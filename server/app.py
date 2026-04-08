"""
Unified server for the AI Banking Retention Sandbox.

This single FastAPI app serves all 3 tasks through the same endpoints:
  - Task 1 & 2 → bank_churn_env.server.environment.BankChurnEnvironment
  - Task 3     → retention_strategy.server.customer_offer_environment.ACREEnvironment

Endpoints:
  POST /reset  — reset with {"task_id": 1|2|3, "seed": N}
  POST /step   — submit action (format depends on active task)
  GET  /state  — current episode state
  GET  /health — health check
"""

import os
import sys

# Ensure project root is on sys.path so both packages are importable.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional, Any, Dict

# ----- bank_churn_env imports -----
from bank_churn_env.server.environment import BankChurnEnvironment
from bank_churn_env.models import ChurnAction, ChurnObservation, ChurnState

# ----- retention_strategy (ACRE) imports -----
from retention_strategy.server.customer_offer_environment import ACREEnvironment
from retention_strategy.models import OutreachAction, OfferType

# ----- Configure the shared DB path -----
import retention_strategy.server.tasks as acre_tasks
acre_tasks._DB_PATH_OVERRIDE = os.path.join(_PROJECT_ROOT, "bank_ops.db")

# ============================================================
# App
# ============================================================
app = FastAPI(
    title="AI Banking Retention Sandbox",
    description="Unified 3-task OpenEnv environment: churn triage → campaign collision → retention orchestration",
    version="1.0.0",
)

# Global environment instances
bank_env = BankChurnEnvironment()
acre_env = ACREEnvironment()

# Track which environment is currently active
_active_task: int = 0


class ResetRequest(BaseModel):
    task_id: int = 1
    seed: int = 42


class StepResponse(BaseModel):
    observation: Any
    reward: float
    done: bool
    info: dict


# ============================================================
# Routes
# ============================================================

@app.get("/health")
async def health():
    return {"status": "ok", "tasks": [1, 2, 3]}


@app.post("/reset")
async def reset(request: Optional[ResetRequest] = None):
    if request is None:
        request = ResetRequest()
    global _active_task
    _active_task = request.task_id

    if request.task_id in (1, 2):
        try:
            # Set working dir to project root so DB + CSV files land there
            bank_env.agent_working_dir = _PROJECT_ROOT
            obs = bank_env.reset(task_id=request.task_id, seed=request.seed)
            return obs.model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif request.task_id == 3:
        try:
            # Map numeric task_id to ACRE's string task_id
            obs = acre_env.reset(
                task_id="task3_retention_orchestration",
                seed=request.seed,
            )
            # Convert CustomerObservation to a dict for consistent JSON response
            return obs.model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid task_id={request.task_id}. Must be 1, 2, or 3.",
        )


@app.post("/step")
async def step(request: Request):
    """
    Accept action for the currently active task.

    Task 1/2 expect ChurnAction fields:
      {customer_id, risk_tier, reasoning, confidence, top_signals_used, ...}

    Task 3 (ACRE) expects OutreachAction fields:
      {offer_type, discount_percentage, cashback_amount}
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    from pydantic import ValidationError

    if _active_task in (1, 2):
        try:
            action = ChurnAction(**body)
            obs, reward, done, info = bank_env.step(action)
            return {
                "observation": obs.model_dump(),
                "reward": reward,
                "done": done,
                "info": info,
            }
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    elif _active_task == 3:
        try:
            # Parse ACRE action
            offer_type = body.get("offer_type", "NO_ACTION")
            discount_pct = float(body.get("discount_percentage", 0.0))
            cashback = float(body.get("cashback_amount", 0.0))

            action = OutreachAction(
                offer_type=OfferType(offer_type),
                discount_percentage=discount_pct,
                cashback_amount=cashback,
            )
            result = acre_env.step(action)

            # Build unified response
            obs_dict = result.model_dump() if hasattr(result, "model_dump") else {}
            reward = float(obs_dict.get("reward", obs_dict.get("immediate_reward", 0.0)))
            done = bool(obs_dict.get("done", obs_dict.get("queue_complete", False) or obs_dict.get("budget_exhausted", False)))
            info = {}
            if done:
                info["final_score"] = obs_dict.get("final_score", 0.0)
                info["grade_breakdown"] = obs_dict.get("grade_breakdown")

            return {
                "observation": obs_dict,
                "reward": reward,
                "done": done,
                "info": info,
            }
        except ValidationError as e:
            raise HTTPException(status_code=422, detail=e.errors())
        except RuntimeError as e:
            raise HTTPException(status_code=400, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    else:
        raise HTTPException(
            status_code=400,
            detail="No active task. Call /reset first.",
        )


@app.get("/state")
async def state():
    if _active_task in (1, 2):
        try:
            return bank_env.state().model_dump()
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    elif _active_task == 3:
        try:
            s = acre_env.state
            return s.model_dump() if hasattr(s, "model_dump") else {}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        return {"status": "not initialized", "task_id": 0}


# ============================================================
# Entry point
# ============================================================
def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run("server.app:app", host=host, port=port)


if __name__ == "__main__":
    main()
