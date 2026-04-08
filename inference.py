"""
Single inference entrypoint for this repository.

Runs all 3 tasks sequentially against the unified server:
  Task 1: Risk Triage (bank_churn_env)
  Task 2: Campaign Collision Resolver (bank_churn_env)
  Task 3: Retention Orchestration (ACRE / retention_strategy)
"""

import json
import os
import sys
from typing import Any, Dict, List

import requests
from openai import OpenAI


# ============================================================
# Configuration
# ============================================================
REQUIRED_ENV_VARS = ["API_BASE_URL", "MODEL_NAME", "HF_TOKEN"]
BENCHMARK = "bank_churn_env"


def _load_dotenv(dotenv_path: str) -> None:
    if not os.path.exists(dotenv_path):
        return
    with open(dotenv_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and not os.getenv(key):
                os.environ[key] = value


def _validate_required_env() -> None:
    missing = [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]
    if missing:
        names = ", ".join(missing)
        print(
            f"ERROR: Missing required environment variables: {names}. "
            "Set them before running inference.",
            flush=True,
        )
        sys.exit(2)


# ============================================================
# Logging helpers
# ============================================================
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: str | None) -> None:
    err = error if error else "null"
    done_value = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_value} error={err}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{x:.2f}" for x in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ============================================================
# JSON helpers
# ============================================================
def _extract_json_from_text(raw: str) -> Dict[str, Any] | List[Any]:
    text = (raw or "").strip()
    if not text:
        raise ValueError("empty model response")

    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(line for line in lines[1:] if not line.strip().startswith("```"))
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        obj_start = text.find("{")
        arr_start = text.find("[")
        start_candidates = [x for x in [obj_start, arr_start] if x >= 0]
        if not start_candidates:
            raise
        start = min(start_candidates)
        candidate = text[start:]
        for end_char in ["}", "]"]:
            end = candidate.rfind(end_char)
            if end >= 0:
                snippet = candidate[: end + 1]
                try:
                    return json.loads(snippet)
                except json.JSONDecodeError:
                    continue
        raise


def call_llm_json(client: OpenAI, model: str, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        max_tokens=500,
    )
    raw = (response.choices[0].message.content or "").strip()
    parsed = _extract_json_from_text(raw)
    if isinstance(parsed, list):
        return {"records": parsed}
    return parsed


# ============================================================
# Campaign resolver (Task 2 helper)
# ============================================================
import csv

def resolve_campaign_files_with_llm(
    client: OpenAI,
    model: str,
    campaign_input_files: Dict[str, str] | None,
    rules_file: str | None,
    expected_output_file: str | None,
) -> None:
    def read_csv(path: str) -> List[Dict[str, str]]:
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return list(csv.DictReader(f))

    files = campaign_input_files or {}
    auto_path = files.get("auto_loan", "auto_loan.csv")
    card_path = files.get("credit_card", "credit_card.csv")
    retention_path = files.get("retention", "retention.csv")
    rules_path = rules_file or "priority_rules.txt"
    output_path = expected_output_file or "master_campaign_deployment.json"

    payload = {
        "auto_loan": read_csv(auto_path),
        "credit_card": read_csv(card_path),
        "retention": read_csv(retention_path),
        "rules": open(rules_path, "r", encoding="utf-8").read() if os.path.exists(rules_path) else "",
    }

    def deterministic_resolve() -> List[Dict[str, Any]]:
        priority = {"retention": 3, "credit_card": 2, "auto_loan": 1}
        picks: Dict[str, Dict[str, Any]] = {}
        for source_key in ["auto_loan", "credit_card", "retention"]:
            for row in payload.get(source_key, []):
                cid = str(row.get("customer_id", "")).strip()
                if not cid:
                    continue
                try:
                    offer_value = float(row.get("offer_value", 0))
                except (TypeError, ValueError):
                    offer_value = 0.0
                source = str(row.get("campaign_source", source_key)).strip() or source_key
                candidate = {
                    "customer_id": cid,
                    "offer_value": round(offer_value, 2),
                    "campaign_source": source,
                }
                current = picks.get(cid)
                if current is None:
                    picks[cid] = candidate
                    continue
                cp = priority.get(str(current.get("campaign_source", "")), 0)
                np = priority.get(source, 0)
                if np > cp:
                    picks[cid] = candidate
                elif np == cp and candidate["offer_value"] > float(current.get("offer_value", 0)):
                    picks[cid] = candidate

        return [picks[cid] for cid in sorted(picks.keys())]

    deployments: List[Dict[str, Any]] = []
    system_prompt = (
        "Resolve campaign collisions. Return JSON array with one row per customer_id "
        "using columns customer_id, offer_value, campaign_source. Follow rules exactly."
    )
    user_prompt = json.dumps(payload)

    try:
        result = call_llm_json(client, model, system_prompt, user_prompt)
        deployments = result.get("deployments") or result.get("records") or []
        if not isinstance(deployments, list) or not deployments:
            deployments = deterministic_resolve()
    except Exception:
        deployments = deterministic_resolve()

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(deployments, f, indent=2)


# ============================================================
# Episode runner - Task 1 & 2
# ============================================================
def run_bank_episode(client: OpenAI, model: str, env_url: str, task_id: int, task_name: str) -> float:
    log_start(task=task_name, env=BENCHMARK, model=model)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    done = False
    last_error: str | None = None

    try:
        reset_resp = requests.post(f"{env_url}/reset", json={"task_id": task_id, "seed": 41}, timeout=60)
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        if task_id == 2:
            resolve_campaign_files_with_llm(
                client,
                model,
                obs.get("campaign_input_files"),
                obs.get("priority_rules_file"),
                obs.get("expected_output_file"),
            )

        while not done:
            steps_taken += 1

            if task_id == 1:
                system_prompt = (
                    "Classify bank churn risk. Return JSON with customer_id,risk_tier,reasoning,confidence,top_signals_used. "
                    "risk_tier must be one of high_risk,medium_risk,low_risk,not_at_risk."
                )
                user_prompt = json.dumps(obs.get("current_customer", {}))
                llm_action = call_llm_json(client, model, system_prompt, user_prompt)
                action = {
                    "customer_id": llm_action.get("customer_id") or obs.get("current_customer", {}).get("customer_id", ""),
                    "risk_tier": llm_action.get("risk_tier", "medium_risk"),
                    "reasoning": llm_action.get("reasoning", "risk assessment"),
                    "confidence": float(llm_action.get("confidence", 0.5)),
                    "top_signals_used": llm_action.get("top_signals_used", ["lstm_output"]),
                }
            elif task_id == 2:
                current_customer = obs.get("current_customer", {})
                task2_context = current_customer.get("task2_context", {})
                current_customer_id = current_customer.get("customer_id", "")
                system_prompt = (
                    "Choose the final campaign assignment for this customer. "
                    "Return JSON with fields: selected_campaign_source (retention|credit_card|auto_loan), "
                    "selected_offer_value (number), reasoning (string), confidence (0..1), "
                    "top_signals_used (array of strings). Apply priority rules strictly."
                )
                user_prompt = json.dumps(task2_context)

                chosen_source = "retention"
                chosen_offer = 0.0
                task2_reasoning = "resolved using campaign priorities"
                task2_confidence = 0.7
                task2_signals = ["campaign_source", "offer_value", "priority_rules"]
                try:
                    llm_action = call_llm_json(client, model, system_prompt, user_prompt)
                    chosen_source = str(
                        llm_action.get("selected_campaign_source", "retention")
                    ).strip() or "retention"
                    chosen_offer = float(llm_action.get("selected_offer_value", 0.0))
                    task2_reasoning = str(
                        llm_action.get("reasoning", task2_reasoning)
                    ).strip() or task2_reasoning
                    task2_confidence = float(llm_action.get("confidence", task2_confidence))
                    task2_signals = llm_action.get("top_signals_used", task2_signals)
                    if not isinstance(task2_signals, list) or not task2_signals:
                        task2_signals = ["campaign_source", "offer_value", "priority_rules"]
                except Exception:
                    offers = task2_context.get("candidate_offers", [])
                    priority = {"retention": 3, "credit_card": 2, "auto_loan": 1}
                    best = None
                    for row in offers:
                        source = str(row.get("campaign_source", "")).strip()
                        try:
                            offer_value = float(row.get("offer_value", 0.0))
                        except (TypeError, ValueError):
                            offer_value = 0.0
                        item = {"campaign_source": source, "offer_value": offer_value}
                        if best is None:
                            best = item
                            continue
                        bp = priority.get(best["campaign_source"], 0)
                        np = priority.get(source, 0)
                        if np > bp or (np == bp and offer_value > best["offer_value"]):
                            best = item
                    if best:
                        chosen_source = best["campaign_source"]
                        chosen_offer = float(best["offer_value"])

                task2_confidence = max(0.0, min(1.0, task2_confidence))

                action = {
                    "customer_id": current_customer_id or "TASK2",
                    "risk_tier": "medium_risk",
                    "reasoning": task2_reasoning,
                    "confidence": round(task2_confidence, 3),
                    "top_signals_used": task2_signals,
                    "selected_campaign_source": chosen_source,
                    "selected_offer_value": round(chosen_offer, 2),
                }
            else:
                break

            step_resp = requests.post(f"{env_url}/step", json=action, timeout=60)
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            obs = result.get("observation", {})
            rewards.append(reward)

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                final_score = float(result.get("info", {}).get("final_score", 0.0))

        success = final_score >= 0.5
        return final_score
    except Exception as exc:
        last_error = str(exc)
        log_step(step=max(1, steps_taken), action="exception", reward=0.0, done=True, error=last_error)
        return 0.0
    finally:
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


# ============================================================
# Episode runner - Task 3 (ACRE)
# ============================================================

ACRE_SYSTEM_PROMPT = """You are an AI agent managing customer retention for a bank. Your goal is to maximize ROI by making smart retention offers to at-risk customers.

For each customer, you will receive metrics about their lifetime value, churn risk, rates, and fees.

Respond with a JSON object:
{
    "offer_type": "NO_ACTION" | "FEE_WAIVER" | "RATE_DISCOUNT" | "CASHBACK" | "PREMIUM_UPGRADE",
    "discount_percentage": 0.0 to 1.0,
    "cashback_amount": dollar amount (for CASHBACK only),
    "reasoning": "Brief explanation"
}

Strategy Tips:
1. HIGH CLV customers are worth spending more on
2. Match offer type to churn reason (rate issues → RATE_DISCOUNT, fee complaints → FEE_WAIVER)
3. Conserve budget for later high-value customers
4. Skip LOW CLV customers (CLV < $500) with NO_ACTION
5. ROI = (retention_probability × CLV) - offer_cost

Always respond with valid JSON only."""


def run_acre_episode(client: OpenAI, model: str, env_url: str) -> float:
    task_name = "task3_retention_orchestration"
    log_start(task=task_name, env=BENCHMARK, model=model)

    rewards: List[float] = []
    steps_taken = 0
    final_score = 0.0
    success = False
    done = False
    last_error: str | None = None

    try:
        reset_resp = requests.post(
            f"{env_url}/reset",
            json={"task_id": 3, "seed": 41},
            timeout=60,
        )
        reset_resp.raise_for_status()
        obs = reset_resp.json()

        while not done:
            steps_taken += 1

            # Format customer info for LLM
            customer_info = {
                k: obs.get(k)
                for k in [
                    "customer_id", "customer_lifetime_value", "monthly_revenue",
                    "account_tenure_months", "churn_risk_score", "primary_churn_reason",
                    "secondary_churn_reasons", "current_rate", "competitor_best_rate",
                    "current_monthly_fee", "sentiment_score", "remaining_budget",
                    "queue_position", "total_customers", "task_id", "task_difficulty",
                ]
                if obs.get(k) is not None
            }

            user_prompt = json.dumps(customer_info, indent=2)

            try:
                llm_action = call_llm_json(client, model, ACRE_SYSTEM_PROMPT, user_prompt)
            except Exception:
                llm_action = {"offer_type": "NO_ACTION", "discount_percentage": 0, "cashback_amount": 0}

            # Normalize action
            valid_offers = ["NO_ACTION", "FEE_WAIVER", "RATE_DISCOUNT", "CASHBACK", "PREMIUM_UPGRADE"]
            offer_type = str(llm_action.get("offer_type", "NO_ACTION")).upper()
            if offer_type not in valid_offers:
                offer_type = "NO_ACTION"

            discount = max(0.0, min(1.0, float(llm_action.get("discount_percentage", 0))))
            cashback = max(0.0, float(llm_action.get("cashback_amount", 0)))

            action = {
                "offer_type": offer_type,
                "discount_percentage": discount,
                "cashback_amount": cashback,
            }

            step_resp = requests.post(f"{env_url}/step", json=action, timeout=60)
            step_resp.raise_for_status()
            result = step_resp.json()

            reward = float(result.get("reward", 0.0))
            done = bool(result.get("done", False))
            rewards.append(reward)

            action_str = json.dumps(action, separators=(",", ":"))
            log_step(step=steps_taken, action=action_str, reward=reward, done=done, error=last_error)

            if done:
                info = result.get("info", {})
                final_score = float(info.get("final_score", 0.0))
            else:
                # Get next customer observation
                obs_data = result.get("observation", {})
                next_cust = obs_data.get("next_customer")
                if next_cust:
                    obs = next_cust
                else:
                    obs = obs_data

        success = final_score >= 0.5
        return final_score
    except Exception as exc:
        last_error = str(exc)
        log_step(step=max(1, steps_taken), action="exception", reward=0.0, done=True, error=last_error)
        return 0.0
    finally:
        if not rewards:
            rewards = [0.0]
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    project_root = os.path.dirname(os.path.abspath(__file__))
    _load_dotenv(os.path.join(project_root, ".env"))

    API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
    MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
    HF_TOKEN = os.getenv("HF_TOKEN")
    ENV_URL = os.getenv("ENV_URL", "http://localhost:8000")

    _validate_required_env()

    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

    score1 = run_bank_episode(client, MODEL_NAME, ENV_URL, task_id=1, task_name="task1_risk_triage")
    score2 = run_bank_episode(client, MODEL_NAME, ENV_URL, task_id=2, task_name="task2_campaign_collision_resolver")
    score3 = run_acre_episode(client, MODEL_NAME, ENV_URL)
