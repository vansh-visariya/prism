from pydantic import BaseModel, Field
from typing import Literal, List, Optional, Dict, Any

class ChurnAction(BaseModel):
    customer_id: str
    risk_tier: Literal["high_risk", "medium_risk", "low_risk", "not_at_risk"]
    reasoning: str
    confidence: float = Field(ge=0.0, le=1.0)
    top_signals_used: List[str]
    selected_campaign_source: Optional[Literal["retention", "credit_card", "auto_loan"]] = None
    selected_offer_value: Optional[float] = Field(default=None, ge=0.0)
    engagement_action: Optional[Literal["contact", "skip"]] = None
    contact_channel: Optional[Literal["call", "email", "sms"]] = None


class ChurnObservation(BaseModel):
    task_id: int
    task_description: str
    step_number: int
    max_steps: int
    current_customer: Dict[str, Any]
    remaining_customer_ids: List[str]
    classified_so_far: List[dict]
    cumulative_reward: float
    last_feedback: Optional[str] = None
    campaign_input_files: Optional[Dict[str, str]] = None
    priority_rules_file: Optional[str] = None
    expected_output_file: Optional[str] = None
    database_file: Optional[str] = None
    task3_customers_remaining: Optional[int] = None
    task3_contact_budget_remaining: Optional[int] = None


class ChurnReward(BaseModel):
    value: float
    reason: Optional[str] = None


class ChurnState(BaseModel):
    episode_id: str = ""
    task_id: int = 1
    step_count: int = 0
    total_customers: int = 10
    classified_count: int = 0
    current_score: float = 0.0
    high_risk_found: int = 0
    dangerous_misses: int = 0
