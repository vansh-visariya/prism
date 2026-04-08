# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Data models for the ACRE (Active Customer Retention Engine) Environment.

ACRE is a reinforcement learning environment where an agent must optimally
allocate a retention budget across a queue of at-risk bank customers by
making personalized counter-offers (fee waivers, rate discounts, etc.).
"""

from enum import Enum
from typing import List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class OfferType(str, Enum):
    """Types of retention offers the agent can make."""
    
    NO_ACTION = "NO_ACTION"  # Skip this customer (cost: $0)
    FEE_WAIVER = "FEE_WAIVER"  # Waive monthly/annual fees (cost: fee amount)
    RATE_DISCOUNT = "RATE_DISCOUNT"  # Reduce loan/credit rate (cost: NPV of discount)
    CASHBACK = "CASHBACK"  # One-time cashback incentive (cost: cashback amount)
    PREMIUM_UPGRADE = "PREMIUM_UPGRADE"  # Free premium tier upgrade (cost: tier difference)


class ChurnReason(str, Enum):
    """Primary reasons for customer churn risk."""
    
    HIGH_COMPETITOR_RATE = "HIGH_COMPETITOR_RATE"  # Competitor offers better rates
    FEE_SENSITIVITY = "FEE_SENSITIVITY"  # Customer complaining about fees
    SERVICE_DISSATISFACTION = "SERVICE_DISSATISFACTION"  # Poor service experience
    LIFE_EVENT = "LIFE_EVENT"  # Relocation, job change, etc.
    INACTIVE_ACCOUNT = "INACTIVE_ACCOUNT"  # Declining engagement
    RATE_SHOPPING = "RATE_SHOPPING"  # Actively comparing rates


class CustomerObservation(Observation):
    """
    Observation representing the current customer in the queue.
    
    This payload simulates what the upstream ML pipeline (churn prediction model)
    would send to the RL environment, containing all information needed to
    make a retention decision.
    """
    
    # Customer identification (anonymized)
    customer_id: str = Field(..., description="Anonymized customer identifier")
    
    # Core financial metrics
    customer_lifetime_value: float = Field(
        ..., 
        description="Expected CLV in dollars over remaining relationship",
        ge=0
    )
    monthly_revenue: float = Field(
        ..., 
        description="Average monthly revenue from this customer",
        ge=0
    )
    account_tenure_months: int = Field(
        ..., 
        description="How long the customer has been with the bank",
        ge=0
    )
    
    # Churn prediction outputs
    churn_risk_score: float = Field(
        ..., 
        description="Probability of churn (0.0 to 1.0) from upstream model",
        ge=0.0,
        le=1.0
    )
    primary_churn_reason: ChurnReason = Field(
        ..., 
        description="Main driver of churn risk from XAI analysis"
    )
    secondary_churn_reasons: List[ChurnReason] = Field(
        default_factory=list,
        description="Additional contributing factors"
    )
    
    # Competitive context
    current_rate: float = Field(
        ..., 
        description="Customer's current interest rate (loan/savings)",
        ge=0.0,
        le=1.0  # As decimal, e.g., 0.05 = 5%
    )
    competitor_best_rate: float = Field(
        ..., 
        description="Best competitor rate from live scraper",
        ge=0.0,
        le=1.0
    )
    current_monthly_fee: float = Field(
        ..., 
        description="Customer's current monthly account fee",
        ge=0
    )
    
    # Sentiment analysis (from FinBERT)
    sentiment_score: float = Field(
        default=0.0,
        description="Customer sentiment from interactions (-1.0 to 1.0)",
        ge=-1.0,
        le=1.0
    )
    
    # Episode context
    remaining_budget: float = Field(
        ..., 
        description="Remaining retention budget for this episode",
        ge=0
    )
    queue_position: int = Field(
        ..., 
        description="Current position in customer queue (1-indexed)",
        ge=1
    )
    total_customers: int = Field(
        ..., 
        description="Total customers in this episode's queue",
        ge=1
    )
    
    # Task identification
    task_id: str = Field(
        default="default",
        description="Identifier for the current task scenario"
    )
    task_difficulty: str = Field(
        default="medium",
        description="Task difficulty level: easy, medium, or hard"
    )


class OutreachAction(Action):
    """
    Action representing the retention offer to make to the current customer.
    
    The agent must decide both the type of offer and the generosity level
    (discount_percentage), balancing retention probability against cost.
    """
    
    offer_type: OfferType = Field(
        ..., 
        description="Type of retention offer to make"
    )
    discount_percentage: float = Field(
        default=0.0,
        description="Discount/waiver percentage (0.0 to 1.0). Only applies to FEE_WAIVER and RATE_DISCOUNT.",
        ge=0.0,
        le=1.0
    )
    cashback_amount: float = Field(
        default=0.0,
        description="One-time cashback amount in dollars. Only applies to CASHBACK offer.",
        ge=0.0
    )
    
    def get_offer_cost(self, customer: CustomerObservation) -> float:
        """
        Calculate the cost of this offer for the given customer.
        
        Returns:
            Cost in dollars that will be deducted from the budget.
        """
        if self.offer_type == OfferType.NO_ACTION:
            return 0.0
        
        elif self.offer_type == OfferType.FEE_WAIVER:
            # Cost = annual fee waived × discount percentage
            annual_fee = customer.current_monthly_fee * 12
            return annual_fee * self.discount_percentage
        
        elif self.offer_type == OfferType.RATE_DISCOUNT:
            # Cost = NPV of rate discount over expected tenure
            # Simplified: monthly savings × expected remaining months × discount
            rate_gap = customer.current_rate - customer.competitor_best_rate
            monthly_savings = customer.monthly_revenue * rate_gap * self.discount_percentage
            expected_months = min(24, customer.account_tenure_months)  # Cap at 2 years
            return monthly_savings * expected_months
        
        elif self.offer_type == OfferType.CASHBACK:
            return self.cashback_amount
        
        elif self.offer_type == OfferType.PREMIUM_UPGRADE:
            # Fixed cost for premium upgrade
            return 150.0 * self.discount_percentage  # $150 max for full upgrade
        
        return 0.0


class ACREState(State):
    """
    Extended state for ACRE environment tracking episode progress.
    """
    
    episode_id: Optional[str] = Field(default=None, description="Unique episode identifier")
    step_count: int = Field(default=0, description="Number of steps taken")
    
    # Budget tracking
    initial_budget: float = Field(default=2000.0, description="Starting budget for episode")
    remaining_budget: float = Field(default=2000.0, description="Current remaining budget")
    total_spent: float = Field(default=0.0, description="Total budget spent on offers")
    
    # Queue tracking
    current_customer_index: int = Field(default=0, description="Index of current customer in queue")
    total_customers: int = Field(default=0, description="Total customers in queue")
    
    # Performance tracking
    total_reward: float = Field(default=0.0, description="Cumulative reward this episode")
    customers_retained: int = Field(default=0, description="Number of customers retained")
    customers_churned: int = Field(default=0, description="Number of customers lost")
    total_clv_retained: float = Field(default=0.0, description="Total CLV of retained customers")
    total_clv_lost: float = Field(default=0.0, description="Total CLV of churned customers")
    
    # Task info
    task_id: str = Field(default="default", description="Current task identifier")
    task_difficulty: str = Field(default="medium", description="Current task difficulty")


class StepResult(Observation):
    """
    Result of a step in the ACRE environment.
    
    Contains the next customer observation (or final state),
    reward from the action, and episode completion status.
    """
    
    # Next customer observation (None if episode done)
    next_customer: Optional[CustomerObservation] = Field(
        default=None,
        description="Next customer in queue, or None if episode complete"
    )
    
    # Action outcome
    action_taken: Optional[OutreachAction] = Field(
        default=None,
        description="The action that was executed"
    )
    offer_cost: float = Field(
        default=0.0,
        description="Cost of the offer made"
    )
    retention_probability: float = Field(
        default=0.0,
        description="Calculated probability of retaining the customer"
    )
    customer_retained: bool = Field(
        default=False,
        description="Whether the customer was retained (simulated outcome)"
    )
    
    # Reward breakdown
    immediate_reward: float = Field(
        default=0.0,
        description="ROI from this single action"
    )
    
    # Episode status
    budget_exhausted: bool = Field(
        default=False,
        description="True if budget hit zero"
    )
    queue_complete: bool = Field(
        default=False,
        description="True if all customers processed"
    )
    
    # Grading (populated only when done=True)
    final_score: Optional[float] = Field(
        default=None,
        description="Final task score (0.0 to 1.0), only present when episode ends"
    )
    grade_breakdown: Optional[dict] = Field(
        default=None,
        description="Detailed grading breakdown"
    )
