# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
Grading logic for ACRE (Active Customer Retention Engine) tasks.

Each grader returns a score between 0.0 and 1.0 based on agent performance.
Grading criteria vary by task difficulty:
- Easy: Basic profitability and valid offer making
- Medium: Precision in discount calibration
- Hard: Strategic prioritization (save VIP, skip negative-value)
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

try:
    from ..models import OfferType, OutreachAction
    from .tasks import TaskConfig, CustomerProfile
except ImportError:
    from models import OfferType, OutreachAction
    from server.tasks import TaskConfig, CustomerProfile


@dataclass
class ActionRecord:
    """Record of an action taken during an episode."""
    
    customer_id: str
    customer_clv: float
    is_vip: bool
    is_negative_value: bool
    action: OutreachAction
    offer_cost: float
    retention_probability: float
    customer_retained: bool
    immediate_reward: float


@dataclass
class EpisodeHistory:
    """Complete history of an episode for grading."""
    
    task_config: TaskConfig
    actions: List[ActionRecord]
    initial_budget: float
    final_budget: float
    total_spent: float
    total_reward: float
    total_clv_retained: float
    total_clv_lost: float
    customers_retained: int
    customers_churned: int


@dataclass
class GradeResult:
    """Result of grading an episode."""
    
    score: float  # 0.0 to 1.0
    passed: bool
    breakdown: Dict[str, float]  # Component scores
    feedback: str  # Human-readable explanation



def grade_hard_task(history: EpisodeHistory) -> GradeResult:
    """
    Grade Task 3: VIP Triage
    
    Scoring criteria:
    - 50%: VIP retention (binary - did they save the VIP?)
    - 25%: Negative-value customer handling (skipped appropriately?)
    - 15%: Total ROI
    - 10%: Budget management (had budget left for VIP?)
    
    Key: Score approaches 1.0 ONLY if VIP is retained.
    
    Returns score 0.0-1.0
    """
    breakdown = {}
    
    # Find VIP and negative-value actions
    vip_action = None
    negative_value_actions = []
    
    for record in history.actions:
        if record.is_vip:
            vip_action = record
        if record.is_negative_value:
            negative_value_actions.append(record)
    
    # VIP Retention Score (50%)
    if vip_action is None:
        # VIP wasn't even reached (budget exhausted early)
        vip_score = 0.0
        vip_retained = False
    else:
        vip_retained = vip_action.customer_retained
        if vip_retained:
            vip_score = 1.0
        else:
            # Partial credit if they at least made an offer
            if vip_action.action.offer_type != OfferType.NO_ACTION:
                vip_score = 0.3  # Tried but failed
            else:
                vip_score = 0.0  # Skipped the VIP - terrible decision
    breakdown["vip_retention"] = vip_score
    
    # Negative-Value Handling Score (25%)
    # Should skip (NO_ACTION) negative-value customers
    if negative_value_actions:
        skipped_negative = sum(
            1 for r in negative_value_actions 
            if r.action.offer_type == OfferType.NO_ACTION
        )
        negative_score = skipped_negative / len(negative_value_actions)
    else:
        negative_score = 1.0  # No negative-value customers to worry about
    breakdown["negative_value_handling"] = negative_score
    
    # ROI Score (15%)
    roi = history.total_reward
    roi_threshold = history.task_config.passing_roi_threshold
    roi_optimal = history.task_config.optimal_roi_estimate
    
    if roi <= 0:
        roi_score = 0.0
    elif roi >= roi_optimal:
        roi_score = 1.0
    else:
        roi_score = roi / roi_optimal
    breakdown["roi_score"] = roi_score
    
    # Budget Management Score (10%)
    # Did they have budget remaining when reaching VIP?
    total_before_vip = 0
    for record in history.actions:
        if record.is_vip:
            break
        total_before_vip += record.offer_cost
    
    budget_at_vip = history.initial_budget - total_before_vip
    # Good if they conserved most budget for VIP
    budget_score = min(1.0, budget_at_vip / (history.initial_budget * 0.6))
    breakdown["budget_management"] = budget_score
    
    # Weighted final score
    final_score = (
        0.50 * vip_score +
        0.25 * negative_score +
        0.15 * roi_score +
        0.10 * budget_score
    )
    
    # Hard task: Only truly passes if VIP retained
    passed = vip_retained and roi > 0
    
    feedback = (
        f"VIP {'RETAINED' if vip_retained else 'LOST'}! "
        f"Skipped {int(negative_score * len(negative_value_actions))}/{len(negative_value_actions)} negative-value customers. "
        f"ROI: ${roi:.2f}. "
        f"Budget at VIP: ${budget_at_vip:.2f}."
    )
    
    return GradeResult(
        score=round(max(0.01, min(0.99, final_score)), 3),
        passed=passed,
        breakdown=breakdown,
        feedback=feedback
    )


def grade_episode(history: EpisodeHistory) -> GradeResult:
    """Grade a completed Task 3 episode."""
    return grade_hard_task(history)
