# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
ACRE (Active Customer Retention Engine) Environment Implementation.

A reinforcement learning environment where an agent must optimally allocate
a retention budget across a queue of at-risk bank customers by making
personalized counter-offers (fee waivers, rate discounts, etc.).

The agent learns to maximize ROI: (retention_probability × CLV) - offer_cost
"""

import random
from typing import List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from ..models import (
        ACREState,
        ChurnReason,
        CustomerObservation,
        OfferType,
        OutreachAction,
        StepResult,
    )
except ImportError:
    from models import (
        ACREState,
        ChurnReason,
        CustomerObservation,
        OfferType,
        OutreachAction,
        StepResult,
    )

try:
    from .tasks import (
        TaskConfig,
        CustomerProfile,
        generate_customer_from_profile,
        get_task,
        list_tasks,
    )
    from .graders import (
        ActionRecord,
        EpisodeHistory,
        GradeResult,
        grade_episode,
    )
except ImportError:
    from server.tasks import (
        TaskConfig,
        CustomerProfile,
        generate_customer_from_profile,
        get_task,
        list_tasks,
    )
    from server.graders import (
        ActionRecord,
        EpisodeHistory,
        GradeResult,
        grade_episode,
    )


class ACREEnvironment(Environment):
    """
    Active Customer Retention Engine - OpenEnv Environment.

    Simulates a bank retention scenario where an RL agent must decide
    how to allocate a limited budget across a queue of at-risk customers.
    
    The agent receives CustomerObservation for each customer, containing:
    - Customer financials (CLV, monthly revenue, tenure)
    - Churn prediction (risk score, primary/secondary reasons)
    - Market context (current rate, competitor rate, fees)
    - Episode context (remaining budget, queue position)
    
    The agent responds with OutreachAction containing:
    - Offer type (fee waiver, rate discount, cashback, premium upgrade, or no action)
    - Discount percentage or cashback amount
    
    Rewards are trajectory-based: immediate ROI after each offer.
    Episode ends when queue is empty or budget is exhausted.

    Example:
        >>> env = ACREEnvironment()
        >>> obs = env.reset(task_id="task1_risk_triage")
        >>> print(f"Customer CLV: ${obs.customer_lifetime_value}")
        >>> 
        >>> action = OutreachAction(offer_type=OfferType.RATE_DISCOUNT, discount_percentage=0.3)
        >>> result = env.step(action)
        >>> print(f"Reward: ${result.immediate_reward}, Retained: {result.customer_retained}")
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        """Initialize the ACRE environment."""
        self._state = ACREState(episode_id=str(uuid4()), step_count=0)
        self._task_config: Optional[TaskConfig] = None
        self._customer_queue: List[CustomerObservation] = []
        self._customer_profiles: List[CustomerProfile] = []
        self._episode_history: List[ActionRecord] = []
        self._random_seed: Optional[int] = None

    def reset(
        self, 
        task_id: str = "task1_risk_triage",
        seed: Optional[int] = None
    ) -> CustomerObservation:
        """
        Reset the environment and start a new episode.

        Args:
            task_id: Task scenario to load. Options:
                - "task1_risk_triage": High CLV, easy competitor rates
                - "task2_campaign_collision_resolver": Aggressive competitors, precision needed
                - "task3_retention_orchestration": Mixed queue with VIP, tight budget
            seed: Optional random seed for reproducibility

        Returns:
            CustomerObservation for the first customer in the queue
        """
        # Load task configuration
        self._task_config = get_task(task_id)
        self._random_seed = seed
        
        if seed is not None:
            random.seed(seed)
        
        # Initialize state
        self._state = ACREState(
            episode_id=str(uuid4()),
            step_count=0,
            initial_budget=self._task_config.initial_budget,
            remaining_budget=self._task_config.initial_budget,
            total_spent=0.0,
            current_customer_index=0,
            total_customers=len(self._task_config.customer_profiles),
            total_reward=0.0,
            customers_retained=0,
            customers_churned=0,
            total_clv_retained=0.0,
            total_clv_lost=0.0,
            task_id=task_id,
            task_difficulty=self._task_config.difficulty,
        )
        
        # Store profiles for grading
        self._customer_profiles = self._task_config.customer_profiles
        
        # Generate customer queue
        self._customer_queue = []
        for i, profile in enumerate(self._task_config.customer_profiles):
            customer = generate_customer_from_profile(
                profile=profile,
                remaining_budget=self._state.remaining_budget,
                queue_position=i + 1,
                total_customers=self._state.total_customers,
                task_id=task_id,
                task_difficulty=self._task_config.difficulty,
                seed=seed + i if seed else None
            )
            self._customer_queue.append(customer)
        
        # Clear episode history
        self._episode_history = []
        
        # Return first customer
        return self._get_current_customer()

    def step(self, action: OutreachAction) -> StepResult:  # type: ignore[override]
        """
        Execute a retention action on the current customer.

        Args:
            action: OutreachAction specifying the offer to make

        Returns:
            StepResult containing:
            - next_customer: Next customer in queue (or None if done)
            - offer_cost: Cost of the action
            - retention_probability: Calculated retention probability
            - customer_retained: Whether customer was retained (simulated)
            - immediate_reward: ROI from this action
            - done: Whether episode is complete
            - final_score: Task score if done (0.0-1.0)
        """
        self._state.step_count += 1
        
        # Get current customer
        current_customer = self._get_current_customer()
        current_profile = self._customer_profiles[self._state.current_customer_index]
        
        # Calculate offer cost
        offer_cost = action.get_offer_cost(current_customer)
        
        # Check budget constraint
        if offer_cost > self._state.remaining_budget:
            # Can't afford this offer - treat as NO_ACTION
            action = OutreachAction(offer_type=OfferType.NO_ACTION)
            offer_cost = 0.0
        
        # Calculate retention probability
        retention_prob = self._calculate_retention_probability(
            current_customer, action
        )
        
        # Simulate retention outcome (deterministic based on probability threshold)
        customer_retained = retention_prob >= 0.5
        
        # Calculate immediate reward (ROI)
        if customer_retained:
            # Reward = CLV retained minus cost
            immediate_reward = current_customer.customer_lifetime_value - offer_cost
            self._state.customers_retained += 1
            self._state.total_clv_retained += current_customer.customer_lifetime_value
        else:
            # Negative reward = cost spent + opportunity cost (small penalty)
            immediate_reward = -offer_cost - (current_customer.customer_lifetime_value * 0.1)
            self._state.customers_churned += 1
            self._state.total_clv_lost += current_customer.customer_lifetime_value
        
        # Update state
        self._state.remaining_budget -= offer_cost
        self._state.total_spent += offer_cost
        self._state.total_reward += immediate_reward
        
        # Record action for grading
        self._episode_history.append(ActionRecord(
            customer_id=current_customer.customer_id,
            customer_clv=current_customer.customer_lifetime_value,
            is_vip=current_profile.is_vip,
            is_negative_value=current_profile.is_negative_value,
            action=action,
            offer_cost=offer_cost,
            retention_probability=retention_prob,
            customer_retained=customer_retained,
            immediate_reward=immediate_reward,
        ))
        
        # Move to next customer
        self._state.current_customer_index += 1
        
        # Check if episode is done
        queue_complete = self._state.current_customer_index >= self._state.total_customers
        budget_exhausted = self._state.remaining_budget <= 0
        done = queue_complete or budget_exhausted
        
        # Prepare result
        next_customer = None if done else self._get_current_customer()
        
        # Grade if done
        final_score = None
        grade_breakdown = None
        if done:
            grade_result = self._grade_episode()
            final_score = grade_result.score
            grade_breakdown = {
                "passed": grade_result.passed,
                "breakdown": grade_result.breakdown,
                "feedback": grade_result.feedback,
            }
        
        return StepResult(
            next_customer=next_customer,
            action_taken=action,
            offer_cost=offer_cost,
            retention_probability=retention_prob,
            customer_retained=customer_retained,
            immediate_reward=immediate_reward,
            budget_exhausted=budget_exhausted,
            queue_complete=queue_complete,
            done=done,
            reward=immediate_reward,
            final_score=final_score,
            grade_breakdown=grade_breakdown,
            metadata={
                "step": self._state.step_count,
                "remaining_budget": self._state.remaining_budget,
                "customers_remaining": self._state.total_customers - self._state.current_customer_index,
                "total_reward_so_far": self._state.total_reward,
            }
        )

    @property
    def state(self) -> ACREState:
        """
        Get the current environment state.

        Returns:
            ACREState with full episode progress
        """
        return self._state

    def _get_current_customer(self) -> CustomerObservation:
        """Get the current customer observation with updated budget info."""
        if self._state.current_customer_index >= len(self._customer_queue):
            raise IndexError("No more customers in queue")
        
        customer = self._customer_queue[self._state.current_customer_index]
        # Update budget info
        customer.remaining_budget = self._state.remaining_budget
        return customer

    def _calculate_retention_probability(
        self, 
        customer: CustomerObservation, 
        action: OutreachAction
    ) -> float:
        """
        Calculate the probability of retaining the customer given the action.
        
        The model considers:
        - Base churn risk (from upstream model)
        - Offer type appropriateness for churn reason
        - Offer generosity vs competitor gap
        - Customer sentiment
        
        Returns:
            Probability of retention (0.0 to 1.0)
        """
        if action.offer_type == OfferType.NO_ACTION:
            # No offer = base retention (inverse of churn risk)
            return 1.0 - customer.churn_risk_score
        
        # Base retention probability
        base_retention = 1.0 - customer.churn_risk_score
        
        # Calculate offer effectiveness
        effectiveness = 0.0
        
        if action.offer_type == OfferType.RATE_DISCOUNT:
            # Effectiveness depends on closing the rate gap
            rate_gap = customer.current_rate - customer.competitor_best_rate
            rate_improvement = rate_gap * action.discount_percentage
            
            # Good if we close most of the gap
            if rate_gap > 0:
                gap_closed = rate_improvement / rate_gap
                effectiveness = gap_closed * 0.8  # Max 80% effectiveness
            
            # Bonus if this matches the churn reason
            if customer.primary_churn_reason in [
                ChurnReason.HIGH_COMPETITOR_RATE,
                ChurnReason.RATE_SHOPPING
            ]:
                effectiveness *= 1.3
        
        elif action.offer_type == OfferType.FEE_WAIVER:
            # Effectiveness based on discount percentage
            effectiveness = action.discount_percentage * 0.7
            
            # Big bonus if fee sensitivity is the issue
            if customer.primary_churn_reason == ChurnReason.FEE_SENSITIVITY:
                effectiveness *= 1.5
        
        elif action.offer_type == OfferType.CASHBACK:
            # Effectiveness based on cashback relative to monthly revenue
            relative_cashback = action.cashback_amount / max(customer.monthly_revenue, 1)
            effectiveness = min(0.7, relative_cashback * 0.5)
            
            # General appeal
            effectiveness += 0.1
        
        elif action.offer_type == OfferType.PREMIUM_UPGRADE:
            # Good for service dissatisfaction
            effectiveness = action.discount_percentage * 0.6
            
            if customer.primary_churn_reason == ChurnReason.SERVICE_DISSATISFACTION:
                effectiveness *= 1.4
        
        # Apply sentiment modifier
        sentiment_modifier = 1.0 + (customer.sentiment_score * 0.2)
        effectiveness *= sentiment_modifier
        
        # Calculate final retention probability
        # Offer improves retention from base
        retention_prob = base_retention + (1.0 - base_retention) * min(effectiveness, 0.9)
        
        # Clamp to valid range
        return max(0.0, min(1.0, retention_prob))

    def _grade_episode(self) -> GradeResult:
        """Grade the completed episode."""
        history = EpisodeHistory(
            task_config=self._task_config,
            actions=self._episode_history,
            initial_budget=self._state.initial_budget,
            final_budget=self._state.remaining_budget,
            total_spent=self._state.total_spent,
            total_reward=self._state.total_reward,
            total_clv_retained=self._state.total_clv_retained,
            total_clv_lost=self._state.total_clv_lost,
            customers_retained=self._state.customers_retained,
            customers_churned=self._state.customers_churned,
        )
        return grade_episode(history)

