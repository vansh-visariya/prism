import uuid
import os
from typing import Dict, Any, List, Optional

try:
    from bank_churn_env.server.customer_generator import (
        CustomerGenerator,
        compute_true_risk,
        generate_campaign_collisions,
        load_campaign_rows,
    )
    from bank_churn_env.server.ml_signals import generate_all_ml_signals
    from bank_churn_env.server.reward import compute_step_reward
    from bank_churn_env.server.grader import grade_task1, grade_task_2
    from bank_churn_env.server.data_store import (
        init_db,
        persist_task1_results,
        persist_task2_candidates,
        persist_task2_deployment,
        read_task2_output,
    )
    from bank_churn_env.models import ChurnAction, ChurnObservation, ChurnState
except ImportError:
    from .customer_generator import (
        CustomerGenerator,
        compute_true_risk,
        generate_campaign_collisions,
        load_campaign_rows,
    )
    from .ml_signals import generate_all_ml_signals
    from .reward import compute_step_reward
    from .grader import grade_task1, grade_task_2
    from .data_store import (
        init_db,
        persist_task1_results,
        persist_task2_candidates,
        persist_task2_deployment,
        read_task2_output,
    )
    from ..models import ChurnAction, ChurnObservation, ChurnState


class BankChurnEnvironment:
    def __init__(self):
        self.customers: List[Dict[str, Any]] = []
        self.customer_truth: Dict[str, str] = {}
        self.customer_ml_signals: Dict[str, Dict[str, Any]] = {}
        self.episode_id: str = ""
        self.task_id: int = 1
        self.step_count: int = 0
        self.classified_results: List[Dict[str, Any]] = []
        self.cumulative_reward: float = 0.0
        self.is_reset: bool = False
        self.max_steps: int = 10
        self.task2_files: Dict[str, str] = {}
        self.agent_working_dir: str = os.getcwd()
        self.db_path: str = ""
        self.task2_rows: List[Dict[str, Any]] = []
        self.task2_customer_order: List[str] = []
        self.task2_expected_by_customer: Dict[str, Dict[str, Any]] = {}
        self.task2_output_map: Dict[str, Dict[str, Any]] = {}
        self.task3_queue: List[Dict[str, Any]] = []
        self.task3_history: List[Dict[str, Any]] = []
        self.task3_contact_budget: int = 0

    def reset(self, task_id: int = 1, seed: int = 42) -> ChurnObservation:
        generator = CustomerGenerator(seed=seed)
        self.customers = generator.generate_customers(10)

        self.customer_truth = {}
        self.customer_ml_signals = {}

        for customer in self.customers:
            customer_id = customer["customer_id"]
            true_risk = compute_true_risk(customer)
            self.customer_truth[customer_id] = true_risk

            score = self._compute_risk_score(customer)
            ml_signals = generate_all_ml_signals(customer, score)
            ml_signals = self._ensure_consistency(customer_id, true_risk, ml_signals)
            self.customer_ml_signals[customer_id] = ml_signals

        self.episode_id = uuid.uuid4().hex[:8]
        self.task_id = task_id
        self.step_count = 0
        self.classified_results = []
        self.cumulative_reward = 0.0
        self.is_reset = True
        self.agent_working_dir = os.getcwd()
        self.db_path = init_db(self.agent_working_dir)
        self.task2_files = {}
        self.task2_rows = []
        self.task2_customer_order = []
        self.task2_expected_by_customer = {}
        self.task2_output_map = {}
        self.task3_queue = []
        self.task3_history = []
        self.task3_contact_budget = 0
        self.max_steps = 10 if task_id in {1, 2} else 1

        if task_id == 2:
            self.task2_files = generate_campaign_collisions(
                self.customers, output_dir=self.agent_working_dir, seed=seed
            )
            self.task2_rows = load_campaign_rows(self.task2_files)

            rows_by_customer: Dict[str, List[Dict[str, Any]]] = {}
            for row in self.task2_rows:
                cid = str(row.get("customer_id", "")).strip()
                if not cid:
                    continue
                rows_by_customer.setdefault(cid, []).append(row)

            self.task2_customer_order = sorted(rows_by_customer.keys())

            priority = {"retention": 3, "credit_card": 2, "auto_loan": 1}
            for cid, rows in rows_by_customer.items():
                best = None
                for row in rows:
                    if best is None:
                        best = row
                        continue
                    bp = priority.get(str(best.get("campaign_source", "")), 0)
                    rp = priority.get(str(row.get("campaign_source", "")), 0)
                    if rp > bp:
                        best = row
                    elif rp == bp and float(row.get("offer_value", 0.0)) > float(
                        best.get("offer_value", 0.0)
                    ):
                        best = row

                if best:
                    self.task2_expected_by_customer[cid] = {
                        "customer_id": cid,
                        "campaign_source": str(best.get("campaign_source", "")).strip(),
                        "offer_value": float(best.get("offer_value", 0.0)),
                        "candidate_offers": rows,
                    }

            persist_task2_candidates(
                self.agent_working_dir,
                self.episode_id,
                self.task2_rows,
            )

        return self._build_observation()

    def _compute_risk_score(self, customer: Dict[str, Any]) -> int:
        core = customer["core_profile"]
        score = 0

        if core["tenure_months"] < 6:
            score += 4
        elif core["tenure_months"] < 12:
            score += 3
        elif core["tenure_months"] < 24:
            score += 1

        if core["balance_trend_90d"] < -0.50:
            score += 4
        elif core["balance_trend_90d"] < -0.30:
            score += 3
        elif core["balance_trend_90d"] < -0.10:
            score += 1

        if core["num_products"] == 1:
            score += 3
        elif core["num_products"] == 2:
            score += 1

        if core["login_frequency_30d"] == 0:
            score += 4
        elif core["login_frequency_30d"] <= 2:
            score += 3
        elif core["login_frequency_30d"] <= 5:
            score += 1

        if core["num_complaints_6m"] >= 3:
            score += 3
        elif core["num_complaints_6m"] >= 1:
            score += 1
        if core["last_complaint_days_ago"] < 14:
            score += 2

        if core["days_since_last_contact"] > 90:
            score += 2
        elif core["days_since_last_contact"] > 60:
            score += 1

        if core["txn_volume_change"] < -0.60:
            score += 3
        elif core["txn_volume_change"] < -0.40:
            score += 2
        elif core["txn_volume_change"] < -0.20:
            score += 1

        if core["min_balance_breaches"] >= 4:
            score += 2
        elif core["min_balance_breaches"] >= 2:
            score += 1

        if core["num_products"] >= 4:
            score -= 2
        if core["tenure_months"] > 60:
            score -= 1
        if core["has_loan"] == "Yes" and core["loan_dpd"] == 0:
            score -= 1
        if core["salary_credit"] == "Yes":
            score -= 1

        return score

    def _ensure_consistency(
        self, customer_id: str, true_risk: str, ml_signals: Dict[str, Any]
    ) -> Dict[str, Any]:
        true_risk_lower = true_risk.lower().replace("_", "")
        is_high_risk = "high" in true_risk_lower
        is_not_at_risk = "not" in true_risk_lower and "risk" in true_risk_lower

        lstm = ml_signals["lstm_output"]
        survival = ml_signals["survival_analysis"]
        anomaly = ml_signals["anomaly_detection"]
        rfm = ml_signals["rfm_segment"]
        sentiment = ml_signals["sentiment_analysis"]
        propensity = ml_signals["propensity_scores"]

        if is_high_risk:
            if lstm["churn_probability_trend"]["current"] < 0.60:
                lstm["churn_probability_trend"]["current"] = (
                    0.60 + (hash(customer_id) % 40) / 100
                )
                lstm["churn_probability_trend"]["month_1_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.6
                )
                lstm["churn_probability_trend"]["month_2_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.3
                )
                lstm["velocity"] = (
                    lstm["churn_probability_trend"]["current"]
                    - lstm["churn_probability_trend"]["month_2_ago"]
                )
                if lstm["velocity"] > 0.30:
                    lstm["trend_direction"] = "accelerating"

            if survival["survival_probability"]["6_months"] >= 0.30:
                survival["survival_probability"]["6_months"] = (
                    0.15 + (hash(customer_id) % 15) / 100
                )
                survival["survival_probability"]["3_months"] = (
                    survival["survival_probability"]["6_months"] ** 0.5
                )
                survival["survival_probability"]["1_month"] = survival[
                    "survival_probability"
                ]["6_months"] ** (1 / 6)
                survival["hazard_rate"] = (
                    1 - survival["survival_probability"]["1_month"]
                )
                survival["estimated_churn_in_months"] = 1 / max(
                    survival["hazard_rate"], 0.01
                )
                if survival["hazard_rate"] > 0.5:
                    survival["risk_interpretation"] = "critical"
                elif survival["hazard_rate"] > 0.3:
                    survival["risk_interpretation"] = "high"

            if not anomaly["is_anomalous"] and anomaly["anomaly_score"] < 0.30:
                anomaly["anomaly_score"] = 0.30 + (hash(customer_id) % 30) / 100
                anomaly["is_anomalous"] = True
                if (
                    "detected_anomalies" not in anomaly
                    or len(anomaly["detected_anomalies"]) == 0
                ):
                    anomaly["detected_anomalies"] = ["behavioral_decline"]
                if anomaly["anomaly_score"] > 0.60:
                    anomaly["severity"] = "critical"
                elif anomaly["anomaly_score"] > 0.35:
                    anomaly["severity"] = "moderate"

            if rfm["segment"] not in ["About_To_Churn", "Hibernating", "At_Risk"]:
                rfm["segment"] = "Hibernating"
                rfm["rfm_total"] = 4
                rfm["segment_churn_rate"] = 0.52

            if sentiment["sentiment_score"] >= 0.50:
                sentiment["sentiment_score"] = 0.30 + (hash(customer_id) % 20) / 100
                if sentiment["sentiment_score"] < 0.45:
                    sentiment["sentiment_label"] = "negative"
                if sentiment["sentiment_score"] < 0.30:
                    sentiment["interaction_quality"] = "poor"
                elif sentiment["sentiment_score"] < 0.50:
                    sentiment["interaction_quality"] = "fair"

            if propensity["overall_engagement"] >= 0.35:
                propensity["overall_engagement"] = 0.15 + (hash(customer_id) % 20) / 100
                propensity["engagement_label"] = "disengaged"

        elif is_not_at_risk:
            if lstm["churn_probability_trend"]["current"] >= 0.25:
                lstm["churn_probability_trend"]["current"] = (
                    0.10 + (hash(customer_id) % 15) / 100
                )
                lstm["churn_probability_trend"]["month_1_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.8
                )
                lstm["churn_probability_trend"]["month_2_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.5
                )
                lstm["velocity"] = (
                    lstm["churn_probability_trend"]["current"]
                    - lstm["churn_probability_trend"]["month_2_ago"]
                )
                if lstm["velocity"] < -0.10:
                    lstm["trend_direction"] = "improving"
                else:
                    lstm["trend_direction"] = "stable"

            if survival["survival_probability"]["6_months"] < 0.70:
                survival["survival_probability"]["6_months"] = (
                    0.70 + (hash(customer_id) % 20) / 100
                )
                survival["survival_probability"]["3_months"] = (
                    survival["survival_probability"]["6_months"] ** 0.5
                )
                survival["survival_probability"]["1_month"] = survival[
                    "survival_probability"
                ]["6_months"] ** (1 / 6)
                survival["hazard_rate"] = (
                    1 - survival["survival_probability"]["1_month"]
                )
                survival["estimated_churn_in_months"] = 1 / max(
                    survival["hazard_rate"], 0.01
                )
                if survival["hazard_rate"] <= 0.15:
                    survival["risk_interpretation"] = "low"

            if anomaly["anomaly_score"] >= 0.20:
                anomaly["anomaly_score"] = 0.05 + (hash(customer_id) % 15) / 100
                anomaly["is_anomalous"] = False
                anomaly["detected_anomalies"] = []
                anomaly["severity"] = "none"

            if rfm["segment"] not in ["Champions", "Loyal", "Potential_Loyalist"]:
                rfm["segment"] = "Loyal"
                rfm["rfm_total"] = 7
                rfm["segment_churn_rate"] = 0.08

            if sentiment["sentiment_score"] < 0.60:
                sentiment["sentiment_score"] = 0.65 + (hash(customer_id) % 25) / 100
                sentiment["sentiment_label"] = "positive"
                sentiment["interaction_quality"] = "excellent"

            if propensity["overall_engagement"] < 0.50:
                propensity["overall_engagement"] = 0.55 + (hash(customer_id) % 30) / 100
                propensity["engagement_label"] = "highly_engaged"

        elif true_risk == "medium_risk":
            if (
                lstm["churn_probability_trend"]["current"] < 0.30
                or lstm["churn_probability_trend"]["current"] > 0.60
            ):
                lstm["churn_probability_trend"]["current"] = (
                    0.35 + (hash(customer_id) % 25) / 100
                )
                lstm["churn_probability_trend"]["month_1_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.75
                )
                lstm["churn_probability_trend"]["month_2_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.50
                )
                lstm["velocity"] = (
                    lstm["churn_probability_trend"]["current"]
                    - lstm["churn_probability_trend"]["month_2_ago"]
                )
                lstm["trend_direction"] = "gradual"

            if (
                survival["survival_probability"]["6_months"] < 0.30
                or survival["survival_probability"]["6_months"] > 0.60
            ):
                survival["survival_probability"]["6_months"] = (
                    0.40 + (hash(customer_id) % 20) / 100
                )
                survival["survival_probability"]["3_months"] = (
                    survival["survival_probability"]["6_months"] ** 0.5
                )
                survival["survival_probability"]["1_month"] = survival[
                    "survival_probability"
                ]["6_months"] ** (1 / 6)
                survival["hazard_rate"] = (
                    1 - survival["survival_probability"]["1_month"]
                )
                survival["estimated_churn_in_months"] = 1 / max(
                    survival["hazard_rate"], 0.01
                )
                survival["risk_interpretation"] = "moderate"

            if rfm["segment"] not in ["At_Risk"]:
                rfm["segment"] = "At_Risk"
                rfm["rfm_total"] = 5
                rfm["segment_churn_rate"] = 0.38

            if (
                sentiment["sentiment_score"] < 0.45
                or sentiment["sentiment_score"] > 0.60
            ):
                sentiment["sentiment_score"] = 0.50 + (hash(customer_id) % 10) / 100
                sentiment["sentiment_label"] = "neutral"
                sentiment["interaction_quality"] = "fair"

        elif true_risk == "low_risk":
            if lstm["churn_probability_trend"]["current"] >= 0.30:
                lstm["churn_probability_trend"]["current"] = (
                    0.15 + (hash(customer_id) % 15) / 100
                )
                lstm["churn_probability_trend"]["month_1_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.70
                )
                lstm["churn_probability_trend"]["month_2_ago"] = (
                    lstm["churn_probability_trend"]["current"] * 0.45
                )
                lstm["velocity"] = (
                    lstm["churn_probability_trend"]["current"]
                    - lstm["churn_probability_trend"]["month_2_ago"]
                )
                lstm["trend_direction"] = "stable"

            if survival["survival_probability"]["6_months"] < 0.60:
                survival["survival_probability"]["6_months"] = (
                    0.65 + (hash(customer_id) % 20) / 100
                )
                survival["survival_probability"]["3_months"] = (
                    survival["survival_probability"]["6_months"] ** 0.5
                )
                survival["survival_probability"]["1_month"] = survival[
                    "survival_probability"
                ]["6_months"] ** (1 / 6)
                survival["hazard_rate"] = (
                    1 - survival["survival_probability"]["1_month"]
                )
                survival["estimated_churn_in_months"] = 1 / max(
                    survival["hazard_rate"], 0.01
                )
                survival["risk_interpretation"] = "low"

            if rfm["segment"] not in ["Potential_Loyalist", "Loyal"]:
                rfm["segment"] = "Potential_Loyalist"
                rfm["rfm_total"] = 6
                rfm["segment_churn_rate"] = 0.15

            if sentiment["sentiment_score"] < 0.55:
                sentiment["sentiment_score"] = 0.60 + (hash(customer_id) % 15) / 100
                sentiment["sentiment_label"] = "positive"
                sentiment["interaction_quality"] = "good"

        return ml_signals

    def _build_observation(self) -> ChurnObservation:
        remaining_ids = [c["customer_id"] for c in self.customers[self.step_count :]]

        current_customer = {}
        if self.step_count < len(self.customers):
            current_customer = self.customers[self.step_count].copy()
            customer_id = current_customer["customer_id"]
            current_customer["ml_signals"] = self.customer_ml_signals.get(
                customer_id, {}
            )

        last_feedback = None
        if self.step_count > 0 and len(self.classified_results) > 0:
            last_result = self.classified_results[-1]
            last_feedback = f"Predicted: {last_result['predicted_tier']}, True: {last_result['true_tier']}, Reward: {last_result['step_reward']}"

        task_description = "Classify 10 bank customers by churn risk tier using behavioral and ML signals."
        if self.task_id == 2:
            task_description = "Resolve campaign collisions from overlapping CSV files and output master_campaign_deployment.json based on priority_rules.txt."

        if self.task_id == 2 and self.task2_customer_order:
            current_id = self.task2_customer_order[self.step_count % len(self.task2_customer_order)]
            expected = self.task2_expected_by_customer.get(current_id, {})
            current_customer = {
                "customer_id": current_id,
                "task2_context": {
                    "customer_id": current_id,
                    "candidate_offers": expected.get("candidate_offers", []),
                    "rules": "Retention priority > Credit Card > Auto Loan; if same priority choose higher offer_value",
                },
            }

        return ChurnObservation(
            task_id=self.task_id,
            task_description=task_description,
            step_number=self.step_count + 1,
            max_steps=self.max_steps,
            current_customer=current_customer,
            remaining_customer_ids=remaining_ids,
            classified_so_far=self.classified_results.copy(),
            cumulative_reward=round(self.cumulative_reward, 4),
            last_feedback=last_feedback,
            campaign_input_files={
                "auto_loan": self.task2_files.get("auto_loan", ""),
                "credit_card": self.task2_files.get("credit_card", ""),
                "retention": self.task2_files.get("retention", ""),
            }
            if self.task_id == 2
            else None,
            priority_rules_file=self.task2_files.get("priority_rules")
            if self.task_id == 2
            else None,
            expected_output_file=os.path.join(
                self.agent_working_dir, "master_campaign_deployment.json"
            )
            if self.task_id == 2
            else None,
            database_file=self.db_path,
        )

    def step(
        self, action: ChurnAction
    ) -> tuple[ChurnObservation, float, bool, Dict[str, Any]]:
        if not self.is_reset:
            raise RuntimeError("Call reset() first")

        if self.task_id == 2:
            if self.task2_customer_order:
                current_id = self.task2_customer_order[self.step_count % len(self.task2_customer_order)]
                expected = self.task2_expected_by_customer.get(current_id, {})
                chosen_source = str(action.selected_campaign_source or "").strip()
                chosen_offer = (
                    float(action.selected_offer_value)
                    if action.selected_offer_value is not None
                    else 0.0
                )

                step_reward = 0.0
                exp_source = str(expected.get("campaign_source", "")).strip()
                exp_offer = float(expected.get("offer_value", 0.0))

                if chosen_source and exp_source and chosen_source == exp_source:
                    if abs(chosen_offer - exp_offer) <= 1e-3:
                        step_reward = 1.0
                    else:
                        step_reward = 0.7
                elif chosen_source:
                    valid_sources = {
                        str(r.get("campaign_source", "")).strip()
                        for r in expected.get("candidate_offers", [])
                    }
                    step_reward = 0.3 if chosen_source in valid_sources else 0.1

                self.task2_output_map[current_id] = {
                    "customer_id": current_id,
                    "campaign_source": chosen_source or exp_source,
                    "offer_value": chosen_offer if chosen_offer > 0 else exp_offer,
                }
            else:
                step_reward = 0.0

            next_step_index = min(self.step_count + 1, self.max_steps)

            self.step_count = next_step_index
            self.cumulative_reward += step_reward
            done = self.step_count >= self.max_steps

            final_score = None
            if done:
                output_path = os.path.join(
                    self.agent_working_dir, "master_campaign_deployment.json"
                )
                with open(output_path, "w", encoding="utf-8") as f:
                    import json

                    json.dump(
                        [
                            self.task2_output_map[cid]
                            for cid in sorted(self.task2_output_map.keys())
                        ],
                        f,
                        indent=2,
                    )

                final_score = grade_task_2(
                    working_dir=self.agent_working_dir,
                    campaign_file_paths=self.task2_files,
                )
                if final_score >= 0.5:
                    deployment_rows = read_task2_output(self.agent_working_dir)
                    persist_task2_deployment(
                        self.agent_working_dir, self.episode_id, deployment_rows
                    )

            return (
                self._build_observation(),
                step_reward,
                done,
                {
                    "final_score": final_score if done else None,
                    "expected_output_file": os.path.join(
                        self.agent_working_dir, "master_campaign_deployment.json"
                    ),
                },
            )

        info = {}
        reward = 0.0

        remaining_ids = [c["customer_id"] for c in self.customers[self.step_count :]]
        expected_customer_id = remaining_ids[0] if remaining_ids else None

        if action.customer_id not in remaining_ids:
            if len(remaining_ids) >= 1 and expected_customer_id:
                action.customer_id = expected_customer_id
                action_sent_wrong_id = True
            else:
                reward = -0.10
                info = {"error": "invalid customer_id", "valid_ids": remaining_ids}

                self.step_count += 1
                self.cumulative_reward += reward

                self.classified_results.append(
                    {
                        "customer_id": action.customer_id,
                        "predicted_tier": action.risk_tier,
                        "true_tier": "error",
                        "step_reward": reward,
                        "error": "invalid customer_id",
                    }
                )

                done = self.step_count >= self.max_steps

                if done:
                    final_score = grade_task1(self.classified_results)
                    persist_task1_results(
                        self.agent_working_dir, self.episode_id, self.classified_results
                    )
                    info["final_score"] = final_score

                return self._build_observation(), reward, done, info
        else:
            action_sent_wrong_id = False

        valid_tiers = ["high_risk", "medium_risk", "low_risk", "not_at_risk"]
        if action.risk_tier not in valid_tiers:
            reward = -0.10
            info = {"error": "invalid risk_tier"}

            self.step_count += 1
            self.cumulative_reward += reward

            self.classified_results.append(
                {
                    "customer_id": action.customer_id,
                    "predicted_tier": action.risk_tier,
                    "true_tier": "error",
                    "step_reward": reward,
                    "error": "invalid risk_tier",
                }
            )

            done = self.step_count >= self.max_steps

            if done:
                final_score = grade_task1(self.classified_results)
                persist_task1_results(
                    self.agent_working_dir, self.episode_id, self.classified_results
                )
                info["final_score"] = final_score

            return self._build_observation(), reward, done, info

        if not action.reasoning or action.reasoning.strip() == "":
            reward = max(
                0.0,
                compute_step_reward(
                    action.risk_tier, self.customer_truth[action.customer_id]
                )
                - 0.05,
            )
        else:
            reward = compute_step_reward(
                action.risk_tier, self.customer_truth[action.customer_id]
            )

        confidence = action.confidence
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0

        result = {
            "customer_id": action.customer_id,
            "predicted_tier": action.risk_tier,
            "true_tier": self.customer_truth[action.customer_id],
            "step_reward": reward,
        }
        if action_sent_wrong_id:
            info["auto_corrected_customer_id"] = True

        self.classified_results.append(result)

        self.step_count += 1
        self.cumulative_reward += reward

        done = self.step_count >= self.max_steps

        if done:
            final_score = grade_task1(self.classified_results)
            persist_task1_results(
                self.agent_working_dir, self.episode_id, self.classified_results
            )
            info["final_score"] = final_score

        return self._build_observation(), reward, done, info

    def state(self) -> ChurnState:
        if not self.is_reset:
            return ChurnState()

        high_risk_found = sum(
            1 for r in self.classified_results if r["true_tier"] == "high_risk"
        )
        dangerous_misses = sum(
            1
            for r in self.classified_results
            if r["true_tier"] == "high_risk" and r["predicted_tier"] == "not_at_risk"
        )

        return ChurnState(
            episode_id=self.episode_id,
            task_id=self.task_id,
            step_count=self.step_count,
            total_customers=self.max_steps,
            classified_count=len(self.classified_results),
            current_score=round(self.cumulative_reward, 4),
            high_risk_found=high_risk_found,
            dangerous_misses=dangerous_misses,
        )
