import random
import math
from typing import Dict, Any, List


def clamp(value: float, min_val: float, max_val: float) -> float:
    return max(min_val, min(max_val, value))


def simulate_lstm_output(
    customer: Dict[str, Any], history: Dict[str, Any]
) -> Dict[str, Any]:
    balance_history = history["balance_history"]
    login_history = history["login_history"]

    month1_balance = balance_history[0]["balance"]
    month3_balance = balance_history[2]["balance"]
    month1_logins = login_history[0]["logins"]
    month3_logins = login_history[2]["logins"]

    if month1_balance > 0:
        balance_decay = (month1_balance - month3_balance) / month1_balance
    else:
        balance_decay = 0.0

    if month1_logins > 0:
        login_decay = (month1_logins - month3_logins) / month1_logins
    else:
        login_decay = 0.5 if month3_logins == 0 else 1.0

    base_risk = (balance_decay * 0.6) + (login_decay * 0.4)
    base_risk = clamp(base_risk, 0.0, 1.0)

    month_2_ago = round(base_risk * 0.35 + 0.05, 3)
    month_1_ago = round(base_risk * 0.65 + 0.05, 3)
    current = round(min(base_risk * 1.05 + 0.10, 0.99), 3)

    velocity = current - month_2_ago

    if velocity > 0.30:
        trend_direction = "accelerating"
    elif velocity > 0.10:
        trend_direction = "gradual"
    elif velocity < -0.10:
        trend_direction = "improving"
    else:
        trend_direction = "stable"

    return {
        "churn_probability_trend": {
            "month_2_ago": month_2_ago,
            "month_1_ago": month_1_ago,
            "current": current,
        },
        "trend_direction": trend_direction,
        "velocity": round(velocity, 3),
    }


def simulate_xai_explanation(
    customer: Dict[str, Any], true_risk_score: int
) -> Dict[str, Any]:
    core = customer["core_profile"]

    tenure = core["tenure_months"]
    if tenure < 6:
        tenure_contribution = 0.31
    elif tenure < 12:
        tenure_contribution = 0.24
    elif tenure < 24:
        tenure_contribution = 0.08
    elif tenure > 60:
        tenure_contribution = -0.06
    else:
        tenure_contribution = 0.0

    balance_trend = core["balance_trend_90d"]
    if balance_trend < -0.50:
        balance_trend_contribution = 0.33
    elif balance_trend < -0.30:
        balance_trend_contribution = 0.26
    elif balance_trend < -0.10:
        balance_trend_contribution = 0.09
    elif balance_trend > 0.10:
        balance_trend_contribution = -0.07
    else:
        balance_trend_contribution = 0.0

    login_freq = core["login_frequency_30d"]
    if login_freq == 0:
        login_contribution = 0.30
    elif login_freq <= 2:
        login_contribution = 0.24
    elif login_freq <= 5:
        login_contribution = 0.09
    elif login_freq >= 15:
        login_contribution = -0.08
    else:
        login_contribution = 0.0

    num_products = core["num_products"]
    if num_products == 1:
        products_contribution = 0.20
    elif num_products == 2:
        products_contribution = 0.07
    elif num_products >= 4:
        products_contribution = -0.12
    else:
        products_contribution = 0.0

    complaints = core["num_complaints_6m"]
    last_complaint = core["last_complaint_days_ago"]
    complaint_contribution = 0.0
    if complaints >= 3:
        complaint_contribution += 0.22
    elif complaints >= 1:
        complaint_contribution += 0.10
    if last_complaint < 14:
        complaint_contribution += 0.12

    contact_gap = core["days_since_last_contact"]
    if contact_gap > 90:
        contact_gap_contribution = 0.14
    elif contact_gap > 60:
        contact_gap_contribution = 0.09
    else:
        contact_gap_contribution = 0.0

    txn_change = core["txn_volume_change"]
    if txn_change < -0.60:
        txn_volume_contribution = 0.25
    elif txn_change < -0.40:
        txn_volume_contribution = 0.18
    elif txn_change < -0.20:
        txn_volume_contribution = 0.09
    else:
        txn_volume_contribution = 0.0

    feature_contributions = {
        "tenure_months": round(tenure_contribution, 2),
        "balance_trend_90d": round(balance_trend_contribution, 2),
        "login_frequency_30d": round(login_contribution, 2),
        "num_products": round(products_contribution, 2),
        "num_complaints_6m": round(complaint_contribution, 2),
        "days_since_last_contact": round(contact_gap_contribution, 2),
        "txn_volume_change": round(txn_volume_contribution, 2),
    }

    positive_contributions = {k: v for k, v in feature_contributions.items() if v > 0}
    negative_contributions = {k: v for k, v in feature_contributions.items() if v < 0}

    top_driver = (
        max(positive_contributions, key=positive_contributions.get)
        if positive_contributions
        else None
    )
    top_protective = (
        min(negative_contributions, key=negative_contributions.get)
        if negative_contributions
        else None
    )

    driver_texts = {
        "tenure_months": "Customer's short tenure is the strongest churn signal.",
        "balance_trend_90d": "Customer's rapidly declining balance is the strongest churn signal.",
        "login_frequency_30d": "Customer's low digital engagement is the strongest churn signal.",
        "num_products": "Customer's single product relationship increases churn risk.",
        "num_complaints_6m": "Customer's complaint history is the strongest churn signal.",
        "days_since_last_contact": "Customer's prolonged contact gap is the strongest churn signal.",
        "txn_volume_change": "Customer's declining transaction volume is the strongest churn signal.",
    }

    explanation_text = (
        driver_texts.get(top_driver, "Multiple risk factors detected.")
        if top_driver
        else "Customer shows stable engagement patterns."
    )

    return {
        "feature_contributions": feature_contributions,
        "top_churn_driver": top_driver,
        "top_protective_factor": top_protective,
        "explanation_text": explanation_text,
    }


def simulate_survival_output(customer: Dict[str, Any]) -> Dict[str, Any]:
    core = customer["core_profile"]

    base_hazard = 0.04
    hazard = base_hazard

    if core["tenure_months"] < 12:
        hazard *= 2.8
    if core["balance_trend_90d"] < -0.30:
        hazard *= 2.2
    if core["num_products"] == 1:
        hazard *= 1.9
    if core["num_complaints_6m"] >= 2:
        hazard *= 1.7
    if core["login_frequency_30d"] <= 2:
        hazard *= 2.0
    if core["txn_volume_change"] < -0.4:
        hazard *= 1.6
    if core["salary_credit"] == "Yes":
        hazard *= 0.7
    if core["num_products"] >= 4:
        hazard *= 0.6
    if core["tenure_months"] > 60:
        hazard *= 0.8

    hazard = clamp(hazard, 0.01, 0.95)

    survival_1m = round((1 - hazard) ** 1, 3)
    survival_3m = round((1 - hazard) ** 3, 3)
    survival_6m = round((1 - hazard) ** 6, 3)

    estimated_churn = round(1 / hazard, 1)

    if hazard > 0.5:
        risk_interpretation = "critical"
    elif hazard > 0.3:
        risk_interpretation = "high"
    elif hazard > 0.15:
        risk_interpretation = "moderate"
    else:
        risk_interpretation = "low"

    return {
        "survival_probability": {
            "1_month": survival_1m,
            "3_months": survival_3m,
            "6_months": survival_6m,
        },
        "estimated_churn_in_months": estimated_churn,
        "hazard_rate": round(hazard, 4),
        "risk_interpretation": risk_interpretation,
    }


def simulate_rfm_segment(customer: Dict[str, Any]) -> Dict[str, Any]:
    core = customer["core_profile"]

    last_txn_days = core["last_txn_days_ago"]
    if last_txn_days < 7:
        r_score = 3
    elif last_txn_days < 30:
        r_score = 2
    else:
        r_score = 1

    txn_count = core["txn_count_30d"]
    if txn_count > 15:
        f_score = 3
    elif txn_count > 5:
        f_score = 2
    else:
        f_score = 1

    avg_balance = core["avg_monthly_balance"]
    if avg_balance > 100000:
        m_score = 3
    elif avg_balance > 25000:
        m_score = 2
    else:
        m_score = 1

    rfm_total = r_score + f_score + m_score

    if rfm_total >= 8:
        segment = "Champions"
    elif rfm_total >= 7:
        segment = "Loyal"
    elif rfm_total >= 6:
        segment = "Potential_Loyalist"
    elif rfm_total >= 5:
        segment = "At_Risk"
    elif rfm_total >= 4:
        segment = "Hibernating"
    else:
        segment = "About_To_Churn"

    segment_churn_rates = {
        "Champions": 0.04,
        "Loyal": 0.08,
        "Potential_Loyalist": 0.15,
        "At_Risk": 0.38,
        "Hibernating": 0.52,
        "About_To_Churn": 0.71,
    }

    return {
        "segment": segment,
        "r_score": r_score,
        "f_score": f_score,
        "m_score": m_score,
        "rfm_total": rfm_total,
        "segment_churn_rate": segment_churn_rates[segment],
    }


def simulate_anomaly_detection(
    customer: Dict[str, Any], history: Dict[str, Any]
) -> Dict[str, Any]:
    core = customer["core_profile"]
    login_history = history["login_history"]
    balance_history = history["balance_history"]

    total_score = 0.0
    detected_anomalies = []

    m1_logins = login_history[0]["logins"]
    m2_logins = login_history[1]["logins"]
    m3_logins = login_history[2]["logins"]
    baseline_logins = (m1_logins + m2_logins) / 2
    current_logins = m3_logins

    if baseline_logins > 0:
        drop_rate = (baseline_logins - current_logins) / baseline_logins
        if drop_rate > 0.70:
            detected_anomalies.append("sudden_login_drop")
            total_score += 0.30

    m1_balance = balance_history[0]["balance"]
    m3_balance = balance_history[2]["balance"]
    if m1_balance > 0:
        drop = (m1_balance - m3_balance) / m1_balance
        if drop > 0.50:
            detected_anomalies.append("large_balance_withdrawal")
            total_score += 0.35

    if core["txn_count_30d"] == 0 and core["tenure_months"] > 6:
        detected_anomalies.append("transaction_freeze")
        total_score += 0.25

    if core["num_complaints_6m"] >= 3 and core["last_complaint_days_ago"] < 30:
        detected_anomalies.append("complaint_spike")
        total_score += 0.20

    if core["upi_active"] == "No" and core["login_frequency_30d"] > 0:
        detected_anomalies.append("upi_deactivation")
        total_score += 0.15

    anomaly_score = round(min(total_score, 1.0), 3)
    is_anomalous = anomaly_score > 0.25

    if anomaly_score > 0.60:
        severity = "critical"
    elif anomaly_score > 0.35:
        severity = "moderate"
    elif anomaly_score > 0.15:
        severity = "mild"
    else:
        severity = "none"

    return {
        "anomaly_score": anomaly_score,
        "is_anomalous": is_anomalous,
        "detected_anomalies": detected_anomalies,
        "severity": severity,
    }


def simulate_sentiment(customer: Dict[str, Any]) -> Dict[str, Any]:
    core = customer["core_profile"]

    base = 0.75

    num_complaints = core["num_complaints_6m"]
    base -= num_complaints * 0.10

    last_complaint_days = core["last_complaint_days_ago"]
    if last_complaint_days < 14:
        base -= 0.18
    elif last_complaint_days < 30:
        base -= 0.10

    min_balance_breaches = core["min_balance_breaches"]
    if min_balance_breaches >= 3:
        base -= 0.08

    days_since_contact = core["days_since_last_contact"]
    if days_since_contact > 90:
        base -= 0.06

    nps_score = core["nps_score"]
    if nps_score is not None:
        contribution = (nps_score / 10) * 0.25
        base = base * 0.75 + contribution

    sentiment_score = clamp(base, 0.0, 1.0)

    if sentiment_score >= 0.70:
        sentiment_label = "positive"
    elif sentiment_score >= 0.45:
        sentiment_label = "neutral"
    elif sentiment_score >= 0.25:
        sentiment_label = "negative"
    else:
        sentiment_label = "highly_negative"

    rng = random.Random(id(customer))
    confidence = round(0.70 + rng.uniform(0, 0.25), 3)
    confidence = clamp(confidence, 0.65, 0.95)

    if sentiment_score < 0.30:
        interaction_quality = "poor"
    elif sentiment_score < 0.50:
        interaction_quality = "fair"
    elif sentiment_score < 0.70:
        interaction_quality = "good"
    else:
        interaction_quality = "excellent"

    return {
        "sentiment_score": round(sentiment_score, 3),
        "sentiment_label": sentiment_label,
        "confidence": confidence,
        "interaction_quality": interaction_quality,
    }


def simulate_propensity_scores(customer: Dict[str, Any]) -> Dict[str, Any]:
    core = customer["core_profile"]

    avg_balance = core["avg_monthly_balance"]
    tenure = core["tenure_months"]
    num_products = core["num_products"]
    credit_score = core["credit_score"]
    age = core["age"]
    is_senior = core["is_senior_citizen"]

    fd_propensity = (avg_balance / 300000) * 0.45 + (tenure / 120) * 0.30
    if num_products >= 2:
        fd_propensity += 0.25
    fd_propensity = clamp(fd_propensity, 0.0, 1.0)

    credit_card_propensity = (credit_score / 900) * 0.55
    if core["salary_credit"] == "Yes":
        credit_card_propensity += 0.25
    if core["has_credit_card"] == "No":
        credit_card_propensity += 0.20
    else:
        credit_card_propensity -= 0.30
    credit_card_propensity = clamp(credit_card_propensity, 0.0, 1.0)

    loan_propensity = (credit_score / 900) * 0.45
    if core["has_loan"] == "No":
        loan_propensity += 0.30
    else:
        loan_propensity -= 0.25
    if tenure > 24:
        loan_propensity += 0.25
    loan_propensity = clamp(loan_propensity, 0.0, 1.0)

    insurance_propensity = (age / 70) * 0.35
    if core["has_insurance"] == "No":
        insurance_propensity += 0.30
    else:
        insurance_propensity -= 0.20
    if num_products >= 3:
        insurance_propensity += 0.20
    if is_senior == 1:
        insurance_propensity += 0.15
    insurance_propensity = clamp(insurance_propensity, 0.0, 1.0)

    overall_engagement = (
        fd_propensity + credit_card_propensity + loan_propensity + insurance_propensity
    ) / 4

    if overall_engagement > 0.55:
        engagement_label = "highly_engaged"
    elif overall_engagement > 0.30:
        engagement_label = "moderately_engaged"
    else:
        engagement_label = "disengaged"

    return {
        "fd_propensity": round(fd_propensity, 3),
        "credit_card_propensity": round(credit_card_propensity, 3),
        "loan_propensity": round(loan_propensity, 3),
        "insurance_propensity": round(insurance_propensity, 3),
        "overall_engagement": round(overall_engagement, 3),
        "engagement_label": engagement_label,
    }


def simulate_peer_benchmark(customer: Dict[str, Any]) -> Dict[str, Any]:
    core = customer["core_profile"]

    tenure = core["tenure_months"]
    if tenure < 12:
        tenure_band = "new"
        cohort_size = 1200
        cohort_avg_balance = 25000
    elif tenure < 36:
        tenure_band = "developing"
        cohort_size = 890
        cohort_avg_balance = 55000
    elif tenure < 72:
        tenure_band = "established"
        cohort_size = 650
        cohort_avg_balance = 90000
    else:
        tenure_band = "loyal"
        cohort_size = 420
        cohort_avg_balance = 140000

    customer_balance = core["avg_monthly_balance"]
    ratio = customer_balance / cohort_avg_balance
    balance_percentile = clamp(int(ratio * 50), 1, 99)

    customer_logins = core["login_frequency_30d"]
    cohort_avg_logins = 12
    ratio = customer_logins / cohort_avg_logins
    login_percentile = clamp(int(ratio * 50), 1, 99)

    customer_products = core["num_products"]
    cohort_avg_products = 2.5
    ratio = customer_products / cohort_avg_products
    products_percentile = clamp(int(ratio * 50), 1, 99)

    overall_percentile = (
        balance_percentile + login_percentile + products_percentile
    ) // 3

    below_avg_count = sum(
        [balance_percentile < 40, login_percentile < 40, products_percentile < 40]
    )
    above_avg_count = sum(
        [balance_percentile > 60, login_percentile > 60, products_percentile > 60]
    )

    if below_avg_count == 3:
        vs_cohort_summary = "Below average on 3 of 3 metrics vs peers"
    elif below_avg_count == 2:
        vs_cohort_summary = "Below average on 2 of 3 metrics vs peers"
    elif below_avg_count == 1:
        vs_cohort_summary = "Below average on 1 of 3 metrics vs peers"
    elif above_avg_count >= 2:
        vs_cohort_summary = "Above average on 2 of 3 metrics vs peers"
    elif above_avg_count == 1:
        vs_cohort_summary = "Above average on 1 of 3 metrics vs peers"
    else:
        vs_cohort_summary = "Average across all metrics vs peers"

    return {
        "tenure_band": tenure_band,
        "cohort_size": cohort_size,
        "balance_percentile": balance_percentile,
        "login_percentile": login_percentile,
        "products_percentile": products_percentile,
        "overall_percentile": overall_percentile,
        "vs_cohort_summary": vs_cohort_summary,
    }


def generate_all_ml_signals(
    customer: Dict[str, Any], true_risk_score: int
) -> Dict[str, Any]:
    core = customer["core_profile"]
    temporal = customer["temporal_history"]

    lstm_output = simulate_lstm_output(customer, temporal)
    xai_explanation = simulate_xai_explanation(customer, true_risk_score)
    survival_analysis = simulate_survival_output(customer)
    rfm_segment = simulate_rfm_segment(customer)
    anomaly_detection = simulate_anomaly_detection(customer, temporal)
    sentiment_analysis = simulate_sentiment(customer)
    propensity_scores = simulate_propensity_scores(customer)
    peer_benchmarking = simulate_peer_benchmark(customer)

    return {
        "lstm_output": lstm_output,
        "xai_explanation": xai_explanation,
        "survival_analysis": survival_analysis,
        "rfm_segment": rfm_segment,
        "anomaly_detection": anomaly_detection,
        "sentiment_analysis": sentiment_analysis,
        "propensity_scores": propensity_scores,
        "peer_benchmarking": peer_benchmarking,
    }
