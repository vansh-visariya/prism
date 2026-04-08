TIERS = ["not_at_risk", "low_risk", "medium_risk", "high_risk"]


def compute_step_reward(predicted_tier: str, true_tier: str) -> float:
    pred_idx = TIERS.index(predicted_tier)
    true_idx = TIERS.index(true_tier)

    if pred_idx == true_idx:
        return 1.00

    if pred_idx < true_idx:
        if true_tier == "high_risk":
            return 0.00
        elif true_tier == "medium_risk":
            return 0.20
        else:
            return 0.40

    distance = pred_idx - true_idx
    if distance == 1:
        return 0.70
    else:
        return 0.50
