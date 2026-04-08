import csv
import json
import os


def grade_task1(episode_results):
    """
    episode_results = list of:
    {customer_id, predicted_tier, true_tier, step_reward}
    """

    if not episode_results:
        return 0.0

    total_reward = sum(r["step_reward"] for r in episode_results)
    base_score = total_reward / len(episode_results)

    high_risk_customers = [r for r in episode_results if r["true_tier"] == "high_risk"]
    if high_risk_customers:
        all_hr_correct = all(
            r["predicted_tier"] == "high_risk" for r in high_risk_customers
        )
        if all_hr_correct:
            base_score = min(1.0, base_score + 0.10)

    if all(r["step_reward"] == 1.0 for r in episode_results):
        base_score = min(1.0, base_score + 0.05)

    dangerous_misses = [
        r
        for r in episode_results
        if r["true_tier"] == "high_risk" and r["predicted_tier"] == "not_at_risk"
    ]
    base_score -= len(dangerous_misses) * 0.20

    return round(max(0.01, min(0.99, base_score)), 4)


def _grade_task_2_internal(
    working_dir: str = ".", campaign_file_paths: dict | None = None
) -> float:
    output_path = os.path.join(working_dir, "master_campaign_deployment.json")
    if not os.path.exists(output_path):
        return 0.0

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            output_data = json.load(f)
    except Exception:
        return 0.0

    if isinstance(output_data, dict):
        if isinstance(output_data.get("deployments"), list):
            deployments = output_data["deployments"]
        elif isinstance(output_data.get("records"), list):
            deployments = output_data["records"]
        else:
            deployments = []
    elif isinstance(output_data, list):
        deployments = output_data
    else:
        deployments = []

    if not deployments:
        return 0.0

    customer_ids = [str(item.get("customer_id", "")).strip() for item in deployments]
    if any(not cid for cid in customer_ids):
        return 0.0
    if len(customer_ids) != len(set(customer_ids)):
        return 0.0

    files = campaign_file_paths or {
        "auto_loan": os.path.join(working_dir, "auto_loan.csv"),
        "credit_card": os.path.join(working_dir, "credit_card.csv"),
        "retention": os.path.join(working_dir, "retention.csv"),
    }

    source_rows = []
    for key in ["auto_loan", "credit_card", "retention"]:
        path = files.get(key, os.path.join(working_dir, f"{key}.csv"))
        if not os.path.exists(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    customer_id = str(row.get("customer_id", "")).strip()
                    if not customer_id:
                        continue
                    offer = row.get("offer_value", 0)
                    try:
                        offer_value = float(offer)
                    except (TypeError, ValueError):
                        offer_value = 0.0
                    campaign_source = str(row.get("campaign_source", "")).strip()
                    source_rows.append(
                        {
                            "customer_id": customer_id,
                            "offer_value": offer_value,
                            "campaign_source": campaign_source,
                        }
                    )
        except Exception:
            return 0.0

    if not source_rows:
        return 0.0

    priority = {"retention": 3, "credit_card": 2, "auto_loan": 1}
    expected = {}
    for row in source_rows:
        cid = row["customer_id"]
        current = expected.get(cid)
        row_priority = priority.get(row["campaign_source"], 0)
        if current is None:
            expected[cid] = row
            continue

        current_priority = priority.get(current["campaign_source"], 0)
        if row_priority > current_priority:
            expected[cid] = row
        elif row_priority == current_priority and row["offer_value"] > current["offer_value"]:
            expected[cid] = row

    output_map = {}
    for row in deployments:
        cid = str(row.get("customer_id", "")).strip()
        try:
            offer_value = float(row.get("offer_value", 0))
        except (TypeError, ValueError):
            offer_value = 0.0
        output_map[cid] = {
            "customer_id": cid,
            "offer_value": offer_value,
            "campaign_source": str(row.get("campaign_source", "")).strip(),
        }

    # Deduplicated output exists at this point; if assignment quality is not perfect,
    # return partial credit.
    for cid, expected_row in expected.items():
        got = output_map.get(cid)
        if not got:
            return 0.5
        if got["campaign_source"] != expected_row["campaign_source"]:
            return 0.5
        if abs(got["offer_value"] - expected_row["offer_value"]) > 1e-6:
            return 0.5

    if set(output_map.keys()) != set(expected.keys()):
        return 0.5

    return 1.0


def grade_task_2(working_dir: str = ".", campaign_file_paths: dict | None = None) -> float:
    score = _grade_task_2_internal(working_dir, campaign_file_paths)
    return float(max(0.01, min(0.99, score)))


def get_task2_expected_map(
    working_dir: str = ".", campaign_file_paths: dict | None = None
) -> dict:
    files = campaign_file_paths or {
        "auto_loan": os.path.join(working_dir, "auto_loan.csv"),
        "credit_card": os.path.join(working_dir, "credit_card.csv"),
        "retention": os.path.join(working_dir, "retention.csv"),
    }

    source_rows = []
    for key in ["auto_loan", "credit_card", "retention"]:
        path = files.get(key, os.path.join(working_dir, f"{key}.csv"))
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                customer_id = str(row.get("customer_id", "")).strip()
                if not customer_id:
                    continue
                try:
                    offer_value = float(row.get("offer_value", 0))
                except (TypeError, ValueError):
                    offer_value = 0.0
                source_rows.append(
                    {
                        "customer_id": customer_id,
                        "offer_value": offer_value,
                        "campaign_source": str(row.get("campaign_source", "")).strip(),
                    }
                )

    priority = {"retention": 3, "credit_card": 2, "auto_loan": 1}
    expected = {}
    for row in source_rows:
        cid = row["customer_id"]
        current = expected.get(cid)
        row_priority = priority.get(row["campaign_source"], 0)
        if current is None:
            expected[cid] = row
            continue
        current_priority = priority.get(current["campaign_source"], 0)
        if row_priority > current_priority:
            expected[cid] = row
        elif row_priority == current_priority and row["offer_value"] > current["offer_value"]:
            expected[cid] = row

    return expected

