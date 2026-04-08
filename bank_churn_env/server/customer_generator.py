import random
import os
import csv
from typing import List, Dict, Any


class CustomerGenerator:
    def __init__(self, seed: int = 42):
        self.seed = seed

    def generate_customers(self, num_customers: int = 10) -> List[Dict[str, Any]]:
        rng = random.Random(self.seed)
        customers = []

        for i in range(num_customers):
            customer_id = f"C_{str(i + 1).zfill(3)}"
            customer = self._generate_single_customer(rng, customer_id, i)
            customers.append(customer)

        return customers

    def _generate_single_customer(
        self, rng: random.Random, customer_id: str, index: int
    ) -> Dict[str, Any]:
        age = rng.randint(25, 70)
        gender = rng.choice(["Male", "Female"])
        city_tier = rng.choice([1, 2, 3])
        occupation = rng.choice(["salaried", "self_employed", "retired", "student"])
        tenure_months = rng.randint(1, 120)
        account_type = rng.choice(["savings", "current", "salary"])
        avg_monthly_balance = rng.uniform(1000, 500000)
        balance_trend_90d = rng.uniform(-0.70, 0.30)
        min_balance_breaches = rng.randint(0, 8)
        num_products = rng.randint(1, 5)
        login_frequency_30d = rng.randint(0, 30)
        txn_count_30d = rng.randint(0, 50)
        txn_volume_change = rng.uniform(-0.80, 0.40)
        last_txn_days_ago = rng.randint(0, 180)
        days_since_last_contact = rng.randint(0, 180)
        num_complaints_6m = rng.randint(0, 6)
        last_complaint_days_ago = rng.randint(0, 365)
        salary_credit = rng.choice(["Yes", "No"])
        credit_score = rng.randint(300, 900)
        has_credit_card = rng.choice(["Yes", "No"])
        has_fd = rng.choice(["Yes", "No"])
        has_loan = rng.choice(["Yes", "No"])
        loan_dpd = rng.randint(0, 90) if has_loan == "Yes" else 0
        has_insurance = rng.choice(["Yes", "No"])
        kyc_status = rng.choice(["active", "expired", "pending"])
        account_status = rng.choice(["active", "dormant", "frozen"])
        is_senior_citizen = 1 if age >= 60 else 0
        nps_score = rng.randint(0, 10) if rng.random() > 0.3 else None
        upi_active = rng.choice(["Yes", "No"])

        balance_history = self._generate_balance_history(
            rng, avg_monthly_balance, balance_trend_90d
        )
        login_history = self._generate_login_history(rng, login_frequency_30d)
        txn_history = self._generate_txn_history(
            rng, txn_count_30d, avg_monthly_balance, txn_volume_change
        )

        core_profile = {
            "age": age,
            "gender": gender,
            "city_tier": city_tier,
            "occupation": occupation,
            "tenure_months": tenure_months,
            "account_type": account_type,
            "avg_monthly_balance": round(avg_monthly_balance, 2),
            "balance_trend_90d": round(balance_trend_90d, 2),
            "min_balance_breaches": min_balance_breaches,
            "num_products": num_products,
            "login_frequency_30d": login_frequency_30d,
            "txn_count_30d": txn_count_30d,
            "txn_volume_change": round(txn_volume_change, 2),
            "last_txn_days_ago": last_txn_days_ago,
            "days_since_last_contact": days_since_last_contact,
            "num_complaints_6m": num_complaints_6m,
            "last_complaint_days_ago": last_complaint_days_ago,
            "salary_credit": salary_credit,
            "credit_score": credit_score,
            "has_credit_card": has_credit_card,
            "has_fd": has_fd,
            "has_loan": has_loan,
            "loan_dpd": loan_dpd,
            "has_insurance": has_insurance,
            "kyc_status": kyc_status,
            "account_status": account_status,
            "is_senior_citizen": is_senior_citizen,
            "nps_score": nps_score,
            "upi_active": upi_active,
        }

        return {
            "customer_id": customer_id,
            "core_profile": core_profile,
            "temporal_history": {
                "balance_history": balance_history,
                "login_history": login_history,
                "txn_history": txn_history,
            },
        }

    def _generate_balance_history(
        self, rng: random.Random, avg_balance: float, trend: float
    ) -> List[Dict[str, Any]]:
        if trend >= 0:
            month1 = avg_balance * (1 - trend * 0.3)
            month2 = avg_balance * (1 - trend * 0.15)
            month3 = avg_balance
        else:
            month1 = avg_balance * (1 - trend * 0.3)
            month2 = avg_balance * (1 - trend * 0.15)
            month3 = avg_balance

        return [
            {"month": "Month_1", "balance": round(month1, 2)},
            {"month": "Month_2", "balance": round(month2, 2)},
            {"month": "Month_3", "balance": round(month3, 2)},
        ]

    def _generate_login_history(
        self, rng: random.Random, current_logins: int
    ) -> List[Dict[str, Any]]:
        if current_logins == 0:
            return [
                {"month": "Month_1", "logins": 0},
                {"month": "Month_2", "logins": 0},
                {"month": "Month_3", "logins": 0},
            ]

        drop_factor = rng.uniform(0.3, 0.8)
        month3 = current_logins
        month2 = int(month3 / drop_factor) if drop_factor > 0 else month3
        month1 = int(month2 / drop_factor) if drop_factor > 0 else month2

        return [
            {"month": "Month_1", "logins": month1},
            {"month": "Month_2", "logins": month2},
            {"month": "Month_3", "logins": month3},
        ]

    def _generate_txn_history(
        self,
        rng: random.Random,
        txn_count: int,
        avg_balance: float,
        volume_change: float,
    ) -> List[Dict[str, Any]]:
        if txn_count == 0:
            return [
                {"month": "Month_1", "txn_count": 0, "txn_volume": 0.0},
                {"month": "Month_2", "txn_count": 0, "txn_volume": 0.0},
                {"month": "Month_3", "txn_count": 0, "txn_volume": 0.0},
            ]

        if volume_change >= 0:
            month3_volume = avg_balance * 2 * (1 + volume_change * 0.3)
            month2_volume = month3_volume / (1 + volume_change * 0.15)
            month1_volume = month3_volume / (1 + volume_change * 0.3)
        else:
            month3_volume = avg_balance * 2 * (1 + volume_change * 0.3)
            month2_volume = month3_volume / (1 + volume_change * 0.15)
            month1_volume = month3_volume / (1 + volume_change * 0.3)

        avg_txn_volume = month3_volume / max(txn_count, 1)

        month1_count = (
            int(txn_count / 0.5) if volume_change < 0 else int(txn_count * 1.5)
        )
        month2_count = (
            int(txn_count / 0.7) if volume_change < 0 else int(txn_count * 1.2)
        )
        month3_count = txn_count

        return [
            {
                "month": "Month_1",
                "txn_count": month1_count,
                "txn_volume": round(month1_volume, 2),
            },
            {
                "month": "Month_2",
                "txn_count": month2_count,
                "txn_volume": round(month2_volume, 2),
            },
            {
                "month": "Month_3",
                "txn_count": month3_count,
                "txn_volume": round(month3_volume, 2),
            },
        ]


def compute_true_risk(customer: Dict[str, Any]) -> str:
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

    if score >= 10:
        return "high_risk"
    elif score >= 6:
        return "medium_risk"
    elif score >= 2:
        return "low_risk"
    else:
        return "not_at_risk"


def generate_campaign_collisions(
    customers_df: Any, output_dir: str = ".", seed: int = 42
) -> Dict[str, str]:
    """
    Generate overlapping campaign files for Task 2.

    The function accepts either:
    - a list of customer dictionaries containing "customer_id", or
    - a dataframe-like object with a "customer_id" column.
    """
    customer_ids: List[str] = []

    if hasattr(customers_df, "columns") and "customer_id" in getattr(
        customers_df, "columns", []
    ):
        customer_ids = [str(cid) for cid in customers_df["customer_id"].tolist()]
    elif isinstance(customers_df, list):
        customer_ids = [str(c.get("customer_id")) for c in customers_df if c]

    customer_ids = [cid for cid in customer_ids if cid]
    if not customer_ids:
        raise ValueError("No customer_id values available to generate campaign files")

    rng = random.Random(seed + 1001)
    os.makedirs(output_dir, exist_ok=True)

    count = len(customer_ids)
    auto_count = max(3, int(count * 0.7))
    card_count = max(3, int(count * 0.7))
    retention_count = max(3, int(count * 0.6))

    auto_ids = customer_ids[:auto_count]
    card_start = max(0, int(count * 0.2))
    card_ids = customer_ids[card_start : min(count, card_start + card_count)]
    retention_start = max(0, int(count * 0.4))
    retention_ids = customer_ids[retention_start : min(count, retention_start + retention_count)]

    # Ensure overlaps always exist across all three lists.
    if count >= 3:
        shared_ids = customer_ids[: min(3, count)]
        for cid in shared_ids:
            if cid not in auto_ids:
                auto_ids.append(cid)
            if cid not in card_ids:
                card_ids.append(cid)
            if cid not in retention_ids:
                retention_ids.append(cid)

    def _rows(ids: List[str], source: str, low: int, high: int) -> List[Dict[str, Any]]:
        return [
            {
                "customer_id": cid,
                "offer_value": round(rng.uniform(low, high), 2),
                "campaign_source": source,
            }
            for cid in ids
        ]

    auto_rows = _rows(auto_ids, "auto_loan", 1500, 8000)
    card_rows = _rows(card_ids, "credit_card", 2000, 10000)
    retention_rows = _rows(retention_ids, "retention", 2500, 12000)

    def _write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["customer_id", "offer_value", "campaign_source"]
            )
            writer.writeheader()
            writer.writerows(rows)

    auto_path = os.path.join(output_dir, "auto_loan.csv")
    card_path = os.path.join(output_dir, "credit_card.csv")
    retention_path = os.path.join(output_dir, "retention.csv")
    rules_path = os.path.join(output_dir, "priority_rules.txt")

    _write_csv(auto_path, auto_rows)
    _write_csv(card_path, card_rows)
    _write_csv(retention_path, retention_rows)

    with open(rules_path, "w", encoding="utf-8") as f:
        f.write("1. No duplicate customers allowed.\n")
        f.write("2. Retention priority > Credit Card > Auto Loan.\n")
        f.write("3. If same priority, higher offer_value wins.\n")

    return {
        "auto_loan": auto_path,
        "credit_card": card_path,
        "retention": retention_path,
        "priority_rules": rules_path,
    }


def load_campaign_rows(campaign_files: Dict[str, str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in ["auto_loan", "credit_card", "retention"]:
        path = campaign_files.get(key)
        if not path or not os.path.exists(path):
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
                rows.append(
                    {
                        "customer_id": customer_id,
                        "offer_value": offer_value,
                        "campaign_source": str(row.get("campaign_source", "")).strip(),
                    }
                )
    return rows
