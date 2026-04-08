import json
import os
import sqlite3
from typing import Any, Dict, List


DB_FILE_NAME = "bank_ops.db"


def get_db_path(base_dir: str) -> str:
    return os.path.join(base_dir, DB_FILE_NAME)


def init_db(base_dir: str) -> str:
    db_path = get_db_path(base_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task1_predictions (
                episode_id TEXT,
                customer_id TEXT,
                predicted_tier TEXT,
                true_tier TEXT,
                step_reward REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task2_candidates (
                episode_id TEXT,
                customer_id TEXT,
                offer_value REAL,
                campaign_source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task2_deployment (
                episode_id TEXT,
                customer_id TEXT,
                offer_value REAL,
                campaign_source TEXT,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task3_actions (
                episode_id TEXT,
                customer_id TEXT,
                action_name TEXT,
                channel TEXT,
                reward REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.commit()
    finally:
        conn.close()

    return db_path


def persist_task1_results(base_dir: str, episode_id: str, results: List[Dict[str, Any]]) -> None:
    if not results:
        return

    db_path = init_db(base_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM task1_predictions WHERE episode_id = ?", (episode_id,))
        cur.executemany(
            """
            INSERT INTO task1_predictions (episode_id, customer_id, predicted_tier, true_tier, step_reward)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (
                    episode_id,
                    str(row.get("customer_id", "")),
                    str(row.get("predicted_tier", "")),
                    str(row.get("true_tier", "")),
                    float(row.get("step_reward", 0.0)),
                )
                for row in results
                if row.get("customer_id")
            ],
        )
        conn.commit()
    finally:
        conn.close()


def persist_task2_candidates(base_dir: str, episode_id: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    db_path = init_db(base_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM task2_candidates WHERE episode_id = ?", (episode_id,))
        cur.executemany(
            """
            INSERT INTO task2_candidates (episode_id, customer_id, offer_value, campaign_source)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    episode_id,
                    str(row.get("customer_id", "")),
                    float(row.get("offer_value", 0.0)),
                    str(row.get("campaign_source", "")),
                )
                for row in rows
                if row.get("customer_id")
            ],
        )
        conn.commit()
    finally:
        conn.close()


def persist_task2_deployment(base_dir: str, episode_id: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return

    db_path = init_db(base_dir)
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        cur.execute("DELETE FROM task2_deployment WHERE episode_id = ?", (episode_id,))
        cur.executemany(
            """
            INSERT INTO task2_deployment (episode_id, customer_id, offer_value, campaign_source)
            VALUES (?, ?, ?, ?)
            """,
            [
                (
                    episode_id,
                    str(row.get("customer_id", "")),
                    float(row.get("offer_value", 0.0)),
                    str(row.get("campaign_source", "")),
                )
                for row in rows
                if row.get("customer_id")
            ],
        )
        conn.commit()
    finally:
        conn.close()


def read_task2_output(base_dir: str) -> List[Dict[str, Any]]:
    output_path = os.path.join(base_dir, "master_campaign_deployment.json")
    if not os.path.exists(output_path):
        return []

    try:
        with open(output_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    except Exception:
        return []

    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        if isinstance(payload.get("deployments"), list):
            return payload["deployments"]
        if isinstance(payload.get("records"), list):
            return payload["records"]
    return []
