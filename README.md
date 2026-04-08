---
title: META SCALER – AI Banking Retention Sandbox
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 7860
huggingface_space_url: https://vansh-myth-prism.hf.space
tags:
  - openenv
  - reinforcement-learning
  - banking
  - customer-retention
  - hackathon
---

# META SCALER — AI Banking Retention Sandbox

An end-to-end autonomous agent evaluation environment for **banking customer retention**. Built on Meta's [OpenEnv](https://github.com/meta-pytorch/openenv) framework, this sandbox chains three interconnected tasks into a unified pipeline — from churn prediction to campaign resolution to personalized outreach.

## 🎯 What It Does

```
Customer Data ──► Task 1: Risk Triage ──► Task 2: Campaign Collision ──► Task 3: Retention Outreach
                    │ classify churn        │ resolve overlapping           │ make personalized
                    │ risk tiers            │ campaign offers               │ counter-offers
                    ▼                       ▼                              ▼
                bank_ops.db ──────────── shared SQLite ──────────────── results & scores
```

An LLM agent interacts with the environment through a standard HTTP API (`/reset`, `/step`, `/state`) and is graded on its ability to maximize customer retention ROI.

## 📊 The 3 Tasks

| Task | Difficulty | What the agent does | Customers | Budget |
|------|-----------|---------------------|-----------|--------|
| **Task 1: Risk Triage** | Easy | Classify customers into `high_risk`, `medium_risk`, `low_risk`, `not_at_risk` | 10 | — |
| **Task 2: Campaign Collision** | Medium | Resolve overlapping offers from auto-loan, credit-card, and retention campaigns | 10 | — |
| **Task 3: Retention Orchestration** | Hard | Make personalized retention offers (fee waivers, rate discounts, cashback) with a limited budget | 6 | $800 |

Task 3 reads the outputs of Tasks 1 & 2 from the shared SQLite database, creating a true end-to-end pipeline.

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- An LLM API endpoint (OpenAI-compatible)

### Setup
```bash
# Clone the repo
git clone https://github.com/Sh1vam09/META_SCALER.git
cd META_SCALER

# Create virtual environment & install dependencies
python -m venv venv
source venv/bin/activate  # or .\venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure your LLM
cp .env.example .env
# Edit .env with your API_BASE_URL, MODEL_NAME, HF_TOKEN
```

### Run
```bash
# Terminal 1: Start the unified server
uvicorn server.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Run the full 3-task pipeline
python inference.py
```

## 🔌 API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/reset` | POST | Start a task: `{"task_id": 1\|2\|3, "seed": 42}` |
| `/step` | POST | Submit an action (format depends on task) |
| `/state` | GET | Current episode state |

### Task 1 & 2 Action Format
```json
{
  "customer_id": "C_001",
  "risk_tier": "high_risk",
  "reasoning": "High churn probability with declining engagement",
  "confidence": 0.85,
  "top_signals_used": ["lstm_output", "transaction_velocity"]
}
```

### Task 3 Action Format (ACRE)
```json
{
  "offer_type": "RATE_DISCOUNT",
  "discount_percentage": 0.3,
  "cashback_amount": 0
}
```

**Offer Types:** `NO_ACTION`, `FEE_WAIVER`, `RATE_DISCOUNT`, `CASHBACK`, `PREMIUM_UPGRADE`

## 📁 Project Structure

```
META_SCALER/
├── server/
│   └── app.py                          # Unified FastAPI server (all 3 tasks)
├── inference.py                        # LLM agent (runs full pipeline)
├── bank_churn_env/                     # Task 1 & 2 engine
│   ├── models.py                       # ChurnAction, ChurnObservation
│   ├── auto_loan.csv / credit_card.csv / retention.csv  # Campaign data
│   └── server/
│       ├── environment.py              # BankChurnEnvironment
│       ├── data_store.py               # SQLite persistence
│       ├── customer_generator.py       # Synthetic customer generation
│       ├── ml_signals.py               # ML signal simulation
│       ├── grader.py                   # Task 1 & 2 grading
│       └── reward.py                   # Step reward computation
├── retention_strategy/                 # Task 3 engine (ACRE)
│   ├── models.py                       # OutreachAction, CustomerObservation
│   └── server/
│       ├── customer_offer_environment.py  # ACREEnvironment
│       ├── tasks.py                    # Customer profiles & DB integration
│       └── graders.py                  # Task 3 grading (VIP triage)
├── Dockerfile                          # HuggingFace Spaces deployment
├── requirements.txt                    # Python dependencies
└── .env.example                        # Environment variable template
```

## 🐳 Deployment

### HuggingFace Spaces (Docker)

Push this repo to a HuggingFace Space with Docker SDK:

```bash
# Create the space on huggingface.co, then:
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/META_SCALER
git push hf main
```

The `Dockerfile` and `README.md` frontmatter are pre-configured for HuggingFace Spaces.

### Local Docker
```bash
docker build -t meta-scaler .
docker run -p 7860:7860 meta-scaler
```

## 📈 Scoring

Each task is graded 0.0 – 1.0:

| Task | Grading Criteria |
|------|-----------------|
| Task 1 | Accuracy of risk tier classification vs ground truth |
| Task 2 | Correct campaign resolution following priority rules |
| Task 3 | 50% VIP retention + 25% skip negative-value + 15% ROI + 10% budget management |

## 🛠️ Tech Stack

- **Framework:** FastAPI + Uvicorn
- **AI:** OpenAI-compatible LLM API
- **Data:** SQLite (cross-task persistence)
- **Evaluation:** Meta OpenEnv
- **Deployment:** Docker / HuggingFace Spaces

## 📜 License

BSD-style license. See LICENSE file for details.