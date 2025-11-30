# AI Agent Performance Intelligence System

> A college group project for **DATA 230 (Data Visualization)** at **San Jose State University**

We built a system that helps you understand how AI agents perform, what they cost, and which one to pick for your task.

---

## What Does This Project Do?

Imagine you have many AI agents (like chatbots, code assistants, data analysts). Each one has different:
- **Performance** (how accurate and fast it is)
- **Cost** (how much it costs per task)
- **Risk** (how likely it is to fail)

Our system:
1. **Analyzes** your agent data
2. **Predicts** which agent will perform best
3. **Recommends** the right agent for your task
4. **Warns** you about agents that might fail

---

## How It Works (Architecture)

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR DATA                                   │
│                    (Agent performance logs)                         │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 1: DATA PREPARATION                         │
│         Clean data, handle missing values, fix formats              │
│                   (TG01_Data_Preparation.ipynb)                     │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STEP 2: FEATURE ENGINEERING                       │
│      Create smart features like "business value score",             │
│      "risk index", "cost efficiency tier"                           │
│                   (TG03_Feature_Engineering.ipynb)                  │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    STEP 3: TRAIN ML MODELS                          │
│  • XGBoost → Predicts performance                                   │
│  • Isolation Forest → Detects failures                              │
│  • Cosine Similarity → Recommends agents                            │
│                   (notebooks/03_model_training.ipynb)               │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      STEP 4: REST API                               │
│         FastAPI endpoints that serve predictions                    │
│                      (src/api/main.py)                              │
└─────────────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   STEP 5: VISUALIZATION                             │
│           Connect to Tableau / Power BI / Dashboard                 │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
agentic-cost-performance-analysis/
│
├── data/
│   ├── raw/                  # Original dataset
│   ├── cleaned/              # Cleaned data
│   ├── processed/            # Another copy of cleaned data
│   └── ml/                   # Feature-engineered data for ML
│
├── notebooks/                # Advanced analysis notebooks
│   ├── 01_advanced_feature_engineering.ipynb
│   ├── 02_strategic_analysis.ipynb
│   └── 03_model_training_optimization.ipynb
│
├── src/                      # Production Python code
│   ├── core/                 # Types, configs, exceptions
│   ├── features/             # Feature engineering logic
│   ├── models/               # ML model classes
│   └── api/                  # FastAPI application
│
├── models/                   # Saved trained models
├── tests/                    # Unit tests (141 tests!)
├── examples/                 # Dashboard integration examples
│
├── TG01_Data_Preparation.ipynb
├── TG03_Feature_Engineering.ipynb
│
├── requirements.txt          # Python dependencies
├── Dockerfile                # Container deployment
├── docker-compose.yml        # Local development setup
└── README.md                 # You are here!
```

---

## API Endpoints

Our API has 5 main endpoints:

| Endpoint | What It Does |
|----------|--------------|
| `POST /v1/optimize-agent-configuration` | Recommends the best agent setup for your task |
| `POST /v1/performance-benchmarking` | Compares your agent against others |
| `POST /v1/cost-performance-tradeoffs` | Shows cost vs performance options |
| `POST /v1/failure-risk-assessment` | Predicts if an agent might fail |
| `POST /v1/agent-recommendation-engine` | Suggests which agent to use |

---

## ML Models We Use

| Model | Purpose |
|-------|---------|
| **XGBoost** | Predicts business value and performance scores |
| **Isolation Forest** | Detects unusual behavior (potential failures) |
| **Cosine Similarity** | Matches tasks to the best agent |

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/Mrnidhi/agentic-cost-performance-analysis.git
cd agentic-cost-performance-analysis

# 2. Create virtual environment
python -m venv myenv
source myenv/bin/activate  # On Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run tests (141 tests should pass)
python -m pytest tests/ -v

# 5. Start the API
python -m uvicorn src.api.main:app --reload --port 8000

# 6. Open browser
# Go to: http://localhost:8000/docs
```

See `COMMANDS.md` for all available commands.

---

## Team

This project was built by students in DATA 230 at San Jose State University.

---

## License

This project is for educational purposes.
