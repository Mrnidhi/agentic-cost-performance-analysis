# Command Reference Guide

All the commands you need to run, test, and deploy this project.

---

## Setup

### Create Virtual Environment
```bash
# Create a new virtual environment
python -m venv myenv

# Activate it (Mac/Linux)
source myenv/bin/activate

# Activate it (Windows)
myenv\Scripts\activate
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Upgrade pip (if needed)
```bash
pip install --upgrade pip
```

---

## Running Tests

### Run All Tests
```bash
python -m pytest tests/ -v
```

### Run Tests with Coverage Report
```bash
python -m pytest tests/ -v --cov=src --cov-report=html
# Then open htmlcov/index.html in your browser
```

### Run Specific Test File
```bash
# Test API endpoints
python -m pytest tests/test_api.py -v

# Test feature engineering
python -m pytest tests/test_features.py -v

# Test ML models
python -m pytest tests/test_models.py -v

# Test business logic
python -m pytest tests/test_business_logic.py -v
```

### Run Tests Matching a Pattern
```bash
# Run only tests with "optimization" in the name
python -m pytest tests/ -k "optimization" -v

# Run only tests with "risk" in the name
python -m pytest tests/ -k "risk" -v
```

### Run Tests in Parallel (Faster)
```bash
python -m pytest tests/ -n auto
```

### Stop on First Failure
```bash
python -m pytest tests/ -x
```

---

## Running the API

### Start API Server (Development Mode)
```bash
python -m uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### Start API Server (Production Mode)
```bash
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Using the run_api.py Script
```bash
python run_api.py --reload --port 8000
```

---

## Testing API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Get API Info
```bash
curl http://localhost:8000/
```

### Optimize Agent Configuration
```bash
curl -X POST http://localhost:8000/v1/optimize-agent-configuration \
  -H "Content-Type: application/json" \
  -d '{
    "task_category": "Data Analysis",
    "required_accuracy": 0.85,
    "budget_constraint": 5.0,
    "latency_requirement": 500,
    "privacy_requirements": "medium",
    "business_criticality": "high"
  }'
```

### Performance Benchmarking
```bash
curl -X POST http://localhost:8000/v1/performance-benchmarking \
  -H "Content-Type: application/json" \
  -d '{
    "agent_type": "Data Analyst",
    "success_rate": 0.87,
    "accuracy_score": 0.82,
    "efficiency_score": 0.79,
    "cost_per_task_cents": 3.2,
    "response_latency_ms": 180.0
  }'
```

### Cost-Performance Tradeoffs
```bash
curl -X POST http://localhost:8000/v1/cost-performance-tradeoffs \
  -H "Content-Type: application/json" \
  -d '{
    "min_performance": 0.75,
    "max_cost": 10.0,
    "max_risk": 0.3,
    "optimization_priority": "balanced"
  }'
```

### Failure Risk Assessment
```bash
curl -X POST http://localhost:8000/v1/failure-risk-assessment \
  -H "Content-Type: application/json" \
  -d '{
    "agent_id": "agent-001",
    "success_rate": 0.75,
    "cpu_usage_percent": 85.0,
    "memory_usage_mb": 450.0,
    "response_latency_ms": 350.0,
    "error_rate": 0.05
  }'
```

### Agent Recommendations
```bash
curl -X POST http://localhost:8000/v1/agent-recommendation-engine \
  -H "Content-Type: application/json" \
  -d '{
    "task_category": "Data Analysis",
    "required_accuracy": 0.85,
    "required_efficiency": 0.75,
    "max_cost_cents": 5.0,
    "top_k": 5
  }'
```

---

## Running Jupyter Notebooks

### Start Jupyter
```bash
jupyter notebook
```

### Run Notebooks in Order
1. `TG01_Data_Preparation.ipynb` - Clean the data
2. `TG03_Feature_Engineering.ipynb` - Create features
3. `notebooks/01_advanced_feature_engineering.ipynb` - Advanced features
4. `notebooks/02_strategic_analysis.ipynb` - Analyze strategies
5. `notebooks/03_model_training_optimization.ipynb` - Train models

---

## Docker Commands

### Build Docker Image
```bash
docker build -t agent-intelligence-api .
```

### Run with Docker
```bash
docker run -p 8000:8000 agent-intelligence-api
```

### Run with Docker Compose
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

---

## Code Quality

### Run Linter (flake8)
```bash
flake8 src/ tests/
```

### Format Code (black)
```bash
black src/ tests/
```

### Sort Imports (isort)
```bash
isort src/ tests/
```

### Type Check (mypy)
```bash
mypy src/
```

---

## Git Commands

### Check Status
```bash
git status
```

### Add All Changes
```bash
git add .
```

### Commit Changes
```bash
git commit -m "Your commit message"
```

### Push to GitHub
```bash
git push origin main
```

### Pull Latest Changes
```bash
git pull origin main
```

---

## Useful URLs (When API is Running)

| URL | Description |
|-----|-------------|
| http://localhost:8000 | API root |
| http://localhost:8000/health | Health check |
| http://localhost:8000/docs | Swagger UI (interactive API docs) |
| http://localhost:8000/redoc | ReDoc (beautiful API docs) |
| http://localhost:8000/openapi.json | OpenAPI schema |

---

## Quick Reference

```bash
# Setup (one time)
python -m venv myenv && source myenv/bin/activate && pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Start API
python -m uvicorn src.api.main:app --reload --port 8000

# Open docs
open http://localhost:8000/docs
```

---

## Troubleshooting

### "Module not found" Error
Make sure your virtual environment is activated:
```bash
source myenv/bin/activate
```

### Tests Fail with Import Error
Use `python -m pytest` instead of just `pytest`:
```bash
python -m pytest tests/ -v
```

### Port Already in Use
Kill the process using the port or use a different port:
```bash
python -m uvicorn src.api.main:app --reload --port 8001
```

### Permission Denied
On Mac/Linux, you might need:
```bash
chmod +x run_api.py
```

