# AI Agent Performance Intelligence System

A project to analyze cost and performance of agentic AI systems, train ML models, and deploy a recommendation API.

This is a college group project for DATA 230 (Data Visualization) at SJSU.

## Project Structure

```
project/
├── data/
│   ├── raw/              # Original datasets
│   ├── processed/        # Cleaned data
│   └── ml/               # Feature engineered data
├── notebooks/
│   ├── 01_advanced_feature_engineering.ipynb
│   ├── 02_strategic_analysis.ipynb
│   └── 03_model_training_optimization.ipynb
├── src/
│   ├── features/         # Feature engineering modules
│   ├── models/           # ML model classes
│   └── api/              # FastAPI schemas
├── models/               # Saved trained models
├── tests/                # Unit tests
├── TG01_Data_Preparation.ipynb
├── TG02_EDA_Analysis.ipynb
├── TG03_Feature_Engineering.ipynb
├── requirements.txt
├── Dockerfile
└── README.md
```

## How to Use

1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order (TG01 → TG02 → TG03 → notebooks/)
4. Train models and deploy API

## Requirements

- Python 3.10+
- See `requirements.txt` for packages

## Goals

- Clean and prepare agent performance data
- Engineer features for ML models
- Train ensemble models to predict performance
- Deploy API for LLM recommendations
- Connect to Tableau/Power BI dashboards

## Contributing

- Open an issue or PR with clear changes
- Keep explanations short and specific
