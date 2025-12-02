# AI Agent Cost & Performance Analysis

**Course:** DATA 230 - Data Visualization at SJSU

A comprehensive data analysis project that prepares AI agent performance data for machine learning and Tableau visualization. This project follows a complete data science workflow from raw data cleaning through exploratory analysis to feature engineering.

---

## 🎯 Project Purpose

This project demonstrates a complete data preparation workflow for AI agent performance analysis:

1. **Data Preparation** - Clean and validate raw AI agent performance data
2. **Exploratory Data Analysis** - Understand relationships, correlations, and patterns
3. **Feature Engineering** - Create composite metrics and interaction features for ML models
4. **Future: ML Integration** - Train models to predict `success_rate` and deploy via TabPy for Tableau

---

## 📁 Project Structure

```
agentic-cost-performance-analysis/
├── TG01_Data_Preparation.ipynb      # Step 1: Data cleaning & validation
├── TG02_EDA_Analysis.ipynb          # Step 2: Exploratory data analysis
├── TG03_Feature_Engineering.ipynb   # Step 3: Feature engineering
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
│
├── data/
│   ├── raw/
│   │   └── agentic_ai.csv           # Original raw dataset (5000 rows, 26 columns)
│   ├── cleaned/
│   │   └── cleaned_data.csv        # Cleaned dataset (5000 rows, 26 columns)
│   └── analytics/
│       └── feature_engineered_data.csv  # Feature-engineered dataset (5000 rows, 37 columns)
│
└── ml/                              # Reserved for ML scripts (future)
```

---

## 📊 Dataset Overview

### Original Dataset (`data/raw/agentic_ai.csv`)
- **Size:** 5,000 AI agent performance records
- **Columns:** 26 features including:
  - Agent metadata: `agent_id`, `agent_type`, `model_architecture`, `deployment_environment`
  - Performance metrics: `success_rate`, `accuracy_score`, `efficiency_score`
  - Resource usage: `memory_usage_mb`, `cpu_usage_percent`, `execution_time_seconds`
  - Cost metrics: `cost_per_task_cents`, `cost_efficiency_ratio`
  - Quality scores: `data_quality_score`, `privacy_compliance_score`, `bias_detection_score`

### Cleaned Dataset (`data/cleaned/cleaned_data.csv`)
- **Validated:** No missing values, no duplicates
- **Ready for analysis:** All 26 original columns preserved

### Feature-Engineered Dataset (`data/analytics/feature_engineered_data.csv`)
- **New Features:** 11 engineered features added:
  1. `overall_performance_score` - Average of success, accuracy, and efficiency
  2. `resource_efficiency_score` - Efficiency relative to memory and CPU usage
  3. `cost_effectiveness` - Performance per cent spent
  4. `weighted_quality_score` - Weighted combination of success, accuracy, and data quality
  5. `complexity_autonomy_ratio` - Autonomy relative to task complexity
  6. `success_autonomy_interaction` - Multiplicative effect of success and autonomy
  7. `latency_per_operation` - Latency burden relative to execution time
  8. `arch_performance_benchmark` - Performance vs median for architecture
  9. `env_avg_cost_per_complexity` - Average cost per complexity unit by environment
  10. `hour_of_day` - Extracted hour from timestamp
  11. `is_weekend` - Binary flag for weekend operations

---

## 🚀 Workflow

### Step 1: Data Preparation (`TG01_Data_Preparation.ipynb`)

**Purpose:** Clean and validate the raw dataset

**Process:**
- Load raw data from `data/raw/agentic_ai.csv`
- Inspect data types and structure
- Check for missing values and duplicates
- Validate data quality
- Save cleaned dataset to `data/cleaned/cleaned_data.csv`

**Output:**
- Clean dataset with 5,000 rows, 26 columns
- No missing values or duplicates
- Ready for analysis

### Step 2: Exploratory Data Analysis (`TG02_EDA_Analysis.ipynb`)

**Purpose:** Understand relationships and patterns in the data

**Analysis Includes:**
- Correlation matrix of key numerical variables
- Descriptive statistics for performance and cost metrics
- Visualization of relationships between metrics
- Key insights:
  - `performance_index` strongly correlated with `accuracy_score` (r ≈ 0.97) and `efficiency_score` (r ≈ 0.96)
  - `cost_per_task_cents` rises linearly with `execution_time_seconds` (r ≈ 0.99)
  - High resource usage correlates negatively with performance (r < -0.8)
  - Faster, lighter models achieve better performance per cost

### Step 3: Feature Engineering (`TG03_Feature_Engineering.ipynb`)

**Purpose:** Create composite metrics and interaction features for ML models

**Feature Categories:**
- **Composite Performance Metrics:** Overall performance, resource efficiency, cost effectiveness
- **Interaction Features:** Complexity-autonomy ratios, success-autonomy interactions
- **Categorical Groupings:** Architecture benchmarks, environment cost patterns
- **Temporal Features:** Hour of day, weekend flags

**Output:**
- Feature-engineered dataset with 37 columns (26 original + 11 new)
- Saved to `data/analytics/feature_engineered_data.csv`

---

## 🔧 Setup & Installation

### Prerequisites
- Python 3.8+
- Jupyter Notebook or JupyterLab

### Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning (for future ML steps)
- `joblib` - Model serialization (for future ML steps)
- `tabpy` - TabPy server (for future Tableau integration)
- `tabpy-client` - TabPy client library (for future deployment)

### Run the Notebooks

1. **Data Preparation:**
   ```bash
   jupyter notebook TG01_Data_Preparation.ipynb
   ```

2. **EDA Analysis:**
   ```bash
   jupyter notebook TG02_EDA_Analysis.ipynb
   ```

3. **Feature Engineering:**
   ```bash
   jupyter notebook TG03_Feature_Engineering.ipynb
   ```

---

## 📈 Key Insights from Analysis

### Performance Drivers
- **Accuracy and efficiency** jointly drive overall performance
- **Resource optimization** (lower memory/CPU) significantly improves performance
- **Task complexity** and **autonomy level** increase cost and resource use but reduce performance

### Cost Behavior
- **Execution time** directly correlates with cost (r ≈ 0.99)
- **Cost efficiency** drops as runtime grows (r ≈ -0.78)
- **Speed optimizations** yield direct cost benefits

### Optimization Opportunities
- Faster, lighter models achieve **better performance per cost**
- Reducing runtime and resource usage offers the **strongest path to efficiency gains**
- Complex tasks are harder to optimize but offer higher potential impact

---

## 📋 Data Dictionary

### Core Performance Metrics
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `success_rate` | float | Task success rate | 0.0 - 1.0 |
| `accuracy_score` | float | Prediction accuracy | 0.0 - 1.0 |
| `efficiency_score` | float | Operational efficiency | 0.0 - 1.0 |
| `performance_index` | float | Composite performance score | 0.0 - 1.0 |

### Resource Usage
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `memory_usage_mb` | float | Memory consumption in MB | ~100 - 600 |
| `cpu_usage_percent` | float | CPU utilization percentage | 0 - 100 |
| `execution_time_seconds` | float | Task execution time | 1 - 157 seconds |
| `response_latency_ms` | float | Response latency in milliseconds | ~100 - 5500 ms |

### Cost Metrics
| Feature | Type | Description | Range |
|---------|------|-------------|-------|
| `cost_per_task_cents` | float | Cost per task in cents | 0.003 - 0.059 |
| `cost_efficiency_ratio` | float | Performance per cost unit | 5.95 - 219.33 |

### Agent Characteristics
| Feature | Type | Description | Values |
|---------|------|-------------|--------|
| `agent_type` | string | Type of AI agent | 16 types (e.g., "Project Manager", "Code Assistant") |
| `model_architecture` | string | ML model architecture | 10 types (e.g., "GPT-4o", "Claude-3.5", "LLaMA-3") |
| `deployment_environment` | string | Deployment location | 6 types (Cloud, Server, Edge, Hybrid, Mobile, Desktop) |
| `task_category` | string | Task type | 10 categories (e.g., "Text Processing", "Decision Making") |
| `task_complexity` | int | Complexity level | 1 - 10 |
| `autonomy_level` | int | Agent autonomy level | 1 - 10 |

---

## 📚 Course Context

This project is part of **DATA 230 - Data Visualization** at San José State University, focusing on:
- Data preparation and cleaning workflows
- Exploratory data analysis techniques
- Feature engineering for machine learning
- Integration of ML models with visualization tools (Tableau via TabPy)

---

## 🤝 Contributing

This is a course project. For questions or suggestions, please contact the course instructor.

---

**Built with ❤️ for DATA 230 at SJSU**
