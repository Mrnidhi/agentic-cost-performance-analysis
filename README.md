# AI Agent Success Rate Prediction

**Course:** DATA 230 - Data Visualization at SJSU

A simple ML project that predicts AI agent `success_rate` and integrates with Tableau via TabPy.

---

## 🎯 Project Purpose

This project demonstrates a complete ML → TabPy → Tableau workflow:

1. **Train** a regression model to predict `success_rate` from agent features
2. **Deploy** the model to TabPy for Tableau integration
3. **Use** predictions in Tableau for row-level analysis and what-if scenarios

---

## 📁 Project Structure

```
agentic-cost-performance-analysis/
├── cleaned_data.csv              # Main dataset
├── train_model.py                # Train ML model
├── deploy_to_tabpy.py            # Deploy to TabPy
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── PROJECT_STRUCTURE.md          # Structure guide
├── tabpy_config.conf             # TabPy configuration
│
├── success_rate_model.pkl        # Generated: Trained model
├── model_info.pkl                # Generated: Model metadata (optional)
│
├── data/
│   ├── raw/                      # Raw dataset
│   ├── cleaned/                  # Cleaned dataset
│   └── analytics/                # Feature engineering outputs
│
└── archive/                       # Legacy files (moved here)
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python train_model.py
```

This will:
- Load `cleaned_data.csv`
- Train a RandomForestRegressor to predict `success_rate`
- Save the model to `success_rate_model.pkl`
- Print R² score and MAE metrics

### Step 3: Start TabPy Server

Open a **new terminal** and run:

```bash
tabpy --disable-auth-warning
```

Wait until you see: `Web service listening on port 9004`

### Step 4: Deploy Model to TabPy

In your original terminal:

```bash
python deploy_to_tabpy.py
```

This deploys the `predict_success_rate` function to TabPy.

### Step 5: Connect Tableau

1. Open Tableau Desktop
2. Connect to your data source (`cleaned_data.csv`)
3. Go to: **Help → Settings and Performance → Manage Analytics Extension Connection**
4. Select **TabPy**, Server: `localhost`, Port: `9004`
5. Click **Test Connection** → **OK**

### Step 6: Create Prediction Calculated Field

Create a new calculated field in Tableau:

```
SCRIPT_REAL("
return tabpy.query('predict_success_rate', 
    _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7)['response']
",
SUM([task_complexity]),
SUM([autonomy_level]),
SUM([accuracy_score]),
SUM([efficiency_score]),
SUM([memory_usage_mb]),
SUM([cpu_usage_percent]),
SUM([cost_per_task_cents])
)
```

Name it **"Predicted Success Rate"** and use it in your visualizations!

---

## 📊 Features Used for Prediction

| Feature | Description |
|---------|-------------|
| `task_complexity` | Complexity level (1-10) |
| `autonomy_level` | Agent autonomy level (1-10) |
| `accuracy_score` | Accuracy score (0-1) |
| `efficiency_score` | Efficiency score (0-1) |
| `memory_usage_mb` | Memory usage in MB |
| `cpu_usage_percent` | CPU usage (0-100%) |
| `cost_per_task_cents` | Cost per task in cents |

**Target:** `success_rate` (0-1)

---

## 🎯 Model Performance

After training, you'll see metrics like:

```
R² Score: 0.85-0.95   (higher is better, max 1.0)
MAE:      0.03-0.08   (lower is better)
```

---

## 💡 Tableau Visualization Ideas

1. **Actual vs Predicted**: Scatter plot comparing real `success_rate` to predicted
2. **Prediction Error**: Bar chart showing prediction accuracy by `agent_type`
3. **What-If Analysis**: Use Tableau parameters to simulate different configurations
4. **Feature Impact**: Show which features most affect predictions

---

## 🔧 Troubleshooting

### "TabPy connection failed"
- Make sure TabPy is running (`tabpy --disable-auth-warning` in a separate terminal)
- Check that port 9004 is not blocked

### "Model file not found"
- Run `python train_model.py` first

### "Column not found"
- Verify your data has all required columns
- Check column names match exactly (case-sensitive)

---

## 📋 Simple Architecture Overview

```
┌─────────────────┐
│ cleaned_data.csv│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ train_model.py   │──► success_rate_model.pkl
└─────────────────┘
         │
         ▼
┌─────────────────┐
│ deploy_to_tabpy │──► TabPy Server (localhost:9004)
└─────────────────┘
         │
         ▼
┌─────────────────┐
│   Tableau       │──► SCRIPT_REAL calls predict_success_rate
└─────────────────┘
```

**No complex frameworks, no over-engineering — just simple, working code.**

---

Built with ❤️ for DATA 230 at SJSU
