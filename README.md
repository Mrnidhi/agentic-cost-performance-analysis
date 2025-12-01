# AI Agent Success Rate Prediction

**Course:** DATA 230 - Data Visualization at SJSU

A simple ML project that predicts AI agent `success_rate` and integrates with Tableau via TabPy.

---

## 📁 Project Structure

```
project/
├── cleaned_data.csv              # Main dataset (also in data/cleaned/)
├── requirements.txt              # Python dependencies
├── tabpy_config.conf             # TabPy configuration (optional)
├── README.md                     # This file
│
├── ml/                           # ML scripts and models
│   ├── train_model.py            # Train the ML model
│   ├── deploy_to_tabpy.py        # Deploy to TabPy for Tableau
│   ├── success_rate_model.pkl    # Saved model (after training)
│   └── model_info.pkl            # Model metadata
│
└── data/
    ├── raw/                      # Raw dataset
    │   └── agentic_ai.csv
    ├── cleaned/                  # Cleaned dataset
    │   └── cleaned_data.csv
    └── analytics/                # Feature engineering outputs
        └── (processed data goes here)
```

---

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the Model

```bash
python ml/train_model.py
```

This will:
- Load `cleaned_data.csv`
- Train a RandomForestRegressor to predict `success_rate`
- Save the model to `success_rate_model.pkl`
- Print model performance metrics

### Step 3: Start TabPy Server

Open a **new terminal** and run:

```bash
tabpy --disable-auth-warning
```

Or with config file:

```bash
tabpy --disable-auth-warning --config tabpy_config.conf
```

Wait until you see: `Web service listening on port 9004`

### Step 4: Deploy Model to TabPy

In your original terminal:

```bash
python ml/deploy_to_tabpy.py
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
RMSE:  0.05-0.10   (lower is better)
MAE:   0.03-0.08   (lower is better)
R²:    0.85-0.95   (higher is better, max 1.0)
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
- Make sure TabPy is running (`tabpy` in a separate terminal)
- Check that port 9004 is not blocked

### "Model file not found"
- Run `python ml/train_model.py` first

### "Column not found"
- Verify your data has all required columns
- Check column names match exactly (case-sensitive)

---

## 📝 Essential Files

For the ML + TabPy + Tableau workflow, you only need:

- `ml/train_model.py` - Trains the model
- `ml/deploy_to_tabpy.py` - Deploys to TabPy
- `cleaned_data.csv` - Your dataset (can be in root or `data/cleaned/`)
- `requirements.txt` - Python dependencies
- `README.md` - This file

**Optional files:**
- `tabpy_config.conf` - TabPy configuration
- `ml/model_info.pkl` - Model metadata (auto-generated)

**Files you can ignore:**
- `src/` - Complex architecture (not needed)
- `tests/` - Unit tests (not needed for simple project)
- `notebooks/` - Jupyter notebooks (optional)
- `dashboard/` - HTML dashboard (optional)
- `examples/` - Example code (optional)

---

Built with ❤️ for DATA 230 at SJSU

