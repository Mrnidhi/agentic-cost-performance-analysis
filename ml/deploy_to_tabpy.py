"""
deploy_to_tabpy.py - Deploy the trained model to TabPy for Tableau integration

This script:
1. Loads the trained success_rate model
2. Connects to a running TabPy server
3. Deploys a prediction function that Tableau can call

Prerequisites:
1. Run train_model.py first to create success_rate_model.pkl
2. Start TabPy server: tabpy
   (Default runs on http://localhost:9004)

Usage:
    python deploy_to_tabpy.py

After deployment, Tableau can call the function using SCRIPT_REAL.
"""

import joblib
import numpy as np
from tabpy.tabpy_tools.client import Client


# ============================================
# CONFIGURATION
# ============================================

# TabPy server URL (default) - note: no trailing slash
TABPY_URL = "http://localhost:9004/"

# Model file path (in ml/ folder)
MODEL_PATH = "success_rate_model.pkl"

# Name of the function in TabPy (Tableau will call this name)
FUNCTION_NAME = "predict_success_rate"


# ============================================
# LOAD THE MODEL
# ============================================

print("📂 Loading trained model...")
try:
    model = joblib.load(MODEL_PATH)
    print(f"   ✓ Model loaded from {MODEL_PATH}")
except FileNotFoundError:
    print(f"   ❌ Error: Could not find {MODEL_PATH}")
    print("   Please run 'python train_model.py' first to train the model.")
    exit(1)


# ============================================
# DEFINE THE PREDICTION FUNCTION
# ============================================

def predict_success_rate(task_complexity,
                         autonomy_level,
                         accuracy_score,
                         efficiency_score,
                         memory_usage_mb,
                         cpu_usage_percent,
                         cost_per_task_cents):
    """
    Predict success_rate for AI agents based on their configuration.
    
    This function is called by Tableau via TabPy.
    Tableau passes each parameter as a list (one value per row in the viz).
    
    Parameters:
    -----------
    task_complexity : list of float
        Complexity level of the task (1-10)
    autonomy_level : list of float
        Agent's autonomy level (1-10)
    accuracy_score : list of float
        Agent's accuracy score (0-1)
    efficiency_score : list of float
        Agent's efficiency score (0-1)
    memory_usage_mb : list of float
        Memory usage in MB
    cpu_usage_percent : list of float
        CPU usage percentage (0-100)
    cost_per_task_cents : list of float
        Cost per task in cents
    
    Returns:
    --------
    list of float
        Predicted success_rate for each row (0-1)
    """
    
    # Stack all inputs into a 2D array (n_samples, n_features)
    # Each input is a list from Tableau, one value per row
    X = np.column_stack([
        task_complexity,
        autonomy_level,
        accuracy_score,
        efficiency_score,
        memory_usage_mb,
        cpu_usage_percent,
        cost_per_task_cents
    ])
    
    # Make predictions using the loaded model
    predictions = model.predict(X)
    
    # Clip predictions to valid range [0, 1]
    predictions = np.clip(predictions, 0.0, 1.0)
    
    # Return as a list (required by TabPy)
    return predictions.tolist()


# ============================================
# DEPLOY TO TABPY
# ============================================

def main():
    print("\n" + "=" * 50)
    print("Deploying Model to TabPy")
    print("=" * 50)
    
    # Connect to TabPy server
    print(f"\n🔌 Connecting to TabPy at {TABPY_URL}...")
    try:
        client = Client(TABPY_URL)
        print("   ✓ Connected to TabPy server")
    except Exception as e:
        print(f"   ❌ Error connecting to TabPy: {e}")
        print("\n   Make sure TabPy is running:")
        print("   1. Open a terminal")
        print("   2. Run: tabpy")
        print("   3. Wait for 'Web service listening on port 9004'")
        print("   4. Run this script again")
        exit(1)
    
    # Deploy the function
    print(f"\n📤 Deploying function '{FUNCTION_NAME}'...")
    try:
        client.deploy(
            FUNCTION_NAME,
            predict_success_rate,
            'Predicts success_rate for AI agents based on configuration parameters. '
            'Input: task_complexity, autonomy_level, accuracy_score, efficiency_score, '
            'memory_usage_mb, cpu_usage_percent, cost_per_task_cents. '
            'Output: predicted success_rate (0-1).',
            override=True  # Replace if function already exists
        )
        print(f"   ✓ Function '{FUNCTION_NAME}' deployed successfully!")
    except Exception as e:
        print(f"   ❌ Error deploying function: {e}")
        exit(1)
    
    # Verify deployment with a test prediction
    print("\n🧪 Testing deployment with sample data...")
    try:
        test_result = predict_success_rate(
            [5.0, 7.0],      # task_complexity
            [6.0, 8.0],      # autonomy_level
            [0.85, 0.90],    # accuracy_score
            [0.80, 0.85],    # efficiency_score
            [256.0, 512.0],  # memory_usage_mb
            [45.0, 60.0],    # cpu_usage_percent
            [0.02, 0.03]     # cost_per_task_cents
        )
        print(f"   ✓ Test predictions: {[f'{p:.4f}' for p in test_result]}")
    except Exception as e:
        print(f"   ⚠️ Test failed: {e}")
    
    # Print success message with Tableau instructions
    print("\n" + "=" * 50)
    print("✅ Deployment Complete!")
    print("=" * 50)
    print(f"""
TabPy is ready! The function '{FUNCTION_NAME}' is now available.

To use in Tableau:
-----------------
1. Connect Tableau to your data (cleaned_data.csv)

2. Go to: Help → Settings and Performance → Manage Analytics Extension Connection
   - Select: TabPy
   - Server: localhost
   - Port: 9004
   - Click "Test Connection" then "OK"

3. Create a Calculated Field with this formula:

SCRIPT_REAL("
import numpy as np
X = np.column_stack([_arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7])
return tabpy.query('predict_success_rate', 
    _arg1, _arg2, _arg3, _arg4, _arg5, _arg6, _arg7)['response']
",
ATTR([task_complexity]),
ATTR([autonomy_level]),
ATTR([accuracy_score]),
ATTR([efficiency_score]),
ATTR([memory_usage_mb]),
ATTR([cpu_usage_percent]),
ATTR([cost_per_task_cents])
)

   OR use the simpler version:

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

4. Name the calculated field "Predicted Success Rate"

5. Use it in your visualizations!
""")


if __name__ == "__main__":
    main()

