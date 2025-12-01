"""
train_model.py - Train a simple ML model to predict success_rate

This script:
1. Loads the cleaned AI agent dataset
2. Selects numeric features for prediction
3. Trains a RandomForestRegressor
4. Evaluates performance on a test set
5. Saves the model for TabPy deployment

Usage:
    python train_model.py

Output:
    success_rate_model.pkl - The trained model file
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from pathlib import Path


# ============================================
# CONFIGURATION
# ============================================

# Path to the dataset
DATA_PATH = "data/cleaned/cleaned_data.csv"

# Features to use for prediction
FEATURE_COLS = [
    "task_complexity",
    "autonomy_level", 
    "accuracy_score",
    "efficiency_score",
    "memory_usage_mb",
    "cpu_usage_percent",
    "cost_per_task_cents"
]

# Target variable
TARGET_COL = "success_rate"

# Output model file (saved in ml/ folder)
MODEL_PATH = "success_rate_model.pkl"


# ============================================
# MAIN TRAINING SCRIPT
# ============================================

def main():
    print("=" * 50)
    print("Training Success Rate Prediction Model")
    print("=" * 50)
    
    # -----------------------------------------
    # Step 1: Load the data
    # -----------------------------------------
    print("\n📂 Loading data...")
    
    data_path = None
    for path in DATA_PATHS:
        if Path(path).exists():
            data_path = path
            break
    
    if data_path is None:
        raise FileNotFoundError(
            f"Could not find cleaned_data.csv. Tried:\n" + 
            "\n".join(f"  - {p}" for p in DATA_PATHS)
        )
    
    df = pd.read_csv(data_path)
    print(f"   Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    # -----------------------------------------
    # Step 2: Validate required columns exist
    # -----------------------------------------
    print("\n🔍 Checking required columns...")
    
    missing_features = [col for col in FEATURE_COLS if col not in df.columns]
    if missing_features:
        raise ValueError(f"Missing feature columns: {missing_features}")
    
    if TARGET_COL not in df.columns:
        raise ValueError(f"Missing target column: {TARGET_COL}")
    
    print(f"   ✓ All required columns found")
    
    # -----------------------------------------
    # Step 3: Prepare features and target
    # -----------------------------------------
    print("\n🔧 Preparing features...")
    
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    
    # Handle any missing values (simple approach: fill with median)
    for col in FEATURE_COLS:
        if X[col].isnull().any():
            median_val = X[col].median()
            X[col] = X[col].fillna(median_val)
            print(f"   Filled {X[col].isnull().sum()} missing values in {col} with median")
    
    print(f"   Features shape: {X.shape}")
    print(f"   Target shape: {y.shape}")
    print(f"   Target range: {y.min():.3f} to {y.max():.3f}")
    
    # -----------------------------------------
    # Step 4: Split into train/test sets
    # -----------------------------------------
    print("\n📊 Splitting data...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=0.2, 
        random_state=42
    )
    
    print(f"   Training set: {len(X_train):,} samples")
    print(f"   Test set: {len(X_test):,} samples")
    
    # -----------------------------------------
    # Step 5: Train the model
    # -----------------------------------------
    print("\n🤖 Training RandomForestRegressor...")
    
    model = RandomForestRegressor(
        n_estimators=100,      # Number of trees
        max_depth=10,          # Limit depth to prevent overfitting
        min_samples_split=5,   # Minimum samples to split a node
        random_state=42,       # For reproducibility
        n_jobs=-1              # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    print("   ✓ Model trained successfully")
    
    # -----------------------------------------
    # Step 6: Evaluate the model
    # -----------------------------------------
    print("\n📈 Evaluating model performance...")
    
    # Predictions on test set
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"   RMSE:  {rmse:.4f}")
    print(f"   MAE:   {mae:.4f}")
    print(f"   R²:    {r2:.4f}")
    
    # Feature importance
    print("\n🎯 Feature Importance:")
    importance = pd.DataFrame({
        'feature': FEATURE_COLS,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for _, row in importance.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"   {row['feature']:25s} {row['importance']:.3f} {bar}")
    
    # -----------------------------------------
    # Step 7: Save the model
    # -----------------------------------------
    print(f"\n💾 Saving model to {MODEL_PATH}...")
    
    joblib.dump(model, MODEL_PATH)
    print(f"   ✓ Model saved successfully")
    
    # Also save feature names for reference
    model_info = {
        'feature_cols': FEATURE_COLS,
        'target_col': TARGET_COL,
        'r2_score': r2,
        'rmse': rmse
    }
    joblib.dump(model_info, 'model_info.pkl')
    print(f"   ✓ Model info saved to model_info.pkl")
    
    # -----------------------------------------
    # Summary
    # -----------------------------------------
    print("\n" + "=" * 50)
    print("✅ Training Complete!")
    print("=" * 50)
    print(f"""
Next steps:
1. Start TabPy server:     tabpy
2. Deploy to TabPy:        python deploy_to_tabpy.py
3. Connect Tableau to TabPy at localhost:9004
4. Use SCRIPT_REAL to call 'predict_success_rate'
""")


if __name__ == "__main__":
    main()

