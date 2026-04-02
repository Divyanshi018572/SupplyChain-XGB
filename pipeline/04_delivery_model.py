import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

from sklearn.model_selection import RandomizedSearchCV

def train_delivery_models():
    print("--- Starting Optimized Delivery Risk Model Training ---")
    
    # Load processed data
    processed_dir = path_utils.PROCESSED_DATA_DIR
    X_train = np.load(os.path.join(processed_dir, 'X_train_del.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test_del.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train_del.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test_del.npy'))

    # Initial Model Performance Check
    print("\nTraining Initial Models...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rf_acc = rf.score(X_test, y_test)
    print(f"Baseline Random Forest Accuracy: {rf_acc:.4f}")

    # XGBoost Hyperparameter Tuning
    print("\nOptimizing XGBoost with RandomizedSearchCV...")
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    
    param_grid = {
        'n_estimators': [100, 300, 500],
        'max_depth': [4, 6, 8, 10],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5]
    }

    random_search = RandomizedSearchCV(
        xgb, param_distributions=param_grid, n_iter=10, 
        scoring='accuracy', cv=3, verbose=1, random_state=42, n_jobs=-1
    )
    
    random_search.fit(X_train, y_train)
    best_xgb = random_search.best_estimator_
    xgb_acc = best_xgb.score(X_test, y_test)
    print(f"Optimized XGBoost Accuracy: {xgb_acc:.4f}")
    print(f"Best Params: {random_search.best_params_}")

    # Baseline XGBoost (no tuning)
    print("\nTraining Baseline XGBoost...")
    xgb_base = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_base.fit(X_train, y_train)
    xgb_base_acc = xgb_base.score(X_test, y_test)
    print(f"Baseline XGBoost Accuracy: {xgb_base_acc:.4f}")

    # Save All 3 Models
    models_to_save = {
        'delivery_xgboost_opt.pkl': best_xgb,
        'delivery_rf.pkl': rf,
        'delivery_xgboost_base.pkl': xgb_base
    }
    
    for filename, model in models_to_save.items():
        path = os.path.join(path_utils.MODELS_DIR, filename)
        joblib.dump(model, path)
        print(f"Saved: {filename}")

    # Compatibility: Save the absolute best to the default path
    if rf_acc > xgb_acc:
        joblib.dump(rf, os.path.join(path_utils.MODELS_DIR, 'delivery_xgboost.pkl'))
    else:
        joblib.dump(best_xgb, os.path.join(path_utils.MODELS_DIR, 'delivery_xgboost.pkl'))

    print("\n--- Optimized Delivery Training Completed (All 3 Models Saved) ---")

if __name__ == "__main__":
    train_delivery_models()
