import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from imblearn.combine import SMOTETomek
from sklearn.metrics import classification_report, f1_score
import joblib
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def train_fraud_model():
    print("--- Starting Fraud Detection Model Training ---")
    
    # Load processed data
    processed_dir = path_utils.PROCESSED_DATA_DIR
    X_train = np.load(os.path.join(processed_dir, 'X_train_fr.npy'))
    X_test = np.load(os.path.join(processed_dir, 'X_test_fr.npy'))
    y_train = np.load(os.path.join(processed_dir, 'y_train_fr.npy'))
    y_test = np.load(os.path.join(processed_dir, 'y_test_fr.npy'))

    # 1. Handle Class Imbalance
    print("\nHandling severe class imbalance with SMOTETomek...")
    smote_tomek = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smote_tomek.fit_resample(X_train, y_train)
    print(f"Resampled Training Set Shape: {X_train_resampled.shape}")
    print(f"Resampled Class Counts: {pd.Series(y_train_resampled).value_counts()}")

    # 2. Train Model (XGBoost)
    print("\nTraining XGBoost on resampled data...")
    # Using scale_pos_weight as per instruction for extra robustness
    ratio = pd.Series(y_train).value_counts()[0] / pd.Series(y_train).value_counts()[1]
    
    model = XGBClassifier(
        n_estimators=300, 
        learning_rate=0.1, 
        max_depth=7, 
        scale_pos_weight=ratio,
        random_state=42, 
        use_label_encoder=False, 
        eval_metric='logloss'
    )
    
    model.fit(X_train_resampled, y_train_resampled)

    # 3. Predict and Evaluate
    y_pred = model.predict(X_test)
    print("\nEvaluation (Fraud Case Highlighted):")
    print(classification_report(y_test, y_pred))

    f1_fraud = f1_score(y_test, y_pred)
    print(f"\nFraud F1-Score: {f1_fraud:.4f}")

    # 4. Save Model
    model_path = os.path.join(path_utils.MODELS_DIR, 'fraud_xgboost.pkl')
    joblib.dump(model, model_path)
    print(f"Saved fraud model to: {model_path}")

    print("\n--- Fraud Training Completed Successfully ---")

if __name__ == "__main__":
    train_fraud_model()
