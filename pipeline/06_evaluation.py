import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, f1_score, accuracy_score
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def evaluate_models():
    print("--- Starting Detailed Model Evaluation ---")
    
    # Define paths
    processed_dir = path_utils.PROCESSED_DATA_DIR
    models_dir = path_utils.MODELS_DIR
    output_dir = path_utils.OUTPUTS_DIR
    
    # Load Models
    print("Loading best models...")
    delivery_model = joblib.load(os.path.join(models_dir, 'delivery_xgboost.pkl'))
    fraud_model = joblib.load(os.path.join(models_dir, 'fraud_xgboost.pkl'))

    # Load Test Data
    print("Loading test data...")
    X_test_del = np.load(os.path.join(processed_dir, 'X_test_del.npy'))
    y_test_del = np.load(os.path.join(processed_dir, 'y_test_del.npy'))
    X_test_fr = np.load(os.path.join(processed_dir, 'X_test_fr.npy'))
    y_test_fr = np.load(os.path.join(processed_dir, 'y_test_fr.npy'))

    # 1. Evaluate Delivery Risk Model
    print("\n--- Delivery Risk Prediction Evaluation ---")
    y_pred_del = delivery_model.predict(X_test_del)
    y_prob_del = delivery_model.predict_proba(X_test_del)[:, 1]
    
    print("\nClassification Report (Delivery):")
    print(classification_report(y_test_del, y_pred_del))

    # Confusion Matrix (Delivery)
    cm_del = confusion_matrix(y_test_del, y_pred_del)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_del, annot=True, fmt='d', cmap='Blues', xticklabels=['On-time', 'Late'], yticklabels=['On-time', 'Late'])
    plt.title('Delivery Risk Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'delivery_confusion_matrix.png'))
    plt.close()
    print("Saved: delivery_confusion_matrix.png")

    # ROC Curve (Delivery)
    fpr_del, tpr_del, _ = roc_curve(y_test_del, y_prob_del)
    roc_auc_del = auc(fpr_del, tpr_del)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_del, tpr_del, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc_del:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Delivery Risk ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'delivery_roc_curve.png'))
    plt.close()
    print("Saved: delivery_roc_curve.png")

    # 2. Evaluate Fraud Detection Model
    print("\n--- Fraud Detection Evaluation ---")
    y_pred_fr = fraud_model.predict(X_test_fr)
    y_prob_fr = fraud_model.predict_proba(X_test_fr)[:, 1]
    
    print("\nClassification Report (Fraud):")
    print(classification_report(y_test_fr, y_pred_fr))

    # Confusion Matrix (Fraud)
    cm_fr = confusion_matrix(y_test_fr, y_pred_fr)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_fr, annot=True, fmt='d', cmap='Reds', xticklabels=['Non-Fraud', 'Fraud'], yticklabels=['Non-Fraud', 'Fraud'])
    plt.title('Fraud Detection Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.savefig(os.path.join(output_dir, 'fraud_confusion_matrix.png'))
    plt.close()
    print("Saved: fraud_confusion_matrix.png")

    # ROC Curve (Fraud)
    fpr_fr, tpr_fr, _ = roc_curve(y_test_fr, y_prob_fr)
    roc_auc_fr = auc(fpr_fr, tpr_fr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_fr, tpr_fr, color='darkred', lw=2, label=f'ROC curve (AUC = {roc_auc_fr:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.title('Fraud Detection ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, 'fraud_roc_curve.png'))
    plt.close()
    print("Saved: fraud_roc_curve.png")

    # 3. Feature Importance (Delivery best model example)
    print("\nGenerating feature importance chart for Delivery model...")
    # Using saved delivery feature names for consistency
    del_feature_names = joblib.load(os.path.join(models_dir, 'delivery_features.joblib'))
    
    # Check if model has feature_importances_ (Random Forest and XGBoost do)
    if hasattr(delivery_model, 'feature_importances_'):
        feat_importances = pd.Series(delivery_model.feature_importances_, index=del_feature_names)
        plt.figure(figsize=(10, 8))
        feat_importances.nlargest(15).plot(kind='barh', color='purple')
        plt.title('Top 15 Feature Importances (Delivery Risk)')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'feature_importance_delivery.png'))
        plt.close()
        print("Saved: feature_importance_delivery.png")
    else:
        print("Current best model does not support feature_importances_ visualization.")

    print("\n--- Model Evaluation Completed Successfully ---")

if __name__ == "__main__":
    evaluate_models()
