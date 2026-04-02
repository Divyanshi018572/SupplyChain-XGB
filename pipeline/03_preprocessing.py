import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_preprocessing():
    print("--- Starting Preprocessing ---")
    
    # Define paths
    engineered_data_path = os.path.join(path_utils.PROCESSED_DATA_DIR, 'supply_chain_engineered.csv')
    processed_dir = path_utils.PROCESSED_DATA_DIR
    models_dir = path_utils.MODELS_DIR
    
    # Load Data
    print(f"Loading engineered data from: {engineered_data_path}")
    df = pd.read_csv(engineered_data_path)

    # 1. Handle missing values
    print("Handling missing values...")
    # Based on EDA, Customer Lname and small zipcode issues. Engineered drop already took care of most.
    # Fill remaining nulls if any (though engineered script dropped some)
    df = df.fillna(0)

    # 1. Drop Irrelevant columns (IDs and non-predictive)
    print("Dropping irrelevant/ID columns...")
    drop_cols = [
        'Order Customer Id', 'Order Item Cardprod Id', 'Product Image', 
        'Customer city', 'Order Zipcode', 'Customer Street',
        'Category Id', 'Department Id' # Redundant with Name
    ]
    # Check if they exist before dropping
    for c in drop_cols:
        if c in df.columns:
            df = df.drop(columns=[c])

    # 2. Handle Cardinality (Frequency Encoding)
    high_card_cols = ['Order City', 'Customer City', 'Order State']
    frequency_encodings = {}
    
    for col in high_card_cols:
        if col in df.columns:
            print(f"Applying Frequency Encoding to {col}...")
            freq = df[col].value_counts() / len(df)
            df[col] = df[col].map(freq)
            frequency_encodings[col] = freq
    
    # Save frequency maps for app.py
    joblib.dump(frequency_encodings, os.path.join(models_dir, 'freq_encodings.joblib'))

    # 3. Identify remaining categoricals
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"Remaining categoricals (to Label Encode): {cat_cols}")

    # 4. Label Encoding for low-to-mid cardinality
    encoders = {}
    for col in cat_cols:
        print(f"Encoding {col}...")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Save encoders
    joblib.dump(encoders, os.path.join(models_dir, 'encoders.joblib'))

    # 5. Handle missing values
    df = df.fillna(0)

    # 6. Define feature sets and targets
    # ... Rest of the function for splits and saves ...
    target_delivery = 'Late_delivery_risk'
    target_fraud = 'is_fraud'

    features_delivery = df.drop(columns=[
        target_delivery, target_fraud, 
        'Delivery Status', 'Days for shipping (real)', 'Order Status', 'shipping_delay'
    ])
    y_delivery = df[target_delivery]
    
    joblib.dump(features_delivery.columns.tolist(), os.path.join(models_dir, 'delivery_features.joblib'))

    features_fraud = df.drop(columns=[
        target_fraud, target_delivery, 
        'Order Status'
    ])
    y_fraud = df[target_fraud]
    
    joblib.dump(features_fraud.columns.tolist(), os.path.join(models_dir, 'fraud_features.joblib'))

    # Scaling
    scaler_delivery = StandardScaler()
    X_delivery_scaled = scaler_delivery.fit_transform(features_delivery)
    scaler_fraud = StandardScaler()
    X_fraud_scaled = scaler_fraud.fit_transform(features_fraud)

    joblib.dump(scaler_delivery, os.path.join(models_dir, 'scaler_delivery.joblib'))
    joblib.dump(scaler_fraud, os.path.join(models_dir, 'scaler_fraud.joblib'))

    # Train/Test Splits
    X_train_del, X_test_del, y_train_del, y_test_del = train_test_split(
        X_delivery_scaled, y_delivery, test_size=0.2, random_state=42, stratify=y_delivery
    )
    X_train_fr, X_test_fr, y_train_fr, y_test_fr = train_test_split(
        X_fraud_scaled, y_fraud, test_size=0.2, random_state=42, stratify=y_fraud
    )

    # Save splits
    np.save(os.path.join(processed_dir, 'X_train_del.npy'), X_train_del)
    np.save(os.path.join(processed_dir, 'X_test_del.npy'), X_test_del)
    np.save(os.path.join(processed_dir, 'y_train_del.npy'), y_train_del)
    np.save(os.path.join(processed_dir, 'y_test_del.npy'), y_test_del)
    np.save(os.path.join(processed_dir, 'X_train_fr.npy'), X_train_fr)
    np.save(os.path.join(processed_dir, 'X_test_fr.npy'), X_test_fr)
    np.save(os.path.join(processed_dir, 'y_train_fr.npy'), y_train_fr)
    np.save(os.path.join(processed_dir, 'y_test_fr.npy'), y_test_fr)

    print("\n--- Optimized Preprocessing Completed Successfully ---")

if __name__ == "__main__":
    run_preprocessing()
