import os
import pandas as pd
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_feature_engineering():
    print("--- Starting Feature Engineering ---")
    
    # Define paths
    raw_data_path = os.path.join(path_utils.RAW_DATA_DIR, 'DataCoSupplyChainDataset.csv')
    processed_dir = path_utils.PROCESSED_DATA_DIR
    
    # Load Data
    print(f"Loading data from: {raw_data_path}")
    df = pd.read_csv(raw_data_path, encoding='ISO-8859-1')

    # 1. Create fraud target
    print("Creating fraud target...")
    df['is_fraud'] = (df['Order Status'] == 'SUSPECTED_FRAUD').astype(int)

    # 2. Shipping delay
    print("Creating shipping delay feature...")
    df['shipping_delay'] = df['Days for shipping (real)'] - df['Days for shipment (scheduled)']

    # 3. Discount ratio
    print("Creating discount ratio feature...")
    # Add small epsilon to avoid division by zero
    df['discount_ratio'] = df['Order Item Discount'] / (df['Order Item Product Price'] + 1e-9)

    # 4. Extract temporal features from Order Date
    print("Extracting temporal features...")
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df['order_month'] = df['order_date'].dt.month
    df['order_day_of_week'] = df['order_date'].dt.dayofweek
    df['order_year'] = df['order_date'].dt.year

    # 5. Drop columns that leak targets or are irrelevant (PII)
    print("Dropping leaky and irrelevant columns...")
    
    # Common PII and irrelevant columns
    pii_columns = [
        'Customer Email', 'Customer Password', 'Customer Fname', 
        'Customer Lname', 'Product Description', 'Order Zipcode',
        'Customer Zipcode', 'Customer Street', 'Order Item Id', 'Order Id',
        'Customer Id', 'Product Card Id', 'Product Category Id',
        'Product Status', 'shipping date (DateOrders)', 'order date (DateOrders)', 'order_date'
    ]
    
    # Columns that leak 'Late_delivery_risk'
    delivery_leaky = ['Delivery Status', 'Days for shipping (real)']
    
    # Columns that leak 'is_fraud'
    fraud_leaky = ['Order Status']

    # Keep original dataframe for processing both targets
    df_engineered = df.drop(columns=pii_columns)
    
    # Save processed dataframe (this will be further processed in step 03)
    output_path = os.path.join(processed_dir, 'supply_chain_engineered.csv')
    df_engineered.to_csv(output_path, index=False)
    print(f"Saved engineered dataset to: {output_path}")

    print("\n--- Feature Engineering Completed Successfully ---")

if __name__ == "__main__":
    run_feature_engineering()
