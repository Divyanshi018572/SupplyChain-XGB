import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sys

# Add parent directory to sys.path to import path_utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import path_utils

def run_eda():
    print("--- Starting Exploratory Data Analysis (EDA) ---")
    
    # Define paths
    raw_data_path = os.path.join(path_utils.RAW_DATA_DIR, 'DataCoSupplyChainDataset.csv')
    output_dir = path_utils.OUTPUTS_DIR
    
    # 1. Load Data
    print(f"Loading data from: {raw_data_path}")
    # Using ISO-8859-1 as per instructions
    try:
        df = pd.read_csv(raw_data_path, encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return

    # 2. Check shape, dtypes, nulls
    print(f"\nDataset Shape: {df.shape}")
    print("\nDataset Info Summary:")
    df.info()
    
    null_counts = df.isnull().sum()
    print(f"\nMissing values summary:\n{null_counts[null_counts > 0]}")

    # Set aesthetic style
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    # 3. Plot Late_delivery_risk distribution
    sns.countplot(x='Late_delivery_risk', data=df, palette='viridis')
    plt.title('Late Delivery Risk Distribution (0=On-time, 1=Late)')
    plt.savefig(os.path.join(output_dir, 'late_delivery_risk_dist.png'))
    plt.close()
    print("Saved: late_delivery_risk_dist.png")

    # 4. Plot Order Status value counts
    plt.figure(figsize=(12, 6))
    df['Order Status'].value_counts().plot(kind='bar', color='skyblue')
    plt.title('Order Status Distribution')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'order_status_dist.png'))
    plt.close()
    print("Saved: order_status_dist.png")

    # 5. Plot late delivery rate by key features
    features_to_check = ['Shipping Mode', 'Market', 'Customer Segment']
    for feature in features_to_check:
        plt.figure(figsize=(12, 6))
        sns.barplot(x=feature, y='Late_delivery_risk', data=df, palette='magma')
        plt.title(f'Late Delivery Risk by {feature}')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'late_risk_by_{feature.lower().replace(" ", "_")}.png'))
        plt.close()
        print(f"Saved: late_risk_by_{feature.lower().replace(' ', '_')}.png")

    # 6. Plot Benefit per order distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Benefit per order'], bins=50, kde=True, color='teal')
    plt.title('Distribution of Benefit per Order (Profit/Loss)')
    plt.savefig(os.path.join(output_dir, 'benefit_per_order_dist.png'))
    plt.close()
    print("Saved: benefit_per_order_dist.png")

    # 7. Plot order volume trend by Order Date
    df['order_date'] = pd.to_datetime(df['order date (DateOrders)'])
    df_monthly = df.resample('M', on='order_date').size()
    plt.figure(figsize=(14, 7))
    df_monthly.plot(marker='o', color='coral')
    plt.title('Monthly Order Volume Trend')
    plt.ylabel('Number of Orders')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'order_volume_trend.png'))
    plt.close()
    print("Saved: order_volume_trend.png")

    # 8. Correlation Heatmap
    # Select numeric columns for heatmap
    numeric_df = df.select_dtypes(include=['float64', 'int64'])
    # Focus on key features to keep heatmap readable
    key_numeric = ['Benefit per order', 'Sales per customer', 'Late_delivery_risk', 
                    'Order Item Discount', 'Order Item Product Price', 'Order Item Profit Ratio', 
                    'Order Item Quantity', 'Sales', 'Product Price']
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(numeric_df[key_numeric].corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Heatmap (Key Numeric Features)')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_heatmap.png'))
    plt.close()
    print("Saved: correlation_heatmap.png")

    print("\n--- EDA Completed Successfully ---")

if __name__ == "__main__":
    run_eda()
