import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import path_utils
from datetime import datetime

# =================================================================
# PAGE CONFIG & STYLING
# =================================================================
st.set_page_config(
    page_title="SupplyChain XGB",
    page_icon="🚚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Deep Navy Dark Theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #f1f5f9 !format;
    }
    
    .stApp {
        background: radial-gradient(circle at top left, #1e293b, #0f172a);
        color: #f1f5f9;
    }
    
    /* Transparent Modern Card / Glass */
    .stCard {
        background: rgba(30, 41, 59, 0.7);
        padding: 25px;
        border-radius: 16px;
        backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 10px 40px rgba(0,0,0,0.3);
        margin-bottom: 25px;
        transition: transform 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-5px);
        border: 1px solid rgba(34, 211, 238, 0.3);
    }
    
    /* Metrics Highlighting */
    .stMetric {
        background: rgba(15, 23, 42, 0.5);
        padding: 20px;
        border-radius: 12px;
        border-left: 4px solid #22d3ee;
    }
    
    .stMetric label { color: #94a3b8 !important; font-weight: 600 !important; }
    .stMetric div[data-testid="stMetricValue"] { color: #f1f5f9 !important; font-size: 1.8rem !important; }
    
    /* Tables & Inputs */
    .stDataFrame { background: rgba(30, 41, 59, 0.5); border-radius: 10px; }
    .stTextInput>div>div>input, .stSelectbox>div>div>div {
        background-color: #1e293b !important;
        color: #f1f5f9 !important;
        border: 1px solid #334155 !important;
    }
    
    .status-box {
        padding: 15px;
        border-radius: 12px;
        font-weight: 700;
        text-align: center;
        margin-top: 15px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .fraud-alert { background: linear-gradient(90deg, #991b1b, #dc2626); color: #fee2e2; box-shadow: 0 0 20px rgba(220, 38, 38, 0.3); }
    .fraud-safe { background: linear-gradient(90deg, #065f46, #059669); color: #ecfdf5; box-shadow: 0 0 20px rgba(5, 150, 105, 0.3); }
    .risk-high { background: linear-gradient(90deg, #9a3412, #ea580c); color: #ffedd5; box-shadow: 0 0 20px rgba(234, 88, 12, 0.3); }
    .risk-low { background: linear-gradient(90deg, #1e40af, #2563eb); color: #dbeafe; box-shadow: 0 0 20px rgba(37, 99, 235, 0.3); }
    
    .sidebar-header {
        font-size: 1.4rem;
        font-weight: 800;
        color: #22d3ee;
        margin-bottom: 25px;
        letter-spacing: -0.5px;
    }
    
    /* Executive Table Styling */
    table { width: 100%; color: #cbd5e1 !important; border-collapse: separate !important; border-spacing: 0 8px !important; }
    th { color: #22d3ee !important; text-transform: uppercase; font-size: 0.8rem; padding: 12px !important; }
    td { background: rgba(15, 23, 42, 0.3); padding: 14px !important; border-top: 1px solid rgba(255,255,255,0.02); }
    tr:hover td { background: rgba(34, 211, 238, 0.05); }
    </style>
""", unsafe_allow_html=True)

# =================================================================
# DATA & MODEL LOADING
# =================================================================
@st.cache_resource
def load_assets():
    models_dir = path_utils.MODELS_DIR
    try:
        # Load all 3 delivery models
        del_model_opt = joblib.load(os.path.join(models_dir, 'delivery_xgboost_opt.pkl'))
        try:
            del_model_rf = joblib.load(os.path.join(models_dir, 'delivery_rf.pkl'))
        except FileNotFoundError:
            del_model_rf = None
        del_model_base = joblib.load(os.path.join(models_dir, 'delivery_xgboost_base.pkl'))
        
        fraud_model = joblib.load(os.path.join(models_dir, 'fraud_xgboost.pkl'))
        scaler_del = joblib.load(os.path.join(models_dir, 'scaler_delivery.joblib'))
        scaler_fra = joblib.load(os.path.join(models_dir, 'scaler_fraud.joblib'))
        encoders = joblib.load(os.path.join(models_dir, 'encoders.joblib'))
        freq_encodings = joblib.load(os.path.join(models_dir, 'freq_encodings.joblib'))
        del_features = joblib.load(os.path.join(models_dir, 'delivery_features.joblib'))
        fra_features = joblib.load(os.path.join(models_dir, 'fraud_features.joblib'))
        
        del_models = {
            'opt': del_model_opt,
            'rf': del_model_rf,
            'base': del_model_base
        }
        
        return del_models, fraud_model, scaler_del, scaler_fra, encoders, freq_encodings, del_features, fra_features
    except Exception as e:
        st.error(f"Failed to load pipeline assets. Please ensure all scripts in `pipeline/` have been executed correctly. Error: {e}")
        return None

assets = load_assets()
if assets:
    del_models, fra_model, scaler_del, scaler_fra, encoders, freq_encodings, del_features, fra_features = assets
else:
    st.stop()

# =================================================================
# SIDEBAR NAVIGATION
# =================================================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/5760/5760599.png", width=80)
    st.markdown("<div class='sidebar-header'>SupplyChain XGB</div>", unsafe_allow_html=True)
    
    selected_task = st.radio(
        "Operations Control",
        ["Dashboard Overview", "Delivery Risk Engine", "Fraud Security Shield", "Performance Analytics"],
        index=0
    )
    
    st.markdown("---")
    st.caption("v2.2.0-optimized | SupplyChain XGB")
    st.info("💡 Accuracy Boost: We now use Frequency Encoding for Cities and States, reaching 83% prediction accuracy.")

# =================================================================
# TAB 1: EXECUTIVE OVERVIEW
# =================================================================
if selected_task == "Dashboard Overview":
    st.title("🏛️ Executive Operations Overview")
    
    st.markdown("""
    ### 📌 Project Strategic Summary
    This intelligence suite provides high-precision predictive modeling for **SupplyChain XGB**. 
    By analyzing 180K+ historical transactions, we address two critical operational risks:
    1. **Logistics Fragility**: Predicting late deliveries before they occur to maintain SLAs.
    2. **Revenue Integrity**: Identifying fraudulent patterns to prevent financial leakage.
    """)
    
    # KPI Row
    st.markdown("### 📈 Core Business KPIs")
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.metric("Delivery Accuracy", "82.7%", "Target: 83-90%")
    with kpi2:
        st.metric("Fraud Recall", "97.4%", "Goal: >75%")
    with kpi3:
        st.metric("Service Reliability", "High", "XGBoost Engine")
    with kpi4:
        st.metric("Risk Coverage", "100%", "Real-time Scoring")

    # Executive Details Card
    st.markdown("""
    <div class='stCard'>
        <h3>🛡️ Model Governance & Performance</h3>
        <p>We have deployed two specialized machine learning architectures optimized for logistics domain-specific complexities.</p>
        <table style='width:100%; border-collapse: collapse; margin-top: 10px;'>
            <tr style='background-color: #f8fafc; border-bottom: 2px solid #e2e8f0;'>
                <th style='padding: 10px; text-align: left;'>Model Objective</th>
                <th style='padding: 10px; text-align: left;'>Best Performing Model</th>
                <th style='padding: 10px; text-align: left;'>Primary Metric</th>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'><b>Late Delivery Prediction</b></td>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'>Optimized XGBoost (Tuned)</td>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'>83% Accuracy</td>
            </tr>
            <tr>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'><b>Fraud Security Shield</b></td>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'>XGBoost + SMOTETomek</td>
                <td style='padding: 10px; border-bottom: 1px solid #edf2f7;'>97% Recall</td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛠️ Strategic Functional Modules")
    c1, c2 = st.columns(2)
    with c1:
        st.info("🚚 **Delivery Risk Engine**: Real-time probability assessment using frequency-encoded geographic features.")
    with c2:
        st.success("🛡️ **Fraud Shield**: Advanced detection system optimized for high-sensitivity financial protection.")

# =================================================================
# TAB 2: DELIVERY RISK
# =================================================================
elif selected_task == "Delivery Risk Engine":
    st.title("🚚 Optimized Delivery Performance")
    st.markdown("Input shipment details to calculate the probability of a delivery delay.")
    
    # User selects which engine to use
    st.markdown("### ⚙️ Engine Selection")
    selected_model_name = st.radio(
        "Select Predictive Model",
        ["Optimized XGBoost (83% Acc)", "Random Forest (~78% Acc)", "Baseline XGBoost (~73% Acc)"],
        horizontal=True
    )
    
    # Map selection to model dictionary key
    if "Optimized XGBoost" in selected_model_name:
        del_model = del_models['opt']
    elif "Random Forest" in selected_model_name:
        del_model = del_models['rf'] if del_models['rf'] is not None else del_models['opt']
        if del_models['rf'] is None:
            st.warning("Note: The 723MB Random Forest model was naturally excluded from this server deploy to preserve fast performance. Using the Optimized XGBoost fallback.")
    else:
        del_model = del_models['base']
    
    st.markdown("---")
    
    with st.form("delivery_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Transport")
            mode = st.selectbox("Shipping Mode", encoders['Shipping Mode'].classes_)
            market = st.selectbox("Market", encoders['Market'].classes_)
            order_city = st.selectbox("Order City", list(freq_encodings['Order City'].index)[:100])
            order_state = st.selectbox("Order State", list(freq_encodings['Order State'].index)[:50])
            
        with col2:
            st.subheader("Logistics")
            scheduled = st.number_input("Scheduled Days", 0, 7, 2)
            payment = st.selectbox("Payment Type", encoders['Type'].classes_)
            segment = st.selectbox("Customer Segment", encoders['Customer Segment'].classes_)
            
        with col3:
            st.subheader("Order Details")
            category = st.selectbox("Category", encoders['Category Name'].classes_)
            sales = st.number_input("Order Sales ($)", 0.0, 5000.0, 200.0)
            benefit = st.number_input("Profit/Benefit ($)", -500.0, 1000.0, 50.0)

        submit = st.form_submit_button("Run Risk Assessment")
        
        if submit:
            # Prepare Input Data
            input_df = pd.DataFrame(columns=del_features)
            input_df.loc[0] = 0 
            
            # Label Encoding
            input_df['Shipping Mode'] = encoders['Shipping Mode'].transform([mode])[0]
            input_df['Market'] = encoders['Market'].transform([market])[0]
            input_df['Type'] = encoders['Type'].transform([payment])[0]
            input_df['Customer Segment'] = encoders['Customer Segment'].transform([segment])[0]
            input_df['Category Name'] = encoders['Category Name'].transform([category])[0]
            
            # Frequency Encoding
            input_df['Order City'] = freq_encodings['Order City'].get(order_city, 0)
            input_df['Order State'] = freq_encodings['Order State'].get(order_state, 0)
            input_df['Customer City'] = freq_encodings['Customer City'].get(order_city, 0) # Fallback to order city for demo
            input_df['Customer State'] = freq_encodings['Order State'].get(order_state, 0)
            
            # Numerical
            input_df['Days for shipment (scheduled)'] = scheduled
            input_df['Sales'] = sales
            input_df['Benefit per order'] = benefit
            input_df['discount_ratio'] = 0.1
            
            # Temporal
            now = datetime.now()
            input_df['order_month'] = now.month
            input_df['order_day_of_week'] = now.weekday()
            input_df['order_year'] = now.year
            
            # Alignment & Scaling
            input_df = input_df[del_features] # Important: Match model column order
            input_scaled = scaler_del.transform(input_df)
            prob = del_model.predict_proba(input_scaled)[0][1]
            
            # Result UI
            st.markdown("---")
            r1, r2 = st.columns([1, 2])
            with r1:
                st.metric("Risk Score", f"{prob:.1%}")
            
            with r2:
                if prob > 0.65:
                    st.markdown(f"<div class='status-box risk-high'>⚠️ HIGH RISK: Delay Highly Probable</div>", unsafe_allow_html=True)
                elif prob > 0.4:
                    st.markdown(f"<div class='status-box risk-high' style='background-color:#fff9e6; color:#9a6300; border-color:#ffeeba;'>⚖️ MODERATE RISK</div>", unsafe_allow_html=True)
                else:
                    st.markdown(f"<div class='status-box risk-low'>✅ LOW RISK: Reliable Delivery Path</div>", unsafe_allow_html=True)

# =================================================================
# TAB 3: FRAUD PROTECTION
# =================================================================
elif selected_task == "Fraud Security Shield":
    st.title("🛡️ Financial Security Analytics")
    st.markdown("Real-time fraud scoring for incoming orders.")
    
    with st.container():
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        f_col1, f_col2 = st.columns(2)
        
        with f_col1:
            f_payment = st.selectbox("Transaction Type", encoders['Type'].classes_)
            f_market = st.selectbox("Operation Market", encoders['Market'].classes_)
            f_sales = st.number_input("Transaction Value ($)", 0.0, 5000.0, 350.0)
            f_status = st.selectbox("Current Operational Status", encoders['Delivery Status'].classes_)
            
        with f_col2:
            f_profit = st.number_input("Calculated Profit ($)", -1000.0, 2000.0, 75.0)
            f_discount = st.slider("Discount Policy Rate", 0.0, 0.7, 0.15)
            f_city = st.selectbox("Delivery City", list(freq_encodings['Order City'].index)[:100])
        
        if st.button("Check Integrity Score", type="primary"):
            # Prepare Input
            f_input = pd.DataFrame(columns=fra_features)
            f_input.loc[0] = 0
            
            f_input['Type'] = encoders['Type'].transform([f_payment])[0]
            f_input['Market'] = encoders['Market'].transform([f_market])[0]
            f_input['Delivery Status'] = encoders['Delivery Status'].transform([f_status])[0]
            f_input['Sales per customer'] = f_sales
            f_input['Benefit per order'] = f_profit
            f_input['Order City'] = freq_encodings['Order City'].get(f_city, 0)
            f_input['Order Item Discount Rate'] = f_discount
            f_input['discount_ratio'] = f_discount
            
            # Temporal
            now = datetime.now()
            f_input['order_month'] = now.month
            f_input['order_day_of_week'] = now.weekday()
            
            # Alignment & Scale
            f_input = f_input[fra_features]
            f_scaled = scaler_fra.transform(f_input)
            f_prob = fra_model.predict_proba(f_scaled)[0][1]
            
            st.markdown("### Security Analysis Result")
            if f_prob > 0.3:
                st.markdown(f"<div class='status-box fraud-alert'>💥 [ALARM] SUSPECTED FRAUD | SCORE: {f_prob:.2f}</div>", unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='status-box fraud-safe'>🛡️ TRANSACTION SECURE | SCORE: {f_prob:.2f}</div>", unsafe_allow_html=True)
                st.balloons()
        st.markdown("</div>", unsafe_allow_html=True)

# =================================================================
# TAB 4: ANALYTICS
# =================================================================
elif selected_task == "Performance Analytics":
    st.title("📊 Model Performance Hub")
    
    st.markdown("### Optimized Model Quality")
    tabs = st.tabs(["Delivery Risk (83%)", "Fraud Shield (97%)", "Logistics Context"])
    
    with tabs[0]:
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "delivery_roc_curve.png"))
        with c2:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "feature_importance_delivery.png"))
            
    with tabs[1]:
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "fraud_roc_curve.png"))
        with c2:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "fraud_confusion_matrix.png"))
            
    with tabs[2]:
        st.markdown("#### Operational Insights")
        c1, c2 = st.columns(2)
        with c1:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "late_risk_by_shipping_mode.png"))
        with c2:
            st.image(os.path.join(path_utils.OUTPUTS_DIR, "benefit_per_order_dist.png"))

# Sidebar footer
st.sidebar.markdown("---")
st.sidebar.write("Optimized with Antigravity AI Engine")
