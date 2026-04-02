**PROJECT INSTRUCTION FILE**

**Supply Chain Analytics — Delivery Performance & Fraud Detection**

Machine Learning Project  |  Classification + Regression  |  Logistics Domain

|<p>Dataset</p><p>**DataCo Supply Chain (Kaggle)**</p>|<p>Rows</p><p>**180K orders**</p>|<p>Difficulty</p><p>**Medium**</p>|<p>Target Accuracy</p><p>**83–90%**</p>|
| :-: | :-: | :-: | :-: |


# **1. Project Overview**
Supply chain analytics applies machine learning to optimize logistics operations — predicting late deliveries, detecting fraudulent orders, and identifying high-risk shipments. This is a multi-target project: you will build two separate models — one for delivery risk classification and one for fraud detection — on the same dataset.

|**Real-World Use Case**|
| :- |
|Logistics companies like FedEx, DHL, Amazon Logistics, and Flipkart Commerce use delivery prediction models to proactively notify customers, reroute shipments, and penalize underperforming carriers. Fraud detection models flag suspicious orders (fake addresses, unusually high-value orders with new accounts) before dispatch, saving millions in losses annually.|

# **2. Dataset Details**
**Source**

- Name: DataCo Smart Supply Chain for Big Data Analysis
- Platform: Kaggle — https://www.kaggle.com/datasets/shashwatwork/dataco-smart-supply-chain-for-big-data-analysis
- Format: CSV — single primary file (DataCoSupplyChainDataset.csv)
- License: Public / Open Use

**Dataset Statistics**

|**Property**|**Value**|
| :- | :- |
|Total Rows|~180,519 orders|
|Total Columns|53 features|
|Target 1|Late\_delivery\_risk (0 = on-time, 1 = late) — classification|
|Target 2|Order Status = 'Suspected\_Fraud' — binary fraud flag|
|Class Distribution (Delivery)|~55% late, ~45% on-time (balanced)|
|Class Distribution (Fraud)|< 1% fraud (highly imbalanced)|
|Missing Values|Minimal — check Product Description and Customer Zipcode|
|Data Types|Mix of numeric, categorical, datetime|

**Key Features**

- Type — payment type (DEBIT, TRANSFER, CASH, PAYMENT)
- Days for shipping (real) — actual days taken to ship
- Days for shipment (scheduled) — planned shipping days
- Benefit per order — profit/loss per order (can be negative = loss)
- Sales per customer — total customer spend
- Delivery Status — Advance shipping, Late delivery, Shipping on time, Shipping canceled
- Late\_delivery\_risk — **TARGET 1**: 0 or 1
- Category Name — product category
- Customer City, Customer Country, Customer Segment — customer demographics
- Customer State — US state
- Market — Americas, Europe, LATAM, Pacific Asia, Africa
- Order City, Order Country, Order Region — order geography
- Order Item Discount, Order Item Discount Rate — discount applied
- Order Item Product Price — unit price
- Order Item Profit Ratio — profit margin per item
- Order Item Quantity — units ordered
- Sales — revenue per item
- Order Profit Per Order — total order profit
- Product Price — product list price
- Order Status — COMPLETE, PENDING, CLOSED, SUSPECTED\_FRAUD, etc. — **TARGET 2 source**
- Shipping Mode — Standard Class, Second Class, First Class, Same Day
- Order Date, Shipping Date — temporal features

# **3. Step-by-Step Workflow**
## **Step 1 — Environment Setup**
Install the required Python libraries before starting:

|pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn|
| :- |

## **Step 2 — Load & Explore Data (EDA)**
1. Load CSV: df = pd.read\_csv('DataCoSupplyChainDataset.csv', encoding='ISO-8859-1')
2. Check shape, dtypes, nulls: df.info(), df.isnull().sum()
3. Plot Late\_delivery\_risk distribution — confirm ~55/45 balance
4. Plot Order Status value counts — identify SUSPECTED\_FRAUD volume
5. Plot late delivery rate by: Shipping Mode, Market, Customer Segment, Category
6. Plot Benefit per order distribution — check for negative values (losses)
7. Plot order volume trend by Order Date (monthly)
8. Correlation heatmap for numeric features

|**Key EDA Finding**|
| :- |
|Shipping Mode is the top predictor of delivery risk — Same Day shipping has ~0% late delivery while Standard Class has ~75%+ late rate. Market and Order Region also show strong geographic patterns. Fraud orders tend to cluster in specific markets and have higher-than-average order values.|

## **Step 3 — Feature Engineering**
1. Create fraud target: df['is\_fraud'] = (df['Order Status'] == 'SUSPECTED\_FRAUD').astype(int)
2. shipping\_delay = df['Days for shipping (real)'] - df['Days for shipment (scheduled)'] (positive = late)
3. discount\_ratio = df['Order Item Discount'] / df['Order Item Product Price'] (normalized discount)
4. profit\_margin = df['Order Item Profit Ratio'] (already computed — verify range)
5. Extract temporal features from Order Date:
   - order\_month = pd.to\_datetime(df['order date (DateOrders)']).dt.month
   - order\_day\_of\_week = pd.to\_datetime(df['order date (DateOrders)']).dt.dayofweek
   - order\_year = pd.to\_datetime(df['order date (DateOrders)']).dt.year
6. Drop columns that directly leak targets:
   - For delivery model: drop Delivery Status, Days for shipping (real) (post-event)
   - For fraud model: drop Order Status (used to create the target)
   - Drop customer name, email, password, address — PII and irrelevant

## **Step 4 — Data Preprocessing**
1. Drop high-cardinality or PII columns: Customer Email, Customer Password, Product Description, Customer Fname, Customer Lname, Order Zipcode
2. Encode categorical columns:
   - Low cardinality (< 10 unique): LabelEncoder or map — Type, Shipping Mode, Customer Segment
   - Medium cardinality (10–50): LabelEncoder — Category Name, Market
   - High cardinality (> 50): Frequency encoding or drop — Customer City, Order City
3. Handle missing values: fill Customer Zipcode nulls with 0 or 'Unknown'
4. Scale numeric features using StandardScaler — required for KNN; optional for tree models
5. Split separately for each model task:
   - Delivery model: drop is\_fraud from features; stratify on Late\_delivery\_risk
   - Fraud model: drop Late\_delivery\_risk from features; stratify on is\_fraud

## **Step 5 — Handle Class Imbalance**

**Delivery model (~55/45):** Minimal imbalance — class\_weight='balanced' is sufficient

**Fraud model (< 1% fraud):** Severe imbalance — requires aggressive handling:
- Use SMOTE + Tomek Links (SMOTETomek) for cleaner boundaries
- Set XGBoost scale\_pos\_weight to ratio of non-fraud to fraud (~100:1)
- Lower classification threshold from 0.5 to 0.2–0.3 to maximize recall

## **Step 6 — Model Building**

**MODEL A — Delivery Risk Prediction:**

|**Model**|**Expected Accuracy**|
| :- | :- |
|Logistic Regression (baseline)|78 – 82%|
|KNN (k=5–15)|80 – 84%|
|Random Forest|85 – 88%|
|XGBoost|86 – 90%|

**MODEL B — Fraud Detection:**

|**Model**|**Expected F1 (Fraud)**|
| :- | :- |
|Logistic Regression (baseline)|30 – 45%|
|Random Forest + class\_weight|55 – 68%|
|XGBoost + scale\_pos\_weight|65 – 78%|
|XGBoost + SMOTE|70 – 82%|

Recommended order for both: Logistic Regression → KNN/Random Forest → XGBoost as final.

## **Step 7 — Hyperparameter Tuning**
1. Use RandomizedSearchCV with cv=5
2. XGBoost key params: n\_estimators (100–400), max\_depth (3–8), learning\_rate (0.01–0.2)
3. KNN key params: n\_neighbors (3–20), metric ('euclidean', 'manhattan'), weights ('uniform', 'distance')
4. Random Forest: n\_estimators (100–300), max\_depth (10–None), min\_samples\_split (2–10)
5. Use scoring='roc\_auc' for delivery model, scoring='f1' for fraud model

## **Step 8 — Evaluate the Models**

**Delivery Risk Model:**

|**Metric**|**Target Value**|
| :- | :- |
|Accuracy|83 – 90%|
|Precision|> 80%|
|Recall|> 82%|
|F1-Score|> 81%|
|AUC-ROC|> 0.88|

**Fraud Detection Model:**

|**Metric**|**Target Value**|
| :- | :- |
|Accuracy|> 99% (not reliable)|
|Precision (Fraud)|> 65%|
|Recall (Fraud)|> 75%|
|F1-Score (Fraud)|> 70%|
|AUC-ROC|> 0.87|

# **4. Feature Importance**

|**Rank**|**Feature**|**Model**|**Business Insight**|
| :- | :- | :- | :- |
|1|Shipping Mode|Delivery|Same Day has near-zero late risk|
|2|shipping\_delay (engineered)|Delivery|Direct lag between scheduled and actual|
|3|Days for shipment (scheduled)|Delivery|Longer planned time = buffer for delays|
|4|Market / Order Region|Both|Geographic risk zones differ widely|
|5|Order Item Profit Ratio|Fraud|Low/negative margin orders flag suspicious|
|6|Benefit per order|Fraud|Loss-generating orders are fraud signals|
|7|Customer Segment|Delivery|Corporate vs Consumer delivery patterns differ|
|8|Category Name|Both|Certain product categories have higher fraud/delay|
|9|discount\_ratio (engineered)|Fraud|Unusually high discounts signal fraud|
|10|Sales per customer|Fraud|New high-value orders from new customers = risk|

# **5. Project Structure**

```
05_supply_chain_analytics/
├── data/
│   ├── raw/DataCoSupplyChainDataset.csv
│   └── processed/
│       ├── delivery_features.csv
│       └── fraud_features.csv
├── models/
│   ├── delivery_xgboost.pkl
│   └── fraud_xgboost.pkl
├── pipeline/
│   ├── 01_eda.py
│   ├── 02_feature_engineering.py
│   ├── 03_preprocessing.py
│   ├── 04_delivery_model.py
│   ├── 05_fraud_model.py
│   └── 06_evaluation.py
├── outputs/
│   ├── delivery_confusion_matrix.png
│   ├── delivery_roc_curve.png
│   ├── fraud_confusion_matrix.png
│   ├── fraud_roc_curve.png
│   └── feature_importance_delivery.png
├── app.py
├── path_utils.py
└── README.md
```

**Pipeline File Descriptions:**

| File | Purpose |
| :- | :- |
| 01\_eda.py | Load data, plot delivery risk, fraud distribution, key feature relationships |
| 02\_feature\_engineering.py | Create shipping\_delay, discount\_ratio, temporal features, fraud target, drop leakage |
| 03\_preprocessing.py | Encode categoricals, scale numerics, split into delivery and fraud feature sets |
| 04\_delivery\_model.py | Train Logistic Reg, KNN, Random Forest, XGBoost for delivery risk — save best model |
| 05\_fraud\_model.py | Train with SMOTE + XGBoost for fraud detection — save model |
| 06\_evaluation.py | Generate all metrics, confusion matrices, ROC curves, feature importances for both models |

# **6. Expected Results Summary**

**Delivery Risk Model:**

|**Metric**|**Logistic Reg.**|**XGBoost**|
| :- | :- | :- |
|Accuracy|78 – 82%|86 – 90%|
|Recall|77 – 82%|83 – 88%|
|AUC-ROC|0.84 – 0.87|0.90 – 0.94|

**Fraud Detection Model:**

|**Metric**|**Logistic Reg.**|**XGBoost + SMOTE**|
| :- | :- | :- |
|F1 (Fraud)|30 – 45%|70 – 82%|
|Recall (Fraud)|25 – 40%|75 – 85%|
|AUC-ROC|0.70 – 0.78|0.88 – 0.93|

# **7. Common Mistakes to Avoid**
- Including Delivery Status or Days for shipping (real) as features in the delivery model — these are post-event (data leakage)
- Including Order Status in the fraud model as a feature — it IS the source of the fraud target
- Not splitting features separately for the two models — they require different leakage-free feature sets
- Using accuracy to evaluate the fraud model — with < 1% fraud rate, a model predicting all non-fraud gets 99%+ accuracy but is completely useless
- One-hot encoding high-cardinality columns like Customer City (~3000+ unique values) — use frequency encoding instead
- Not applying SMOTE for the fraud model — class\_weight alone is insufficient for 100:1 imbalance
- Keeping PII columns (email, password, customer name) — not features and must be dropped

# **8. Recommended Tools & Libraries**

|**Library**|**Purpose**|
| :- | :- |
|pandas|Data loading, feature engineering, datetime parsing|
|numpy|Numerical operations|
|scikit-learn|KNN, Random Forest, preprocessing, metrics, cross-validation|
|xgboost|Best classifier for both delivery and fraud models|
|imbalanced-learn|SMOTE + SMOTETomek for fraud class oversampling|
|matplotlib / seaborn|EDA plots, confusion matrices, ROC curves|
|joblib|Save and load trained models|

# **9. Project Deliverables Checklist**
- pipeline/ folder with 6 modular .py files (EDA → engineering → preprocessing → delivery model → fraud model → evaluation)
- Two trained models saved as .pkl: delivery\_xgboost.pkl and fraud\_xgboost.pkl
- Classification Reports + Confusion Matrix visualizations for both models
- ROC Curves comparing all models for each task
- Feature Importance charts for both models
- README.md clearly distinguishing the two prediction tasks and their evaluation metrics
- Streamlit app (app.py) with two tabs — Tab 1: Delivery Risk Predictor (user inputs shipping mode, market, category, days scheduled, customer segment → returns on-time probability + risk level), Tab 2: Fraud Risk Detector (user inputs order value, discount rate, profit margin, market, payment type → returns fraud probability with risk flag)

Supply Chain Analytics  |  ML Project Instruction File  |  Classification Project #5
