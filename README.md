
# Customer Segmentation with KMeans Clustering

This project applies unsupervised machine learning to segment credit card customers based on their financial behavior. By grouping customers into distinct behavioral clusters, fintech companies can personalize marketing strategies, manage credit risk more effectively, and improve customer retention.

---

## Business Problem

Most fintech platforms — from digital wallets to loan apps — treat users the same way, regardless of how they behave. Some customers are inactive, others are high spenders, and some rely heavily on short-term credit. Yet, they receive the same messages, limits, and offers.
While the dataset is global, the behavioral insights and clustering strategy can be adapted to fintech customers in Nigeria or other emerging markets.

This project explores how customer segmentation can help fintechs:
- Personalize product offers (e.g. credit, loyalty, BNPL)
- Identify risky customers based on behavior (e.g. cash advance usage)
- Proactively re-engage low-activity users
- Support data-driven decisions across marketing, lending, and product teams

---

## Project Objective

> Segment credit card users into distinct behavioral groups using **unsupervised learning** (KMeans) and **real financial usage data**, then make the results actionable through a deployed app.

---

## Dataset

- **Source**: [Kaggle – Credit Card Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)
- **Records**: ~8,900 customers
- **Key Features**:
  - Spending behavior (`PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`)
  - Cash usage (`CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`)
  - Repayment patterns (`PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`)
  - Tenure and credit limits
  - Frequency of usage (`PURCHASES_TRX`, `BALANCE_FREQUENCY`)

---

## Data Cleaning & Preprocessing

- Dropped `CUST_ID` (identifier only)
- Handled missing values using median imputation
- Scaled features using `StandardScaler`
- Retained 17 core numerical features for clustering

---

## Clustering Model

- **Algorithm**: KMeans clustering
- **Optimal Clusters**: 4 (determined using Elbow + Silhouette Score)
- **Dimensionality Reduction**: PCA used for visualization only (not for clustering)
- **Model Serialization**: Saved `KMeans` model and `StandardScaler` using `joblib`

---

## Cluster Insights

| Cluster | Segment Name                       | Behavior Summary                                     |
|--------:|------------------------------------|-----------------------------------------------------|
| 0       | Low Activity, Low Engagement       | Infrequent use, low balance, rarely pays in full    |
| 1       | High-Spending, Full-Payment Users  | High purchases, pays in full, low credit risk       |
| 2       | Cash Advance Heavy, Risk-Prone     | High cash advance usage, low full repayment         |
| 3       | Installment-Focused Active Users   | Regular usage, prefers installments, steady payment |

Each segment can be used by fintech teams to tailor engagement, credit offers, or risk monitoring.

---

## Streamlit App

A lightweight Streamlit app was built to allow real-time customer segmentation based on manual input.

### App Features:
- Accepts 17 numeric inputs from users
- Scales input using trained scaler
- Predicts cluster and displays:
  - Segment label
  - Behavioral summary

### Run Locally:
```bash
streamlit run app.py
```

---

## Project Structure

```
customer-segmentation-kmeans/
├── customer_segmentation_kmeans.py                       # Streamlit app
├── model/
│   ├── kmeans.pkl               # Trained KMeans model
│   └── scaler.pkl               # Trained StandardScaler
├── notebooks/
│   └── clustering_pipeline.ipynb  # EDA + modeling                   
├── requirements.txt
└── README.md
```

---

## Tech Stack

- **Python**
- **scikit-learn** (KMeans, StandardScaler, PCA)
- **Pandas / NumPy**
- **Streamlit** (App interface)
- **joblib** (Model saving/loading)
- **matplotlib / seaborn** (Visualizations)

---

## How to Use

1. Clone this repository:
```bash
git clone https://github.com/your-username/customer-segmentation-kmeans.git
cd customer-segmentation-kmeans
```

2. Install the required libraries:
```bash
pip install -r requirements.txt
```

3. Run the app:
```bash
streamlit run app.py
```

---

## Future Improvements

- Try alternative clustering methods (DBSCAN, GMM)
- Deploy the Streamlit app on Streamlit Cloud or EC2
- Add automatic profiling reports per cluster
- Build dashboards for real-time product or marketing use

---

## Author

**Chukwunonyelim Gabriel Okonji**  
📧 nonnyokonji@gmail.com  
🔗 [GitHub](https://github.com/Nonny-123)  
🔗 [LinkedIn](https://linkedin.com/in/your-profile)

---

## License

This project is licensed under the **MIT License** — feel free to use, adapt, or extend with credit.

---

## If you found this useful...

Give the repo a star and share your feedback! I'm always open to suggestions or collaboration.
