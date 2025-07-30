import streamlit as st
import numpy as np
import joblib

# Load saved scaler and model
scaler = joblib.load("kmeans_scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

def interpret_cluster(cluster):
    labels = {
        0: "Low Activity, Low Engagement Customer",
        1: "High-Spending, Full-Payment Customer",
        2: "Cash Advance Heavy, Risk-Prone Customer",
        3: "Installment-Focused, Active User"
    }
    insights = {
        0: "This user rarely uses their card and is at risk of churning.",
        1: "This user is valuable and should receive loyalty perks.",
        2: "This user heavily depends on cash advances — monitor closely.",
        3: "This user is active and prefers installment-based purchases."
    }
    return labels[cluster], insights[cluster]

st.title("Customer Segmentation App")
st.markdown("Input a customer’s behavior to predict their segment and understand their profile.")

#Input Fields
balance = st.number_input("Average Balance", 0.0, 50000.0, 1000.0)
balance_freq = st.slider("Balance Frequency (0 to 1)", 0.0, 1.0, 0.8)
purchases = st.number_input("Total Purchases", 0.0, 20000.0, 500.0)
oneoff_purchases = st.number_input("One-off Purchases", 0.0, 20000.0, 300.0)
installment_purchases = st.number_input("Installment Purchases", 0.0, 20000.0, 200.0)
cash_advance = st.number_input("Cash Advance Amount", 0.0, 20000.0, 0.0)
purchases_freq = st.slider("Purchase Frequency (0 to 1)", 0.0, 1.0, 0.5)
oneoff_freq = st.slider("One-off Purchase Frequency", 0.0, 1.0, 0.3)
installment_freq = st.slider("Installment Purchase Frequency", 0.0, 1.0, 0.3)
cash_freq = st.slider("Cash Advance Frequency", 0.0, 1.0, 0.1)
cash_trx = st.number_input("Cash Advance Transactions", 0, 100, 2)
purchase_trx = st.number_input("Purchase Transactions", 0, 200, 10)
credit_limit = st.number_input("Credit Limit", 0.0, 50000.0, 2000.0)
payments = st.number_input("Total Payments", 0.0, 50000.0, 1000.0)
min_payments = st.number_input("Minimum Payments", 0.0, 20000.0, 500.0)
full_payment_ratio = st.slider("Percentage of Full Payments (0 to 1)", 0.0, 1.0, 0.2)
tenure = st.slider("Tenure (in months)", 6, 24, 12)

#Predict Button
if st.button("Predict Segment"):
    user_input = np.array([[
        balance, balance_freq, purchases, oneoff_purchases, installment_purchases,
        cash_advance, purchases_freq, oneoff_freq, installment_freq,
        cash_freq, cash_trx, purchase_trx, credit_limit,
        payments, min_payments, full_payment_ratio, tenure
    ]])

    scaled_input = scaler.transform(user_input)
    cluster = kmeans.predict(scaled_input)[0]
    label, insight = interpret_cluster(cluster)

    st.success(f"Predicted Segment: **{label}**")
    st.markdown(f"💡 _{insight}_")

    
