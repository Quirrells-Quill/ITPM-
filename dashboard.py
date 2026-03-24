import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="Smart Retail Dashboard", layout="wide")

st.title("🛒 Smart Retail Analytics Dashboard")

# File path
file_path = "output/reports.csv"

# Check if file exists
if not os.path.exists(file_path):
    st.error("⚠️ No report found. Please run app.py first.")
else:
    df = pd.read_csv(file_path)

    st.subheader("📊 Summary Metrics")

    col1, col2, col3 = st.columns(3)

    unique = df[df["Metric"] == "Total Unique Customers"]["Value"].values[0]
    entry = df[df["Metric"] == "Total Entry"]["Value"].values[0]
    exit = df[df["Metric"] == "Total Exit"]["Value"].values[0]

    col1.metric("👥 Unique Customers", unique)
    col2.metric("🟢 Entry", entry)
    col3.metric("🔴 Exit", exit)

    st.subheader("📄 Report Data")
    st.dataframe(df)
    st.subheader("📈 Visualization")

    st.bar_chart(df.set_index("Metric"))

    st.success("✅ Dashboard Loaded Successfully")