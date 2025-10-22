# app.py
import streamlit as st
import pandas as pd
import os
import json
from src.fetch_historical import fetch_and_save
from src.features import build_features_from_csv
from src.train import train_xgb, collect_processed_features
from src.predict_live import predict_for_tickers
from src.utils import ensure_dirs
from src.news_screener import summarize_news_for_prompt


st.set_page_config(page_title="Stock Market AI Dashboard", layout="wide")

st.title("📈 AI  Stock Market Analysis Dashboard")

st.markdown("""
Use this app to analyze and predict **Up/Down market movement** for Indian stocks (NSE: `.NS`).
It automatically fetches data, processes features, trains (if needed), and runs live predictions.
""")

# Step 1: Input tickers
tickers_input = st.text_input("Enter stock tickers (comma-separated, e.g. RELIANCE.NS, TCS.NS, HDFCBANK.NS):")
run_button = st.button("🚀 Run Live Prediction")

ensure_dirs()

if run_button and tickers_input.strip():
    tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]

    st.info(f"Fetching latest data for: {', '.join(tickers)} ...")
    fetch_and_save(tickers)

    st.success("✅ Data fetched successfully!")

    # Prepare features
    st.info("Preparing features...")
    for t in tickers:
        raw_path = f"data/raw/{t.replace('.', '_')}.csv"
        if os.path.exists(raw_path):
            df = build_features_from_csv(raw_path)
            df.to_csv(f"data/processed/{t.replace('.', '_')}_features.csv")
    st.success("✅ Feature preparation completed!")

    # Train or load model
    model_path = "models/xgb_model.json"
    if not os.path.exists(model_path):
        st.warning("No trained model found — training a new one...")
        df = collect_processed_features()
        exclude = ['label','next_close','future_ret','ticker']
        features = [c for c in df.columns if c not in exclude]
        train_xgb(df, features)
        st.success("✅ Model trained and saved.")
    else:
        st.success("✅ Using existing trained model.")

    # Run live prediction
    st.info("Running live prediction & portfolio optimization...")
    results = predict_for_tickers(tickers)
    st.success("✅ Prediction completed!")

    # Display results
    st.subheader("📊 Market Direction Prediction Results")
    df_res = pd.DataFrame(results)
    st.dataframe(df_res, width='stretch')   

    # Show portfolio weights only if multiple tickers
    if len(tickers) > 1:
      st.subheader("💼 Portfolio Weights")
      st.bar_chart(df_res.set_index("ticker")["weight"])

    # Display fundamentals summary
    st.subheader("📘 Generated Reports")
    for t in tickers:
        rpath = f"reports/{t.replace('.', '_')}_report.md"
        if os.path.exists(rpath):
            with open(rpath, "r", encoding="utf-8") as f:
                st.markdown(f"### {t}")
                st.markdown(f.read(), unsafe_allow_html=True)
                st.divider()

else:
    st.info("👈 Enter tickers above and click **Run Live Prediction** to begin.")
