# src/predict_live.py
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
import os

# ✅ absolute imports — required when running from project root
from src.features import create_technical_features
from src.portfolio_opt import optimize_portfolio
from src.reporting import generate_report


MODEL_PATH = "models/xgb_model.json"
FEATURES_PATH = "models/feature_list.pkl"

def load_model_and_features():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(FEATURES_PATH):
        raise RuntimeError("Model or feature list not found. Train model first.")
    model = xgb.Booster()
    model.load_model(MODEL_PATH)
    feature_list = joblib.load(FEATURES_PATH)
    return model, feature_list

def make_live_prediction_for_ticker(ticker):
    path = f"data/raw/{ticker.replace('.','_')}.csv"
    if not os.path.exists(path):
        raise RuntimeError(f"No raw data for {ticker}. Run fetch_historical first.")

    df = pd.read_csv(path, index_col=0, parse_dates=True)

    # ✅ Clean numeric columns — handle commas, strings, or None
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # ✅ Drop invalid rows
    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])

    df = create_technical_features(df)
    last = df.iloc[-1:]
    return df, last


def predict_for_tickers(tickers, vol_cap=0.02, max_weight=0.35):
    model, feat_list = load_model_and_features()
    probs = {}
    price_close = {}

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    for t in tickers:
        price_df, last_row = make_live_prediction_for_ticker(t)
        price_close[t] = price_df['Close']

        # ensure correct features, fill missing values
        X = last_row.reindex(columns=feat_list).fillna(0).values
        dmat = xgb.DMatrix(X)
        p_raw = model.predict(dmat)[0]
        probs[t] = float(sigmoid(p_raw))  # probability of Up

        # generate report
        rep = generate_report(t, price_df)
        print("Report created:", rep)

    tickers_list = list(probs.keys())
    pvec = np.array([probs[t] for t in tickers_list])
    expected_returns = (pvec - 0.5) * 0.02

    returns_df = pd.concat([price_close[t].pct_change().dropna().tail(240) for t in tickers_list], axis=1)
    returns_df.columns = tickers_list
    cov = returns_df.cov().fillna(0).values
    cov = cov + np.eye(cov.shape[0]) * 1e-8

    weights = optimize_portfolio(expected_returns, cov, vol_cap=vol_cap, max_weight=max_weight)

    out = []
    for i, t in enumerate(tickers_list):
        out.append({'ticker': t, 'prob_up': round(float(probs[t]), 4), 'weight': round(float(weights[i]), 4)})

    return out

if __name__ == "__main__":
    tickers = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
    res = predict_for_tickers(tickers)
    import json
    print(json.dumps(res, indent=2))
