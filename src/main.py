# src/main.py
import argparse
from fetch_historical import fetch_and_save
from features import build_features_from_csv
from train import train_xgb, collect_processed_features
import glob, os, pandas as pd
from predict_live import predict_for_tickers
from utils import ensure_dirs
import warnings
warnings.filterwarnings("ignore")


def prepare_features_for_all(raw_dir="data/raw", processed_dir="data/processed"):
    os.makedirs(processed_dir, exist_ok=True)
    files = glob.glob(os.path.join(raw_dir, "*.csv"))
    for f in files:
        base = os.path.basename(f).replace('.csv','')
        print("Processing:", base)
        df = build_features_from_csv(f)
        out = os.path.join(processed_dir, f"{base}_features.csv")
        df.to_csv(out)
    print("Processed feature CSVs saved to", processed_dir)

def train_pipeline():
    df = None
    try:
        df = collect_processed_features()
    except Exception as e:
        raise RuntimeError("Need processed features. Run prepare first.") from e
    exclude = ['label','next_close','future_ret','ticker']
    feature_cols = [c for c in df.columns if c not in exclude]
    train_xgb(df, feature_cols)

def cli():
    parser = argparse.ArgumentParser(description="Stock Up/Down project orchestration")
    parser.add_argument("--tickers", nargs='+', help="Tickers to analyze (e.g. RELIANCE.NS TCS.NS)", required=True)
    parser.add_argument("--fetch", action='store_true', help="Download historical data")
    parser.add_argument("--prepare", action='store_true', help="Prepare features (after fetch)")
    parser.add_argument("--train", action='store_true', help="Train model")
    parser.add_argument("--live", action='store_true', help="Run live prediction & reports")
    args = parser.parse_args()
    ensure_dirs()
    if args.fetch:
        fetch_and_save(args.tickers)
    if args.prepare:
        prepare_features_for_all()
    if args.train:
        train_pipeline()
    if args.live:
        out = predict_for_tickers(args.tickers)
        print("Live prediction & portfolio weights:")
        import json
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    cli()
