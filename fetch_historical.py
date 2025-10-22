# src/fetch_historical.py
import yfinance as yf
import os
from datetime import datetime

OUT_DIR = "data/raw"

def download_ticker(ticker, start='2017-01-01', end=None):
    if end is None:
        end = datetime.today().strftime('%Y-%m-%d')
    df = yf.download(ticker, start=start, end=end, progress=False)
    if df.empty:
        raise RuntimeError(f"No data for {ticker}")
    df.index = df.index.tz_localize(None)
    return df

def save_csv(ticker, df, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    fname = os.path.join(out_dir, f"{ticker.replace('.','_')}.csv")
    df.to_csv(fname)
    return fname

def fetch_and_save(tickers, start='2017-01-01', end=None):
    results = {}
    for t in tickers:
        print("Downloading:", t)
        df = download_ticker(t, start, end)
        p = save_csv(t, df)
        results[t] = p
    return results

if __name__ == "__main__":
    ts = ["RELIANCE.NS","TCS.NS"]
    fetch_and_save(ts)
