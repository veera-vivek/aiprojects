# src/features.py
import pandas as pd
from ta import momentum, trend, volatility

def create_technical_features(df):
    df = df.copy()
    df['return_1'] = df['Close'].pct_change(1)
    df['return_3'] = df['Close'].pct_change(3)
    df['ma20'] = df['Close'].rolling(20).mean()
    df['ma50'] = df['Close'].rolling(50).mean()
    df['ma200'] = df['Close'].rolling(200).mean()
    df['ma20_50'] = df['ma20'] - df['ma50']
    df['rsi14'] = momentum.rsi(df['Close'], window=14)
    df['macd_diff'] = trend.macd_diff(df['Close'])
    df['atr14'] = volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    df['vol_20'] = df['Volume'].rolling(20).mean()
    df['vol_ratio'] = df['Volume'] / (df['vol_20'] + 1e-9)
    df = df.dropna()
    return df

def create_label(df, threshold=0.0):
    df = df.copy()
    df['next_close'] = df['Close'].shift(-1)
    df['future_ret'] = (df['next_close'] - df['Close']) / df['Close']
    df['label'] = (df['future_ret'] > threshold).astype(int)
    df = df.dropna(subset=['label'])
    return df

def build_features_from_csv(path, label_threshold=0.0):
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    
    # Convert all numeric columns properly (handle commas and strings)
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with too many missing or non-numeric values
    df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
    
    df = create_technical_features(df)
    df = create_label(df, threshold=label_threshold)
    return df

