# src/train.py
import pandas as pd
import numpy as np
import glob, os, joblib
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import roc_auc_score
import xgboost as xgb

def collect_processed_features(processed_dir="data/processed"):
    files = glob.glob(os.path.join(processed_dir, "*_features.csv"))
    if not files:
        raise RuntimeError("No processed feature files found in data/processed.")
    dfs = []
    for f in files:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        df['ticker'] = os.path.basename(f).split('_')[0]
        dfs.append(df)
    full = pd.concat(dfs, ignore_index=False)
    return full

def train_xgb(df, feature_cols, label_col='label', model_out='models/xgb_model.json'):
    X = df[feature_cols].values
    y = df[label_col].values
    tscv = TimeSeriesSplit(n_splits=5)
    params = {'objective':'binary:logistic', 'eval_metric':'auc', 'verbosity':0}
    aucs = []
    for train_idx, test_idx in tscv.split(X):
        dtrain = xgb.DMatrix(X[train_idx], label=y[train_idx])
        dtest = xgb.DMatrix(X[test_idx], label=y[test_idx])
        bst = xgb.train(params, dtrain, num_boost_round=150)
        ypred = bst.predict(dtest)
        auc = roc_auc_score(y[test_idx], ypred)
        aucs.append(auc)
        print("Fold AUC:", round(auc,4))
    print("Mean CV AUC:", round(np.mean(aucs),4))
    dfull = xgb.DMatrix(X, label=y)
    final = xgb.train(params, dfull, num_boost_round=200)
    os.makedirs(os.path.dirname(model_out), exist_ok=True)
    final.save_model(model_out)
    joblib.dump(feature_cols, os.path.join(os.path.dirname(model_out),'feature_list.pkl'))
    print("Model saved to", model_out)
    return final

if __name__ == "__main__":
    df = collect_processed_features()
    exclude = ['label','next_close','future_ret','ticker']
    feature_cols = [c for c in df.columns if c not in exclude]
    train_xgb(df, feature_cols)
