# src/utils.py
import os

def ensure_dirs():
    for d in ["data/raw", "data/processed", "models", "reports"]:
        os.makedirs(d, exist_ok=True)
