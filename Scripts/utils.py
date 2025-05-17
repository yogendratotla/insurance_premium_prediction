# utils.py
import joblib

def save_model(model, filepath):
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")

def load_model(filepath):
    return joblib.load(filepath)
