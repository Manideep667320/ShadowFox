import joblib
import pandas as pd

def load_model():
    return joblib.load("models/loan_approval_model.pkl")

def preprocess_input(data):
    df = pd.DataFrame([data])
    return df
