# Enhanced Flask Web App (app.py)
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load("models/loan_approval_model.pkl")

# Load training feature names to ensure order consistency
training_data = pd.read_csv("dataset/cleaned_loan_data.csv")
feature_columns = training_data.drop(columns=["Loan_Status"]).columns  # Exclude target variable

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.form.to_dict()
    
    # Convert input into DataFrame and ensure correct feature order
    df = pd.DataFrame([data])
    
    # Convert numerical columns to appropriate data types
    numerical_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History']
    for col in numerical_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # One-hot encode categorical variables to match training features
    df = pd.get_dummies(df)
    df = df.reindex(columns=feature_columns, fill_value=0)  # Ensure columns match training set
    
    # Check if any missing values exist in final input
    if df.isnull().values.any():
        return render_template("result.html", result="Error: Invalid Input Data")
    
    # Predict loan status
    prediction = model.predict(df)[0]
    result = "Approved" if prediction == 1 else "Denied"
    
    return render_template("result.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
