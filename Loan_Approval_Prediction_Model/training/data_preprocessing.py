import pandas as pd

def fill_missing_values(df):
    df['Gender'].fillna(df['Gender'].mode()[0], inplace=True)
    df['Married'].fillna(df['Married'].mode()[0], inplace=True)
    df['Dependents'].fillna(df['Dependents'].mode()[0], inplace=True)
    df['Self_Employed'].fillna(df['Self_Employed'].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)
    df['Credit_History'].fillna(df['Credit_History'].mode()[0], inplace=True)

def encode_categorical_variables(df):
    df.replace({
        "Loan_Status": {'N': 0, 'Y': 1},
        "Gender": {'Male': 0, 'Female': 1},
        "Education": {'Not Graduate': 0, 'Graduate': 1},
        "Married": {'No': 0, 'Yes': 1},
        "Self_Employed": {'No': 0, 'Yes': 1}
    }, inplace=True)

def preprocess_data(df):
    fill_missing_values(df)
    encode_categorical_variables(df)
    return df

def save_cleaned_data(df, file_path):
    df.to_csv(file_path, index=False)

if __name__ == "__main__":
    df = pd.read_csv("dataset/loan_prediction.csv")
    df = preprocess_data(df)
    save_cleaned_data(df, 'cleaned_loan_data.csv')
    print("Preprocessed data saved as 'cleaned_loan_data.csv'")
