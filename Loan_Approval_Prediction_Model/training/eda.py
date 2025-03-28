import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    df = pd.read_csv("organized_loan_data.csv")

    return df

def perform_eda(df):
    print(df.info())
    print(df.describe())
    print(df.isna().sum() * 100 / len(df))

    # Visualizations
    sns.countplot(x=df['Gender'])
    plt.title('Gender Distribution')
    plt.show()

    sns.countplot(x=df['Married'])
    plt.title('Married Distribution')
    plt.show()

    sns.countplot(x=df['Dependents'])
    plt.title('Dependents Distribution')
    plt.show()

    sns.countplot(x=df['Self_Employed'])
    plt.title('Self Employed Distribution')
    plt.show()

    sns.countplot(x=df['Credit_History'])
    plt.title('Credit History Distribution')
    plt.show()

    # Additional plots for numerical features
    sns.histplot(df['LoanAmount'], kde=True)
    plt.title('Loan Amount Distribution')
    plt.show()

    sns.histplot(df['Loan_Amount_Term'], kde=True)
    plt.title('Loan Amount Term Distribution')
    plt.show()

    # Correlation matrix
    corr_matrix = df.select_dtypes(include=['float64', 'int64']).corr()

    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

if __name__ == "__main__":
    df = load_data("cleaned_loan_data.csv")
    perform_eda(df)
