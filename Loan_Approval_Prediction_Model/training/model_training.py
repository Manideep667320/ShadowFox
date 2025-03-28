import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt
import joblib

def train_test_split_and_features(df):
    y = df["Loan_Status"]
    x = df.drop(['Loan_Status', 'Loan_ID'], axis=1)
    x = pd.get_dummies(data=x, columns=["Property_Area", "Dependents"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
    features = list(x.columns)
    return x_train, x_test, y_train, y_test, features

def fit_and_evaluate_model(x_train, x_test, y_train, y_test):
    random_forest = RandomForestClassifier(random_state=0, max_depth=5, min_samples_split=0.01, max_features=0.8, max_samples=0.8)
    model = random_forest.fit(x_train, y_train)
    random_forest_predict = random_forest.predict(x_test)
    random_forest_conf_matrix = confusion_matrix(y_test, random_forest_predict)
    random_forest_acc_score = accuracy_score(y_test, random_forest_predict)
    print("Confusion Matrix:")
    print(random_forest_conf_matrix)
    print("\nAccuracy of Random Forest:", random_forest_acc_score * 100, '\n')
    print(classification_report(y_test, random_forest_predict))

    # Save the model
    joblib.dump(model, 'loan_approval_model.pkl')

    return model

def plot_feature_importances(model, features):
    importances = pd.DataFrame(model.feature_importances_, columns=['importance'])
    importances['feature'] = features
    importances.sort_values(by='importance', ascending=True, inplace=True)
    plt.barh(importances['feature'], importances['importance'])
    plt.show()

if __name__ == "__main__":
    df = pd.read_csv("dataset/loan_prediction.csv")
    from data_preprocessing import preprocess_data
    df = preprocess_data(df)
    x_train, x_test, y_train, y_test, features = train_test_split_and_features(df)
    model = fit_and_evaluate_model(x_train, x_test, y_train, y_test)
    plot_feature_importances(model, features)
