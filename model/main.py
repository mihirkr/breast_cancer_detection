import pickle

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def get_clean_data():
    data = pd.read_csv("data/data.csv")

    # Preprocessing
    data = data.drop(columns=["Unnamed: 32", "id"], axis=1)
    data["diagnosis"] = data["diagnosis"].map({"M": 1, "B": 0})
    return data


def create_model(data):
    X = data.drop(columns=["diagnosis"], axis=1)
    y = data["diagnosis"]

    # Scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    # train
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # test
    y_pred = model.predict(X_test)
    print("Accuracy of the model: ", accuracy_score(y_test, y_pred))
    print("Classification Report: \n", classification_report(y_test, y_pred))

    return model, scaler


def main():
    data = get_clean_data()
    # Create the model
    model, scaler = create_model(data)

    with open("model/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)


if __name__ == "__main__":
    main()
