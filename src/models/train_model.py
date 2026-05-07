import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier


# Prepare target column
def prepare_data(df):

    # 1 if next day price increases else 0
    df["Target"] = (
        df["Close"].shift(-1) > df["Close"]
    ).astype(int)

    df.dropna(inplace=True)

    return df


# Train model function
def train_model(ticker):

    ticker = ticker.lower()

    # Load features dataset
    df = pd.read_csv(
        f"data/{ticker}_features.csv"
    )

    # Prepare target
    df = prepare_data(df)

    # Input features
    X = df[
        [
            "moving_avg5",
            "moving_avg10",
            "return",
            "volatility",
            "lag_1",
            "lag_2",
            "lag_3"
        ]
    ]

    # Output target
    y = df["Target"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        shuffle=False
    )

    # Model save path
    model_path = f"src/models/{ticker}_model.pkl"

    # Load existing model if available
    if os.path.exists(model_path):

        print("Loading existing model...")

        model = joblib.load(model_path)

    else:

        print("Training new XGBoost model...")

        model = XGBClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )

        model.fit(X_train, y_train)

        # Save model
        joblib.dump(model, model_path)

    # Predictions
    y_pred = model.predict(X_test)

    # Accuracy
    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    print("Model Accuracy:", accuracy)

    return model, accuracy


if __name__ == "__main__":

    ticker = input("Enter ticker: ")

    train_model(ticker)