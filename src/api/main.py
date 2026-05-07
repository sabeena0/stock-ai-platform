from fastapi import FastAPI
import pandas as pd
import os
import joblib

# Import pipeline functions
from src.data.data_loader import download_stock_data
from src.features.feature_engineering import create_features
from src.models.train_model import train_model
from src.assistant.assistant import generate_explanation

app = FastAPI()


@app.get("/")
def home():

    return {
        "message": "AI Stock Prediction API Running"
    }


@app.get("/predict/{ticker}")
def predict_stock(ticker: str):

    ticker = ticker.lower()

    try:

        # -----------------------------
        # Step 1: Download data if missing
        # -----------------------------
        stock_path = f"data/{ticker}_stock_data.csv"

        if not os.path.exists(stock_path):

            print("Downloading stock data...")

            download_stock_data(ticker)

        # -----------------------------
        # Step 2: Create features if missing
        # -----------------------------
        feature_path = f"data/{ticker}_features.csv"

        if not os.path.exists(feature_path):

            print("Creating features...")

            create_features(ticker)

        # -----------------------------
        # Step 3: Train model if missing
        # -----------------------------
        model_path = f"src/models/{ticker}_model.pkl"

        if not os.path.exists(model_path):

            print("Training model...")

            train_model(ticker)

        # -----------------------------
        # Step 4: Load feature dataset
        # -----------------------------
        df = pd.read_csv(feature_path)

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

        # -----------------------------
        # Step 5: Load model
        # -----------------------------
        model = joblib.load(model_path)

        # Latest row prediction
        latest = X.iloc[-1].values.reshape(1, -1)

        prediction = model.predict(latest)[0]

        result = (
            "UP 📈"
            if prediction == 1
            else "DOWN 📉"
        )

        # -----------------------------
        # Step 6: AI explanation
        # -----------------------------
        explanation = generate_explanation(
            ticker,
            prediction
        )

        return {

            "ticker": ticker.upper(),

            "prediction": result,

            "explanation": explanation
        }

    except Exception as e:

        return {
            "error": str(e)
        }