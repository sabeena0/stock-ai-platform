from fastapi import FastAPI
import pandas as pd
import joblib
import os
from src.assistant.assistant import generate_explanation
app=FastAPI()
@app.get("/")
def home():
    return {"message":"Stock Prediction is running"}
@app.get("/predict/{ticker}")
def predict_stock(ticker:str):
    ticker=ticker.lower()
    file_path=f"data/{ticker}_features.csv"
    if not os.path.exists(file_path):
        return {"error":"Data not found. run data loader first"}
    df=pd.read_csv(file_path)
    x=df[["moving_avg5", "moving_avg10", "return", "volatility", "lag_1", "lag_2", "lag_3"]]
    model_path=f"src/models/{ticker}_model.pkl"
    if not os.path.exists(model_path):
        return {"error":"Model not found. run train model first"}
    model=joblib.load(model_path)
    # Latest data
    latest = x.iloc[-1].values.reshape(1, -1)

    prediction = model.predict(latest)[0]

    explanation = generate_explanation(ticker, prediction)

    result = "UP 📈" if prediction == 1 else "DOWN 📉"

    return {
        "ticker": ticker.upper(),
        "prediction": result,
        "explanation": explanation
    }
