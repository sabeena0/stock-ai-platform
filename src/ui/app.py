import streamlit as st
import requests
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Stock Prediction Platform",
    layout="wide"
)

st.title("📊 AI Stock Prediction Platform")

st.write("Predict stock movement using Machine Learning + AI")

# User Input
ticker = st.text_input(
    "Enter Stock Ticker",
    value="tsla"
).lower()

# Predict Button
if st.button("Predict"):

    try:

        # API URL
        url = f"http://127.0.0.1:8000/predict/{ticker}"

        # API Request
        response = requests.get(url)

        # Convert response to JSON
        data = response.json()

        # Check if backend returned error
        if "error" in data:

            st.error(data["error"])

        else:

            st.subheader(f"Prediction for {data['ticker']}")

            if "UP" in data["prediction"]:
                st.success(data["prediction"])
            else:
                st.error(data["prediction"])

            st.subheader("🤖 AI Explanation")

            st.info(data["explanation"])

            # Load stock CSV
            df = pd.read_csv(f"data/{ticker}_stock_data.csv")

            # Plot chart
            fig = px.line(
                df,
                x="Date",
                y="Close",
                title=f"{ticker.upper()} Stock Price"
            )

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:

        st.error(f"Error: {str(e)}")