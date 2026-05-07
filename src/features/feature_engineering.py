import pandas as pd


# Clean columns
def clean_data(df):

    if isinstance(df.columns, pd.MultiIndex):

        df.columns = df.columns.get_level_values(0)

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    return df


# Create features
def create_features(ticker):

    ticker = ticker.lower()

    # Load stock CSV
    df = pd.read_csv(
        f"data/{ticker}_stock_data.csv"
    )

    # Clean dataset
    df = clean_data(df)

    # Feature Engineering
    df["moving_avg5"] = df["Close"].rolling(5).mean()

    df["moving_avg10"] = df["Close"].rolling(10).mean()

    df["return"] = df["Close"].pct_change()

    df["volatility"] = df["return"].rolling(10).std()

    df["lag_1"] = df["Close"].shift(1)

    df["lag_2"] = df["Close"].shift(2)

    df["lag_3"] = df["Close"].shift(3)

    # Remove null rows
    df.dropna(inplace=True)

    # Save features CSV
    df.to_csv(
        f"data/{ticker}_features.csv",
        index=False
    )

    return df


if __name__ == "__main__":

    ticker = input("Enter ticker: ")

    df = create_features(ticker)

    print(df.head())