import pandas as pd
import yfinance as yf


def download_stock_data(ticker, start="2018-01-01"):

    ticker = ticker.lower()

    print(f"Downloading dataset for {ticker.upper()}....")

    data = yf.download(ticker, start=start)

    # Fix multi-level columns issue
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    data.reset_index(inplace=True)

    # Save CSV automatically
    data.to_csv(f"data/{ticker}_stock_data.csv", index=False)

    return data


if __name__ == "__main__":

    ticker = input("Input stock ticker: ")

    df = download_stock_data(ticker)

    print(df.head())