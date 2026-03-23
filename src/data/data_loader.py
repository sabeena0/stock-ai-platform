import pandas as pd
import yfinance as yf
def download_stock_data(ticker, start="2018-01-01"):
    print(f"Downloading dataset for {ticker}....")
    data=yf.download(ticker,start=start)
    data.reset_index(inplace=True)
    return data
if __name__=="__main__":
    ticker=input("input stock ticker").upper()
    df=download_stock_data(ticker)
    print(df.head())
    df.to_csv(f"data/{ticker.lower()}_stock_data.csv",index=False)