import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
def prepare_data(df):
    #target: 1 if next day prices are high else 0
    df["Target"]=df["Close"].shift(-1)>df["Close"].astype(int)
    df.dropna(inplace=True)