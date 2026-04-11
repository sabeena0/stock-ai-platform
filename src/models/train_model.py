import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os
def prepare_data(df):
    #target: 1 if next day prices are high else 0
    df["Target"]=df["Close"].shift(-1)>df["Close"].astype(int)
    df.dropna(inplace=True)
    return df
if __name__ == "__main__":
    ticker=input("Enter ticker: ").lower()
    df=pd.read_csv(f"data/{ticker}_features.csv")
    df=prepare_data(df)
    x=df[["moving_avg5", "moving_avg10", "return", "volatility", "lag_1", "lag_2", "lag_3"]]
    y=df["Target"]
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=False)
    model_path = f"src/models/{ticker}_model.pkl"
    if os.path.exists(model_path):
        print("Loading existing model...")
        model=joblib.load(model_path)
    else:
        print("Training new model")
        model=RandomForestClassifier()
        model.fit(x_train,y_train)
        joblib.dump(model, model_path)
    y_pred=model.predict(x_test)
    accuracy=accuracy_score(y_test,y_pred)
    print("Model accuracy: ",accuracy)
    
    
