from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LinearRegression
import pandas as pd
import pickle
from datetime import datetime
import numpy as np


with open('C:/Users/teeyob/Sentiment-Driven-Financial-Forecasting/notebooks/sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)


app = FastAPI()


class SentimentInput(BaseModel):
    sentiment_score: float
    date: str  


@app.post("/predict/")
def predict_daily_return(input_data: SentimentInput):
   
    sentiment_score = input_data.sentiment_score
    date_str = input_data.date
    
  
    try:
        date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        return {"error": "Invalid date format. Please use YYYY-MM-DD."}
    

    date_ordinal = date.toordinal()
    

    features = np.array([[sentiment_score, date_ordinal]])
    

    predicted_return = model.predict(features)[0]
    
    return {"predicted_daily_return": predicted_return}

