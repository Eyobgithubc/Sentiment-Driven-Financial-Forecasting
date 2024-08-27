from transformers import pipeline
import pandas as pd

# Load the sentiment-analysis pipeline
sentiment_pipeline = pipeline('sentiment-analysis')

def analyze_sentiment_advanced(headline):
    """Analyze sentiment of a headline using a pre-trained model."""
    result = sentiment_pipeline(headline)[0]
    sentiment = result['label']
    return sentiment.lower()

def analyze_headlines_sentiment_advanced(df):
    """Analyze sentiment for each headline in the DataFrame using an advanced model."""
    df['sentiment'] = df['headline'].apply(analyze_sentiment_advanced)
    return df
