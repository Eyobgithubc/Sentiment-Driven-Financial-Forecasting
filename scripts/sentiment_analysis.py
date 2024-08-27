from nltk.sentiment.vader import SentimentIntensityAnalyzer
import pandas as pd

def analyze_sentiment_vader(df):
    sia = SentimentIntensityAnalyzer()
    
    def classify_sentiment_vader(text):
        score = sia.polarity_scores(text)
        if score['compound'] > 0:
            return 'Positive'
        elif score['compound'] < 0:
            return 'Negative'
        else:
            return 'Neutral'
    
    df['sentiment'] = df['headline'].apply(classify_sentiment_vader)
    sentiment_counts = df['sentiment'].value_counts().reset_index()
    sentiment_counts.columns = ['Sentiment', 'Count']
    return sentiment_counts

# Example usage:
df = pd.read_csv('../data/raw_analyst_ratings.csv')
sentiment_count_df = analyze_sentiment_vader(df.sample(frac=0.1, random_state=1))  # Use 10% of the data
print(sentiment_count_df)
