import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import talib
import pynance as pn
import os
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.dates as mdates
import plotly.express as px

# Helper functions (you can add your existing functions here)

# Analyze headline length
def analyze_headline_length(df):
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe()

# Count articles per publisher
def count_articles_per_publisher(df):
    return df['publisher'].value_counts()

# Analyze publication dates
def analyze_publication_dates(df):
    df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    df.dropna(subset=['date'], inplace=True)
    df.set_index('date', inplace=True)
    daily_counts = df.resample('D').size()
    st.pyplot(fig=plt.figure(figsize=(10, 6)))
    plt.plot(daily_counts, label='Daily Publication Frequency')
    plt.title('Trend of Publications Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Publications')
    plt.grid(True)
    plt.legend()
    plt.show()
    return daily_counts

# Standardize column names
def standardize_columns(df):
    df.columns = df.columns.str.lower()
    return df

# Load historical stock data
def load_historical_data(file_paths):
    historical_data = {}
    for path in file_paths:
        filename = os.path.basename(path)
        company_name = filename.split('_')[0].upper()
        df = pd.read_csv(path)
        df = standardize_columns(df)
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        else:
            raise ValueError("Column 'date' not found in file.")
        historical_data[company_name] = df
    return historical_data

# Plot financial indicators (SMA, EMA, etc.)
def plot_indicators(df, company_name):
    plt.figure(figsize=(14, 7))
    plt.plot(df['date'], df['close'], label='Close Price', color='black')
    if 'SMA_50' in df.columns:
        plt.plot(df['date'], df['SMA_50'], label='50-day SMA', linestyle='--')
    if 'EMA_20' in df.columns:
        plt.plot(df['date'], df['EMA_20'], label='20-day EMA', linestyle='--')
    plt.title(f'{company_name} Stock Price with Moving Averages')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot RSI
    plt.figure(figsize=(14, 5))
    if 'RSI' in df.columns:
        plt.plot(df['date'], df['RSI'], label='RSI', color='purple')
        plt.axhline(70, color='red', linestyle='--', label='Overbought')
        plt.axhline(30, color='green', linestyle='--', label='Oversold')
        plt.title(f'{company_name} Relative Strength Index (RSI)')
        plt.xlabel('Date')
        plt.ylabel('RSI Value')
        plt.legend()
        plt.grid()
        plt.show()

# Sentiment analysis (VADER, TextBlob)
def analyze_display_and_plot_sentiments(df):
    sid = SentimentIntensityAnalyzer()
    df['vader_sentiment'] = df['headline'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment_label'] = df['vader_sentiment'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))
    df['textblob_sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_sentiment_label'] = df['textblob_sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))
    vader_counts = df['vader_sentiment_label'].value_counts()
    textblob_counts = df['textblob_sentiment_label'].value_counts()

    # Plot VADER sentiment distribution
    st.subheader("VADER Sentiment Distribution")
    sns.barplot(x=vader_counts.index, y=vader_counts.values, palette='viridis')
    plt.title('VADER Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

    # Plot TextBlob sentiment distribution
    st.subheader("TextBlob Sentiment Distribution")
    sns.barplot(x=textblob_counts.index, y=textblob_counts.values, palette='viridis')
    plt.title('TextBlob Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    st.pyplot(plt)

    return df


# Streamlit UI setup
st.title("Financial Dashboard")

# Tabs for navigation
tab_selection = st.sidebar.radio("Select Tab", ['Analysis', 'Financial Indicators', 'Sentiment Correlation'])

if tab_selection == 'Analysis':
    st.header("Headline Analysis")
    df = pd.read_csv('../data/raw_analyst_ratings.csv')
    st.subheader("Headline Length Analysis")
    st.write(analyze_headline_length(df))
    st.subheader("Articles per Publisher")
    st.write(count_articles_per_publisher(df))
    st.subheader("Publication Dates Trend")
    daily_counts = analyze_publication_dates(df)
    st.write(daily_counts)

elif tab_selection == 'Financial Indicators':
    st.header("Stock Financial Indicators")
    file_paths = glob.glob('C:/Users/teeyob/Sentiment-Driven-Financial-Forecasting/data/yfinance_data/*.csv')
    historical_data = load_historical_data(file_paths)
    company = st.selectbox("Select Company", list(historical_data.keys()))
    df = historical_data[company]
    st.subheader(f"{company} Financial Indicators")
    st.write(calculate_financial_metrics(df))
    plot_indicators(df, company)

elif tab_selection == 'Sentiment Correlation':
    st.header("Sentiment Correlation with Stock Performance")
    news_df = pd.read_csv('../data/raw_analyst_ratings.csv')
    stock_df = pd.read_csv('C:/Users/teeyob/Sentiment-Driven-Financial-Forecasting/data/yfinance_data/AAPL_Historical_data.csv')
    sentiment_df = analyze_display_and_plot_sentiments(news_df)
    st.write(sentiment_df)

# Add custom styles for better visuals
st.markdown("""
    <style>
    .css-18e3th9 {
        padding-top: 20px;
    }
    </style>
""", unsafe_allow_html=True)