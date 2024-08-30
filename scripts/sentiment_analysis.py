import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import re



def analyze_display_and_plot_sentiments(df, n_samples=5):
   
    sid = SentimentIntensityAnalyzer()

    df['vader_sentiment'] = df['headline'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment_label'] = df['vader_sentiment'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

    df['textblob_sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_sentiment_label'] = df['textblob_sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))

    
    vader_counts = df['vader_sentiment_label'].value_counts()
    textblob_counts = df['textblob_sentiment_label'].value_counts()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=vader_counts.index, y=vader_counts.values, palette='viridis')
    plt.title('VADER Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()
    plt.figure(figsize=(10, 5))
    sns.barplot(x=textblob_counts.index, y=textblob_counts.values, palette='viridis')
    plt.title('TextBlob Sentiment Distribution')
    plt.xlabel('Sentiment')
    plt.ylabel('Count')
    plt.show()

    return df


def pie_chart(df):
   
    sid = SentimentIntensityAnalyzer()

    df['vader_sentiment'] = df['headline'].apply(lambda x: sid.polarity_scores(x)['compound'])
    df['vader_sentiment_label'] = df['vader_sentiment'].apply(lambda x: 'positive' if x >= 0.05 else ('negative' if x <= -0.05 else 'neutral'))

    df['textblob_sentiment'] = df['headline'].apply(lambda x: TextBlob(x).sentiment.polarity)
    df['textblob_sentiment_label'] = df['textblob_sentiment'].apply(lambda x: 'positive' if x > 0 else ('negative' if x < 0 else 'neutral'))


    vader_counts = df['vader_sentiment_label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(vader_counts, labels=vader_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(vader_counts)))
    plt.title('VADER Sentiment Distribution')
    plt.show()
    
    textblob_counts = df['textblob_sentiment_label'].value_counts()
    plt.figure(figsize=(8, 8))
    plt.pie(textblob_counts, labels=textblob_counts.index, autopct='%1.1f%%', colors=sns.color_palette('viridis', len(textblob_counts)))
    plt.title('TextBlob Sentiment Distribution')
    plt.show()

    return df





def inspect_dates(df):
    unique_dates = df['date'].unique()
    return unique_dates

def clean_and_parse_dates(date_str):
    if pd.isna(date_str):  
        return pd.NaT
    try:
        if isinstance(date_str, str):  
            clean_date_str = date_str.split('+')[0]  
            clean_date_str = clean_date_str.split('.')[0]  
            return pd.to_datetime(clean_date_str, format="%Y-%m-%d %H:%M:%S")
        else:
            return pd.NaT  
    except (ValueError, TypeError):
        return pd.NaT  

def analyze_publication_frequency(df, time_interval='D', spike_threshold=2):
    
    inspect_dates(df)
    df['date'] = df['date'].apply(clean_and_parse_dates)

    if df['date'].isnull().any():
        print("Warning: There are NaT values in the 'date' column after conversion. These will be dropped.")
        df = df.dropna(subset=['date'])

    
    freq_df = df.groupby(pd.Grouper(key='date', freq=time_interval)).size().reset_index(name='article_count')

    
    mean_count = freq_df['article_count'].mean()
    freq_df['spike'] = freq_df['article_count'] > mean_count * spike_threshold

    
    df_freq_spike = pd.merge(df, freq_df[['date', 'spike']], on='date', how='left')

    
    plt.figure(figsize=(14, 7))  
    sns.lineplot(data=freq_df, x='date', y='article_count', marker='o')
    
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
    
    plt.title(f'Publication Frequency Over Time ({time_interval} Interval)')
    plt.xlabel('Date')
    plt.ylabel('Number of Articles')
    plt.xticks(rotation=45)  
    plt.tight_layout()  
    plt.show()

    
    spike_dates = freq_df[freq_df['spike']]['date']
    spike_articles = df_freq_spike[df_freq_spike['date'].isin(spike_dates)]
    
    if not spike_articles.empty:
        print("\nHeadlines during spike periods:")
        print(spike_articles[['date', 'headline']].sort_values(by='date'))

    return freq_df


def perform_topic_modeling(df, num_topics=5, num_words=10):

    vectorizer = CountVectorizer(stop_words='english')
    dtm = vectorizer.fit_transform(df['headline'])

    
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(dtm)

    topics = []
    for index, topic in enumerate(lda.components_):
        top_words = [vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-num_words:]]
        topics.append(f"Topic {index+1}: " + " ".join(top_words))
        print(f"Topic {index+1}: " + ", ".join(top_words))

    return lda, topics






def is_email_address(s):
    """Check if a string is an email address."""
    return isinstance(s, str) and '@' in s

def extract_domain(email):
    """Extract domain from an email address."""
    if is_email_address(email):
        match = re.search(r'@([a-zA-Z0-9.-]+)', email)
        if match:
            return match.group(1)
    return None

def analyze_publisher_domains(df):
   

    df['domain'] = df['publisher'].apply(lambda x: extract_domain(x) if isinstance(x, str) else None)
    
  
    df = df.dropna(subset=['domain'])
    
  
    domain_counts = df['domain'].value_counts().reset_index()
    domain_counts.columns = ['Domain', 'Frequency']
    
    return domain_counts


















# Example 
#df = pd.read_csv('C:/Users/teeyob/Sentiment-Driven-Financial-Forecasting/data/raw_analyst_ratings.csv')
#freq_df = analyze_publication_time(df, time_interval='D', spike_threshold=2)
#df_with_sentiments = analyze_display_and_plot_sentiments(df, n_samples=5)