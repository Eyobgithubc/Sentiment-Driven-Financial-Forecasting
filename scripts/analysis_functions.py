import pandas as pd 
import matplotlib.pyplot as plt
def analyze_headline_length(df):
    df['headline_length'] = df['headline'].apply(len)
    return df['headline_length'].describe()


def count_articles_per_publisher(df):
    return df['publisher'].value_counts()










import pandas as pd
import matplotlib.pyplot as plt

def analyze_publication_dates(df):
    if 'date' in df.columns:
        # Convert 'date' to datetime, handling multiple formats
        df['date'] = pd.to_datetime(df['date'], errors='coerce', utc=True)
    else:
        raise KeyError("The DataFrame does not contain a 'date' column.")

    # Drop rows where 'date' could not be parsed
    df.dropna(subset=['date'], inplace=True)

    # Set 'date' as index
    df.set_index('date', inplace=True)

    # Resample to get daily counts
    daily_counts = df.resample('D').size()

    plt.figure(figsize=(10, 6))
    plt.plot(daily_counts, label='Daily Publication Frequency')
    plt.title('Trend of Publications Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Publications')
    plt.grid(True)
    plt.legend()
    plt.show()

    return daily_counts













