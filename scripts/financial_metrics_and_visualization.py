import pandas as pd
import talib
import matplotlib.pyplot as plt
import pynance as pn
import glob
import os


def standardize_columns(df):
    df.columns = df.columns.str.lower()
    return df


def load_historical_data(file_paths):
    historical_data = {}
    for path in file_paths:
        
        filename = os.path.basename(path)
        print(f"Processing file: {filename}")  
        company_name = filename.split('_')[0].upper()  
        print(f"Extracted company name: {company_name}")  
        
       
        try:
            df = pd.read_csv(path)
            df = standardize_columns(df)  
            
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            else:
                raise ValueError("Column 'date' not found in file.")
            
            historical_data[company_name] = df
        except ValueError as e:
            print(f"Error loading {path}: {e}")
        except Exception as e:
            print(f"Unexpected error loading {path}: {e}")

    return historical_data


def calculate_financial_metrics(df):
    close = df['close']  
    
    
    df['returns'] = df['close'].pct_change().dropna()
    
    
    annualized_return = df['returns'].mean() * 252  
    
    
    volatility = df['returns'].std() * (252 ** 0.5)
    
    return annualized_return, volatility


def calculate_metrics_for_all(companies_data):
    metrics = {}
    for company, df in companies_data.items():
        print(f"Calculating metrics for {company}...")
        try:
            annualized_return, volatility = calculate_financial_metrics(df)
            metrics[company] = {
                'Annualized Return': annualized_return,
                'Volatility': volatility
            }
        except Exception as e:
            print(f"Error calculating metrics for {company}: {e}")
    return metrics


def calculate_indicators(df):
    
    close = df['Close'] if 'Close' in df.columns else df['close']

   
    df['SMA_50'] = talib.SMA(close, timeperiod=50)
    
    
    df['EMA_20'] = talib.EMA(close, timeperiod=20)
    
    
    df['RSI'] = talib.RSI(close, timeperiod=14)
    
    
    macd, macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_Signal'] = macd_signal
    df['MACD_Hist'] = macd_hist
    
    return df


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
    else:
        print("RSI column is missing")

   
    plt.figure(figsize=(14, 7))
    if 'MACD' in df.columns:
        plt.plot(df['date'], df['MACD'], label='MACD', color='blue')
    if 'MACD_Signal' in df.columns:
        plt.plot(df['date'], df['MACD_Signal'], label='Signal Line', color='orange')
    if 'MACD_Hist' in df.columns:
        plt.bar(df['date'], df['MACD_Hist'], label='MACD Histogram', color='grey', alpha=0.5)
    plt.title(f'{company_name} MACD Analysis')
    plt.xlabel('Date')
    plt.ylabel('MACD Value')
    plt.legend()
    plt.grid()
    plt.show()

