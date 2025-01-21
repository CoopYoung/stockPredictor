import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from textblob import TextBlob
import yfinance as yf

def get_all_stocks():
    csvfile = "assets/nasdaq_screener.csv" #Saved locally 
    stock_data = pd.read_csv(csvfile) 
    stock_data_list = list(stock_data["Symbol"]) #Only the tickers 

    spec_columns = stock_data[["Symbol", "Market Cap", "Volume"]] #tickers, market cap, and volume
   
    volume_data_list = list(stock_data["Volume"])
    market_cap_list = list(stock_data["Market Cap"])
    sdl = {}

    i = 0
    for stock in stock_data_list:
        if isinstance(stock, str) and '^' not in stock:
            sdl[stock] = {"Market Cap" : market_cap_list[i], "Volume" : volume_data_list[i]}
        i += 1

    for index, row in spec_columns.iterrows():
        symbol = row["Symbol"]
        spec_columns.at[index, 'Symbol'] = str(symbol).replace("^", "")
        
    return spec_columns, sdl

def calculate_technical_indicators(df):
    try:
        # Moving averages
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['EMA_20'] = df['Close'].ewm(span=20).mean()
        
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # MACD
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = exp1 - exp2
        df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
        
        # Bollinger Bands
        middle_band = df['Close'].rolling(window=20).mean()
        std_dev = df['Close'].rolling(window=20).std()
        df['BB_upper'] = middle_band + (std_dev * 2)
        df['BB_lower'] = middle_band - (std_dev * 2)
        df['BB_middle'] = middle_band
        
        # Fill NaN values
        df = df.ffill()
        
        # Ensure all required columns exist
        required_columns = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        return df
    
    except Exception as e:
        print(f"Error in calculate_technical_indicators: {str(e)}")
        raise


def get_news_sentiment(ticker):
    try:
        stock = yf.Ticker(ticker)
        news = stock.news
        
        sentiment_scores = []
        for article in news[:10]:
            blob = TextBlob(article['title'])
            sentiment_scores.append(blob.sentiment.polarity)
        
        avg_sentiment = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0
        return avg_sentiment, news[:5]
    except:
        return 0, []

def calculate_risk_metrics(df):
    returns = df['Close'].pct_change()
    
    metrics = {
        'VaR_95': np.percentile(returns.dropna(), 5),
        'Volatility': returns.std() * np.sqrt(252),
        'Sharpe_Ratio': np.sqrt(252) * (returns.mean() - 0.02/252) / returns.std(),
        'Max_Drawdown': (1 + returns).cumprod().div((1 + returns).cumprod().expanding().max()).min() - 1
    }
    
    return metrics

# Function to calculate optimal window size based on market volatility
def calculate_optimal_window(df):
    # Calculate volatility
    volatility = df['Close'].pct_change().std()
    
    if volatility > 0.03:  # High volatility
        return 30  # Shorter window for volatile markets
    elif volatility > 0.02:  # Medium volatility
        return 60
    else:  # Low volatility
        return 90