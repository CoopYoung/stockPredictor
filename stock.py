import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import datetime
import talib as ta
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from flask import Flask, jsonify, request, render_template, Response
import os

'''
Uploading onto the web server, the only thing that needs to be 
using Flask methods is the prediction function. The ML function 
does not need to be a Flask method.
'''

# Step 1: Download data from yfinance
def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    if data.empty:
        print(f"No data found for ticker {ticker}")
        return None

    if len(data) < 10:
        print(f"Insufficient data for ticker {ticker}. Minimum 10 data points required\n")
        return None
    return data
# Step 2: Feature Engineering (using previous day's data)

#Extra step: Get all stocks into a list and dataframe from yfinance

def get_all_stocks():
    csvfile = "nasdaq_screener.csv" #Saved locally 
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
        
        #if not isinstance(symbol, str) or '^' in symbol:
        #    spec_columns.at[index, 'Symbol'].replace("^", "")
        #else:
        #    spec_columns.at[index, 'Symbol'] = str(symbol).replace("^", "")
    #spec_columns.dropna(subset=['Symbol'], inplace=True)
    #print(dict(list(sdl.items())[:4]))
    print("spec_columns after: ", spec_columns, "\n")
    return spec_columns, sdl

# Enhanced feature creation with more technical indicators
def create_features(df):
    if len(df) < 5:
        print("Not enough data for rolling mean or standard deviation.")
        return df

    # Previous day prices
    df['Prev Close'] = df['Close'].shift(1)
    df['Prev High'] = df['High'].shift(1)
    df['Prev Low'] = df['Low'].shift(1)
    df['Prev Open'] = df['Open'].shift(1)
    
    # Moving averages
    df['SMA_10'] = df['Close'].rolling(window=10).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential moving average
    df['EMA_10'] = df['Close'].ewm(span=10, adjust=False).mean()
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    
    # Ensure 'Close' column is a 1-dimensional array
    close_prices = df['Close'].values.astype(float).reshape(-1)

    
    # RSI (Relative Strength Index)
    df['RSI'] = ta.RSI(close_prices, timeperiod=14)

    
    # MACD (Moving Average Convergence Divergence)
    df['MACD'], df['MACD_signal'], _ = ta.MACD(close_prices, fastperiod=12, slowperiod=26, signalperiod=9)
    
    # Bollinger Bands
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = ta.BBANDS(close_prices, timeperiod=20)

    # Rolling statistics
    df['Rolling Mean'] = df['Close'].rolling(window=5).mean()
    df['Rolling Std'] = df['Close'].rolling(window=5).std()

    # Drop rows with missing values after features creation
    df = df.dropna()

    return df


# Step 3: Prepare Data (Features and Labels)
def prepare_data(df):
    # Features: Using various lagged features
    features = df[['Prev Close', 'Prev High', 'Prev Low', 'Prev Open', 'Rolling Mean', 'Rolling Std']]
    
    # Labels: The next day's closing price
    labels = df['Close']
    
    # Split the data into training and testing datasets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    
    # Check if data is not empty
    if X_train.empty or X_test.empty:
        raise ValueError("Training or testing data is empty!")
    
    # Feature Scaling (Standardization)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler



def train_xgboost_model(X_train, y_train):
    model = xgb.XGBRegressor(objective="reg:squarederror", colsample_bytree=0.3, learning_rate=0.1,
                             max_depth=5, alpha=10, n_estimators=100)
    model.fit(X_train, y_train)
    return model

def tune_random_forest(X_train, y_train):
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Best parameters found: ", grid_search.best_params_)
    return grid_search.best_estimator_

# Step 4: Train the Model
def train_model(X_train, y_train):
    model = tune_random_forest(X_train, y_train)
    return model

# Step 5: Evaluate the Model

def evaluate_model(model, X_test, y_test, ticker):
    y_pred = model.predict(X_test)
    
    y_test_reshaped = y_test.values.astype(float).reshape(-1) #Need to become 1D array for evaluation
    mse = mean_squared_error(y_test_reshaped, y_pred)
    mae = mean_absolute_error(y_test_reshaped, y_pred)
    r2 = r2_score(y_test_reshaped, y_pred)
    
    print(f'MSE: {mse}')
    print(f'MAE: {mae}')
    print(f'R² Score: {r2}')
    
    # Plotting the actual vs predicted values
    
    plt.figure(figsize=(12, 6))
    plt.plot(y_test.index, y_test, label='Actual', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted', color='red')
    plt.title(f"{ticker} Price Prediction: Actual vs Predicted")
    plt.xlabel('Date')
    plt.ylabel('Price')
    ax = plt.gca()
    ax.yaxis.set_major_locator(plt.MaxNLocator(nbins=10))
    plt.legend()
    
    
     # Save the plot as an image file
    plot_path = os.path.join('static', 'plot.png')
    plt.savefig(plot_path)
    plt.close()
    
    return plot_path, mse, mae, r2
  
def time_series_cross_val(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    for train_index, test_index in tscv.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        model = train_model(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        print(f'MSE for fold: {mse}')

# Step 6: Predict Future Price (if user inputs a ticker and days)
def predict_future(model, scaler, ticker):

    
    # Fetch the most recent data

    end_date = datetime.datetime.today().strftime('%Y-%m-%d')
    start_date = (datetime.datetime.today() - datetime.timedelta(days=1095)).strftime('%Y-%m-%d')
    latest_data = fetch_data(ticker, start_date, end_date)
    
    print("Latest data before the loop : ", latest_data.values, "\n")


    last_row = latest_data.iloc[-1].values #np array of the last row of the dataframe
    print("Lastrow: ", last_row, "\n")
    '''
    # Check if data is not empty
    if latest_data is None or latest_data.empty:
        print(f"No data found for ticker {ticker}")
        return []
    
    print("Initial latest_data:")
    print(latest_data.tail(10))
    
    latest_data = create_features(latest_data)

    print("latest_data after feature engineering:")
    print(latest_data.tail(10))
    '''

    # Extract features
    latest_data['Prev Close'] = latest_data['Close'].shift(1)
    latest_data['Prev High'] = latest_data['High'].shift(1)
    latest_data['Prev Low'] = latest_data['Low'].shift(1)
    latest_data['Prev Open'] = latest_data['Open'].shift(1)

    if len(latest_data) < 5:
        return TypeError("Not enough data for rolling calculations")
    
    latest_data['Rolling Mean'] = latest_data['Close'].rolling(window=5).mean()
    latest_data['Rolling Std'] = latest_data['Close'].rolling(window=5).std()
    latest_data.dropna(ignore_index=True, inplace=True) #May need to change this?
        
    # Check for missing values before dropping them
    if latest_data.isnull().values.any():
        print("Missing values detected, filling with previous values.")
        latest_data.fillna(method='ffill', inplace=True)
        latest_data.dropna(inplace=True)
        
        # Extract features
    features = latest_data[['Prev Close', 'Prev High', 'Prev Low', 'Prev Open', 'Rolling Mean', 'Rolling Std']].tail(1)
        # Scale the features
    features_scaled = scaler.transform(features)
        
        # Predict the next day's closing price
    predicted_price = model.predict(features_scaled)[0]
        
    print(f"Predictions for the next 1 day: {predicted_price}")
    return predicted_price
    
