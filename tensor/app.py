import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
from datetime import datetime, timedelta
import yfinance as yf
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import logging 
import sys
from flask import Flask

from models import create_advanced_model
from analysis import calculate_technical_indicators, get_news_sentiment, calculate_risk_metrics, get_all_stocks



#INITIALIZE LOGGING FOR DEBUGGING
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
server = Flask(__name__)
# Initialize Dash app
app = dash.Dash(__name__, server=server)

# Available tickers

spec_columns, sdl = get_all_stocks()

def get_scalar_value(series_or_value):
    """Convert pandas Series or numpy array to scalar value"""
    try:
        if isinstance(series_or_value, pd.Series):
            return float(series_or_value.iloc[0])
        elif isinstance(series_or_value, np.ndarray):
            return float(series_or_value.item())
        else:
            return float(series_or_value)
    except Exception as e:
        logger.error(f"Error converting value: {str(e)}")
        return 0.0

def calculate_metrics(df, next_day_price, train_loss, val_loss):
    """Calculate all metrics with proper error handling"""
    try:
        # Get scalar values
        current_price = get_scalar_value(df['Close'].iloc[-1])
        initial_price = get_scalar_value(df['Close'].iloc[0])
        next_day_price = get_scalar_value(next_day_price)
        
        # Calculate metrics
        price_change = ((current_price / initial_price) - 1) * 100
        daily_returns = df['Close'].pct_change()
        volatility = get_scalar_value(daily_returns.std() * np.sqrt(252))
        avg_volume = int(get_scalar_value(df['Volume'].mean()))
        pred_change = ((next_day_price/current_price) - 1) * 100
        
        # Determine sentiment
        sentiment = "Bullish" if next_day_price > current_price else "Bearish"
        
        return {
            'current_price': current_price,
            'price_change': price_change,
            'next_day_price': next_day_price,
            'volatility': volatility,
            'avg_volume': avg_volume,
            'pred_change': pred_change,
            'sentiment': sentiment,
            'train_loss': get_scalar_value(train_loss),
            'val_loss': get_scalar_value(val_loss)
        }
    except Exception as e:
        logger.error(f"Error in calculate_metrics: {str(e)}")
        return None

# Layout
app.layout = html.Div([
    # Header
    html.Div([
        html.Img(src='/assets/logo.png', className='logo'),
        html.H1("Advanced Stock Prediction Dashboard")
    ], className='header'),
    
    # Control Panel
    html.Div([
        html.P("Select a stock from NASDAQ listing"),
        html.Div([
            html.Label("Select Stock:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} 
                        for ticker in spec_columns['Symbol']],
                value='AAPL'
            )
        ], style={'width': '30%'}),
        
        html.Div([
            html.Label("Analysis Timeframe:"),
            dcc.Dropdown(
                id='timeframe-dropdown',
                options=[
                    {'label': '5 Days', 'value': '5d'},
                    {'label': '1 Month', 'value': '1mo'},
                    {'label': '3 Months', 'value': '3mo'},
                    {'label': '6 Months', 'value': '6mo'},
                    {'label': '1 Year', 'value': '1y'}
                ],
                value='3mo'
            )
        ], style={'width': '30%'}),
        
        html.Button(
            'Analyze',
            id='analyze-button',
            className='button-primary'
        )
    ], className='control-panel'),
    
    # Main Content
    html.Div([
        # Price Chart
        html.Div([
            html.H3("Price Prediction"),
            dcc.Graph(id='price-chart')
        ], className='chart-container'),
        
        # Technical Indicators
        html.Div([
            html.H3("Technical Indicators"),
            dcc.Graph(id='technical-chart')
        ], className='chart-container'),
        
        # Metrics Panels
        html.Div([
            # Performance Metrics
            html.Div([
                html.H3("Performance Metrics"),
                html.Div(id='performance-metrics')
            ], className='metric-card'),
            
            # Risk Metrics
            html.Div([
                html.H3("Risk Metrics"),
                html.Div(id='risk-metrics')
            ], className='metric-card'),
            
            # Sentiment Analysis
            html.Div([
                html.H3("Market Sentiment"),
                html.Div(id='sentiment-indicator')
            ], className='metric-card')
        ], className='metrics-panel'),
        
        # News Feed
        html.Div([
            html.H3("Recent News"),
            html.Div(id='news-feed')
        ], className='news-container')
    ], className='main-container')
])

@app.callback(
    [Output('price-chart', 'figure'),
     Output('technical-chart', 'figure'),
     Output('performance-metrics', 'children'),
     Output('risk-metrics', 'children'),
     Output('sentiment-indicator', 'children'),
     Output('news-feed', 'children')],
    [Input('analyze-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('timeframe-dropdown', 'value')]
)
def update_dashboard(n_clicks, ticker, timeframe):
    if n_clicks is None or n_clicks == 0:
        empty_fig = {
            'data': [],
            'layout': {'title': 'Click Analyze to see data'}
        }
        return empty_fig, empty_fig, '', '', '', ''
    try:

        logger.info(f"Processing request for {ticker} with timeframe {timeframe}")
        # Get data
        end_date = datetime.now()
        extra_days = 120
        if timeframe == '1mo':
            start_date = end_date - timedelta(days=30 + extra_days)
        elif timeframe == '3mo':
            start_date = end_date - timedelta(days=90 + extra_days)
        elif timeframe == '6mo':
            start_date = end_date - timedelta(days=180 + extra_days)
        elif timeframe == '5d':
            start_date = end_date - timedelta(days=5 + extra_days)
        else:
            start_date = end_date - timedelta(days=365 + extra_days)
            
        df = yf.download(ticker, start=start_date - timedelta(days=120), end=end_date)

        if df.empty:
            raise Exception("No data downloaded from Yahoo Finance")
        
        logger.info(f"Downloaded {len(df)} data points")

        df = calculate_technical_indicators(df)
        

        
        # Create sequences with multiple features
        
        features = ['Close', 'Volume', 'RSI', 'MACD', 'BB_upper', 'BB_lower', 'SMA_20']
        
        for feature in features:
            if feature not in df.columns:
                raise Exception(f"Missing required column: {feature}")
            
        # Scale all features
        feature_data = df[features].values
        print("\nFeature Data shape:", feature_data.shape)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(feature_data)
        print("\nScaled Features shape:", scaled_data.shape)
        print("\nLen Scaled Features:", len(scaled_data))
        # Create sequences with multiple features

        X = []
        y = []
        for i in range(60, len(scaled_data)):
            sequence = scaled_data[i-60:i]
            target = scaled_data[i, 0]  # Predicting Close price
        
            if sequence.shape == (60, 7):
                X.append(sequence)
                y.append(target)

        X = np.array(X)
        y = np.array(y)
        

        logger.info(f"Prepared sequences - X shape: {X.shape}, y shape: {y.shape}")

        if len(X) == 0:
            raise Exception("Not enough data for prediction")
        
        print("\nX shape:", X.shape)
        print("\ny shape:", y.shape)
        # Create and train the advanced model
        model, lr_scheduler = create_advanced_model()  # This is where we call the function from models.py

        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, 
                                             monitor='val_loss', 
                                             restore_best_weights=True)
        ]
        history = model.fit(X, y, 
                            validation_split = 0.2,
                            batch_size=32,
                            epochs=50,
                            callbacks=callbacks, 
                            verbose=0)
        
        # Make prediction
        last_60_days = scaled_data[-60:].reshape(1, 60, 7)  # 7 features
        next_day_pred = model.predict(last_60_days, verbose=0)
        
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]

        pred_array = np.zeros((1, 7))
        pred_array[0, 0] = next_day_pred[0]

        next_day_price = scaler.inverse_transform(pred_array)[0, 0]

        logger.info(f"Prediction complete - next day price: {next_day_price}")
        # Convert prediction back to price
       # next_day_price = scaler.inverse_transform(
       #     np.array([next_day_pred[0]] + [0] * (len(features)-1)).reshape(1, -1)
       # )[0][0]

        # Get metrics
        metrics = calculate_metrics(df, next_day_price, train_loss, val_loss)
        
        if metrics is None:
            raise Exception("Failed to calculate metrics")
        

        # Create price chart
        price_fig = go.Figure()
        
        # Add historical data
        price_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Close'],
            mode='lines',
            name='Historical',
            line=dict(color='blue')
        ))
        
        # Add prediction point
        price_fig.add_trace(go.Scatter(
            x=[df.index[-1] + timedelta(days=1)],
            y=[next_day_price],
            mode='markers',
            name='Prediction',
            marker=dict(color='red', size=10, symbol='star')
        ))
        
        #Must update layout to show graph with title
        price_fig.update_layout(
            title=f'{ticker} Stock Price and Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True
        )
        # Add technical indicators to separate chart
        tech_fig = go.Figure()

        tech_fig.add_trace(go.Scatter(
            x=df.index, 
            y=df['Close'],
            mode='lines',
            name='Price',
            line=dict(color='blue')
            
        ))

        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_20'],
            mode='lines',
            name='SMA 20',
            line=dict(color='orange')
        ))
        
        tech_fig.add_trace(go.Scatter(
            x=df.index, y=df['SMA_50'],
            mode='lines',
            name='SMA 50',
            line=dict(color='green')
        ))
        tech_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_upper'],
            mode='lines',
            name='BB Upper',
            line=dict(color='green')
        ))
                
        tech_fig.add_trace(go.Scatter(
            x=df.index,
            y=df['BB_lower'],
            mode='lines',
            name='BB Lower',
            line=dict(color='red')
        ))

        #Update layout to show graph with title
        tech_fig.update_layout(
            title='Technical Indicators',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            height=600,
            showlegend=True
        )
        # Calculate metrics
        
                
                # Create display components
        performance_metrics = html.Div([
                html.P(f"Current Price: ${safe_format(metrics['current_price'], 'currency')}"),
                html.P(f"Price Change: {safe_format(metrics['price_change'], 'percentage')}"),
                html.P(f"Predicted Price: ${safe_format(metrics['next_day_price'], 'currency')}"),
                html.P(f"Model Training Loss: {safe_format(metrics['train_loss'], decimals=4)}"),
                html.P(f"Model Validation Loss: {safe_format(metrics['val_loss'], decimals=4)}")
        ])
                
        risk_metrics = html.Div([
                html.P(f"Volatility: {safe_format(metrics['volatility'], 'percentage')}"),
                html.P(f"Average Volume: {safe_format(metrics['avg_volume'], 'integer')}")
        ])
                
        sentiment_indicator = html.Div([
                html.P(f"Market Sentiment: {metrics['sentiment']}"),
                html.P(f"Prediction Change: {safe_format(metrics['pred_change'], 'percentage')}")
        ])

        news_metrics = html.Div([
                html.P("Coming soon...")
        ])

        return price_fig, tech_fig, performance_metrics, risk_metrics, sentiment_indicator, news_metrics
         
    except Exception as e:
        logger.error(f"Error in callback: {str(e)}", exc_info=True)
        empty_fig = {
            'data': [],
            'layout': {'title': f'Error: {str(e)}'}
        }
        return empty_fig, empty_fig, 'Error', 'Error', 'Error', 'Error'

def safe_format(value, format_type='number', decimals=2):
    """
    Safely format values with proper error handling
    format_type can be: 'number', 'percentage', 'currency', 'integer'
    """
    try:
        if isinstance(value, pd.Series):
            value = value.iloc[0]
        elif isinstance(value, np.ndarray):
            value = value.item()
        value = float(value)
        
        if format_type == 'percentage':
            return f"{value:.{decimals}f}%"
        elif format_type == 'currency':
            return f"${value:,.{decimals}f}"
        elif format_type == 'integer':
            return f"{int(value):,}"
        else:
            return f"{value:.{decimals}f}"
    except Exception as e:
        logger.error(f"Error formatting value: {str(e)}")
        return "N/A"



def main():
    try:
        logger.info("Initializing TensorFlow...")
        # Verify TensorFlow is working
        tf.constant([1, 2, 3])
        
        logger.info("Starting the Dash application...")
        app.run_server(
            debug=True,
            host='0.0.0.0',
            port=8050,
            use_reloader=False  # Disable reloader in production
        )
    except Exception as e:
        logger.error(f"Application failed: {str(e)}", exc_info=True)
        sys.exit(1)    

if __name__ == '__main__':
    main()