import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objs as go
from dash.exceptions import PreventUpdate

# Initialize the Flask app with Dash
app = Dash(__name__)

# List of popular stock tickers
AVAILABLE_TICKERS = [
    'AAPL', 'GOOGL', 'MSFT', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 
    'BAC', 'WMT', 'DIS', 'NFLX', 'INTC', 'AMD', 'UBER'
]

def validate_ticker(ticker):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return True if info else False
    except:
        return False

def create_model(sequence_length=60):
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
        tf.keras.layers.LSTM(50, return_sequences=False),
        tf.keras.layers.Dense(25),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_data(ticker, start_date, end_date):
    df = yf.download(ticker, start=start_date, end=end_date)

    print("Sample of downloaded data:")
    print(df.head())
    print(f"Number of data points: {len(df)}")

    if df.empty:
        raise ValueError(f"No data found for ticker {ticker}")
    
    data = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X = []
    y = []
    for i in range(60, len(scaled_data)):
        X.append(scaled_data[i-60:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler, df, scaled_data

# Modified Layout with initial graph
app.layout = html.Div([
    html.H1("Stock Price Prediction", 
            style={'textAlign': 'center', 'marginBottom': '20px'},
            className='head-title'),
    html.P("Created by: AI is the Ally LLC",
           style={'textAlign': 'center', 'marginBottom': '20px'},
           className='head-title'),
         html.Div([
            html.Img(
                src='/assets/logo.png',
                className='logo-image'
            ),
        ], className='logo-header'),
        # Dropdown for predefined tickers
        html.Div([
            html.Label("Select from popular tickers:"),
            dcc.Dropdown(
                id='ticker-dropdown',
                options=[{'label': ticker, 'value': ticker} 
                        for ticker in AVAILABLE_TICKERS],
                value='AAPL'  # Set default value
            ),
             
        ], style={'width': '48%', 'display': 'inline-block'}),
        
       
        # Custom ticker input
        html.Div([
            html.Label("Or enter custom ticker:"),
            dcc.Input(
                id='custom-ticker-input',
                type='text',
                placeholder='Enter custom ticker (e.g., UBER)',
            ),
        ], style={'width': '48%', 'display': 'inline-block', 'float': 'right'}),
    
    
    html.Div([
        html.Button(
            'Predict', 
            id='predict-button',
            n_clicks=0,  # Initialize n_clicks
            style={'margin-top': '20px'}
        ),
    ], style={'textAlign': 'center'}),
    
    html.Div(id='error-message', style={'color': 'red', 'margin-top': '10px'}),
    
    dcc.Loading(
        id="loading",
        type="default",
        children=[
            dcc.Graph(id='stock-graph',
                      figure={
                            'data': [],
                            'layout': {
                                'title': 'Click Predict to see stock prediction',
                                'xaxis': {'title': 'Date'},
                                'yaxis': {'title': 'Price ($)'}
                            }
                      },
                      config={
                            'displayModeBar': True,
                            'scrollZoom': True
                      },
                      style={'height': '600px'}
                      ),
            html.Div(id='prediction-output', 
                    style={'textAlign': 'center', 'margin-top': '20px'})
        ]
    ),
    
    html.Div(id='stock-metrics', style={'margin-top': '20px'}),
], style={'padding': '20px'})

# Modified callback
@app.callback(
    [Output('stock-graph', 'figure'),
     Output('prediction-output', 'children'),
     Output('error-message', 'children'),
     Output('stock-metrics', 'children')],
    [Input('predict-button', 'n_clicks')],
    [State('ticker-dropdown', 'value'),
     State('custom-ticker-input', 'value')]
)
def update_graph(n_clicks, dropdown_ticker, custom_ticker):

    empty_fig = {
        'data': [],
        'layout': {
            'title': 'Select a ticker and click Predict to start',
            'xaxis': {'title': 'Date'},
            'yaxis': {'title': 'Price ($)'}
        }
    }
    # Print for debugging
    print(f"Callback triggered - n_clicks: {n_clicks}, ticker: {dropdown_ticker}")
    
    if n_clicks is None or n_clicks == 0:
        # Return empty figure with message for initial load
        return empty_fig, '', '', ''
    
    ticker = custom_ticker if custom_ticker else dropdown_ticker
    
    if not ticker:
        return empty_fig, '', 'Please select or enter a ticker symbol.', ''
    
    ticker = ticker.upper().strip()
    
    try:
        # Get historical data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=365)
        
        # Download stock data
        df = yf.download(ticker, start=start_date, end=end_date)
        print(f"Data downloaded for {ticker}: {len(df)} rows")  # Debug print
        
        if df.empty:
            return empty_fig, '', f'No data found for ticker {ticker}', ''
        
        # Create the graph
        fig = go.Figure()

        # Debug prints
        print("Close prices shape:", df['Close'].shape)
        print("Date range:", df.index[0], "to", df.index[-1])   

        # Add historical data trace
        '''
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['Close'],
                mode='lines',
                name='Historical',
                line=dict(color='blue')
            )
        )
        '''
        # Add historical data trace with explicit values
        historical_trace = go.Scatter(
            x=df.index.values,  # Convert pandas index to numpy array
            y=df['Close'].values,  # Convert pandas series to numpy array
            mode='lines',
            name='Historical',
            line=dict(color='blue', width=2)
        )
        fig.add_trace(historical_trace)
        # Prepare data for prediction
        data = df['Close'].values.reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(data)
        
        # Create sequences
        X = []
        y = []
        for i in range(60, len(scaled_data)):
            X.append(scaled_data[i-60:i, 0])
            y.append(scaled_data[i, 0])
        
        X = np.array(X)
        y = np.array(y)
        
        # Split data and train model
        split = int(0.8 * len(X))
        X_train = X[:split]
        y_train = y[:split]
        
        model = create_model()
        model.fit(X_train, y_train, batch_size=32, epochs=10, verbose=0)
        
        # Make prediction
        last_60_days = scaled_data[-60:]
        next_day_pred = model.predict(last_60_days.reshape(1, 60, 1))
        next_day_price = float(scaler.inverse_transform(next_day_pred)[0][0])
        print(f"Prediction value: {next_day_price}")
        # Add prediction point
        '''
        fig.add_trace(
            go.Scatter(
                x=[df.index[-1] + timedelta(days=1)],
                y=[next_day_price],
                mode='markers',
                name='Prediction',
                marker=dict(
                    color='red',
                    size=10,
                    symbol='star'
                )
            )
        )
        '''
        # Add prediction point with explicit values
        prediction_trace = go.Scatter(
            x=[df.index[-1] + timedelta(days=1)],  # Next day
            y=[next_day_price],
            mode='markers',
            name='Prediction',
            marker=dict(
                color='red',
                size=12,
                symbol='star',
                line=dict(color='red', width=2)
            )
        )
        fig.add_trace(prediction_trace)  

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'{ticker} Stock Price Prediction',
                x=0.5,
                xanchor='center',
                font=dict(size=20)
            ),
            xaxis=dict(
                    title='Date',
                    showgrid=True,
                    gridwidth=1,
                    gridcolor='LightGray',
                    range=[df.index[0], df.index[-1] + timedelta(days=5)]  # Extend range slightly
            ),
            yaxis=dict(
                title='Price ($)',
                showgrid=True,
                gridwidth=1,
                gridcolor='LightGray',
                range=[df['Close'].min() * 0.95, df['Close'].max() * 1.05]  # Add 5% padding
            ),
            xaxis_title='Date',
            yaxis_title='Price ($)',
            hovermode='x unified',
            showlegend=True,
            template='plotly_white',
            height=600,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        # Add range selector
        fig.update_xaxes(
            rangeslider_visible=False,
            rangeselector=dict(
                buttons=list([
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ]),
                bgcolor='WhiteSmoke',
                activecolor='LightSkyBlue'
            )
        )
        # Force the figure to update
        fig.update_layout(uirevision=True)    
        # Calculate metrics for display
        current_price = df['Close'].iloc[-1].item()
        price_change = next_day_price - current_price
        price_change_pct = (price_change / current_price) * 100
        
        prediction_text = html.Div([
            html.H3(f"Prediction Results for {ticker}"),
            html.P(f"Current Price: ${current_price:.2f}"),
            html.P(f"Predicted Price: ${next_day_price:.2f}"),
            html.P(f"Predicted Change: ${price_change:.2f} ({price_change_pct:.2f}%)")
        ])
        
        volume = df['Volume'].iloc[-1].item()
        high = df['High'].max().item()
        low = df['Low'].min().item()
        avg_volume = df['Volume'].mean().item()
        
        metrics = html.Div([
            html.H3("Additional Metrics"),
            html.Div([
                html.P(f"Volume: {volume:,}"),
                html.P(f"52-Week High: ${high:.2f}"),
                html.P(f"52-Week Low: ${low:.2f}"),
                html.P(f"Average Volume: {avg_volume:,}")
            ])
        ])
        #Debug print
        print("Figure data points:", len(df['Close']))

        return fig, prediction_text, '', metrics
        
    except Exception as e:
        print(f"Error: {str(e)}")  # For debugging
        return {}, '', f'Error processing {ticker}: {str(e)}', ''

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)