import plotly.graph_objects as go

import pandas as pd
from datetime import datetime
import yfinance as yf

ticker = yf.Ticker("AAPL")
df = ticker.history(period="1y")

def plot_candlestick(df, ticker):
    fig = go.Figure(data=[go.Candlestick(x=df.index,
                                         open=df['Open'],
                                         high=df['High'],
                                         low=df['Low'],
                                         close=df['Close'])])

    fig.update_layout(
        title=f'{ticker} Candlestick Chart',
        xaxis_title='Date',
        yaxis_title='Price',
        font=dict(
            family="Courier New, monospace",
            size=18))
    fig.show()
plot_candlestick(df, "AAPL")