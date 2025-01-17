from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go
import pandas as pd
import yfinance as yf



app = Dash(__name__)

app.layout = html.Div([
    html.H4('Apple stock candlestick chart'),
    dcc.Checklist(
        id='toggle-rangeslider',
        options=[{'label': 'Include Rangeslider', 
                  'value': 'slider'}],
        value=['slider']
    ),
    dcc.Graph(id="graph"),
])


@app.callback(
    Output("graph", "figure"), 
    Input("toggle-rangeslider", "value"))
def display_candlestick(value):
    ticker = yf.Ticker("AAPL")
    df = ticker.history(period="1y")
    times = df['Open'].index
    df.insert(0, 'Date', times, True)
    fig = go.Figure(go.Candlestick(
        x=df['Date'],
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close']
    ))

    fig.update_layout(
        xaxis_rangeslider_visible='slider' in value
    )

    return fig


app.run_server(debug=True)