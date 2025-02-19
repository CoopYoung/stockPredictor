from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import openai
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import timedelta, datetime
from fbprophet import Prophet

# Initialize the FastAPI app
app = FastAPI()

# Set API keys
OPENAI_API_KEY = "your_openai_api_key"  # Replace with your OpenAI API key
TWELVE_DATA_API_KEY = "your_twelve_data_api_key"  # Replace with your Twelve Data API key
openai.api_key = OPENAI_API_KEY

# Mount the static directory (for CSS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up templates directory (for index.html)
templates = Jinja2Templates(directory="templates")

# Define Pydantic models for request payload validation
class PredictRequest(BaseModel):
    stock_symbol: str

# Stock market prediction endpoint using Prophet and GPT-4
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        stock = request.stock_symbol.upper().strip()

        # Fetch 180 days of stock data from Twelve Data
        url = f"https://api.twelvedata.com/time_series?symbol={stock}&interval=1day&outputsize=180&apikey={TWELVE_DATA_API_KEY}"
        response = requests.get(url)
        data = response.json()

        # Handle API errors
        if "values" not in data:
            return JSONResponse(content={"error": f"Could not fetch data for {stock}. {data.get('message', '')}"}, status_code=400)

        # Process stock data
        stock_data = data["values"]
        df = pd.DataFrame(stock_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)
        df = df.rename(columns={"datetime": "ds", "close": "y"})
        df = df.sort_values(by="ds")

        # Train Prophet model
        model = Prophet()
        model.fit(df)
        future = model.make_future_dataframe(periods=5)
        forecast = model.predict(future)
        predicted_price = forecast.iloc[-1]["yhat"]
        price_min = forecast.iloc[-1]["yhat_lower"]
        price_max = forecast.iloc[-1]["yhat_upper"]
        predicted_date = forecast.iloc[-1]["ds"].strftime("%Y-%m-%d")

        # Market sentiment analysis with GPT-4
        prompt = f"Analyze the recent trend of {stock} stock and provide insights on whether it is bullish, bearish, or neutral."
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=150
        )
        sentiment_analysis = response['choices'][0]['message']['content'].strip()

        # Generate a candlestick chart
        fig = go.Figure(data=[go.Candlestick(
            x=df["ds"].dt.strftime("%Y-%m-%d"),
            open=df["y"], high=df["y"], low=df["y"], close=df["y"]
        )])
        fig.add_trace(go.Scatter(x=[predicted_date], y=[predicted_price], mode="markers", marker=dict(color="red", size=10), name="Predicted Price"))
        fig.update_layout(title=f"{stock} - Trend & AI Prediction", xaxis_title="Date", yaxis_title="Price", xaxis_rangeslider_visible=False)
        chart_html = fig.to_html(full_html=False)

        return {
            "prediction": f"Predicted closing price for {stock} on {predicted_date} is ${predicted_price:.2f} (Range: ${price_min:.2f} - ${price_max:.2f})",
            "sentiment": sentiment_analysis,
            "chart": chart_html
        }

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

