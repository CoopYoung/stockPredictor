from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import openai
import requests
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
import plotly.graph_objects as go
from datetime import timedelta, datetime

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
class QueryRequest(BaseModel):
    question: str

class PredictRequest(BaseModel):
    stock_symbol: str

# Route to serve the homepage
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Chatbot endpoint using OpenAI
@app.post("/query")
async def query(request: QueryRequest):
    try:
        question = request.question.lower()

        # Check if the question is about the date or time
        if "date" in question or "time" in question or "تاريخ" in question or "وقت" in question:
            now = datetime.now()
            if "arabic" in question or "عربي" in question:
                date_time = now.strftime("%Y-%m-%d %H:%M:%S") + " (التاريخ والوقت الحالي)"
            else:
                date_time = now.strftime("%Y-%m-%d %H:%M:%S") + " (Current Date and Time)"
            return {"answer": date_time}

        # Check if the question is about generating an AI picture
        if "generate" in question or "picture" in question or "صورة" in question:
            response = openai.Image.create(
                prompt=question,
                n=1,
                size="1024x1024"
            )
            image_url = response['data'][0]['url']
            return {"answer": f"Here is your generated image: <img src='{image_url}' alt='Generated Image'>"}

        # Otherwise, use GPT-3.5-turbo for general questions
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": request.question}],
            max_tokens=150
        )
        answer = response['choices'][0]['message']['content'].strip()
        return {"answer": answer}
    
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

# Stock market prediction endpoint using Twelve Data
@app.post("/predict")
async def predict(request: PredictRequest):
    try:
        stock = request.stock_symbol.upper().strip()

        # Fetch last 30 days of stock data from Twelve Data
        url = f"https://api.twelvedata.com/time_series?symbol={stock}&interval=1day&outputsize=30&apikey={TWELVE_DATA_API_KEY}"
        response = requests.get(url)
        data = response.json()

        # Handle API errors
        if "values" not in data:
            return JSONResponse(content={"error": f"Could not fetch data for {stock}. {data.get('message', '')}"}, status_code=400)

        # Process stock data
        stock_data = data["values"]
        stock_data.reverse()  # API returns data in descending order, reverse for chronological order

        # Convert to DataFrame
        df = pd.DataFrame(stock_data)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df["close"] = df["close"].astype(float)
        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)

        # Use last 5 trading days for prediction
        if len(df) < 5:
            return JSONResponse(content={"error": f"Not enough data for {stock} to make a prediction."}, status_code=400)

        last_week_data = df.tail(5).reset_index()
        last_week_data["Day"] = np.arange(len(last_week_data))
        X = last_week_data["Day"].values.reshape(-1, 1)
        y = last_week_data["close"].values

        # Train a simple linear regression model
        model = LinearRegression().fit(X, y)
        next_day_index = np.array([[len(last_week_data)]])
        predicted_price = model.predict(next_day_index).item()

        # Generate a candlestick chart using the last week's data
        fig = go.Figure(data=[go.Candlestick(
            x=last_week_data["datetime"].dt.strftime("%Y-%m-%d"),
            open=last_week_data["open"],
            high=last_week_data["high"],
            low=last_week_data["low"],
            close=last_week_data["close"]
        )])

        # Calculate the predicted date (next day after the last date)
        last_date = last_week_data["datetime"].iloc[-1]
        predicted_date = (last_date + timedelta(days=1)).strftime("%Y-%m-%d")

        # Add a marker for the predicted price
        fig.add_trace(go.Scatter(
            x=[predicted_date],
            y=[predicted_price],
            mode="markers",
            marker=dict(color="red", size=10),
            name="Predicted Price"
        ))

        # Update chart layout
        fig.update_layout(
            title=f"{stock} - Last Week Candlestick Chart with Prediction",
            xaxis_title="Date",
            yaxis_title="Price",
            xaxis_rangeslider_visible=False
        )

        # Convert the chart to HTML
        chart_html = fig.to_html(full_html=False)

        prediction_text = f"The predicted closing price for {stock} on {predicted_date} is ${predicted_price:.2f}"
        return {"prediction": prediction_text, "chart": chart_html}

    except Exception as e:
        return {"error": f"Error: {str(e)}"}

# Run the app if this file is executed directly.
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
