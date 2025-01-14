from flask import Flask, render_template, request
import yfinance as yf

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        # Get the stock symbol entered by the user
        stock_symbol = request.form["symbol"]
        
        # Fetch stock data using yfinance
        stock_data = yf.Ticker(stock_symbol)
        stock_info = stock_data.history(period="5d")
        
        # Process stock data to return the predicted price (latest closing price)
        prediction = {
            'stock_symbol': stock_symbol,
            'predicted_price': stock_info['Close'].iloc[-1]  # Latest closing price
        }
    
    # Render the template and pass the prediction if available
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
