from flask import Flask, request, render_template
import datetime
import os
from stock import fetch_data, create_features, get_all_stocks, prepare_data, train_model, evaluate_model, predict_future
import logging 
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(levelname)s %(message)s')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            ticker = request.form['ticker']
            #days = int(request.form['days'])
            # Define the period for the data (e.g., last 1 year)
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.today() - datetime.timedelta(days=1095)).strftime('%Y-%m-%d')
            
            # Step 1: Fetch historical data
            df = fetch_data(ticker, start_date, end_date)
            
            # Step 2: Create features
            if df is None or df.empty:
                return f"No data available for ticker {ticker}."
            
            df = create_features(df)
            
            # Step 3: Prepare Data (Train-Test Split & Scaling)
            X_train, X_test, y_train, y_test, scaler = prepare_data(df)
            
            # Step 4: Train the Random Forest model
            model = train_model(X_train, y_train)
        except Exception as e: 
            app.logger.error(f"An error occurred: {str(e)}")
            return "An internal error occured. ", 500
            # Step 5: Evaluate the model and generate the plot
        plot_path, mse, mae, r2 = evaluate_model(model, X_test, y_test, ticker)
            
            # Save the model and scaler for future use
        app.config['MODEL'] = model
        app.config['SCALER'] = scaler
        app.config['TICKER'] = ticker
        #app.config['DAYS'] = days
            # Step 6: Predict the future price for the input ticker
        predictions = predict_future(model, scaler, ticker)
    
    return render_template('result.html', ticker=ticker, plot_url=plot_path, mse=mse, mae=mae, r2=r2, predictions=predictions)

@app.route('/future_predictions', methods=['POST'])
def future_predictions():
    model = app.config['MODEL']
    scaler = app.config['SCALER']
    ticker = app.config['TICKER']
    #days = app.config['DAYS']
    
    # Step 6: Predict the future price for the input ticker
    predictions = predict_future(model, scaler, ticker)
    return render_template('predictions.html', ticker=ticker, predictions=predictions)


@app.route('/all', methods=['POST'])
def all_stocks():
    if request.method == 'POST':
        try:
            spec_columns, sdl = get_all_stocks() #every stock from the csv file
    
            #This is an example and will need to be trained and analyzed by the model.
            #For now, just taking whichever stock has the largest market cap. 
            
            largest_market_cap = spec_columns["Market Cap"].idxmax()
            ticker = spec_columns.at[largest_market_cap, "Symbol"]
            end_date = datetime.datetime.today().strftime('%Y-%m-%d')
            start_date = (datetime.datetime.today() - datetime.timedelta(days=1095)).strftime('%Y-%m-%d')
    
    
            # Step 1: Fetch historical data
            df = fetch_data(ticker, start_date, end_date)
            print("Dead yet?")
            # Step 2: Create features
            if df is None or df.empty:
                return f"No data available for ticker {ticker}."
            
            df = create_features(df)
            
            # Step 3: Prepare Data (Train-Test Split & Scaling)
            X_train, X_test, y_train, y_test, scaler = prepare_data(df)
            
            # Step 4: Train the Random Forest model
            model = train_model(X_train, y_train)

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return "An internal error occured. ", 500
        
        # Step 5: Evaluate the model and generate the plot
        plot_path, mse, mae, r2 = evaluate_model(model, X_test, y_test, ticker)

        app.config['MODEL'] = model
        app.config['SCALER'] = scaler
        app.config['TICKER'] = largest_market_cap

        predictions = predict_future(model, scaler, ticker)
    return render_template('all.html', stock=ticker, predictions=predictions, plot_path=plot_path, mse=mse, mae=mae, r2=r2)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=True)