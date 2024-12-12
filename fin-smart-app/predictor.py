import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import datetime

def predict_next_5_days(ticker):
    try:
        # Fetch historical stock data
        stock = yf.Ticker(ticker)
        data = stock.history(period="1y", interval="1d")

        if data.empty:
            raise ValueError(f"No data available for ticker: {ticker}")

        # Prepare data for ML model
        data['Day'] = np.arange(len(data))  # Add numerical day column
        X = data[['Day']]  # Feature: Days
        y = data['Close']  # Target: Closing prices

        # Standardize the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Train a Linear Regression model
        model = LinearRegression()
        model.fit(X_scaled, y)

        # Predict prices for the next 5 days
        future_days = np.arange(len(data), len(data) + 5).reshape(-1, 1)
        future_days_scaled = scaler.transform(future_days)
        predictions = model.predict(future_days_scaled)

        # Prepare predictions as a list of dictionaries with dates
        start_date = data.index[-1] + datetime.timedelta(days=1)
        dates = [start_date + datetime.timedelta(days=i) for i in range(5)]
        return [{"date": date.strftime("%Y-%m-%d"), "predicted_price": round(price, 2)} for date, price in zip(dates, predictions)]

    except Exception as e:
        print(f"Error predicting stock prices: {e}")
        return None
