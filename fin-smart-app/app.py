from flask import Flask, render_template, request
import yfinance as yf
import numpy as np
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import plotly.graph_objs as go
import plotly.io as pio
from predictor import predict_next_5_days  # Import the prediction function

app = Flask(__name__, template_folder='template')

newsapi = NewsApiClient(api_key='d563ec80927e4b1998a3635fc05d06a1')
vader = SentimentIntensityAnalyzer()

def fetch_stock_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        data = stock.history(period="6mo", interval="1d")
        if data.empty:
            raise ValueError(f"No stock data found for ticker: {ticker}")
        data['SMA_20'] = data['Close'].rolling(window=20).mean()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['RSI'] = calculate_rsi(data['Close'])
        data['MACD'], data['Signal_Line'] = calculate_macd(data['Close'])
        return data
    except:
        return None

def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_macd(series, fast=12, slow=26, signal=9):
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

def fetch_news_sentiment(ticker):
    try:
        articles = newsapi.get_everything(
            q=f"{ticker} OR stock",
            language="en",
            sort_by="publishedAt",
            page_size=10
        )
        sentiments = [
            vader.polarity_scores(article.get('description', '') or article['title'])['compound']
            for article in articles['articles']
        ]
        return np.mean(sentiments) if sentiments else 0
    except:
        return 0

def calculate_risk_analysis(data, market_data=None):
    try:
        daily_returns = data['Close'].pct_change().dropna()
        risk_free_rate = 0.02 / 252
        avg_return = daily_returns.mean()
        volatility = daily_returns.std()
        sharpe_ratio = (avg_return - risk_free_rate) / volatility if volatility > 0 else 0
        var_95 = np.percentile(daily_returns, 5)
        beta = None
        if market_data is not None:
            market_returns = market_data['Close'].pct_change().dropna()
            cov_matrix = np.cov(daily_returns, market_returns)
            beta = cov_matrix[0, 1] / cov_matrix[1, 1] if cov_matrix[1, 1] > 0 else 0
        risk_level = "Low Risk"
        if volatility > 0.02:
            risk_level = "High Risk"
        elif volatility > 0.01:
            risk_level = "Mild Risk"
        return {
            "sharpe_ratio": sharpe_ratio,
            "var_95": var_95,
            "beta": beta,
            "risk_level": risk_level
        }
    except:
        return {
            "sharpe_ratio": None,
            "var_95": None,
            "beta": None,
            "risk_level": "Unknown"
        }

def make_recommendation(data, sentiment):
    try:
        current_price = data['Close'].iloc[-1]
        sma_20 = data['SMA_20'].iloc[-1]
        sma_50 = data['SMA_50'].iloc[-1]
        macd = data['MACD'].iloc[-1]
        signal_line = data['Signal_Line'].iloc[-1]
        rsi = data['RSI'].iloc[-1]
        recommendation = "Hold"
        if sentiment >= 0.05 and sma_20 > sma_50 and macd > signal_line and rsi < 70:
            recommendation = "Buy"
        elif sentiment < -0.05 or (rsi > 70 and macd < signal_line) or (sma_20 < sma_50):
            recommendation = "Sell"
        return recommendation
    except:
        return "Hold"

def generate_plot(data, ticker):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=data.index,
                                 open=data['Open'],
                                 high=data['High'],
                                 low=data['Low'],
                                 close=data['Close'],
                                 name='Candlestick'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_20'], mode='lines', name='SMA 20'))
    fig.add_trace(go.Scatter(x=data.index, y=data['SMA_50'], mode='lines', name='SMA 50'))
    fig.add_trace(go.Scatter(x=data.index, y=data['MACD'], mode='lines', name='MACD'))
    fig.add_trace(go.Scatter(x=data.index, y=data['Signal_Line'], mode='lines', name='Signal Line'))
    buy_signals = data[(data['SMA_20'] > data['SMA_50']) & (data['MACD'] > data['Signal_Line'])]
    sell_signals = data[(data['SMA_20'] < data['SMA_50']) & (data['MACD'] < data['Signal_Line'])]
    fig.add_trace(
        go.Scatter(x=buy_signals.index, y=buy_signals['Close'], mode='markers', marker=dict(color='green', size=10),
                   name='Buy Signal'))
    fig.add_trace(
        go.Scatter(x=sell_signals.index, y=sell_signals['Close'], mode='markers', marker=dict(color='red', size=10),
                   name='Sell Signal'))
    fig.update_layout(
        title=f'Stock Analysis for {ticker}',
        xaxis_title='Date',
        yaxis_title='Price',
        height=600,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5)
    )
    return pio.to_html(fig, full_html=False)

@app.route("/", methods=["GET", "POST"])
def index():
    recommendation = None
    stock_data = None
    sentiment = None
    ticker = None
    error_message = None
    plot_html = None
    risk_metrics = None
    predictions = None

    if request.method == "POST":
        ticker = request.form["ticker"].upper()
        stock_data = fetch_stock_data(ticker)

        if stock_data is None or stock_data.empty:
            error_message = f"Unable to fetch stock data for ticker: {ticker}. Please check the symbol or try again later."
        else:
            sentiment = fetch_news_sentiment(ticker)
            recommendation = make_recommendation(stock_data, sentiment)
            market_data = yf.Ticker('^GSPC').history(period="6mo", interval="1d")
            risk_metrics = calculate_risk_analysis(stock_data, market_data)
            plot_html = generate_plot(stock_data, ticker)
            predictions = predict_next_5_days(ticker)

    return render_template(
        'index.html',
        stock_data=stock_data,
        recommendation=recommendation,
        ticker=ticker,
        error_message=error_message,
        plot_html=plot_html,
        risk_metrics=risk_metrics,
        predictions=predictions
    )

if __name__ == "__main__":
    app.run(debug=True)
