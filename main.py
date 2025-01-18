import streamlit as st
import pandas as pd  # Import pandas
import pickle
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import os
from datetime import date
from statsmodels.tsa.arima.model import ARIMA
from plotly import graph_objs as go

# ---- File Management Setup ---- #
DATA_DIR = "data"  # All dynamic files are stored in this directory
os.makedirs(DATA_DIR, exist_ok=True)

BANK_FILE = os.path.join(DATA_DIR, 'bank_accounts.pkl')
PORTFOLIO_FILE = os.path.join(DATA_DIR, 'portfolio.pkl')

# ---- Utility Functions ---- #
def load_data(file_path):
    try:
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        return {}
    except Exception as e:
        st.warning(f"Error loading data from {file_path}: {e}")
        return {}

def save_data(file_path, data):
    try:
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.error(f"Error saving data to {file_path}: {e}")

# ---- Bank Functions ---- #
bank_accounts = load_data(BANK_FILE)

def add_bank_account(account_name, balance):
    bank_accounts[account_name] = balance
    save_data(BANK_FILE, bank_accounts)
    return f"Added bank account '{account_name}' with balance ${balance:.2f}."

def remove_bank_account(account_name):
    if account_name in bank_accounts:
        del bank_accounts[account_name]
        save_data(BANK_FILE, bank_accounts)
        return f"Removed bank account '{account_name}'."
    else:
        return f"Bank account '{account_name}' not found."

def show_bank_accounts():
    if bank_accounts:
        response = "Your bank accounts:\n"
        for account_name, balance in bank_accounts.items():
            response += f"{account_name}: ${balance:.2f}\n"
        return response.strip()
    else:
        return "No bank accounts found."

def total_balance():
    return f"Total balance across all accounts: ${sum(bank_accounts.values()):.2f}"

# ---- Portfolio Functions ---- #
portfolio = load_data(PORTFOLIO_FILE)

def add_stock(ticker, amount):
    ticker = ticker.upper()
    if ticker in portfolio:
        portfolio[ticker] += amount
    else:
        portfolio[ticker] = amount
    save_data(PORTFOLIO_FILE, portfolio)
    return f"Added {amount} shares of {ticker} to your portfolio."

def show_portfolio():
    if portfolio:
        response = "Your current portfolio:\n"
        for ticker, shares in portfolio.items():
            response += f"{ticker}: {shares} shares\n"
        return response.strip()
    else:
        return "Your portfolio is empty."

# ---- Portfolio Optimization ---- #
@st.cache_data(ttl=3600)
def get_stock_data(tickers, period='1y'):
    try:
        df = yf.download(tickers, period=period)['Close']
        return df
    except Exception as e:
        st.error(f"Error retrieving stock data: {e}")
        return None

def optimize_portfolio(mean_returns, cov_matrix, risk_free_rate=0.02):
    num_assets = len(mean_returns)
    args = (mean_returns, cov_matrix, risk_free_rate)

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds: weights between 0 and 1
    bounds = tuple((0, 1) for _ in range(num_assets))

    # Initial weights
    initial_weights = num_assets * [1. / num_assets]

    # Define the Sharpe ratio optimization function
    def negative_sharpe_ratio(weights, mean_returns, cov_matrix, risk_free_rate):
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
        return -sharpe_ratio  # Negative because we are minimizing

    # Optimize
    result = minimize(negative_sharpe_ratio, initial_weights, args=args, method='SLSQP', bounds=bounds, constraints=constraints)
    return result

def display_allocation_pie_chart(tickers, weights):
    fig, ax = plt.subplots()
    ax.pie(weights, labels=tickers, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("husl", len(tickers)))
    ax.axis('equal')
    st.pyplot(fig)

def recommend_mpt_allocation():
    st.header("Portfolio Optimization Using Modern Portfolio Theory")

    tickers = list(portfolio.keys())
    if not tickers:
        return st.write("Your portfolio is empty. Please add stocks to proceed with optimization.")

    # Fetch stock data
    df = get_stock_data(tickers)
    if df is None:
        return st.write("Error fetching stock data. Please try again later.")

    returns = df.pct_change().dropna()
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    # Optimize the portfolio
    optimized_result = optimize_portfolio(mean_returns, cov_matrix)
    optimized_weights = optimized_result.x

    st.subheader("Optimized Portfolio Allocation")
    display_allocation_pie_chart(tickers, optimized_weights)

    st.write("### Recommended Allocation:")
    for ticker, weight in zip(tickers, optimized_weights):
        st.write(f"{ticker}: **{weight * 100:.2f}%**")

# ---- Stock Forecast Using ARIMA ---- #
def stock_forecast_app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    # User Input
    user_input_stock = st.text_input('Enter stock symbol:', 'AAPL')
    n_days = st.slider('Days of prediction:', 1, 30)

    @st.cache_data
    def load_data(ticker):
        data = yf.download(ticker, START, TODAY)
        data.reset_index(inplace=True)
        return data

    # Load and Display Data
    data = load_data(user_input_stock)
    st.subheader('Raw data')
    st.write(data.tail())

    # Plot Data
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Close Price"))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # ARIMA Forecast
    st.subheader(f'{user_input_stock} Price Prediction for {n_days} Days')

    def arima_forecast(data, n_days):
        close_prices = data['Close']
        model = ARIMA(close_prices, order=(5, 1, 0))  # ARIMA parameters (p, d, q)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=n_days)
        return forecast

    # Predict for 30 days
    n_days = 30
    forecast = arima_forecast(data, n_days)

    # Generate future dates for plotting
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, n_days + 1)]

    # Prepare forecast DataFrame
    forecast_df = pd.DataFrame({'Date': future_dates, 'Forecast': forecast})

    # Plot the forecast
    st.subheader("Forecasted Prices for the Next 30 Days")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="Historical Prices"))
    fig.add_trace(go.Scatter(x=forecast_df['Date'], y=forecast_df['Forecast'], name="Forecasted Prices"))
    fig.layout.update(title_text=f'{user_input_stock} Price Prediction for 30 Days', xaxis_title="Date",
                      yaxis_title="Price")
    st.plotly_chart(fig)

    # Display Forecast Data
    st.write("Forecasted Prices:")
    st.write(forecast_df)


# ---- Main App ---- #
def main():
    st.title("Investment Portfolio Management")
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ("Bank Accounts", "Stocks", "Portfolio Optimization", "Stock Forecast"))

    if options == "Bank Accounts":
        st.header("Manage Bank Accounts")
        action = st.selectbox("Action", ("Add Account", "Show Accounts"))
        if action == "Add Account":
            account_name = st.text_input("Account Name")
            balance = st.number_input("Balance", min_value=0.0)
            if st.button("Add Account"):
                st.write(add_bank_account(account_name, balance))
        elif action == "Show Accounts":
            st.write(show_bank_accounts())
            st.write(total_balance())

    elif options == "Stocks":
        st.header("Manage Stock Portfolio")
        action = st.selectbox("Action", ("Add Stock", "Show Portfolio"))
        if action == "Add Stock":
            ticker = st.text_input("Stock Ticker")
            amount = st.number_input("Number of Shares", min_value=0)
            if st.button("Add Stock"):
                st.write(add_stock(ticker, amount))
        elif action == "Show Portfolio":
            st.write(show_portfolio())

    elif options == "Portfolio Optimization":
        recommend_mpt_allocation()

    elif options == "Stock Forecast":
        stock_forecast_app()

if __name__ == "__main__":
    main()

