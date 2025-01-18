import streamlit as st
import pickle
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import seaborn as sns
import os
from datetime import date
from prophet import Prophet
from prophet.plot import plot_plotly
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

# ---- Stock Prediction Chart ---- #
@st.cache_data(ttl=3600)
def stock_prediction_chart(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1mo")
    if not hist.empty:
        last_price = hist['Close'].iloc[-1]
        st.line_chart(hist['Close'])
        st.write(f"The last price of {ticker} is ${last_price:.2f}.")
    else:
        st.write(f"No historical data available for {ticker}.")

# ---- Stock Forecast ---- #
def stock_forecast_app():
    START = "2015-01-01"
    TODAY = date.today().strftime("%Y-%m-%d")

    # User Input
    user_input_stock = st.text_input('Enter stock symbol:', 'AAPL')
    n_years = st.slider('Years of prediction:', 1, 4)
    period = n_years * 365

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
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close"))
    fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

    # Forecast
    df_train = data[['Date', 'Close']].rename(columns={"Date": "ds", "Close": "y"})
    m = Prophet()
    m.fit(df_train)
    future = m.make_future_dataframe(periods=period)
    forecast = m.predict(future)

    # Display Forecast
    st.write(f"Forecast for {n_years} years")
    st.plotly_chart(plot_plotly(m, forecast))

# ---- Main App ---- #
def main():
    st.title("Investment Portfolio Management")
    st.sidebar.title("Navigation")
    options = st.sidebar.radio("Select an option", ("Bank Accounts", "Stocks", "Stock Forecast"))

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

    elif options == "Stock Forecast":
        stock_forecast_app()

if __name__ == "__main__":
    main()
