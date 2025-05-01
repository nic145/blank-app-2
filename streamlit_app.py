import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import os
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Set page configuration
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

# Constants
DATA_DIR = "local_data"
os.makedirs(DATA_DIR, exist_ok=True)
DEFAULT_COIN = "BTC/USDT"
TOP_COINS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "SOL/USDT", "ADA/USDT", "DOGE/USDT", "AVAX/USDT"]
INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Settings")
    selected_coin = st.selectbox("Select Coin", TOP_COINS + ["Custom"])
    if selected_coin == "Custom":
        selected_coin = st.text_input("Enter custom coin (e.g., LTC/USDT)", "LTC/USDT")

    selected_interval = st.radio("Candlestick Interval", list(INTERVALS.keys()), index=3)
    leverage = st.slider("Leverage (x)", 1, 50, 25)
    capital = st.number_input("Capital per Trade ($)", 100, 10000, 400, step=50)
    show_discord = st.toggle("Enable Discord Alerts", value=False)
    webhook_url = st.text_input("Discord Webhook URL", type="password")

# Fetch data from Kraken
def fetch_ohlcv(symbol, interval="1h", limit=200):
    pair = symbol.replace("/", "")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={INTERVALS[interval]}"
    try:
        res = requests.get(url).json()
        key = list(res["result"].keys())[0]
        ohlcv = res["result"][key]
        df = pd.DataFrame(ohlcv, columns=["Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])
        df["Time"] = pd.to_datetime(df["Time"], unit="s")
        df.set_index("Time", inplace=True)
        df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)
        return df.tail(limit)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load model from disk
@st.cache_resource
def load_model():
    try:
        with open(f"{DATA_DIR}/model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(f"{DATA_DIR}/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        return model, scaler
    except:
        return None, None

# Save model
def save_model(model, scaler):
    with open(f"{DATA_DIR}/model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(f"{DATA_DIR}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

# Train model
def train_model(df):
    df["Return"] = df["Close"].pct_change()
    df.dropna(inplace=True)
    features = df[["Close", "Volume"]].values
    target = df["Close"].shift(-1).dropna().values

    scaler = MinMaxScaler()
    scaled_features = scaler.fit_transform(features[:-1])

    model = RandomForestRegressor()
    model.fit(scaled_features, target)
    return model, scaler

# Predict next price
def predict_price(model, scaler, df):
    last_data = df[["Close", "Volume"]].iloc[-1:].values
    scaled = scaler.transform(last_data)
    return model.predict(scaled)[0] if model else None

# Display dashboard
df = fetch_ohlcv(selected_coin, selected_interval)
if not df.empty:
    st.markdown(f"### {selected_coin} — Interval: {selected_interval}")
    st.line_chart(df["Close"])

    model, scaler = load_model()
    if model and scaler:
        pred = predict_price(model, scaler, df)
        latest_price = df["Close"].iloc[-1]
        signal = "BUY" if pred > latest_price else "SELL" if pred < latest_price else "HOLD"
        st.metric("Current Price", f"${latest_price:,.2f}")
        st.metric("Predicted Price", f"${pred:,.2f}" if pred else "N/A")
        st.metric("Signal", signal)
        
        # Simulated PnL
        change = ((pred - latest_price) / latest_price) * 100
        pnl = (change / 100) * capital * leverage * (1 if signal == "BUY" else -1)
        st.markdown(f"**Simulated PnL (@{leverage}x on ${capital}):** ${pnl:.2f}")

        if show_discord and webhook_url and signal in ["BUY", "SELL"]:
            msg = {
                "content": f"**{signal} Signal for {selected_coin}**\nPrice: ${latest_price:.2f}\nPrediction: ${pred:.2f}\nSimulated PnL: ${pnl:.2f}"
            }
            try:
                requests.post(webhook_url, json=msg)
                st.success("Discord alert sent.")
            except:
                st.warning("Failed to send Discord alert.")
    else:
        st.info("Model not trained yet.")

    if st.button("Train Model Now"):
        with st.spinner("Training model..."):
            model, scaler = train_model(df)
            save_model(model, scaler)
            st.success("Model trained and cached. Reloading...")
            st.rerun()
else:
    st.warning("No data to display.")
