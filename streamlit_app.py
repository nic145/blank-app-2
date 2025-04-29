# streamlit_app.py

import streamlit as st
import pandas as pd
import numpy as np
import requests
import pytz
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import mplfinance as mpf

# Must be first
st.set_page_config(page_title="Crypto Predictor", layout="wide")

# Sidebar - Timezone selector
timezones = pytz.all_timezones
selected_tz = st.sidebar.selectbox("Select your time zone", ["UTC"] + sorted(timezones))
local_tz = pytz.timezone(selected_tz)

# Constants
KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC"
ASSET_PAIR = "XBTUSD"
INTERVAL_MAP = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
DEFAULT_INTERVAL = "1h"
REFRESH_INTERVAL_MIN = 1

# Global state
if "last_refreshed" not in st.session_state:
    st.session_state.last_refreshed = datetime.utcnow()

# Fetch Kraken OHLC data
@st.cache_data(ttl=REFRESH_INTERVAL_MIN * 60)
def fetch_ohlc(pair="XBTUSD", interval=60):
    params = {"pair": pair, "interval": interval}
    response = requests.get(KRAKEN_API_URL, params=params).json()
    result = list(response["result"].values())[0]
    df = pd.DataFrame(result, columns=[
        "time", "open", "high", "low", "close", "vwap", "volume", "count"
    ])
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

# Predict future price
def predict(df):
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]

    if len(X) != len(y):
        min_len = min(len(X), len(y))
        X = X.iloc[:min_len]
        y = y.iloc[:min_len]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    current = df[["open", "high", "low", "close", "volume"]].iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(current)[0]
    predicted_time = df.index[-1] + timedelta(minutes=INTERVAL_MAP[selected_interval])
    return predicted_price, predicted_time

# Buy/Sell/Hold signal
def get_signal(current_price, predicted_price):
    change = (predicted_price - current_price) / current_price
    if change > 0.10:
        return "BUY"
    elif change < -0.10:
        return "SELL"
    else:
        return "HOLD"

# Interface
st.title("📈 Live Crypto Dashboard")
st.markdown("Made with ❤️ using Kraken API")

selected_interval = st.radio("Select Interval", list(INTERVAL_MAP.keys()), index=list(INTERVAL_MAP).index(DEFAULT_INTERVAL), horizontal=True)

df = fetch_ohlc(ASSET_PAIR, INTERVAL_MAP[selected_interval])

# Chart 1: Candlestick
st.subheader(f"Candlestick Chart - BTC/USD ({selected_interval})")
mpf_plot = mpf.plot(df[-60:], type='candle', style='charles', volume=False, returnfig=True)
st.pyplot(mpf_plot[0])

# Compute EMA/SMA
st.subheader("Price & Prediction Comparison")
show_sma = st.checkbox("Show SMA", value=True)
show_ema = st.checkbox("Show EMA", value=True)
sma = df["close"].rolling(window=10).mean()
ema = df["close"].ewm(span=10).mean()

plt.figure(figsize=(12, 5))
plt.plot(df.index, df["close"], label="Actual Price")
if show_sma:
    plt.plot(sma.index, sma, label="SMA (10)")
if show_ema:
    plt.plot(ema.index, ema, label="EMA (10)")
plt.legend()
plt.grid(True)
plt.tight_layout()
st.pyplot(plt)

# Prediction
predicted_price, predicted_time = predict(df)
local_pred_time = predicted_time.astimezone(local_tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')
local_now = datetime.now(pytz.utc).astimezone(local_tz).strftime('%Y-%m-%d %I:%M:%S %p %Z')

current_price = df["close"].iloc[-1]
signal = get_signal(current_price, predicted_price)

# Price Display
st.subheader("🔮 Price Prediction")
st.metric("Predicted Price", f"${predicted_price:,.2f}", delta=f"{(predicted_price - current_price):.2f}")
st.metric("Current Price", f"${current_price:,.2f}")
st.write(f"**Prediction Time:** {local_pred_time}")
st.write(f"**Last Updated:** {local_now}")
st.write(f"**Signal:** `{signal}`")

# Buttons
col1, col2 = st.columns(2)
if col1.button("🔁 Refresh Charts"):
    st.cache_data.clear()
    st.rerun()
if col2.button("📈 Refresh Prediction"):
    st.rerun()
