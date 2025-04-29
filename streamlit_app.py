import streamlit as st
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# ==========================
# Utility Functions
# ==========================

def fetch_ohlc_data(pair="BTC/USD", interval="60", since=None):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    if since:
        url += f"&since={since}"
    response = requests.get(url)
    data = response.json()

    if not data["error"]:
        key = list(data["result"].keys())[0]
        df = pd.DataFrame(data["result"][key], columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["time"] = pd.to_datetime(df["time"], unit='s')
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df
    return pd.DataFrame()

def add_indicators(df, sma=True, ema=True):
    if sma:
        df["SMA_20"] = df["close"].rolling(window=20).mean()
    if ema:
        df["EMA_20"] = df["close"].ewm(span=20, adjust=False).mean()
    return df

def generate_predictions(df):
    df = df.dropna()
    df["target"] = df["close"].shift(-1)
    X = df[["open", "high", "low", "close", "volume"]].values[:-1]
    y = df["target"].values[:-1]
    model = RandomForestRegressor()
    model.fit(X, y)
    prediction = model.predict([df[["open", "high", "low", "close", "volume"]].values[-1]])[0]
    return prediction

def generate_scalp_signals(df, threshold=0.10):
    df["scalp_signal"] = np.where(df["close"].pct_change() > threshold, "SELL",
                           np.where(df["close"].pct_change() < -threshold, "BUY", "HOLD"))
    return df

# ==========================
# Streamlit UI
# ==========================

st.set_page_config(layout="wide")
st.title("📊 Crypto Dashboard with Live Charts")

# Coin and interval selection
pair = st.selectbox("Choose trading pair:", ["BTC/USD", "ETH/USD", "ADA/USD"])
interval = st.radio("Select interval for candlestick chart:", ["1", "5", "15", "60", "240", "1440"], horizontal=True)
refresh = st.button("🔄 Refresh All Charts")
prediction_refresh = st.button("🔁 Refresh Prediction")

# Technical Indicator toggles
col1, col2, col3 = st.columns(3)
with col1:
    show_sma = st.toggle("Show SMA", value=True)
with col2:
    show_ema = st.toggle("Show EMA", value=True)
with col3:
    scalp_threshold = st.slider("Scalp threshold (%)", min_value=0.10, max_value=0.35, value=0.10, step=0.01)

# ==========================
# Data Fetching and Display
# ==========================

df = fetch_ohlc_data(pair.replace("/", ""), interval)
if df.empty:
    st.error("Failed to fetch data.")
else:
    current_price = df["close"].iloc[-1]
    last_updated = df.index[-1].strftime("%Y-%m-%d %H:%M:%S")
    st.subheader(f"💰 Current Price: {current_price:.2f} {pair.split('/')[1]}  |  Last Update: {last_updated}")

    # Top: Candlestick chart
    st.markdown("### 🕯️ Candlestick Chart")
    mpf_fig = mpf.figure(style='charles', figsize=(12, 6))
    mpf.plot(df, type='candle', ax=mpf_fig.gca(), volume=False)
    st.pyplot(mpf_fig)

    # Mid: Line chart with indicators
    st.markdown("### 📈 Price Chart with Indicators")
    df = add_indicators(df, show_sma, show_ema)
    df = generate_scalp_signals(df, threshold=scalp_threshold)

    chart_df = df[["close"]].copy()
    if show_sma:
        chart_df["SMA_20"] = df["SMA_20"]
    if show_ema:
        chart_df["EMA_20"] = df["EMA_20"]

    st.line_chart(chart_df)

    # Bottom: Prediction chart
    st.markdown("### 🤖 Predicted Price vs Actual Price")
    predicted_price = generate_predictions(df)
    prediction_time = (df.index[-1] + timedelta(minutes=int(interval))).strftime("%Y-%m-%d %H:%M:%S")

    pred_df = pd.DataFrame({
        "Actual": [df["close"].iloc[-1]],
        "Predicted": [predicted_price]
    }, index=[prediction_time])
    st.line_chart(pred_df)

    st.success(f"📍 Predicted price at {prediction_time}: {predicted_price:.2f} {pair.split('/')[1]}")

