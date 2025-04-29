import streamlit as st
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Mapping
PAIR_MAPPING = {
    "BTC/USD": "XBTUSD",
    "ETH/USD": "ETHUSD",
    "SOL/USD": "SOLUSD"
}
INTERVAL_MAPPING = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "1h": 60,
    "4h": 240,
    "1d": 1440
}

# Kraken OHLC fetcher
def fetch_ohlcv(pair, interval):
    url = "https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair, "interval": interval}
    response = requests.get(url, params=params)
    data = response.json()
    result = list(data["result"].values())[0]
    df = pd.DataFrame(result, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

def add_indicators(df, show_sma, show_ema):
    if show_sma:
        df["SMA_50"] = df["close"].rolling(window=50).mean()
    if show_ema:
        df["EMA_50"] = df["close"].ewm(span=50).mean()
    return df

def predict_price(df):
    df = df.copy()
    df["target"] = df["close"].shift(-1)
    df.dropna(inplace=True)
    X = df[["open", "high", "low", "close", "volume"]]
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    future_price = model.predict([X.iloc[-1].values])[0]
    return future_price

# App config
st.set_page_config(layout="wide")
st.title("📈 Live Crypto Dashboard (Kraken)")

# Sidebar
pair_name = st.sidebar.selectbox("Select Pair", list(PAIR_MAPPING.keys()))
interval_label = st.sidebar.radio("Chart Interval", list(INTERVAL_MAPPING.keys()), index=3)
interval = INTERVAL_MAPPING[interval_label]
show_sma = st.sidebar.checkbox("Show SMA (50)", True)
show_ema = st.sidebar.checkbox("Show EMA (50)", True)
scalp_min = st.sidebar.slider("Min Gain %", 10, 15, 10)
scalp_max = st.sidebar.slider("Max Gain %", 10, 15, 15)
refresh = st.sidebar.button("🔄 Refresh Now")

# Data
pair = PAIR_MAPPING[pair_name]
df = fetch_ohlcv(pair, interval)
df = add_indicators(df, show_sma, show_ema)
current_price = df["close"].iloc[-1]
predicted_price = predict_price(df)
local_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# Display info
st.markdown(f"### {pair_name} Current Price: **${current_price:.2f}**")
st.markdown(f"**Predicted Price (next step):** ${predicted_price:.2f} @ {local_time}")
st.markdown(f"**Last Updated:** {local_time}")

# Scalp suggestion
scalp_min_price = current_price * (1 + scalp_min / 100)
scalp_max_price = current_price * (1 + scalp_max / 100)
st.markdown(f"💡 Scalping Zone: **${scalp_min_price:.2f} - ${scalp_max_price:.2f}**")

# Indicator Chart
st.subheader("📊 Line Chart with Indicators")
line_df = df[["close"]]
if show_sma:
    line_df["SMA_50"] = df["SMA_50"]
if show_ema:
    line_df["EMA_50"] = df["EMA_50"]
st.line_chart(line_df)

# Prediction comparison
st.subheader("📈 Actual vs Predicted")
compare_df = df[["close"]].copy()
compare_df["predicted"] = np.nan
compare_df.iloc[-1, compare_df.columns.get_loc("predicted")] = predicted_price
st.line_chart(compare_df)

# Candlestick chart
st.subheader("🕯️ Candlestick Chart")
mpf_df = df[["open", "high", "low", "close", "volume"]]
mpf.plot(mpf_df, type='candle', style='charles', title=pair_name, ylabel='Price', volume=True)

# Global refresh
if not refresh:
    st.experimental_rerun()
