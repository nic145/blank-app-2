import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestRegressor

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# ------------------ SETTINGS ------------------
REFRESH_INTERVAL = 60  # seconds
DEFAULT_COIN = "BTC/USDT"
COINS = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT", "DOGE/USDT", "AVAX/USDT"]
INTERVALS = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}

# ------------------ SIDEBAR ------------------
st.sidebar.header("Settings")
selected_timezone = st.sidebar.selectbox("Select Timezone", pytz.all_timezones, index=pytz.all_timezones.index("UTC"))
custom_coin = st.sidebar.text_input("Add Custom Coin (e.g., LTC/USDT)")
if custom_coin and custom_coin not in COINS:
    COINS.append(custom_coin)

# ------------------ MAIN PANEL ------------------
st.title("🪙 Live Crypto Dashboard")
col1, col2 = st.columns(2)

with col1:
    selected_coin = st.selectbox("Select Cryptocurrency", COINS, index=COINS.index(DEFAULT_COIN))
with col2:
    gain_range = st.slider("Scalping % Gain Threshold", 0, 50, (10, 15), step=1)

interval_choice = st.radio("Candlestick Interval", list(INTERVALS.keys()), horizontal=True, index=3)
future_minutes = st.slider("Predict Minutes Into Future", 15, 480, 60, step=15)

# ------------------ HELPER FUNCTIONS ------------------
def fetch_ohlc(pair: str, interval: str, since=None):
    url = f"https://api.kraken.com/0/public/OHLC"
    params = {"pair": pair.replace("/", ""), "interval": INTERVALS[interval]}
    resp = requests.get(url, params=params).json()
    key = list(resp["result"].keys())[0]
    df = pd.DataFrame(resp["result"][key], columns=[
        "Time", "Open", "High", "Low", "Close", "VWAP", "Volume", "Count"])
    df["Time"] = pd.to_datetime(df["Time"], unit="s", utc=True)
    df.set_index("Time", inplace=True)
    df = df.astype(float)
    return df[["Open", "High", "Low", "Close", "Volume"]]

def predict(df, future_minutes):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    df.dropna(inplace=True)

    X = df[["Open", "High", "Low", "Volume", "Volatility"]]
    y = df["Close"].shift(-1).dropna()
    X = X.iloc[:-1]

    model = RandomForestRegressor()
    model.fit(X, y)

    last_row = df.iloc[-1:]
    for _ in range(future_minutes):
        features = last_row[["Open", "High", "Low", "Volume", "Volatility"]]
        pred = model.predict(features)[0]
        new_row = last_row.copy()
        new_row["Close"] = pred
        new_row["Open"] = new_row["Close"]
        new_row["High"] = new_row["Close"]
        new_row["Low"] = new_row["Close"]
        new_row["Return"] = 0
        new_row["Volatility"] = df["Return"].rolling(5).std().iloc[-1]
        df = pd.concat([df, new_row])
        last_row = new_row

    pred_time = df.index[-1]
    return float(pred), pred_time

def get_signal(current, predicted, lower, upper):
    if predicted > current * (1 + upper / 100):
        return "BUY"
    elif predicted < current * (1 - lower / 100):
        return "SELL"
    else:
        return "HOLD"

def plot_volatility(df):
    df["Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Return"].rolling(5).std()
    df["ATR"] = (df["High"] - df["Low"]).rolling(14).mean()
    df["BollingerUpper"] = df["Close"].rolling(20).mean() + 2 * df["Close"].rolling(20).std()
    df["BollingerLower"] = df["Close"].rolling(20).mean() - 2 * df["Close"].rolling(20).std()
    df.dropna(inplace=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(df.index, df["Volatility"], label="Volatility")
    ax.plot(df.index, df["ATR"], label="ATR", linestyle="--")
    ax.fill_between(df.index, df["BollingerUpper"], df["BollingerLower"], color="gray", alpha=0.2, label="Bollinger Bands")
    ax.set_title("Volatility Indicators")
    ax.legend()
    st.pyplot(fig)

# ------------------ MAIN ------------------
df = fetch_ohlc(selected_coin, interval_choice)
predicted_price, predicted_time = predict(df.copy(), future_minutes)
current_price = df["Close"].iloc[-1]
local_tz = pytz.timezone(selected_timezone)
local_pred_time = predicted_time.tz_convert(local_tz)
formatted_time = local_pred_time.strftime("%Y-%m-%d %I:%M %p %Z")
signal = get_signal(current_price, predicted_price, gain_range[0], gain_range[1])

st.markdown(f"### Current Price: ${current_price:.2f}")
st.markdown(f"### Predicted Price: ${predicted_price:.2f} at {formatted_time}")
st.markdown(f"### Signal: {'🟢 BUY' if signal=='BUY' else '🔴 SELL' if signal=='SELL' else '⚪ HOLD'}")

# ------------------ CANDLESTICK CHART ------------------
arrow_color = {"BUY": "green", "SELL": "red"}.get(signal, None)
apds = []
if arrow_color:
    last_idx = df.index[-1]
    apds = [mpf.make_addplot(
        [np.nan] * (len(df) - 1) + [predicted_price],
        type='scatter',
        markersize=200,
        marker='^' if signal == 'BUY' else 'v',
        color=arrow_color
    )]

fig1, _ = mpf.plot(
    df[-60:],
    type='candle',
    volume=True,
    style='yahoo',
    title=f"Candlestick Chart - {selected_coin} ({interval_choice})",
    returnfig=True,
    addplot=apds
)
st.pyplot(fig1)

# ------------------ VOLATILITY CHART ------------------
plot_volatility(df)

# ------------------ AUTO REFRESH ------------------
if "last_refresh" not in st.session_state:
    st.session_state.last_refresh = datetime.now()

if (datetime.now() - st.session_state.last_refresh).seconds > REFRESH_INTERVAL:
    st.session_state.last_refresh = datetime.now()
    st.experimental_rerun()
