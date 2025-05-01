import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import time
import os
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator

# === Configuration ===
DATA_DIR = "data"
MODEL_DIR = "models"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

TOP_COINS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "BCH/USDT", "ADA/USDT", "DOT/USDT"]

# === Streamlit Setup ===
st.set_page_config(page_title="AI Crypto Dashboard", layout="wide")
st.title("🤖 AI Crypto Dashboard")

# === Sidebar Settings ===
st.sidebar.title("⚙️ Settings")
selected_coin = st.sidebar.selectbox("Top Coins", TOP_COINS)
custom_coin = st.sidebar.text_input("Or Custom Coin (e.g., SOL/USDT)").upper()
if custom_coin:
    selected_coin = custom_coin

prediction_timeframe = st.sidebar.selectbox("Prediction Timeframe", ["15m", "30m", "1h", "4h"])
leverage = st.sidebar.slider("Leverage", 1, 50, 25)
position_size = st.sidebar.number_input("Position Size ($)", min_value=50, value=200, step=50)

st.sidebar.markdown("### 🔔 Notification Toggles")
notify_buy = st.sidebar.checkbox("BUY Alerts", True)
notify_sell = st.sidebar.checkbox("SELL Alerts", True)
notify_hold = st.sidebar.checkbox("HOLD Alerts", False)

# === Helper Functions ===
def fetch_ohlc(coin_pair):
    symbol = coin_pair.replace("/", "")
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval=1"
    try:
        res = requests.get(url).json()
        key = next(k for k in res['result'].keys() if k != 'last')
        ohlc = pd.DataFrame(res['result'][key], columns=["Time", "Open", "High", "Low", "Close", "Vwap", "Volume", "Count"])
        ohlc["Time"] = pd.to_datetime(ohlc["Time"], unit="s")
        ohlc.set_index("Time", inplace=True)
        ohlc = ohlc.astype(float)
        return ohlc
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

def train_lstm_model(data):
    df = data.copy()
    df["Target"] = df["Close"].shift(-1)
    df.dropna(inplace=True)
    X = df[["Close", "High", "Low", "Open", "Volume"]].values
    y = df["Target"].values
    generator = TimeseriesGenerator(X, y, length=10, batch_size=1)

    model = Sequential()
    model.add(LSTM(64, activation="relu", input_shape=(10, 5)))
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")
    model.fit(generator, epochs=5, verbose=0)
    return model

def calculate_gain_loss(entry_price, predicted_price, capital, leverage):
    price_change = (predicted_price - entry_price) / entry_price
    return round(capital * leverage * price_change, 2)

def send_discord_alert(signal_type, coin, price, gain):
    if signal_type == "BUY" and not notify_buy: return
    if signal_type == "SELL" and not notify_sell: return
    if signal_type == "HOLD" and not notify_hold: return
    st.write(f"📤 [Discord] {signal_type} alert for {coin} at ${price:.2f} | Est. P/L: ${gain:.2f}")
    # To send: requests.post(WEBHOOK_URL, json={"content": message})

# === Load Data ===
df = fetch_ohlc(selected_coin)
if df.empty:
    st.stop()

symbol = selected_coin.replace("/", "")
current_price = df["Close"].iloc[-1]
model_path = os.path.join(MODEL_DIR, f"{symbol}_lstm.pkl")

# === Load or Train Model ===
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = train_lstm_model(df)
    joblib.dump(model, model_path)

# === Make Prediction ===
last_input = df[["Close", "High", "Low", "Open", "Volume"]].values[-10:]
last_input = last_input.reshape((1, 10, 5))
predicted_price = float(model.predict(last_input)[0][0])
price_diff = predicted_price - current_price

# === Generate Signal ===
signal = "HOLD"
if price_diff > current_price * 0.0015:
    signal = "BUY"
elif price_diff < -current_price * 0.0015:
    signal = "SELL"

est_gain = calculate_gain_loss(current_price, predicted_price, position_size, leverage)

# === Display Metrics ===
st.markdown(f"**🕒 Local Time:** `{datetime.datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}`")
st.metric("Current Price", f"${current_price:,.2f}")
st.metric("Predicted Price", f"${predicted_price:,.2f}")
st.metric("Signal", signal)
st.metric("Est. P/L", f"${est_gain:,.2f}")

# === Trigger Discord Alert ===
send_discord_alert(signal, selected_coin, current_price, est_gain)

# === TradingView Embed ===
st.components.v1.html(f"""
<iframe src="https://s.tradingview.com/widgetembed/?symbol=KRAKEN:{symbol}&interval=60&hidesidetoolbar=1&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc/UTC&withdateranges=1&hidevolume=0&allow_symbol_change=true&details=true&hotlist=true&calendar=true&news=stock&autosize=true"
width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
""", height=500)

# === Future Feature Placeholder ===
st.subheader("🔄 Backtesting (Coming Soon)")
st.info("You'll be able to upload a CSV and evaluate historical strategy performance.")
