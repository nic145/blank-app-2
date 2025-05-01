import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense

st.set_page_config(layout="wide")

# Constants
INTERVALS = {
    "1 Minute": "1",
    "5 Minutes": "5",
    "15 Minutes": "15",
    "1 Hour": "60",
    "4 Hours": "240",
    "1 Day": "1440"
}

DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1367114033421094983/ZrE7E_ule4aOQHR-rEc8zfnSAIxHLvDO88tzIhIegCulIBKtQDmIMYBc8rpps2B4gnYp"

def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def fetch_ohlcv(pair, interval="60"):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    response = requests.get(url)
    data = response.json()
    if not data["error"]:
        ohlc = list(data["result"].values())[0]
        df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
        df["time"] = pd.to_datetime(df["time"], unit='s')
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df
    return pd.DataFrame()

def train_model(df):
    data = df["close"].values.reshape(-1, 1)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    X_train, y_train = [], []
    for i in range(60, len(scaled_data)):
        X_train.append(scaled_data[i-60:i])
        y_train.append(scaled_data[i])
    model = create_model()
    model.fit(np.array(X_train), np.array(y_train), epochs=5, batch_size=1, verbose=0)
    return model, scaler

def predict_price(model, df, scaler):
    last_60 = df["close"].values[-60:].reshape(-1, 1)
    scaled = scaler.transform(last_60)
    X_test = np.array([scaled])
    predicted = model.predict(X_test, verbose=0)
    return scaler.inverse_transform(predicted)[0][0]

# Sidebar Settings
st.sidebar.title("Settings")

top_coins = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "DOGE/USDT", "SOL/USDT", "LTC/USDT"]
selected_coin = st.sidebar.selectbox("Select Top Coin", top_coins)
custom_coin = st.sidebar.text_input("Or Enter Custom Pair (e.g., MATIC/USDT)").upper()
if custom_coin:
    selected_coin = custom_coin

selected_interval = st.sidebar.selectbox("Candlestick Interval", list(INTERVALS.keys()), index=3)
gain_threshold = st.sidebar.slider("Set Gain Threshold (%)", min_value=0, max_value=50, value=10)
leverage = st.sidebar.number_input("Leverage (x)", min_value=1, max_value=100, value=25)
capital = st.sidebar.number_input("Capital ($)", min_value=10, value=200)
enable_discord = st.sidebar.checkbox("Send Discord Alert (BUY/SELL only)")

# Main Header
st.title("📈 Crypto AI Dashboard")
st.markdown(f"**Selected Coin:** `{selected_coin}`")
st.markdown(f"**Interval:** `{selected_interval}` — **Gain Threshold:** `{gain_threshold}%`")

# Fetch and predict
pair = selected_coin.replace("/", "")
interval_code = INTERVALS[selected_interval]
df = fetch_ohlcv(pair, interval_code)

if not df.empty and len(df) > 60:
    model, scaler = train_model(df)
    predicted_price = predict_price(model, df, scaler)
    current_price = df["close"].iloc[-1]
    timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')

    # Signal logic
    signal = "HOLD"
    gain_pct = ((predicted_price - current_price) / current_price) * 100
    if gain_pct >= gain_threshold:
        signal = "BUY"
    elif gain_pct <= -gain_threshold:
        signal = "SELL"

    # Simulated gain/loss
    if signal == "BUY":
        simulated_gain = (gain_pct / 100) * leverage * capital
    elif signal == "SELL":
        simulated_gain = (-gain_pct / 100) * leverage * capital
    else:
        simulated_gain = 0

    # Display
    st.metric(label="Current Price", value=f"${current_price:.2f}")
    st.metric(label="Predicted Price", value=f"${predicted_price:.2f}", delta=f"{gain_pct:.2f}%")
    st.metric(label="Signal", value=signal)
    st.metric(label="Simulated Gain/Loss", value=f"${simulated_gain:.2f}")

    # Discord alert
    if enable_discord and signal in ["BUY", "SELL"]:
        message = {
            "content": f"**{signal} SIGNAL:** {selected_coin}\nPrice: ${current_price:.2f}\nPredicted: ${predicted_price:.2f}\nGain: {gain_pct:.2f}%\nSimulated Gain: ${simulated_gain:.2f}"
        }
        try:
            requests.post(DISCORD_WEBHOOK_URL, json=message)
        except Exception as e:
            st.warning(f"Discord alert failed: {e}")

    # TradingView Chart
    st.subheader("📊 TradingView Chart")
    st.components.v1.html(f"""
    <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{pair}&symbol=KRAKEN:{pair}&interval={interval_code}&theme=dark&style=1&locale=en&toolbar_bg=rgba(0, 0, 0, 1)&enable_publishing=false&hide_side_toolbar=false&allow_symbol_change=true" width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """, height=500)

else:
    st.warning("Not enough data to generate prediction.")

# Backtest placeholder
st.subheader("🧪 Backtest (Coming Soon)")
st.markdown("Backtest UI for historical predictions is under development.")

# Credits
st.markdown("---")
st.markdown("Made with ❤️ using Kraken API, TensorFlow, and TradingView.")
