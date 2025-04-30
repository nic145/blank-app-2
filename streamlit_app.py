import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import datetime
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import json

# Webhook URL
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1367114033421094983/ZrE7E_ule4aOQHR-rEc8zfnSAIxHLvDO88tzIhIegCulIBKtQDmIMYBc8rpps2B4gnYp"

st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.title("📈 Real-Time Crypto AI Dashboard")
st.caption("Powered by Kraken + Hybrid AI (Random Forest + LSTM)")

# -------------------------------
# FETCH DATA FROM KRAKEN API
# -------------------------------

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol: str, interval="60", limit=100):
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}"
    response = requests.get(url)
    data = response.json()

    if "error" in data and data["error"]:
        st.error(f"Kraken API Error: {data['error']}")
        return pd.DataFrame()

    if "result" not in data:
        st.error("Invalid response from Kraken API.")
        return pd.DataFrame()

    key = list(data["result"].keys())[0]
    ohlc = data["result"][key]

    df = pd.DataFrame(ohlc, columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

# -------------------------------
# AI HYBRID MODEL: RF + LSTM
# -------------------------------

def prepare_lstm_data(df, n_steps=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']])
    X, y = [], []
    for i in range(len(scaled) - n_steps):
        X.append(scaled[i:i+n_steps])
        y.append(scaled[i+n_steps][0])
    return np.array(X), np.array(y), scaler

def predict_price(df, minutes_forward=60):
    df = df.copy().dropna()

    # Random Forest Model
    df['returns'] = df['close'].pct_change().fillna(0)
    df['volatility'] = df['returns'].rolling(5).std().fillna(0)
    features = df[['open', 'high', 'low', 'close', 'volume', 'returns', 'volatility']]
    X_rf = features.iloc[:-minutes_forward]
    y_rf = df['close'].shift(-minutes_forward).dropna()
    X_rf = X_rf.loc[y_rf.index]
    rf_model = RandomForestRegressor()
    rf_model.fit(X_rf, y_rf)
    rf_pred = rf_model.predict([features.iloc[-1]])[0]

    # LSTM Model
    X_lstm, y_lstm, scaler = prepare_lstm_data(df)
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X_lstm.shape[1], X_lstm.shape[2])),
        Dense(1)
    ])
    model.compile(optimizer=Adam(), loss='mse')
    model.fit(X_lstm, y_lstm, epochs=10, verbose=0)
    last_sequence = scaler.transform(df[['close']])[-10:]
    lstm_pred = model.predict(last_sequence.reshape(1, 10, 1), verbose=0)[0][0]
    lstm_pred = scaler.inverse_transform([[lstm_pred]])[0][0]

    # Final hybrid prediction
    predicted = (rf_pred + lstm_pred) / 2
    return round(predicted, 2), datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# -------------------------------
# DISCORD ALERT
# -------------------------------

def send_discord_alert(message: str):
    data = {"content": message}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code != 204:
            st.warning("Discord webhook error.")
    except Exception as e:
        st.error(f"Webhook error: {e}")

# -------------------------------
# UI COMPONENTS
# -------------------------------

symbol = st.selectbox("Select a Kraken trading pair", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT"])
time_forward = st.slider("⏩ Predict how far into the future (minutes)", 15, 480, 60, step=15)

df = fetch_ohlcv(symbol, interval="60")

if df.empty:
    st.stop()

current_price = df["close"].iloc[-1]
predicted_price, predicted_time = predict_price(df, minutes_forward=time_forward)

# Determine signal
if predicted_price > current_price * 1.01:
    signal = "BUY"
    dot_color = "green"
elif predicted_price < current_price * 0.99:
    signal = "SELL"
    dot_color = "red"
else:
    signal = "HOLD"
    dot_color = "yellow"

# Send Discord alert (once per run or on button press)
if st.button("🔔 Send Discord Alert"):
    send_discord_alert(f"Signal for {symbol}: **{signal}**\nCurrent: ${current_price:.2f}\nPredicted: ${predicted_price:.2f}")

# Display prices and signal
st.markdown(f"""
### 💰 Current Price: `${current_price:.2f}`  
### 🤖 Predicted Price: `{predicted_price:.2f}` ({predicted_time})  
### 🔍 Signal: **:{dot_color}_circle: {signal}**
""")

# -------------------------------
# TRADINGVIEW CHART EMBED
# -------------------------------

tradingview_widget = f"""
<iframe src="https://www.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol}&symbol=KRAKEN:{symbol}&interval=60&theme=dark&style=1&locale=en&utm_source=streamlit" 
width="100%" height="600" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
"""

st.markdown("### 📊 Live Kraken Candlestick Chart")
st.components.v1.html(tradingview_widget, height=600)

st.caption("Made with ❤️ using Kraken, TradingView, and AI")
