import streamlit as st
import pandas as pd
import numpy as np
import requests
import threading
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

# Constants
COINS = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "LTC/USDT", "ADA/USDT", "DOT/USDT", "SOL/USDT"]
INTERVALS = {
    "1 Minute": "1",
    "5 Minutes": "5",
    "15 Minutes": "15",
    "1 Hour": "60",
    "4 Hours": "240",
    "1 Day": "1440"
}
REFRESH_INTERVAL = 300  # seconds
WEBHOOK_URL = "https://discord.com/api/webhooks/your_webhook_here"

st.set_page_config("Crypto Dashboard", layout="wide")

# Sidebar
selected_coin = st.selectbox("Choose a coin", COINS)
custom_coin = st.text_input("Or enter custom coin pair (e.g. MATIC/USDT)")
if custom_coin:
    selected_coin = custom_coin.upper()
selected_interval = st.radio("Interval", list(INTERVALS.keys()), index=3, horizontal=True)
discord_alerts = st.toggle("Send Discord BUY/SELL alerts", value=True)
show_chart = st.toggle("Show TradingView Chart", value=False)
threshold_percent = st.slider("Alert Threshold (%)", min_value=5, max_value=50, value=15, step=1)

# Main display
st.title("📈 Crypto Prediction Dashboard")
price_col, prediction_col = st.columns(2)
price_col.metric("Current Price", "Loading...")
prediction_col.metric("Predicted Price", "Loading...")

@st.cache_data(ttl=300)
def fetch_ohlcv(coin, interval):
    pair = coin.replace("/", "")
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    try:
        r = requests.get(url)
        data = list(r.json()["result"].values())[0]
        df = pd.DataFrame(data, columns=["Time", "Open", "High", "Low", "Close", "Vwap", "Volume", "Count"])
        df["Time"] = pd.to_datetime(df["Time"], unit="s")
        df["Close"] = pd.to_numeric(df["Close"])
        return df
    except Exception as e:
        st.error(f"Data error: {e}")
        return pd.DataFrame()

@st.cache_resource
def train_model(df):
    df = df[["Close"]]
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)
    X, y = [], []
    for i in range(60, len(df_scaled)):
        X.append(df_scaled[i-60:i, 0])
        y.append(df_scaled[i, 0])
    X, y = np.array(X), np.array(y)
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    return model, scaler

def predict_price(model, scaler, df):
    last_60 = df["Close"].values[-60:]
    scaled = scaler.transform(last_60.reshape(-1, 1))
    X = scaled.reshape(1, -1)
    pred_scaled = model.predict(X)
    predicted_price = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
    return round(predicted_price, 2)

def send_discord_alert(signal, coin, price):
    if discord_alerts:
        message = {
            "content": f"📢 **{signal} Signal** for `{coin}` at price **${price:.2f}**"
        }
        try:
            requests.post(WEBHOOK_URL, json=message)
        except:
            st.warning("Failed to send Discord alert.")

def show_tradingview_chart(symbol):
    tv_symbol = symbol.replace("/", "")
    st.components.v1.html(f'''
        <iframe src="https://www.tradingview.com/embed-widget/advanced-chart/?symbol=KRAKEN:{tv_symbol}" 
                width="100%" height="500" frameborder="0"></iframe>
    ''', height=500)

# Load data and model
df = fetch_ohlcv(selected_coin, INTERVALS[selected_interval])
if not df.empty:
    def model_thread_func():
        global model, scaler
        model, scaler = train_model(df)

    model_thread = threading.Thread(target=model_thread_func)
    model_thread.start()
    model_thread.join()

    predicted_price = predict_price(model, scaler, df)
    current_price = df["Close"].iloc[-1]
    change_percent = ((predicted_price - current_price) / current_price) * 100

    price_col.metric("Current Price", f"${current_price:.2f}")
    prediction_col.metric("Predicted Price", f"${predicted_price:.2f}", delta=f"{change_percent:.2f}%")

    if change_percent >= threshold_percent:
        send_discord_alert("BUY", selected_coin, predicted_price)
    elif change_percent <= -threshold_percent:
        send_discord_alert("SELL", selected_coin, predicted_price)

    if show_chart:
        show_tradingview_chart(selected_coin)
else:
    st.warning("No data to display.")
