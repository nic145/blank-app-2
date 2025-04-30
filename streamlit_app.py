import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
import mplfinance as mpf
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import pytz

# === CONFIG ===
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# === DISCORD WEBHOOK ===
WEBHOOK_URL = "https://discord.com/api/webhooks/1367114033421094983/ZrE7E_ule4aOQHR-rEc8zfnSAIxHLvDO88tzIhIegCulIBKtQDmIMYBc8rpps2B4gnYp"

def send_discord_alert(message):
    payload = {"content": message}
    try:
        requests.post(WEBHOOK_URL, json=payload)
    except Exception as e:
        st.warning(f"Failed to send Discord alert: {e}")

# === FUNCTIONS ===
def get_ohlc(coin, interval='60'):
    url = f"https://api.kraken.com/0/public/OHLC?pair={coin}&interval={interval}"
    resp = requests.get(url).json()
    key = list(resp['result'].keys())[0]
    data = resp['result'][key]
    df = pd.DataFrame(data, columns=[
        'Time', 'Open', 'High', 'Low', 'Close', 'VWAP', 'Volume', 'Count'])
    df['Time'] = pd.to_datetime(df['Time'], unit='s')
    df.set_index('Time', inplace=True)
    df = df.astype(float)
    return df

def calculate_volatility(df):
    df['Returns'] = df['Close'].pct_change()
    df['Volatility (%)'] = df['Returns'].rolling(window=10).std() * 100
    df['ATR'] = (df['High'] - df['Low']).rolling(window=14).mean()
    df['Price Change %'] = df['Close'].pct_change(periods=3) * 100
    return df

def predict(df, future_minutes=60):
    df = df[['Close']].copy()
    df['Target'] = df['Close'].shift(-1)
    df.dropna(inplace=True)
    X = np.arange(len(df)).reshape(-1, 1)
    y = df['Target'].values
    model = RandomForestRegressor()
    model.fit(X, y)

    next_index = len(df)
    future_index = np.array([[next_index + (future_minutes // 15)]])
    pred = model.predict(future_index)[0]
    pred_time = df.index[-1] + timedelta(minutes=future_minutes)
    return round(pred, 2), pred_time

def determine_signal(current_price, predicted_price, threshold):
    change_pct = (predicted_price - current_price) / current_price * 100
    if change_pct >= threshold:
        return "BUY", "🟢"
    elif change_pct <= -threshold:
        return "SELL", "🔴"
    else:
        return "HOLD", "⚪"

# === UI ===
st.title("💹 Real-Time Crypto Dashboard")

# Coin Selection
main_coins = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT']
coin = st.selectbox("Choose Coin Pair", main_coins)
custom_coin = st.text_input("Or enter a custom Kraken pair (e.g., LTCUSDT)")
pair = custom_coin.upper() if custom_coin else coin

# Sliders
threshold = st.slider("Scalp % Gain Threshold", 0.1, 50.0, 10.0)
future_minutes = st.slider("Prediction Horizon (minutes)", 15, 480, 60, step=15)

# Timezone Selection
user_tz = st.selectbox("Select Your Timezone", pytz.all_timezones, index=pytz.all_timezones.index("UTC"))
local_tz = pytz.timezone(user_tz)

# === DATA FETCHING ===
try:
    df = get_ohlc(pair, interval='60')
    df = calculate_volatility(df)
    current_price = df['Close'].iloc[-1]

    # === PREDICTION ===
    predicted_price, predicted_time = predict(df.copy(), future_minutes)
    predicted_time = predicted_time.replace(tzinfo=pytz.utc).astimezone(local_tz)
    timestamp = predicted_time.strftime("%I:%M %p %Z")

    # === SIGNAL ===
    signal, icon = determine_signal(current_price, predicted_price, threshold)

    # Send Discord Alert
    if signal in ["BUY", "SELL"]:
        send_discord_alert(f"{icon} {pair} {signal} signal triggered! Predicted price: ${predicted_price} at {timestamp}")

    # === CHART ===
    st.subheader(f"Candlestick Chart - {pair} (1h)")
    st.write(f"**Current Price:** ${current_price:.2f}   \n**Predicted Price:** ${predicted_price} at {timestamp}   \n**Signal:** {signal} {icon}")

    mpf_df = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ap = []
    if signal != "HOLD":
        arrow_color = 'green' if signal == 'BUY' else 'red'
        arrow = mpf.make_addplot([np.nan]*(len(mpf_df)-1)+[predicted_price], type='scatter', markersize=100, marker='^' if signal == 'BUY' else 'v', color=arrow_color)
        ap.append(arrow)

    fig, _ = mpf.plot(
        mpf_df,
        type='candle',
        volume=True,
        style='yahoo',  # light mode
        addplot=ap,
        returnfig=True,
        figsize=(12, 6)
    )
    st.pyplot(fig)

    # === VOLATILITY CHART ===
    st.subheader("Volatility Indicators")
    st.line_chart(df[['Volatility (%)', 'ATR', 'Price Change %']].dropna())

    # === AUTO REFRESH WORKAROUND ===
    if 'last_refresh' not in st.session_state:
        st.session_state.last_refresh = datetime.now()

    if (datetime.now() - st.session_state.last_refresh).seconds > 60:
        st.session_state.last_refresh = datetime.now()
        st.experimental_rerun()

except Exception as e:
    st.error(f"Something went wrong: {e}")
