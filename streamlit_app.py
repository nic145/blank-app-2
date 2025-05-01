import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime
import pytz
import json
from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.preprocessing import MinMaxScaler

# Constants
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1367114033421094983/ZrE7E_ule4aOQHR-rEc8zfnSAIxHLvDO88tzIhIegCulIBKtQDmIMYBc8rpps2B4gnYp"
TOP_COINS = ['BTC', 'ETH', 'XRP', 'SOL', 'ADA', 'LTC', 'DOT']
TIMEZONE = 'US/Eastern'
PREDICTION_HORIZON = 8

st.set_page_config(layout="wide", page_title="Crypto AI Dashboard")

# Sidebar
st.sidebar.title("⚙️ Prediction Controls")
selected_coin = st.sidebar.selectbox("Choose a Coin", TOP_COINS)
prediction_hours = st.sidebar.slider("Prediction Time Horizon (hours)", 1, PREDICTION_HORIZON, 1)
scalp_gain = st.sidebar.slider("Scalp Gain Target (%)", 1, 50, 10)
scalp_loss = st.sidebar.slider("Scalp Loss Limit (%)", 1, 50, 10)

st.title("📈 Real-Time Crypto AI Dashboard")
symbol = f"{selected_coin}USDT"

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol):
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval=1"
    res = requests.get(url).json()
    try:
        key = list(res["result"].keys())[0]
        df = pd.DataFrame(res["result"][key], columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV: {e}")
        return pd.DataFrame()

def generate_features(df):
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['rsi'] = compute_rsi(df['close'], 14)
    df['macd'] = compute_macd(df['close'])
    df = df.dropna()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_macd(series):
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    return ema12 - ema26

def train_models(df, prediction_hours):
    feature_df = generate_features(df)
    X = feature_df[['close', 'returns', 'sma_10', 'sma_50', 'rsi', 'macd']].values
    y = feature_df['close'].shift(-prediction_hours).dropna().values

    # Align X and y
    X = X[:-prediction_hours]

    # Train RF
    rf = RandomForestRegressor()
    rf.fit(X, y)

    # Scale for LSTM
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[['close']].dropna())
    gen = TimeseriesGenerator(scaled, scaled, length=20, batch_size=1)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(20, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(gen, epochs=5, verbose=0)

    return rf, model, scaler, feature_df

def predict_next_price(df, rf, lstm, scaler, prediction_hours):
    latest_features = generate_features(df).iloc[-1][['close', 'returns', 'sma_10', 'sma_50', 'rsi', 'macd']].values.reshape(1, -1)
    rf_pred = rf.predict(latest_features)[0]

    recent_close = df[['close']].dropna().values[-20:]
    scaled_input = scaler.transform(recent_close)
    lstm_input = scaled_input.reshape((1, 20, 1))
    lstm_pred = scaler.inverse_transform(lstm.predict(lstm_input, verbose=0))[0][0]

    hybrid_pred = (rf_pred + lstm_pred) / 2
    return hybrid_pred

def send_discord_alert(signal, coin, current_price, predicted_price):
    color = "🟢" if signal == "BUY" else "🔴" if signal == "SELL" else "🟡"
    msg = {
        "content": f"{color} **{signal} ALERT for {coin}**\n"
                   f"Current Price: ${current_price:.2f}\n"
                   f"Predicted Price: ${predicted_price:.2f}"
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=msg)
    except Exception as e:
        st.warning(f"Discord alert failed: {e}")

# Main logic
df = fetch_ohlcv(symbol)
if not df.empty:
    rf_model, lstm_model, scaler, features = train_models(df, prediction_hours)
    predicted_price = predict_next_price(df, rf_model, lstm_model, scaler, prediction_hours)
    current_price = df['close'].iloc[-1]

    # Signal logic
    gain_pct = (predicted_price - current_price) / current_price * 100
    if gain_pct > scalp_gain:
        signal = "BUY"
    elif gain_pct < -scalp_loss:
        signal = "SELL"
    else:
        signal = "HOLD"

    send_discord_alert(signal, selected_coin, current_price, predicted_price)

    # Display results
    st.markdown(f"### 💰 Current Price of {selected_coin}: **${current_price:.2f}**")
    st.markdown(f"### 🤖 Predicted Price ({prediction_hours}h): **${predicted_price:.2f}**")
    st.markdown(f"### 📊 AI Signal: **{signal}**")

    # TradingView Embed
    st.markdown("---")
    st.components.v1.html(f"""
        <iframe src="https://s.tradingview.com/embed-widget/symbol-overview/?locale=en#%7B%22symbols%22%3A%5B%5B%22KRAKEN%3A{selected_coin}USDT%22%5D%5D%2C%22width%22%3A%22100%25%22%2C%22height%22%3A300%2C%22colorTheme%22%3A%22dark%22%2C%22isTransparent%22%3Afalse%7D"
                width="100%" height="300" frameborder="0"></iframe>
    """, height=300)

    # Optional: Display prediction horizon table
    st.markdown("### 📅 Upcoming Predictions")
    future_prices = []
    for h in range(1, PREDICTION_HORIZON + 1):
        pred = predict_next_price(df, rf_model, lstm_model, scaler, h)
        future_prices.append((h, pred))
    table = pd.DataFrame(future_prices, columns=['Hours Ahead', 'Predicted Price'])
    st.dataframe(table.set_index('Hours Ahead'), height=250)
