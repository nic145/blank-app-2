import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
import pytz

# ========== Settings ==========
COINS = {
    "BTC": "XBTUSD",
    "ETH": "ETHUSD",
    "ADA": "ADAUSD",
    "XRP": "XRPUSD",
    "SOL": "SOLUSD",
    "ETC": "ETCUSD"
}
TIMEZONE = 'US/Eastern'  # Change to your local timezone

# ========== Utility Functions ==========

@st.cache_data(ttl=300)
def fetch_ohlc(coin_pair, interval='60'):
    url = f'https://api.kraken.com/0/public/OHLC?pair={coin_pair}&interval={interval}'
    response = requests.get(url).json()
    try:
        key = list(response['result'].keys())[0]
        df = pd.DataFrame(response['result'][key],
                          columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = df['close'].astype(float)
        return df
    except:
        return None

def get_latest_price(df):
    return df['close'].iloc[-1] if df is not None else None

def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

def generate_signals(df, gain_thresh):
    signals = []
    for i in range(1, len(df)):
        gain = (df['close'].iloc[i] - df['close'].iloc[i-1]) / df['close'].iloc[i-1]
        if gain >= gain_thresh:
            signals.append("SELL")
        elif gain <= -gain_thresh:
            signals.append("BUY")
        else:
            signals.append("")
    signals.insert(0, "")
    df['Signal'] = signals
    return df

def predict_prices(df):
    df = df.dropna().copy()
    df['return'] = df['close'].pct_change()
    df.dropna(inplace=True)

    for lag in range(1, 6):
        df[f'lag_{lag}'] = df['return'].shift(lag)

    df.dropna(inplace=True)

    X = df[[f'lag_{i}' for i in range(1, 6)]]
    y = df['close']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    last_data = X.iloc[-1].values.reshape(1, -1)
    predictions = {}

    for minutes_ahead in [15, 30, 60]:
        future_time = datetime.now(pytz.timezone(TIMEZONE)) + timedelta(minutes=minutes_ahead)
        future_price = model.predict(last_data)[0]
        predictions[minutes_ahead] = (future_price, future_time.strftime('%Y-%m-%d %I:%M %p %Z'))

    return predictions

# ========== Streamlit App ==========

st.set_page_config(page_title="Crypto Streamlit AI", layout="wide")

st.title("🧠 Crypto Price AI + Scalp Signals")
st.markdown("Made with ❤️ using Kraken + Streamlit")

# Coin Selection and Gain Toggle
coin = st.selectbox("Choose a coin:", list(COINS.keys()))
gain_thresh = st.slider("Scalp % Gain Threshold", 0.001, 0.05, 0.01, 0.001)
refresh_prices = st.button("🔄 Refresh Prices")
refresh_predictions = st.button("🔁 Refresh Predictions")

# Data Fetch
pair = COINS[coin]
df = fetch_ohlc(pair)

# Price Display
if df is not None:
    price = get_latest_price(df)
    local_time = datetime.now(pytz.timezone(TIMEZONE)).strftime('%Y-%m-%d %I:%M %p %Z')
    st.metric(label=f"📈 Current {coin} Price", value=f"${price:,.2f}")
    st.caption(f"Last Updated: {local_time}")
else:
    st.error("Failed to fetch OHLCV data.")
    st.stop()

# Add Indicators and Signals
df = calculate_indicators(df)
df = generate_signals(df, gain_thresh)

# Chart
st.subheader("📊 Price Chart with SMA/EMA + Buy/Sell")
st.line_chart(df.set_index('time')[['close', 'SMA_20', 'EMA_20']])

# Signal Table
st.dataframe(df[['time', 'close', 'Signal']].tail(10), use_container_width=True)

# Predictions
if refresh_predictions or refresh_prices:
    st.subheader("🔮 AI Price Predictions (Local Time)")
    preds = predict_prices(df)
    for k, (price, ts) in preds.items():
        st.write(f"**{k} min** → ${price:.2f} at *{ts}*")

