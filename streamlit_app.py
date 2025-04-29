import streamlit as st
import pandas as pd
import krakenex
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests

# Initialize Kraken API client
k = krakenex.API()

# Supported Kraken crypto pairs
CRYPTO_PAIRS = {
    "BTC": "XXBTZUSD",
    "ETH": "XETHZUSD",
    "ETC": "XETCZUSD",
    "SOL": "SOLUSD",
    "XRP": "XXRPZUSD",
    "ADA": "ADAUSD"
}

st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

st.title("💹 Crypto AI Dashboard (Powered by Kraken)")

# Sidebar
st.sidebar.header("Select Options")
selected_coin = st.sidebar.selectbox("Select a coin", list(CRYPTO_PAIRS.keys()))
scalping_toggle = st.sidebar.toggle("💥 Enable Scalping Predictions")

# Utility: Fetch historical OHLC data
@st.cache_data(show_spinner=False)
def fetch_ohlcv(pair, interval="60", since_hours=48):
    since = int((datetime.utcnow() - timedelta(hours=since_hours)).timestamp())
    res = k.query_public('OHLC', {'pair': pair, 'interval': interval, 'since': since})
    data = res['result'][pair]
    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close',
        'vwap', 'volume', 'count'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df[['open', 'high', 'low', 'close']] = df[['open', 'high', 'low', 'close']].astype(float)
    return df

# Add indicators
def calculate_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['EMA_20'] = df['close'].ewm(span=20).mean()
    return df

# Detect crossover signals
def detect_crossovers(df):
    signals = []
    for i in range(1, len(df)):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        if prev['SMA_20'] < prev['EMA_20'] and curr['SMA_20'] > curr['EMA_20']:
            signals.append((curr['timestamp'], curr['close'], "BUY"))
        elif prev['SMA_20'] > prev['EMA_20'] and curr['SMA_20'] < curr['EMA_20']:
            signals.append((curr['timestamp'], curr['close'], "SELL"))
    return signals

# Chart plotting
def plot_chart(df, signals):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df['timestamp'], open=df['open'], high=df['high'],
        low=df['low'], close=df['close'], name="Candles"
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['SMA_20'], name='SMA 20', line=dict(color='orange')
    ))
    fig.add_trace(go.Scatter(
        x=df['timestamp'], y=df['EMA_20'], name='EMA 20', line=dict(color='blue')
    ))

    for ts, price, signal in signals:
        fig.add_trace(go.Scatter(
            x=[ts], y=[price], mode='markers+text',
            marker=dict(color='green' if signal == 'BUY' else 'red', size=10),
            text=signal, name=signal
        ))

    fig.update_layout(
        title=f"{selected_coin} Price Chart with Indicators",
        xaxis_title="Time", yaxis_title="Price (USD)", height=600
    )
    return fig

try:
    pair = CRYPTO_PAIRS[selected_coin]
    df = fetch_ohlcv(pair)
    df = calculate_indicators(df)
    signals = detect_crossovers(df) if scalping_toggle else []

    st.subheader(f"📈 {selected_coin} Historical Chart")
    st.plotly_chart(plot_chart(df, signals), use_container_width=True)

    if signals:
        st.subheader("📍 Scalping Signals")
        for ts, price, signal in signals[-5:]:
            st.write(f"{signal} at {price:.2f} on {ts.strftime('%Y-%m-%d %H:%M:%S')}")

except Exception as e:
    st.error("🚨 Failed to load chart.")
    st.exception(e)
