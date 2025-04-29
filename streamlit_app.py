import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import krakenex
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime, timedelta

# --- Kraken API Setup ---
k = krakenex.API()

# --- Functions ---
def fetch_ohlcv(symbol, interval, lookback_days=30):
    pair = f"{symbol}USD"
    since = int((datetime.now() - timedelta(days=lookback_days)).timestamp() * 1e9)
    try:
        ohlc_data = k.query_public('OHLC', {'pair': pair, 'interval': interval, 'since': since})
        result = list(ohlc_data['result'].values())[0]
        df = pd.DataFrame(result, columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df['close'] = df['close'].astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return None

def add_indicators(df):
    df['SMA_20'] = df['close'].rolling(window=20).mean()
    df['SMA_50'] = df['close'].rolling(window=50).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['EMA_50'] = df['close'].ewm(span=50, adjust=False).mean()

    df['signal'] = 0
    df.loc[df['EMA_20'] > df['EMA_50'], 'signal'] = 1
    df.loc[df['EMA_20'] < df['EMA_50'], 'signal'] = -1
    return df

def predict_next(df):
    df['returns'] = df['close'].pct_change()
    df = df.dropna()
    X = df[['returns']]
    y = df['close']
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X, y)
    latest_return = df['returns'].iloc[-1]
    prediction = model.predict(np.array([[latest_return]]))[0]
    return prediction

def calculate_gain_percentages(df):
    trades = []
    last_buy_price = None
    for i in range(1, len(df)):
        if df.iloc[i]['signal'] == 1 and last_buy_price is None:
            last_buy_price = df.iloc[i]['close']
            buy_time = df.iloc[i]['time']
        elif df.iloc[i]['signal'] == -1 and last_buy_price is not None:
            sell_price = df.iloc[i]['close']
            sell_time = df.iloc[i]['time']
            gain = ((sell_price - last_buy_price) / last_buy_price) * 100
            trades.append({'buy_time': buy_time, 'sell_time': sell_time, 'gain': gain})
            last_buy_price = None
    return trades

# --- Streamlit UI ---
st.set_page_config(page_title="Kraken Crypto Scalper", layout="wide")
st.title("🚀 Kraken Crypto Scalper")

coins = ['BTC', 'ETH', 'SOL', 'XRP', 'ADA', 'ETC']
selected_coin = st.selectbox("Choose a cryptocurrency:", coins)

# Timeframe selection
interval_map = {"1 Hour": "60", "4 Hours": "240", "1 Day": "1440"}
selected_tf = st.selectbox("Timeframe:", list(interval_map.keys()))
interval = interval_map[selected_tf]

# SMA/EMA Toggles
col1, col2, col3, col4 = st.columns(4)
show_sma20 = col1.toggle("Show SMA 20", True)
show_sma50 = col2.toggle("Show SMA 50", False)
show_ema20 = col3.toggle("Show EMA 20", True)
show_ema50 = col4.toggle("Show EMA 50", False)

# Buttons
refresh_data = st.button("🔁 Refresh Price Data")
refresh_prediction = st.button("🎯 Refresh Prediction")
show_gains = st.toggle("📊 Show Gain % Between Trades", value=False)

# --- Fetch or refresh data ---
if 'price_data' not in st.session_state or refresh_data or st.session_state.get('last_interval') != interval:
    df = fetch_ohlcv(selected_coin, interval)
    if df is not None:
        df = add_indicators(df)
        st.session_state['price_data'] = df
        st.session_state['last_interval'] = interval
else:
    df = st.session_state['price_data']

# --- Live Price ---
if df is not None and not df.empty:
    latest_price = df['close'].iloc[-1]
    st.metric(label=f"💰 Latest {selected_coin}/USD Price", value=f"${latest_price:,.2f}")
else:
    st.error("❌ No price data available.")

# --- Plot Chart ---
if df is not None and not df.empty:
    st.subheader("📈 Price Chart")

    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df['time'], df['close'], label='Close', color='blue')
    if show_sma20:
        ax.plot(df['time'], df['SMA_20'], label='SMA 20', color='orange')
    if show_sma50:
        ax.plot(df['time'], df['SMA_50'], label='SMA 50', color='brown')
    if show_ema20:
        ax.plot(df['time'], df['EMA_20'], label='EMA 20', color='green')
    if show_ema50:
        ax.plot(df['time'], df['EMA_50'], label='EMA 50', color='red')

    buy_signals = df[df['signal'] == 1]
    sell_signals = df[df['signal'] == -1]
    ax.scatter(buy_signals['time'], buy_signals['close'], marker='^', color='lime', label='BUY', s=100)
    ax.scatter(sell_signals['time'], sell_signals['close'], marker='v', color='red', label='SELL', s=100)

    if show_gains:
        trades = calculate_gain_percentages(df)
        for trade in trades:
            mid_time = trade['buy_time'] + (trade['sell_time'] - trade['buy_time']) / 2
            mid_price = df[(df['time'] >= trade['buy_time']) & (df['time'] <= trade['sell_time'])]['close'].mean()
            gain_color = 'green' if trade['gain'] > 0 else 'red'
            ax.text(mid_time, mid_price, f"{trade['gain']:.2f}%", color=gain_color, fontsize=8, ha='center')

    ax.legend()
    ax.set_xlabel("Time")
    ax.set_ylabel("Price (USD)")
    st.pyplot(fig)
else:
    st.warning("Please refresh to load data.")

# --- Prediction ---
if df is not None and not df.empty:
    if 'price_prediction' not in st.session_state or refresh_prediction:
        st.session_state['price_prediction'] = predict_next(df)
    st.subheader("🔮 AI Price Prediction")
    st.metric(label="Next Predicted Price", value=f"${st.session_state['price_prediction']:.2f}")
