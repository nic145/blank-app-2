import streamlit as st
import pandas as pd
import numpy as np
import time
import os
import datetime as dt
from ta import add_all_ta_features
from ta.utils import dropna
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler
import requests
import json

# === CONFIG ===
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")

# === LOCAL STORAGE ===
DATA_DIR = "./data"
os.makedirs(DATA_DIR, exist_ok=True)

# === STATE INIT ===
if 'custom_coin' not in st.session_state:
    st.session_state['custom_coin'] = ''
if 'selected_coin' not in st.session_state:
    st.session_state['selected_coin'] = 'BTC/USDT'
if 'show_notifications' not in st.session_state:
    st.session_state['show_notifications'] = True
if 'prediction_timeframe' not in st.session_state:
    st.session_state['prediction_timeframe'] = 15

# === HEADER ===
st.title("📈 Crypto AI Dashboard")
st.markdown(f"**Time (Local):** {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# === SIDEBAR: Signal & Settings ===
with st.sidebar:
    st.header("🔔 AI Prediction Signals")
    st.session_state['show_notifications'] = st.checkbox("Enable Notifications", True)
    leverage = st.slider("Leverage", 1, 50, 25)
    capital = st.number_input("Capital ($)", value=200)
    gain_threshold = st.slider("Scalp Gain %", 0.5, 50.0, 10.0)

    st.markdown("---")
    st.header("⚙️ Settings")
    main_coins = ["BTC/USDT", "ETH/USDT", "ADA/USDT", "XRP/USDT", "SOL/USDT", "AVAX/USDT", "DOGE/USDT"]
    selected = st.selectbox("Choose a coin", main_coins)
    custom = st.text_input("Or add custom (e.g. TRX/USDT)")
    if custom:
        st.session_state['selected_coin'] = custom
    else:
        st.session_state['selected_coin'] = selected

    st.session_state['prediction_timeframe'] = st.radio(
        "Prediction Timeframe",
        [15, 30, 60, 240], format_func=lambda x: f"{x} min"
    )

# === FETCHING DATA ===
@st.cache_data
def fetch_ohlc_data(symbol, interval='15'):
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol.replace('/', '')}&interval={interval}"
    try:
        response = requests.get(url)
        raw = response.json()
        pair_key = list(raw['result'].keys())[0]
        df = pd.DataFrame(raw['result'][pair_key], columns=[
            'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
        ])
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df = df.set_index('time')
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# === PREDICTION MODEL ===
def create_model():
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(10, 1)))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def train_and_predict(df, steps_ahead=1):
    data = df[['close']].copy()
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(10, len(scaled) - steps_ahead):
        X.append(scaled[i-10:i, 0])
        y.append(scaled[i + steps_ahead, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    model = create_model()
    model.fit(X, y, epochs=5, batch_size=16, verbose=0)

    last_sequence = scaled[-10:].reshape((1, 10, 1))
    prediction_scaled = model.predict(last_sequence)
    predicted_price = scaler.inverse_transform(prediction_scaled)[0][0]
    return float(predicted_price)

# === MAIN PANEL ===
coin = st.session_state['selected_coin']
interval = '15'
df = fetch_ohlc_data(coin.replace('/', ''), interval)

if not df.empty:
    current_price = df['close'].iloc[-1]
    predicted_price = train_and_predict(df, steps_ahead=int(st.session_state['prediction_timeframe'] // 15))

    price_diff = predicted_price - current_price
    signal = 'Buy' if price_diff > 0 else 'Sell' if price_diff < 0 else 'Hold'

    # === SIMULATED SCALP GAIN ===
    is_long = signal.lower() == 'buy'
    price_movement = predicted_price - current_price
    pct_change = (price_movement / current_price) * leverage * (1 if is_long else -1)
    dollar_gain = (capital * pct_change / 100)

    # === PRICE DISPLAY ===
    st.subheader(f"🔍 {coin}")
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${current_price:,.2f}")
    col2.metric("Predicted", f"${predicted_price:,.2f}", delta=f"{price_diff:+.2f}")
    col3.metric("Signal", signal)

    st.markdown("---")

    st.markdown("### 💸 Simulated Trade Outcome")
    st.write(f"Leverage: {leverage}x | Capital: ${capital}")
    st.success(f"Estimated {'Gain' if dollar_gain >= 0 else 'Loss'}: ${dollar_gain:,.2f} ({pct_change:.2f}%)")

    # === CHART ===
    st.markdown("### 📊 TradingView Chart")
    symbol_code = coin.replace('/', '')
    st.components.v1.html(f"""
        <iframe src="https://www.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol_code}&symbol=KRAKEN:{symbol_code}&interval=15&symboledit=1&saveimage=1&toolbarbg=f1f3f6&studies=[]&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1&hide_side_toolbar=0&allow_symbol_change=1&details=1&hotlist=1&calendar=1&news=1&locale=en"
        width="100%" height="550" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
    """, height=550)

    # === BACKTEST PLACEHOLDER ===
    st.markdown("### 🧪 Backtest Results (coming soon)")
    st.info("Backtest UI and local CSV-based learning are under development.")

else:
    st.error("Unable to fetch OHLC data. Please check symbol format or try again later.")
