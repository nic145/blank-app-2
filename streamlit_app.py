import streamlit as st
import pandas as pd
import numpy as np
import requests
import mplfinance as mpf
import time
import pytz
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor

# --- CONFIG ---
st.set_page_config("Crypto Dashboard", layout="wide")

# --- TIMEZONE HANDLING ---
all_timezones = pytz.all_timezones
default_tz = 'US/Eastern'
selected_tz = st.sidebar.selectbox("Select Timezone", all_timezones, index=all_timezones.index(default_tz))
local_tz = pytz.timezone(selected_tz)

# --- APP TITLE ---
st.title("🔮 Real-Time Crypto Dashboard with Predictions")

# --- COINS ---
top_coins = ['BTC', 'ETH', 'ADA', 'XRP', 'SOL', 'DOT', 'DOGE']
coin = st.selectbox("Choose a coin:", top_coins)
custom_coin = st.text_input("Or enter custom coin symbol (e.g., MATIC):")
coin = custom_coin.upper() if custom_coin else coin.upper()
pair = f"{coin}USDT"

# --- PREDICTION TIME SLIDER ---
hours_ahead = st.slider("Predict how far ahead (in hours):", 0.25, 8.0, 1.0, 0.25)
future_minutes = int(hours_ahead * 60)

# --- SCALP GAIN THRESHOLDS ---
col1, col2 = st.columns(2)
with col1:
    scalp_min = st.slider("Minimum gain for scalp (%)", 10, 15, 10)
with col2:
    scalp_max = st.slider("Maximum gain for scalp (%)", 10, 15, 15)

# --- SMA/EMA TOGGLES ---
col1, col2 = st.columns(2)
with col1:
    show_sma = st.checkbox("Show SMA", value=True)
with col2:
    show_ema = st.checkbox("Show EMA", value=True)

# --- HELPER: Fetch OHLCV Data from Kraken ---
@st.cache_data(ttl=60)
def fetch_ohlcv(pair, interval='60'):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair}&interval={interval}"
    r = requests.get(url).json()
    key = list(r['result'].keys())[0]
    df = pd.DataFrame(r['result'][key], columns=['time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'])
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df = df.astype(float)
    return df[['open', 'high', 'low', 'close', 'volume']]

# --- HELPER: Predict Future Price ---
def predict(df, future_minutes):
    df['return'] = df['close'].pct_change().fillna(0)
    df['sma'] = df['close'].rolling(window=5).mean().fillna(method='bfill')
    df['ema'] = df['close'].ewm(span=5).mean().fillna(method='bfill')
    df.dropna(inplace=True)

    X = df[['close', 'sma', 'ema']]
    y = df['close'].shift(-future_minutes)

    X, y = X.iloc[:-future_minutes], y.dropna()
    model = RandomForestRegressor()
    model.fit(X, y)

    latest_input = X.iloc[-1].values.reshape(1, -1)
    predicted_price = model.predict(latest_input)[0]
    predicted_time = df.index[-1] + timedelta(minutes=future_minutes)
    return predicted_price, predicted_time

# --- FETCH AND PREDICT ---
df = fetch_ohlcv(pair)
current_price = df['close'].iloc[-1]
predicted_price, predicted_time = predict(df.copy(), future_minutes)

# --- LOCALIZE TIMES ---
local_now = datetime.now(pytz.utc).astimezone(local_tz)
pred_time_local = predicted_time.astimezone(local_tz)
last_updated = local_now.strftime('%I:%M %p %Z')
pred_time_str = pred_time_local.strftime('%I:%M %p %Z')

# --- DISPLAY CURRENT/PREDICTED ---
st.subheader(f"📈 {coin}/USDT Price")
st.metric("Current Price", f"${current_price:,.4f}")
st.metric("Predicted Price", f"${predicted_price:,.4f}", f"at {pred_time_str}")

# --- SIGNAL ---
change_pct = ((predicted_price - current_price) / current_price) * 100
if change_pct > scalp_max:
    signal = "BUY"
elif change_pct < -scalp_max:
    signal = "SELL"
else:
    signal = "HOLD"
st.success(f"Signal: {signal} ({change_pct:.2f}%)")
st.caption(f"Last updated: {last_updated}")

# --- CANDLESTICK INTERVALS ---
intervals = {
    "1m": '1',
    "5m": '5',
    "15m": '15',
    "1h": '60',
    "4h": '240',
    "1d": '1440'
}
selected_label = st.radio("Candlestick Interval", list(intervals.keys()), horizontal=True)
interval = intervals[selected_label]
candles_df = fetch_ohlcv(pair, interval)

# --- CANDLESTICK CHART ---
st.subheader(f"Candlestick Chart - {coin}/USDT ({selected_label})")
mc = mpf.make_marketcolors(up='g', down='r', wick='inherit', edge='inherit')
s = mpf.make_mpf_style(base_mpf_style='default', marketcolors=mc, facecolor='white', edgecolor='black')

fig, _ = mpf.plot(
    candles_df,
    type='candle',
    volume=True,
    style=s,
    returnfig=True,
    figsize=(10, 6),
    tight_layout=True,
    datetime_format='%I:%M %p'
)
st.pyplot(fig)

# --- PRICE AND PREDICTION CHART ---
st.subheader("Price vs Predicted Price")
df_plot = df[['close']].copy()
df_plot['Predicted'] = np.nan
df_plot.iloc[-1, df_plot.columns.get_loc('Predicted')] = predicted_price
st.line_chart(df_plot)

# --- MANUAL REFRESH BUTTONS ---
col1, col2 = st.columns(2)
with col1:
    if st.button("🔁 Refresh Price"):
        st.rerun()
with col2:
    if st.button("🔮 Refresh Prediction"):
        st.rerun()
