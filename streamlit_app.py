import streamlit as st
import pandas as pd
import numpy as np
import requests
import datetime as dt
import os
import json
import time

from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator

# App settings
st.set_page_config(page_title="Crypto AI Dashboard", layout="wide")
st.title("📈 Real-Time Crypto AI Dashboard")
st.markdown("**Made with love & AI · Live Kraken Data**")

# Storage
DATA_DIR = "data/crypto_ai_data"
os.makedirs(DATA_DIR, exist_ok=True)

# Top 7 + custom coin input
top_coins = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "DOGE/USDT", "DOT/USDT"]
selected_coin = st.selectbox("Select Coin", top_coins)
custom_coin = st.text_input("Or Enter Custom Kraken Pair (e.g. LTC/USDT)", "")
symbol = custom_coin if custom_coin else selected_coin

# Prediction timeframe
prediction_timedelta = st.select_slider("Predict Ahead", options=["15m", "30m", "1h", "4h"], value="1h")

# Notification toggles
with st.sidebar.expander("🔔 Discord Notification Settings"):
    notify_buy = st.checkbox("Notify on BUY", value=True)
    notify_sell = st.checkbox("Notify on SELL", value=True)
    notify_hold = st.checkbox("Notify on HOLD", value=False)

# Gain thresholds
min_gain = st.slider("Scalp Min Gain (%)", 1, 50, 10)
max_gain = st.slider("Scalp Max Gain (%)", 1, 50, 15)

# TradingView chart embed
def render_tradingview_chart(symbol):
    base = symbol.replace("/", "").upper()
    st.components.v1.html(f"""
        <iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{base}&symbol=KRAKEN:{base}&interval=60&theme=dark&style=1&timezone=Etc%2FUTC&withdateranges=1"
        width="100%" height="500" frameborder="0"></iframe>
    """, height=500)

render_tradingview_chart(symbol)

# Kraken OHLC fetcher
def get_ohlc(pair, interval=1, since=None):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair.replace('/', '')}&interval={interval}"
    if since:
        url += f"&since={since}"
    try:
        response = requests.get(url).json()
        data = list(response["result"].values())[0]
        df = pd.DataFrame(data, columns=[
            "time", "open", "high", "low", "close", "vwap", "volume", "count"
        ])
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.set_index("time", inplace=True)
        df = df.astype(float)
        return df
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()

# Load and cache data
@st.cache_data(ttl=60)
def load_data(symbol, interval):
    df = get_ohlc(symbol, interval)
    if not df.empty:
        df["returns"] = df["close"].pct_change()
        df["volatility"] = df["returns"].rolling(window=10).std()
        bb = BollingerBands(close=df["close"])
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_lower"] = bb.bollinger_lband()
        macd = MACD(close=df["close"])
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        rsi = RSIIndicator(close=df["close"])
        df["rsi"] = rsi.rsi()
        return df.dropna()
    return pd.DataFrame()

data = load_data(symbol, 1)

# Save locally
csv_path = os.path.join(DATA_DIR, f"{symbol.replace('/', '_')}.csv")
if not data.empty:
    data.to_csv(csv_path)

# Features and target
def create_features(df):
    df = df.copy()
    df["target"] = df["close"].shift(-1)
    return df.dropna()

df_feat = create_features(data)

# Prepare for ML
features = ["close", "rsi", "macd", "macd_signal", "bb_upper", "bb_lower", "volatility"]
X = df_feat[features].values
y = df_feat["target"].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split & generate sequences
generator = TimeseriesGenerator(X_scaled, y, length=10, batch_size=1)

# Train model
model = Sequential()
model.add(LSTM(64, input_shape=(10, X_scaled.shape[1]), return_sequences=False))
model.add(Dense(1))
model.compile(loss="mse", optimizer="adam")
model.fit(generator, epochs=3, verbose=0)

# Predict future price
latest_seq = X_scaled[-10:].reshape(1, 10, X_scaled.shape[1])
predicted_price = model.predict(latest_seq)[0][0]
current_price = df_feat["close"].iloc[-1]
prediction_time = df_feat.index[-1] + pd.Timedelta(minutes=prediction_time_map[predict_minutes])

# Signal logic
if predicted_price > current_price * (1 + scalp_pct / 100):
    signal = "BUY"
elif predicted_price < current_price * (1 - scalp_pct / 100):
    signal = "SELL"
else:
    signal = "HOLD"

# Price target estimation
scalp_entry_price = current_price
scalp_exit_price = predicted_price if signal != "HOLD" else np.nan

# Notification toggle logic
alerts_enabled = notify_buy if signal == "BUY" else notify_sell if signal == "SELL" else notify_hold

# Discord webhook notification
def send_discord_alert(signal, coin, current_price, predicted_price):
    if not alerts_enabled:
        return
    msg = {
        "content": f"**{signal} ALERT for {coin}**\nCurrent: ${current_price:.2f}\nPredicted: ${predicted_price:.2f}\nTarget: {prediction_time.strftime('%Y-%m-%d %H:%M:%S')}"
    }
    try:
        requests.post(DISCORD_WEBHOOK_URL, json=msg)
    except Exception as e:
        st.warning(f"Discord alert failed: {e}")

send_discord_alert(signal, symbol, current_price, predicted_price)

# UI Layout
st.markdown(f"### {symbol} Price Dashboard ({time.tzname[0]})")
col1, col2 = st.columns(2)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("AI Predicted Price", f"${predicted_price:.2f}", delta=f"{signal}")

# TradingView Chart
st.markdown("### Candlestick Chart")
tradingview_symbol = f"KRAKEN:{symbol.replace('/', '')}"
tradingview_html = f"""
<iframe src="https://s.tradingview.com/widgetembed/?frameElementId=tradingview_{symbol}&symbol={tradingview_symbol}&interval={interval}&theme=dark&style=1&locale=en" width="100%" height="500" frameborder="0" allowtransparency="true" scrolling="no"></iframe>
"""
st.components.v1.html(tradingview_html, height=500)

# Show predictions table
st.markdown("### Recent Predictions")
st.dataframe(pd.DataFrame({
    "Timestamp": [prediction_time],
    "Signal": [signal],
    "Current Price": [current_price],
    "Predicted Price": [predicted_price],
    "Target Price": [scalp_exit_price],
}).set_index("Timestamp"))

# Local model save
with open(MODEL_FILE, "wb") as f:
    pickle.dump(model, f)
with open(SCALER_FILE, "wb") as f:
    pickle.dump(scaler, f)

# Backtesting UI (placeholder)
st.markdown("### Backtesting")
uploaded = st.file_uploader("Upload historical CSV to backtest", type=["csv"])
if uploaded:
    try:
        df_backtest = pd.read_csv(uploaded, parse_dates=["time"])
        st.success("Backtest data loaded.")
        st.dataframe(df_backtest.tail())
    except Exception as e:
        st.error(f"Failed to read backtest data: {e}")

# --- AI Retrain Button ---
st.markdown("### Model Training")
if st.button("Retrain AI Model"):
    with st.spinner("Retraining..."):
        try:
            model.fit(X, y, epochs=10, verbose=0)
            with open(MODEL_FILE, "wb") as f:
                pickle.dump(model, f)
            st.success("Model retrained and saved locally.")
        except Exception as e:
            st.error(f"Error during retraining: {e}")

# --- Auto-Refresh every 60 seconds ---
def should_refresh():
    if "last_refresh" not in st.session_state:
        st.session_state.last_refresh = time.time()
        return False
    elapsed = time.time() - st.session_state.last_refresh
    if elapsed > 60:
        st.session_state.last_refresh = time.time()
        return True
    return False

if should_refresh():
    st.experimental_rerun()

# --- Optional Footer / Debug Info ---
st.markdown("---")
st.markdown(
    "📈 Made with ❤️ using Kraken API, TradingView charts, and AI predictions. "
    "All times shown in your local timezone."
)
