
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="📱 Mobile Crypto Alerts",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Sound alert function
def play_sound():
    sound_url = "https://www.soundjay.com/buttons/sounds/beep-07.mp3"
    st.markdown(f'''
        <audio autoplay>
            <source src="{sound_url}" type="audio/mpeg">
        </audio>
    ''', unsafe_allow_html=True)

# Title + Description
st.title("📡 Crypto Signal Monitor")
st.caption("Live scalping alerts — optimized for mobile")

# Select coin
coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "MATIC"]
symbol = st.selectbox("Select Coin", coins, index=0, key="mobile_coin")

# Prediction timeframe and threshold
forecast_minutes = st.slider("Predict Minutes Ahead", 1, 30, 5, key="mobile_minutes")
alert_threshold = st.slider("Trigger Alert If Move > $", 0.5, 10.0, 2.0, step=0.5, key="mobile_threshold")

symbol_full = f"{symbol}/USDT"
exchange = ccxt.kraken()
try:
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol_full, "5m", limit=100)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")

    # Feature engineering
    df["returns"] = df["close"].pct_change()
    df["sma"] = df["close"].rolling(10).mean()
    df["ema"] = df["close"].ewm(span=10).mean()
    df["rsi"] = df["returns"].rolling(14).mean() / (df["returns"].rolling(14).std() + 1e-8)
    df["future"] = df["close"].shift(-forecast_minutes)
    df.dropna(inplace=True)

    # Model
    features = ["open", "high", "low", "close", "volume", "returns", "sma", "ema", "rsi"]
    X = df[features]
    y = df["future"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_scaled[:-1], y[:-1])
    pred = model.predict([X_scaled[-1]])[0]
    last_price = df["close"].iloc[-1]
    delta = pred - last_price
    direction = "UP 📈" if delta > 0 else "DOWN 📉"
    alert = abs(delta) >= alert_threshold

    # Display results
    st.metric(label="Current Price", value=f"${last_price:.2f}")
    st.metric(label="Predicted Price", value=f"${pred:.2f}")
    st.metric(label="Expected Move", value=f"{direction} ${abs(delta):.2f}")

    if alert:
        st.success("🔔 ALERT TRIGGERED!")
        play_sound()
    else:
        st.info("No alert triggered.")

    with st.expander("📊 View Chart"):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["close"], name="Close"))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["sma"], name="SMA", line=dict(dash="dot")))
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["ema"], name="EMA", line=dict(dash="dot")))
        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Failed to load data: {e}")
