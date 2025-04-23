
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import ccxt
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
import streamlit.components.v1 as components

st.set_page_config(
    page_title="📱 Local Crypto Alerts",
    layout="centered",
    initial_sidebar_state="collapsed"
)

def play_sound():
    sound_url = "https://www.soundjay.com/buttons/sounds/beep-07.mp3"
    st.markdown(f'''
        <audio autoplay>
            <source src="{sound_url}" type="audio/mpeg">
        </audio>
    ''', unsafe_allow_html=True)

    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
    try:
        time.sleep(3)
        driver.find_element(By.XPATH, "//input[@type='text']").send_keys(username)
        driver.find_element(By.XPATH, "//input[@type='password']").send_keys(password)
        driver.find_element(By.XPATH, "//button").click()
        time.sleep(5)
        if "#/pages" not in driver.current_url:
            return None, "Login failed or redirected unexpectedly."
        time.sleep(5)
        elements = driver.find_elements(By.CLASS_NAME, "coin-price")
        prices = [el.text for el in elements if el.text.strip()]
        return prices, None
    except Exception as e:
        return None, str(e)
    finally:
        driver.quit()

st.title("📡 Local Crypto Alert App")

default_coins = ["BTC", "ETH", "SOL", "XRP", "DOGE", "MATIC"]
custom_coin = st.text_input("➕ Add Custom Coin (e.g. ADA)", key="custom_coin_input")
coin_list = default_coins + ([custom_coin.upper()] if custom_coin and custom_coin.upper() not in default_coins else [])
symbol = st.selectbox("Select Coin", coin_list, index=0, key="mobile_coin")
forecast_minutes = st.slider("Predict Minutes Ahead", 1, 30, 5, key="mobile_minutes")
alert_threshold = st.slider("Trigger Alert If Move > $", 0.5, 10.0, 2.0, step=0.5, key="mobile_threshold")

if st.button("🔄 Refresh Prediction"):
    st.session_state["trigger_refresh"] = True

symbol_full = f"{symbol}/USDT"
exchange = ccxt.kraken()
try:
    exchange.load_markets()
    ohlcv = exchange.fetch_ohlcv(symbol_full, "5m", limit=100)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df["returns"] = df["close"].pct_change()
    df["sma"] = df["close"].rolling(10).mean()
    df["ema"] = df["close"].ewm(span=10).mean()
    df["rsi"] = df["returns"].rolling(14).mean() / (df["returns"].rolling(14).std() + 1e-8)
    df["future"] = df["close"].shift(-forecast_minutes)
    df.dropna(inplace=True)

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
    st.error(f"Failed to load data for {symbol_full}: {e}")

            with st.spinner("Logging in and retrieving market prices..."):
                if error:
                    st.error(f"❌ {error}")
                elif prices:
                    for price in prices:
                        st.write(price)
                else:
                    st.warning("No data found.")
        else:
            st.warning("Please enter both username and password.")

# Gold price ticker and market news feed
st.markdown("### 🟡 Gold Price Ticker (USD)")
components.html("""
    <iframe src="https://goldbroker.com/widget/live-price/gold/1?currency=USD" 
            width="100%" height="60" frameborder="0" scrolling="no"></iframe>
""", height=60)

st.markdown("### 🗞️ Market News Feed")
components.html("""
    <iframe src="https://rss.app/embed/v1/wall/your_widget_id" 
            width="100%" height="600" frameborder="0" scrolling="no"></iframe>
""", height=600)
