import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ----------------- Settings ----------------- #
KRAKEN_API_URL = "https://api.kraken.com/0/public/OHLC"
SUPPORTED_COINS = {
    'BTC': 'XBTUSD',
    'ETH': 'ETHUSD',
    'SOL': 'SOLUSD',
    'ADA': 'ADAUSD',
    'XRP': 'XRPUSD',
    'ETC': 'ETCUSD'
}

LOCAL_TIMEZONE = 'America/New_York'  # Change this as needed

# ----------------- Helper Functions ----------------- #
def get_ohlcv_data(pair, interval='60', since=None):
    params = {
        'pair': pair,
        'interval': interval
    }
    if since:
        params['since'] = since
    response = requests.get(KRAKEN_API_URL, params=params).json()
    result = list(response['result'].values())[0]
    df = pd.DataFrame(result, columns=[
        'time', 'open', 'high', 'low', 'close', 'vwap', 'volume', 'count'
    ])
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['close'] = df['close'].astype(float)
    df.set_index('time', inplace=True)
    return df[['close']]

def calculate_indicators(df, show_sma, show_ema):
    if show_sma:
        df['SMA_20'] = df['close'].rolling(window=20).mean()
    if show_ema:
        df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

def generate_predictions(df):
    df['return'] = df['close'].pct_change().shift(-1)
    df = df.dropna()

    df['hour'] = df.index.hour
    df['dayofweek'] = df.index.dayofweek

    X = df[['close', 'hour', 'dayofweek']]
    y = df['return']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    last_row = df.iloc[[-1]]
    pred = model.predict(last_row[['close', 'hour', 'dayofweek']])[0]

    gain = pred * 100
    now_local = datetime.now(pytz.timezone(LOCAL_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S %Z')

    recommendation = "HOLD"
    if gain >= 10 and gain <= 15:
        recommendation = "BUY (Scalp)"
    elif gain <= -10 and gain >= -15:
        recommendation = "SELL (Scalp)"

    return {
        "prediction": round(last_row['close'].values[0] * (1 + pred), 2),
        "gain_percent": round(gain, 2),
        "recommendation": recommendation,
        "timestamp": now_local
    }

# ----------------- Streamlit UI ----------------- #
st.set_page_config(page_title="Crypto Predictor", layout="wide")

st.title("💸 Crypto Price Dashboard + AI Prediction")
st.markdown(f"**Timezone:** {LOCAL_TIMEZONE}")

selected_coin = st.selectbox("Choose a coin:", list(SUPPORTED_COINS.keys()))
pair = SUPPORTED_COINS[selected_coin]

show_sma = st.checkbox("📊 Show SMA (20)", value=True)
show_ema = st.checkbox("📈 Show EMA (20)", value=True)

refresh = st.button("🔁 Refresh Data")
predict_now = st.button("🎯 Refresh Prediction")

@st.cache_data(ttl=3600, show_spinner=False)
def load_data():
    return get_ohlcv_data(pair)

if refresh:
    st.cache_data.clear()
df = load_data()

if df.empty:
    st.error("Failed to load data.")
else:
    current_price = df['close'].iloc[-1]
    st.subheader(f"💰 {selected_coin} Current Price: ${current_price:.2f}")
    st.markdown(f"**Last update:** {datetime.now(pytz.timezone(LOCAL_TIMEZONE)).strftime('%Y-%m-%d %H:%M:%S %Z')}")

    df = calculate_indicators(df, show_sma, show_ema)

    st.line_chart(df)

    if predict_now:
        pred_info = generate_predictions(df)
        st.success(f"📈 **Predicted Price:** ${pred_info['prediction']}")
        st.info(f"📊 Gain: {pred_info['gain_percent']}% — {pred_info['recommendation']}")
        st.caption(f"🕒 Prediction Time: {pred_info['timestamp']}")
