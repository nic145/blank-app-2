# --- streamlit_app.py ---
import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import pytz
import mplfinance as mpf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Set page config FIRST
st.set_page_config(page_title="Crypto Dashboard", layout="wide")

# ---- Sidebar: Timezone ----
st.sidebar.title("Settings")
timezones = pytz.all_timezones
user_timezone = st.sidebar.selectbox("Select Timezone", options=timezones, index=timezones.index("UTC"))
local_tz = pytz.timezone(user_timezone)

# ---- Main UI: Coin selection ----
st.title("🪙 Crypto Prediction Dashboard")
top_coins = ["BTC/USDT", "ETH/USDT", "XRP/USDT", "ADA/USDT", "SOL/USDT", "DOT/USDT", "LTC/USDT"]
selected_coin = st.selectbox("Choose a Coin", top_coins)
custom_coin = st.text_input("Or enter a custom coin (e.g. DOGE/USDT)").strip().upper()
pair = custom_coin if custom_coin else selected_coin

# ---- Sliders ----
col1, col2, col3 = st.columns(3)
with col1:
    scalp_min = st.slider("Scalp Min %", 5, 30, 10)
with col2:
    scalp_max = st.slider("Scalp Max %", 10, 35, 15)
with col3:
    future_hours = st.slider("Predict up to Hours Ahead", 0.25, 8.0, 1.0, 0.25)

# ---- Interval Selection ----
interval_map = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
st.subheader(f"Candlestick Chart - {pair}")
interval_choice = st.radio("Select Interval", list(interval_map.keys()), horizontal=True)

# ---- Fetch OHLC Data ----
def fetch_ohlc(pair, interval):
    url = f"https://api.kraken.com/0/public/OHLC?pair={pair.replace('/', '')}&interval={interval}"
    resp = requests.get(url).json()
    pair_key = list(resp["result"].keys())[0]
    df = pd.DataFrame(resp["result"][pair_key],
                      columns=["time", "open", "high", "low", "close", "vwap", "volume", "count"])
    df["time"] = pd.to_datetime(df["time"], unit="s").dt.tz_localize("UTC").dt.tz_convert(local_tz)
    df.set_index("time", inplace=True)
    df = df.astype(float)
    return df

# ---- Fetch Price ----
def fetch_price(pair):
    url = f"https://api.kraken.com/0/public/Ticker?pair={pair.replace('/', '')}"
    data = requests.get(url).json()
    pair_key = list(data["result"].keys())[0]
    return float(data["result"][pair_key]["c"][0])

# ---- Prediction ----
def predict(df, future_hours):
    df = df[["close"]].copy()
    df["return"] = df["close"].pct_change()
    df.dropna(inplace=True)
    X, y = [], []
    window = 5
    step = int(future_hours * 4)  # 15-min intervals
    for i in range(len(df) - window - step):
        X.append(df["return"].iloc[i:i+window].values)
        y.append(df["close"].iloc[i+window+step])
    X, y = np.array(X), np.array(y)
    scaler = StandardScaler()
    model = RandomForestRegressor(n_estimators=100)
    model.fit(scaler.fit_transform(X), y)
    latest = scaler.transform(df["return"].iloc[-window:].values.reshape(1, -1))
    pred_price = model.predict(latest)[0]
    pred_time = df.index[-1] + timedelta(minutes=15 * step)
    return pred_price, pred_time

# ---- Signal Logic ----
def trade_signal(current, predicted, min_th, max_th):
    pct = ((predicted - current) / current) * 100
    if pct >= max_th:
        return f"🟢 BUY ({pct:.2f}%)"
    elif pct <= -min_th:
        return f"🔴 SELL ({pct:.2f}%)"
    else:
        return f"🟡 HOLD ({pct:.2f}%)"

# ---- Run ----
df = fetch_ohlc(pair, interval_map[interval_choice])
current_price = fetch_price(pair)
predicted_price, pred_time = predict(df, future_hours)

# ---- Candlestick Chart ----
mc = mpf.make_marketcolors(up='g', down='r', inherit=True)
style = mpf.make_mpf_style(marketcolors=mc, base_mpl_style='dark_background')
fig, _ = mpf.plot(df, type='candle', style=style, volume=True, returnfig=True)
st.pyplot(fig)

# ---- Price + Signal ----
st.subheader("📈 Live Prediction")
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:,.2f}")
col2.metric("Predicted Price", f"${predicted_price:,.2f}")
col3.metric("Prediction Time", pred_time.strftime('%I:%M %p %Z'))
st.markdown(f"### Signal: {trade_signal(current_price, predicted_price, scalp_min, scalp_max)}")

# ---- requirements.txt ----
with open("requirements.txt", "w") as f:
    f.write("""streamlit\npandas\nnumpy\nrequests\nscikit-learn\nmatplotlib\nmplfinance""")
with open("requirements.txt", "rb") as f:
    st.sidebar.download_button("Download requirements.txt", f, file_name="requirements.txt")

# ---- Auto Refresh Every 60s ----
st_autorefresh = st.experimental_rerun if 'rerun_now' in st.session_state else None
st.session_state['rerun_now'] = True
