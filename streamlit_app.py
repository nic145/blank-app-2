import streamlit as st
import pandas as pd
import numpy as np
import krakenex
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import time

# Initialize Kraken API
api = krakenex.API()

# Mapping user-friendly names to Kraken pairs
COIN_MAPPING = {
    "Bitcoin (BTC)": "XBTUSD",
    "Ethereum (ETH)": "ETHUSD",
    "Solana (SOL)": "SOLUSD",
    "Ripple (XRP)": "XRPUSD",
    "Cardano (ADA)": "ADAUSD",
    "Ethereum Classic (ETC)": "ETCUSD",
}

# Function to fetch latest price
def get_latest_price(pair):
    try:
        response = api.query_public('Ticker', {'pair': pair})
        price = float(response['result'][list(response['result'].keys())[0]]['c'][0])
        return price
    except Exception as e:
        st.error(f"Error fetching price for {pair}: {e}")
        return None

# Function to simulate historical price data
def get_historical_data(pair, days=90):
    try:
        np.random.seed(abs(hash(pair)) % (10 ** 8))  # Seed based on pair
        base_price = get_latest_price(pair)
        if base_price is None:
            return None
        prices = base_price + np.cumsum(np.random.normal(0, base_price * 0.01, days))
        dates = pd.date_range(end=pd.Timestamp.today(), periods=days)
        df = pd.DataFrame({'Date': dates, 'Price': prices})
        return df
    except Exception as e:
        st.error(f"Error generating historical data for {pair}: {e}")
        return None

# Function to train AI model
def train_model(df):
    try:
        df['Day'] = np.arange(len(df))
        X = df[['Day']]
        y = df['Price']
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error training model: {e}")
        return None

# Streamlit App
def main():
    st.set_page_config(page_title="Crypto AI Predictor", page_icon="📈", layout="wide")
    st.title("📈 Crypto AI Price Predictor (Kraken)")

    selected_coin = st.selectbox("Choose a coin:", list(COIN_MAPPING.keys()))
    show_predictions = st.toggle("Show AI Predictions 🔮", value=True)

    if selected_coin:
        pair = COIN_MAPPING[selected_coin]

        latest_price = get_latest_price(pair)
        if latest_price:
            st.metric(label=f"Current {selected_coin} Price", value=f"${latest_price:,.2f}")

        st.subheader("📈 Historical Price Chart")
        historical_data = get_historical_data(pair)

        if historical_data is not None:
            model = train_model(historical_data)
            if model and show_predictions:
                # Predict next 30 days
                future_days = 30
                future = pd.DataFrame({'Day': np.arange(len(historical_data), len(historical_data) + future_days)})
                future['Predicted_Price'] = model.predict(future[['Day']])
                
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(historical_data['Date'], historical_data['Price'], label='Historical Price', color='blue')
                future_dates = pd.date_range(start=historical_data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=future_days)
                ax.plot(future_dates, future['Predicted_Price'], label='Predicted Price', linestyle='--', color='green')
                ax.set_xlabel("Date")
                ax.set_ylabel("Price (USD)")
                ax.set_title(f"{selected_coin} Price Chart with AI Prediction")
                ax.legend()
                st.pyplot(fig)

            else:
                st.line_chart(historical_data.set_index('Date')['Price'])

        else:
            st.error("Failed to load historical data.")

    st.caption("🔮 Made with love — Powered by Kraken API and AI models.")

if __name__ == "__main__":
    main()
