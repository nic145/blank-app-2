import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime
import io

# ------------------------------
# Kraken OHLCV Fetcher
# ------------------------------

def fetch_ohlcv_data(symbol, interval='60', since=None):
    url = f"https://api.kraken.com/0/public/OHLC?pair={symbol}&interval={interval}"
    if since:
        url += f"&since={since}"
    response = requests.get(url)
    data = response.json()
    if data['error']:
        st.error(f"Kraken API Error: {data['error']}")
        return None
    result = list(data['result'].values())[0]
    df = pd.DataFrame(result, columns=["timestamp", "open", "high", "low", "close", "vwap", "volume", "count"])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.astype({"open": float, "high": float, "low": float, "close": float})
    return df

# ------------------------------
# Technical Indicators
# ------------------------------

def calculate_indicators(df):
    df['SMA_10'] = df['close'].rolling(window=10).mean()
    df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()
    return df

# ------------------------------
# Scalping Signal (Crossover)
# ------------------------------

def generate_scalping_signals(df):
    df['Buy_Signal'] = (df['SMA_10'].shift(1) < df['EMA_20'].shift(1)) & (df['SMA_10'] > df['EMA_20'])
    df['Sell_Signal'] = (df['SMA_10'].shift(1) > df['EMA_20'].shift(1)) & (df['SMA_10'] < df['EMA_20'])
    return df

# ------------------------------
# Chart Drawing
# ------------------------------

def plot_chart(df, selected_crypto):
    fig = go.Figure()

    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], name='Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_10'], name='SMA 10', line=dict(color='orange')))
    fig.add_trace(go.Scatter(x=df['timestamp'], y=df['EMA_20'], name='EMA 20', line=dict(color='green')))

    # Plot Buy/Sell signals
    fig.add_trace(go.Scatter(
        x=df[df['Buy_Signal']]['timestamp'],
        y=df[df['Buy_Signal']]['close'],
        mode='markers',
        marker=dict(color='lime', size=10, symbol='triangle-up'),
        name='Buy Signal'
    ))

    fig.add_trace(go.Scatter(
        x=df[df['Sell_Signal']]['timestamp'],
        y=df[df['Sell_Signal']]['close'],
        mode='markers',
        marker=dict(color='red', size=10, symbol='triangle-down'),
        name='Sell Signal'
    ))

    fig.update_layout(
        title=f"{selected_crypto} Price with SMA/EMA + Scalping Signals",
        xaxis_title="Time",
        yaxis_title="Price (USD)",
        template="plotly_white"
    )

    return fig

# ------------------------------
# Main App
# ------------------------------

def main():
    st.title("🚀 Kraken Crypto Scalping Assistant")

    # Select Coin
    crypto_options = {
        'Bitcoin (BTC/USD)': 'XBTUSD',
        'Ethereum (ETH/USD)': 'ETHUSD',
        'Solana (SOL/USD)': 'SOLUSD',
        'XRP (XRP/USD)': 'XRPUSD',
        'Cardano (ADA/USD)': 'ADAUSD',
        'Ethereum Classic (ETC/USD)': 'ETCUSD'
    }

    selected_crypto_name = st.selectbox("Select Cryptocurrency", list(crypto_options.keys()))
    selected_crypto_symbol = crypto_options[selected_crypto_name]

    # Fetch + Process Data
    price_df = fetch_ohlcv_data(selected_crypto_symbol)
    
    if price_df is not None and not price_df.empty:
        price_df = calculate_indicators(price_df)
        price_df = generate_scalping_signals(price_df)

        st.subheader(f"📈 {selected_crypto_name} Chart + Indicators")
        fig = plot_chart(price_df, selected_crypto_name)
        st.plotly_chart(fig)

        # Save Buttons
        st.subheader("💾 Save Options")

        buffer = io.BytesIO()
        chart_csv = price_df.to_csv(index=False).encode('utf-8')
        buffer.write(chart_csv)
        buffer.seek(0)

        st.download_button(
            label="💾 Download Full Chart Data (CSV)",
            data=buffer,
            file_name=f"{selected_crypto_symbol}_chartdata_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

        signals_df = price_df[(price_df['Buy_Signal']) | (price_df['Sell_Signal'])][['timestamp', 'close', 'Buy_Signal', 'Sell_Signal']]

        if not signals_df.empty:
            signals_buffer = io.BytesIO()
            signals_csv = signals_df.to_csv(index=False).encode('utf-8')
            signals_buffer.write(signals_csv)
            signals_buffer.seek(0)

            st.download_button(
                label="💾 Download Buy/Sell Signals (CSV)",
                data=signals_buffer,
                file_name=f"{selected_crypto_symbol}_signals_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

    else:
        st.error("Failed to fetch price! Data is empty.")

if __name__ == "__main__":
    main()
