import krakenex
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

class CryptoApp:
    def __init__(self):
        self.last_refresh = None

    # Fetch Kraken OHLCV data
    def fetch_kraken_ohlcv(self, symbol="BTCUSD", interval=5):
        # Initialize Kraken API client
        k = krakenex.API()

        try:
            # Fetch OHLC data from Kraken API
            ohlcv_data = k.query_public('OHLC', {
                'pair': symbol,         # Currency pair (e.g., 'BTCUSD')
                'interval': interval    # Timeframe interval (5, 15, 60, 1440, etc.)
            })
            
            # Check if the response is successful
            if ohlcv_data.get('error'):
                st.error(f"Error fetching data from Kraken: {ohlcv_data['error']}")
                return None
            
            # Extract and convert data into DataFrame
            ohlcv = ohlcv_data['result'][symbol]
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume', 'closed', 'count', 'average'])
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')

            return df
        except Exception as e:
            st.error(f"Error fetching data: {e}")
            return None

    # Function to calculate technical indicators (Simple example: SMA)
    def calculate_technical_indicators(self, df):
        # Moving Average (SMA) as an example of technical indicator
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        return df

    # Function to create and display the historical chart
    def create_price_history_chart(self, df, symbol):
        fig = go.Figure()

        # Adding the Close Price Line
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['close'], mode='lines', name=f'{symbol} Close', line=dict(color='blue')))
        
        # Adding the 50-period Moving Average (SMA)
        fig.add_trace(go.Scatter(x=df['timestamp'], y=df['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange')))

        # Layout customization
        fig.update_layout(
            title=f"{symbol} Price History with SMA50",
            xaxis_title="Time",
            yaxis_title="Price (USD)",
            template="plotly_dark",
            xaxis_rangeslider_visible=False
        )
        return fig

    # Function to display technical indicators
    def display_technical_indicators(self, df):
        st.write("## Technical Indicators")
        st.write(df[['timestamp', 'close', 'SMA_50']].tail(10))

    # Function to display predictions (dummy example)
    def display_predictions(self):
        st.write("## AI Predictions (Placeholder)")

    # Function to run the app
    def run(self):
        selected_crypto = st.sidebar.selectbox('Select Cryptocurrency', ['BTCUSD', 'ETHUSD', 'SOLUSD', 'XRPUSD', 'ADAUSD'])
        timeframe = st.sidebar.selectbox('Timeframe', ['5m', '15m', '1h', '1d', '1wk'])
        refresh = st.sidebar.button('Refresh Data')

        if refresh:
            interval = 5  # Default interval of 5 minutes
            if timeframe == '15m':
                interval = 15
            elif timeframe == '1h':
                interval = 60
            elif timeframe == '1d':
                interval = 1440
            elif timeframe == '1wk':
                interval = 10080  # 7 days in minutes

            # Fetch data
            price_df = self.fetch_kraken_ohlcv(symbol=selected_crypto, interval=interval)
            
            if price_df is not None and not price_df.empty:
                # Calculate technical indicators (e.g., SMA50)
                price_df = self.calculate_technical_indicators(price_df)
                
                # Display the technical indicators
                self.display_technical_indicators(price_df)

                # Create and display the chart
                fig = self.create_price_history_chart(price_df, selected_crypto)
                st.plotly_chart(fig, use_container_width=True)

                # Placeholder for predictions (if you want to integrate AI predictions)
                self.display_predictions()
            else:
                st.error("Failed to fetch OHLCV data.")
                

# Run the app
if __name__ == "__main__":
    app = CryptoApp()
    app.run()
