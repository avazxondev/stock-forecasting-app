import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# Sahifa sozlamalari
st.set_page_config(page_title="Stock & Crypto Forecasting App", page_icon="📈", layout="wide")

st.title("📈 Stock & Crypto Trend & Volatility Forecasting Dashboard")
st.markdown("Matematik tahlil, vaqtli qatorlar (Time Series) va Machine Learning yordamida bozor narxlarini bashorat qilish ilovasi.")

# Yon panel (Sidebar) sozlamalari
st.sidebar.header("Parametrlar sozlamasi")
ticker_symbol = st.sidebar.text_input("Ticker kiriting (Masalan: AAPL, TSLA, BTC-USD, ETH-USD):", value="AAPL")
start_date = st.sidebar.date_input("Boshlanish sanasi", value=datetime.today() - timedelta(days=365*2))
forecast_days = st.sidebar.slider("Bashorat qilish kunlari soni:", min_value=7, max_value=90, value=30)

# Ma'lumotlarni yuklab olish funksiyasi
@st.cache_data
def load_data(ticker, start):
    df = yf.download(ticker, start=start)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.reset_index(inplace=True)
    return df

try:
    with st.spinner(f"'{ticker_symbol}' ma'lumotlari yuklanmoqda..."):
        data = load_data(ticker_symbol, start_date)
    
    if data.empty:
        st.error("Kiritilgan Ticker bo'yicha ma'lumot topilmadi. Qaytadan urinib ko'ring.")
    else:
        # 1. Feature Engineering & Mathematical Indicators
        data['Daily_Return'] = data['Close'].pct_change()
        data['Volatility'] = data['Daily_Return'].rolling(window=21).std() * np.sqrt(252) # Yillik volatillik
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['SMA_200'] = data['Close'].rolling(window=200).mean()

        st.subheader(f"📊 {ticker_symbol.upper()} - Narxlar va Harakatlanuvchi O'rtachalar")
        
        # Asosiy narx grafigi
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], mode='lines', name='Close Price', line=dict(color='royalblue', width=2)))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_50'], mode='lines', name='SMA 50', line=dict(color='orange', width=1)))
        fig.add_trace(go.Scatter(x=data['Date'], y=data['SMA_200'], mode='lines', name='SMA 200', line=dict(color='crimson', width=1)))
        fig.update_layout(xaxis_title="Sana", yaxis_title="Narx (USD)", template="plotly_dark", height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Volatillik grafigi
        st.subheader("⚡ Narxlar Volatilligi (Risk Tahlili)")
        fig_vol = px.line(data, x='Date', y='Volatility', title="21 kunlik Rolling Volatillik (Yillashtirilgan)", template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)

        # 2. Prophet Model yordamida Bashorat (Forecasting)
        st.subheader(f"🔮 Kelgusi {forecast_days} kun uchun Trend Bashorati (Meta Prophet)")
        
        df_prophet = data[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
        
        model = Prophet(daily_seasonality=True)
        model.fit(df_prophet)
        
        future = model.make_future_dataframe(periods=forecast_days)
        forecast = model.predict(future)
        
        # Bashorat grafigi
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Bashorat (Forecast)', line=dict(color='green', width=2)))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], mode='lines', name='Yuqori chegara (Upper)', line=dict(color='gray', dash='dash')))
        fig_forecast.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], mode='lines', name='Pastki chegara (Lower)', line=dict(color='gray', dash='dash'), fill='tonexty'))
        fig_forecast.add_trace(go.Scatter(x=df_prophet['ds'], y=df_prophet['y'], mode='lines', name='Haqiqiy narx', line=dict(color='white', width=1)))
        
        fig_forecast.update_layout(template="plotly_dark", height=500, xaxis_title="Sana", yaxis_title="Bashorat qilingan narx")
        st.plotly_chart(fig_forecast, use_container_width=True)

        st.success("Tahlil va bashorat muvaffaqiyatli yakunlandi!")

except Exception as e:
    st.error(f"Xatolik yuz berdi: {e}")
