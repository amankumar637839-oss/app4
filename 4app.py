import streamlit as st
import pandas as pd
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go

st.set_page_config(page_title="Pro Trading Dashboard", layout="wide")

# ---------- STYLE ----------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg,#0f172a,#1e293b);
    color: white;
}
.big-title {
    text-align:center;
    font-size:48px;
    color:#38bdf8;
    font-weight:bold;
}
.metric-box {
    background: rgba(255,255,255,0.08);
    padding: 20px;
    border-radius: 15px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📈 Professional Stock Dashboard</div>', unsafe_allow_html=True)

# ---------- SIDEBAR ----------
st.sidebar.title("Trading Panel")
ticker = st.sidebar.text_input("Ticker", "AAPL")
years = st.sidebar.slider("Years", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)

# ---------- INDICATORS ----------
def RSI(data, period=14):
    delta = data.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def MACD(data):
    ema12 = data.ewm(span=12).mean()
    ema26 = data.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    return macd, signal

if st.sidebar.button("Run Analysis"):

    df = yf.download(ticker, period=f"{years}y", auto_adjust=True)

    if df.empty:
        st.error("Invalid ticker")
        st.stop()

    close = df["Close"]
    volume = df["Volume"]

    # Moving averages
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    # Bollinger Bands
    df["STD"] = close.rolling(20).std()
    df["Upper"] = df["MA20"] + (df["STD"] * 2)
    df["Lower"] = df["MA20"] - (df["STD"] * 2)

    # RSI
    df["RSI"] = RSI(close)

    # MACD
    df["MACD"], df["Signal"] = MACD(close)

    # ARIMA Forecast
    model = ARIMA(close.dropna(), order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    latest = float(close.tail(1).values[0])
  predicted = float(forecast[-1])
    growth = ((predicted - latest) / latest) * 100

    # Signal engine
    signal = "HOLD"
    if df["RSI"].iloc[-1] < 30 and df["MACD"].iloc[-1] > df["Signal"].iloc[-1]:
        signal = "STRONG BUY"
    elif growth > 5:
        signal = "BUY"
    elif growth < -5:
        signal = "SELL"

    # Metrics
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${latest:.2f}")
    col2.metric("Forecast Price", f"${predicted:.2f}")
    col3.metric("Expected Return", f"{growth:.2f}%")
    col4.metric("Signal", signal)

    # Candlestick
    st.subheader("Candlestick Chart")

    fig = go.Figure(data=[
        go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"]
        )
    ])

    fig.add_trace(go.Scatter(x=df.index, y=df["MA20"], name="MA20"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA50"], name="MA50"))
    fig.add_trace(go.Scatter(x=df.index, y=df["MA200"], name="MA200"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper"], name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower"], name="Lower Band"))

    fig.update_layout(template="plotly_dark", height=700)
    st.plotly_chart(fig, use_container_width=True)

    # RSI
    st.subheader("RSI Indicator")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    rsi_fig.add_hline(y=70)
    rsi_fig.add_hline(y=30)
    rsi_fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(rsi_fig, use_container_width=True)

    # MACD
    st.subheader("MACD Indicator")

    macd_fig = go.Figure()
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))
    macd_fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(macd_fig, use_container_width=True)

    # Volume
    st.subheader("Volume Analysis")

    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(x=df.index, y=volume, name="Volume"))
    volume_fig.update_layout(template="plotly_dark", height=400)
    st.plotly_chart(volume_fig, use_container_width=True)

    # Forecast Table
    future_dates = pd.date_range(start=close.index[-1], periods=forecast_days)
    forecast_df = pd.DataFrame({"Forecast": forecast.values}, index=future_dates)

    st.subheader("Forecast Data")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode("utf-8")
    st.download_button("Download Forecast", csv, f"{ticker}_forecast.csv")

    # Risk Meter
    volatility = close.pct_change().std() * 100

    st.subheader("Risk Analysis")
    st.metric("Volatility", f"{volatility:.2f}%")

    if volatility < 2:
        st.success("Low Risk")
    elif volatility < 4:
        st.warning("Medium Risk")
    else:
        st.error("High Risk")
