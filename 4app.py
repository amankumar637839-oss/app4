
import streamlit as st
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Pro Stock Analyzer", layout="wide")

# -------------------- STYLE --------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #0f172a, #1e293b);
    color: white;
}
.big-title {
    text-align:center;
    font-size:48px;
    color:#38bdf8;
    font-weight:bold;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="big-title">📊 Pro Stock Analyzer</div>', unsafe_allow_html=True)

# -------------------- SIDEBAR --------------------
st.sidebar.title("Stock Analysis Panel")
ticker = st.sidebar.text_input("Enter Stock Ticker", "AAPL")
years = st.sidebar.slider("Historical Data (Years)", 1, 10, 5)
forecast_days = st.sidebar.slider("Forecast Days", 30, 365, 90)

# -------------------- FUNCTIONS --------------------
def calculate_rsi(data, period=14):
    delta = data.diff()

    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()

    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    return rsi

def calculate_macd(data):
    ema12 = data.ewm(span=12).mean()
    ema26 = data.ewm(span=26).mean()

    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()

    return macd, signal

# -------------------- MAIN --------------------
if st.sidebar.button("Analyze Stock"):

    stock = yf.Ticker(ticker)

    df = stock.history(period=f"{years}y")

    if df.empty:
        st.error("Invalid stock ticker")
        st.stop()

    close = df["Close"].squeeze()
    close = pd.to_numeric(close, errors="coerce").dropna()

    # Technical Indicators
    df["MA20"] = close.rolling(20).mean()
    df["MA50"] = close.rolling(50).mean()
    df["MA200"] = close.rolling(200).mean()

    df["RSI"] = calculate_rsi(close)

    df["MACD"], df["Signal"] = calculate_macd(close)

    df["STD"] = close.rolling(20).std()
    df["Upper Band"] = df["MA20"] + (df["STD"] * 2)
    df["Lower Band"] = df["MA20"] - (df["STD"] * 2)

    # Support and Resistance
    support = round(df["Low"].tail(30).min(), 2)
    resistance = round(df["High"].tail(30).max(), 2)

    # Forecast
    model = ARIMA(close, order=(5,1,0))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    latest = float(close.tail(1).values[0])
    predicted = float(forecast[-1])

    growth = ((predicted - latest) / latest) * 100

    # Fundamentals
    info = stock.info

    market_cap = info.get("marketCap", "N/A")
    pe_ratio = info.get("trailingPE", "N/A")
    eps = info.get("trailingEps", "N/A")
    dividend_yield = info.get("dividendYield", "N/A")
    beta = info.get("beta", "N/A")
    week_high = info.get("fiftyTwoWeekHigh", "N/A")
    week_low = info.get("fiftyTwoWeekLow", "N/A")
    revenue_growth = info.get("revenueGrowth", "N/A")
    profit_margin = info.get("profitMargins", "N/A")

    # Metrics
    st.subheader("Key Metrics")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Current Price", f"${latest:.2f}")
    col2.metric("Forecast Price", f"${predicted:.2f}")
    col3.metric("Expected Return", f"{growth:.2f}%")
    col4.metric("RSI", f"{df['RSI'].iloc[-1]:.2f}")

    # Fundamentals
    st.subheader("Fundamental Analysis")

    f1, f2, f3, f4 = st.columns(4)

    f1.metric("Market Cap", market_cap)
    f2.metric("P/E Ratio", pe_ratio)
    f3.metric("EPS", eps)
    f4.metric("Beta", beta)

    f5, f6, f7, f8 = st.columns(4)

    f5.metric("Dividend Yield", dividend_yield)
    f6.metric("52W High", week_high)
    f7.metric("52W Low", week_low)
    f8.metric("Profit Margin", profit_margin)

    st.metric("Revenue Growth", revenue_growth)

    # Candlestick Chart
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
    fig.add_trace(go.Scatter(x=df.index, y=df["Upper Band"], name="Upper Band"))
    fig.add_trace(go.Scatter(x=df.index, y=df["Lower Band"], name="Lower Band"))

    fig.update_layout(template="plotly_dark", height=700)

    st.plotly_chart(fig, use_container_width=True)

    # RSI Chart
    st.subheader("RSI Indicator")

    rsi_fig = go.Figure()
    rsi_fig.add_trace(go.Scatter(x=df.index, y=df["RSI"], name="RSI"))
    rsi_fig.add_hline(y=70)
    rsi_fig.add_hline(y=30)

    rsi_fig.update_layout(template="plotly_dark", height=350)

    st.plotly_chart(rsi_fig, use_container_width=True)

    # MACD Chart
    st.subheader("MACD Indicator")

    macd_fig = go.Figure()

    macd_fig.add_trace(go.Scatter(x=df.index, y=df["MACD"], name="MACD"))
    macd_fig.add_trace(go.Scatter(x=df.index, y=df["Signal"], name="Signal"))

    macd_fig.update_layout(template="plotly_dark", height=350)

    st.plotly_chart(macd_fig, use_container_width=True)

    # Volume
    st.subheader("Volume Analysis")

    volume_fig = go.Figure()
    volume_fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume"))

    volume_fig.update_layout(template="plotly_dark", height=350)

    st.plotly_chart(volume_fig, use_container_width=True)

    # Support & Resistance
    st.subheader("Support & Resistance")

    s1, s2 = st.columns(2)

    s1.metric("Support Level", support)
    s2.metric("Resistance Level", resistance)

    # Forecast Table
    future_dates = pd.date_range(
        start=close.index[-1],
        periods=forecast_days
    )

    forecast_df = pd.DataFrame({
        "Forecast Price": forecast.values
    }, index=future_dates)

    st.subheader("Forecast Table")
    st.dataframe(forecast_df)

    csv = forecast_df.to_csv().encode("utf-8")

    st.download_button(
        "Download Forecast CSV",
        csv,
        f"{ticker}_forecast.csv"
    )
