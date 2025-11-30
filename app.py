# ==============================================
#   FUTURES AI TRADING TERMINAL – FULL VERSION
#   Multi-Timeframe + Strategies + TV Chart
# ==============================================
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import time
from datetime import datetime, timedelta
import requests
POLYGON_API_KEY = "NVksWgBFh0_zyqR5GT76CNitsRAGKVBp"
# ----------------------------------------------
# PAGE + GLOBAL SETTINGS
# ----------------------------------------------
st.set_page_config(
    page_title="Futures AI Terminal",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------
# THEME SYSTEM (User Controlled)
# ----------------------------------------------
THEMES = {
    "Light": {
        "bg": "#FFFFFF",
        "text": "#000000",
        "grid": "#E0E0E0",
        "bull": "#26A69A",
        "bear": "#EF5350",
        "volume_bull": "#26A69A",
        "volume_bear": "#EF5350",
    },
    "Dark": {
        "bg": "#0E1117",
        "text": "#FAFAFA",
        "grid": "#333333",
        "bull": "#00D27F",
        "bear": "#E2474B",
        "volume_bull": "#00D27F",
        "volume_bear": "#E2474B",
    },
    "Midnight": {
        "bg": "#000000",
        "text": "#E6E6E6",
        "grid": "#1F1F1F",
        "bull": "#2AFFA3",
        "bear": "#FF4D4D",
        "volume_bull": "#2AFFA3",
        "volume_bear": "#FF4D4D",
    }
}

# Sidebar options
st.sidebar.header("⚙️ Settings")
theme_choice = st.sidebar.selectbox("Theme", list(THEMES.keys()))
chart_type = st.sidebar.selectbox(
    "Chart Type",
    ["Candles", "Bars", "Line"],
    index=0
)
theme = THEMES[theme_choice]

# ----------------------------------------------
# SYMBOL LIST – Index Futures, Commodities, Crypto
# ----------------------------------------------
SYMBOLS = {
    "ES (S&P 500 Futures)": "ES=F",
    "MES (Micro S&P)": "MES=F",
    "NQ (Nasdaq Futures)": "NQ=F",
    "MNQ (Micro Nasdaq)": "MNQ=F",
    "YM (Dow Futures)": "YM=F",
    "MYM (Micro Dow)": "MYM=F",
    "RTY (Russell 2000)": "RTY=F",
    "M2K (Micro Russell)": "M2K=F",
    "CL (Crude Oil)": "CL=F",
    "GC (Gold)": "GC=F",
    "SI (Silver)": "SI=F",
    "NG (Natural Gas)": "NG=F",
    "BTC (Bitcoin)": "BTC-USD",
    "ETH (Ethereum)": "ETH-USD",
}

symbol_name = st.sidebar.selectbox("Market", list(SYMBOLS.keys()))
symbol = SYMBOLS[symbol_name]

# ----------------------------------------------
# TIMEFRAME OPTIONS
# ----------------------------------------------
TIMEFRAMES = {
    "1 Minute": "1m",
    "5 Minute": "5m",
    "15 Minute": "15m",
    "30 Minute": "30m",
    "1 Hour": "60m",
}

tf_name = st.sidebar.selectbox("Timeframe", list(TIMEFRAMES.keys()))
tf_interval = TIMEFRAMES[tf_name]

# AUTO REFRESH (5 sec)
REFRESH_RATE = 5
st.sidebar.markdown(f"⏱ Refresh every **{REFRESH_RATE} seconds** (free tier limit).")

# ----------------------------------------------
# HEADER
# ----------------------------------------------
st.markdown(f"""
# 🔥 Futures AI Trading Terminal  
### Market: **{symbol_name}**  
Timeframe: **{tf_name}**  
Theme: **{theme_choice}**
""")
# ==============================================
# PART 2 – INDICATORS + AI MODEL
# ==============================================

# -------------------------------
# 1) Exponential Moving Average
# -------------------------------
def ema(series, length):
    return series.ewm(span=length, adjust=False).mean()

# -------------------------------
# 2) RSI
# -------------------------------
def rsi(series, period=14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# -------------------------------
# 3) MACD
# -------------------------------
def macd(close):
    ema12 = ema(close, 12)
    ema26 = ema(close, 26)
    macd_line = ema12 - ema26
    signal = ema(macd_line, 9)
    hist = macd_line - signal
    return macd_line, signal, hist

# -------------------------------
# 4) VWAP
# -------------------------------
def vwap(df):
    typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
    volume = df["Volume"]
    vwap_series = (typical_price * volume).cumsum() / volume.cumsum()
    return vwap_series
# -------------------------------
# 5) Support / Resistance Detection
# -------------------------------
def find_support_resistance(df, distance=3):
    lows = df["Low"].values
    highs = df["High"].values
    index = df.index

    support = []
    resistance = []

    for i in range(distance, len(df) - distance):
        # Support detection
        if lows[i] == min(lows[i - distance:i + distance + 1]):
            support.append((index[i], lows[i]))

        # Resistance detection
        if highs[i] == max(highs[i - distance:i + distance + 1]):
            resistance.append((index[i], highs[i]))

    return support, resistance

# -------------------------------
# 6) AI MODEL (Linear Regression)
# -------------------------------
def ai_predict(df):
    df = df.reset_index()
    df["index"] = np.arange(len(df))

    X = df[["index"]]
    y = df["Close"]

    model = LinearRegression()
    model.fit(X, y)

    next_index = np.array([[len(df)]])
    prediction = model.predict(next_index)[0]

    last_close = float(df["Close"].iat[-1])

    trend = "UP" if prediction > last_close else "DOWN"
    confidence = abs(prediction - last_close) / last_close * 100

    return prediction, trend, confidence
# ==============================================
# PART 3 – DATA LOADER + STRATEGY ENGINE
# ==============================================
def polygon_load_futures(symbol, interval):
    """
    Load futures candles from Polygon.io.
    Example symbol: ES=F → X:ESH2025 for index futures.
    """

    # ==========================
    # 1 — Map Yahoo symbols → Polygon futures tickers
    # ==========================
    POLY_SYMBOL_MAP = {
        "ES=F": "X:ES",   # S&P 500 E-mini
        "MES=F": "X:MES", # Micro ES
        "NQ=F": "X:NQ",   # Nasdaq
        "MNQ=F": "X:MNQ",
        "YM=F": "X:YM",   # Dow
        "MYM=F": "X:MYM",
        "RTY=F": "X:RTY", # Russell
        "M2K=F": "X:M2K",

        "CL=F": "X:CL",   # Crude oil
        "GC=F": "X:GC",   # Gold
        "SI=F": "X:SI",   # Silver
        "NG=F": "X:NG",   # Nat Gas
    }

    if symbol not in POLY_SYMBOL_MAP:
        return None
    
    base = POLY_SYMBOL_MAP[symbol]

    # ==========================
    # 2 — Interval conversion
    # ==========================
    interval_map = {
        "1m": "1",
        "5m": "5",
        "15m": "15",
        "30m": "30",
        "60m": "60",
    }
    timespan = interval_map.get(interval, "1")

    # ==========================
    # 3 — API URL
    # ==========================
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{base}/range/"
        f"{timespan}/minute/2023-01-01/2025-12-31"
        f"?adjusted=true&limit=50000&sort=asc&apiKey={POLYGON_API_KEY}"
    )

    # ==========================
    # 4 — Request data
    # ==========================
    r = requests.get(url)
    if r.status_code != 200:
        return None

    data = r.json()
    if "results" not in data:
        return None

    # ==========================
    # 5 — Convert → DataFrame
    # ==========================
    results = data["results"]
    df = pd.DataFrame(results)

    df.rename(columns={
        "t": "timestamp",
        "o": "Open",
        "h": "High",
        "l": "Low",
        "c": "Close",
        "v": "Volume",
    }, inplace=True)

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)

    return df
# ----------------------------------------------
# DATA LOADER (Multi-Timeframe via yfinance)
# ----------------------------------------------
def load_data(symbol, interval):
    """
    Try Polygon first → then Yahoo fallback.
    """

    # 1 — Try Polygon.io
    poly_df = polygon_load_futures(symbol, interval)
    if poly_df is not None and len(poly_df) > 10:
        st.info("Loaded from Polygon.io ✓ (delayed futures data)")
        return poly_df

    # 2 — Yahoo fallback
    try:
        st.warning("Polygon returned no data — falling back to Yahoo Finance.")

        if interval == "1m":
            period = "2d"
        elif interval in ["5m", "15m", "30m"]:
            period = "5d"
        else:
            period = "60d"

        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            progress=False,
            auto_adjust=False
        )

        df.dropna(inplace=True)
        df.index = pd.to_datetime(df.index)

        return df

    except Exception as e:
        st.error(f"Data load error: {e}")
        return None


# ----------------------------------------------
# APPLY ALL INDICATORS
# ----------------------------------------------
def apply_indicators(df):
    df["EMA9"] = ema(df["Close"], 9)
    df["EMA21"] = ema(df["Close"], 21)
    df["RSI"] = rsi(df["Close"])
    df["VWAP"] = vwap(df)
    macd_line, signal, hist = macd(df["Close"])
    df["MACD"] = macd_line
    df["MACD_Signal"] = signal
    df["MACD_Hist"] = hist

    support, resistance = find_support_resistance(df)
    return df, support, resistance


# ----------------------------------------------
# STRATEGY ENGINE
# ----------------------------------------------
def generate_signals(df, support, resistance):
    signals = []

    # Last candle values
    close       = float(df["Close"].iat[-1])
    open_price  = float(df["Open"].iat[-1])
    ema9        = float(df["EMA9"].iat[-1])
    ema21       = float(df["EMA21"].iat[-1])
    rsi_value   = float(df["RSI"].iat[-1])
    macd_val    = float(df["MACD"].iat[-1])
    macd_signal = float(df["MACD_Signal"].iat[-1])
    vwap_val    = float(df["VWAP"].iat[-1])

    # ATR (stop loss baseline)
    df["TR"] = np.maximum(
        df["High"] - df["Low"],
        np.maximum(abs(df["High"] - df["Close"].shift(1)),
                   abs(df["Low"] - df["Close"].shift(1)))
    )
    df["ATR"] = df["TR"].rolling(14).mean()
    atr = float(df["ATR"].iat[-1])

    stop_loss = None
    take_profit = None

    # ============= BUY CONDITIONS =============
    if (
        df["EMA9"].iloc[-2] < df["EMA21"].iloc[-2] and ema9 > ema21
        and close > vwap_val
        and rsi_value > 35
    ):
        signals.append(("BUY", "Momentum + Trend Reversal"))

        # Structure SL: last swing low
        if len(support) > 0:
            stop_loss = support[-1][1]
        else:
            stop_loss = close - (atr * 1.5)

        take_profit = close + (atr * 3)

    # ============= SELL CONDITIONS =============
    if (
        df["EMA9"].iloc[-2] > df["EMA21"].iloc[-2] and ema9 < ema21
        and close < vwap_val
        and rsi_value < 65
    ):
        signals.append(("SELL", "Momentum Reversal Down"))

        # Structure SL: last swing high
        if len(resistance) > 0:
            stop_loss = resistance[-1][1]
        else:
            stop_loss = close + (atr * 1.5)

        take_profit = close - (atr * 3)

    # AI MODEL
    prediction, trend, confidence = ai_predict(df)
    confidence = float(confidence)
    if trend == "UP":
        signals.append(("BUY", f"AI Trend Up ({confidence:.2f}% strength)"))
    else:
        signals.append(("SELL", f"AI Trend Down ({confidence:.2f}% strength)"))

    return signals, prediction, trend, confidence, stop_loss, take_profit
# ==============================================
# PART 4 – TRADINGVIEW-STYLE CHART + VOLUME
# ==============================================

def tradingview_chart(df, support, resistance, theme, chart_type):
    fig = go.Figure()

    # --------------------------------------------------
    # 1) PRICE CHART FIRST – ensures candles draw on TOP
    # --------------------------------------------------
    if chart_type == "Candles":
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],

            increasing_line_color=theme["bull"],
            decreasing_line_color=theme["bear"],
            increasing_fillcolor=theme["bull"],
            decreasing_fillcolor=theme["bear"],

            increasing_line_width=2,
            decreasing_line_width=2,
            name="Candles"
        ))

    elif chart_type == "Bars":
        fig.add_trace(go.Ohlc(
            x=df.index,
            open=df["Open"],
            high=df["High"],
            low=df["Low"],
            close=df["Close"],
            increasing_line_color=theme["bull"],
            decreasing_line_color=theme["bear"],
            name="Bars"
        ))

    else:  # Line chart
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            line=dict(color=theme["text"], width=2),
            name="Price"
        ))

    # --------------------------------------------------
    # 2) Volume (second layer)
    # --------------------------------------------------
    volume_colors = []
    for i in range(len(df)):
        close_i = float(df["Close"].iloc[i])
        open_i = float(df["Open"].iloc[i])
        volume_colors.append(
            theme["volume_bull"] if close_i > open_i else theme["volume_bear"]
        )

    fig.add_trace(go.Bar(
        x=df.index,
        y=df["Volume"],
        marker_color=volume_colors,
        name="Volume",
        opacity=0.3,
        yaxis="y2"
    ))

    # --------------------------------------------------
    # 3) Indicators (third layer)
    # --------------------------------------------------
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA9"],
        mode="lines",
        line=dict(color="#00FFAA", width=1.5),
        name="EMA 9"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["EMA21"],
        mode="lines",
        line=dict(color="#FFAA00", width=1.5),
        name="EMA 21"
    ))

    fig.add_trace(go.Scatter(
        x=df.index,
        y=df["VWAP"],
        mode="lines",
        line=dict(color="#0099FF", width=1.2, dash="dot"),
        name="VWAP"
    ))

    # --------------------------------------------------
    # 4) Support & Resistance (top layer)
    # --------------------------------------------------
    for t, level in support:
        fig.add_hline(
            y=level,
            line=dict(color="#00FF00", width=1, dash="dash"),
            opacity=0.35
        )

    for t, level in resistance:
        fig.add_hline(
            y=level,
            line=dict(color="#FF0000", width=1, dash="dash"),
            opacity=0.35
        )

    # --------------------------------------------------
    # Chart Layout
    # --------------------------------------------------
    fig.update_layout(
        height=600,
        plot_bgcolor=theme["bg"],
        paper_bgcolor=theme["bg"],
        font=dict(color=theme["text"]),

        xaxis=dict(
            gridcolor=theme["grid"],
            showgrid=True,
            rangeslider=dict(visible=False)
        ),
        yaxis=dict(
            gridcolor=theme["grid"],
            showgrid=True
        ),
        yaxis2=dict(
            overlaying="y",
            side="right",
            showgrid=False,
            range=[0, df["Volume"].max() * 6],
        ),
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h"),
    )

    return fig

# ==============================================
# PART 5 – UI LAYOUT + AUTO REFRESH + PANELS
# ==============================================

# ----------------------------------------------
# LOAD DATA
# ----------------------------------------------
df = load_data(symbol, tf_interval)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

st.write("RAW DF:", df)


if df is None or len(df) < 50:
    st.error("Not enough data to load chart.")
    st.stop()

# Apply Indicators
df, support, resistance = apply_indicators(df)

# Strategy Signals
signals, prediction, trend, confidence, stop_loss, take_profit = generate_signals(df, support, resistance)

# ----------------------------------------------
# TABS LAYOUT (Dashboard, Signals, AI, Log, Settings)
# ----------------------------------------------
tabs = st.tabs(["📊 Dashboard", "📈 Signals", "🤖 AI Prediction", "📘 Indicator Data", "📝 Trade Log"])

# ----------------------------------------------
# TAB 1 – DASHBOARD WITH CHART
# ----------------------------------------------
with tabs[0]:
    st.subheader("📊 Market Chart")
    chart = tradingview_chart(df, support, resistance, theme, chart_type)
    st.plotly_chart(chart, width="stretch")

# ----------------------------------------------
# TAB 2 – SIGNALS PANEL
# ----------------------------------------------
with tabs[1]:
    st.subheader("📈 Strategy Signals")

    if len(signals) == 0:
        st.info("No signals currently.")
    else:
        for sig_type, sig_text in signals:
            st.markdown(f"### {sig_type} — {sig_text}")

            # Stop Loss + Take Profit display
            if stop_loss is not None:
                st.warning(f"🛑 Suggested Stop Loss: **{stop_loss:.2f}**")

            if take_profit is not None:
                st.success(f"🎯 Suggested Take Profit: **{take_profit:.2f}**")

# ----------------------------------------------
# TAB 3 – AI PREDICTION
# ----------------------------------------------
with tabs[2]:
    st.subheader("🤖 AI Trend Prediction")
    prediction = float(prediction)
    confidence = float(confidence)
    st.metric(
        label="Next Candle Prediction",
        value=f"${prediction:.2f}",
        delta=f"{trend} ({confidence:.2f}% confidence)"
    )

    st.success(f"AI predicts **{trend}** move with **{confidence:.2f}% confidence**.")

# ----------------------------------------------
# TAB 4 – RAW INDICATOR PANEL
# ----------------------------------------------
with tabs[3]:
    st.subheader("📘 Indicator Data")
    st.dataframe(df[["Open", "High", "Low", "Close", "EMA9", "EMA21", "RSI", "MACD", "MACD_Signal", "VWAP"]].tail(50))

# ----------------------------------------------
# TAB 5 – TRADE LOG (Session-based)
# ----------------------------------------------
with tabs[4]:
    st.subheader("📝 Trade Log")

    if "trade_log" not in st.session_state:
        st.session_state.trade_log = []

    # Save signals to log if user wants
    if st.button("💾 Log Current Signals"):
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        for s in signals:
            st.session_state.trade_log.append([timestamp, symbol, s[0], s[1]])
        st.success("Saved to log!")

    st.write("### Logged Signals")
    if len(st.session_state.trade_log) == 0:
        st.info("No trades logged yet.")
    else:
        log_df = pd.DataFrame(st.session_state.trade_log, columns=["Time", "Symbol", "Type", "Signal"])
        st.dataframe(log_df)

# ----------------------------------------------
# AUTO REFRESH ENGINE
# ----------------------------------------------
st.rerun()
