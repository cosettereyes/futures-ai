# Futures AI Trading Terminal
Real-Time Futures, Commodities, and Crypto Trading Dashboard  
Multi-Timeframe • AI Trend Prediction • Strategy Signals • Smart Risk Tools

A live trading terminal built with Python and Streamlit.  
Features include real-time data, TradingView-style charts, indicators, AI forecasting, and automatic buy/sell signals.

---

## Features

### Real-Time Market Data
Supports:
- ES, MES, NQ, MNQ, YM, MYM
- RTY, M2K
- CL, GC, SI, NG
- BTC, ETH

Data sources:
- Polygon.io (delayed futures data)
- Yahoo Finance (fallback source)

### Charting System
- Candlestick, bar, or line charts
- Colored volume bars
- EMA9, EMA21, RSI, MACD, VWAP
- Automatic support and resistance detection
- Configurable themes (Light, Dark, Midnight)

### AI Trend Prediction
Uses a linear regression model to predict next-candle direction:
- Trend: UP or DOWN  
- Confidence percentage

### Automated Strategy Signals
Signals include:
- EMA 9/21 crossover
- RSI oversold/overbought
- MACD bullish/bearish cross
- VWAP breakout/breakdown
- Support bounce / resistance rejection
- AI trend confirmation

### Stop-Loss and Take-Profit Suggestions
Generated using:
- Nearest support or resistance levels
- Volatility levels

### Trade Logging
Each signal can be logged with:
- Timestamp
- Symbol
- Action (BUY/SELL)
- Strategy name

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/stock-ai-app.git
cd stock-ai-app
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it:

Windows:
```bash
venv\Scripts\activate
```

Mac/Linux:
```bash
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Polygon API Setup

Inside `app.py`, set your API key:

```python
POLYGON_API_KEY = "YOUR_POLYGON_API_KEY_HERE"
```

Get a free API key at:
https://polygon.io/

---

## Running the Application

Run the Streamlit server:

```bash
streamlit run app.py
```

Access the application in your browser:

```
http://localhost:8501
```

---

## Project Structure

```
stock-ai-app/
│
├── app.py                 # Main application file
├── README.md              # This documentation
├── .gitignore             # Git ignore rules
└── venv/                  # Virtual environment (ignored)
```

---

## Recommended .gitignore

```gitignore
venv/
__pycache__/
*.pyc
.env
.DS_Store
*.log
```

---


