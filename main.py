import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import plotly.graph_objects as go
from scipy.stats import t
from numba import jit, prange
from joblib import Parallel, delayed
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings

warnings.filterwarnings("ignore")

# Streamlit app title
st.title("Quantum-Charged Stock Prediction: Beyond All Tools")

# Sidebar for user input
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Stock Ticker", "AAPL")
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=60))
end_date = st.sidebar.date_input("End Date", datetime.now())
window_base = st.sidebar.slider("Base Window Size", 10, 50, 20)
n_particles = st.sidebar.slider("Particle Count", 50, 500, 100)
hurst_threshold = st.sidebar.slider("Hurst Threshold", 0.5, 0.8, 0.65)


# Fetch real data
@st.cache_data
def fetch_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)


data = fetch_data(ticker, start_date, end_date)
if data.empty:
    st.error("No data fetched. Please check the ticker or date range.")
    st.stop()

# Fetch S&P 500 for benchmark
sp500 = fetch_data("^GSPC", start_date, end_date)

# Calculate log-returns and realized semivariance
data['Log_Return'] = np.log(data['Close'] / data['Close'].shift(1))
data['RSV_Up'] = (data['Log_Return'].where(data['Log_Return'] > 0, 0) ** 2).rolling(5).sum()
data['RSV_Down'] = (data['Log_Return'].where(data['Log_Return'] < 0, 0) ** 2).rolling(5).sum()


# 1. MA-DFA with Volatility Entropy and Fractal Dimension
@jit(nopython=True, parallel=True)
def calculate_ma_dfa(series, window_base, volatility):
    fluctuations = np.empty(len(series))
    hurst = np.empty(len(series))
    entropy = -volatility * np.log(volatility + 1e-10)
    entropy_mean = np.nanmean(entropy)
    for i in prange(len(series)):
        fractal_dim = 2 - hurst[i - 1] if i > 0 and not np.isnan(hurst[i - 1]) else 2
        window = int(window_base * np.exp(-entropy[i] / entropy_mean + 1 / fractal_dim)) if not np.isnan(
            entropy[i]) else window_base
        window = max(10, min(50, window))
        if i < window - 1:
            fluctuations[i] = np.nan
            hurst[i] = np.nan
        else:
            segment = series[i - window:i]
            t = np.arange(len(segment))
            poly = np.polyfit(t, segment, 1)
            detrended = segment - np.polyval(poly, t)
            fluctuation = np.mean(detrended ** 2) ** 0.5
            fluctuations[i] = fluctuation
            hurst[i] = np.log(fluctuation) / np.log(window) if fluctuation > 0 else np.nan
    return hurst, fluctuations


data['Volatility'] = data['Log_Return'].rolling(window_base).std()
data['Hurst'], data['Fluctuations'] = calculate_ma_dfa(data['Close'].values, window_base, data['Volatility'].values)


# 2. Asymmetric MA-DFA
@jit(nopython=True, parallel=True)
def calculate_amfdfa(series, window_base, volatility, returns):
    hurst_up = np.empty(len(series))
    hurst_down = np.empty(len(series))
    entropy = -volatility * np.log(volatility + 1e-10)
    entropy_mean = np.nanmean(entropy)
    for i in prange(len(series)):
        fractal_dim = 2 - hurst_up[i - 1] if i > 0 and not np.isnan(hurst_up[i - 1]) else 2
        window = int(window_base * np.exp(-entropy[i] / entropy_mean + 1 / fractal_dim)) if not np.isnan(
            entropy[i]) else window_base
        window = max(10, min(50, window))
        if i < window - 1:
            hurst_up[i] = np.nan
            hurst_down[i] = np.nan
        else:
            segment = series[i - window:i]
            t = np.arange(len(segment))
            poly = np.polyfit(t, segment, 1)
            detrended = segment - np.polyval(poly, t)
            fluctuation = np.mean(detrended ** 2) ** 0.5
            segment_returns = returns[max(0, i - window):i]
            if len(segment_returns) > 0 and np.mean(segment_returns > 0) > 0.5:
                hurst_up[i] = np.log(fluctuation) / np.log(window) if fluctuation > 0 else np.nan
                hurst_down[i] = np.nan
            else:
                hurst_up[i] = np.nan
                hurst_down[i] = np.log(fluctuation) / np.log(window) if fluctuation > 0 else np.nan
    return hurst_up, hurst_down


data['Hurst_Up'], data['Hurst_Down'] = calculate_amfdfa(data['Close'].values, window_base, data['Volatility'].values,
                                                        data['Log_Return'].values)


# 3. Particle Filter Volatility with Semivariance
def calculate_particle_volatility(returns, rsv_up, rsv_down, n_particles):
    def parallel_particle_update(particles, r, rsv_up_val, rsv_down_val):
        particles = 0.9 * particles + t.rvs(df=5, size=n_particles) * 0.01
        scale = np.sqrt(np.exp(particles / 2) + (rsv_up_val if r > 0 else rsv_down_val))
        likelihood = t.pdf(r, df=5, loc=0, scale=scale + 1e-10)
        return particles, likelihood

    returns = returns.dropna()
    rsv_up = rsv_up.dropna()
    rsv_down = rsv_down.dropna()
    particles = np.random.normal(0, 0.01, n_particles)
    weights = np.ones(n_particles) / n_particles
    volatilities = []

    for r, ru, rd in zip(returns, rsv_up, rsv_down):
        results = Parallel(n_jobs=-1)(delayed(parallel_particle_update)(particles, r, ru, rd) for _ in range(1))
        particles, likelihood = results[0]
        weights *= likelihood
        weights /= np.sum(weights + 1e-10)
        indices = np.random.choice(range(n_particles), size=n_particles, p=weights)
        particles = particles[indices]
        weights = np.ones(n_particles) / n_particles
        volatilities.append(np.mean(np.exp(particles / 2)))

    return np.concatenate([np.array([np.nan] * (len(data) - len(volatilities))), volatilities])


data['Particle_Volatility'] = calculate_particle_volatility(data['Log_Return'], data['RSV_Up'], data['RSV_Down'],
                                                            n_particles)


# 4. Quantum-Inspired Chaos Optimization
@jit(nopython=True)
def qco_tune(hurst, volatility, base_threshold=0.65):
    thresholds = np.empty(len(hurst))
    x = 0.5  # Initial chaotic state
    r = 4.0  # Logistic map parameter
    eta = 0.01  # Learning rate
    for i in range(len(hurst)):
        if np.isnan(hurst[i]) or np.isnan(volatility[i]):
            thresholds[i] = base_threshold
        else:
            entropy = -volatility[i] * np.log(volatility[i] + 1e-10)
            x = r * x * (1 - x)  # Logistic map
            thresholds[i] = base_threshold + eta * x * np.sin(np.pi * entropy)
    return thresholds


data['Hurst_Threshold'] = qco_tune(data['Hurst'], data['Volatility'], hurst_threshold)


# 5. Lightweight FPN-DRL
def create_fpn_drl_model():
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(5, 5)),
        LSTM(16),
        Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    return model


def fpn_drl_predict(states):
    model = create_fpn_drl_model()
    probs = np.random.random((len(states), 3))  # Placeholder
    probs /= probs.sum(axis=1, keepdims=True)
    actions = ['Buy' if p[0] > 0.7 else 'Sell' if p[1] > 0.7 else 'Hold' for p in probs]
    return actions


states = np.array(
    [data[['Hurst', 'Hurst_Up', 'Hurst_Down', 'Particle_Volatility', 'Log_Return']].fillna(0).values[-5:]])
data['FPN_Signal'] = ['Hold'] * (len(data) - 5) + fpn_drl_predict(states)


# 6. Advanced Candlestick Patterns
def detect_candlestick_patterns(df):
    patterns = []
    for i in range(3, len(df)):
        o, c, h, l = df['Open'].iloc[i], df['Close'].iloc[i], df['High'].iloc[i], df['Low'].iloc[i]
        o1, c1, h1, l1 = df['Open'].iloc[i - 1], df['Close'].iloc[i - 1], df['High'].iloc[i - 1], df['Low'].iloc[i - 1]
        o2, c2 = df['Open'].iloc[i - 2], df['Close'].iloc[i - 2]
        # Three White Soldiers
        if (c2 > o2 and c1 > o1 and c > o and (df['Close'].iloc[i - 2:i + 1].diff().iloc[1:] > 0).all() and
                (df['High'].iloc[i - 2:i + 1].diff() > 0).all()):
            patterns.append((df.index[i], 'Three White Soldiers'))
        # Three Black Crows
        elif (c2 < o2 and c1 < o1 and c < o and (df['Close'].iloc[i - 2:i + 1].diff().iloc[1:] < 0).all() and
              (df['Low'].iloc[i - 2:i + 1].diff() < 0).all()):
            patterns.append((df.index[i], 'Three Black Crows'))
        # Morning Star
        elif c2 < o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and c > o:
            patterns.append((df.index[i], 'Morning Star'))
        # Evening Star
        elif c2 > o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and c < o:
            patterns.append((df.index[i], 'Evening Star'))
        # Bullish Abandoned Baby
        elif c2 < o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and c > o and l > h1:
            patterns.append((df.index[i], 'Bullish Abandoned Baby'))
        # Bearish Abandoned Baby
        elif c2 > o2 and abs(c1 - o1) < 0.1 * (h1 - l1) and c < o and h < l1:
            patterns.append((df.index[i], 'Bearish Abandoned Baby'))
        # Bullish Kicker
        elif c2 < o2 and c > o and o > c1 and c > o1:
            patterns.append((df.index[i], 'Bullish Kicker'))
        # Bearish Kicker
        elif c2 > o2 and c < o and o < c1 and c < o1:
            patterns.append((df.index[i], 'Bearish Kicker'))
        else:
            patterns.append((df.index[i], None))
    return [(df.index[0], None), (df.index[1], None), (df.index[2], None)] + patterns


patterns = detect_candlestick_patterns(data)


# 7. Trading Strategy
def generate_trading_signals(df):
    signals = []
    vol_threshold = df['Particle_Volatility'].quantile(0.75)
    for i in range(len(df)):
        if pd.isna(df['Hurst'].iloc[i]) or pd.isna(df['Particle_Volatility'].iloc[i]):
            signals.append('Hold')
        elif (df['Hurst'].iloc[i] > df['Hurst_Threshold'].iloc[i] and
              df['Hurst_Up'].iloc[i] > df['Hurst_Down'].iloc[i] and
              df['Particle_Volatility'].iloc[i] < vol_threshold and
              df['FPN_Signal'].iloc[i] == 'Buy'):
            signals.append('Buy')
        elif (df['Hurst'].iloc[i] < (1 - df['Hurst_Threshold'].iloc[i]) or
              df['Hurst_Down'].iloc[i] > df['Hurst_Up'].iloc[i] or
              df['FPN_Signal'].iloc[i] == 'Sell'):
            signals.append('Sell')
        else:
            signals.append('Hold')
    return signals


data['Signal'] = generate_trading_signals(data)


# 8. Performance Metrics
def calculate_metrics(df, sp500):
    returns = df['Close'].pct_change().where(df['Signal'] == 'Buy', 0) - df['Close'].pct_change().where(
        df['Signal'] == 'Sell', 0)
    cum_return = (1 + returns).cumprod()[-1] - 1 if not returns.empty else 0
    sp500_return = (sp500['Close'][-1] / sp500['Close'][0] - 1) if not sp500.empty else 0
    sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
    sortino = returns.mean() / returns[returns < 0].std() * np.sqrt(252) if returns[returns < 0].std() != 0 else 0
    max_drawdown = (returns.cumsum() - returns.cumsum().cummax()).min()
    calmar = returns.mean() * 252 / abs(max_drawdown) if max_drawdown != 0 else 0
    trades = len(df[df['Signal'].isin(['Buy', 'Sell'])])
    win_rate = len(returns[returns > 0]) / trades if trades > 0 else 0
    return cum_return, sp500_return, sharpe, sortino, max_drawdown, calmar, trades, win_rate


cum_return, sp500_return, sharpe, sortino, max_drawdown, calmar, trades, win_rate = calculate_metrics(data, sp500)

# 9. Candlestick Chart
fig = go.Figure()

# Candlestick
fig.add_trace(go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Candlestick'
))

# Indicators
fig.add_trace(
    go.Scatter(x=data.index, y=data['Hurst'] * 100, name='MA-DFA Hurst (x100)', line=dict(color='orange'), yaxis='y2'))
fig.add_trace(go.Scatter(x=data.index, y=data['Hurst_Up'] * 100, name='Hurst Uptrend (x100)',
                         line=dict(color='green', dash='dash'), yaxis='y2'))
fig.add_trace(go.Scatter(x=data.index, y=data['Hurst_Down'] * 100, name='Hurst Downtrend (x100)',
                         line=dict(color='red', dash='dash'), yaxis='y2'))
fig.add_trace(go.Scatter(x=data.index, y=data['Particle_Volatility'] * 1000, name='Particle Volatility (x1000)',
                         line=dict(color='purple'), yaxis='y3'))
fig.add_trace(go.Scatter(x=data.index, y=data['Hurst_Threshold'] * 100, name='QCO Hurst Threshold (x100)',
                         line=dict(color='blue', dash='dot'), yaxis='y2'))

# Candlestick patterns
for date, pattern in patterns:
    if pattern:
        fig.add_annotation(x=date, y=data.loc[date, 'High'] * 1.02, text=pattern, showarrow=True, arrowhead=1, ax=20,
                           ay=-30)

# Trading signals
buy_signals = data[data['Signal'] == 'Buy']
sell_signals = data[data['Signal'] == 'Sell']
fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['Low'] * 0.98, mode='markers', name='Buy Signal',
                         marker=dict(symbol='triangle-up', size=10, color='green')))
fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['High'] * 1.02, mode='markers', name='Sell Signal',
                         marker=dict(symbol='triangle-down', size=10, color='red')))

# Update layout
fig.update_layout(
    title=f"{ticker} Stock with Quantum-Charged Indicators",
    yaxis=dict(title="Price (USD)"),
    yaxis2=dict(title="Hurst Exponent", overlaying='y', side='right', range=[0, 100]),
    yaxis3=dict(title="Volatility", overlaying='y', side='right', range=[0, 20], showgrid=False),
    xaxis_title="Date",
    showlegend=True,
    height=800
)

# Display chart
st.plotly_chart(fig, use_container_width=True)

# Display data and metrics
st.subheader("Stock Data and Signals")
st.dataframe(data[['Open', 'High', 'Low', 'Close', 'Hurst', 'Hurst_Up', 'Hurst_Down', 'Particle_Volatility',
                   'Hurst_Threshold', 'Signal']])
st.subheader("Performance Metrics")
st.write(f"Cumulative Return: {cum_return:.2%}")
st.write(f"S&P 500 Return: {sp500_return:.2%}")
st.write(f"Sharpe Ratio: {sharpe:.2f}")
st.write(f"Sortino Ratio: {sortino:.2f}")
st.write(f"Max Drawdown: {max_drawdown:.2%}")
st.write(f"Calmar Ratio: {calmar:.2f}")
st.write(f"Number of Trades: {trades}")
st.write(f"Win Rate: {win_rate:.2%}")

