# **Sniper Trading System**

```
███████╗███╗   ██╗██╗██████╗ ███████╗██████╗ 
██╔════╝████╗  ██║██║██╔══██╗██╔════╝██╔══██╗
███████╗██╔██╗ ██║██║██████╔╝█████╗  ██████╔╝
╚════██║██║╚██╗██║██║██╔═══╝ ██╔══╝  ██╔══██╗
███████║██║ ╚████║██║██║     ███████╗██║  ██║
╚══════╝╚═╝  ╚═══╝╚═╝╚═╝     ╚══════╝╚═╝  ╚═╝
```

**High-frequency algorithmic trading system for MNQ futures with real-time tick execution**

---

## 🎯 **Overview**

Sniper is a sophisticated multi-strategy trading system that combines advanced technical analysis with machine learning for MNQ (Micro NASDAQ-100) futures trading. The system features real-time tick-based execution, achieving 100-200x faster response times compared to traditional bar-close systems.

### **Key Features**
- **⚡ Real-time Execution**: Tick-based decision making with 50-200ms latency
- **🧠 Multi-Strategy**: Mean reversion, momentum, volatility carry, and breakout strategies
- **🤖 AI-Powered**: Machine learning allocation and reinforcement learning execution
- **🛡️ Risk Management**: Position limits, daily loss limits, and emergency controls
- **📊 Professional Integration**: NinjaTrader 8 with TCP socket communication

---

## 🏗️ **Architecture**

```
┌─────────────────┐    TCP    ┌──────────────────┐
│   NinjaTrader   │◄──5556───►│   Python Core    │
│   (C# Strategy) │   5557    │  (Trading Brain) │
└─────────────────┘           └──────────────────┘
         │                             │
         ▼                             ▼
  ┌─────────────┐              ┌─────────────────┐
  │ Market Data │              │ Strategy Engine │
  │ OHLC + Ticks│              │ • Mean Reversion│
  │ Real-time   │              │ • Momentum      │
  └─────────────┘              │ • Vol Carry     │
                               │ • Vol Breakout  │
                               └─────────────────┘
```

---

## 🚀 **Tech Stack**

### **Core Platform**
- **Python 3.12** - Trading logic and ML models
- **NinjaTrader 8** - Market data and order execution
- **DuckDB** - High-performance data storage

### **Data & Analytics**
- **pandas/polars** - Data manipulation
- **numpy** - Numerical computing  
- **vectorbt** - Backtesting framework

### **Machine Learning**
- **scikit-learn 1.6** - Traditional ML models
- **LightGBM 4** - Gradient boosting
- **Stable-Baselines3** - Reinforcement learning (PPO)

### **Infrastructure**
- **Prefect** - Workflow orchestration
- **TCP Sockets** - Real-time communication
- **Threading** - Concurrent processing

---

## 📁 **Project Structure**

```
sniper_v1/
├── src/
│   ├── strategies/          # Trading strategy implementations
│   │   ├── mean_reversion.py
│   │   ├── momentum.py
│   │   ├── volatility_carry.py
│   │   └── volatility_breakout.py
│   ├── models/              # ML/RL models
│   │   ├── meta_allocator.py
│   │   └── ppo_execution.py
│   ├── orchestration/       # System coordination
│   │   ├── signal_processor.py
│   │   ├── risk_manager.py
│   │   └── connection_manager.py
│   ├── infra/              # Infrastructure
│   │   └── nt_bridge.py    # NinjaTrader communication
│   ├── utils/              # Utilities
│   │   ├── data_manager.py
│   │   └── price_history_manager.py
│   ├── config.py           # System configuration
│   └── main.py            # Main orchestrator
├── ninja/                  # NinjaScript integration
│   └── NinjaTraderStrategy.cs
├── data/                  # Market data storage
├── tests/                 # Test suite
└── README.md             # This file
```

---

## ⚙️ **Installation & Setup**

### **1. Python Environment**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **2. NinjaTrader Setup**
1. Install NinjaTrader 8
2. Import `NinjaTraderStrategy.cs` as a custom strategy
3. Configure TCP ports (5556 for data, 5557 for signals)
4. Enable real-time data feed

### **3. Configuration**
```bash
# Set environment variables
export ENABLE_ML_ALLOCATOR=true
export ENABLE_RL_EXECUTION=true
export FORCE_MARKET_ORDERS=true
```

---

## 🎮 **Usage**

### **Start Trading System**
```bash
# Start the main trading system
python src/main.py

# With specific options
python src/main.py --no-ml --no-rl  # Disable ML/RL
```

### **Data Management**
```bash
# Download historical data
make dl symbol=MNQ

# Run backtests
make backtest strat=all

# Train ML models
make train-ml window=60
make train-rl steps=5e6
```

### **Live Trading**
```bash
# Start live trading (requires NinjaTrader connection)
make live
```

---

## 📊 **Strategies**

### **1. Mean Reversion**
- **Logic**: RSI divergence + VWAP mean reversion
- **Timeframes**: 5m, 15m analysis
- **Risk**: 2.0x ATR stops

### **2. Momentum** 
- **Logic**: EMA crossovers + trend strength
- **Timeframes**: 30m, 1h analysis
- **Confirmation**: Volume + duration filters

### **3. Volatility Carry**
- **Logic**: Term structure slope analysis
- **Signals**: Contango/backwardation detection
- **Edge**: Volatility mean reversion

### **4. Volatility Breakout**
- **Logic**: Statistical volatility expansion
- **Triggers**: 2.0σ+ breakouts from mean
- **Timing**: Immediate breakout capture

---

## 🛡️ **Risk Management**

### **Position Limits**
- **Max Contracts**: 5 micro futures
- **Daily Loss**: 50% of equity
- **Emergency Close**: Automatic risk breach response

### **Technical Controls**
- **ATR-based Stops**: Dynamic based on market volatility
- **Real-time Monitoring**: Tick-level risk assessment
- **Connection Monitoring**: Auto-reconnect on failures

---

## 🔧 **Configuration**

### **Key Settings** (`src/config.py`)
```python
# Risk Management
max_position_size: int = 5
max_daily_loss_pct: float = 0.5
stop_loss_atr_multiplier: float = 2.0

# Strategy Parameters
min_confidence: float = 0.6
force_market_orders: bool = True

# Network
data_port: int = 5556
signal_port: int = 5557
```

### **Environment Variables**
```bash
TRADING_MODE=live                    # live/paper/backtest
LOG_LEVEL=INFO                      # DEBUG/INFO/WARNING/ERROR
ENABLE_ML_ALLOCATOR=true            # Enable ML allocation
ENABLE_RL_EXECUTION=false           # Enable RL execution
FORCE_MARKET_ORDERS=true            # Market orders only
```

---

## 📈 **Performance**

### **Execution Speed**
- **Traditional**: 30-60 seconds (bar close)
- **Sniper**: 50-200 milliseconds (real-time)
- **Improvement**: **150-1200x faster**

### **Strategy Performance**
- **Required Metrics**: Sharpe ≥ 1.2, Calmar ≥ 0.8
- **Walk-forward**: 1-year out-of-sample validation
- **Slippage**: Included in all backtests

---

## 🔬 **Testing**

### **Run Test Suite**
```bash
# Unit tests
pytest tests/

# Strategy validation
python test_integration.py

# Real-time execution test
python test_true_realtime.py
```

### **Validation Process**
1. **Historical Backtesting**: 10+ years of data
2. **Paper Trading**: Live market validation
3. **Walk-forward Analysis**: Out-of-sample testing
4. **Risk Scenario Testing**: Stress testing

---

## 📡 **Communication Protocol**

### **Message Types**
- **`historical_data`**: Initial bar data (startup)
- **`live_data`**: Complete market data (bar close)
- **`realtime_tick`**: Lightweight tick data (real-time)
- **`trade_completion`**: Execution confirmations

### **TCP Flow**
```
NinjaTrader → Python: Market data (port 5556)
Python → NinjaTrader: Trade signals (port 5557)
```

---

## 🚨 **Monitoring & Alerts**

### **System Health**
- **Connection Status**: NinjaTrader bridge monitoring
- **Data Quality**: Timestamp validation, price checks
- **Performance**: Execution latency tracking

### **Trading Alerts**
- **Risk Breaches**: Position/loss limit violations
- **Strategy Signals**: Trade generation notifications
- **System Events**: Startup, shutdown, errors

---

## 🐛 **Troubleshooting**

### **Common Issues**

**Connection Problems**
```bash
# Check NinjaTrader strategy is running
# Verify TCP ports 5556/5557 are open
# Restart bridge: python src/main.py
```

**Data Issues**
```bash
# Timestamp mismatches: Automatically handled
# Missing bars: Check NT data feed
# Invalid prices: Built-in validation
```

**Performance Issues**
```bash
# High CPU: Reduce tick processing frequency
# Memory usage: Check data buffer sizes
# Latency: Verify network connectivity
```

---

## 📝 **Development**

### **Code Style**
- **PEP 8** compliance with `black` formatting
- **Type hints** for all functions
- **Google docstrings** for documentation
- **Max 50 lines** per function

### **Contributing**
1. Branch: `feature/<ticket>-desc` or `bugfix/<desc>`
2. Test: All tests must pass
3. Commit: Include `Backtest:` line if PnL changes
4. PR: Attach backtest report + NT replay diff

---

## ⚠️ **Disclaimer**

**This software is for educational and research purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.**

---

## 📄 **License**

**Proprietary** - All rights reserved

---

## 📞 **Contact**

For questions or support regarding the Sniper trading system, please refer to the internal documentation or contact the development team.

---

*Built with precision. Engineered for speed. Designed to dominate markets.* 🎯