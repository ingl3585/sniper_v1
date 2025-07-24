# claude.md
# Purpose: Define a two-phase refactor and indicator implementation for a trading repo using Claude Code via CLI. 
# This document must be followed exactly. All instructions are deterministic, CLI-safe, and idempotent.

--------------------------------------------------------------------------------
Objective:
Refactor the algorithmic trading repository into a clean, hexagonal layout and implement two core indicators:
- FVDR (Flow-Vol Drift Ratio)
- NFVGS (Normalized Fair Value Gap Strength)

The goals are:
- Clean and testable architecture
- Shared, typed domain models
- Accurate and centralized technical indicators
- A single-pass CLI validation script (quick_check.py)

--------------------------------------------------------------------------------
Global Rules:
- Python version: 3.11+
- Use mypy strict mode compliance
- All functions must use full type hints
- All input/output arrays must be np.ndarray or List[float] with shape (N,)
- Use absolute imports only (from src.core import Bar), never relative
- snake_case for functions/vars; PascalCase for class names
- Every public symbol must have a Google-style docstring
- Every file must use: log = get_logger(__name__)
- Do not invent placeholder features, add extra libraries, or introduce notebooks

--------------------------------------------------------------------------------
PHASE 1 — ARCHITECTURE CLEANUP + CORE INDICATORS

Target layout:
repo/
├── src/
│   ├── core/
│   │   ├── models.py             # frozen Pydantic dataclasses: Bar, Tick, TradeSignal, Position
│   │   ├── constants.py          # central RISK_* sizing multipliers (RISK.PER_TRADE, etc.)
│   │   └── logging_config.py     # structlog JSON logger with get_logger()
│   ├── services/
│   │   ├── technical_indicators.py   # canonical source of ATR, VWAP, RSI, FVDR, NFVGS
│   │   └── data_validation.py        # Pydantic validators for inbound socket JSON
│   ├── strategies/              # e.g., mean_reversion.py, momentum.py
│   ├── engine/                  # meta_allocator.py, position_sizer.py, risk_manager.py
│   └── infra/
│       ├── nt_bridge.py         # socket IO layer
│       ├── live_gateway.py      # order routing
│       └── storage.py           # DuckDB helpers
├── scripts/
│   └── quick_check.py           # CLI entry point for single-session sanity pass
└── conf/                        # reserved for future Hydra YAMLs

--------------------------------------------------------------------------------
Tasks:

1. File Moves + Imports
- Move files to the directory layout above
- Add __init__.py barrels where appropriate so code can `from src.core import Bar`
- Fix all import paths to use absolute import syntax
- Ensure no circular imports

2. Logging Setup (src/core/logging_config.py)
def get_logger(name: str):
    import logging, structlog
    logging.basicConfig(level=logging.INFO)
    structlog.configure(processors=[structlog.processors.JSONRenderer()])
    return structlog.get_logger(name)

3. Models (src/core/models.py)
- Define the following frozen=True Pydantic dataclasses:
  - Bar
  - Tick
  - TradeSignal
  - Position
- Each class must have full type hints and per-field docstrings

4. Indicators (src/services/technical_indicators.py)
- Keep the current ATR, VWAP, RSI logic
- Add:

def calculate_fvdr(buys, sells, highs, lows, closes, atr_period=14):
    """Flow-Vol Drift Ratio (FVDR)
    Inputs: Lists or arrays of float, shape (N,)
    Logic:
    - net_flow[t] = buys[t] - sells[t]
    - tr[t] = max(high[t] - low[t], abs(high[t] - close[t-1]), abs(low[t] - close[t-1]))
    - clip tr > 0.0001
    - raw_ratio = net_flow / tr
    - Return: Wilder EMA(raw_ratio, period=14)
    """

def calculate_nfvgs(highs, lows, closes, atr_period=14, decay_ema=5):
    """Normalized Fair Value Gap Strength (NFVGS)
    - bullish gap if low[t-1] > high[t-3]
    - gap_size = |low[t-1] - high[t-3]|
    - score = gap_size / ATR14
    - divide score by (age+1) while unfilled
    - smooth with EMA(alpha=1/decay_ema)
    - output: +bullish, -bearish signal
    """

5. Sanity Checks
- Add:
assert not np.isnan(fvdr).any()
assert abs(atr14[-1] - np.mean(last14_tr)) < 1e-6

6. Strategy Cleanup
- Delete all duplicate instances of ATR/VWAP/RSI in strategies/
- Replace all usages with imports from src.services.technical_indicators
- Replace hard-coded risk values (e.g., 1.2 * atr) with constants from src.core.constants

7. CLI Check (scripts/quick_check.py)
- Must be runnable as: python -m scripts.quick_check --date YYYY-MM-DD
- Load 1 historical session via storage.py
- Pipe through indicators, generate a TradeSignal, send to nt_bridge
- If no data available, generate synthetic OHLCV with 50 bars and dummy signal
- Print:
  - "ATR diff vs pandas-ta: [value]"
  - "FVDR tail: [fvdr[-3:]]"
  - "NFVGS tail: [nfvgs[-3:]]"
  - "Signal OK" if trade signal dispatch successful
- Exit cleanly with sys.exit(0) on success

8. Format and Lint
Run the following at repo root:
ruff --fix .
black .

--------------------------------------------------------------------------------
PHASE 2 — FEATURE WIRING

1. Strategy Logic Update
- Momentum: size multiplier = fvdr if fvdr > 0 else 0
- Mean-reversion: skip signal generation when abs(nfvgs) > 0.5

2. Feature Injection (src/engine/meta_allocator.py)
- Add fvdr and nfvgs to LightGBM feature vector
- Retrain meta-allocator using updated feature set

3. Documentation (docs/alpha_design.md)
- Append both indicator formulas with math, example parameters, and rationale
- Include a summary block like:
  - FVDR: momentum proxy based on order flow / range
  - NFVGS: gap exhaustion risk scored over time

--------------------------------------------------------------------------------
Deliverables (pass conditions):

- [x] src/ matches hex layout
- [x] All models are typed and shared
- [x] Indicators centralized in services/technical_indicators.py
- [x] No repeated helper code in strategies/*
- [x] constants.py houses all RISK multipliers
- [x] scripts/quick_check.py passes with CLI args or fallback mode
- [x] Output includes FVDR, NFVGS preview, ATR diff, and "Signal OK"
- [x] ruff and black pass cleanly

--------------------------------------------------------------------------------
PROGRESS TRACKER

Use this section to track task completion. Claude Code or any CLI agent should update it as each item is completed successfully. All checkboxes default to [ ] and must be marked [x] when done.

PHASE 1 — ARCHITECTURE + INDICATORS

[x] Restructure repo layout into src/core, services, strategies, engine, infra
[x] Add __init__.py barrels and fix all imports to use absolute paths
[x] Create logging_config.py with get_logger()
[x] Define frozen Pydantic models: Bar, Tick, TradeSignal, Position
[x] Implement calculate_fvdr() in technical_indicators.py
[x] Implement calculate_nfvgs() in technical_indicators.py
[x] Add assert guards for fvdr and atr14 sanity
[x] Delete duplicated ATR/VWAP/RSI logic from strategies/*
[x] Replace hardcoded risk values with constants.RISK.*
[x] Create scripts/quick_check.py with full CLI fallback
[x] Confirm quick_check outputs: ATR diff, FVDR/NFVGS tail, "Signal OK"
[x] Run and pass: ruff --fix .
[x] Run and pass: black .

PHASE 2 — FEATURE WIRING

[x] Update momentum strategy to use fvdr > 0
[x] Update mean-reversion strategy to disable on abs(nfvgs) > 0.5
[x] Add fvdr and nfvgs as LightGBM features in meta_allocator
[x] Retrain meta-allocator with updated feature set

Legend:
[ ] = Not started
[~] = In progress
[x] = Complete


End of claude.md