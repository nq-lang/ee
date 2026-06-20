# ⬡ QUANTITATIVE BACKTESTING TERMINAL

A professional-grade, institutional-style backtesting harness for systematic
trading strategies written in C++ and Python.  Runs entirely inside a
**Python / Streamlit** terminal — no paid data subscriptions required for
prototyping.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Installation](#installation)
3. [Directory Structure](#directory-structure)
4. [Preparing CSV Data Files](#preparing-csv-data-files)
5. [Preparing C++ Strategy Files](#preparing-c-strategy-files)
6. [How to Run the Terminal](#how-to-run-the-terminal)
7. [Terminal Sections — User Guide](#terminal-sections--user-guide)
8. [Free Historical Data Sources](#free-historical-data-sources)
9. [Known Limitations & Workarounds](#known-limitations--workarounds)
10. [Configuration Reference](#configuration-reference)
11. [C++ Interface Contract](#c-interface-contract)
12. [Frequently Asked Questions](#frequently-asked-questions)

---

## Quick Start

```bash
# 1. Clone / download the terminal
git clone https://github.com/yourname/quant-terminal
cd quant-terminal

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

---

## Installation

### Requirements

- Python **3.10+**
- `g++` (optional — only needed for native C++ compilation mode)

### Install All Dependencies

```bash
pip install -r requirements.txt
```

### Optional: Install `kaleido` for PDF chart export

```bash
pip install kaleido
```

### Optional: Verify `g++` is available

```bash
g++ --version
```

If `g++` is not found, native C++ compilation mode will be disabled but all
Python-based backtesting features remain fully functional.

---

## Directory Structure

```
quant_terminal/
├── app.py                  ← Main Streamlit entry point (launch this)
├── config.py               ← Instrument specs, tick values, colour palette
├── data_loader.py          ← CSV loading, yfinance API, parquet caching
├── strategy_parser.py      ← C++ source parsing engine (regex-based AST)
├── backtest_engine.py      ← Bar-by-bar backtesting loop + C++ exec path
├── metrics.py              ← All performance metric calculations
├── monte_carlo.py          ← Monte Carlo simulation engine (5 sampling modes)
├── charts.py               ← All Plotly chart construction functions
├── ui_components.py        ← Reusable dark-theme CSS + UI building blocks
├── export.py               ← PDF, CSV, JSON export functions
├── requirements.txt        ← Python dependencies
│
├── cpp/
│   └── strategy_template.cpp   ← C++ interface contract template
│
├── config/
│   └── run_config_schema.json  ← Config file schema (auto-generated per run)
│
└── data/
    ├── raw/                ← Downloaded raw data (yfinance cache)
    ├── processed/          ← Normalized CSV/Parquet files fed to the engine
    └── results/            ← C++ binary output: trades.csv, equity.csv
```

---

## Preparing CSV Data Files

### Supported Formats

The terminal accepts CSV files from any platform:

| Platform         | Expected Format                                  |
|-----------------|--------------------------------------------------|
| TradeStation     | Date, Time, Open, High, Low, Close, Volume       |
| NinjaTrader      | Date, Time, Open, High, Low, Close, Volume       |
| Sierra Chart     | Date, Time, Open, High, Low, Close, Volume, OI  |
| Rithmic          | Timestamp, Open, High, Low, Last, Volume        |
| CQG              | Date/Time, Open, High, Low, Close, Volume        |
| Interactive Brokers | datetime, open, high, low, close, volume     |
| yfinance         | (fetched automatically — no CSV needed)          |

### Minimum Required Columns

```
Open, High, Low, Close  — price columns (names case-insensitive)
```

`Volume` is optional but recommended for VWAP indicator calculation.

### Timestamp Column

The terminal accepts many timestamp formats:

```
# Separate date + time columns:
Date,Time,Open,High,Low,Close,Volume
2024-01-02,09:30:00,16820.25,16835.50,16815.00,16828.75,12345

# Combined datetime:
Datetime,Open,High,Low,Close,Volume
2024-01-02 09:30:00,16820.25,16835.50,16815.00,16828.75,12345

# ISO 8601:
timestamp,open,high,low,close,volume
2024-01-02T09:30:00Z,16820.25,16835.50,16815.00,16828.75,12345
```

### Multi-Year Files

You can upload multiple CSV files at once (e.g. one per year). The terminal
will concatenate them chronologically and remove duplicates automatically.

### Instrument Auto-Detection

Upon upload, the terminal scans:
1. **Price range** — NQ typically 8,000–25,000; ES typically 1,800–7,000
2. **Filename keywords** — `NQ`, `ES`, `MNQ`, `MES`, `nasdaq`, `sp500`, `emini`
3. **Column headers** — any `symbol`, `instrument`, or `ticker` column

If detection is confident a green banner appears. If ambiguous, select the
instrument manually in **Section 1 — Configuration**.

---

## Preparing C++ Strategy Files

### File Requirements

- File extension: `.cpp`, `.h`, `.hpp`, or `.cxx`
- Maximum file size: 10 MB per file
- Multiple files accepted (combined before parsing/compilation)
- C++ standard: C++23 (compiled with `g++ -std=c++23 -O2`)

### What the Parser Extracts

The Python parser scans your C++ source and extracts:

| Element | What the Parser Looks For |
|---------|--------------------------|
| **Signal names** | String literals or constants like `"MOM_LONG"`, `MOM_SHORT`, `ABSORPTION_SHORT` |
| **Entry conditions** | `if` blocks containing signal assignments |
| **Stop loss logic** | Variables named `stop`, `sl`, `stopPrice`, `stopLevel` |
| **Take profit logic** | Variables named `target`, `tp`, `takeProfit`, `tpLevel` |
| **ATR/sigma stops** | Patterns like `1.5 * atr`, `0.75 * sigma` |
| **R-multiple targets** | Patterns like `2.0 * risk`, `1.5 * stop` |
| **Constants** | `#define`, `const double`, `constexpr` numeric values |
| **Regime labels** | String literals matching `TRENDING`, `ROTATIONAL`, `MEAN_REVERT`, `VOLATILE` |
| **Indicators** | References to `vwap`, `atr`, `delta`, `cvd`, `momentum`, `rsi`, etc. |
| **Position sizing** | Changes to `size`, `qty`, `contracts` variables |
| **Cooldown logic** | `cooldown`, `minBars`, `barsSince` comparisons |
| **Daily limits** | `dailyLoss`, `dailyLimit` comparisons |

### Using the C++ Template

Copy `cpp/strategy_template.cpp` and replace the `generate_signal()` function
body with your own strategy logic:

```cpp
static std::string generate_signal(const Bar& bar, const Bar& prev,
                                    const std::string& regime) {
    // YOUR STRATEGY LOGIC HERE
    // Example:
    if (bar.z_ret > 0.45 && bar.z_mom > 0.30 && bar.close > bar.vwap)
        return "MOM_LONG";
    if (bar.z_ret < -0.45 && bar.z_mom < -0.30 && bar.close < bar.vwap)
        return "MOM_SHORT";
    return "";   // no signal this bar
}
```

### Native Compilation Mode (Advanced)

Enable **"Compile & Execute C++ Natively"** in Section 2 to:
1. Write your `.cpp` files to a temporary directory
2. Compile with `g++ -std=c++23 -O2`
3. Execute the binary with the normalized data CSV
4. Parse `data/results/trades.csv` + `data/results/equity.csv`

If compilation fails, the terminal automatically falls back to the Python
simulation engine and notifies you of the fallback.

---

## How to Run the Terminal

### Launch

```bash
streamlit run app.py
```

### Typical Workflow

1. **Section 1 — Configuration**: Select instrument (NQ/ES/MNQ/MES), set account
   balance, contracts, commission, slippage, stop/target parameters.

2. **Section 2 — Strategy Loader**: Upload your `.cpp` file or use the built-in
   demo strategy (MOM_LONG / MOM_SHORT). Review the parse report.

3. **Section 3 — Data Loader**: Upload CSV data **or** use the yfinance API
   tab to fetch daily futures data automatically.

4. **Section 4 — Backtest Engine**: Review the pre-run checklist. Click
   **▶ RUN BACKTEST**. Watch the live progress bar.

5. **Section 5 — Results Dashboard**: View KPI cards, equity curve, annual
   P&L chart, and detailed trade-level statistics.

6. **Section 6 — Candlestick Chart**: Inspect trade entries/exits overlaid on
   price bars. Filter by signal type or outcome.

7. **Section 7 — Monte Carlo**: Forward-project robustness. Configure stress
   tests and combine challenge simulation.

8. **Section 8 — Annual Breakdown**: Year / month / quarter / DOW / TOD splits.

9. **Section 9 — Signal History Log**: Full paginated trade table with filters.

10. **Section 10 — Risk Metrics**: Complete institutional metric suite.

11. **Section 11 — Export**: Download PDF report, trade log CSV, equity CSV,
    Monte Carlo CSV, and config JSON.

---

## Terminal Sections — User Guide

### Section 1: Configuration & Inputs

| Subsection | Key Settings |
|------------|-------------|
| Instrument Selector | NQ / ES / MNQ / MES — governs all P&L calculations |
| Account & Capital | Starting balance, contract count, sizing mode |
| Commission & Fees | Per-side commission, exchange fees, NFA fee, slippage ticks |
| Stop & Target | Mode (Points / Ticks / Dollars / R-Multiple / Strategy), value |
| Partial Exits | % to exit at first target, trailing remainder options |
| Time Stops | Max bars in trade, EOD forced exit time |
| Date Range | Full history / custom range / last N days / last N years |
| Session Filter | Full session / RTH only / overnight / pre-market |
| DOW Filter | Select which weekdays to include |
| Display Scaling | Scale all P&L instantly without re-running (e.g. 1 NQ = 10 MNQ) |

### Section 7: Monte Carlo Sampling Methods

| Method | Description |
|--------|-------------|
| Trade-by-Trade Bootstrap | Resample individual trades with replacement |
| Daily P&L Bootstrap | Resample daily P&L totals with replacement |
| Parametric (Normal) | Fit N(μ,σ) to trade P&L, sample forward |
| Parametric (T-Distribution) | Fit heavy-tailed t-dist (better for fat tails) |
| Block Bootstrap | Resample contiguous blocks — preserves autocorrelation |

---

## Free Historical Data Sources

### Futures (NQ / ES)

| Source | Access | Notes |
|--------|--------|-------|
| **yfinance** (built-in) | Free | Daily bars only via `NQ=F` / `ES=F`. Intraday uses SPY/QQQ proxy (×40 / ×10 scaling). Not suitable for live production. |
| **Databento** | Pay-as-you-go | Clean CME continuous contracts, 1-min to tick. Best free-tier trial available. |
| **Norgate Data** | Subscription | Back-adjusted continuous contracts, local database. |
| **Interactive Brokers** | Account required | Historical bars via TWS API (`ib_insync`). |

### Options (SPY / QQQ)

| Source | Access | Notes |
|--------|--------|-------|
| **yfinance** (built-in) | Free | Current chains only — no historical snapshots. |
| **Kaggle / GitHub datasets** | Free | Historical EOD chains for SPY/QQQ. Download `.parquet` files and place in `data/processed/options_eod_spy.parquet`. |
| **ThetaData** | Affordable | Dedicated options backtesting data, REST API. |
| **CBOE DataShop** | Paid | Official SPY/QQQ option historical data. |

### Loading Historical Options Parquet Files

1. Download a free EOD options dataset (e.g. from [github.com/optionstrat](https://github.com/optionstrat))
2. Place the file at:
   ```
   data/processed/options_eod_spy.parquet
   data/processed/options_eod_qqq.parquet
   ```
3. The terminal will automatically detect and query these files in Section 11.

---

## Known Limitations & Workarounds

### 1. yfinance Intraday Futures Data

**Limitation**: `yfinance` only provides daily OHLCV for `NQ=F` / `ES=F`.
Intraday requests use `SPY` / `QQQ` as proxies scaled by ×40 / ×10.

**Workaround**: Upload your own 1-min / 5-min CSV data (from NinjaTrader,
Sierra Chart, TradeStation, Rithmic, etc.) for proper intraday backtesting.

### 2. No Historical Options Chains via yfinance

**Limitation**: `yfinance` only returns the *current* options chain.

**Workaround**: Download free historical EOD options data from Kaggle or
GitHub (optionstrat datasets) and place `.parquet` files in `data/processed/`.

### 3. C++ Parser Coverage

**Limitation**: The regex-based parser has ~60–90% coverage on typical
systematic strategy code. Highly obfuscated code, templates, or unusual naming
conventions may not be fully parsed.

**Workaround**: The parse report clearly shows what was and was not detected.
For undetected logic, manually override stop/target settings in Section 1
before running. Use native compilation mode for exact replication.

### 4. Large Datasets (2.5M+ bars)

**Limitation**: Pure Python bar-by-bar iteration on 10 years of 1-min data
(~2.5M bars) takes approximately 30–90 seconds depending on hardware.

**Workaround**: Use native C++ compilation mode for maximum speed. Alternatively,
downsample to 5-min bars for exploratory runs and use 1-min only for final
validation.

### 5. Monte Carlo Memory

**Limitation**: 10,000 paths × 252 days stores 2.52M floats (~20 MB). For
very large simulations (50,000+ paths), memory usage may be significant.

**Workaround**: Use 1,000–5,000 paths for typical analysis. Results are
statistically stable above 1,000 paths for most strategies.

### 6. PDF Export Without kaleido

**Limitation**: Chart images are not embedded in PDF reports unless `kaleido`
is installed.

**Workaround**: `pip install kaleido`. Without it, PDF still exports — just
without chart images. All tables and metrics are included.

---

## Configuration Reference

### Instrument Specs

| Symbol | Name | Point Value | Tick Size | Tick Value |
|--------|------|------------|-----------|------------|
| NQ | E-mini NASDAQ-100 | $20/pt | 0.25 | $5.00 |
| ES | E-mini S&P 500 | $50/pt | 0.25 | $12.50 |
| MNQ | Micro E-mini NASDAQ-100 | $2/pt | 0.25 | $0.50 |
| MES | Micro E-mini S&P 500 | $5/pt | 0.25 | $0.25 |

### Default Commission Structure

| Fee | Default | Notes |
|-----|---------|-------|
| Commission | $0.50/contract/side | Varies by broker |
| Exchange Fee | $0.85/contract | CME Globex standard |
| NFA Fee | $0.02/contract | Fixed regulatory |
| Slippage | 1 tick | Adjustable per strategy |
| **Total RT (NQ)** | **$2.74 + $10/slip** | Per contract per round trip |

---

## C++ Interface Contract

### Config File: `run_config.json`

Auto-generated by the terminal at `config/run_config.json` before each run.

```json
{
  "schema_version":   "1.0",
  "instrument":       "NQ",
  "data_file":        "data/processed/backtest_input.csv",
  "trades_output":    "data/results/trades.csv",
  "equity_output":    "data/results/equity.csv",
  "starting_balance": 50000.0,
  "num_contracts":    1,
  "commission":       2.74,
  "slippage_ticks":   1,
  "eod_exit":         true,
  "eod_exit_time":    "15:45",
  "daily_loss_limit": 3000.0
}
```

### Input Data: `backtest_input.csv`

```
timestamp,open,high,low,close,volume
2020-01-02T09:30:00Z,16820.25,16835.50,16815.00,16828.75,12345
```

### Output: `trades.csv`

```
trade_id,signal,direction,entry_time,exit_time,entry_price,exit_price,
stop_price,target_price,contracts,outcome,gross_pnl,commission,
slippage,net_pnl,hold_bars,hold_minutes,mae,mfe,is_gap_fill
```

### Output: `equity.csv`

```
timestamp,equity,drawdown,drawdown_pct
```

### Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Config file error |
| 2 | Data file error |
| 3 | Runtime error |

---

## Frequently Asked Questions

**Q: Can I use this with Python-only strategies (no C++)?**
A: Yes — the demo strategy and the Python backtest engine run without any C++
file. The C++ loader is optional. All 11 sections work fully without it.

**Q: How accurate is the C++ parser?**
A: Typical systematic strategy files parse at 60–95% confidence. The parse
report tells you exactly what was and was not extracted. Use native compilation
mode for 100% fidelity to your C++ logic.

**Q: Can I backtest options strategies?**
A: The data pipeline supports options chains (via yfinance or local parquet).
The backtesting engine is currently optimised for futures. Options strategy
backtesting requires uploading historical chain data and is a planned
enhancement.

**Q: The equity curve looks wrong after scaling. Why?**
A: Display scaling (Section 1) multiplies *relative P&L* by the scale factor.
It does not re-run the backtest — it is a display multiplier only. For exact
scaled results, change `num_contracts` in Section 1 and re-run.

**Q: How do I replicate a specific TradeStation report format?**
A: Export your TS data as CSV (OHLCV format), upload it in Section 3. The
annual and trade-level statistics in Sections 5 and 8 match standard TS report
columns. Use Section 11 to download the full CSV trade log for side-by-side
comparison.

**Q: Can multiple strategies run simultaneously?**
A: Not in one session — the terminal is single-strategy per session. Run each
strategy in a separate browser tab (`streamlit run app.py --server.port 8502`).

---

## Support & Development

- All metric formulas are documented in source comments with references
- All errors are caught and displayed as styled banners — no raw tracebacks
- The terminal is designed for iterative use: upload → configure → run → adjust

```
streamlit run app.py
```

---

*Built to institutional specification. All 11 sections are fully implemented.*
*Zero placeholder panels.*
