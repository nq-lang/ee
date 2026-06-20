/**
 * strategy_template.cpp
 * ─────────────────────────────────────────────────────────────────────────────
 * QUANT TERMINAL — C++ Strategy Interface Contract  (C++23)
 *
 * This file is the canonical template and documentation for writing a C++
 * trading strategy that integrates with the Python Streamlit backtest
 * terminal. Follow this exact interface contract so the terminal can:
 *   1. Compile this file with g++ -std=c++23 -O2
 *   2. Execute the binary:   ./strategy  path/to/run_config.json
 *   3. Parse trades.csv and equity.csv for the results dashboard
 *
 * C++ STRATEGY INTERFACE CONTRACT
 * ──────────────────────────────────────────────────────────────────────────────
 * INPUT  : run_config.json (path passed as argv[1])
 * OUTPUT : trades.csv      (one row per completed trade)
 *          equity.csv      (one row per bar — cumulative P&L + drawdown)
 *
 * CONFIG SCHEMA (run_config.json)
 * ──────────────────────────────────────────────────────────────────────────────
 * {
 *   "schema_version"  : "1.0",
 *   "instrument"      : "NQ",
 *   "data_file"       : "data/processed/backtest_input.csv",
 *   "trades_output"   : "data/results/trades.csv",
 *   "equity_output"   : "data/results/equity.csv",
 *   "starting_balance": 50000.0,
 *   "num_contracts"   : 1,
 *   "commission"      : 2.74,        // total round-trip per contract
 *   "slippage_ticks"  : 1,
 *   "eod_exit"        : true,
 *   "eod_exit_time"   : "15:45",
 *   "daily_loss_limit": 3000.0       // 0 = disabled
 * }
 *
 * DATA FILE SCHEMA (backtest_input.csv)
 * ──────────────────────────────────────────────────────────────────────────────
 * timestamp,open,high,low,close,volume
 * 2020-01-02T09:30:00Z,16820.25,16835.50,16815.00,16828.75,12345
 * ...
 *
 * TRADES OUTPUT SCHEMA (trades.csv)
 * ──────────────────────────────────────────────────────────────────────────────
 * trade_id,signal,direction,entry_time,exit_time,entry_price,exit_price,
 * stop_price,target_price,contracts,outcome,gross_pnl,commission,
 * slippage,net_pnl,hold_bars,hold_minutes,mae,mfe,is_gap_fill
 *
 * EQUITY OUTPUT SCHEMA (equity.csv)
 * ──────────────────────────────────────────────────────────────────────────────
 * timestamp,equity,drawdown,drawdown_pct
 *
 * EXIT CODES
 * ──────────────────────────────────────────────────────────────────────────────
 * 0 = success
 * 1 = config file not found / parse error
 * 2 = data file not found / parse error
 * 3 = runtime error during backtest
 */

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

// ─── Instrument specs (point values per contract) ────────────────────────────
static const std::unordered_map<std::string, double> POINT_VALUE = {
    {"NQ", 20.0}, {"ES", 50.0}, {"MNQ", 2.0}, {"MES", 5.0},
};
static const std::unordered_map<std::string, double> TICK_VALUE = {
    {"NQ", 5.0}, {"ES", 12.50}, {"MNQ", 0.50}, {"MES", 0.25},
};

// ─── Config ──────────────────────────────────────────────────────────────────
struct Config {
    std::string schema_version  = "1.0";
    std::string instrument      = "NQ";
    std::string data_file;
    std::string trades_output;
    std::string equity_output;
    double      starting_balance = 50'000.0;
    int         num_contracts    = 1;
    double      commission       = 2.74;      // round-trip per contract
    int         slippage_ticks   = 1;
    bool        eod_exit         = true;
    std::string eod_exit_time    = "15:45";
    double      daily_loss_limit = 0.0;
};

// ─── Bar ─────────────────────────────────────────────────────────────────────
struct Bar {
    std::string timestamp;
    double open, high, low, close;
    long   volume;
    // Pre-computed indicators
    double atr   = 0.0;
    double z_ret = 0.0;
    double z_mom = 0.0;
    double vwap  = 0.0;
};

// ─── Trade ───────────────────────────────────────────────────────────────────
struct Trade {
    int    trade_id    = 0;
    std::string signal;
    std::string direction;   // "LONG" | "SHORT"
    std::string entry_time;
    std::string exit_time;
    double entry_price  = 0.0;
    double exit_price   = 0.0;
    double stop_price   = 0.0;
    double target_price = 0.0;
    int    contracts    = 1;
    std::string outcome;     // "WIN" | "LOSS" | "TIME" | "EOD" | "GAP"
    double gross_pnl    = 0.0;
    double commission   = 0.0;
    double slippage     = 0.0;
    double net_pnl      = 0.0;
    int    hold_bars    = 0;
    double hold_minutes = 0.0;
    double mae          = 0.0;
    double mfe          = 0.0;
    bool   is_gap_fill  = false;
};

// ─── Simple JSON parser for run_config.json ──────────────────────────────────
static std::unordered_map<std::string, std::string>
parse_json_flat(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Config file not found: " + path);

    std::unordered_map<std::string, std::string> kv;
    std::string line;
    while (std::getline(f, line)) {
        // Extract: "key" : "value"  or  "key" : 123
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        auto key_start = line.find('"');
        auto key_end   = line.find('"', key_start + 1);
        if (key_start == std::string::npos || key_end == std::string::npos) continue;
        std::string key  = line.substr(key_start + 1, key_end - key_start - 1);

        std::string val_part = line.substr(colon + 1);
        // Strip whitespace, commas, quotes
        auto strip = [](std::string s) {
            while (!s.empty() && (s.front() == ' ' || s.front() == '"' || s.front() == ','))
                s.erase(s.begin());
            while (!s.empty() && (s.back()  == ' ' || s.back()  == '"' || s.back()  == ','))
                s.pop_back();
            return s;
        };
        kv[key] = strip(val_part);
    }
    return kv;
}

static Config load_config(const std::string& path) {
    auto kv = parse_json_flat(path);
    Config cfg;
    if (kv.count("instrument"))       cfg.instrument       = kv.at("instrument");
    if (kv.count("data_file"))        cfg.data_file        = kv.at("data_file");
    if (kv.count("trades_output"))    cfg.trades_output    = kv.at("trades_output");
    if (kv.count("equity_output"))    cfg.equity_output    = kv.at("equity_output");
    if (kv.count("starting_balance")) cfg.starting_balance = std::stod(kv.at("starting_balance"));
    if (kv.count("num_contracts"))    cfg.num_contracts    = std::stoi(kv.at("num_contracts"));
    if (kv.count("commission"))       cfg.commission       = std::stod(kv.at("commission"));
    if (kv.count("slippage_ticks"))   cfg.slippage_ticks   = std::stoi(kv.at("slippage_ticks"));
    if (kv.count("eod_exit"))         cfg.eod_exit         = kv.at("eod_exit") == "true";
    if (kv.count("eod_exit_time"))    cfg.eod_exit_time    = kv.at("eod_exit_time");
    if (kv.count("daily_loss_limit")) cfg.daily_loss_limit = std::stod(kv.at("daily_loss_limit"));
    return cfg;
}

// ─── Load CSV data ────────────────────────────────────────────────────────────
static std::vector<Bar> load_data(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Data file not found: " + path);

    std::vector<Bar> bars;
    std::string line;
    std::getline(f, line); // skip header

    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        Bar b;
        int col = 0;
        while (std::getline(ss, tok, ',')) {
            try {
                switch (col) {
                    case 0: b.timestamp = tok;                    break;
                    case 1: b.open      = std::stod(tok);         break;
                    case 2: b.high      = std::stod(tok);         break;
                    case 3: b.low       = std::stod(tok);         break;
                    case 4: b.close     = std::stod(tok);         break;
                    case 5: b.volume    = std::stol(tok);         break;
                }
            } catch (...) {}
            ++col;
        }
        if (col >= 5) bars.push_back(b);
    }
    return bars;
}

// ─── Pre-compute indicators ───────────────────────────────────────────────────
static void compute_indicators(std::vector<Bar>& bars, int window = 20) {
    int n = static_cast<int>(bars.size());
    if (n < 2) return;

    // ATR (simplified Wilder)
    std::vector<double> tr(n);
    tr[0] = bars[0].high - bars[0].low;
    for (int i = 1; i < n; ++i) {
        double hl  = bars[i].high - bars[i].low;
        double hpc = std::abs(bars[i].high - bars[i-1].close);
        double lpc = std::abs(bars[i].low  - bars[i-1].close);
        tr[i] = std::max({hl, hpc, lpc});
    }
    double atr_sum = 0.0;
    for (int i = 0; i < std::min(window, n); ++i) atr_sum += tr[i];
    bars[window-1].atr = atr_sum / window;
    for (int i = window; i < n; ++i)
        bars[i].atr = (bars[i-1].atr * (window - 1) + tr[i]) / window;

    // Rolling return z-score
    for (int i = window; i < n; ++i) {
        double mean = 0.0, var = 0.0;
        for (int j = i - window; j < i; ++j) {
            double r = (bars[j].close - bars[j-1].close) / bars[j-1].close;
            mean += r;
        }
        mean /= window;
        for (int j = i - window; j < i; ++j) {
            double r = (bars[j].close - bars[j-1].close) / bars[j-1].close;
            var  += (r - mean) * (r - mean);
        }
        double std_ = std::sqrt(var / window);
        double ret  = (bars[i].close - bars[i-1].close) / bars[i-1].close;
        bars[i].z_ret = (std_ > 1e-9) ? (ret - mean) / std_ : 0.0;
    }

    // VWAP (approximate)
    double cum_pv = 0.0, cum_v = 0.0;
    for (auto& b : bars) {
        double typ = (b.high + b.low + b.close) / 3.0;
        cum_pv += typ * b.volume;
        cum_v  += (b.volume > 0) ? b.volume : 1;
        b.vwap  = cum_pv / cum_v;
    }

    // Composite momentum
    for (auto& b : bars)
        b.z_mom = b.z_ret * 0.6 + (b.vwap > 0 ? (b.close / b.vwap - 1.0) * 100.0 : 0.0) * 0.4;
}

// ─── Signal generation ────────────────────────────────────────────────────────
/**
 * STRATEGY SIGNAL LOGIC
 * ──────────────────────────────────────────────────────────────────────────────
 * Replace the body of this function with YOUR strategy's entry conditions.
 * The Python parser will scan this function for:
 *   - Signal names (MOM_LONG, MOM_SHORT, ABSORPTION_SHORT, etc.)
 *   - Threshold constants (Z_RET_THRESHOLD, Z_VOL_THRESHOLD, etc.)
 *   - Regime labels (TRENDING, ROTATIONAL, MEAN_REVERT, etc.)
 *
 * Return: signal name string, or "" for no signal.
 */

// ── Configurable thresholds (Python parser will extract these) ────────────────
constexpr double Z_RET_THRESHOLD   = 0.45;   // minimum z-return for MOM signal
constexpr double Z_MOM_THRESHOLD   = 0.30;   // minimum z-momentum
constexpr double ATR_STOP_MULT     = 1.50;   // stop = ATR × 1.50
constexpr double TP_R_MULTIPLE     = 2.00;   // target = 2.0R
constexpr double DAILY_LOSS_LIMIT  = 3000.0; // daily loss circuit breaker ($)

static std::string generate_signal(
    const Bar& bar,
    const Bar& prev,
    const std::string& regime
) {
    if (bar.atr == 0.0) return "";

    bool trend_up   = bar.close > bar.vwap;
    bool trend_down = bar.close < bar.vwap;

    // ── MOM_LONG: momentum long entry ─────────────────────────────────────────
    if (bar.z_ret > Z_RET_THRESHOLD &&
        bar.z_mom > Z_MOM_THRESHOLD &&
        trend_up &&
        regime == "TRENDING")
    {
        return "MOM_LONG";
    }

    // ── MOM_SHORT: momentum short entry ──────────────────────────────────────
    if (bar.z_ret < -Z_RET_THRESHOLD &&
        bar.z_mom < -Z_MOM_THRESHOLD &&
        trend_down &&
        regime == "TRENDING")
    {
        return "MOM_SHORT";
    }

    // ── ABSORPTION_SHORT: supply absorption short (add your own logic) ────────
    // Example: strong up move on below-average volume = absorption
    // if (bar.z_ret > 0.6 && bar.volume < avg_volume * 0.8 && trend_down)
    //     return "ABSORPTION_SHORT";

    return "";
}

// ─── Regime detector ─────────────────────────────────────────────────────────
static std::string detect_regime(const Bar& bar) {
    // Simple regime: trending if price well away from VWAP
    double dist = std::abs(bar.close - bar.vwap) / bar.vwap;
    if (dist > 0.003) return "TRENDING";
    if (dist > 0.001) return "ROTATIONAL";
    return "MEAN_REVERT";
}

// ─── Write outputs ────────────────────────────────────────────────────────────
static void write_trades(const std::string& path, const std::vector<Trade>& trades) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write trades file: " + path);

    f << "trade_id,signal,direction,entry_time,exit_time,entry_price,exit_price,"
      << "stop_price,target_price,contracts,outcome,gross_pnl,commission,"
      << "slippage,net_pnl,hold_bars,hold_minutes,mae,mfe,is_gap_fill\n";

    f << std::fixed << std::setprecision(4);
    for (const auto& t : trades) {
        f << t.trade_id    << ','
          << t.signal      << ','
          << t.direction   << ','
          << t.entry_time  << ','
          << t.exit_time   << ','
          << t.entry_price << ','
          << t.exit_price  << ','
          << t.stop_price  << ','
          << t.target_price<< ','
          << t.contracts   << ','
          << t.outcome     << ','
          << t.gross_pnl   << ','
          << t.commission  << ','
          << t.slippage    << ','
          << t.net_pnl     << ','
          << t.hold_bars   << ','
          << t.hold_minutes<< ','
          << t.mae         << ','
          << t.mfe         << ','
          << (t.is_gap_fill ? "true" : "false") << '\n';
    }
}

static void write_equity(
    const std::string& path,
    const std::vector<std::string>& ts,
    const std::vector<double>& equity
) {
    std::ofstream f(path);
    if (!f) throw std::runtime_error("Cannot write equity file: " + path);

    f << "timestamp,equity,drawdown,drawdown_pct\n";
    double peak = equity.empty() ? 0.0 : equity[0];
    f << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < ts.size() && i < equity.size(); ++i) {
        if (equity[i] > peak) peak = equity[i];
        double dd     = peak - equity[i];
        double dd_pct = (peak > 0) ? dd / peak * 100.0 : 0.0;
        f << ts[i] << ',' << equity[i] << ',' << dd << ',' << dd_pct << '\n';
    }
}

// ─── Core backtest loop ───────────────────────────────────────────────────────
static void run_backtest(const Config& cfg) {
    auto bars = load_data(cfg.data_file);
    if (bars.empty()) throw std::runtime_error("No bars loaded from data file.");

    compute_indicators(bars);

    const double pv    = POINT_VALUE.count(cfg.instrument)
                         ? POINT_VALUE.at(cfg.instrument) : 20.0;
    const double tv    = TICK_VALUE.count(cfg.instrument)
                         ? TICK_VALUE.at(cfg.instrument)  : 5.0;
    const double slip  = cfg.slippage_ticks * tv * 2.0;
    const double comm  = cfg.commission * cfg.num_contracts;
    const double cost  = comm + slip;

    double balance    = cfg.starting_balance;
    double daily_pnl  = 0.0;
    std::string last_date;

    std::vector<Trade>       completed;
    std::vector<double>      equity_curve;
    std::vector<std::string> equity_ts;
    int trade_id = 0;

    bool       in_trade   = false;
    Trade      open_trade;
    int        cooldown   = 0;
    const int  COOLDOWN_BARS = 5;

    for (int i = 1; i < static_cast<int>(bars.size()); ++i) {
        const Bar& bar  = bars[i];
        const Bar& prev = bars[i - 1];

        // Date tracker for daily P&L reset
        std::string bar_date = bar.timestamp.substr(0, 10);
        if (bar_date != last_date) { daily_pnl = 0.0; last_date = bar_date; }

        // EOD exit
        if (cfg.eod_exit && in_trade) {
            std::string bar_time = bar.timestamp.length() > 10
                                   ? bar.timestamp.substr(11, 5) : "00:00";
            if (bar_time >= cfg.eod_exit_time) {
                double sign = (open_trade.direction == "LONG") ? 1.0 : -1.0;
                double gross = sign * (bar.open - open_trade.entry_price) * pv * open_trade.contracts;
                open_trade.gross_pnl  = gross;
                open_trade.net_pnl    = gross - cost;
                open_trade.commission = comm;
                open_trade.slippage   = slip;
                open_trade.exit_price = bar.open;
                open_trade.exit_time  = bar.timestamp;
                open_trade.outcome    = "EOD";
                completed.push_back(open_trade);
                balance    += open_trade.net_pnl;
                daily_pnl  += open_trade.net_pnl;
                in_trade    = false;
                cooldown    = COOLDOWN_BARS;
            }
        }

        // Manage open trade
        if (in_trade) {
            open_trade.hold_bars++;
            double sign = (open_trade.direction == "LONG") ? 1.0 : -1.0;
            bool closed = false;

            if (open_trade.direction == "LONG") {
                // Gap fill
                if (bar.open <= open_trade.stop_price) {
                    open_trade.exit_price = bar.open;
                    open_trade.outcome    = (bar.open < open_trade.stop_price) ? "GAP" : "LOSS";
                    open_trade.is_gap_fill = bar.open < open_trade.stop_price;
                    closed = true;
                } else if (bar.open >= open_trade.target_price) {
                    open_trade.exit_price = bar.open;
                    open_trade.outcome    = "GAP";
                    open_trade.is_gap_fill = true;
                    closed = true;
                } else if (bar.low <= open_trade.stop_price) {
                    open_trade.exit_price = open_trade.stop_price;
                    open_trade.outcome    = "LOSS";
                    closed = true;
                } else if (bar.high >= open_trade.target_price) {
                    open_trade.exit_price = open_trade.target_price;
                    open_trade.outcome    = "WIN";
                    closed = true;
                } else {
                    open_trade.mae = std::min(open_trade.mae, bar.low  - open_trade.entry_price);
                    open_trade.mfe = std::max(open_trade.mfe, bar.high - open_trade.entry_price);
                }
            } else { // SHORT
                if (bar.open >= open_trade.stop_price) {
                    open_trade.exit_price = bar.open;
                    open_trade.outcome    = (bar.open > open_trade.stop_price) ? "GAP" : "LOSS";
                    open_trade.is_gap_fill = bar.open > open_trade.stop_price;
                    closed = true;
                } else if (bar.open <= open_trade.target_price) {
                    open_trade.exit_price = bar.open;
                    open_trade.outcome    = "GAP";
                    open_trade.is_gap_fill = true;
                    closed = true;
                } else if (bar.high >= open_trade.stop_price) {
                    open_trade.exit_price = open_trade.stop_price;
                    open_trade.outcome    = "LOSS";
                    closed = true;
                } else if (bar.low <= open_trade.target_price) {
                    open_trade.exit_price = open_trade.target_price;
                    open_trade.outcome    = "WIN";
                    closed = true;
                } else {
                    open_trade.mae = std::min(open_trade.mae, open_trade.entry_price - bar.high);
                    open_trade.mfe = std::max(open_trade.mfe, open_trade.entry_price - bar.low);
                }
            }

            if (closed) {
                double gross = sign * (open_trade.exit_price - open_trade.entry_price)
                               * pv * open_trade.contracts;
                open_trade.gross_pnl  = gross;
                open_trade.net_pnl    = gross - cost;
                open_trade.commission = comm;
                open_trade.slippage   = slip;
                open_trade.exit_time  = bar.timestamp;
                completed.push_back(open_trade);
                balance   += open_trade.net_pnl;
                daily_pnl += open_trade.net_pnl;
                in_trade   = false;
                cooldown   = COOLDOWN_BARS;
            }
        }

        // Daily loss circuit breaker
        if (cfg.daily_loss_limit > 0.0 && daily_pnl < -cfg.daily_loss_limit)
            goto record_equity;

        // Generate new signal
        if (!in_trade && cooldown <= 0) {
            std::string regime = detect_regime(bar);
            std::string sig    = generate_signal(bar, prev, regime);

            if (!sig.empty()) {
                double atr        = (bar.atr > 0.0) ? bar.atr : 20.0;
                double stop_dist  = atr * ATR_STOP_MULT;
                double tgt_dist   = stop_dist * TP_R_MULTIPLE;
                bool   is_long    = (sig.find("LONG") != std::string::npos ||
                                     sig.find("BUY")  != std::string::npos);

                Trade t;
                t.trade_id    = ++trade_id;
                t.signal      = sig;
                t.direction   = is_long ? "LONG" : "SHORT";
                t.entry_time  = bar.timestamp;
                t.entry_price = bar.close;
                t.contracts   = cfg.num_contracts;
                t.stop_price  = is_long
                                ? bar.close - stop_dist
                                : bar.close + stop_dist;
                t.target_price = is_long
                                 ? bar.close + tgt_dist
                                 : bar.close - tgt_dist;
                t.mae         = 0.0;
                t.mfe         = 0.0;
                open_trade    = t;
                in_trade      = true;
            }
        } else if (cooldown > 0) {
            --cooldown;
        }

        record_equity:
        equity_curve.push_back(balance);
        equity_ts.push_back(bar.timestamp);
    }

    // Force-close any remaining trade at last bar
    if (in_trade && !bars.empty()) {
        const Bar& last = bars.back();
        double sign  = (open_trade.direction == "LONG") ? 1.0 : -1.0;
        double gross = sign * (last.close - open_trade.entry_price)
                       * pv * open_trade.contracts;
        open_trade.exit_price = last.close;
        open_trade.exit_time  = last.timestamp;
        open_trade.outcome    = "EOD";
        open_trade.gross_pnl  = gross;
        open_trade.net_pnl    = gross - cost;
        open_trade.commission = comm;
        open_trade.slippage   = slip;
        completed.push_back(open_trade);
    }

    write_trades(cfg.trades_output, completed);
    write_equity(cfg.equity_output, equity_ts, equity_curve);

    // Summary to stdout (terminal can capture this)
    std::cout << "[COMPLETE] trades=" << completed.size()
              << " | final_equity=" << std::fixed << std::setprecision(2) << balance
              << " | data_bars=" << bars.size()
              << '\n';
}

// ─── Entry point ─────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <path/to/run_config.json>\n";
        return 1;
    }

    try {
        Config cfg = load_config(argv[1]);
        run_backtest(cfg);
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << '\n';
        return 3;
    }
}
