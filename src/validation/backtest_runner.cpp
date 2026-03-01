/// @file src/validation/backtest_runner.cpp
/// @brief Empirical Validation Binary — Three-Strategy Backtest Runner
///
/// # Binary: backtest_runner
///
/// ## Purpose
/// Reads a regime CSV (output from regime_validator) and runs all three
/// SRFM strategies (RAW, RELATIVISTIC, GEODESIC_DEVIATION) via
/// ExtendedBacktester.  Writes one output row per strategy per ticker:
///
///   ticker, strategy, sharpe, sortino, max_drawdown
///
/// Also prints a formatted comparison table to stdout.
///
/// ## Usage
///   backtest_runner --input <regime_csv> --output-dir <dir> [--ticker <name>]
///
/// ## Input Format (regime CSV)
///   ticker, bar_index, interval_type, next_bar_abs_return, beta, geodesic_deviation
///
/// ## Exit Codes
///   0 — success
///   1 — fatal error

#include "srfm/geodesic_strategy.hpp"
#include "srfm/backtest.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

// ─── CSV Parsing ─────────────────────────────────────────────────────────────

static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) {
        auto start = field.find_first_not_of(" \t\r\n");
        auto end   = field.find_last_not_of(" \t\r\n");
        if (start != std::string::npos) {
            fields.push_back(field.substr(start, end - start + 1));
        } else {
            fields.push_back("");
        }
    }
    return fields;
}

static std::optional<double> safe_double(const std::string& s) {
    if (s.empty()) return std::nullopt;
    try {
        std::size_t pos = 0;
        double v = std::stod(s, &pos);
        if (pos != s.size()) return std::nullopt;
        if (!std::isfinite(v)) return std::nullopt;
        return v;
    } catch (...) {
        return std::nullopt;
    }
}

static int find_col(const std::vector<std::string>& hdr, const std::string& name) {
    for (int i = 0; i < static_cast<int>(hdr.size()); ++i) {
        std::string h = hdr[static_cast<std::size_t>(i)];
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        if (h == name) return i;
    }
    return -1;
}

// ─── Regime Row ──────────────────────────────────────────────────────────────

struct RegimeRow {
    std::size_t bar_index          = 0;
    std::string interval_type;
    double      next_bar_abs_return = 0.0;
    double      beta               = 0.0;
    double      geodesic_deviation = 0.0;
};

static std::vector<RegimeRow> load_regime_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open: " + path);
    }

    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty file: " + path);
    }
    auto hdr = split_csv(line);

    int col_idx  = find_col(hdr, "bar_index");
    int col_type = find_col(hdr, "interval_type");
    int col_ret  = find_col(hdr, "next_bar_abs_return");
    int col_beta = find_col(hdr, "beta");
    int col_geo  = find_col(hdr, "geodesic_deviation");

    if (col_type < 0 || col_ret < 0) {
        throw std::runtime_error("CSV missing required columns interval_type / next_bar_abs_return");
    }

    std::vector<RegimeRow> rows;
    std::size_t auto_idx = 0;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        auto fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= col_type) continue;

        auto maybe_ret = safe_double(fields[static_cast<std::size_t>(col_ret)]);
        if (!maybe_ret) continue;

        RegimeRow row;
        row.bar_index          = (col_idx >= 0)
            ? static_cast<std::size_t>(
                  safe_double(fields[static_cast<std::size_t>(col_idx)]).value_or(
                      static_cast<double>(auto_idx)))
            : auto_idx;
        row.interval_type      = fields[static_cast<std::size_t>(col_type)];
        row.next_bar_abs_return = *maybe_ret;
        row.beta               = (col_beta >= 0)
            ? safe_double(fields[static_cast<std::size_t>(col_beta)]).value_or(0.0)
            : 0.0;
        row.geodesic_deviation = (col_geo >= 0)
            ? safe_double(fields[static_cast<std::size_t>(col_geo)]).value_or(0.0)
            : 0.0;

        rows.push_back(row);
        ++auto_idx;
    }
    return rows;
}

// ─── Build GeodesicBarData ────────────────────────────────────────────────────

/// Convert regime rows to GeodesicBarData.
///
/// Signal convention: +1 for TIMELIKE (trust causal momentum), −1 for SPACELIKE.
/// Asset return = next_bar_abs_return with sign derived from interval type.
static std::pair<std::vector<srfm::backtest::GeodesicBarData>, std::vector<double>>
build_bars(const std::vector<RegimeRow>& rows) {
    std::vector<srfm::backtest::GeodesicBarData> bars;
    std::vector<double> asset_returns;
    bars.reserve(rows.size());
    asset_returns.reserve(rows.size());

    for (const auto& row : rows) {
        // Raw signal: causal bars get +1, stochastic get -1
        double raw_signal = (row.interval_type == "Timelike") ? 1.0 : -1.0;
        double beta_val   = std::clamp(row.beta, 0.0, srfm::constants::BETA_MAX_SAFE - 0.001);

        srfm::backtest::BarData base{
            .raw_signal = raw_signal,
            .beta       = srfm::BetaVelocity{beta_val},
            .benchmark  = 0.0,  // no benchmark return available
        };

        bars.push_back(srfm::backtest::GeodesicBarData{
            .base               = base,
            .geodesic_deviation = row.geodesic_deviation,
        });

        // Asset return = unsigned abs return; Backtester applies direction via raw_signal.
        // Passing raw_signal * abs_return would double-sign: sign(raw_signal) * raw_signal * abs_return
        // = abs_return (always positive), making Sortino nullopt (no returns below zero).
        asset_returns.push_back(row.next_bar_abs_return);
    }

    return {bars, asset_returns};
}

// ─── Output ───────────────────────────────────────────────────────────────────

static void write_backtest_csv(
    const std::string& ticker,
    const srfm::backtest::ExtendedBacktester::TripleComparison& result,
    const std::string& output_dir)
{
    fs::create_directories(output_dir);
    std::string safe_ticker = ticker;
    std::replace(safe_ticker.begin(), safe_ticker.end(), '-', '_');

    std::string out_path = output_dir + "/" + safe_ticker + "_backtest.csv";
    std::ofstream out(out_path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output: " + out_path);
    }

    out << "ticker,strategy,sharpe,sortino,max_drawdown\n";
    auto write_row = [&](const std::string& strategy,
                          const srfm::backtest::PerformanceMetrics& m) {
        out << ticker << "," << strategy << ",";
        out << std::fixed;
        out.precision(6);
        out << m.sharpe_ratio << ","
            << m.sortino_ratio << ","
            << m.max_drawdown << "\n";
    };

    write_row("RAW",                result.raw);
    write_row("RELATIVISTIC",       result.relativistic);
    write_row("GEODESIC_DEVIATION", result.geodesic);

    std::cout << "[" << ticker << "] Backtest CSV written to " << out_path << "\n";
}

// ─── CLI Args ─────────────────────────────────────────────────────────────────

struct Args {
    std::string input_path;
    std::string output_dir;
    std::string ticker;
};

static std::optional<Args> parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc - 1; ++i) {
        std::string key(argv[i]);
        std::string val(argv[i + 1]);
        if (key == "--input") {
            args.input_path = val; ++i;
        } else if (key == "--output-dir") {
            args.output_dir = val; ++i;
        } else if (key == "--ticker") {
            args.ticker = val; ++i;
        }
    }
    if (args.input_path.empty() || args.output_dir.empty()) return std::nullopt;
    if (args.ticker.empty()) args.ticker = "UNKNOWN";
    return args;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto maybe_args = parse_args(argc, argv);
    if (!maybe_args.has_value()) {
        std::cerr << "Usage: backtest_runner --input <regime_csv> "
                     "--output-dir <dir> [--ticker <name>]\n";
        return 1;
    }
    const Args& args = *maybe_args;

    // ── Load regime data ───────────────────────────────────────────────────────
    std::vector<RegimeRow> rows;
    try {
        rows = load_regime_csv(args.input_path);
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "[" << args.ticker << "] Loaded " << rows.size() << " regime rows\n";

    if (rows.size() < static_cast<std::size_t>(srfm::constants::MIN_RETURN_SERIES_LENGTH)) {
        std::cerr << "[FATAL] Too few rows (" << rows.size()
                  << ") — need at least " << srfm::constants::MIN_RETURN_SERIES_LENGTH << "\n";
        return 1;
    }

    // ── Build bars ─────────────────────────────────────────────────────────────
    auto [bars, asset_returns] = build_bars(rows);

    // ── Run backtest ───────────────────────────────────────────────────────────
    srfm::backtest::ExtendedBacktester backtester(
        srfm::backtest::BacktestConfig{},
        100   // rolling window for p75
    );

    auto maybe_result = backtester.run_triple(bars, asset_returns, args.ticker);
    if (!maybe_result.has_value()) {
        std::cerr << "[FATAL] Backtest failed for " << args.ticker
                  << " — check data quality and minimum series length\n";
        return 1;
    }

    // ── Print summary ──────────────────────────────────────────────────────────
    std::cout << maybe_result->to_string();

    // ── Write CSV ──────────────────────────────────────────────────────────────
    try {
        write_backtest_csv(args.ticker, *maybe_result, args.output_dir);
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
