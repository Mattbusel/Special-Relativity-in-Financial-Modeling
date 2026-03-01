/// @file src/validation/regime_validator.cpp
/// @brief Empirical Validation Binary — Regime Classification + Return Recording
///
/// # Binary: regime_validator
///
/// ## Purpose
/// Reads a 1-minute OHLCV CSV file (produced by validation/fetch_data.py),
/// runs the full SRFM pipeline on each bar, and writes one output row per bar:
///
///   ticker, bar_index, interval_type, next_bar_abs_return, beta, geodesic_deviation
///
/// This output feeds the Q1 and Q2 Python analysis scripts.
///
/// ## Pipeline
///   1. Parse OHLCV CSV (no external library — std::stringstream)
///   2. Compute SpacetimeEvent for each bar (time, close, volume, momentum)
///   3. Normalize events via CoordinateNormalizer (rolling z-score, window=20)
///   4. Classify each bar via MarketManifold::process (TIMELIKE/SPACELIKE/LIGHTLIKE)
///   5. Compute β for each bar via MarketManifold::beta
///   6. Compute geodesic deviation via GeodesicDeviationCalculator::compute
///   7. Write output CSV
///
/// ## Usage
///   regime_validator --input <csv> --output <csv> --ticker <name>
///
/// ## Exit Codes
///   0 — success
///   1 — fatal error (missing file, bad args, etc.)

#include "srfm/manifold.hpp"
#include "srfm/normalizer.hpp"
#include "srfm/tensor.hpp"
#include "srfm/geodesic_signal.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// ─── Raw OHLCV Bar ────────────────────────────────────────────────────────────

struct OhlcvBar {
    std::string timestamp;
    double open   = 0.0;
    double high   = 0.0;
    double low    = 0.0;
    double close  = 0.0;
    double volume = 0.0;
};

// ─── CSV Helpers ──────────────────────────────────────────────────────────────

/// Split a CSV line on commas (no quote handling needed for numeric data).
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    std::istringstream ss(line);
    while (std::getline(ss, field, ',')) {
        // Trim whitespace
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

/// Safely parse a double from a string.  Returns nullopt on failure.
static std::optional<double> safe_parse_double(const std::string& s) {
    if (s.empty()) return std::nullopt;
    try {
        std::size_t pos = 0;
        double val = std::stod(s, &pos);
        if (pos != s.size()) return std::nullopt;
        if (!std::isfinite(val)) return std::nullopt;
        return val;
    } catch (...) {
        return std::nullopt;
    }
}

/// Find the column index of a header name (case-insensitive).
static int find_col(const std::vector<std::string>& header,
                    const std::string& name) {
    for (int i = 0; i < static_cast<int>(header.size()); ++i) {
        std::string h = header[static_cast<std::size_t>(i)];
        std::transform(h.begin(), h.end(), h.begin(), ::tolower);
        if (h == name) return i;
    }
    return -1;
}

// ─── Data Loading ─────────────────────────────────────────────────────────────

static std::vector<OhlcvBar> load_csv(const std::string& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open input file: " + path);
    }

    std::vector<OhlcvBar> bars;
    std::string line;

    // Read header
    if (!std::getline(file, line)) {
        throw std::runtime_error("Empty CSV file: " + path);
    }
    auto header = split_csv(line);

    int col_ts     = find_col(header, "timestamp");
    int col_open   = find_col(header, "open");
    int col_high   = find_col(header, "high");
    int col_low    = find_col(header, "low");
    int col_close  = find_col(header, "close");
    int col_volume = find_col(header, "volume");

    if (col_close < 0 || col_volume < 0) {
        throw std::runtime_error("CSV missing required columns 'close' or 'volume'");
    }

    std::size_t row_idx = 0;
    std::size_t skipped = 0;
    while (std::getline(file, line)) {
        ++row_idx;
        if (line.empty()) continue;

        auto fields = split_csv(line);
        if (static_cast<int>(fields.size()) <= std::max(col_close, col_volume)) {
            ++skipped;
            continue;
        }

        auto maybe_close  = safe_parse_double(fields[static_cast<std::size_t>(col_close)]);
        auto maybe_volume = safe_parse_double(fields[static_cast<std::size_t>(col_volume)]);

        if (!maybe_close || !maybe_volume) {
            ++skipped;
            continue;
        }
        if (*maybe_close <= 0.0 || *maybe_volume < 0.0) {
            ++skipped;
            continue;
        }

        OhlcvBar bar;
        bar.timestamp = (col_ts >= 0) ? fields[static_cast<std::size_t>(col_ts)] : std::to_string(row_idx);
        bar.close     = *maybe_close;
        bar.volume    = *maybe_volume;

        if (col_open >= 0) {
            auto v = safe_parse_double(fields[static_cast<std::size_t>(col_open)]);
            bar.open = v.value_or(bar.close);
        }
        if (col_high >= 0) {
            auto v = safe_parse_double(fields[static_cast<std::size_t>(col_high)]);
            bar.high = v.value_or(bar.close);
        }
        if (col_low >= 0) {
            auto v = safe_parse_double(fields[static_cast<std::size_t>(col_low)]);
            bar.low = v.value_or(bar.close);
        }

        bars.push_back(bar);
    }

    if (skipped > 0) {
        std::cerr << "[regime_validator] Skipped " << skipped << " non-finite rows\n";
    }
    return bars;
}

// ─── SRFM Pipeline ────────────────────────────────────────────────────────────

struct ClassifiedBar {
    std::size_t bar_index          = 0;
    std::string interval_type;        // "Timelike", "Spacelike", "Lightlike"
    double      next_bar_abs_return = 0.0;
    double      beta               = 0.0;
    double      geodesic_deviation = 0.0;
};

static std::vector<ClassifiedBar> classify_bars(
    const std::vector<OhlcvBar>& bars,
    const std::string& ticker)
{
    const std::size_t n = bars.size();
    if (n < 3) {
        std::cerr << "[" << ticker << "] Too few bars (" << n << ") — need at least 3\n";
        return {};
    }

    // ── Build SpacetimeEvents (raw) ────────────────────────────────────────────
    // momentum = (close - prev_close) / prev_close  (simple return as momentum proxy)
    std::vector<srfm::manifold::SpacetimeEvent> raw_events;
    raw_events.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        double momentum = 0.0;
        if (i > 0 && bars[i - 1].close > 0.0) {
            momentum = (bars[i].close - bars[i - 1].close) / bars[i - 1].close;
        }
        raw_events.push_back({
            .time     = static_cast<double>(i),
            .price    = bars[i].close,
            .volume   = bars[i].volume,
            .momentum = momentum,
        });
    }

    // ── Normalizer ────────────────────────────────────────────────────────────
    srfm::CoordinateNormalizer normalizer(20);

    // Warm up normalizer on first 2 bars (these become the "prev" for pipeline)
    srfm::manifold::SpacetimeEvent prev_normalized = normalizer.normalize(raw_events[0]);
    (void)normalizer.normalize(raw_events[1]);  // warm-up second bar

    // Pre-normalize all events for geodesic computation
    srfm::CoordinateNormalizer norm2(20);
    std::vector<srfm::manifold::SpacetimeEvent> all_normalized;
    all_normalized.reserve(n);
    for (const auto& ev : raw_events) {
        all_normalized.push_back(norm2.normalize(ev));
    }

    // ── Geodesic deviation ────────────────────────────────────────────────────
    auto metric = srfm::tensor::MetricTensor::make_minkowski(1.0, 1.0);
    srfm::tensor::GeodesicDeviationCalculator geo_calc(metric);
    auto geo_signals = geo_calc.compute(all_normalized);

    // ── Main classification loop ───────────────────────────────────────────────
    // Start from bar 2 (warmup bars 0,1 are skipped)
    std::vector<ClassifiedBar> results;
    results.reserve(n - 2);

    // Reset normalizer — re-normalize in order for classify loop
    srfm::CoordinateNormalizer norm3(20);
    prev_normalized = norm3.normalize(raw_events[0]);

    for (std::size_t i = 1; i + 1 < n; ++i) {
        // Classify interval i-1 → i
        auto maybe_type = srfm::manifold::MarketManifold::process(
            norm3, prev_normalized, raw_events[i]);
        prev_normalized = norm3.normalize(raw_events[i]);  // update for next step

        if (!maybe_type.has_value()) {
            continue;
        }

        // β = |Δspatial| / (c · |Δt|)
        auto maybe_beta = srfm::manifold::MarketManifold::beta(
            all_normalized[i - 1], all_normalized[i]);
        double beta = maybe_beta.value_or(0.0);

        // next-bar absolute return  = |close[i+1] / close[i] - 1|
        double next_abs_ret = 0.0;
        if (bars[i].close > 0.0) {
            next_abs_ret = std::abs(bars[i + 1].close / bars[i].close - 1.0);
        }

        // Geodesic deviation at bar i
        double geo_dev = 0.0;
        if (i < geo_signals.size() && geo_signals[i].is_valid) {
            geo_dev = geo_signals[i].geodesic_deviation;
        }

        const char* type_str = srfm::manifold::to_string(*maybe_type);

        results.push_back(ClassifiedBar{
            .bar_index          = i,
            .interval_type      = std::string(type_str),
            .next_bar_abs_return = next_abs_ret,
            .beta               = beta,
            .geodesic_deviation = geo_dev,
        });
    }

    return results;
}

// ─── Output ───────────────────────────────────────────────────────────────────

static void write_output(
    const std::string& ticker,
    const std::vector<ClassifiedBar>& bars,
    const std::string& output_path)
{
    std::ofstream out(output_path);
    if (!out.is_open()) {
        throw std::runtime_error("Cannot open output file: " + output_path);
    }

    out << "ticker,bar_index,interval_type,next_bar_abs_return,beta,geodesic_deviation\n";

    for (const auto& b : bars) {
        out << ticker << ","
            << b.bar_index << ","
            << b.interval_type << ","
            << std::fixed;
        out.precision(10);
        out << b.next_bar_abs_return << ","
            << b.beta << ","
            << b.geodesic_deviation << "\n";
    }
}

// ─── CLI Argument Parsing ─────────────────────────────────────────────────────

struct Args {
    std::string input_path;
    std::string output_path;
    std::string ticker;
};

static std::optional<Args> parse_args(int argc, char* argv[]) {
    Args args;
    for (int i = 1; i < argc - 1; ++i) {
        std::string key(argv[i]);
        std::string val(argv[i + 1]);
        if (key == "--input") {
            args.input_path = val;
            ++i;
        } else if (key == "--output") {
            args.output_path = val;
            ++i;
        } else if (key == "--ticker") {
            args.ticker = val;
            ++i;
        }
    }
    if (args.input_path.empty() || args.output_path.empty()) {
        return std::nullopt;
    }
    if (args.ticker.empty()) {
        args.ticker = "UNKNOWN";
    }
    return args;
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char* argv[]) {
    auto maybe_args = parse_args(argc, argv);
    if (!maybe_args.has_value()) {
        std::cerr << "Usage: regime_validator --input <csv> --output <csv> "
                     "[--ticker <name>]\n";
        return 1;
    }

    const Args& args = *maybe_args;

    // ── Load data ─────────────────────────────────────────────────────────────
    std::vector<OhlcvBar> bars;
    try {
        bars = load_csv(args.input_path);
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }

    if (bars.empty()) {
        std::cerr << "[FATAL] No valid bars loaded from " << args.input_path << "\n";
        return 1;
    }

    std::cout << "[" << args.ticker << "] Loaded " << bars.size() << " bars\n";

    // ── Classify ──────────────────────────────────────────────────────────────
    auto classified = classify_bars(bars, args.ticker);

    if (classified.empty()) {
        std::cerr << "[FATAL] No bars classified — check data quality\n";
        return 1;
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::size_t n_tl = 0, n_sl = 0, n_ll = 0;
    for (const auto& b : classified) {
        if (b.interval_type == "Timelike")  ++n_tl;
        else if (b.interval_type == "Spacelike") ++n_sl;
        else                                     ++n_ll;
    }

    std::cout << "[" << args.ticker << "] Classified " << classified.size() << " bars\n";
    std::cout << "  TIMELIKE:  " << n_tl << "  (" << (100.0 * n_tl / classified.size()) << "%)\n";
    std::cout << "  SPACELIKE: " << n_sl << "  (" << (100.0 * n_sl / classified.size()) << "%)\n";
    std::cout << "  LIGHTLIKE: " << n_ll << "  (" << (100.0 * n_ll / classified.size()) << "%)\n";

    // ── Write output ──────────────────────────────────────────────────────────
    try {
        write_output(args.ticker, classified, args.output_path);
    } catch (const std::exception& ex) {
        std::cerr << "[FATAL] " << ex.what() << "\n";
        return 1;
    }

    std::cout << "[" << args.ticker << "] Output written to " << args.output_path << "\n";
    return 0;
}
