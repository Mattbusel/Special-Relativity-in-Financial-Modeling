/// @file src/main.cpp
/// @brief SRFM CLI entry point — AGT-06.
///
/// Usage:
///   srfm --backtest <csv_file>   Run a relativistic backtest on OHLCV data
///   srfm --stream                Read OHLCV bars from stdin (one CSV row each)
///   srfm --help                  Print usage

#include "srfm/engine.hpp"
#include "srfm/data_loader.hpp"

#include <fmt/core.h>

#include <cstring>
#include <iostream>
#include <string>

namespace {

void print_usage() {
    fmt::print(
        "Usage:\n"
        "  srfm --backtest <csv_file>   Backtest on OHLCV CSV data\n"
        "  srfm --stream                Stream OHLCV bars from stdin\n"
        "  srfm --help                  Show this help\n"
        "\n"
        "CSV format (header required):\n"
        "  timestamp,open,high,low,close,volume\n"
    );
}

/// Run a full batch backtest on the given CSV file.
/// Returns 0 on success, 1 on error.
int run_backtest(const std::string& filepath) {
    auto bars = srfm::core::DataLoader::load_csv(filepath);
    if (!bars) {
        fmt::print(stderr, "Error: cannot open file '{}'\n", filepath);
        return 1;
    }

    if (bars->empty()) {
        fmt::print(stderr, "Error: no valid bars loaded from '{}'\n", filepath);
        return 1;
    }

    fmt::print("Loaded {} bars from '{}'\n", bars->size(), filepath);

    srfm::core::Engine engine;
    auto result = engine.run_backtest(*bars);

    if (!result) {
        fmt::print(stderr,
            "Error: backtest failed — too few bars ({} loaded, {} required)\n",
            bars->size(), srfm::constants::MIN_RETURN_SERIES_LENGTH);
        return 1;
    }

    fmt::print("{}\n", result->to_string());
    return 0;
}

/// Read OHLCV CSV rows from stdin and process each bar through the engine.
/// The header line is the first line of stdin.
/// Returns 0 on success, 1 on error.
int run_stream() {
    srfm::core::Engine engine;

    std::string line;
    bool header_skipped = false;
    std::size_t bar_count = 0;

    fmt::print("SRFM streaming mode. Enter OHLCV rows (timestamp,open,high,low,close,volume).\n");
    fmt::print("First line: header. Ctrl-D to finish.\n");

    while (std::getline(std::cin, line)) {
        // Trim carriage return for Windows-style line endings.
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!header_skipped) {
            if (!line.empty() && line[0] != '#') {
                header_skipped = true;
            }
            continue;
        }

        // Parse this row as a standalone CSV string with a dummy header.
        const std::string wrapped = "timestamp,open,high,low,close,volume\n" + line;
        auto batch = srfm::core::DataLoader::parse_csv_string(wrapped);
        if (batch.empty()) {
            fmt::print(stderr, "Skipping malformed row: {}\n", line);
            continue;
        }

        ++bar_count;
        auto result = engine.process_stream_bar(batch[0]);
        if (result) {
            fmt::print(
                "Bar {:4d}: close={:.4f}  β={:.4f}  γ={:.4f}  interval={}\n",
                bar_count,
                result->raw.close,
                result->beta.value,
                result->gamma.value,
                srfm::manifold::to_string(result->interval_type)
            );
        }
    }

    fmt::print("Processed {} bars.\n", bar_count);
    return 0;
}

}  // anonymous namespace

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage();
        return 1;
    }

    const std::string mode(argv[1]);

    if (mode == "--help" || mode == "-h") {
        print_usage();
        return 0;
    }

    if (mode == "--backtest") {
        if (argc < 3) {
            fmt::print(stderr, "Error: --backtest requires a CSV file path\n");
            print_usage();
            return 1;
        }
        return run_backtest(std::string(argv[2]));
    }

    if (mode == "--stream") {
        return run_stream();
    }

    fmt::print(stderr, "Unknown option: {}\n", mode);
    print_usage();
    return 1;
}
