/// @file src/core/data_loader.cpp
/// @brief CSV DataLoader for OHLCV market data — AGT-06.

#include "srfm/data_loader.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace srfm::core {

// ─── DataLoader::validate_bar ─────────────────────────────────────────────────

bool DataLoader::validate_bar(const OHLCV& bar) noexcept {
    // All fields must be finite.
    if (!std::isfinite(bar.timestamp) ||
        !std::isfinite(bar.open)      ||
        !std::isfinite(bar.high)      ||
        !std::isfinite(bar.low)       ||
        !std::isfinite(bar.close)     ||
        !std::isfinite(bar.volume)) {
        return false;
    }

    // OHLC consistency.
    if (bar.high < bar.low)   return false;
    if (bar.open  > bar.high) return false;
    if (bar.open  < bar.low)  return false;
    if (bar.close > bar.high) return false;
    if (bar.close < bar.low)  return false;

    // Volume is non-negative.
    if (bar.volume < 0.0) return false;

    return true;
}

// ─── DataLoader::parse_row ────────────────────────────────────────────────────

std::optional<OHLCV>
DataLoader::parse_row(const std::string& line) noexcept {
    // Skip blank lines and comment lines.
    if (line.empty() || line[0] == '#') {
        return std::nullopt;
    }

    std::istringstream ss(line);
    std::string token;
    std::vector<double> fields;
    fields.reserve(6);

    while (std::getline(ss, token, ',')) {
        // Trim leading/trailing whitespace.
        const auto first = token.find_first_not_of(" \t\r\n");
        const auto last  = token.find_last_not_of(" \t\r\n");
        if (first == std::string::npos) {
            return std::nullopt;  // empty token
        }
        token = token.substr(first, last - first + 1);

        double val = 0.0;
        try {
            std::size_t pos = 0;
            val = std::stod(token, &pos);
            if (pos != token.size()) {
                return std::nullopt;  // trailing garbage
            }
        } catch (...) {
            return std::nullopt;
        }

        if (!std::isfinite(val)) {
            return std::nullopt;
        }

        fields.push_back(val);
    }

    if (fields.size() != 6) {
        return std::nullopt;
    }

    OHLCV bar{
        .timestamp = fields[0],
        .open      = fields[1],
        .high      = fields[2],
        .low       = fields[3],
        .close     = fields[4],
        .volume    = fields[5],
    };

    if (!validate_bar(bar)) {
        return std::nullopt;
    }

    return bar;
}

// ─── DataLoader::parse_csv_string ────────────────────────────────────────────

std::vector<OHLCV>
DataLoader::parse_csv_string(const std::string& csv_content) noexcept {
    std::vector<OHLCV> bars;
    std::istringstream stream(csv_content);
    std::string line;
    bool header_skipped = false;

    while (std::getline(stream, line)) {
        // Trim carriage return.
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }

        if (!header_skipped) {
            // First non-empty, non-comment line is the header.
            if (!line.empty() && line[0] != '#') {
                header_skipped = true;
            }
            continue;
        }

        auto bar = parse_row(line);
        if (bar) {
            bars.push_back(*bar);
        }
    }

    return bars;
}

// ─── DataLoader::load_csv ────────────────────────────────────────────────────

std::optional<std::vector<OHLCV>>
DataLoader::load_csv(const std::string& filepath) noexcept {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        return std::nullopt;
    }

    std::string contents;
    std::string line;
    while (std::getline(file, line)) {
        contents += line;
        contents += '\n';
    }

    return parse_csv_string(contents);
}

}  // namespace srfm::core
