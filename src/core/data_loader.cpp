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

namespace {

/// Days since 1970-01-01 for a proleptic Gregorian date (H. Hinnant's
/// days_from_civil).
long long days_from_civil(long long y, unsigned m, unsigned d) noexcept {
    y -= m <= 2 ? 1 : 0;
    const long long era = (y >= 0 ? y : y - 399) / 400;
    const unsigned yoe = static_cast<unsigned>(y - era * 400);
    const unsigned doy = (153 * (m + (m > 2 ? -3 : 9)) + 2) / 5 + d - 1;
    const unsigned doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    return era * 146097 + static_cast<long long>(doe) - 719468;
}

/// Parse an ISO-8601 style timestamp ("2021-03-01", "2021-03-01 14:30:00",
/// "2021-03-01T14:30:00+00:00") into Unix epoch seconds (UTC). Any UTC
/// offset suffix is ignored. Returns nullopt if the text is not a date.
std::optional<double> parse_iso_timestamp(const std::string& s) noexcept {
    auto digits = [&](std::size_t pos, std::size_t n, int& out) {
        if (pos + n > s.size()) return false;
        int v = 0;
        for (std::size_t i = pos; i < pos + n; ++i) {
            if (s[i] < '0' || s[i] > '9') return false;
            v = v * 10 + (s[i] - '0');
        }
        out = v;
        return true;
    };
    int y = 0, mo = 0, d = 0, h = 0, mi = 0, sec = 0;
    if (!digits(0, 4, y) || s.size() < 10 || s[4] != '-' || s[7] != '-' ||
        !digits(5, 2, mo) || !digits(8, 2, d)) {
        return std::nullopt;
    }
    if (mo < 1 || mo > 12 || d < 1 || d > 31) return std::nullopt;
    if (s.size() > 10) {
        if ((s[10] != ' ' && s[10] != 'T') || !digits(11, 2, h) ||
            s.size() < 16 || s[13] != ':' || !digits(14, 2, mi)) {
            return std::nullopt;
        }
        if (s.size() >= 19 && s[16] == ':' && !digits(17, 2, sec)) {
            return std::nullopt;
        }
    }
    const long long days = days_from_civil(y, static_cast<unsigned>(mo),
                                           static_cast<unsigned>(d));
    return static_cast<double>(days * 86400LL + h * 3600LL + mi * 60LL + sec);
}

}  // namespace

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
        if (fields.empty()) {
            // The timestamp column may be numeric or an ISO-8601 date.
            if (auto ts = parse_iso_timestamp(token)) {
                fields.push_back(*ts);
                continue;
            }
        }
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
