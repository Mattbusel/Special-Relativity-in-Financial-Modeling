#pragma once

/// @file include/srfm/data_loader.hpp
/// @brief CSV data loader for OHLCV market data — AGT-06.
///
/// # Module: DataLoader
///
/// ## Responsibility
/// Parse CSV files containing OHLCV market data into `std::vector<OHLCV>`.
/// Malformed or non-finite rows are skipped with a warning; the loader never
/// crashes on bad input.
///
/// ## Expected CSV Format
/// ```
/// timestamp,open,high,low,close,volume
/// 1,100.0,105.0,99.0,103.0,1000000
/// 2,103.0,107.0,102.0,106.5,1200000
/// ```
/// The first line is treated as a header and skipped.
///
/// ## Guarantees
/// - Never throws; returns `nullopt` on unrecoverable errors
/// - Skips individual bad rows rather than failing the entire load
/// - Does not modify any file or external state

#include "srfm/engine.hpp"

#include <optional>
#include <string>
#include <vector>

namespace srfm::core {

/// Loads OHLCV data from CSV files and stdin streams.
class DataLoader {
public:
    /// Load OHLCV bars from a CSV file on disk.
    ///
    /// # Arguments
    /// * `filepath` — Path to a CSV file with header row
    ///                (timestamp, open, high, low, close, volume)
    ///
    /// # Returns
    /// - `nullopt` if the file cannot be opened
    /// - Empty vector if the file has a header but no valid data rows
    /// - Vector of parsed bars, skipping any malformed or non-finite rows
    [[nodiscard]] static std::optional<std::vector<OHLCV>>
    load_csv(const std::string& filepath) noexcept;

    /// Parse OHLCV bars from a CSV-formatted string (useful for testing).
    ///
    /// Same format as `load_csv` — first line is treated as a header.
    ///
    /// # Returns
    /// Parsed bars.  Never returns `nullopt`; returns empty vector on
    /// completely empty or header-only input.
    [[nodiscard]] static std::vector<OHLCV>
    parse_csv_string(const std::string& csv_content) noexcept;

    /// Validate a single OHLCV bar.
    ///
    /// A bar is valid if:
    /// - All fields are finite
    /// - low <= open, close <= high
    /// - high >= low
    /// - volume >= 0
    [[nodiscard]] static bool validate_bar(const OHLCV& bar) noexcept;

private:
    /// Parse a single CSV data row into an OHLCV.
    /// Returns `nullopt` if the row is malformed or values are non-finite.
    [[nodiscard]] static std::optional<OHLCV>
    parse_row(const std::string& line) noexcept;
};

}  // namespace srfm::core
