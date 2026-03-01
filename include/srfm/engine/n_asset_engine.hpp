#pragma once
/**
 * @file  n_asset_engine.hpp
 * @brief Full pipeline engine: N-asset OHLCV → relativistic momentum signals.
 *
 * Module:  include/srfm/engine/
 * Stage:   4 — N-Asset Manifold
 *
 * ## Responsibility
 * Orchestrate the full SRFM pipeline for N assets:
 *   1. Accumulate OHLCV bars in a ring buffer.
 *   2. Estimate the N×N covariance matrix from log-returns.
 *   3. Construct an NAssetManifold from the covariance estimate.
 *   4. Compute per-asset relativistic momenta:
 *        β_i    = |Δprice_i / (c_market × prev_close_i)|
 *        γ_i    = 1 / sqrt(1 - β_i²)
 *        m_eff_i = volume_i / adv_baseline
 *        p_rel_i = γ_i × m_eff_i × close_i
 *   5. Compute the portfolio spacetime interval.
 *   6. Classify each asset and the portfolio into TIMELIKE/SPACELIKE/LIGHTLIKE.
 *
 * ## Guarantees
 * - All public methods are noexcept.
 * - All fallible operations return std::optional.
 * - No raw pointers.
 *
 * ## NOT Responsible For
 * - Fetching market data.
 * - Persistence of history.
 * - Cross-process coordination.
 */

#include "../manifold/n_asset_interval.hpp"
#include "../tensor/n_asset_manifold.hpp"

#include <deque>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

namespace srfm::engine {

using srfm::manifold::IntervalType;
using srfm::manifold::NAssetEvent;
using srfm::manifold::NAssetInterval;
using srfm::tensor::NAssetManifold;

// ── Constants ─────────────────────────────────────────────────────────────────

/// Maximum safe beta value (mirrors BETA_MAX_SAFE from momentum.hpp).
inline constexpr double ENGINE_BETA_MAX_SAFE = 0.9999;

// ── Input types ───────────────────────────────────────────────────────────────

/**
 * @brief A single OHLCV bar for one asset.
 */
struct OHLCVBar {
    double open;       ///< Opening price.
    double high;       ///< Highest price.
    double low;        ///< Lowest price.
    double close;      ///< Closing price.
    double volume;     ///< Volume traded.
    double timestamp;  ///< Bar timestamp (seconds since epoch or bar index).
};

/**
 * @brief The universe of assets processed by the engine.
 */
struct AssetUniverse {
    std::vector<std::string> names;  ///< Asset names in order.

    /**
     * @brief Return the number of assets.
     *
     * @return static_cast<int>(names.size()).
     */
    [[nodiscard]] int n() const noexcept {
        return static_cast<int>(names.size());
    }
};

// ── Configuration ─────────────────────────────────────────────────────────────

/**
 * @brief Engine configuration parameters.
 */
struct EngineConfig {
    double c_market{1.0};              ///< Market speed of light.
    double adv_baseline{1'000'000.0};  ///< ADV baseline for effective mass.
    int    lookback_bars{20};          ///< Bars required before covariance is valid.
    double lightlike_threshold{1e-10}; ///< ds² threshold for LIGHTLIKE classification.
};

// ── Output types ──────────────────────────────────────────────────────────────

/**
 * @brief Relativistic momentum result for a single asset.
 */
struct AssetMomentumResult {
    std::string  asset_name;            ///< Name of the asset.
    double       beta;                  ///< Lorentz beta (normalised velocity).
    double       gamma;                 ///< Lorentz gamma factor.
    double       m_eff;                 ///< Effective mass (volume / ADV_baseline).
    double       relativistic_momentum; ///< p_rel = γ × m_eff × close.
    IntervalType regime;                ///< Interval type for this asset.
};

/**
 * @brief Aggregated engine output for one process() call.
 */
struct EngineOutput {
    std::vector<AssetMomentumResult> assets;                  ///< Per-asset results.
    IntervalType                      portfolio_regime;        ///< Portfolio-level regime.
    double                            portfolio_interval_sq;   ///< Portfolio ds².
    double                            timestamp;               ///< Timestamp of latest bar.
};

// ── Engine ────────────────────────────────────────────────────────────────────

/**
 * @brief Full SRFM pipeline engine for N assets.
 *
 * Accumulates OHLCV bars, estimates covariance, builds an NAssetManifold,
 * computes relativistic momenta, and classifies market regimes.
 */
class NAssetEngine {
public:
    // ── Construction ─────────────────────────────────────────────────────────

    /**
     * @brief Construct an engine for the given asset universe.
     *
     * @param universe  The set of assets (names must be non-empty).
     * @param cfg       Engine configuration.
     */
    explicit NAssetEngine(AssetUniverse universe,
                          EngineConfig  cfg = {}) noexcept;

    // ── Data ingestion ────────────────────────────────────────────────────────

    /**
     * @brief Ingest one bar for every asset in the universe.
     *
     * Bars must be provided in the same order as universe.names.
     * The bar is appended to the history ring buffer; if history exceeds
     * lookback_bars * 2 the oldest entry is dropped.
     *
     * @param bars  Span of bars, one per asset (must have length n_assets()).
     * @return      std::monostate on success, std::nullopt if bar count mismatches.
     */
    [[nodiscard]] std::optional<std::monostate>
    ingest(std::span<const OHLCVBar> bars) noexcept;

    // ── Processing ────────────────────────────────────────────────────────────

    /**
     * @brief Run the full pipeline on accumulated history.
     *
     * Requires ready() == true (at least lookback_bars entries in history).
     *
     * Steps:
     *   1. Estimate covariance from log-returns.
     *   2. Build NAssetManifold.
     *   3. Compute per-asset β, γ, m_eff, p_rel, regime.
     *   4. Compute portfolio interval and regime.
     *
     * @return EngineOutput, or std::nullopt if not ready or covariance fails.
     */
    [[nodiscard]] std::optional<EngineOutput> process() const noexcept;

    /**
     * @brief Ingest bars then immediately process.
     *
     * Convenience wrapper: ingest(bars) followed by process().
     *
     * @param bars  Bars to ingest.
     * @return      EngineOutput or std::nullopt.
     */
    [[nodiscard]] std::optional<EngineOutput>
    ingest_and_process(std::span<const OHLCVBar> bars) noexcept;

    // ── Status ────────────────────────────────────────────────────────────────

    /**
     * @brief Return the number of assets.
     *
     * @return universe_.n().
     */
    [[nodiscard]] int n_assets() const noexcept;

    /**
     * @brief Return true if enough bars have been accumulated.
     *
     * @return history_.size() >= static_cast<size_t>(cfg_.lookback_bars).
     */
    [[nodiscard]] bool ready() const noexcept;

private:
    // ── Internal pipeline stages ──────────────────────────────────────────────

    /**
     * @brief Compute the N×N empirical covariance from accumulated log-returns.
     *
     * Uses the last lookback_bars bars. Log-return for bar t, asset i:
     *   r_ti = log(close[t][i] / close[t-1][i])
     * Covariance: Cov_ij = (1/(T-1)) Σ_t (r_ti - r̄_i)(r_tj - r̄_j)
     *
     * @return N×N covariance matrix, or std::nullopt if history too short or
     *         any close price is non-positive.
     */
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    estimate_covariance() const noexcept;

    /**
     * @brief Compute beta for one asset from two consecutive bars.
     *
     * beta = |Δprice / (c_market * prev_close)|, clamped to
     * [0, ENGINE_BETA_MAX_SAFE - 1e-9].
     *
     * @param prev_close  Previous closing price (must be > 0).
     * @param curr_close  Current closing price.
     * @return            Beta value, or std::nullopt if prev_close <= 0.
     */
    [[nodiscard]] std::optional<double>
    compute_beta(double prev_close, double curr_close) const noexcept;

    /**
     * @brief Compute the portfolio spacetime interval from two consecutive events.
     *
     * Builds NAssetEvents from the last two bar snapshots and calls
     * NAssetInterval::compute.
     *
     * @param manifold  The manifold to use for the metric.
     * @return          ds² and type, or std::nullopt on failure.
     */
    [[nodiscard]] std::optional<std::pair<double, IntervalType>>
    compute_portfolio_interval(const NAssetManifold& manifold) const noexcept;

    // ── Data members ──────────────────────────────────────────────────────────

    AssetUniverse universe_;   ///< Asset universe.
    EngineConfig  cfg_;        ///< Engine configuration.

    /// Ring buffer: each entry is a snapshot of N bars (one per asset).
    std::deque<std::vector<OHLCVBar>> history_;
};

// ── Inline trivial accessors ──────────────────────────────────────────────────

inline int NAssetEngine::n_assets() const noexcept {
    return universe_.n();
}

inline bool NAssetEngine::ready() const noexcept {
    return static_cast<int>(history_.size()) >= cfg_.lookback_bars;
}

} // namespace srfm::engine
