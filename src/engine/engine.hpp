#pragma once
/**
 * @file  engine.hpp
 * @brief Full SRFM pipeline engine: CSV → relativistic signal (AGT-13 / SRFM)
 *
 * Module:  src/engine/
 * Owner:   AGT-13  (Adversarial hardening)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Orchestrate the complete Special Relativity Financial Modelling pipeline:
 *
 *   CSV bytes → parse prices → BetaCalculator → SpacetimeManifold
 *             → GeodesicSolver → RelativisticSignalProcessor → PipelineResult
 *
 * The Engine is the primary fuzzing surface: it must handle arbitrary byte
 * sequences without crashing, UB, or infinite loops.
 *
 * Design Constraints
 * ------------------
 *   • Accepts std::string_view (including binary garbage).
 *   • Returns std::nullopt for any unparseable or physically invalid input.
 *   • Must never abort, crash, or access out-of-bounds memory.
 *   • All public methods are noexcept.
 *
 * NOT Responsible For
 *   • Network I/O or file system access.
 *   • Multi-frame state (stateless).
 */

#include <array>
#include <cmath>
#include <optional>
#include <string_view>
#include <vector>

#include "../manifold/spacetime_manifold.hpp"

namespace srfm::engine {

using manifold::Regime;

// ── PipelineResult ────────────────────────────────────────────────────────────

/**
 * @brief Full output of one Engine pipeline run.
 */
struct PipelineResult {
    double beta{0.0};              ///< Normalised market velocity β
    double gamma{1.0};             ///< Lorentz factor γ ≥ 1
    double rapidity{0.0};          ///< Rapidity φ = atanh(β)
    double doppler{1.0};           ///< Doppler factor D(β) > 0
    Regime regime{Regime::Newtonian}; ///< Relativistic regime classification
    double relativistic_signal{0.0};  ///< γ-corrected representative signal
    std::size_t price_count{0};       ///< Number of price observations parsed
};

// ── Engine ────────────────────────────────────────────────────────────────────

/**
 * @brief Stateless end-to-end SRFM pipeline engine.
 *
 * @example
 * @code
 *   Engine engine;
 *   auto result = engine.process("100.0,101.0,102.5,101.8,103.0");
 *   if (result) {
 *       // result->beta ≈ relativistic velocity of this price window
 *   }
 * @endcode
 */
class Engine {
public:
    Engine() noexcept = default;

    /**
     * @brief Process a CSV-like byte sequence through the full pipeline.
     *
     * Parsing rules:
     *   • Tokens are split on comma, newline, space, or tab.
     *   • Tokens that parse as finite positive doubles become price observations.
     *   • At least 2 valid price observations are required.
     *   • Non-numeric tokens are silently skipped.
     *
     * @param data  Arbitrary byte sequence (may contain binary, NaN text, etc.)
     * @return PipelineResult, or std::nullopt if fewer than 2 prices are found
     *         or if any physics computation fails.
     */
    [[nodiscard]] std::optional<PipelineResult>
    process(std::string_view data) const noexcept;

private:
    /// Parse tokens from data into a vector of finite positive prices.
    [[nodiscard]] std::vector<double>
    parse_prices(std::string_view data) const noexcept;
};

} // namespace srfm::engine
