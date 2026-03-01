#pragma once

#include <cstddef>

/// @file include/srfm/constants.hpp
/// @brief Physical and financial constants for the SRFM system.
///
/// READ-ONLY for all agents except the designated maintainer.
/// Submit a CROSS_AGENT_REQUESTS.md entry to propose additions.

namespace srfm::constants {

// ─── Relativistic Bounds ──────────────────────────────────────────────────────

/// Maximum safe beta value. β must stay strictly below 1.0 (speed of light).
/// Set to 0.9999 to avoid numerical instability near the singularity.
static constexpr double BETA_MAX_SAFE = 0.9999;

/// Below this β, relativistic corrections are negligible (γ ≈ 1 + β²/2).
static constexpr double BETA_NEWTONIAN_THRESHOLD = 0.1;

// ─── Numerical Tolerances ─────────────────────────────────────────────────────

/// General floating-point comparison epsilon.
static constexpr double FLOAT_EPSILON = 1e-12;

/// Epsilon for metric invertibility check (det(g) must exceed this).
static constexpr double METRIC_SINGULARITY_EPSILON = 1e-14;

/// Minimum volatility to prevent a singular diagonal metric entry.
static constexpr double MIN_VOLATILITY = 1e-8;

// ─── Financial Spacetime ──────────────────────────────────────────────────────

/// Normalised speed of information propagation in the market frame.
/// Analogous to c = 1 in natural units.
static constexpr double SPEED_OF_INFORMATION = 1.0;

/// Default proper-time step for geodesic integration.
static constexpr double DEFAULT_GEODESIC_STEP = 0.01;

/// Default finite-difference step for numerical metric derivatives.
static constexpr double DEFAULT_FD_STEP = 1e-5;

// ─── Backtester Defaults ──────────────────────────────────────────────────────

/// Minimum number of bars required for a valid backtest run.
/// Below this threshold all metric computations return nullopt.
static constexpr std::size_t MIN_RETURN_SERIES_LENGTH = 30;

/// Default annualised risk-free rate (zero — excess-return framing by default).
static constexpr double DEFAULT_RISK_FREE_RATE = 0.0;

/// Default annualisation factor: 252 trading days per year.
static constexpr double ANNUALISATION_FACTOR = 252.0;

} // namespace srfm::constants
