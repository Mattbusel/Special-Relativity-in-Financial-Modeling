#pragma once

/// @file include/srfm/types.hpp
/// @brief Shared primitive types for the Special Relativity in Financial
///        Modeling (SRFM) system.
///
/// All agent modules include this file. It defines the core value types and
/// Eigen-based linear-algebra aliases used throughout the system.
///
/// This file is READ-ONLY for all agents except the one designated to
/// maintain shared types. Submit a `CROSS_AGENT_REQUESTS.md` entry to
/// propose changes.

#include <Eigen/Dense>
#include <optional>
#include <array>

namespace srfm {

/// Dimensionality of the financial spacetime manifold (1 time + 3 assets).
static constexpr int SPACETIME_DIM = 4;

// ─── Strong Scalar Types ──────────────────────────────────────────────────────

/// Market velocity as a fraction of the speed of information propagation.
/// Analogous to β = v/c in special relativity.
/// Valid range: (-BETA_MAX_SAFE, BETA_MAX_SAFE).
struct BetaVelocity {
    double value;
};

/// Lorentz factor γ = 1/√(1−β²). Always ≥ 1.0 for valid beta.
struct LorentzFactor {
    double value;
};

// ─── Linear Algebra Aliases ───────────────────────────────────────────────────

/// A point in the 4-dimensional financial spacetime manifold.
/// Component layout: [t, x¹, x², x³] = [time, asset₁, asset₂, asset₃].
using SpacetimePoint = Eigen::Vector<double, SPACETIME_DIM>;

/// A tangent vector at a spacetime point (four-velocity: dx^μ/dτ).
using FourVelocity = Eigen::Vector<double, SPACETIME_DIM>;

/// The covariant metric tensor g_μν: a 4×4 symmetric matrix.
using MetricMatrix = Eigen::Matrix<double, SPACETIME_DIM, SPACETIME_DIM>;

// ─── Signal Type ──────────────────────────────────────────────────────────────

/// A financial signal with relativistic corrections applied.
struct RelativisticSignal {
    double        raw_value;      ///< Original signal before correction
    LorentzFactor gamma;          ///< Lorentz factor used
    double        adjusted_value; ///< γ · m_eff · raw_value
    std::optional<double> time;   ///< Proper time stamp (caller-set)
};

} // namespace srfm
