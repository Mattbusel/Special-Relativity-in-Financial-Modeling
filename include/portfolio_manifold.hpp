#pragma once

/// @file include/portfolio_manifold.hpp
/// @brief N-Asset Minkowski Covariance Matrix and Spacetime Causal Graph.
///
/// # Module: Portfolio Manifold
///
/// ## Responsibility
/// Extend the 4D financial spacetime manifold to an N-asset setting.
/// Each asset is represented as a spacetime event in an (N+1)-dimensional
/// Lorentzian manifold (1 time axis + N price axes). The module provides:
///
///   - `AssetEvent`            — A single market observation for one asset.
///   - `MinkowskiCovariance`   — Accumulates N asset events and computes an
///                               NxN spacetime-interval covariance matrix.
///   - `SpacetimeCausalGraph`  — Directed graph where edge (i→j) exists iff
///                               asset_i is TIMELIKE-separated from asset_j.
///
/// ## Physical Interpretation
/// Two assets are TIMELIKE-separated when their spacetime interval
///   ds²(i,j) = −c²·Δt² + ΔP² + ΔV² + ΔM²  < 0
/// meaning asset_i "precedes" asset_j causally inside the market light cone.
/// The hypothesis is that causal (TIMELIKE) pairs exhibit positive lead-lag
/// correlation — asset_i's move CAUSES a future move in asset_j.
///
/// ## Guarantees
/// - All fallible operations return std::optional.
/// - No raw pointers; Eigen3 owns all matrix storage.
/// - Thread-safe reads: const methods are safe for concurrent access.
///
/// ## Dependencies
/// - Eigen3 (matrix arithmetic)
/// - srfm/manifold/n_asset_interval.hpp (NAssetEvent, IntervalType)
/// - srfm/tensor/n_asset_manifold.hpp   (NAssetManifold metric)

#include "srfm/manifold/n_asset_interval.hpp"
#include "srfm/tensor/n_asset_manifold.hpp"
#include "srfm/constants.hpp"

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

namespace srfm::portfolio {

// ─── AssetEvent ───────────────────────────────────────────────────────────────

/// A single market observation for one identified asset.
///
/// Embeds the observation as a 4-vector in financial spacetime:
///   [t, P, V, M]  (time, price, volume, market_cap)
struct AssetEvent {
    std::string asset_id;  ///< Ticker or unique identifier for this asset
    double      t;         ///< Market time coordinate (bar index or epoch seconds)
    double      P;         ///< Price coordinate (spatial axis 1)
    double      V;         ///< Volume coordinate  (spatial axis 2)
    double      M;         ///< Market-cap coordinate (spatial axis 3)

    /// Convert to a 4-vector [t, P, V, M] compatible with the 4D manifold.
    [[nodiscard]] Eigen::Vector4d to_four_vector() const noexcept;

    /// Convert to an NAssetEvent with a single price coordinate (for use with
    /// manifold::NAssetInterval when operating in 1-asset mode).
    [[nodiscard]] manifold::NAssetEvent to_nasset_event() const noexcept;
};

// ─── MinkowskiCovariance ──────────────────────────────────────────────────────

/// Accumulates a set of AssetEvents and computes an NxN covariance matrix
/// whose (i,j) entry is derived from the spacetime interval between asset i
/// and asset j.
///
/// ## Algorithm
/// For each ordered pair (i, j):
///   ds²(i,j) = −c²·Δt² + ΔP² + ΔV² + ΔM²
/// where Δx = event_j − event_i in each coordinate.
///
/// The covariance entry C(i,j) is defined as:
///   C(i,j) = exp(−|ds²(i,j)|)            (Gaussian kernel over interval)
///
/// This maps:
///   - TIMELIKE  pairs (ds² ≪ 0) → near 0 (strongly causal, large |ds²|)
///   - LIGHTLIKE pairs (ds² ≈ 0) → 1.0  (maximally correlated at light cone)
///   - SPACELIKE pairs (ds² ≫ 0) → near 0 (stochastic, acausal)
///
/// The diagonal is set to 1.0 (each asset is perfectly correlated with itself).
///
/// ## Usage
/// ```cpp
/// MinkowskiCovariance mc;
/// mc.add_asset(AssetEvent{"AAPL", 1.0, 150.0, 1e8, 2.4e12});
/// mc.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12});
/// auto cov = mc.compute_spacetime_covariance();
/// // cov is a 2x2 Eigen::MatrixXd
/// ```
class MinkowskiCovariance {
public:
    /// Construct with optional speed-of-information parameter.
    ///
    /// @param c_market  Speed-of-information constant (default: 1.0).
    explicit MinkowskiCovariance(double c_market =
                                     constants::SPEED_OF_INFORMATION) noexcept;

    /// Add an asset event to the manifold.
    ///
    /// Events are stored in insertion order. The i-th added event becomes
    /// row/column i of the output covariance matrix.
    ///
    /// @param event  Asset event to add.
    void add_asset(AssetEvent event);

    /// Return the number of asset events currently stored.
    [[nodiscard]] std::size_t size() const noexcept;

    /// Compute the NxN Minkowski covariance matrix.
    ///
    /// Requires at least 2 assets; returns nullopt if fewer are stored.
    ///
    /// @return NxN Eigen::MatrixXd where N = size(), or nullopt on failure.
    [[nodiscard]] std::optional<Eigen::MatrixXd>
    compute_spacetime_covariance() const noexcept;

    /// Compute the raw spacetime interval ds²(i, j) between asset pair (i, j).
    ///
    /// @param i  Index of first asset (0-based).
    /// @param j  Index of second asset (0-based).
    /// @return   ds²(i, j), or nullopt if indices are out of range.
    [[nodiscard]] std::optional<double>
    interval_correlation(std::size_t i, std::size_t j) const noexcept;

    /// Classify the spacetime interval between asset pair (i, j).
    ///
    /// @param i  Index of first asset (0-based).
    /// @param j  Index of second asset (0-based).
    /// @return   IntervalType (TIMELIKE / LIGHTLIKE / SPACELIKE), or nullopt
    ///           if indices are out of range.
    [[nodiscard]] std::optional<manifold::IntervalType>
    classify_pair(std::size_t i, std::size_t j) const noexcept;

    /// Read-only access to the stored asset events.
    [[nodiscard]] const std::vector<AssetEvent>& events() const noexcept;

    /// Clear all stored events.
    void clear() noexcept;

private:
    /// Compute ds²(i, j) from raw asset events.
    [[nodiscard]] double raw_interval(std::size_t i,
                                      std::size_t j) const noexcept;

    std::vector<AssetEvent> events_;
    double                  c_market_;
};

// ─── SpacetimeCausalGraph ─────────────────────────────────────────────────────

/// Directed causal graph over a set of asset events.
///
/// An edge (i → j) is added when asset_i is TIMELIKE-separated from asset_j:
///   ds²(i, j) < −LIGHTLIKE_THRESHOLD
///
/// Under the "causal influence hypothesis", a TIMELIKE edge means asset_i's
/// price dynamics causally precede and may predict asset_j's dynamics.
///
/// ## Representation
/// The adjacency matrix A is an NxN boolean matrix (stored as Eigen::MatrixXi):
///   A(i, j) = 1  iff  ds²(i, j) < −LIGHTLIKE_THRESHOLD
///   A(i, i) = 0  (no self-loops)
///
/// ## Usage
/// ```cpp
/// MinkowskiCovariance mc;
/// mc.add_asset(AssetEvent{"AAPL", 0.0, 150.0, 1e8, 2.4e12});
/// mc.add_asset(AssetEvent{"MSFT", 1.0, 290.0, 8e7, 2.1e12});
/// auto graph = SpacetimeCausalGraph::build(mc);
/// if (graph) {
///     auto adj = graph->adjacency_matrix();
///     // adj(0,1) == 1 means AAPL causally precedes MSFT
/// }
/// ```
class SpacetimeCausalGraph {
public:
    /// Build the causal graph from a MinkowskiCovariance instance.
    ///
    /// Requires at least 2 asset events. Returns nullopt if fewer are available.
    ///
    /// @param mc  MinkowskiCovariance with N >= 2 stored events.
    /// @return    SpacetimeCausalGraph, or nullopt on failure.
    [[nodiscard]] static std::optional<SpacetimeCausalGraph>
    build(const MinkowskiCovariance& mc) noexcept;

    /// Return the NxN adjacency matrix.
    /// A(i,j) = 1 means edge (i→j) exists (asset_i TIMELIKE before asset_j).
    [[nodiscard]] const Eigen::MatrixXi& adjacency_matrix() const noexcept;

    /// Return true iff an edge (i → j) exists.
    ///
    /// @param i  Source asset index (0-based).
    /// @param j  Destination asset index (0-based).
    [[nodiscard]] bool has_edge(std::size_t i, std::size_t j) const noexcept;

    /// Return the number of assets (graph nodes).
    [[nodiscard]] std::size_t n_assets() const noexcept;

    /// Return the out-degree of node i (number of assets that i causally precedes).
    [[nodiscard]] int out_degree(std::size_t i) const noexcept;

    /// Return the in-degree of node j (number of assets that causally precede j).
    [[nodiscard]] int in_degree(std::size_t j) const noexcept;

    /// Return asset IDs in the order they were added.
    [[nodiscard]] const std::vector<std::string>& asset_ids() const noexcept;

private:
    explicit SpacetimeCausalGraph(Eigen::MatrixXi adjacency,
                                   std::vector<std::string> ids) noexcept;

    Eigen::MatrixXi          adj_;       ///< NxN adjacency matrix
    std::vector<std::string> asset_ids_; ///< Asset identifiers in index order
};

} // namespace srfm::portfolio
