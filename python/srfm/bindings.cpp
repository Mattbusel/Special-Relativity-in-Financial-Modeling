/// @file python/srfm/bindings.cpp
/// @brief pybind11 Python bindings for the SRFM C++ library.
///
/// Exposes the core SRFM types and algorithms to Python:
///
///   from srfm import SpacetimeInterval, LorentzTransform, Backtester
///   from srfm import MultiAssetEvent, CorrelationMetric, PortfolioGeodesic
///
/// ## Building
/// ```bash
/// pip install pybind11 eigen3
/// cmake -B build -DSRFM_BUILD_PYTHON=ON
/// cmake --build build --target srfm_python
/// pip install -e .
/// ```

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <pybind11/functional.h>

#include "srfm/constants.hpp"
#include "srfm/types.hpp"
#include "srfm/manifold.hpp"
#include "srfm/backtest.hpp"
#include "srfm/multi_asset.hpp"

#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace py = pybind11;
using namespace py::literals;

// ---------------------------------------------------------------------------
// Helper: convert std::optional to Python (None on nullopt)
// ---------------------------------------------------------------------------
template<typename T>
py::object opt_to_py(const std::optional<T>& opt) {
    if (opt) return py::cast(*opt);
    return py::none();
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(srfm, m) {
    m.doc() = R"doc(
Special Relativity in Financial Modeling (SRFM)

Python bindings for the C++ SRFM library.  Applies special-relativistic
mathematics (Lorentz transforms, spacetime intervals, geodesics) to
financial price series.

Core Hypothesis
---------------
Financial price series can be embedded in a pseudo-Riemannian spacetime:

    x^μ = (c·t,  P,  V^(1/4),  M^(1/3))

The spacetime interval ds² = −c²dt² + dP² + dV² + dM² classifies each
market bar as TIMELIKE (causal, predictive) or SPACELIKE (stochastic).

Empirical Result (Q1 2025)
--------------------------
TIMELIKE bars exhibit 1.27× lower next-bar absolute return variance than
SPACELIKE bars across 10 liquid instruments
(Bartlett p = 6×10⁻¹⁶, 10/11 assets significant after Bonferroni correction).

Quickstart
----------
>>> from srfm import SpacetimeInterval, LorentzTransform, Backtester
>>> SpacetimeInterval.classify(dt=1.0, dp=0.5, dv=0.1, dm=0.05)
'TIMELIKE'
>>> LorentzTransform.gamma(beta=0.8)
1.6666666666666667
>>> bt = Backtester()
>>> result = bt.run(prices=[100, 101, 99, 102], volumes=[1e6, 2e6, 1.5e6, 3e6])
>>> result.sharpe
...
)doc";

    // -----------------------------------------------------------------------
    // Constants
    // -----------------------------------------------------------------------
    m.attr("SPEED_OF_LIGHT")   = srfm::constants::SPEED_OF_INFORMATION;
    m.attr("BETA_MAX_SAFE")    = srfm::constants::BETA_MAX_SAFE;
    m.attr("LIGHTLIKE_EPS")    = srfm::constants::FLOAT_EPSILON;

    // -----------------------------------------------------------------------
    // IntervalType enum
    // -----------------------------------------------------------------------
    py::enum_<srfm::manifold::IntervalType>(m, "IntervalType",
        "Causal character of a spacetime interval.")
        .value("TIMELIKE",  srfm::manifold::IntervalType::Timelike,
               "ds² < 0 — causal, subluminal market movement.")
        .value("LIGHTLIKE", srfm::manifold::IntervalType::Lightlike,
               "ds² ≈ 0 — information propagation at the market speed of light.")
        .value("SPACELIKE", srfm::manifold::IntervalType::Spacelike,
               "ds² > 0 — stochastic, superluminal (no causal link).")
        .export_values();

    // -----------------------------------------------------------------------
    // SpacetimeInterval
    // -----------------------------------------------------------------------
    py::class_<srfm::manifold::SpacetimeInterval>(m, "SpacetimeInterval",
        R"doc(
Compute and classify spacetime intervals between market events.

The spacetime interval is:

    ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²

Classification:
    TIMELIKE   — ds² < 0  — causal regime (momentum is predictive)
    LIGHTLIKE  — ds² ≈ 0  — critical boundary
    SPACELIKE  — ds² > 0  — stochastic regime (bars are decorrelated)

Empirical Result
----------------
TIMELIKE bars show 1.27× lower next-bar absolute return variance.
(Bartlett p = 6×10⁻¹⁶, Q1 2025 data, 10 liquid instruments.)
)doc")
        .def_static("compute_ds2",
            [](double dt, double dp, double dv, double dm, double c) -> double {
                // ds² = −c²·Δt² + ΔP² + ΔV² + ΔM²
                return -(c * c * dt * dt) + dp * dp + dv * dv + dm * dm;
            },
            "dt"_a, "dp"_a, "dv"_a, "dm"_a,
            "c"_a = srfm::constants::SPEED_OF_INFORMATION,
            R"doc(
Compute ds² = −c²·Δt² + ΔP² + ΔV² + ΔM².

Parameters
----------
dt : float
    Time separation between bars (seconds or bar index).
dp : float
    Price separation ΔP = P₂ − P₁.
dv : float
    Volume separation ΔV = V₂^(1/4) − V₁^(1/4) (normalised).
dm : float
    Momentum separation ΔM = M₂^(1/3) − M₁^(1/3) (normalised).
c : float, optional
    Market speed of light (default 1.0).

Returns
-------
float
    Spacetime interval ds².
)doc")
        .def_static("classify",
            [](double dt, double dp, double dv, double dm, double c) -> std::string {
                const double ds2 = -(c * c * dt * dt) + dp * dp + dv * dv + dm * dm;
                const auto t = srfm::manifold::SpacetimeInterval::classify(ds2);
                return srfm::manifold::to_string(t);
            },
            "dt"_a, "dp"_a, "dv"_a, "dm"_a,
            "c"_a = srfm::constants::SPEED_OF_INFORMATION,
            R"doc(
Classify a spacetime interval.

Returns
-------
str
    "TIMELIKE", "LIGHTLIKE", or "SPACELIKE".
)doc")
        .def_static("classify_ds2",
            [](double ds2) -> std::string {
                return srfm::manifold::to_string(
                    srfm::manifold::SpacetimeInterval::classify(ds2));
            }, "ds2"_a,
            "Classify a pre-computed ds² value. Returns 'TIMELIKE', 'LIGHTLIKE', or 'SPACELIKE'.");

    // -----------------------------------------------------------------------
    // LorentzTransform
    // -----------------------------------------------------------------------
    py::class_<srfm::BetaVelocity>(m, "_BetaVelocity")
        .def(py::init<double>(), "value"_a)
        .def_readwrite("value", &srfm::BetaVelocity::value);

    py::class_<srfm::LorentzFactor>(m, "_LorentzFactor")
        .def(py::init<double>(), "value"_a)
        .def_readwrite("value", &srfm::LorentzFactor::value);

    py::class_<py::object>(m, "LorentzTransform",
        R"doc(
Utility class for Lorentz factor and relativistic momentum calculations.

All methods are static — no instance is needed.

Mathematical Definitions
------------------------
β = ΔP / (c · Δt)         — normalised price velocity
γ(β) = 1 / √(1 − β²)      — Lorentz factor
p_rel = γ · m_eff · p_raw  — relativistic momentum signal
)doc");
    // Since LorentzTransform in C++ is split across multiple files, we
    // expose a pure-Python static class backed by inline math.
    py::object lt_cls = m.attr("LorentzTransform");

    // Override with a concrete class.
    py::class_<srfm::LorentzFactor> lt(m, "LorentzTransform",
        "Static utility: Lorentz factor and relativistic momentum.");
    lt.def_static("gamma",
        [](double beta) -> double {
            const double b = std::clamp(std::abs(beta),
                                        0.0, srfm::constants::BETA_MAX_SAFE);
            return 1.0 / std::sqrt(1.0 - b * b);
        }, "beta"_a,
        R"doc(
Compute the Lorentz factor γ(β) = 1 / √(1 − β²).

Parameters
----------
beta : float
    Normalised market velocity β = ΔP / (c·Δt). Clamped to BETA_MAX_SAFE.

Returns
-------
float
    Lorentz factor γ ≥ 1.0.

Examples
--------
>>> LorentzTransform.gamma(0.0)
1.0
>>> LorentzTransform.gamma(0.8)
1.6666666666666667
>>> LorentzTransform.gamma(0.99)
7.088812050083354
)doc")
    .def_static("beta",
        [](double dp, double dt, double c) -> double {
            if (std::abs(dt) < 1e-15 || std::abs(c) < 1e-15) return 0.0;
            return std::clamp(dp / (c * dt),
                              -srfm::constants::BETA_MAX_SAFE,
                               srfm::constants::BETA_MAX_SAFE);
        }, "dp"_a, "dt"_a, "c"_a = srfm::constants::SPEED_OF_INFORMATION,
        "Compute β = ΔP / (c·Δt). Clamped to BETA_MAX_SAFE.")
    .def_static("relativistic_momentum",
        [](double gamma, double m_eff, double p_raw) -> double {
            return gamma * m_eff * p_raw;
        }, "gamma"_a, "m_eff"_a, "p_raw"_a,
        "Compute p_rel = γ · m_eff · p_raw.");

    // -----------------------------------------------------------------------
    // BacktestResult
    // -----------------------------------------------------------------------
    py::class_<srfm::backtest::PerformanceMetrics>(m, "PerformanceMetrics",
        "Performance metrics for a single (raw or relativistic) strategy.")
        .def_readonly("sharpe_ratio",     &srfm::backtest::PerformanceMetrics::sharpe_ratio,
                      "Annualised Sharpe ratio (mean_ret − r_f) / σ.")
        .def_readonly("sortino_ratio",    &srfm::backtest::PerformanceMetrics::sortino_ratio,
                      "Annualised Sortino ratio (mean_ret − r_f) / σ_down.")
        .def_readonly("max_drawdown",     &srfm::backtest::PerformanceMetrics::max_drawdown,
                      "Maximum peak-to-trough fractional loss (≥ 0).")
        .def_readonly("gamma_weighted_ir",&srfm::backtest::PerformanceMetrics::gamma_weighted_ir,
                      "γ-weighted information ratio vs benchmark.")
        .def("to_string", &srfm::backtest::PerformanceMetrics::to_string);

    py::class_<srfm::backtest::BacktestComparison>(m, "BacktestResult",
        R"doc(
Side-by-side comparison of raw vs relativistic strategy performance.

Attributes
----------
raw : PerformanceMetrics
    Metrics from the unmodified (unit-position) strategy.
relativistic : PerformanceMetrics
    Metrics from the γ-scaled (Lorentz-corrected) strategy.
mean_gamma : float
    Mean Lorentz factor γ across all bars.
max_gamma_applied : float
    Maximum γ multiplier applied (capped at BacktestConfig::max_gamma).
relativistic_lift : float
    IR_γ_relativistic / IR_γ_raw.  > 1 → relativistic lift.
sharpe : float
    Shortcut: relativistic.sharpe_ratio.
sortino : float
    Shortcut: relativistic.sortino_ratio.
max_drawdown : float
    Shortcut: relativistic.max_drawdown.
relativistic_sharpe : float
    Alias: relativistic.sharpe_ratio (for convenience).
)doc")
        .def_readonly("raw",              &srfm::backtest::BacktestComparison::raw)
        .def_readonly("relativistic",     &srfm::backtest::BacktestComparison::relativistic)
        .def_readonly("mean_gamma",       &srfm::backtest::BacktestComparison::mean_gamma)
        .def_readonly("max_gamma_applied",&srfm::backtest::BacktestComparison::max_gamma_applied)
        .def_readonly("relativistic_lift",&srfm::backtest::BacktestComparison::relativistic_lift)
        .def_property_readonly("sharpe",
            [](const srfm::backtest::BacktestComparison& c) {
                return c.relativistic.sharpe_ratio; })
        .def_property_readonly("sortino",
            [](const srfm::backtest::BacktestComparison& c) {
                return c.relativistic.sortino_ratio; })
        .def_property_readonly("max_drawdown",
            [](const srfm::backtest::BacktestComparison& c) {
                return c.relativistic.max_drawdown; })
        .def_property_readonly("relativistic_sharpe",
            [](const srfm::backtest::BacktestComparison& c) {
                return c.relativistic.sharpe_ratio; })
        .def("sharpe_lift",    &srfm::backtest::BacktestComparison::sharpe_lift)
        .def("sortino_lift",   &srfm::backtest::BacktestComparison::sortino_lift)
        .def("drawdown_delta", &srfm::backtest::BacktestComparison::drawdown_delta)
        .def("ir_lift",        &srfm::backtest::BacktestComparison::ir_lift)
        .def("to_string",      &srfm::backtest::BacktestComparison::to_string)
        .def("__repr__",       &srfm::backtest::BacktestComparison::to_string);

    // -----------------------------------------------------------------------
    // Backtester
    // -----------------------------------------------------------------------
    py::class_<srfm::backtest::BacktestConfig>(m, "BacktestConfig",
        "Configuration for a Backtester run.")
        .def(py::init<>())
        .def_readwrite("risk_free_rate", &srfm::backtest::BacktestConfig::risk_free_rate,
                       "Annualised risk-free rate (default 0.0).")
        .def_readwrite("annualisation",  &srfm::backtest::BacktestConfig::annualisation,
                       "Annualisation factor (default 252).")
        .def_readwrite("effective_mass", &srfm::backtest::BacktestConfig::effective_mass,
                       "m_eff in p_rel = γ m_eff signal (default 1.0).")
        .def_readwrite("max_gamma",      &srfm::backtest::BacktestConfig::max_gamma,
                       "Cap on γ position multiplier (default 3.0).")
        .def_readwrite("verbose",        &srfm::backtest::BacktestConfig::verbose);

    py::class_<srfm::backtest::Backtester>(m, "Backtester",
        R"doc(
Run a relativistic backtest over a price/volume series.

Usage
-----
>>> bt = Backtester()
>>> result = bt.run(prices=[100, 101, 99, 102], volumes=[1e6, 2e6, 1.5e6, 3e6])
>>> print(f"Sharpe: {result.sharpe:.3f}")
>>> print(f"Relativistic lift: {result.relativistic_lift:.3f}x")
>>> print(result.to_string())

The backtester:
1. Converts prices to log-returns.
2. Computes β = ΔP / (c · Δt) for each bar.
3. Computes γ(β) and applies it as a position multiplier.
4. Evaluates Sharpe, Sortino, max-drawdown, and γ-weighted IR for
   both raw (unit-position) and relativistic (γ-scaled) strategies.
)doc")
        .def(py::init<>())
        .def(py::init<srfm::backtest::BacktestConfig>(), "config"_a)
        .def("run",
            [](srfm::backtest::Backtester& self,
               const std::vector<double>& prices,
               const std::vector<double>& volumes) -> srfm::backtest::BacktestComparison
            {
                if (prices.size() < 3)
                    throw std::invalid_argument("Need at least 3 price bars.");
                if (!volumes.empty() && volumes.size() != prices.size())
                    throw std::invalid_argument("volumes must have the same length as prices.");

                // Build BarData from prices.
                std::vector<srfm::backtest::BarData> bars;
                bars.reserve(prices.size() - 1);

                const double c = srfm::constants::SPEED_OF_INFORMATION;
                for (std::size_t i = 1; i < prices.size(); ++i) {
                    const double dp   = prices[i] - prices[i-1];
                    const double dt   = 1.0; // unit bar time
                    const double beta = std::clamp(dp / (c * dt),
                                                   -srfm::constants::BETA_MAX_SAFE,
                                                    srfm::constants::BETA_MAX_SAFE);
                    const double ret  = prices[i-1] > 0 ?
                        std::log(prices[i] / prices[i-1]) : 0.0;
                    bars.push_back({ret, srfm::BetaVelocity{beta}, 0.0});
                }

                auto result = self.run(bars);
                if (!result)
                    throw std::runtime_error("Backtest failed (insufficient data or NaN inputs).");
                return *result;
            },
            "prices"_a, "volumes"_a = std::vector<double>{},
            R"doc(
Run a backtest over a price series.

Parameters
----------
prices : List[float]
    Closing prices (at least 3 bars required).
volumes : List[float], optional
    Traded volumes (unused in the current metric; reserved for future use).

Returns
-------
BacktestResult
    Side-by-side raw vs relativistic performance metrics.
)doc");

    // -----------------------------------------------------------------------
    // MultiAsset bindings
    // -----------------------------------------------------------------------

    // MultiAssetEvent
    py::class_<srfm::multi_asset::MultiAssetEvent>(m, "MultiAssetEvent",
        R"doc(
A snapshot of N correlated financial assets at a point in time.

Parameters
----------
symbols : List[str]
    Asset ticker symbols (optional, for labelling).
prices : List[float]
    Mid-prices for each asset.
volumes : List[float]
    Traded volumes for each asset.
timestamp : int
    Unix epoch milliseconds.
)doc")
        .def(py::init<>())
        .def_readwrite("symbols",   &srfm::multi_asset::MultiAssetEvent::symbols)
        .def_readwrite("prices",    &srfm::multi_asset::MultiAssetEvent::prices)
        .def_readwrite("volumes",   &srfm::multi_asset::MultiAssetEvent::volumes)
        .def_readwrite("timestamp", &srfm::multi_asset::MultiAssetEvent::timestamp)
        .def("n_assets",            &srfm::multi_asset::MultiAssetEvent::n_assets)
        .def("is_valid",            &srfm::multi_asset::MultiAssetEvent::is_valid);

    // CorrelationMetric
    using CMConfig = srfm::multi_asset::CorrelationMetric::Config;
    py::class_<CMConfig>(m, "CorrelationMetricConfig",
        "Configuration for CorrelationMetric.")
        .def(py::init<>())
        .def_readwrite("window_size",       &CMConfig::window_size)
        .def_readwrite("c_market",          &CMConfig::c_market)
        .def_readwrite("regularisation_eps",&CMConfig::regularisation_eps);

    py::class_<srfm::multi_asset::CorrelationMetric>(m, "CorrelationMetric",
        R"doc(
Builds the Lorentzian metric tensor from a rolling correlation matrix.

Maintains a sliding window of N-asset price observations and computes
the (N+1)×(N+1) metric tensor:

    g_00     = -c_market²
    g_ij     = Σ_ij  (rolling covariance)

Usage
-----
>>> cm = CorrelationMetric(n_assets=3, window_size=60)
>>> for row in price_rows:
...     metric = cm.update(row)
...     if metric is not None:
...         ds2 = MultiAssetInterval.compute(event_a, event_b, metric)
)doc")
        .def(py::init<std::size_t, CMConfig>(),
             "n_assets"_a, "config"_a = CMConfig{})
        .def("update",
            [](srfm::multi_asset::CorrelationMetric& self,
               const std::vector<double>& prices) -> py::object {
                auto result = self.update(prices);
                return result ? py::cast(*result) : py::none();
            }, "prices"_a, "Add a price observation. Returns (N+1)×(N+1) metric or None.")
        .def("current_metric",
            [](const srfm::multi_asset::CorrelationMetric& self) -> py::object {
                return opt_to_py(self.current_metric());
            }, "Return the last computed metric tensor, or None.")
        .def("correlation_matrix",
            [](const srfm::multi_asset::CorrelationMetric& self) -> py::object {
                return opt_to_py(self.correlation_matrix());
            }, "Return the N×N rolling correlation matrix, or None.")
        .def("n_assets",          &srfm::multi_asset::CorrelationMetric::n_assets)
        .def("observation_count", &srfm::multi_asset::CorrelationMetric::observation_count)
        .def("reset",             &srfm::multi_asset::CorrelationMetric::reset);

    // MultiAssetInterval
    using MAI = srfm::multi_asset::MultiAssetInterval;
    py::class_<MAI>(m, "MultiAssetInterval",
        "Compute ds² between N-asset spacetime events using the correlation metric.")
        .def_static("compute",
            [](const srfm::multi_asset::MultiAssetEvent& a,
               const srfm::multi_asset::MultiAssetEvent& b,
               const Eigen::MatrixXd& metric) -> py::object {
                return opt_to_py(MAI::compute(a, b, metric));
            }, "a"_a, "b"_a, "metric"_a,
            "Compute ds² = g_μν Δx^μ Δx^ν. Returns float or None on failure.")
        .def_static("classify",
            [](double ds2) -> std::string {
                return srfm::manifold::to_string(MAI::classify(ds2));
            }, "ds2"_a, "Classify ds² → 'TIMELIKE', 'LIGHTLIKE', or 'SPACELIKE'.");

    // MultiAssetLorentz
    using MAL = srfm::multi_asset::MultiAssetLorentz;
    using TRes = MAL::TransformResult;
    py::class_<TRes>(m, "MultiAssetTransformResult",
        "Result of a multi-asset Lorentz transform.")
        .def_readonly("adjusted_prices",  &TRes::adjusted_prices)
        .def_readonly("beta_portfolio",   &TRes::beta_portfolio)
        .def_readonly("gamma_portfolio",  &TRes::gamma_portfolio)
        .def_readonly("per_asset_beta",   &TRes::per_asset_beta)
        .def_readonly("per_asset_gamma",  &TRes::per_asset_gamma);

    py::class_<MAL>(m, "MultiAssetLorentz",
        "Apply simultaneous Lorentz boosts to N correlated price series.")
        .def_static("transform",
            [](const srfm::multi_asset::MultiAssetEvent& a,
               const srfm::multi_asset::MultiAssetEvent& b,
               const Eigen::MatrixXd& metric) -> py::object {
                return opt_to_py(MAL::transform(a, b, metric));
            }, "a"_a, "b"_a, "metric"_a,
            "Apply multi-asset Lorentz transform. Returns MultiAssetTransformResult or None.")
        .def_static("portfolio_beta",
            [](const std::vector<double>& betas, const Eigen::MatrixXd& metric) -> double {
                return MAL::portfolio_beta(betas, metric);
            }, "betas"_a, "metric"_a,
            "Compute metric-weighted portfolio velocity β_portfolio.");

    // PortfolioGeodesic
    using PG = srfm::multi_asset::PortfolioGeodesic;
    using PGStep = PG::GeodesicStep;
    py::class_<PGStep>(m, "GeodesicStep",
        "One step on the portfolio geodesic path.")
        .def_readonly("event",       &PGStep::event)
        .def_readonly("proper_time", &PGStep::proper_time)
        .def_readonly("path_length", &PGStep::path_length)
        .def_readonly("deviation",   &PGStep::deviation);

    py::class_<PG>(m, "PortfolioGeodesic",
        R"doc(
Compute geodesic paths and portfolio weights in multi-asset spacetime.

Under a constant metric (flat manifold), the geodesic is a straight line —
the "inertial" portfolio trajectory.  Deviations of actual prices from this
path are trading signals.
)doc")
        .def_static("integrate",
            [](const srfm::multi_asset::MultiAssetEvent& initial,
               const Eigen::VectorXd& four_velocity,
               const Eigen::MatrixXd& metric,
               std::size_t n_steps,
               double dt) -> std::vector<PGStep> {
                return PG::integrate(initial, four_velocity, metric, n_steps, dt);
            },
            "initial"_a, "four_velocity"_a, "metric"_a,
            "n_steps"_a, "dt"_a,
            "Integrate the geodesic equation for n_steps with step size dt.")
        .def_static("deviation_series",
            [](const std::vector<PGStep>& predicted,
               const std::vector<srfm::multi_asset::MultiAssetEvent>& actual)
            -> std::vector<double> {
                return PG::deviation_series(predicted, actual);
            },
            "predicted"_a, "actual"_a,
            "Compute ||predicted − actual||₂ at each step.")
        .def_static("portfolio_weights",
            [](const std::vector<PGStep>& steps,
               const Eigen::VectorXd& four_velocity,
               double gross_exposure) -> std::vector<Eigen::VectorXd> {
                return PG::portfolio_weights(steps, four_velocity, gross_exposure);
            },
            "steps"_a, "four_velocity"_a, "gross_exposure"_a = 1.0,
            "Return per-step portfolio weight vectors normalised to gross_exposure.");

} // PYBIND11_MODULE
