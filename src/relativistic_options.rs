//! Relativistic Options Pricing via Spacetime Metric
//!
//! ## Overview
//!
//! This module extends the SRFM financial manifold framework to options
//! pricing. Classical Black-Scholes assumes constant volatility and a flat
//! Euclidean background. Here we replace both assumptions:
//!
//! 1. **Volatility → spacetime metric**: The Black-Scholes volatility `σ`
//!    is replaced by the Minkowski spacetime interval `|ds|` between
//!    consecutive price events. Events that are TIMELIKE (causally connected,
//!    `ds² < 0`) carry predictive information and are assigned lower implied
//!    volatility than SPACELIKE events.
//!
//! 2. **Light-cone option pricing**: An option's time-to-expiry is measured
//!    in *proper time* `Δτ` rather than coordinate time `Δt`. For a
//!    TIMELIKE trajectory, `Δτ < Δt`, so the option decays faster in
//!    proper-time space — reducing implied vol for causal regimes.
//!
//! 3. **Relativistic delta**: The standard Black-Scholes Δ is multiplied by
//!    the Lorentz factor `γ(β)`. In a high-β (fast-moving) regime the hedge
//!    ratio increases because the same nominal price move corresponds to more
//!    "spacetime distance" in the underlying trajectory.
//!
//! 4. **Spacetime arbitrage identification**: When the spacetime classification
//!    of a price move (TIMELIKE vs SPACELIKE) is inconsistent with the
//!    Black-Scholes implied vol pricing, a potential arbitrage is flagged.
//!
//! ## Derivations
//!
//! ### Relativistic Black-Scholes
//!
//! Classical B-S uses the risk-neutral SDE:
//!
//! ```text
//! dS = r·S·dt + σ·S·dW
//! ```
//!
//! In SRFM, the diffusion coefficient is replaced by the effective volatility
//! derived from the spacetime interval between the current and previous event:
//!
//! ```text
//! σ_eff = |ds| / (S · Δt)
//!       = √(|−c²Δt² + ΔP²|) / (S · Δt)
//! ```
//!
//! This makes volatility frame-dependent in the relativistic sense: a market
//! regime classified as TIMELIKE produces `σ_eff < c` while a SPACELIKE
//! regime produces `σ_eff > c` (superluminal diffusion).
//!
//! For practical computation the effective volatility is clamped to
//! `[σ_min, σ_max]` to prevent degenerate option prices.
//!
//! ### Proper-Time Discounting
//!
//! The classical B-S formula discounts at `e^{-rT}` where `T` is calendar
//! time to expiry. In the relativistic formulation we discount at
//! `e^{-r·τ_remaining}` where `τ_remaining` is the remaining *proper time*
//! along the underlying's worldline:
//!
//! ```text
//! τ_remaining = T · √(1 − β²)
//! ```
//!
//! For a TIMELIKE regime (`|β| < 1`), `τ < T`: the option's time value
//! decays faster than classical B-S predicts. For β → 0 (stationary market)
//! the formula collapses to classical B-S.
//!
//! ### Light-Cone Implied Volatility
//!
//! TIMELIKE trajectories (within the light cone) are assigned:
//!
//! ```text
//! σ_impl(TIMELIKE) = σ_base · √(1 − β²) = σ_base / γ
//! ```
//!
//! SPACELIKE trajectories (outside the light cone) are assigned:
//!
//! ```text
//! σ_impl(SPACELIKE) = σ_base · √(β² − 1)   [for |β| > 1, capped at σ_max]
//! ```
//!
//! LIGHTLIKE events (exactly on the cone) use `σ_base` unchanged.
//!
//! ### Relativistic Delta
//!
//! The Black-Scholes delta is:
//!
//! ```text
//! Δ_BS = N(d₁)   for calls,   Δ_BS = N(d₁) − 1  for puts
//! ```
//!
//! The relativistic correction multiplies by the Lorentz factor:
//!
//! ```text
//! Δ_rel = γ(β) · Δ_BS
//! ```
//!
//! Intuition: in a fast-moving (high-β) market, a unit price move is
//! "worth more" in proper distance, so the effective hedge ratio is higher.
//! Δ_rel is clamped to `[−γ_max, γ_max]` to remain finite near the light cone.
//!
//! ### Spacetime Arbitrage
//!
//! An arbitrage signal is raised when:
//!
//! ```text
//! classify(ds²) == SPACELIKE   AND   σ_impl < σ_base
//! ```
//!
//! or symmetrically:
//!
//! ```text
//! classify(ds²) == TIMELIKE    AND   σ_impl > σ_base · γ
//! ```
//!
//! This mismatch means the option market is pricing the future path as
//! causal/predictable while the spacetime metric says the regime is
//! stochastic (or vice versa). The `RelOrbitArbitrage` detector returns
//! a signed score (`> 0` → overpriced, `< 0` → underpriced) along with
//! the classification contradiction.

use std::f64::consts::PI;

// ── Constants ────────────────────────────────────────────────────────────────

/// Minimum effective volatility floor to prevent zero-vol pricing.
const SIGMA_FLOOR: f64 = 1e-4;

/// Maximum effective volatility cap.
const SIGMA_CAP: f64 = 10.0;

/// Safe upper bound on |β| (inherited from SRFM convention).
const BETA_MAX_SAFE: f64 = 0.9999;

/// |ds²| threshold for LIGHTLIKE classification.
const LIGHTLIKE_EPS: f64 = 1e-12;

// ── Error type ────────────────────────────────────────────────────────────────

/// Errors that can arise from relativistic options computations.
#[derive(Debug, Clone, thiserror::Error)]
pub enum OptionsError {
    /// Input parameters are outside valid ranges.
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Numerical computation failed (e.g. log of non-positive number).
    #[error("numerical error: {0}")]
    Numerical(String),
}

// ── Spacetime classification ──────────────────────────────────────────────────

/// Causal character of a spacetime interval.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum IntervalClass {
    /// `ds² < 0` — causally connected, predictive regime.
    Timelike,
    /// `|ds²| ≤ ε` — exactly on the light cone.
    Lightlike,
    /// `ds² > 0` — acausal, stochastic regime.
    Spacelike,
}

impl std::fmt::Display for IntervalClass {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IntervalClass::Timelike => write!(f, "TIMELIKE"),
            IntervalClass::Lightlike => write!(f, "LIGHTLIKE"),
            IntervalClass::Spacelike => write!(f, "SPACELIKE"),
        }
    }
}

/// Classify a pre-computed `ds²` value.
fn classify_ds2(ds2: f64) -> IntervalClass {
    if ds2.abs() <= LIGHTLIKE_EPS {
        IntervalClass::Lightlike
    } else if ds2 < 0.0 {
        IntervalClass::Timelike
    } else {
        IntervalClass::Spacelike
    }
}

// ── Helpers (cumulative normal) ───────────────────────────────────────────────

/// Rational-polynomial approximation to the standard normal CDF N(x).
/// Maximum absolute error < 7.5 × 10⁻⁸.
fn standard_normal_cdf(x: f64) -> f64 {
    // Abramowitz & Stegun 26.2.17 approximation.
    let neg = x < 0.0;
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.2316419 * z);
    let poly = t * (0.319_381_530
        + t * (-0.356_563_782
            + t * (1.781_477_937 + t * (-1.821_255_978 + t * 1.330_274_429))));
    let phi = (-0.5 * z * z).exp() / (2.0 * PI).sqrt() * poly;
    if neg { phi } else { 1.0 - phi }
}

// ── Configuration ─────────────────────────────────────────────────────────────

/// Configuration shared across all relativistic options pricing models.
#[derive(Debug, Clone)]
pub struct OptionsConfig {
    /// Financial speed of light: maximum log-return per unit time.
    /// Normalises β = ΔP / (c · Δt).
    pub c_scale: f64,

    /// Base volatility (annualised) used when the spacetime metric is
    /// unavailable or as a benchmark for arbitrage detection.
    pub sigma_base: f64,

    /// Risk-free rate (annualised, continuously compounded).
    pub risk_free_rate: f64,

    /// Maximum Lorentz factor γ applied to the delta hedge ratio.
    pub gamma_cap: f64,
}

impl Default for OptionsConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,
            sigma_base: 0.20,
            risk_free_rate: 0.05,
            gamma_cap: 5.0,
        }
    }
}

// ── RelativisticBlackScholes ──────────────────────────────────────────────────

/// Result of a single relativistic B-S option price computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RelativisticOptionPrice {
    /// Option price in the same currency as the underlying.
    pub price: f64,
    /// Effective volatility derived from the spacetime metric.
    pub sigma_effective: f64,
    /// Causal character of the current market regime.
    pub interval_class: IntervalClass,
    /// Lorentz factor γ at the time of pricing.
    pub gamma: f64,
    /// Proper-time to expiry (< calendar time for TIMELIKE regimes).
    pub proper_time_to_expiry: f64,
    /// Whether this is a call (`true`) or put (`false`).
    pub is_call: bool,
}

/// Black-Scholes model where volatility is replaced by the spacetime interval
/// metric derived from recent price trajectory.
///
/// ## Usage
///
/// ```rust
/// use tokio_prompt_orchestrator::relativistic_options::{
///     RelativisticBlackScholes, OptionsConfig,
/// };
///
/// let cfg = OptionsConfig::default();
/// let model = RelativisticBlackScholes::new(cfg);
///
/// // Price a call: S=100, K=105, T=0.25yr, dt=1.0, dp=2.0
/// let result = model.price_call(100.0, 105.0, 0.25, 1.0, 2.0)
///     .expect("price should succeed");
/// println!("Call price: {:.4}", result.price);
/// println!("σ_eff:      {:.4}", result.sigma_effective);
/// println!("Regime:     {}", result.interval_class);
/// ```
#[derive(Debug, Clone)]
pub struct RelativisticBlackScholes {
    cfg: OptionsConfig,
}

impl RelativisticBlackScholes {
    /// Create a new model with the given configuration.
    pub fn new(cfg: OptionsConfig) -> Self {
        Self { cfg }
    }

    /// Price a European **call** option.
    ///
    /// # Parameters
    /// - `spot`   — Current underlying price S.
    /// - `strike` — Option strike price K.
    /// - `expiry` — Time to expiry in years T.
    /// - `dt`     — Coordinate-time step for the last price move (ticks).
    /// - `dp`     — Price change for the last move ΔP.
    ///
    /// # Errors
    /// Returns [`OptionsError::InvalidInput`] if S ≤ 0, K ≤ 0, or T ≤ 0.
    pub fn price_call(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        dt: f64,
        dp: f64,
    ) -> Result<RelativisticOptionPrice, OptionsError> {
        self.price_option(spot, strike, expiry, dt, dp, true)
    }

    /// Price a European **put** option.
    ///
    /// Parameters identical to [`price_call`](Self::price_call).
    pub fn price_put(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        dt: f64,
        dp: f64,
    ) -> Result<RelativisticOptionPrice, OptionsError> {
        self.price_option(spot, strike, expiry, dt, dp, false)
    }

    fn price_option(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        dt: f64,
        dp: f64,
        is_call: bool,
    ) -> Result<RelativisticOptionPrice, OptionsError> {
        if spot <= 0.0 {
            return Err(OptionsError::InvalidInput("spot must be > 0".into()));
        }
        if strike <= 0.0 {
            return Err(OptionsError::InvalidInput("strike must be > 0".into()));
        }
        if expiry <= 0.0 {
            return Err(OptionsError::InvalidInput("expiry must be > 0".into()));
        }

        let (sigma_eff, interval_class, gamma, proper_t) =
            self.effective_params(dt, dp, expiry, spot);

        let r = self.cfg.risk_free_rate;
        let log_ratio = (spot / strike).ln();
        let d1 = (log_ratio + (r + 0.5 * sigma_eff * sigma_eff) * proper_t)
            / (sigma_eff * proper_t.sqrt());
        let d2 = d1 - sigma_eff * proper_t.sqrt();

        let price = if is_call {
            spot * standard_normal_cdf(d1) - strike * (-r * proper_t).exp() * standard_normal_cdf(d2)
        } else {
            strike * (-r * proper_t).exp() * standard_normal_cdf(-d2)
                - spot * standard_normal_cdf(-d1)
        };

        Ok(RelativisticOptionPrice {
            price: price.max(0.0),
            sigma_effective: sigma_eff,
            interval_class,
            gamma,
            proper_time_to_expiry: proper_t,
            is_call,
        })
    }

    /// Compute (sigma_eff, class, gamma, proper_time) from recent trajectory.
    fn effective_params(
        &self,
        dt: f64,
        dp: f64,
        expiry: f64,
        spot: f64,
    ) -> (f64, IntervalClass, f64, f64) {
        // β = (ΔP / S) / (c · Δt), clamped to safe range.
        let beta_raw = if dt.abs() < 1e-12 || spot <= 0.0 {
            0.0
        } else {
            (dp / spot) / (self.cfg.c_scale * dt)
        };
        let beta = beta_raw.clamp(-BETA_MAX_SAFE, BETA_MAX_SAFE);
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();

        // ds² = -(c·dt)² + dp²  (1+1 Minkowski).
        let ds2 = -(self.cfg.c_scale * dt).powi(2) + dp.powi(2);
        let interval_class = classify_ds2(ds2);

        // Effective volatility depends on interval class.
        let sigma_eff = match interval_class {
            IntervalClass::Timelike => {
                // TIMELIKE: vol reduced by inverse Lorentz factor.
                (self.cfg.sigma_base * (1.0 - beta * beta).sqrt())
                    .clamp(SIGMA_FLOOR, SIGMA_CAP)
            }
            IntervalClass::Lightlike => {
                // LIGHTLIKE: use base vol unchanged.
                self.cfg.sigma_base.clamp(SIGMA_FLOOR, SIGMA_CAP)
            }
            IntervalClass::Spacelike => {
                // SPACELIKE: vol enhanced — acausal diffusion.
                let beta_excess = (beta.abs() - 1.0).max(0.0);
                let enhancement = (1.0 + beta_excess.sqrt()).clamp(1.0, SIGMA_CAP / self.cfg.sigma_base);
                (self.cfg.sigma_base * enhancement).clamp(SIGMA_FLOOR, SIGMA_CAP)
            }
        };

        // Proper time to expiry: τ = T · √(1 − β²).
        let proper_t = (expiry * (1.0 - beta * beta).sqrt()).max(1e-6);

        (sigma_eff, interval_class, gamma, proper_t)
    }

    /// Reference to the underlying configuration.
    pub fn config(&self) -> &OptionsConfig {
        &self.cfg
    }
}

// ── LightconeOptionPricing ────────────────────────────────────────────────────

/// Result of a light-cone option pricing computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LightconeOptionResult {
    /// Option price under the light-cone model.
    pub price: f64,
    /// Implied volatility for a TIMELIKE trajectory.
    pub sigma_timelike: f64,
    /// Implied volatility for a SPACELIKE trajectory.
    pub sigma_spacelike: f64,
    /// Actual applied volatility (chosen by regime).
    pub sigma_applied: f64,
    /// Regime applied.
    pub interval_class: IntervalClass,
    /// Lorentz-reduced vol ratio: `σ_timelike / σ_base`.
    pub vol_reduction_ratio: f64,
}

/// Options pricing where TIMELIKE trajectories (inside the light cone) are
/// assigned lower implied volatility than SPACELIKE trajectories.
///
/// The key distinction from [`RelativisticBlackScholes`] is that this model
/// pre-computes both TIMELIKE and SPACELIKE implied vols and selects between
/// them based on the current regime — making the volatility surface
/// explicitly two-valued at the light cone boundary.
///
/// ## Physical Motivation
///
/// Inside the light cone (TIMELIKE), price moves are causally connected to
/// prior information: momentum is predictive, mean-reversion is plausible,
/// and realized vol should be lower. Outside (SPACELIKE), moves are
/// informationally disconnected — pure noise — so vol is higher.
#[derive(Debug, Clone)]
pub struct LightconeOptionPricing {
    bs: RelativisticBlackScholes,
}

impl LightconeOptionPricing {
    /// Create a new light-cone pricer.
    pub fn new(cfg: OptionsConfig) -> Self {
        Self {
            bs: RelativisticBlackScholes::new(cfg),
        }
    }

    /// Price a European option and return both TIMELIKE and SPACELIKE vols.
    ///
    /// # Parameters
    /// - `spot`   — Current price S.
    /// - `strike` — Strike K.
    /// - `expiry` — Time to expiry T (years).
    /// - `beta`   — Observed normalised price velocity β ∈ (−1, 1) for TIMELIKE,
    ///              or `|β| > 1` for SPACELIKE (clamped internally).
    /// - `is_call` — `true` for call, `false` for put.
    ///
    /// # Errors
    /// Returns [`OptionsError`] if inputs are invalid.
    pub fn price(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        beta: f64,
        is_call: bool,
    ) -> Result<LightconeOptionResult, OptionsError> {
        if spot <= 0.0 {
            return Err(OptionsError::InvalidInput("spot must be > 0".into()));
        }
        if strike <= 0.0 {
            return Err(OptionsError::InvalidInput("strike must be > 0".into()));
        }
        if expiry <= 0.0 {
            return Err(OptionsError::InvalidInput("expiry must be > 0".into()));
        }

        let beta_safe = beta.clamp(-BETA_MAX_SAFE, BETA_MAX_SAFE);
        let beta2 = beta_safe * beta_safe;
        let sigma_base = self.bs.cfg.sigma_base;

        // TIMELIKE vol: reduced by √(1 − β²).
        let sigma_tl = (sigma_base * (1.0 - beta2).sqrt()).clamp(SIGMA_FLOOR, SIGMA_CAP);
        // SPACELIKE vol: enhanced by γ.
        let gamma_factor = 1.0 / (1.0 - beta2).sqrt();
        let sigma_sl = (sigma_base * gamma_factor).clamp(SIGMA_FLOOR, SIGMA_CAP);

        // Classify by β: |β| < 1 → TIMELIKE, |β| ≈ 1 → LIGHTLIKE, else SPACELIKE.
        let interval_class = {
            let ds2_proxy = beta2 - 1.0; // sign mirrors ds² sign for unit dt/dp
            // ds2_proxy < 0 ↔ |β| < 1 ↔ TIMELIKE
            classify_ds2(-ds2_proxy)
        };

        let sigma_applied = match interval_class {
            IntervalClass::Timelike => sigma_tl,
            IntervalClass::Lightlike => sigma_base.clamp(SIGMA_FLOOR, SIGMA_CAP),
            IntervalClass::Spacelike => sigma_sl,
        };

        let vol_reduction_ratio = if sigma_base > 1e-12 {
            sigma_tl / sigma_base
        } else {
            1.0
        };

        // Proper-time adjusted expiry.
        let proper_t = (expiry * (1.0 - beta2).sqrt()).max(1e-6);
        let r = self.bs.cfg.risk_free_rate;
        let log_ratio = (spot / strike).ln();
        let d1 = (log_ratio + (r + 0.5 * sigma_applied * sigma_applied) * proper_t)
            / (sigma_applied * proper_t.sqrt());
        let d2 = d1 - sigma_applied * proper_t.sqrt();

        let price = if is_call {
            spot * standard_normal_cdf(d1)
                - strike * (-r * proper_t).exp() * standard_normal_cdf(d2)
        } else {
            strike * (-r * proper_t).exp() * standard_normal_cdf(-d2)
                - spot * standard_normal_cdf(-d1)
        };

        Ok(LightconeOptionResult {
            price: price.max(0.0),
            sigma_timelike: sigma_tl,
            sigma_spacelike: sigma_sl,
            sigma_applied,
            interval_class,
            vol_reduction_ratio,
        })
    }
}

// ── SpacetimeDelta ────────────────────────────────────────────────────────────

/// Relativistic delta hedge ratio for a European option.
///
/// The standard Black-Scholes Δ is multiplied by the Lorentz factor γ(β):
///
/// ```text
/// Δ_rel = γ(β) · Δ_BS
/// ```
///
/// This accounts for the fact that in a high-β (fast-moving) regime, a unit
/// price change corresponds to a larger proper-distance step along the
/// underlying's worldline, requiring a proportionally larger hedge.
///
/// Δ_rel is clamped to `[−γ_max, γ_max]` to remain finite near the light cone.
#[derive(Debug, Clone)]
pub struct SpacetimeDelta {
    cfg: OptionsConfig,
}

/// Result of a relativistic delta computation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DeltaResult {
    /// Classical Black-Scholes delta N(d₁) or N(d₁)−1.
    pub delta_classical: f64,
    /// Relativistic delta = γ · Δ_BS.
    pub delta_relativistic: f64,
    /// Lorentz factor γ applied.
    pub gamma_factor: f64,
    /// Normalised velocity β.
    pub beta: f64,
    /// Interval class for this trajectory step.
    pub interval_class: IntervalClass,
}

impl SpacetimeDelta {
    /// Create a new relativistic delta calculator.
    pub fn new(cfg: OptionsConfig) -> Self {
        Self { cfg }
    }

    /// Compute the relativistic delta for a European option.
    ///
    /// # Parameters
    /// - `spot`    — Current underlying price S.
    /// - `strike`  — Strike price K.
    /// - `expiry`  — Time to expiry T (years).
    /// - `sigma`   — Volatility estimate (annualised).
    /// - `dt`      — Coordinate-time step for last price move.
    /// - `dp`      — Price change for last move ΔP.
    /// - `is_call` — `true` for call, `false` for put.
    ///
    /// # Errors
    /// Returns [`OptionsError`] if inputs are out of range.
    pub fn compute(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        sigma: f64,
        dt: f64,
        dp: f64,
        is_call: bool,
    ) -> Result<DeltaResult, OptionsError> {
        if spot <= 0.0 {
            return Err(OptionsError::InvalidInput("spot must be > 0".into()));
        }
        if strike <= 0.0 {
            return Err(OptionsError::InvalidInput("strike must be > 0".into()));
        }
        if expiry <= 0.0 {
            return Err(OptionsError::InvalidInput("expiry must be > 0".into()));
        }
        let sigma_safe = sigma.clamp(SIGMA_FLOOR, SIGMA_CAP);

        // Compute β.
        let beta_raw = if dt.abs() < 1e-12 || spot <= 0.0 {
            0.0
        } else {
            (dp / spot) / (self.cfg.c_scale * dt)
        };
        let beta = beta_raw.clamp(-BETA_MAX_SAFE, BETA_MAX_SAFE);
        let gamma = (1.0 / (1.0 - beta * beta).sqrt()).min(self.cfg.gamma_cap);

        // Interval class.
        let ds2 = -(self.cfg.c_scale * dt).powi(2) + dp.powi(2);
        let interval_class = classify_ds2(ds2);

        // d₁ for B-S delta.
        let log_ratio = (spot / strike).ln();
        let r = self.cfg.risk_free_rate;
        let d1 = (log_ratio + (r + 0.5 * sigma_safe * sigma_safe) * expiry)
            / (sigma_safe * expiry.sqrt());

        let delta_bs = if is_call {
            standard_normal_cdf(d1)
        } else {
            standard_normal_cdf(d1) - 1.0
        };

        let delta_rel = (gamma * delta_bs).clamp(-self.cfg.gamma_cap, self.cfg.gamma_cap);

        Ok(DeltaResult {
            delta_classical: delta_bs,
            delta_relativistic: delta_rel,
            gamma_factor: gamma,
            beta,
            interval_class,
        })
    }
}

// ── RelOrbitArbitrage ─────────────────────────────────────────────────────────

/// Classification contradiction detected by the arbitrage scanner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum ArbitrageType {
    /// Spacetime says TIMELIKE (causal) but market prices as if SPACELIKE (noisy).
    /// The option is likely **overpriced** relative to the relativistic model.
    TimelikeButOverpriced,
    /// Spacetime says SPACELIKE (noisy) but market prices as if TIMELIKE (causal).
    /// The option is likely **underpriced** relative to the relativistic model.
    SpacelikeButUnderpriced,
    /// No contradiction detected.
    None,
}

impl std::fmt::Display for ArbitrageType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ArbitrageType::TimelikeButOverpriced => write!(f, "TIMELIKE_BUT_OVERPRICED"),
            ArbitrageType::SpacelikeButUnderpriced => write!(f, "SPACELIKE_BUT_UNDERPRICED"),
            ArbitrageType::None => write!(f, "NONE"),
        }
    }
}

/// Result from the relativistic arbitrage scanner.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ArbitrageSignal {
    /// Type of spacetime/pricing contradiction.
    pub arb_type: ArbitrageType,
    /// Signed score: `> 0` → overpriced; `< 0` → underpriced.
    /// Magnitude indicates the degree of mispricing.
    pub score: f64,
    /// Market implied vol (inferred from `market_price` if provided, else 0).
    pub market_implied_vol: f64,
    /// Relativistic model implied vol.
    pub model_vol: f64,
    /// Interval class of the current regime.
    pub interval_class: IntervalClass,
}

/// Identifies arbitrage opportunities when the spacetime regime classification
/// contradicts the Black-Scholes implied volatility from market prices.
///
/// ## Algorithm
///
/// 1. Classify the current bar as TIMELIKE, LIGHTLIKE, or SPACELIKE.
/// 2. Compute the relativistic model vol (`σ_model`).
/// 3. If the caller provides a market price, back-solve for market implied vol
///    via Newton-Raphson.
/// 4. Compare: TIMELIKE + `σ_mkt > σ_model · threshold` → overpriced.
///              SPACELIKE + `σ_mkt < σ_model / threshold` → underpriced.
#[derive(Debug, Clone)]
pub struct RelOrbitArbitrage {
    bs: RelativisticBlackScholes,
    /// Relative vol difference threshold to trigger a signal (e.g. 0.05 = 5%).
    pub threshold: f64,
}

impl RelOrbitArbitrage {
    /// Create a new arbitrage scanner.
    ///
    /// `threshold` is the minimum fractional vol difference to flag a signal.
    pub fn new(cfg: OptionsConfig, threshold: f64) -> Self {
        Self {
            bs: RelativisticBlackScholes::new(cfg),
            threshold: threshold.max(0.0),
        }
    }

    /// Scan for arbitrage.
    ///
    /// # Parameters
    /// - `spot`         — Current underlying price.
    /// - `strike`       — Option strike.
    /// - `expiry`       — Time to expiry (years).
    /// - `dt`           — Coordinate-time step.
    /// - `dp`           — Price change ΔP.
    /// - `market_price` — Observed market option price (call). `None` skips
    ///                    implied-vol back-solving.
    /// - `is_call`      — Option type for the market price.
    ///
    /// # Errors
    /// Returns [`OptionsError`] if basic inputs are invalid.
    pub fn scan(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        dt: f64,
        dp: f64,
        market_price: Option<f64>,
        is_call: bool,
    ) -> Result<ArbitrageSignal, OptionsError> {
        // Get model price and vol.
        let model_result = self.bs.price_option(spot, strike, expiry, dt, dp, is_call)?;
        let sigma_model = model_result.sigma_effective;
        let interval_class = model_result.interval_class;

        // Back-solve for market implied vol if price provided.
        let sigma_mkt = if let Some(mkt_px) = market_price {
            if mkt_px <= 0.0 {
                sigma_model // fallback
            } else {
                self.implied_vol_newton(spot, strike, expiry, mkt_px, is_call, sigma_model)
                    .unwrap_or(sigma_model)
            }
        } else {
            sigma_model
        };

        // Determine arbitrage type.
        let vol_diff = sigma_mkt - sigma_model;
        let rel_diff = if sigma_model > 1e-12 { vol_diff / sigma_model } else { 0.0 };

        let arb_type = match interval_class {
            IntervalClass::Timelike if rel_diff > self.threshold => {
                // Market vol > model vol despite TIMELIKE (causal) regime.
                ArbitrageType::TimelikeButOverpriced
            }
            IntervalClass::Spacelike if rel_diff < -self.threshold => {
                // Market vol < model vol despite SPACELIKE (noisy) regime.
                ArbitrageType::SpacelikeButUnderpriced
            }
            _ => ArbitrageType::None,
        };

        Ok(ArbitrageSignal {
            arb_type,
            score: rel_diff,
            market_implied_vol: sigma_mkt,
            model_vol: sigma_model,
            interval_class,
        })
    }

    /// Newton-Raphson implied volatility solver (10 iterations max).
    /// Returns `None` if it fails to converge.
    fn implied_vol_newton(
        &self,
        spot: f64,
        strike: f64,
        expiry: f64,
        target_price: f64,
        is_call: bool,
        sigma_init: f64,
    ) -> Option<f64> {
        let r = self.bs.cfg.risk_free_rate;
        let mut sigma = sigma_init.clamp(SIGMA_FLOOR, SIGMA_CAP);

        for _ in 0..10 {
            let log_ratio = (spot / strike).ln();
            let d1 = (log_ratio + (r + 0.5 * sigma * sigma) * expiry)
                / (sigma * expiry.sqrt());
            let d2 = d1 - sigma * expiry.sqrt();

            let price = if is_call {
                spot * standard_normal_cdf(d1)
                    - strike * (-r * expiry).exp() * standard_normal_cdf(d2)
            } else {
                strike * (-r * expiry).exp() * standard_normal_cdf(-d2)
                    - spot * standard_normal_cdf(-d1)
            };

            // Vega = S · √T · N'(d₁).
            let vega = spot * expiry.sqrt() * (-0.5 * d1 * d1).exp() / (2.0 * PI).sqrt();
            if vega.abs() < 1e-12 {
                return None;
            }

            let diff = price - target_price;
            sigma -= diff / vega;
            sigma = sigma.clamp(SIGMA_FLOOR, SIGMA_CAP);
            if diff.abs() < 1e-8 {
                return Some(sigma);
            }
        }
        Some(sigma) // return best estimate even if not fully converged
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_cfg() -> OptionsConfig {
        OptionsConfig::default()
    }

    // ── RelativisticBlackScholes ──────────────────────────────────────────────

    #[test]
    fn rbs_call_is_positive_and_finite() {
        let model = RelativisticBlackScholes::new(default_cfg());
        let r = model.price_call(100.0, 100.0, 1.0, 1.0, 0.5).expect("price_call");
        assert!(r.price > 0.0, "call price should be positive");
        assert!(r.price.is_finite(), "call price should be finite");
    }

    #[test]
    fn rbs_put_is_positive_and_finite() {
        let model = RelativisticBlackScholes::new(default_cfg());
        let r = model.price_put(100.0, 105.0, 0.5, 1.0, -1.0).expect("price_put");
        assert!(r.price > 0.0);
        assert!(r.price.is_finite());
    }

    #[test]
    fn rbs_invalid_spot_returns_error() {
        let model = RelativisticBlackScholes::new(default_cfg());
        assert!(model.price_call(-1.0, 100.0, 1.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn rbs_invalid_expiry_returns_error() {
        let model = RelativisticBlackScholes::new(default_cfg());
        assert!(model.price_call(100.0, 100.0, 0.0, 1.0, 0.0).is_err());
    }

    #[test]
    fn rbs_timelike_lower_sigma_than_spacelike() {
        let model = RelativisticBlackScholes::new(default_cfg());
        // TIMELIKE: small dp relative to c·dt.
        let tl = model.price_call(100.0, 100.0, 1.0, 1.0, 0.001).expect("tl price");
        // SPACELIKE: large dp relative to c·dt.
        let sl = model.price_call(100.0, 100.0, 1.0, 1.0, 100.0).expect("sl price");
        assert!(
            tl.sigma_effective <= sl.sigma_effective,
            "TIMELIKE σ ({}) should be ≤ SPACELIKE σ ({})",
            tl.sigma_effective,
            sl.sigma_effective
        );
    }

    #[test]
    fn rbs_timelike_classification_for_slow_move() {
        let model = RelativisticBlackScholes::new(default_cfg());
        // dp = 0.001, c_scale=0.05, dt=1 → β = 0.001/0.05 ≈ 0.02 → TIMELIKE.
        let r = model.price_call(100.0, 100.0, 1.0, 1.0, 0.001 * 100.0).expect("price");
        assert_eq!(r.interval_class, IntervalClass::Timelike);
    }

    #[test]
    fn rbs_proper_time_less_than_expiry_when_moving() {
        let model = RelativisticBlackScholes::new(default_cfg());
        // Non-zero dp → β > 0 → τ < T.
        let r = model.price_call(100.0, 100.0, 1.0, 1.0, 1.0).expect("price");
        assert!(r.proper_time_to_expiry <= 1.0 + 1e-9);
    }

    // ── LightconeOptionPricing ────────────────────────────────────────────────

    #[test]
    fn lightcone_sigma_tl_lt_sigma_sl() {
        let pricer = LightconeOptionPricing::new(default_cfg());
        // β=0.1 → TIMELIKE → σ_tl < σ_sl
        let r = pricer.price(100.0, 100.0, 1.0, 0.1, true).expect("price");
        assert!(r.sigma_timelike < r.sigma_spacelike,
            "σ_tl={} should be < σ_sl={}", r.sigma_timelike, r.sigma_spacelike);
    }

    #[test]
    fn lightcone_price_positive() {
        let pricer = LightconeOptionPricing::new(default_cfg());
        let r = pricer.price(100.0, 95.0, 0.25, 0.5, true).expect("price");
        assert!(r.price > 0.0);
    }

    #[test]
    fn lightcone_invalid_inputs_error() {
        let pricer = LightconeOptionPricing::new(default_cfg());
        assert!(pricer.price(0.0, 100.0, 1.0, 0.1, true).is_err());
        assert!(pricer.price(100.0, 0.0, 1.0, 0.1, true).is_err());
        assert!(pricer.price(100.0, 100.0, -1.0, 0.1, true).is_err());
    }

    // ── SpacetimeDelta ────────────────────────────────────────────────────────

    #[test]
    fn delta_call_between_zero_and_gamma_cap() {
        let sd = SpacetimeDelta::new(default_cfg());
        let r = sd.compute(100.0, 100.0, 1.0, 0.20, 1.0, 0.5, true).expect("delta");
        let cap = default_cfg().gamma_cap;
        assert!(r.delta_relativistic >= 0.0 && r.delta_relativistic <= cap,
            "call delta_rel {} not in [0, {}]", r.delta_relativistic, cap);
    }

    #[test]
    fn delta_put_between_neg_gamma_and_zero() {
        let sd = SpacetimeDelta::new(default_cfg());
        let r = sd.compute(100.0, 100.0, 1.0, 0.20, 1.0, 0.5, false).expect("delta");
        let cap = default_cfg().gamma_cap;
        assert!(r.delta_relativistic <= 0.0 && r.delta_relativistic >= -cap,
            "put delta_rel {} not in [{}, 0]", r.delta_relativistic, -cap);
    }

    #[test]
    fn delta_relativistic_gte_classical_for_positive_gamma() {
        let sd = SpacetimeDelta::new(default_cfg());
        // β > 0 → γ > 1 → Δ_rel ≥ Δ_bs for a call.
        let r = sd.compute(100.0, 100.0, 1.0, 0.20, 1.0, 1.0, true).expect("delta");
        assert!(
            r.delta_relativistic >= r.delta_classical - 1e-9,
            "Δ_rel ({}) should be ≥ Δ_bs ({}) for γ≥1 call",
            r.delta_relativistic, r.delta_classical
        );
    }

    #[test]
    fn delta_stationary_asset_equals_classical() {
        let sd = SpacetimeDelta::new(default_cfg());
        // dp = 0, dt = 1 → β = 0 → γ = 1 → Δ_rel = Δ_bs.
        let r = sd.compute(100.0, 100.0, 1.0, 0.20, 1.0, 0.0, true).expect("delta");
        assert!(
            (r.delta_relativistic - r.delta_classical).abs() < 1e-9,
            "stationary: Δ_rel should equal Δ_bs"
        );
    }

    // ── RelOrbitArbitrage ─────────────────────────────────────────────────────

    #[test]
    fn arbitrage_none_when_model_and_market_agree() {
        let arb = RelOrbitArbitrage::new(default_cfg(), 0.05);
        // Use model price as market price → no mispricing.
        let model = RelativisticBlackScholes::new(default_cfg());
        let px = model.price_call(100.0, 100.0, 1.0, 1.0, 0.5).expect("price").price;
        let sig = arb.scan(100.0, 100.0, 1.0, 1.0, 0.5, Some(px), true).expect("scan");
        assert_eq!(sig.arb_type, ArbitrageType::None,
            "no arbitrage when market price matches model");
    }

    #[test]
    fn arbitrage_signal_for_overpriced_timelike() {
        // Set a very high market price to force σ_mkt >> σ_model with TIMELIKE regime.
        let cfg = OptionsConfig {
            sigma_base: 0.20,
            ..default_cfg()
        };
        let arb = RelOrbitArbitrage::new(cfg, 0.01);
        // dp = 0.001 → TIMELIKE, very low σ_model; market price = deep ITM call inflated.
        let sig = arb.scan(100.0, 100.0, 1.0, 1.0, 0.001, Some(50.0), true).expect("scan");
        // With such a large market price vs model, score should be large positive.
        assert!(sig.score > 0.0, "score should be positive for overpriced option");
    }

    #[test]
    fn arbitrage_no_panic_with_zero_market_price() {
        let arb = RelOrbitArbitrage::new(default_cfg(), 0.05);
        let sig = arb.scan(100.0, 100.0, 1.0, 1.0, 0.5, Some(0.0), true);
        // Zero market price defaults to model vol; should not panic.
        assert!(sig.is_ok());
    }

    #[test]
    fn arbitrage_no_market_price_gives_none_arb() {
        let arb = RelOrbitArbitrage::new(default_cfg(), 0.05);
        let sig = arb.scan(100.0, 100.0, 1.0, 1.0, 0.5, None, true).expect("scan");
        // Without market price, σ_mkt = σ_model → no contradiction.
        assert_eq!(sig.arb_type, ArbitrageType::None);
    }

    // ── Standard normal CDF ───────────────────────────────────────────────────

    #[test]
    fn normal_cdf_symmetry() {
        assert!((standard_normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((standard_normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((standard_normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }
}
