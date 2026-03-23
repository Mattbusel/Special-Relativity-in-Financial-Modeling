//! Relativistic momentum factor for cross-sectional asset ranking.
//!
//! ## Motivation
//!
//! Classic momentum factors (e.g. 12-1 month returns) treat all price paths
//! as equivalent: a stock that doubles in one week scores the same as one
//! that doubles over twelve months.  Special relativity provides a natural
//! way to *weight* momentum by the Lorentz factor γ: a fast-moving worldline
//! accumulates *less* proper time than a slow one.  Assets with high
//! relativistic momentum are "hard to stop" — they are moving quickly in
//! Minkowski spacetime, compressing their proper-time clocks.
//!
//! ## Relativistic Momentum
//!
//! For a particle of (unit) rest mass:
//!
//! ```text
//! p_rel = γ · v / c      where γ = 1 / √(1 - β²),   β = v/c
//! ```
//!
//! Applied to finance, for each tick `i` of an asset:
//!
//! ```text
//! β_i   = Δx_i / (c · Δt_i)          (normalised velocity)
//! γ_i   = 1 / √(1 - β_i²)            (Lorentz factor, clamped to [1, γ_max])
//! p_i   = γ_i · β_i                   (relativistic momentum for this tick)
//! ```
//!
//! The **cumulative relativistic momentum** over a window is the sum of `p_i`.
//! This penalises assets with erratic, high-speed bursts (they spend less
//! proper time in each state) while rewarding those with sustained, moderate
//! trends that compound efficiently in proper-time space.
//!
//! ## Factor Construction
//!
//! Given a cross-section of `N` assets and their return series:
//!
//! 1. Compute the cumulative relativistic momentum `P` for each asset.
//! 2. Rank assets by `P` (descending = long, ascending = short).
//! 3. Form a zero-cost long-short factor: buy top quintile, sell bottom.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::relativistic_momentum_factor::{
//!     RelativisticMomentumEngine, MomentumConfig,
//! };
//!
//! let cfg = MomentumConfig::default();
//! let engine = RelativisticMomentumEngine::new(cfg);
//!
//! let returns = vec![0.01, -0.005, 0.02, 0.015, -0.003];
//! let score = engine.cumulative_momentum(&returns);
//! println!("Relativistic momentum score: {score:.4}");
//! ```

use serde::{Deserialize, Serialize};

// ─── CONFIGURATION ───────────────────────────────────────────────────────────

/// Configuration for the relativistic momentum engine.
#[derive(Debug, Clone)]
pub struct MomentumConfig {
    /// Speed-of-light scale `c` — maximum meaningful log-return per tick.
    pub c_scale: f64,
    /// Maximum Lorentz factor cap (prevents numerical explosion near β=1).
    pub gamma_max: f64,
    /// Look-back window length in ticks for cumulative momentum.
    pub window: usize,
    /// Quintile cutoff (0.2 = top/bottom 20%).
    pub quintile: f64,
    /// Minimum number of ticks required to compute a valid score.
    pub min_ticks: usize,
}

impl Default for MomentumConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,
            gamma_max: 50.0,
            window: 60,
            quintile: 0.20,
            min_ticks: 10,
        }
    }
}

// ─── PER-TICK RESULT ─────────────────────────────────────────────────────────

/// Relativistic quantities for a single tick.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TickMomentum {
    /// Normalised velocity `β = Δx / (c · Δt)` ∈ (-1, 1).
    pub beta: f64,
    /// Lorentz factor `γ = 1/√(1-β²)`, capped at `gamma_max`.
    pub gamma: f64,
    /// Relativistic momentum `p = γ · β`.
    pub momentum: f64,
    /// Proper-time increment `Δτ = Δt · √(1-β²)`.
    pub proper_time_increment: f64,
}

// ─── ASSET SCORE ─────────────────────────────────────────────────────────────

/// The full momentum summary for one asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetMomentumScore {
    /// Asset symbol / label.
    pub symbol: String,
    /// Cumulative relativistic momentum `Σ p_i`.
    pub cumulative_momentum: f64,
    /// Average Lorentz factor over the window.
    pub mean_gamma: f64,
    /// Fraction of time the asset was moving "faster than c/2" (`|β| > 0.5`).
    pub high_speed_fraction: f64,
    /// Total proper time accumulated over the window (less = faster-moving asset).
    pub total_proper_time: f64,
    /// Total coordinate time (number of ticks used).
    pub total_coord_time: f64,
    /// Per-tick breakdown (optional, populated when `verbose=true`).
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub ticks: Vec<TickMomentum>,
    /// Percentile rank (0-100) within the cross-section (set by ranker).
    pub percentile_rank: f64,
}

// ─── FACTOR PORTFOLIO ────────────────────────────────────────────────────────

/// A zero-cost long-short factor portfolio based on relativistic momentum.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MomentumFactorPortfolio {
    /// Assets in the long leg (top quintile by momentum).
    pub long_leg: Vec<AssetMomentumScore>,
    /// Assets in the short leg (bottom quintile by momentum).
    pub short_leg: Vec<AssetMomentumScore>,
    /// Average momentum of the long leg.
    pub long_avg_momentum: f64,
    /// Average momentum of the short leg.
    pub short_avg_momentum: f64,
    /// Factor spread: `long_avg - short_avg`.
    pub factor_spread: f64,
    /// Number of assets in each leg.
    pub leg_size: usize,
}

impl MomentumFactorPortfolio {
    pub fn summary_line(&self) -> String {
        format!(
            "Rel-momentum factor: long={} assets (avg p={:.4}), \
             short={} assets (avg p={:.4}), spread={:.4}",
            self.long_leg.len(),
            self.long_avg_momentum,
            self.short_leg.len(),
            self.short_avg_momentum,
            self.factor_spread,
        )
    }
}

// ─── ENGINE ──────────────────────────────────────────────────────────────────

/// The relativistic momentum computation engine.
pub struct RelativisticMomentumEngine {
    cfg: MomentumConfig,
}

impl RelativisticMomentumEngine {
    pub fn new(cfg: MomentumConfig) -> Self {
        Self { cfg }
    }

    // ── Per-tick computation ──────────────────────────────────────────────────

    /// Compute relativistic quantities for a single tick with
    /// `log_return = Δx` and `dt` coordinate-time step (default 1.0).
    pub fn tick_momentum(&self, log_return: f64, dt: f64) -> TickMomentum {
        let c = self.cfg.c_scale;
        // Normalise velocity, clamped strictly inside (-1, 1).
        let beta_raw = log_return / (c * dt.max(1e-12));
        let beta = beta_raw.clamp(-0.9999, 0.9999);

        let one_minus_beta_sq = (1.0 - beta * beta).max(1e-12);
        let gamma_raw = 1.0 / one_minus_beta_sq.sqrt();
        let gamma = gamma_raw.min(self.cfg.gamma_max);

        let momentum = gamma * beta;
        let proper_time_increment = dt * one_minus_beta_sq.sqrt();

        TickMomentum { beta, gamma, momentum, proper_time_increment }
    }

    // ── Asset-level computation ───────────────────────────────────────────────

    /// Compute cumulative relativistic momentum for a return series.
    ///
    /// Uses the trailing `window` observations. Returns 0.0 when the series
    /// is shorter than `min_ticks`.
    pub fn cumulative_momentum(&self, log_returns: &[f64]) -> f64 {
        if log_returns.len() < self.cfg.min_ticks {
            return 0.0;
        }
        let window = self.cfg.window.min(log_returns.len());
        let slice = &log_returns[log_returns.len() - window..];
        slice.iter().map(|&r| self.tick_momentum(r, 1.0).momentum).sum()
    }

    /// Full per-asset score with optional verbose tick breakdown.
    pub fn score_asset(
        &self,
        symbol: impl Into<String>,
        log_returns: &[f64],
        verbose: bool,
    ) -> AssetMomentumScore {
        let sym = symbol.into();
        if log_returns.len() < self.cfg.min_ticks {
            return AssetMomentumScore {
                symbol: sym,
                cumulative_momentum: 0.0,
                mean_gamma: 1.0,
                high_speed_fraction: 0.0,
                total_proper_time: 0.0,
                total_coord_time: 0.0,
                ticks: Vec::new(),
                percentile_rank: 50.0,
            };
        }

        let window = self.cfg.window.min(log_returns.len());
        let slice = &log_returns[log_returns.len() - window..];

        let mut cumulative_momentum = 0.0_f64;
        let mut gamma_sum = 0.0_f64;
        let mut high_speed_count = 0usize;
        let mut total_proper_time = 0.0_f64;
        let mut ticks_out: Vec<TickMomentum> = Vec::new();

        for &ret in slice {
            let tm = self.tick_momentum(ret, 1.0);
            cumulative_momentum += tm.momentum;
            gamma_sum += tm.gamma;
            if tm.beta.abs() > 0.5 {
                high_speed_count += 1;
            }
            total_proper_time += tm.proper_time_increment;
            if verbose {
                ticks_out.push(tm);
            }
        }

        let n = slice.len() as f64;
        AssetMomentumScore {
            symbol: sym,
            cumulative_momentum,
            mean_gamma: gamma_sum / n,
            high_speed_fraction: high_speed_count as f64 / n,
            total_proper_time,
            total_coord_time: n,
            ticks: ticks_out,
            percentile_rank: 50.0, // filled in by cross-section ranker
        }
    }

    // ── Cross-section ranking ─────────────────────────────────────────────────

    /// Rank a set of assets by cumulative relativistic momentum and assign
    /// percentile ranks (0 = lowest momentum, 100 = highest).
    pub fn rank_cross_section(&self, scores: &mut Vec<AssetMomentumScore>) {
        let n = scores.len();
        if n == 0 {
            return;
        }
        // Sort by cumulative_momentum descending to assign percentile ranks.
        let mut indexed: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, s)| (i, s.cumulative_momentum))
            .collect();
        indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (rank, &(idx, _)) in indexed.iter().enumerate() {
            scores[idx].percentile_rank = rank as f64 / (n - 1).max(1) as f64 * 100.0;
        }
    }

    /// Build a long-short factor portfolio from a cross-section of return series.
    ///
    /// `assets` is a slice of `(symbol, log_returns)` pairs.
    pub fn build_factor_portfolio(
        &self,
        assets: &[(&str, &[f64])],
    ) -> MomentumFactorPortfolio {
        let mut scores: Vec<AssetMomentumScore> = assets
            .iter()
            .map(|&(sym, rets)| self.score_asset(sym, rets, false))
            .collect();

        self.rank_cross_section(&mut scores);

        // Sort descending by momentum for leg assignment.
        scores.sort_by(|a, b| {
            b.cumulative_momentum
                .partial_cmp(&a.cumulative_momentum)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let n = scores.len();
        let leg_size = ((n as f64 * self.cfg.quintile).ceil() as usize).max(1);

        let long_leg: Vec<AssetMomentumScore> = scores[..leg_size.min(n)].to_vec();
        let short_leg: Vec<AssetMomentumScore> =
            scores[n.saturating_sub(leg_size)..].to_vec();

        let long_avg = if long_leg.is_empty() {
            0.0
        } else {
            long_leg.iter().map(|s| s.cumulative_momentum).sum::<f64>() / long_leg.len() as f64
        };
        let short_avg = if short_leg.is_empty() {
            0.0
        } else {
            short_leg.iter().map(|s| s.cumulative_momentum).sum::<f64>()
                / short_leg.len() as f64
        };

        MomentumFactorPortfolio {
            long_leg,
            short_leg,
            long_avg_momentum: long_avg,
            short_avg_momentum: short_avg,
            factor_spread: long_avg - short_avg,
            leg_size,
        }
    }

    // ── Rolling factor time-series ────────────────────────────────────────────

    /// Compute the factor spread at each rolling window position across a
    /// set of return series.  Returns a `Vec<f64>` of factor spread values.
    ///
    /// `asset_returns` is a slice of (symbol, full_return_series) pairs.
    /// The factor is re-computed at each step from `t = window` onwards.
    pub fn rolling_factor_spread(
        &self,
        asset_returns: &[(&str, Vec<f64>)],
        step: usize,
    ) -> Vec<f64> {
        let window = self.cfg.window;
        let max_len = asset_returns
            .iter()
            .map(|(_, r)| r.len())
            .max()
            .unwrap_or(0);

        if max_len < window {
            return Vec::new();
        }

        let mut spreads = Vec::new();
        let step = step.max(1);
        let mut t = window;
        while t <= max_len {
            let snapshot: Vec<(&str, &[f64])> = asset_returns
                .iter()
                .filter_map(|(sym, rets)| {
                    if rets.len() >= t {
                        Some((*sym, &rets[t - window..t]))
                    } else {
                        None
                    }
                })
                .collect();

            if snapshot.len() >= 2 {
                let portfolio = self.build_factor_portfolio(&snapshot);
                spreads.push(portfolio.factor_spread);
            }
            t += step;
        }
        spreads
    }
}

// ─── TESTS ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> RelativisticMomentumEngine {
        RelativisticMomentumEngine::new(MomentumConfig::default())
    }

    #[test]
    fn beta_clamped_in_unit_interval() {
        let e = engine();
        // A huge return should still clamp β below 1.
        let tm = e.tick_momentum(10.0, 1.0);
        assert!(tm.beta.abs() < 1.0, "β must be < 1");
    }

    #[test]
    fn gamma_always_at_least_one() {
        let e = engine();
        for ret in [-0.05, 0.0, 0.01, 0.05] {
            let tm = e.tick_momentum(ret, 1.0);
            assert!(tm.gamma >= 1.0, "γ must be >= 1.0, got {}", tm.gamma);
        }
    }

    #[test]
    fn zero_return_zero_momentum() {
        let e = engine();
        let tm = e.tick_momentum(0.0, 1.0);
        assert!(tm.momentum.abs() < 1e-12);
        assert!((tm.gamma - 1.0).abs() < 1e-12);
    }

    #[test]
    fn positive_trend_positive_momentum() {
        let e = engine();
        let returns: Vec<f64> = vec![0.01; 60];
        let score = e.score_asset("AAPL", &returns, false);
        assert!(score.cumulative_momentum > 0.0);
    }

    #[test]
    fn negative_trend_negative_momentum() {
        let e = engine();
        let returns: Vec<f64> = vec![-0.01; 60];
        let score = e.score_asset("BRKB", &returns, false);
        assert!(score.cumulative_momentum < 0.0);
    }

    #[test]
    fn rank_assigns_percentile_100_to_highest() {
        let e = engine();
        let mut scores = vec![
            e.score_asset("A", &vec![0.02; 60], false),
            e.score_asset("B", &vec![-0.01; 60], false),
            e.score_asset("C", &vec![0.005; 60], false),
        ];
        e.rank_cross_section(&mut scores);
        let max_pct = scores.iter().map(|s| s.percentile_rank as u64).max().unwrap();
        assert_eq!(max_pct, 100);
    }

    #[test]
    fn factor_portfolio_has_two_legs() {
        let e = engine();
        let assets: Vec<(&str, Vec<f64>)> = vec![
            ("A", vec![0.02; 60]),
            ("B", vec![-0.02; 60]),
            ("C", vec![0.01; 60]),
            ("D", vec![-0.01; 60]),
            ("E", vec![0.005; 60]),
        ];
        let asset_refs: Vec<(&str, &[f64])> =
            assets.iter().map(|(s, r)| (*s, r.as_slice())).collect();
        let portfolio = e.build_factor_portfolio(&asset_refs);
        assert!(!portfolio.long_leg.is_empty());
        assert!(!portfolio.short_leg.is_empty());
        assert!(portfolio.factor_spread.is_finite());
    }

    #[test]
    fn proper_time_less_than_coord_time_when_moving() {
        let e = engine();
        let returns: Vec<f64> = vec![0.02; 60]; // consistently moving
        let score = e.score_asset("X", &returns, false);
        assert!(
            score.total_proper_time < score.total_coord_time,
            "proper time < coord time for moving asset"
        );
    }

    #[test]
    fn rolling_factor_spread_length_sensible() {
        let e = engine();
        let asset_returns: Vec<(&str, Vec<f64>)> = vec![
            ("A", vec![0.01; 120]),
            ("B", vec![-0.01; 120]),
        ];
        let spreads = e.rolling_factor_spread(&asset_returns, 10);
        assert!(!spreads.is_empty());
        for s in &spreads {
            assert!(s.is_finite());
        }
    }
}
