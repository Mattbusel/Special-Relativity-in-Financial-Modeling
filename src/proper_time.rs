//! Proper-time portfolio correlation.
//!
//! ## Motivation
//!
//! Conventional (coordinate-time) correlation between two assets can be
//! spurious: if both assets happened to move during the same calendar
//! interval but for unrelated reasons, the correlation metric will register
//! a high value even though the assets share no causal driver.
//!
//! In special relativity, **proper time** is the Lorentz-invariant interval
//! accumulated along a worldline — it does not depend on the choice of
//! reference frame.  Applied to finance:
//!
//! ```text
//! Δτ = √( Δt² - (Δx/c)² )
//! ```
//!
//! where `Δt` is the calendar-time step and `Δx` is the log-price change.
//! Assets that traverse similar proper-time profiles share a structural
//! (invariant) path rather than a coincidental calendar-time overlap.
//!
//! ## Algorithm
//!
//! 1. Each asset accumulates its proper-time series `{τ_0, τ_1, …, τ_n}`
//!    tick-by-tick.
//! 2. The **proper-time profile** of an asset over `W` ticks is the vector
//!    of normalised proper-time increments: `Δτ_i / Σ Δτ_i`.
//! 3. Two assets are **properly correlated** if the cosine similarity of
//!    their normalised profiles exceeds a threshold.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::proper_time::{ProperTimeTracker, ProperTimeConfig};
//!
//! let cfg = ProperTimeConfig::default();
//! let mut btc = ProperTimeTracker::new("BTC".to_string(), cfg.clone());
//! let mut eth = ProperTimeTracker::new("ETH".to_string(), cfg);
//!
//! btc.update(1.0, 0.003);
//! eth.update(1.0, 0.003);
//!
//! let sim = btc.profile_similarity(&eth);
//! println!("proper-time cosine similarity: {sim:.4}");
//! ```

/// Configuration for proper-time computation.
#[derive(Debug, Clone)]
pub struct ProperTimeConfig {
    /// Speed-of-light scale `c` — maximum meaningful log-return per unit time.
    /// Price moves faster than this are treated as speed-of-light violations
    /// and clamped.
    pub c_scale: f64,

    /// Rolling window `W` for proper-time profile comparison.
    pub window: usize,
}

impl Default for ProperTimeConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,
            window: 50,
        }
    }
}

/// Summary of the proper-time trajectory for a single asset.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProperTimeSummary {
    /// Asset identifier.
    pub symbol: String,
    /// Total proper time accumulated since construction.
    pub total_proper_time: f64,
    /// Total coordinate time elapsed.
    pub total_coord_time: f64,
    /// Average Lorentz factor `γ = Δt / Δτ` over the history (> 1 when moving).
    pub mean_lorentz_factor: f64,
    /// Number of ticks processed.
    pub tick_count: usize,
}

/// Per-asset proper-time accumulator.
#[derive(Debug, Clone)]
pub struct ProperTimeTracker {
    symbol: String,
    cfg: ProperTimeConfig,
    total_proper_time: f64,
    total_coord_time: f64,
    lorentz_sum: f64,
    tick_count: usize,
    /// Rolling window of recent proper-time increments for profile comparison.
    increment_window: std::collections::VecDeque<f64>,
}

impl ProperTimeTracker {
    /// Create a new tracker for `symbol`.
    pub fn new(symbol: String, cfg: ProperTimeConfig) -> Self {
        Self {
            symbol,
            cfg,
            total_proper_time: 0.0,
            total_coord_time: 0.0,
            lorentz_sum: 0.0,
            tick_count: 0,
            increment_window: std::collections::VecDeque::new(),
        }
    }

    /// Ingest one tick: `dt` is the coordinate-time step (e.g. 1.0 per tick),
    /// `log_return` is `ln(price_t / price_{t-1})`.
    ///
    /// Returns the proper-time increment `Δτ` for this step.
    pub fn update(&mut self, dt: f64, log_return: f64) -> f64 {
        self.tick_count += 1;
        self.total_coord_time += dt;

        // Normalised velocity β = (Δx/Δt) / c, clamped strictly inside (-1, 1).
        let beta = (log_return / (dt * self.cfg.c_scale)).clamp(-0.9999, 0.9999);
        let gamma_inv = (1.0 - beta * beta).sqrt(); // 1/γ
        let dtau = dt * gamma_inv;

        self.total_proper_time += dtau;
        // γ = Δt / Δτ (time dilation factor).
        let gamma = if dtau > 1e-12 { dt / dtau } else { 1.0 };
        self.lorentz_sum += gamma;

        // Update rolling window.
        self.increment_window.push_back(dtau);
        if self.increment_window.len() > self.cfg.window {
            self.increment_window.pop_front();
        }

        dtau
    }

    /// Cosine similarity of the normalised proper-time increment profiles
    /// between `self` and `other`.  Returns `None` if either tracker has
    /// fewer than 2 ticks in its window.
    pub fn profile_similarity(&self, other: &ProperTimeTracker) -> Option<f64> {
        let a: Vec<f64> = self.normalised_profile();
        let b: Vec<f64> = other.normalised_profile();
        if a.len() < 2 || b.len() < 2 {
            return None;
        }
        let len = a.len().min(b.len());
        let dot: f64 = a.iter().zip(b.iter()).take(len).map(|(x, y)| x * y).sum();
        let mag_a: f64 = a.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
        let mag_b: f64 = b.iter().take(len).map(|x| x * x).sum::<f64>().sqrt();
        if mag_a < 1e-12 || mag_b < 1e-12 {
            return None;
        }
        Some((dot / (mag_a * mag_b)).clamp(-1.0, 1.0))
    }

    /// Normalised proper-time increment vector (window entries divided by their sum).
    pub fn normalised_profile(&self) -> Vec<f64> {
        let sum: f64 = self.increment_window.iter().sum();
        if sum < 1e-12 {
            return self.increment_window.iter().map(|_| 0.0).collect();
        }
        self.increment_window.iter().map(|&v| v / sum).collect()
    }

    /// Summary statistics for this asset's proper-time trajectory.
    pub fn summary(&self) -> ProperTimeSummary {
        let mean_lorentz = if self.tick_count > 0 {
            self.lorentz_sum / self.tick_count as f64
        } else {
            1.0
        };
        ProperTimeSummary {
            symbol: self.symbol.clone(),
            total_proper_time: self.total_proper_time,
            total_coord_time: self.total_coord_time,
            mean_lorentz_factor: mean_lorentz,
            tick_count: self.tick_count,
        }
    }

    /// Total accumulated proper time.
    pub fn total_proper_time(&self) -> f64 {
        self.total_proper_time
    }

    /// Symbol name.
    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

/// Compute pairwise proper-time similarity for a portfolio of assets.
///
/// Returns a vector of `(symbol_a, symbol_b, similarity)` triples for
/// all pairs where similarity is `Some`.
pub fn portfolio_proper_time_correlation(
    trackers: &[ProperTimeTracker],
) -> Vec<(String, String, f64)> {
    let mut result = Vec::new();
    for i in 0..trackers.len() {
        for j in (i + 1)..trackers.len() {
            if let Some(sim) = trackers[i].profile_similarity(&trackers[j]) {
                result.push((
                    trackers[i].symbol().to_string(),
                    trackers[j].symbol().to_string(),
                    sim,
                ));
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_paths_have_unit_similarity() {
        let cfg = ProperTimeConfig::default();
        let mut a = ProperTimeTracker::new("A".to_string(), cfg.clone());
        let mut b = ProperTimeTracker::new("B".to_string(), cfg);
        for _ in 0..20 {
            a.update(1.0, 0.002);
            b.update(1.0, 0.002);
        }
        let sim = a.profile_similarity(&b).expect("similarity should be Some");
        assert!((sim - 1.0).abs() < 1e-9, "identical paths should yield similarity 1.0, got {sim}");
    }

    #[test]
    fn proper_time_less_than_coord_time_when_moving() {
        let cfg = ProperTimeConfig::default();
        let mut tracker = ProperTimeTracker::new("X".to_string(), cfg);
        for _ in 0..10 {
            tracker.update(1.0, 0.01); // moving asset
        }
        let s = tracker.summary();
        assert!(
            s.total_proper_time < s.total_coord_time,
            "proper time ({}) must be < coord time ({}) for a moving asset",
            s.total_proper_time,
            s.total_coord_time
        );
    }

    #[test]
    fn stationary_asset_proper_time_equals_coord_time() {
        let cfg = ProperTimeConfig::default();
        let mut tracker = ProperTimeTracker::new("Y".to_string(), cfg);
        for _ in 0..10 {
            tracker.update(1.0, 0.0); // no price move
        }
        let s = tracker.summary();
        assert!(
            (s.total_proper_time - s.total_coord_time).abs() < 1e-10,
            "stationary asset proper time should equal coord time"
        );
    }

    #[test]
    fn portfolio_correlation_produces_all_pairs() {
        let cfg = ProperTimeConfig::default();
        let mut trackers: Vec<ProperTimeTracker> = ["A", "B", "C"]
            .iter()
            .map(|&s| ProperTimeTracker::new(s.to_string(), cfg.clone()))
            .collect();
        for t in &mut trackers {
            for _ in 0..10 {
                t.update(1.0, 0.001);
            }
        }
        let pairs = portfolio_proper_time_correlation(&trackers);
        assert_eq!(pairs.len(), 3, "expected 3 pairs for 3 assets");
    }

    #[test]
    fn similarity_none_before_two_ticks() {
        let cfg = ProperTimeConfig::default();
        let a = ProperTimeTracker::new("A".to_string(), cfg.clone());
        let b = ProperTimeTracker::new("B".to_string(), cfg);
        assert!(a.profile_similarity(&b).is_none());
    }
}
