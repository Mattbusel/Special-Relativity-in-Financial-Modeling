//! Twin paradox applied to competing trading strategies.
//!
//! ## Motivation
//!
//! The twin paradox from special relativity states that a twin who accelerates
//! through high-velocity travel ages less (experiences less proper time) than
//! their stationary sibling.  Applied to trading:
//!
//! - A **high-frequency trader (HFT)** executes many trades per day, operating
//!   close to the "speed of light" of market information velocity.  Their
//!   subjective experience of time is compressed — they process events quickly
//!   but experience less "narrative time" per coordinate day.
//! - A **long-term investor** moves slowly through market spacetime and
//!   accumulates proper time at nearly the full coordinate rate.
//!
//! ## Proper Time
//!
//! For a strategy with velocity fraction β = v/c (trades relative to max
//! possible trade rate):
//!
//! ```text
//! τ = t · √(1 − β²)
//! ```
//!
//! The HFT's subjective time is compressed; the investor's is not.

use serde::{Deserialize, Serialize};

// ─── Strategy Frame ───────────────────────────────────────────────────────────

/// Represents a trading strategy's "reference frame" in market spacetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyFrame {
    /// Human-readable name, e.g. "HFT" or "Long-Term Investor".
    pub name: String,
    /// Number of trades executed per trading day.
    pub trades_per_day: f64,
    /// β = v/c — velocity fraction derived from trade rate vs max trade rate.
    /// Clamped to [0.0, 0.9999].
    pub velocity_fraction: f64,
    /// Proper time accumulated so far (days).
    pub proper_time: f64,
}

impl StrategyFrame {
    /// Construct a frame, computing β from the trade rate.
    ///
    /// `max_trade_rate` is the theoretical maximum trades per day (the "speed
    /// of light" of the market).
    pub fn new(name: impl Into<String>, trades_per_day: f64, max_trade_rate: f64) -> Self {
        let beta = (trades_per_day / max_trade_rate).clamp(0.0, 0.9999);
        Self {
            name: name.into(),
            trades_per_day,
            velocity_fraction: beta,
            proper_time: 0.0,
        }
    }
}

// ─── Core functions ───────────────────────────────────────────────────────────

/// Lorentz factor γ = 1/√(1−β²).  Returns 1.0 for β=0, grows toward ∞ as β→1.
pub fn time_dilation_factor(beta: f64) -> f64 {
    let beta = beta.clamp(0.0, 0.9999);
    1.0 / (1.0 - beta * beta).sqrt()
}

/// Compute proper time for both strategies over `coordinate_days`.
///
/// Returns `(hft_proper_days, investor_proper_days)`.
pub fn twin_paradox_aging(
    hft: &StrategyFrame,
    investor: &StrategyFrame,
    coordinate_days: u64,
) -> (f64, f64) {
    let t = coordinate_days as f64;
    let hft_proper = t * (1.0 - hft.velocity_fraction * hft.velocity_fraction).sqrt();
    let investor_proper = t * (1.0 - investor.velocity_fraction * investor.velocity_fraction).sqrt();
    (hft_proper, investor_proper)
}

/// Absolute difference in proper time experienced (|investor_proper - hft_proper|).
pub fn aging_difference(hft_proper: f64, investor_proper: f64) -> f64 {
    (investor_proper - hft_proper).abs()
}

// ─── Twin Trajectory ─────────────────────────────────────────────────────────

/// A time-series of (coordinate_day, proper_time) pairs for a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwinTrajectory {
    /// Sequence of (coordinate_time, proper_time) snapshots.
    pub frames: Vec<(u64, f64)>,
    /// The strategy frame whose trajectory this represents.
    pub strategy: StrategyFrame,
}

impl TwinTrajectory {
    /// Compute the full trajectory for a strategy over `days` coordinate days.
    /// Snapshots are recorded daily.
    pub fn compute(strategy: &StrategyFrame, days: u64) -> Self {
        let beta = strategy.velocity_fraction;
        let tau_rate = (1.0 - beta * beta).sqrt(); // proper time per coordinate day
        let frames: Vec<(u64, f64)> = (0..=days)
            .map(|day| (day, day as f64 * tau_rate))
            .collect();
        Self {
            frames,
            strategy: strategy.clone(),
        }
    }

    /// Proper time accumulated at the end of the trajectory.
    pub fn final_proper_time(&self) -> f64 {
        self.frames.last().map(|(_, tau)| *tau).unwrap_or(0.0)
    }
}

// ─── Paradox Report ───────────────────────────────────────────────────────────

/// Summary of the twin paradox comparison between two strategies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParadoxReport {
    /// Proper time experienced by the HFT strategy (days).
    pub hft_proper_days: f64,
    /// Proper time experienced by the long-term investor (days).
    pub investor_proper_days: f64,
    /// Absolute difference in proper time experienced by the two strategies.
    pub age_difference_days: f64,
    /// Human-readable explanation of the result.
    pub interpretation: String,
}

/// Compute a full paradox report for the two strategies over `days`.
pub fn paradox_resolution(
    hft: &StrategyFrame,
    investor: &StrategyFrame,
    days: u64,
) -> ParadoxReport {
    let (hft_proper, investor_proper) = twin_paradox_aging(hft, investor, days);
    let diff = aging_difference(hft_proper, investor_proper);
    let gamma_hft = time_dilation_factor(hft.velocity_fraction);

    let interpretation = format!(
        "Over {} coordinate trading days, '{}' (β={:.4}) experiences {:.2} proper days \
         while '{}' (β={:.4}) experiences {:.2} proper days. \
         The HFT ages {:.2} days less (γ={:.4}x time dilation). \
         The long-term investor accumulates more subjective experience per calendar day.",
        days,
        hft.name,
        hft.velocity_fraction,
        hft_proper,
        investor.name,
        investor.velocity_fraction,
        investor_proper,
        diff,
        gamma_hft,
    );

    ParadoxReport {
        hft_proper_days: hft_proper,
        investor_proper_days: investor_proper,
        age_difference_days: diff,
        interpretation,
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn beta_zero_proper_time_equals_coordinate_time() {
        let hft = StrategyFrame::new("slow", 0.0, 1_000_000.0); // β = 0
        let investor = StrategyFrame::new("slow2", 0.0, 1_000_000.0);
        let (h, i) = twin_paradox_aging(&hft, &investor, 365);
        assert!((h - 365.0).abs() < EPS);
        assert!((i - 365.0).abs() < EPS);
    }

    #[test]
    fn high_beta_strong_time_dilation() {
        // β = 0.99 → proper time ≈ 14% of coordinate time
        let hft = StrategyFrame {
            name: "HFT".to_string(),
            trades_per_day: 990.0,
            velocity_fraction: 0.99,
            proper_time: 0.0,
        };
        let investor = StrategyFrame {
            name: "Investor".to_string(),
            trades_per_day: 1.0,
            velocity_fraction: 0.001,
            proper_time: 0.0,
        };
        let (hft_proper, investor_proper) = twin_paradox_aging(&hft, &investor, 100);
        assert!(hft_proper < investor_proper, "HFT should age slower");
        // γ for β=0.99 ≈ 7.09 → proper_time ≈ 100 / 7.09 ≈ 14.1
        let expected_hft = 100.0 * (1.0_f64 - 0.99_f64 * 0.99_f64).sqrt();
        assert!((hft_proper - expected_hft).abs() < 1e-10);
    }

    #[test]
    fn time_dilation_factor_beta_zero() {
        assert!((time_dilation_factor(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn time_dilation_factor_increases_with_beta() {
        let g1 = time_dilation_factor(0.5);
        let g2 = time_dilation_factor(0.9);
        assert!(g2 > g1);
    }

    #[test]
    fn trajectory_compute_length() {
        let frame = StrategyFrame::new("Test", 100.0, 10_000.0);
        let traj = TwinTrajectory::compute(&frame, 10);
        assert_eq!(traj.frames.len(), 11); // days 0..=10
        assert_eq!(traj.frames[0].0, 0);
        assert!((traj.frames[0].1 - 0.0).abs() < EPS);
        assert_eq!(traj.frames[10].0, 10);
    }

    #[test]
    fn trajectory_final_matches_formula() {
        let frame = StrategyFrame {
            name: "HFT".to_string(),
            trades_per_day: 800.0,
            velocity_fraction: 0.8,
            proper_time: 0.0,
        };
        let traj = TwinTrajectory::compute(&frame, 100);
        let expected = 100.0 * (1.0 - 0.64_f64).sqrt(); // √(1-0.8²) = 0.6
        assert!((traj.final_proper_time() - expected).abs() < EPS);
    }

    #[test]
    fn aging_difference_is_positive() {
        let hft = StrategyFrame::new("HFT", 9_000.0, 10_000.0);
        let investor = StrategyFrame::new("Investor", 1.0, 10_000.0);
        let (h, i) = twin_paradox_aging(&hft, &investor, 252);
        let diff = aging_difference(h, i);
        assert!(diff > 0.0);
    }

    #[test]
    fn paradox_report_interpretation_not_empty() {
        let hft = StrategyFrame::new("HFT", 5_000.0, 10_000.0);
        let investor = StrategyFrame::new("Investor", 5.0, 10_000.0);
        let report = paradox_resolution(&hft, &investor, 252);
        assert!(!report.interpretation.is_empty());
        assert!(report.hft_proper_days < report.investor_proper_days);
    }

    #[test]
    fn frame_new_clamps_beta() {
        let frame = StrategyFrame::new("extreme", 1_000_000.0, 1.0);
        assert!(frame.velocity_fraction <= 0.9999);
    }
}
