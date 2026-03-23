//! Relativistic Doppler effect for price signal propagation.
//!
//! ## Motivation
//!
//! In special relativity, when a source emits signals at frequency f_e and an
//! observer moves relative to the source, the observed frequency is shifted:
//!
//! - **Moving toward** the source (β > 0): blueshift — higher observed frequency.
//! - **Moving away** from the source (β < 0 or treated as negative): redshift —
//!   lower observed frequency.
//!
//! Applied to markets: a market participant's "trade rate" determines their
//! velocity through market spacetime.  A high-frequency trader moving *toward*
//! the information source observes signals at a higher rate (blueshift).  A
//! slow participant moving away sees fewer signals per unit time (redshift).
//!
//! ## Formulas
//!
//! Moving toward source (β > 0):
//! ```text
//! f_obs = f_e · √((1 + β) / (1 − β))
//! ```
//!
//! Moving away from source (β < 0):
//! ```text
//! f_obs = f_e · √((1 − |β|) / (1 + |β|))
//! ```

use serde::{Deserialize, Serialize};

// ─── Core function ────────────────────────────────────────────────────────────

/// Compute the relativistic Doppler-shifted observed frequency.
///
/// - `emitted_freq`: frequency emitted by the source (Hz or any consistent unit).
/// - `beta`: signed velocity fraction (positive = moving toward source, negative = away).
///
/// Clamps |β| to [0, 0.9999] to avoid division by zero.
pub fn relativistic_doppler(emitted_freq: f64, beta: f64) -> f64 {
    let beta_clamped = beta.clamp(-0.9999, 0.9999);
    if beta_clamped >= 0.0 {
        // Moving toward source: blueshift
        emitted_freq * ((1.0 + beta_clamped) / (1.0 - beta_clamped)).sqrt()
    } else {
        // Moving away: redshift
        let b = beta_clamped.abs();
        emitted_freq * ((1.0 - b) / (1.0 + b)).sqrt()
    }
}

// ─── Observer ─────────────────────────────────────────────────────────────────

/// A single market observer characterised by their velocity toward/away from
/// the information source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopplerObserver {
    /// Velocity fraction β (positive = toward source, negative = away).
    pub beta: f64,
    /// Observed frequency given the current β.
    pub observed_frequency: f64,
}

impl DopplerObserver {
    /// Construct a new observer at velocity `beta` observing a source at `emitted_freq`.
    pub fn new(beta: f64, emitted_freq: f64) -> Self {
        let observed_frequency = relativistic_doppler(emitted_freq, beta);
        Self { beta, observed_frequency }
    }
}

// ─── Doppler Shift ────────────────────────────────────────────────────────────

/// Full characterisation of a Doppler shift event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DopplerShift {
    /// Frequency emitted at the source.
    pub emitted_freq: f64,
    /// Frequency observed by the moving observer.
    pub observed_freq: f64,
    /// Percentage shift: (observed - emitted) / emitted * 100.
    pub shift_pct: f64,
    /// True if observed < emitted (redshift; observer moving away).
    pub redshifted: bool,
}

impl DopplerShift {
    /// Compute the Doppler shift for a given emitted frequency and velocity fraction.
    pub fn compute(emitted: f64, beta: f64) -> Self {
        let observed_freq = relativistic_doppler(emitted, beta);
        let shift_pct = (observed_freq - emitted) / emitted * 100.0;
        let redshifted = observed_freq < emitted;
        Self {
            emitted_freq: emitted,
            observed_freq,
            shift_pct,
            redshifted,
        }
    }
}

// ─── Price Signal Doppler ─────────────────────────────────────────────────────

/// Models price signal frequency as observed by different market participants.
pub struct PriceSignalDoppler;

impl PriceSignalDoppler {
    /// Compute the signal frequency observed by a participant with the given
    /// trade rate.
    ///
    /// - `base_freq_hz`: the frequency at which price signals are emitted (e.g. 1 Hz = 1 signal/s).
    /// - `observer_trade_rate`: this participant's trades per unit time.
    /// - `max_trade_rate`: the theoretical maximum (speed-of-light equivalent).
    ///
    /// Returns the observed frequency.
    pub fn signal_frequency_for_timeframe(
        base_freq_hz: f64,
        observer_trade_rate: f64,
        max_trade_rate: f64,
    ) -> f64 {
        let beta = (observer_trade_rate / max_trade_rate).min(0.99);
        relativistic_doppler(base_freq_hz, beta)
    }
}

// ─── Multi-observer analysis ──────────────────────────────────────────────────

/// Compute Doppler shifts for multiple market participants simultaneously.
///
/// Each element of `observer_betas` is a signed β for one participant.
pub fn multi_observer_analysis(signal_freq: f64, observer_betas: &[f64]) -> Vec<DopplerShift> {
    observer_betas
        .iter()
        .map(|&beta| DopplerShift::compute(signal_freq, beta))
        .collect()
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    #[test]
    fn beta_zero_no_shift() {
        let f_obs = relativistic_doppler(1.0, 0.0);
        assert!((f_obs - 1.0).abs() < EPS, "Expected no shift at β=0, got {f_obs}");
    }

    #[test]
    fn moving_toward_causes_blueshift() {
        let f_e = 1.0;
        let f_obs = relativistic_doppler(f_e, 0.5);
        assert!(f_obs > f_e, "Should blueshift when moving toward source");
    }

    #[test]
    fn moving_away_causes_redshift() {
        let f_e = 1.0;
        let f_obs = relativistic_doppler(f_e, -0.5);
        assert!(f_obs < f_e, "Should redshift when moving away from source");
    }

    #[test]
    fn frequency_ratio_correct() {
        // For β=0.5 toward: ratio = √(1.5/0.5) = √3 ≈ 1.7320508...
        let ratio = relativistic_doppler(1.0, 0.5);
        let expected = ((1.0 + 0.5_f64) / (1.0 - 0.5_f64)).sqrt();
        assert!((ratio - expected).abs() < EPS);
    }

    #[test]
    fn frequency_ratio_away_correct() {
        // For β=-0.6 (away): ratio = √(0.4/1.6) = √0.25 = 0.5
        let ratio = relativistic_doppler(2.0, -0.6);
        let expected = 2.0 * ((0.4_f64) / (1.6_f64)).sqrt();
        assert!((ratio - expected).abs() < 1e-10);
    }

    #[test]
    fn doppler_shift_compute_redshifted_flag() {
        let shift = DopplerShift::compute(10.0, -0.5);
        assert!(shift.redshifted);
        assert!(shift.shift_pct < 0.0);

        let shift_blue = DopplerShift::compute(10.0, 0.5);
        assert!(!shift_blue.redshifted);
        assert!(shift_blue.shift_pct > 0.0);
    }

    #[test]
    fn multi_observer_analysis_count() {
        let betas = vec![-0.9, -0.5, 0.0, 0.5, 0.9];
        let shifts = multi_observer_analysis(1.0, &betas);
        assert_eq!(shifts.len(), 5);
        // Verify ordering: first should be most redshifted, last most blueshifted.
        assert!(shifts[0].observed_freq < shifts[4].observed_freq);
    }

    #[test]
    fn price_signal_doppler_increases_with_trade_rate() {
        let base = 1.0;
        let max_rate = 10_000.0;
        let f_slow = PriceSignalDoppler::signal_frequency_for_timeframe(base, 100.0, max_rate);
        let f_fast = PriceSignalDoppler::signal_frequency_for_timeframe(base, 9_000.0, max_rate);
        assert!(f_fast > f_slow, "Faster trader should observe higher signal frequency");
    }

    #[test]
    fn doppler_observer_construction() {
        let obs = DopplerObserver::new(0.6, 5.0);
        let expected = relativistic_doppler(5.0, 0.6);
        assert!((obs.observed_frequency - expected).abs() < EPS);
    }

    #[test]
    fn beta_clamped_for_extreme_values() {
        // Should not panic or return infinity.
        let f = relativistic_doppler(1.0, 1.5);
        assert!(f.is_finite());
        let f2 = relativistic_doppler(1.0, -1.5);
        assert!(f2.is_finite());
    }
}
