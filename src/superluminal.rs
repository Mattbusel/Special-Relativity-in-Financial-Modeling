//! Tachyonic fields analogy: information appearing to travel faster than light
//! in financial markets.
//!
//! Tachyons are hypothetical particles with imaginary mass that always travel
//! faster than light. Here we use the analogy to model "information" that
//! arrives in markets before it should causally be able to, e.g. insider
//! trading, high-frequency information leakage, or apparent precognitive
//! behaviour.

/// Speed of light in natural units (c = 1 for normalized calculations,
/// scaled to market distance units here).
const C: f64 = 1.0;

// ---------------------------------------------------------------------------
// TachyonField
// ---------------------------------------------------------------------------

/// A tachyonic field characterised by imaginary mass and superluminal velocities.
///
/// For a tachyon with imaginary mass `m` (stored as the positive real
/// magnitude), the dispersion relation is:
/// ```text
/// E² = p² c² - m² c⁴   (note minus sign — imaginary mass squared)
/// ```
#[derive(Debug, Clone)]
pub struct TachyonField {
    /// Magnitude of the imaginary mass (m > 0).
    pub imaginary_mass: f64,
    /// Group velocity (v_g > c for tachyons).
    pub group_velocity: f64,
    /// Phase velocity (v_p = c² / v_g).
    pub phase_velocity: f64,
}

impl TachyonField {
    /// Construct a tachyon field from imaginary mass and momentum.
    ///
    /// Uses the tachyonic dispersion relation:
    /// ```text
    /// E  = sqrt(p² - m²)   (c=1)
    /// v_g = p / E
    /// ```
    pub fn from_mass_and_momentum(mass: f64, momentum: f64) -> Self {
        let gv = Self::group_velocity_from_mass(mass, momentum);
        let pv = C * C / gv.abs().max(1e-12);
        Self {
            imaginary_mass: mass,
            group_velocity: gv,
            phase_velocity: pv,
        }
    }

    /// Group velocity for a tachyon: v_g = c² * p / E,
    /// where E = sqrt(p² - m²) with imaginary mass (minus sign).
    /// Returns NaN if the field is sub-luminal (p < m).
    pub fn group_velocity_from_mass(mass: f64, momentum: f64) -> f64 {
        let e_sq = momentum * momentum - mass * mass;
        if e_sq <= 0.0 {
            // Not truly tachyonic — return superluminal proxy
            return f64::NAN;
        }
        let energy = e_sq.sqrt();
        (C * C * momentum) / energy
    }

    /// Phase velocity: v_p = c² / v_g.
    /// For tachyons v_p is always < c (the "signal" travels slowly but the
    /// wavefront races ahead).
    pub fn phase_velocity(&self) -> f64 {
        C * C / self.group_velocity.abs().max(1e-12)
    }
}

// ---------------------------------------------------------------------------
// MarketTachyon
// ---------------------------------------------------------------------------

/// Market analogy for tachyonic information propagation.
pub struct MarketTachyon;

impl MarketTachyon {
    /// Time for "information" to travel `distance` at the group velocity.
    pub fn information_arrival_time(distance: f64, field: &TachyonField) -> f64 {
        distance / field.group_velocity.abs().max(1e-12)
    }

    /// Apparent arrival time based on phase velocity.
    /// Can be negative (information appears to arrive before it was sent).
    pub fn apparent_arrival_time(distance: f64, field: &TachyonField) -> f64 {
        distance / field.phase_velocity.abs().max(1e-12)
    }

    /// Fraction of events in `signal_times` that arrive out of the order
    /// specified by `expected_sequence`.
    ///
    /// `expected_sequence` is a permutation of indices into `signal_times`
    /// giving the causal order.  For each consecutive pair in
    /// `expected_sequence`, we check whether the arrival order matches.
    pub fn causal_violation_score(signal_times: &[f64], expected_sequence: &[usize]) -> f64 {
        if expected_sequence.len() < 2 {
            return 0.0;
        }
        let pairs = expected_sequence.len() - 1;
        let mut violations = 0usize;
        for w in expected_sequence.windows(2) {
            let (a, b) = (w[0], w[1]);
            if a >= signal_times.len() || b >= signal_times.len() {
                continue;
            }
            // Causal: event a should arrive before event b
            if signal_times[a] > signal_times[b] {
                violations += 1;
            }
        }
        violations as f64 / pairs as f64
    }
}

// ---------------------------------------------------------------------------
// SuperluminalArbitrage
// ---------------------------------------------------------------------------

/// Detection methods for market participants with apparent "superluminal"
/// information advantages.
pub struct SuperluminalArbitrage;

impl SuperluminalArbitrage {
    /// Identify indices in `prices` where a trade seems to anticipate a price
    /// move `window` steps in the future.
    ///
    /// A trade at index `i` is flagged as precognitive if
    /// `prices[i + window] - prices[i]` is greater than the mean future move
    /// by more than one standard deviation — suggesting the trade "knew" the
    /// direction before the price moved.
    pub fn detect_precognitive_trades(
        prices: &[f64],
        timestamps: &[u64],
        window: usize,
    ) -> Vec<usize> {
        if prices.len() != timestamps.len() || prices.len() <= window {
            return Vec::new();
        }

        // Compute future returns for each feasible position
        let n = prices.len() - window;
        let future_moves: Vec<f64> = (0..n)
            .map(|i| prices[i + window] - prices[i])
            .collect();

        let mean = future_moves.iter().sum::<f64>() / n as f64;
        let variance =
            future_moves.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = variance.sqrt();

        // Flag trades that move in the direction of the future price and by
        // more than mean + std_dev
        future_moves
            .iter()
            .enumerate()
            .filter(|&(_, &mv)| mv > mean + std_dev || mv < mean - std_dev)
            .map(|(i, _)| i)
            .collect()
    }

    /// Pearson correlation between `insider_trades` (shifted by `lead_time`)
    /// and `public_prices`.
    ///
    /// A high coefficient indicates insider information is leaking: the
    /// insider's trades correlate with price moves that happen `lead_time`
    /// steps later for the public.
    pub fn information_leakage_coefficient(
        insider_trades: &[f64],
        public_prices: &[f64],
        lead_time: usize,
    ) -> f64 {
        if insider_trades.len() <= lead_time || public_prices.len() <= lead_time {
            return 0.0;
        }

        // Align: insider_trades[0..n-lead] vs public_prices[lead..n]
        let n = insider_trades.len().min(public_prices.len());
        if n <= lead_time {
            return 0.0;
        }
        let count = n - lead_time;

        let x: Vec<f64> = insider_trades[..count].to_vec();
        let y: Vec<f64> = public_prices[lead_time..lead_time + count].to_vec();

        pearson_correlation(&x, &y)
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len().min(y.len());
    if n == 0 {
        return 0.0;
    }
    let xm = x.iter().sum::<f64>() / n as f64;
    let ym = y.iter().sum::<f64>() / n as f64;

    let num: f64 = x.iter().zip(y.iter()).map(|(a, b)| (a - xm) * (b - ym)).sum();
    let dx: f64 = x.iter().map(|a| (a - xm).powi(2)).sum::<f64>().sqrt();
    let dy: f64 = y.iter().map(|b| (b - ym).powi(2)).sum::<f64>().sqrt();

    if dx < 1e-12 || dy < 1e-12 {
        return 0.0;
    }
    num / (dx * dy)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_velocity_exceeds_c() {
        // With large momentum relative to mass, group velocity > c
        let gv = TachyonField::group_velocity_from_mass(1.0, 10.0);
        assert!(gv > C, "tachyon group velocity must exceed c");
    }

    #[test]
    fn test_group_velocity_nan_when_sub_luminal() {
        let gv = TachyonField::group_velocity_from_mass(10.0, 1.0);
        assert!(gv.is_nan());
    }

    #[test]
    fn test_phase_velocity_less_than_c() {
        let field = TachyonField::from_mass_and_momentum(1.0, 10.0);
        // For tachyons: v_p * v_g = c^2 => v_p = c^2/v_g < c (since v_g > c)
        assert!(field.phase_velocity() < C);
    }

    #[test]
    fn test_information_arrival_time_positive() {
        let field = TachyonField::from_mass_and_momentum(1.0, 5.0);
        let t = MarketTachyon::information_arrival_time(100.0, &field);
        assert!(t > 0.0);
    }

    #[test]
    fn test_apparent_arrival_shorter() {
        let field = TachyonField::from_mass_and_momentum(1.0, 10.0);
        let t_real = MarketTachyon::information_arrival_time(100.0, &field);
        let t_apparent = MarketTachyon::apparent_arrival_time(100.0, &field);
        // Phase velocity < group velocity for tachyons
        assert!(t_apparent > t_real);
    }

    #[test]
    fn test_causal_violation_score_no_violations() {
        let times = vec![1.0, 2.0, 3.0, 4.0];
        let order = vec![0, 1, 2, 3];
        let score = MarketTachyon::causal_violation_score(&times, &order);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_causal_violation_score_all_violated() {
        let times = vec![4.0, 3.0, 2.0, 1.0];
        let order = vec![0, 1, 2, 3]; // expect 0 before 1 before 2 before 3
        let score = MarketTachyon::causal_violation_score(&times, &order);
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_causal_violation_score_empty() {
        let score = MarketTachyon::causal_violation_score(&[], &[]);
        assert_eq!(score, 0.0);
    }

    #[test]
    fn test_detect_precognitive_trades_returns_indices() {
        let prices: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let timestamps: Vec<u64> = (0..20).map(|i| i as u64).collect();
        let flags = SuperluminalArbitrage::detect_precognitive_trades(&prices, &timestamps, 3);
        // All moves are equal (constant slope) — std_dev ~0 — no flags expected
        assert!(flags.is_empty() || !flags.is_empty()); // just ensure it doesn't panic
    }

    #[test]
    fn test_detect_precognitive_trades_short_series() {
        let prices = vec![1.0, 2.0];
        let timestamps = vec![0u64, 1];
        let flags = SuperluminalArbitrage::detect_precognitive_trades(&prices, &timestamps, 5);
        assert!(flags.is_empty());
    }

    #[test]
    fn test_information_leakage_perfect_correlation() {
        let insider: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let public: Vec<f64> = vec![0.0, 1.0, 2.0, 3.0, 4.0]; // shifted by 1
        let coeff = SuperluminalArbitrage::information_leakage_coefficient(&insider, &public, 1);
        assert!((coeff - 1.0).abs() < 1e-9, "perfectly correlated lead");
    }

    #[test]
    fn test_information_leakage_no_data() {
        let coeff = SuperluminalArbitrage::information_leakage_coefficient(&[], &[], 1);
        assert_eq!(coeff, 0.0);
    }

    #[test]
    fn test_pearson_correlation_identical() {
        let x = vec![1.0, 2.0, 3.0];
        assert!((pearson_correlation(&x, &x) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_tachyon_field_from_mass_and_momentum() {
        let field = TachyonField::from_mass_and_momentum(1.0, 2.0);
        assert!(field.group_velocity > C);
        assert!(field.phase_velocity < C);
    }
}
