//! Heisenberg uncertainty analogy for financial price/momentum trade-offs.
//!
//! In quantum mechanics, position and momentum cannot be simultaneously known
//! with arbitrary precision.  This module applies the same analogy to price
//! observations and price-momentum estimates in financial markets: a high-
//! precision price measurement (tight bid-ask) disturbs the momentum estimate,
//! and vice versa.

/// A single price measurement with its associated precision.
#[derive(Debug, Clone, PartialEq)]
pub struct PriceObservation {
    /// Observed mid price.
    pub price: f64,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Standard deviation of the measurement error (tighter = more precise).
    pub measurement_precision: f64,
}

/// An estimate of price momentum dp/dt with associated precision.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentumObservation {
    /// Rate of price change (units: currency per millisecond).
    pub momentum: f64,
    /// Window over which momentum was estimated (ms).
    pub window_ms: u64,
    /// Standard deviation of the momentum estimate.
    pub precision: f64,
}

/// Represents an uncertainty relation between price and momentum measurements.
#[derive(Debug, Clone, PartialEq)]
pub struct UncertaintyRelation {
    /// Uncertainty in price (σ_x).
    pub price_uncertainty: f64,
    /// Uncertainty in momentum (σ_p).
    pub momentum_uncertainty: f64,
    /// Planck-analogue constant (financial ℏ).
    pub planck_analog: f64,
}

impl UncertaintyRelation {
    /// Returns true when σ_x · σ_p ≥ planck_analog / 2.
    pub fn satisfies_bound(&self) -> bool {
        self.price_uncertainty * self.momentum_uncertainty >= self.planck_analog / 2.0
    }

    /// Given a desired momentum precision, compute the minimum achievable price uncertainty.
    pub fn minimum_price_uncertainty(momentum_precision: f64, planck: f64) -> f64 {
        if momentum_precision <= 0.0 {
            return f64::INFINITY;
        }
        planck / (2.0 * momentum_precision)
    }
}

/// Analyzes market uncertainty using Heisenberg-inspired bounds.
#[derive(Debug, Default)]
pub struct MarketUncertaintyAnalyzer {
    /// Financial Planck constant analogue (calibrated from market microstructure data).
    pub planck_analog: f64,
}

impl MarketUncertaintyAnalyzer {
    pub fn new(planck_analog: f64) -> Self {
        Self { planck_analog }
    }

    /// Estimate the price uncertainty as the sample standard deviation of observed prices.
    pub fn estimate_price_uncertainty(&self, observations: &[PriceObservation]) -> f64 {
        sample_std(&observations.iter().map(|o| o.price).collect::<Vec<_>>())
    }

    /// Estimate the momentum uncertainty as the sample standard deviation of momentum values.
    pub fn estimate_momentum_uncertainty(&self, observations: &[MomentumObservation]) -> f64 {
        sample_std(&observations.iter().map(|o| o.momentum).collect::<Vec<_>>())
    }

    /// Build an `UncertaintyRelation` from two slices of observations.
    pub fn check_uncertainty_bound(
        &self,
        prices: &[PriceObservation],
        momenta: &[MomentumObservation],
    ) -> UncertaintyRelation {
        UncertaintyRelation {
            price_uncertainty: self.estimate_price_uncertainty(prices),
            momentum_uncertainty: self.estimate_momentum_uncertainty(momenta),
            planck_analog: self.planck_analog,
        }
    }

    /// Model the observer effect: measuring price with high intensity disturbs it.
    ///
    /// Returns: true_price + measurement_intensity * ε  where ε ~ N(0, 1) is approximated
    /// by a deterministic perturbation proportional to measurement_intensity.
    pub fn observer_effect(&self, true_price: f64, measurement_intensity: f64) -> f64 {
        // Deterministic proxy for disturbance: intensity × planck_analog.
        true_price + measurement_intensity * self.planck_analog
    }
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────

fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return 0.0;
    }
    let mean = values.iter().sum::<f64>() / n as f64;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    variance.sqrt()
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn price_obs(prices: &[f64]) -> Vec<PriceObservation> {
        prices
            .iter()
            .enumerate()
            .map(|(i, &p)| PriceObservation {
                price: p,
                timestamp_ms: i as u64 * 100,
                measurement_precision: 0.01,
            })
            .collect()
    }

    fn momentum_obs(momenta: &[f64]) -> Vec<MomentumObservation> {
        momenta
            .iter()
            .map(|&m| MomentumObservation {
                momentum: m,
                window_ms: 1000,
                precision: 0.001,
            })
            .collect()
    }

    #[test]
    fn test_satisfies_bound_true() {
        let rel = UncertaintyRelation {
            price_uncertainty: 1.0,
            momentum_uncertainty: 1.0,
            planck_analog: 1.0,
        };
        assert!(rel.satisfies_bound()); // 1.0 * 1.0 = 1.0 >= 0.5
    }

    #[test]
    fn test_satisfies_bound_false() {
        let rel = UncertaintyRelation {
            price_uncertainty: 0.1,
            momentum_uncertainty: 0.1,
            planck_analog: 1.0,
        };
        assert!(!rel.satisfies_bound()); // 0.01 < 0.5
    }

    #[test]
    fn test_minimum_price_uncertainty() {
        let min_pu = UncertaintyRelation::minimum_price_uncertainty(2.0, 1.0);
        assert!((min_pu - 0.25).abs() < 1e-9);
    }

    #[test]
    fn test_minimum_price_uncertainty_zero_momentum_precision() {
        let min_pu = UncertaintyRelation::minimum_price_uncertainty(0.0, 1.0);
        assert!(min_pu.is_infinite());
    }

    #[test]
    fn test_estimate_price_uncertainty() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.5);
        let obs = price_obs(&[100.0, 101.0, 102.0, 99.0, 100.5]);
        let sigma = analyzer.estimate_price_uncertainty(&obs);
        assert!(sigma > 0.0);
        // std of [100, 101, 102, 99, 100.5] is roughly 1.12
        assert!((sigma - 1.12).abs() < 0.05);
    }

    #[test]
    fn test_estimate_price_uncertainty_single_obs() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.5);
        let obs = price_obs(&[100.0]);
        assert_eq!(analyzer.estimate_price_uncertainty(&obs), 0.0);
    }

    #[test]
    fn test_estimate_momentum_uncertainty() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.5);
        let obs = momentum_obs(&[0.1, 0.2, -0.1, 0.05, 0.15]);
        let sigma = analyzer.estimate_momentum_uncertainty(&obs);
        assert!(sigma > 0.0);
    }

    #[test]
    fn test_check_uncertainty_bound_returns_relation() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.5);
        let prices = price_obs(&[100.0, 101.0, 102.0]);
        let momenta = momentum_obs(&[0.1, 0.2, 0.15]);
        let rel = analyzer.check_uncertainty_bound(&prices, &momenta);
        assert!((rel.planck_analog - 0.5).abs() < 1e-9);
        assert!(rel.price_uncertainty >= 0.0);
        assert!(rel.momentum_uncertainty >= 0.0);
    }

    #[test]
    fn test_observer_effect_positive_intensity() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.1);
        let disturbed = analyzer.observer_effect(100.0, 1.0);
        assert!((disturbed - 100.1).abs() < 1e-9);
    }

    #[test]
    fn test_observer_effect_zero_intensity() {
        let analyzer = MarketUncertaintyAnalyzer::new(0.1);
        let disturbed = analyzer.observer_effect(100.0, 0.0);
        assert!((disturbed - 100.0).abs() < 1e-9);
    }

    #[test]
    fn test_sample_std_known_values() {
        // std([2, 4, 4, 4, 5, 5, 7, 9]) = 2.0 (population); sample ≈ 2.138
        let values = vec![2.0f64, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let std = super::sample_std(&values);
        assert!((std - 2.138).abs() < 0.01);
    }
}
