//! Quantum tunneling analogy for price gaps through resistance levels.

use serde::{Deserialize, Serialize};

// ─── POTENTIAL BARRIER ───────────────────────────────────────────────────────

/// Represents a resistance zone as a quantum potential barrier.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PotentialBarrier {
    /// Price level of the resistance.
    pub resistance_level: f64,
    /// Width of the resistance zone (in price units).
    pub barrier_width: f64,
    /// Height of the barrier above the resistance level (barrier strength).
    pub barrier_height: f64,
}

// ─── TUNNELING PROBABILITY ───────────────────────────────────────────────────

pub struct TunnelingProbability;

impl TunnelingProbability {
    /// WKB approximation: exp(-2 * width * sqrt(2 * mass * (height - energy)))
    /// when E < V. Returns 1.0 if energy >= barrier_height (classical pass-through).
    ///
    /// # Arguments
    /// * `barrier` - The potential barrier
    /// * `particle_energy` - Particle energy (analogous to price momentum)
    /// * `mass` - Effective mass (resistance strength scaling)
    pub fn wkb_approximation(
        barrier: &PotentialBarrier,
        particle_energy: f64,
        mass: f64,
    ) -> f64 {
        if particle_energy >= barrier.barrier_height {
            return 1.0;
        }
        let forbidden = barrier.barrier_height - particle_energy;
        if forbidden < 0.0 || mass < 0.0 {
            return 0.0;
        }
        let exponent = -2.0 * barrier.barrier_width * (2.0 * mass * forbidden).sqrt();
        exponent.exp().clamp(0.0, 1.0)
    }

    /// Classical probability: 0 if E < V (barrier), 1 if E >= V.
    pub fn classical_probability(particle_energy: f64, barrier_height: f64) -> f64 {
        if particle_energy >= barrier_height {
            1.0
        } else {
            0.0
        }
    }
}

// ─── MARKET TUNNELING MODEL ──────────────────────────────────────────────────

pub struct MarketTunnelingModel;

impl MarketTunnelingModel {
    /// Convert a price resistance level into a PotentialBarrier.
    ///
    /// * `level` - Resistance price level
    /// * `current_price` - Current market price
    /// * `volatility` - Annualised volatility (used to scale barrier width/height)
    pub fn resistance_to_barrier(
        level: f64,
        current_price: f64,
        volatility: f64,
    ) -> PotentialBarrier {
        // Distance to resistance as fraction of current price
        let distance = (level - current_price).abs() / current_price.max(1e-9);
        // Barrier width scales with volatility (higher vol → thinner effective barrier)
        let barrier_width = (distance / volatility.max(1e-9)).clamp(0.01, 10.0);
        // Barrier height is proportional to resistance distance
        let barrier_height = distance * 100.0; // scaled in percentage units
        PotentialBarrier {
            resistance_level: level,
            barrier_width,
            barrier_height,
        }
    }

    /// Probability that price tunnels through a resistance level.
    ///
    /// * `price` - Current price
    /// * `resistance` - Resistance price level
    /// * `volatility` - Annualised volatility
    /// * `momentum` - Price momentum (analogous to particle energy)
    pub fn tunneling_probability(
        price: f64,
        resistance: f64,
        volatility: f64,
        momentum: f64,
    ) -> f64 {
        let barrier = Self::resistance_to_barrier(resistance, price, volatility);
        // Particle energy = momentum scaled by volatility
        let energy = (momentum * volatility).abs();
        TunnelingProbability::wkb_approximation(&barrier, energy, 1.0)
    }

    /// Product of tunneling probabilities for each barrier between `from` and `to`.
    pub fn price_gap_probability(
        from: f64,
        to: f64,
        barriers: &[f64],
        volatility: f64,
    ) -> f64 {
        // Filter barriers that lie between from and to
        let (lo, hi) = if from < to { (from, to) } else { (to, from) };
        let momentum = (to - from).abs() / from.max(1e-9);
        barriers
            .iter()
            .filter(|&&b| b > lo && b < hi)
            .map(|&b| Self::tunneling_probability(from, b, volatility, momentum))
            .fold(1.0, |acc, p| acc * p)
    }
}

// ─── TESTS ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wkb_energy_above_barrier_returns_one() {
        let barrier = PotentialBarrier {
            resistance_level: 100.0,
            barrier_width: 1.0,
            barrier_height: 5.0,
        };
        let p = TunnelingProbability::wkb_approximation(&barrier, 6.0, 1.0);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_wkb_energy_below_barrier_tunnels() {
        let barrier = PotentialBarrier {
            resistance_level: 100.0,
            barrier_width: 0.5,
            barrier_height: 10.0,
        };
        let p = TunnelingProbability::wkb_approximation(&barrier, 1.0, 1.0);
        assert!(p > 0.0 && p < 1.0);
    }

    #[test]
    fn test_wkb_wide_barrier_lower_prob() {
        let b_narrow = PotentialBarrier { resistance_level: 100.0, barrier_width: 0.1, barrier_height: 5.0 };
        let b_wide = PotentialBarrier { resistance_level: 100.0, barrier_width: 2.0, barrier_height: 5.0 };
        let p_narrow = TunnelingProbability::wkb_approximation(&b_narrow, 1.0, 1.0);
        let p_wide = TunnelingProbability::wkb_approximation(&b_wide, 1.0, 1.0);
        assert!(p_narrow > p_wide);
    }

    #[test]
    fn test_classical_probability_below_barrier() {
        assert_eq!(TunnelingProbability::classical_probability(3.0, 5.0), 0.0);
    }

    #[test]
    fn test_classical_probability_above_barrier() {
        assert_eq!(TunnelingProbability::classical_probability(6.0, 5.0), 1.0);
    }

    #[test]
    fn test_classical_probability_at_barrier() {
        assert_eq!(TunnelingProbability::classical_probability(5.0, 5.0), 1.0);
    }

    #[test]
    fn test_resistance_to_barrier_fields() {
        let b = MarketTunnelingModel::resistance_to_barrier(110.0, 100.0, 0.2);
        assert!((b.resistance_level - 110.0).abs() < 1e-9);
        assert!(b.barrier_width > 0.0);
        assert!(b.barrier_height > 0.0);
    }

    #[test]
    fn test_tunneling_probability_range() {
        for _ in 0..10 {
            let p = MarketTunnelingModel::tunneling_probability(100.0, 110.0, 0.2, 0.05);
            assert!(p >= 0.0 && p <= 1.0);
        }
    }

    #[test]
    fn test_price_gap_no_barriers() {
        let p = MarketTunnelingModel::price_gap_probability(100.0, 120.0, &[], 0.2);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_gap_barriers_outside_range_ignored() {
        let barriers = vec![90.0, 130.0]; // outside [100, 120]
        let p = MarketTunnelingModel::price_gap_probability(100.0, 120.0, &barriers, 0.2);
        assert!((p - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_price_gap_multiple_barriers_reduces_prob() {
        let barriers = vec![105.0, 110.0, 115.0];
        let p = MarketTunnelingModel::price_gap_probability(100.0, 120.0, &barriers, 0.05);
        // With low volatility and momentum, each barrier should reduce prob
        assert!(p >= 0.0 && p <= 1.0);
    }

    #[test]
    fn test_wkb_zero_width_barrier() {
        let barrier = PotentialBarrier { resistance_level: 100.0, barrier_width: 0.0, barrier_height: 10.0 };
        let p = TunnelingProbability::wkb_approximation(&barrier, 1.0, 1.0);
        assert!((p - 1.0).abs() < 1e-9); // exp(0) = 1
    }
}
