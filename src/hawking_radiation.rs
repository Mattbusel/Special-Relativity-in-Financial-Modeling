//! Hawking radiation analogy for information leakage from financial "black holes".
//!
//! A *financial black hole* represents an extremely large, concentrated position
//! that has sucked in so much capital that it distorts the surrounding market.
//! Hawking-radiation-inspired formulas model how information (and risk) slowly
//! leaks back out as the position unwinds over time.

// ---------------------------------------------------------------------------
// Physical-analogy constants (adapted for finance, dimensionless)
// ---------------------------------------------------------------------------

/// ℏ (reduced Planck constant) analogue — sets the scale of "quantum" leakage.
const HBAR_FINANCE: f64 = 1.0546e-3;
/// *c*³ analogue — market propagation speed cubed (set to 1 in natural units).
const C_CUBED: f64 = 1.0;
/// 8π (from Hawking temperature formula denominator).
const EIGHT_PI: f64 = 8.0 * std::f64::consts::PI;
/// *G* analogue — "gravitational" coupling of capital concentration.
const G_FINANCE: f64 = 6.674e-2;
/// *k_B* (Boltzmann constant) analogue.
const K_B_FINANCE: f64 = 1.380649e-3;

// ---------------------------------------------------------------------------
// FinancialBlackHole
// ---------------------------------------------------------------------------

/// A concentrated market position modelled as a black hole.
///
/// * `mass`   — total accumulated notional position (always positive).
/// * `spin`   — directional asymmetry of the book, in `[0, 1]`.
/// * `charge` — net margin balance / leverage charge, in `(-∞, ∞)`.
#[derive(Debug, Clone, PartialEq)]
pub struct FinancialBlackHole {
    /// Accumulated position notional.  Must be positive.
    pub mass:   f64,
    /// Spin parameter (0 = Schwarzschild, 1 = extremal Kerr).  Clamped to [0, 1).
    pub spin:   f64,
    /// Net leverage charge.
    pub charge: f64,
}

impl FinancialBlackHole {
    /// Construct a new black hole, clamping `mass` to a small positive value
    /// and `spin` to `[0, 0.9999]`.
    pub fn new(mass: f64, spin: f64, charge: f64) -> Self {
        Self {
            mass:   mass.max(1e-12),
            spin:   spin.clamp(0.0, 0.9999),
            charge,
        }
    }

    /// Effective mass adjusted for spin (Kerr black hole analogy).
    ///
    /// A spinning black hole has a smaller effective Schwarzschild radius,
    /// so the effective mass for radiation purposes is reduced by the spin
    /// factor: `M_eff = M × (1 − a²)` where `a` is the spin parameter.
    pub fn effective_mass(&self) -> f64 {
        self.mass * (1.0 - self.spin * self.spin)
    }
}

// ---------------------------------------------------------------------------
// Hawking temperature
// ---------------------------------------------------------------------------

/// Hawking temperature analogue for a financial black hole.
///
/// Adapted from `T_H = ℏ c³ / (8 π G M k_B)` using finance-analogue constants.
///
/// Higher temperature ⇒ faster information leakage ⇒ faster unwind.
pub fn hawking_temperature(mass: f64) -> f64 {
    let m = mass.max(1e-12);
    (HBAR_FINANCE * C_CUBED) / (EIGHT_PI * G_FINANCE * m * K_B_FINANCE)
}

// ---------------------------------------------------------------------------
// Information leakage rate
// ---------------------------------------------------------------------------

/// Rate of information leakage from the financial black hole (bits / unit-time).
///
/// Proportional to the Hawking temperature divided by the spin-adjusted mass.
/// Larger, less-spinning positions leak information more slowly — they are
/// harder to unwind without market impact.
pub fn information_leakage_rate(bh: &FinancialBlackHole) -> f64 {
    let temp   = hawking_temperature(bh.effective_mass());
    // Leakage rate ∝ T² / M² (Stefan-Boltzmann analogy for a 2-D surface)
    let area   = bh.effective_mass() * bh.effective_mass();
    let area   = area.max(1e-24);
    temp * temp / area
}

// ---------------------------------------------------------------------------
// Evaporation time
// ---------------------------------------------------------------------------

/// Time until the position has fully unwound (evaporation time).
///
/// Adapted from `t_evap ∝ M³ / (ℏ c⁴)` — large positions take much longer
/// to unwind than small ones.  Spin accelerates evaporation.
pub fn evaporation_time(bh: &FinancialBlackHole) -> f64 {
    let m_eff = bh.effective_mass();
    // t_evap = M_eff³ / (ℏ × C⁴)
    let c4 = C_CUBED; // C⁴ ~ 1 in our unit system
    (m_eff * m_eff * m_eff) / (HBAR_FINANCE * c4)
}

// ---------------------------------------------------------------------------
// Entanglement pairs (Hawking pairs)
// ---------------------------------------------------------------------------

/// A pair of entangled market orders created at the "event horizon".
///
/// In Hawking radiation the pair-creation produces one particle that escapes
/// and one that falls in.  Here `particle` is the observable order flow
/// (visible to the market) and `antiparticle` is the hidden dark-pool flow
/// (equal and opposite, absorbed back into the position).
#[derive(Debug, Clone, PartialEq)]
pub struct EntanglementPair {
    /// Visible order size / direction (+positive = buy).
    pub particle:      f64,
    /// Hidden counter-order (-particle in a perfectly entangled pair).
    pub antiparticle:  f64,
    /// Pearson-like correlation of the pair's time series: 1.0 = perfectly
    /// anti-correlated, 0.0 = uncorrelated (decoherence).
    pub correlation:   f64,
}

/// Generate an entangled order pair from a volatility level and a seed.
///
/// Uses a simple LCG for reproducibility with no external crate.
pub fn generate_entangled_orders(volatility: f64, seed: u64) -> EntanglementPair {
    // LCG: Knuth multiplier.
    let s1 = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let s2 = s1
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let s3 = s2
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);

    // Map to floats in [0,1).
    let r1 = ((s1 >> 11) as f64) / (1u64 << 53) as f64;
    let r2 = ((s2 >> 11) as f64) / (1u64 << 53) as f64;
    let r3 = ((s3 >> 11) as f64) / (1u64 << 53) as f64;

    // Particle size in (-volatility, +volatility).
    let particle = (r1 * 2.0 - 1.0) * volatility;
    // Antiparticle is opposite but slightly noisy.
    let noise    = (r2 * 2.0 - 1.0) * volatility * 0.05;
    let antiparticle = -particle + noise;
    // Correlation decays with volatility: very volatile ⇒ decoherence.
    let correlation = (1.0 - volatility.min(1.0) * r3).max(0.0);

    EntanglementPair { particle, antiparticle, correlation }
}

// ---------------------------------------------------------------------------
// Schwarzschild radius analogue
// ---------------------------------------------------------------------------

/// The "event horizon" radius (Schwarzschild radius analogue).
///
/// `r_s = 2 G M / c²`
pub fn schwarzschild_radius(mass: f64) -> f64 {
    2.0 * G_FINANCE * mass.max(0.0) / (C_CUBED * C_CUBED)
}

// ---------------------------------------------------------------------------
// Black hole entropy (Bekenstein-Hawking)
// ---------------------------------------------------------------------------

/// Information entropy of the financial black hole (Bekenstein-Hawking analogue).
///
/// `S_BH = A / (4 ℏ)` where `A ∝ r_s²` is the event-horizon area.
pub fn bekenstein_hawking_entropy(bh: &FinancialBlackHole) -> f64 {
    let r_s = schwarzschild_radius(bh.mass);
    let area = r_s * r_s * std::f64::consts::PI; // π r²
    area / (4.0 * HBAR_FINANCE)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // --- FinancialBlackHole ---

    #[test]
    fn bh_mass_clamped_above_zero() {
        let bh = FinancialBlackHole::new(-10.0, 0.0, 0.0);
        assert!(bh.mass > 0.0);
    }

    #[test]
    fn bh_spin_clamped_below_one() {
        let bh = FinancialBlackHole::new(1.0, 2.0, 0.0);
        assert!(bh.spin < 1.0);
    }

    #[test]
    fn effective_mass_without_spin_equals_mass() {
        let bh = FinancialBlackHole::new(100.0, 0.0, 0.0);
        assert!((bh.effective_mass() - 100.0).abs() < EPS);
    }

    #[test]
    fn effective_mass_reduced_by_spin() {
        let bh_no_spin   = FinancialBlackHole::new(100.0, 0.0, 0.0);
        let bh_with_spin = FinancialBlackHole::new(100.0, 0.5, 0.0);
        assert!(bh_with_spin.effective_mass() < bh_no_spin.effective_mass());
    }

    // --- hawking_temperature ---

    #[test]
    fn temperature_positive() {
        assert!(hawking_temperature(1.0) > 0.0);
    }

    #[test]
    fn temperature_decreases_with_mass() {
        let t1 = hawking_temperature(1.0);
        let t2 = hawking_temperature(10.0);
        assert!(t1 > t2, "smaller mass should be hotter: {} vs {}", t1, t2);
    }

    #[test]
    fn temperature_zero_mass_is_finite() {
        let t = hawking_temperature(0.0);
        assert!(t.is_finite());
        assert!(t > 0.0);
    }

    // --- information_leakage_rate ---

    #[test]
    fn leakage_rate_positive() {
        let bh = FinancialBlackHole::new(1.0, 0.0, 0.0);
        assert!(information_leakage_rate(&bh) > 0.0);
    }

    #[test]
    fn smaller_bh_leaks_faster() {
        let big   = FinancialBlackHole::new(1000.0, 0.0, 0.0);
        let small = FinancialBlackHole::new(1.0, 0.0, 0.0);
        assert!(information_leakage_rate(&small) > information_leakage_rate(&big));
    }

    // --- evaporation_time ---

    #[test]
    fn evaporation_time_positive() {
        let bh = FinancialBlackHole::new(5.0, 0.0, 0.0);
        assert!(evaporation_time(&bh) > 0.0);
    }

    #[test]
    fn evaporation_time_scales_cubically_with_mass() {
        let bh1 = FinancialBlackHole::new(1.0, 0.0, 0.0);
        let bh2 = FinancialBlackHole::new(2.0, 0.0, 0.0);
        let ratio = evaporation_time(&bh2) / evaporation_time(&bh1);
        assert!((ratio - 8.0).abs() < 1e-6, "expected 8×, got {}", ratio);
    }

    #[test]
    fn spin_reduces_evaporation_time() {
        let bh_static  = FinancialBlackHole::new(10.0, 0.0, 0.0);
        let bh_spinning = FinancialBlackHole::new(10.0, 0.5, 0.0);
        assert!(evaporation_time(&bh_spinning) < evaporation_time(&bh_static));
    }

    // --- generate_entangled_orders ---

    #[test]
    fn entangled_orders_opposite_sign() {
        let pair = generate_entangled_orders(0.1, 42);
        // particle and antiparticle should have approximately opposite signs.
        assert!(pair.particle * pair.antiparticle < 0.0 || pair.particle.abs() < 1e-12);
    }

    #[test]
    fn entangled_orders_same_seed_reproducible() {
        let p1 = generate_entangled_orders(0.2, 1234);
        let p2 = generate_entangled_orders(0.2, 1234);
        assert_eq!(p1.particle, p2.particle);
        assert_eq!(p1.antiparticle, p2.antiparticle);
        assert_eq!(p1.correlation, p2.correlation);
    }

    #[test]
    fn entangled_orders_correlation_in_range() {
        for seed in [0, 1, 42, 999, u64::MAX / 2] {
            let pair = generate_entangled_orders(0.5, seed);
            assert!(pair.correlation >= 0.0 && pair.correlation <= 1.0,
                    "correlation out of [0,1]: {}", pair.correlation);
        }
    }

    #[test]
    fn high_volatility_reduces_correlation() {
        let low_vol  = generate_entangled_orders(0.01, 1);
        let high_vol = generate_entangled_orders(0.99, 1);
        // High volatility should produce lower or equal correlation.
        assert!(high_vol.correlation <= low_vol.correlation + 0.1);
    }

    // --- schwarzschild_radius ---

    #[test]
    fn schwarzschild_radius_positive_for_positive_mass() {
        assert!(schwarzschild_radius(1.0) > 0.0);
    }

    #[test]
    fn schwarzschild_radius_zero_for_zero_mass() {
        assert!((schwarzschild_radius(0.0)).abs() < EPS);
    }

    // --- bekenstein_hawking_entropy ---

    #[test]
    fn entropy_positive() {
        let bh = FinancialBlackHole::new(10.0, 0.0, 0.0);
        assert!(bekenstein_hawking_entropy(&bh) > 0.0);
    }

    #[test]
    fn entropy_larger_for_bigger_bh() {
        let small = FinancialBlackHole::new(1.0, 0.0, 0.0);
        let large = FinancialBlackHole::new(100.0, 0.0, 0.0);
        assert!(bekenstein_hawking_entropy(&large) > bekenstein_hawking_entropy(&small));
    }
}
