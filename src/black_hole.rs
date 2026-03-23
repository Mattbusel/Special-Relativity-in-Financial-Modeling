//! Schwarzschild black hole physics, Hawking radiation, Penrose diagram, and
//! market black-hole analogues.

/// Gravitational constant × solar mass in SI: G·M_sun ≈ 1.327×10²⁰ m³/s²
const GM_SUN_SI: f64 = 1.327_124_4e20;
/// Speed of light in m/s.
const C_SI: f64 = 2.997_924_58e8;
/// Schwarzschild radius per solar mass in metres (≈ 2954 m ≈ 2.95 km).
const RS_PER_SOLAR_MASS_M: f64 = 2.0 * GM_SUN_SI / (C_SI * C_SI);

/// Hawking temperature coefficient: T_H = COEFF / M_solar [Kelvin]
/// T_H = ℏc³/(8πGMk_B)  ≈ 6.17×10⁻⁸ / M_sun  K
const HAWKING_T_COEFF: f64 = 6.17e-8;

/// Stefan–Boltzmann constant in W m⁻² K⁻⁴.
const SIGMA_SB: f64 = 5.670_374_419e-8;

/// A Schwarzschild (or Kerr-approximation) black hole.
#[derive(Debug, Clone)]
pub struct SchwarzschildBH {
    /// Mass in solar masses.
    pub mass_solar: f64,
    /// Dimensionless spin parameter a* ∈ [0, 1).  0 = Schwarzschild, >0 = Kerr-like.
    pub spin_param: f64,
}

impl SchwarzschildBH {
    /// Create a Schwarzschild (non-spinning) black hole.
    pub fn new(mass_solar: f64) -> Self {
        Self {
            mass_solar,
            spin_param: 0.0,
        }
    }

    /// Create a Kerr-like black hole with the given spin parameter.
    pub fn kerr(mass_solar: f64, spin_param: f64) -> Self {
        assert!(
            (0.0..1.0).contains(&spin_param),
            "spin_param must be in [0, 1)"
        );
        Self { mass_solar, spin_param }
    }

    /// Schwarzschild radius r_s = 2GM/c² [metres].
    pub fn schwarzschild_radius(&self) -> f64 {
        RS_PER_SOLAR_MASS_M * self.mass_solar
    }

    /// Photon sphere radius r_ph = (3/2) r_s  [metres].
    pub fn photon_sphere(&self) -> f64 {
        1.5 * self.schwarzschild_radius()
    }

    /// Innermost stable circular orbit (ISCO).
    /// For Schwarzschild: r_ISCO = 3 r_s.
    /// For Kerr (prograde): approximate as r_ISCO = r_s · (3 - a*·√6/3 + ...).
    pub fn innermost_stable_orbit(&self) -> f64 {
        let rs = self.schwarzschild_radius();
        if self.spin_param == 0.0 {
            3.0 * rs
        } else {
            // Prograde Kerr approximation
            rs * (3.0 - self.spin_param * 2.0_f64.sqrt())
        }
    }

    /// Metric component g_tt(r) = -(1 - r_s/r).
    /// Undefined (returns f64::NAN) inside the horizon (r ≤ r_s).
    pub fn metric_g_tt(&self, r: f64) -> f64 {
        let rs = self.schwarzschild_radius();
        -(1.0 - rs / r)
    }

    /// Metric component g_rr(r) = 1/(1 - r_s/r).
    pub fn metric_g_rr(&self, r: f64) -> f64 {
        let rs = self.schwarzschild_radius();
        1.0 / (1.0 - rs / r)
    }

    /// Gravitational redshift factor between emission radius r_emit and
    /// observation radius r_obs.
    /// z_grav = sqrt(g_tt(r_obs) / g_tt(r_emit))
    pub fn gravitational_redshift(&self, r_emit: f64, r_obs: f64) -> f64 {
        let rs = self.schwarzschild_radius();
        if r_emit <= rs || r_obs <= rs {
            return f64::INFINITY;
        }
        let g_tt_obs = (1.0 - rs / r_obs).abs();
        let g_tt_emit = (1.0 - rs / r_emit).abs();
        (g_tt_obs / g_tt_emit).sqrt()
    }

    /// Hawking temperature T_H ≈ 6.17×10⁻⁸ / M_solar  [Kelvin].
    pub fn hawking_temperature(&self) -> f64 {
        HAWKING_T_COEFF / self.mass_solar
    }

    /// Hawking radiation power using Stefan–Boltzmann law over the BH surface area.
    /// P = σ · (4π r_s²) · T_H⁴  [Watts]
    pub fn hawking_radiation_power(&self) -> f64 {
        let rs = self.schwarzschild_radius();
        let area = 4.0 * std::f64::consts::PI * rs * rs;
        let t = self.hawking_temperature();
        SIGMA_SB * area * t.powi(4)
    }

    /// Evaporation time: t ∝ M³.
    /// t_evap ≈ 5120 π G² M³ / (ℏ c⁴)  — in seconds (Planck units approximation).
    /// Returns time in years.
    pub fn evaporation_time(&self) -> f64 {
        // Coefficient in years per solar_mass³:
        // t_evap [yr] ≈ 2.09e67 * (M/M_sun)^3
        2.09e67 * self.mass_solar.powi(3)
    }

    /// Tidal (spaghettification) force on an object near the black hole.
    /// F_tidal = 2GM · m · l / r³  [Newtons]
    /// * r — distance from BH centre [metres]
    /// * object_mass [kg]
    /// * object_length [metres]
    pub fn spaghettification_force(r: f64, object_mass: f64, object_length: f64) -> f64 {
        if r <= 0.0 {
            return f64::INFINITY;
        }
        // Using GM_sun as proxy; caller should scale mass_solar accordingly
        2.0 * GM_SUN_SI * object_mass * object_length / r.powi(3)
    }
}

/// Tools for computing Kruskal–Szekeres coordinates and regions in the
/// Penrose (conformal) diagram of a Schwarzschild black hole.
pub struct PenroseDiagram;

impl PenroseDiagram {
    /// Kruskal U coordinate for an exterior point (r > r_s).
    /// U = -exp(-(t - r*) / (2r_s))  where r* = r + r_s·ln|r/r_s - 1|
    pub fn kruskal_u(t: f64, r: f64, r_s: f64) -> f64 {
        if r <= r_s || r_s <= 0.0 {
            return f64::NAN;
        }
        let r_star = r + r_s * ((r / r_s - 1.0).abs()).ln();
        -((-(t - r_star) / (2.0 * r_s)).exp())
    }

    /// Kruskal V coordinate for an exterior point (r > r_s).
    /// V = exp((t + r*) / (2r_s))
    pub fn kruskal_v(t: f64, r: f64, r_s: f64) -> f64 {
        if r <= r_s || r_s <= 0.0 {
            return f64::NAN;
        }
        let r_star = r + r_s * ((r / r_s - 1.0).abs()).ln();
        ((t + r_star) / (2.0 * r_s)).exp()
    }

    /// Determine the causal region of the Kruskal diagram from (U, V) coordinates.
    /// Regions:
    ///  - UV < 0  → exterior (our universe)
    ///  - V > 0, U > 0 → future interior (black hole)
    ///  - V < 0, U < 0 → past interior (white hole)
    ///  - UV > 0, U < 0 → parallel exterior (other universe)
    pub fn region(u: f64, v: f64) -> &'static str {
        if u < 0.0 && v > 0.0 {
            "exterior"
        } else if u > 0.0 && v > 0.0 {
            "future_interior"
        } else if u < 0.0 && v < 0.0 {
            "past_interior"
        } else if u > 0.0 && v < 0.0 {
            "parallel_exterior"
        } else {
            "horizon"
        }
    }
}

/// Market analogues of black-hole phenomena.
pub struct MarketBlackHole;

impl MarketBlackHole {
    /// Detect a flash crash — a singularity in the price series.
    /// Returns the index of the largest single-step price drop exceeding
    /// 3 standard deviations of all returns, or None if no such event.
    pub fn market_singularity(prices: &[f64]) -> Option<usize> {
        if prices.len() < 3 {
            return None;
        }
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[1] / w[0]) - 1.0)
            .collect();

        let n = returns.len() as f64;
        let mean = returns.iter().sum::<f64>() / n;
        let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
        let std = variance.sqrt();
        if std <= 0.0 {
            return None;
        }

        let threshold = -3.0 * std;
        returns
            .iter()
            .enumerate()
            .filter(|(_, &r)| r < threshold)
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i + 1) // index into prices
    }

    /// KL divergence D_KL(P || Q) between return distributions of two price series.
    /// Uses a simple histogram with `n_bins` bins over the joint range.
    pub fn information_loss(pre_crash: &[f64], post_crash: &[f64]) -> f64 {
        if pre_crash.len() < 2 || post_crash.len() < 2 {
            return 0.0;
        }
        let pre_ret: Vec<f64> = pre_crash
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let post_ret: Vec<f64> = post_crash
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();

        let n_bins = 20usize;
        let all: Vec<f64> = pre_ret.iter().chain(post_ret.iter()).cloned().collect();
        let min_v = all.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_v = all.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = max_v - min_v;
        if range <= 0.0 {
            return 0.0;
        }

        let bin_index = |v: f64| {
            ((v - min_v) / range * n_bins as f64)
                .floor()
                .clamp(0.0, (n_bins - 1) as f64) as usize
        };

        let mut p = vec![0.0f64; n_bins];
        let mut q = vec![0.0f64; n_bins];

        for v in &pre_ret {
            p[bin_index(*v)] += 1.0;
        }
        for v in &post_ret {
            q[bin_index(*v)] += 1.0;
        }

        let pn = pre_ret.len() as f64;
        let qn = post_ret.len() as f64;
        let eps = 1e-10;

        (0..n_bins)
            .map(|i| {
                let pi = (p[i] + eps) / (pn + eps * n_bins as f64);
                let qi = (q[i] + eps) / (qn + eps * n_bins as f64);
                pi * (pi / qi).ln()
            })
            .sum()
    }

    /// Hawking radiation analogue: information leakage rate after a crash.
    /// Models the rate at which market memory of the pre-crash state dissipates.
    /// hawking_analog = crash_speed / (1 + recovery_time)
    pub fn hawking_analog(crash_speed: f64, recovery_time: f64) -> f64 {
        crash_speed / (1.0 + recovery_time.max(0.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_schwarzschild_radius() {
        let bh = SchwarzschildBH::new(1.0);
        let rs = bh.schwarzschild_radius();
        // Should be ~2954 m for 1 solar mass
        assert!((rs - 2953.0).abs() < 50.0, "r_s = {}", rs);
    }

    #[test]
    fn test_photon_sphere_is_1_5_rs() {
        let bh = SchwarzschildBH::new(10.0);
        let rs = bh.schwarzschild_radius();
        let rph = bh.photon_sphere();
        assert!((rph - 1.5 * rs).abs() < 1e-6);
    }

    #[test]
    fn test_isco_schwarzschild() {
        let bh = SchwarzschildBH::new(5.0);
        let rs = bh.schwarzschild_radius();
        let isco = bh.innermost_stable_orbit();
        assert!((isco - 3.0 * rs).abs() < 1e-6);
    }

    #[test]
    fn test_hawking_temperature_inversely_proportional_to_mass() {
        let bh1 = SchwarzschildBH::new(1.0);
        let bh2 = SchwarzschildBH::new(2.0);
        let ratio = bh1.hawking_temperature() / bh2.hawking_temperature();
        assert!((ratio - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_gravitational_redshift() {
        let bh = SchwarzschildBH::new(1.0);
        let rs = bh.schwarzschild_radius();
        // At large distances the factor should approach 1
        let z_far = bh.gravitational_redshift(rs * 100.0, rs * 10000.0);
        assert!((z_far - 1.0).abs() < 0.01);
    }

    #[test]
    fn test_kruskal_region_exterior() {
        let region = PenroseDiagram::region(-1.0, 1.0);
        assert_eq!(region, "exterior");
    }

    #[test]
    fn test_kruskal_region_future_interior() {
        let region = PenroseDiagram::region(1.0, 1.0);
        assert_eq!(region, "future_interior");
    }

    #[test]
    fn test_market_singularity_detection() {
        let mut prices: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64).collect();
        // Insert a flash crash at index 10
        prices[10] = 50.0;
        let result = MarketBlackHole::market_singularity(&prices);
        assert!(result.is_some());
    }

    #[test]
    fn test_information_loss_identical_series() {
        let prices: Vec<f64> = (1..=20).map(|i| 100.0 + i as f64 * 0.5).collect();
        let kl = MarketBlackHole::information_loss(&prices, &prices);
        // KL divergence of a distribution against itself should be ~0
        assert!(kl < 0.01, "KL = {}", kl);
    }

    #[test]
    fn test_hawking_analog_positive() {
        let rate = MarketBlackHole::hawking_analog(0.1, 5.0);
        assert!(rate > 0.0);
        assert!(rate < 0.1);
    }
}
