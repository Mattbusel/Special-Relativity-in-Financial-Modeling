//! FLRW cosmology, Hubble expansion, and market cosmology analogues.

/// Cosmological parameters for a flat ΛCDM (or general FLRW) universe.
#[derive(Debug, Clone)]
pub struct CosmologicalParams {
    /// Hubble constant in km/s/Mpc.
    pub h0: f64,
    /// Matter density parameter Ω_m.
    pub omega_m: f64,
    /// Radiation density parameter Ω_r.
    pub omega_r: f64,
    /// Dark-energy density parameter Ω_Λ.
    pub omega_lambda: f64,
    /// Curvature density parameter Ω_k = 1 - Ω_m - Ω_r - Ω_Λ.
    pub omega_k: f64,
}

impl CosmologicalParams {
    /// Construct a flat ΛCDM model given H₀ and Ω_m.
    /// Radiation is neglected (Ω_r = 0); dark energy fills the rest.
    pub fn flat_lcdm(h0: f64, omega_m: f64) -> Self {
        let omega_lambda = 1.0 - omega_m;
        Self {
            h0,
            omega_m,
            omega_r: 0.0,
            omega_lambda,
            omega_k: 0.0,
        }
    }

    /// E²(z) = Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_k(1+z)² + Ω_Λ
    pub fn e_squared(&self, z: f64) -> f64 {
        let zp1 = 1.0 + z;
        self.omega_m * zp1.powi(3)
            + self.omega_r * zp1.powi(4)
            + self.omega_k * zp1.powi(2)
            + self.omega_lambda
    }

    /// H(z) = H₀ · √(E²(z))  [km/s/Mpc]
    pub fn hubble_at_z(&self, z: f64) -> f64 {
        self.h0 * self.e_squared(z).sqrt()
    }
}

/// Speed of light in km/s.
const C_KM_S: f64 = 299_792.458;

/// Numerical integration helper using the trapezoidal rule.
fn trapz<F: Fn(f64) -> f64>(f: F, a: f64, b: f64, n: usize) -> f64 {
    if n == 0 {
        return 0.0;
    }
    let h = (b - a) / n as f64;
    let mut sum = 0.5 * (f(a) + f(b));
    for i in 1..n {
        sum += f(a + i as f64 * h);
    }
    sum * h
}

/// Cosmological distance and time calculator.
pub struct CosmologicalCalculator;

impl CosmologicalCalculator {
    /// Comoving distance D_C in Mpc.
    /// D_C = (c/H₀) ∫₀ᶻ dz'/E(z')
    pub fn comoving_distance(z: f64, params: &CosmologicalParams, n_steps: usize) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let integral = trapz(
            |zp| {
                let e2 = params.e_squared(zp);
                if e2 <= 0.0 { 0.0 } else { 1.0 / e2.sqrt() }
            },
            0.0,
            z,
            n_steps.max(100),
        );
        (C_KM_S / params.h0) * integral
    }

    /// Luminosity distance D_L = (1+z) · D_C  [Mpc]
    pub fn luminosity_distance(z: f64, params: &CosmologicalParams) -> f64 {
        let dc = Self::comoving_distance(z, params, 1000);
        (1.0 + z) * dc
    }

    /// Angular diameter distance D_A = D_C / (1+z)  [Mpc]
    pub fn angular_diameter_distance(z: f64, params: &CosmologicalParams) -> f64 {
        let dc = Self::comoving_distance(z, params, 1000);
        dc / (1.0 + z)
    }

    /// Lookback time in Gyr (Giga-years).
    /// t_lb = (1/H₀) ∫₀ᶻ dz'/[(1+z') · E(z')]
    /// H₀ is in km/s/Mpc; 1 Mpc / (1 km/s) ≈ 1.0227 Gyr.
    pub fn lookback_time(z: f64, params: &CosmologicalParams) -> f64 {
        if z <= 0.0 {
            return 0.0;
        }
        let hubble_time_gyr = (1.0 / params.h0) * 978.0; // 978 Gyr·km/s/Mpc
        let integral = trapz(
            |zp| {
                let e2 = params.e_squared(zp);
                if e2 <= 0.0 { 0.0 } else { 1.0 / ((1.0 + zp) * e2.sqrt()) }
            },
            0.0,
            z,
            1000,
        );
        hubble_time_gyr * integral
    }

    /// Age of the universe in Gyr (integrates to z = 1000).
    pub fn age_of_universe(params: &CosmologicalParams) -> f64 {
        Self::lookback_time(1000.0, params)
    }

    /// Comoving volume shell between z1 and z2 in Gpc³.
    /// V = (4π/3) · [D_C(z2)³ - D_C(z1)³] / (1000 Mpc/Gpc)³
    pub fn volume_shell(z1: f64, z2: f64, params: &CosmologicalParams) -> f64 {
        let dc1 = Self::comoving_distance(z1, params, 1000); // Mpc
        let dc2 = Self::comoving_distance(z2, params, 1000); // Mpc
        let mpc_to_gpc = 1.0e-3;
        let dc1_gpc = dc1 * mpc_to_gpc;
        let dc2_gpc = dc2 * mpc_to_gpc;
        (4.0 * std::f64::consts::PI / 3.0) * (dc2_gpc.powi(3) - dc1_gpc.powi(3))
    }

    /// Growth factor D₊(z) — approximation for a flat ΛCDM universe.
    /// Uses the fitting formula: D₊ ∝ H(z)/H₀ · ∫_z^∞ dz'(1+z')/E³(z')
    pub fn perturbation_growth(z: f64, params: &CosmologicalParams) -> f64 {
        let integrand = |zp: f64| {
            let e3 = params.e_squared(zp).sqrt().powi(3);
            if e3 <= 0.0 { 0.0 } else { (1.0 + zp) / e3 }
        };
        // Integrate from z to z_max = 100 (approximating infinity)
        let z_max = 100.0_f64.max(z + 1.0);
        let integral_z = trapz(integrand, z, z_max, 1000);
        let integral_0 = trapz(integrand, 0.0, z_max, 1000);
        // Normalized so D₊(0) = 1
        let ez = params.e_squared(z).sqrt();
        let e0 = params.e_squared(0.0).sqrt();
        if integral_0 <= 0.0 || e0 <= 0.0 {
            return 0.0;
        }
        (ez * integral_z) / (e0 * integral_0)
    }
}

/// Market cosmology — analogues of cosmological quantities applied to price series.
pub struct MarketCosmology;

impl MarketCosmology {
    /// Local "Hubble constant" analog: rate of price expansion in a rolling window.
    /// Returns dln(P)/dt over each window — i.e., the log-return rate.
    pub fn market_expansion_rate(prices: &[f64], window: usize) -> Vec<f64> {
        if prices.len() < 2 || window < 2 {
            return Vec::new();
        }
        let w = window.min(prices.len());
        let mut rates = Vec::new();
        for i in w..=prices.len() {
            let p_start = prices[i - w];
            let p_end = prices[i - 1];
            if p_start > 0.0 {
                let log_return = (p_end / p_start).ln() / (w as f64);
                rates.push(log_return);
            }
        }
        rates
    }

    /// Predictability horizon: analogous to the causal horizon in cosmology.
    /// Models the distance beyond which correlations decay below significance.
    /// horizon = correlation_length / volatility
    pub fn horizon_distance(volatility: f64, correlation_length: f64) -> f64 {
        if volatility <= 0.0 {
            return f64::INFINITY;
        }
        correlation_length / volatility
    }

    /// Dark energy fraction: portion of total variance not explained by momentum/trend.
    /// Analogous to Ω_Λ — the "mysterious" component driving unexplained expansion.
    pub fn dark_energy_fraction(trend: f64, total_variance: f64) -> f64 {
        if total_variance <= 0.0 {
            return 0.0;
        }
        let explained = trend.powi(2);
        let unexplained = (total_variance - explained).max(0.0);
        unexplained / total_variance
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn planck_params() -> CosmologicalParams {
        CosmologicalParams::flat_lcdm(67.4, 0.315)
    }

    #[test]
    fn test_flat_lcdm() {
        let p = planck_params();
        assert!((p.omega_m + p.omega_lambda - 1.0).abs() < 1e-10);
        assert_eq!(p.omega_k, 0.0);
    }

    #[test]
    fn test_e_squared_at_zero() {
        let p = planck_params();
        // E²(0) = Ω_m + Ω_Λ = 1 for flat universe
        assert!((p.e_squared(0.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_luminosity_distance_positive() {
        let p = planck_params();
        let dl = CosmologicalCalculator::luminosity_distance(1.0, &p);
        assert!(dl > 0.0);
    }

    #[test]
    fn test_angular_diameter_distance_less_than_luminosity() {
        let p = planck_params();
        let z = 2.0;
        let da = CosmologicalCalculator::angular_diameter_distance(z, &p);
        let dl = CosmologicalCalculator::luminosity_distance(z, &p);
        // D_A = D_L / (1+z)²
        assert!(da < dl);
    }

    #[test]
    fn test_lookback_time_positive() {
        let p = planck_params();
        let t = CosmologicalCalculator::lookback_time(1.0, &p);
        assert!(t > 0.0);
    }

    #[test]
    fn test_market_expansion_rate() {
        let prices: Vec<f64> = (1..=10).map(|i| i as f64 * 10.0).collect();
        let rates = MarketCosmology::market_expansion_rate(&prices, 3);
        assert!(!rates.is_empty());
        // All rates should be positive for monotonically increasing prices
        for r in &rates {
            assert!(*r > 0.0);
        }
    }

    #[test]
    fn test_dark_energy_fraction_bounds() {
        let frac = MarketCosmology::dark_energy_fraction(0.5, 1.0);
        assert!((0.0..=1.0).contains(&frac));
    }
}
