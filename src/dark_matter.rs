//! Dark matter halo models and market analogs.
//!
//! Implements NFW and Hernquist density profiles, WIMP detection models,
//! and market analogues for hidden autocorrelation and rotation-curve anomalies.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Gravitational constant in units of (km/s)^2 * pc / M_sun
const G_KPC: f64 = 4.301e-3; // pc M_sun^-1 (km/s)^2
/// Proton mass in GeV/c^2
const PROTON_MASS_GEV: f64 = 0.938_272;

// ---------------------------------------------------------------------------
// NfwProfile
// ---------------------------------------------------------------------------

/// Navarro-Frenk-White dark matter density profile.
///
/// * `rho_0`    – central density normalisation (M_sun / pc^3)
/// * `r_s`      – scale radius (pc)
#[derive(Debug, Clone)]
pub struct NfwProfile {
    pub rho_0: f64,
    pub r_s: f64,
}

impl NfwProfile {
    /// Create a new NFW profile.
    pub fn new(rho_0: f64, r_s: f64) -> Self {
        Self { rho_0, r_s }
    }

    /// NFW density at radius `r` (pc).
    ///
    /// `rho(r) = rho_0 / ((r/r_s) * (1 + r/r_s)^2)`
    pub fn density(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return f64::INFINITY;
        }
        let x = r / self.r_s;
        self.rho_0 / (x * (1.0 + x).powi(2))
    }

    /// Mass enclosed within radius `r` via numerical integration (100 shells).
    ///
    /// `M(<r) = 4 pi int_0^r rho(r') r'^2 dr'`
    pub fn mass_enclosed(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return 0.0;
        }
        let n = 1000usize;
        let dr = r / n as f64;
        let mut mass = 0.0;
        for i in 0..n {
            let r_mid = (i as f64 + 0.5) * dr;
            mass += self.density(r_mid) * r_mid * r_mid * dr;
        }
        4.0 * PI * mass
    }

    /// Circular velocity at radius `r` (km/s).
    ///
    /// `v_c(r) = sqrt(G * M(<r) / r)`
    pub fn circular_velocity(&self, r: f64, g: f64) -> f64 {
        let m = self.mass_enclosed(r);
        if r <= 0.0 || m <= 0.0 {
            return 0.0;
        }
        (g * m / r).sqrt()
    }

    /// Mass-concentration relation.
    ///
    /// `c ≈ 7 * (M200 / 1e12)^(-0.1)`
    pub fn concentration(m200: f64) -> f64 {
        7.0 * (m200 / 1.0e12_f64).powf(-0.1)
    }
}

// ---------------------------------------------------------------------------
// HernquistProfile
// ---------------------------------------------------------------------------

/// Hernquist (1990) profile — a useful analytic model for bulges and ellipticals.
///
/// * `mass`         – total mass (M_sun)
/// * `scale_radius` – scale radius `a` (pc)
#[derive(Debug, Clone)]
pub struct HernquistProfile {
    pub mass: f64,
    pub scale_radius: f64,
}

impl HernquistProfile {
    /// Create a Hernquist profile.
    pub fn new(mass: f64, scale_radius: f64) -> Self {
        Self { mass, scale_radius }
    }

    /// Hernquist density at radius `r`.
    ///
    /// `rho(r) = M/(2*pi) * a / (r * (r+a)^3)`
    pub fn density(&self, r: f64) -> f64 {
        if r <= 0.0 {
            return f64::INFINITY;
        }
        let a = self.scale_radius;
        (self.mass / (2.0 * PI)) * a / (r * (r + a).powi(3))
    }

    /// Gravitational potential at radius `r`.
    ///
    /// `Phi(r) = -G * M / (r + a)`
    ///
    /// Returns in units where G is absorbed into the caller's conventions.
    pub fn potential(&self, r: f64) -> f64 {
        let a = self.scale_radius;
        -G_KPC * self.mass / (r + a)
    }
}

// ---------------------------------------------------------------------------
// WimpModel
// ---------------------------------------------------------------------------

/// Weakly Interacting Massive Particle (WIMP) dark matter model.
///
/// * `mass_gev`         – WIMP mass in GeV/c^2
/// * `cross_section_pb` – spin-independent cross section in picobarn (pb)
#[derive(Debug, Clone)]
pub struct WimpModel {
    pub mass_gev: f64,
    pub cross_section_pb: f64,
}

impl WimpModel {
    /// Construct a WIMP model.
    pub fn new(mass_gev: f64, cross_section_pb: f64) -> Self {
        Self { mass_gev, cross_section_pb }
    }

    /// Local number density of WIMPs given a DM mass density `rho_dm` (GeV/cm^3).
    ///
    /// `n = rho_dm / (m_chi * m_p)`  (approximate, using proton-mass normalisation)
    pub fn number_density(&self, rho_dm: f64) -> f64 {
        rho_dm / (self.mass_gev * PROTON_MASS_GEV)
    }

    /// Direct detection event rate (counts / kg / day).
    ///
    /// `R = n * sigma * v * N_target`
    ///
    /// * `density`  – local DM number density (cm^-3)
    /// * `velocity` – WIMP velocity (cm/s), typically ~220 km/s = 2.2e7 cm/s
    /// * `n_target` – number of target nuclei per kg of detector
    pub fn detection_rate(&self, density: f64, velocity: f64, n_target: f64) -> f64 {
        // Convert cross section from pb to cm^2: 1 pb = 1e-36 cm^2
        let sigma_cm2 = self.cross_section_pb * 1.0e-36;
        density * sigma_cm2 * velocity * n_target * 86_400.0 // per day
    }

    /// Upper limit on cross section from null-result exposure.
    ///
    /// Simple Poisson 90% CL: sigma_limit ~ 2.3 / (exposure * R_background)
    pub fn direct_detection_limit(&self, exposure: f64, background: f64) -> f64 {
        if exposure <= 0.0 || background <= 0.0 {
            return f64::INFINITY;
        }
        2.3 / (exposure * background)
    }
}

// ---------------------------------------------------------------------------
// DarkEnergyMarket
// ---------------------------------------------------------------------------

/// Market analogues of dark matter / dark energy concepts.
///
/// Treats "missing" predictive power in return models as analogous to the
/// mass-to-light ratio discrepancy in galactic dynamics.
#[derive(Debug, Clone, Default)]
pub struct DarkEnergyMarket;

impl DarkEnergyMarket {
    /// Fraction of return variance explained by "hidden" autocorrelation.
    ///
    /// Fits an AR(1) model and returns the unexplained residual fraction —
    /// the "dark matter" of the return series.
    pub fn dark_matter_fraction(returns: &[f64]) -> f64 {
        let n = returns.len();
        if n < 4 {
            return 0.0;
        }
        let mean = returns.iter().sum::<f64>() / n as f64;
        let var_total: f64 = returns.iter().map(|&r| (r - mean).powi(2)).sum::<f64>() / n as f64;
        if var_total == 0.0 {
            return 0.0;
        }
        // AR(1) coefficient via OLS
        let demeaned: Vec<f64> = returns.iter().map(|&r| r - mean).collect();
        let cov: f64 = demeaned[..n - 1].iter().zip(&demeaned[1..]).map(|(a, b)| a * b).sum::<f64>() / (n - 1) as f64;
        let var_lag: f64 = demeaned[..n - 1].iter().map(|a| a * a).sum::<f64>() / (n - 1) as f64;
        let phi = if var_lag > 0.0 { cov / var_lag } else { 0.0 };
        // Variance explained by AR(1): phi^2 * var
        let explained = phi.powi(2) * var_total;
        let dark = (var_total - explained).max(0.0) / var_total;
        dark
    }

    /// Velocity dispersion analog — the "rotation curve" of a market-cap series.
    ///
    /// Returns the RMS velocity (rolling std) of the log-market-cap changes at radius `r`
    /// (interpreted as a smoothing scale index from 0 to 1).
    pub fn rotation_curve_analog(market_cap_history: &[f64], r: f64) -> f64 {
        let n = market_cap_history.len();
        if n < 2 {
            return 0.0;
        }
        // r in [0,1] selects a sub-window
        let window = ((r.clamp(0.01, 1.0)) * n as f64).ceil() as usize;
        let slice = &market_cap_history[n.saturating_sub(window)..];
        if slice.len() < 2 {
            return 0.0;
        }
        let returns: Vec<f64> = slice.windows(2)
            .map(|w| (w[1] / w[0].max(1e-12)).ln())
            .collect();
        let mean = returns.iter().sum::<f64>() / returns.len() as f64;
        let var = returns.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / returns.len() as f64;
        var.sqrt()
    }

    /// Regime concentration parameter — analogous to NFW concentration index.
    ///
    /// High volatility + low drift → concentrated (cuspy) regime.
    pub fn halo_concentration(volatility: f64, drift: f64) -> f64 {
        if volatility <= 0.0 {
            return 0.0;
        }
        // Sharpe-like ratio, inverted and scaled
        let sharpe = drift.abs() / volatility;
        // High Sharpe = spread-out (cored) profile; low Sharpe = concentrated (cuspy)
        7.0 * (-sharpe).exp()
    }

    /// Missing mass: the unaccounted return needed to reconcile observed vs predicted.
    ///
    /// Analogous to the mass discrepancy in rotation curves.
    pub fn missing_mass(
        observed_returns: f64,
        predicted_returns: f64,
        luminous_fraction: f64,
    ) -> f64 {
        let discrepancy = observed_returns - predicted_returns * luminous_fraction.clamp(0.0, 1.0);
        discrepancy
    }
}
