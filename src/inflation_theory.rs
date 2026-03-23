//! Cosmic Inflation Theory: slow-roll, power spectrum, CMB predictions

use std::f64::consts::PI;

/// Reduced Planck mass M_Pl = 1 (natural units)
/// Hubble constant during inflation H

/// Slow-roll parameters from potential V(phi)
pub struct SlowRollParameters {
    pub epsilon: f64, // ε = (M_Pl^2/2)(V'/V)^2  — flatness condition ε << 1
    pub eta: f64,     // η = M_Pl^2 * V''/V      — curvature condition |η| << 1
}

impl SlowRollParameters {
    /// Compute from potential and its derivatives (M_Pl = 1)
    pub fn from_potential(v: f64, dv: f64, d2v: f64) -> Self {
        let epsilon = 0.5 * (dv / v).powi(2);
        let eta = d2v / v;
        SlowRollParameters { epsilon, eta }
    }

    pub fn is_slow_rolling(&self) -> bool {
        self.epsilon < 1.0 && self.eta.abs() < 1.0
    }

    /// Number of e-folds from current field value to end of inflation
    /// N ~ (1/M_Pl^2) * integral V/V' dphi
    /// For chaotic inflation V = m^2 phi^2/2: N ~ phi^2/4 - phi_end^2/4
    pub fn efolds_chaotic(phi_start: f64, phi_end: f64) -> f64 {
        (phi_start.powi(2) - phi_end.powi(2)) / 4.0
    }
}

/// Primordial power spectrum of scalar perturbations
/// P_R(k) = (H^2 / (8 pi^2 epsilon)) evaluated at k = aH
#[derive(Debug, Clone)]
pub struct PowerSpectrum {
    pub amplitude: f64,        // A_s ~ 2.1e-9
    pub spectral_index: f64,   // n_s ~ 0.965
    pub pivot_scale: f64,      // k_* = 0.05 Mpc^{-1}
    pub tensor_to_scalar: f64, // r = 16*epsilon
}

impl PowerSpectrum {
    pub fn new(epsilon: f64, eta: f64, h_inf: f64) -> Self {
        let amplitude = h_inf.powi(2) / (8.0 * PI * PI * epsilon);
        let spectral_index = 1.0 - 6.0 * epsilon + 2.0 * eta;
        PowerSpectrum {
            amplitude,
            spectral_index,
            pivot_scale: 0.05,
            tensor_to_scalar: 16.0 * epsilon,
        }
    }

    /// Scale-dependent power spectrum P(k) = A_s * (k/k_*)^{n_s - 1}
    pub fn evaluate(&self, k: f64) -> f64 {
        self.amplitude * (k / self.pivot_scale).powf(self.spectral_index - 1.0)
    }

    /// CMB temperature anisotropy amplitude delta_T/T ~ sqrt(P_R) * transfer_function
    pub fn cmb_cl(&self, ell: u32) -> f64 {
        // Simplified: C_ell ~ A_s * (l*(l+1))^{(n_s-1)/2} * sachs_wolfe
        // Sachs-Wolfe plateau: C_ell ~ const for l < 200
        let l = ell as f64;
        let k_eff = l / 14000.0; // Mpc^{-1} rough conversion
        let sw_factor = if ell < 200 { 1.0 } else { (200.0 / l).powi(2) };
        self.evaluate(k_eff) * sw_factor / (l * (l + 1.0))
    }
}

/// Starobinsky (R^2) inflation model
pub struct StarobinskyModel {
    pub m: f64, // mass parameter ~ 1.3e13 GeV
}

impl StarobinskyModel {
    pub fn new(m: f64) -> Self { StarobinskyModel { m } }

    /// Potential in Einstein frame: V(phi) = (3/4) m^2 (1 - e^{-sqrt(2/3)*phi})^2
    pub fn potential(&self, phi: f64) -> f64 {
        let x = (-phi * (2.0_f64/3.0).sqrt()).exp();
        0.75 * self.m * self.m * (1.0 - x).powi(2)
    }

    pub fn dpotential(&self, phi: f64) -> f64 {
        let sqrt23 = (2.0_f64/3.0).sqrt();
        let x = (-phi * sqrt23).exp();
        0.75 * self.m * self.m * 2.0 * (1.0 - x) * sqrt23 * x
    }

    pub fn d2potential(&self, phi: f64) -> f64 {
        let sqrt23 = (2.0_f64/3.0).sqrt();
        let x = (-phi * sqrt23).exp();
        0.75 * self.m * self.m * 2.0 * sqrt23.powi(2) * x * (2.0 * x - 1.0)
    }

    pub fn slow_roll(&self, phi: f64) -> SlowRollParameters {
        let v = self.potential(phi);
        let dv = self.dpotential(phi);
        let d2v = self.d2potential(phi);
        SlowRollParameters::from_potential(v, dv, d2v)
    }

    /// Predictions: n_s = 1 - 2/N, r = 12/N^2
    pub fn cmb_predictions(&self, n_efolds: f64) -> (f64, f64) {
        let ns = 1.0 - 2.0 / n_efolds;
        let r = 12.0 / (n_efolds * n_efolds);
        (ns, r)
    }
}

/// Reheating after inflation
pub fn reheating_temperature(gamma: f64, g_star: f64) -> f64 {
    // T_reh = (90 / (pi^2 * g_*))^{1/4} * sqrt(Gamma * M_Pl)
    // In natural units with M_Pl = 2.4e18 GeV
    let m_pl = 2.4e18_f64;
    (90.0 / (PI * PI * g_star)).powf(0.25) * (gamma * m_pl).sqrt()
}
