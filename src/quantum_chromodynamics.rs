//! Quantum Chromodynamics: color algebra, running coupling, asymptotic freedom

use std::f64::consts::PI;

/// SU(3) color algebra constants
pub const N_C: f64 = 3.0;      // number of colors
pub const N_F: f64 = 6.0;      // number of quark flavors
pub const C_F: f64 = 4.0 / 3.0; // Casimir of fundamental representation
pub const C_A: f64 = 3.0;       // Casimir of adjoint representation
pub const T_F: f64 = 0.5;       // trace normalization

/// Beta function coefficients for QCD
pub const BETA_0: f64 = (11.0 * C_A - 4.0 * T_F * N_F) / (4.0 * PI);
pub const BETA_1: f64 = (34.0 * C_A * C_A - 20.0 * C_A * T_F * N_F - 12.0 * C_F * T_F * N_F) / (16.0 * PI * PI);

/// One-loop running coupling alpha_s(Q^2)
/// alpha_s(Q^2) = alpha_s(mu^2) / (1 + alpha_s(mu^2) * beta0 * ln(Q^2/mu^2))
pub fn running_coupling_1loop(alpha_s_ref: f64, q_sq: f64, mu_sq: f64) -> f64 {
    let log_ratio = (q_sq / mu_sq).ln();
    let b0 = (11.0 * C_A - 4.0 * T_F * N_F) / (12.0 * PI);
    let denom = 1.0 + alpha_s_ref * b0 * log_ratio;
    if denom <= 0.0 { return 0.0; } // Landau pole
    alpha_s_ref / denom
}

/// Two-loop running coupling (implicit, solved iteratively)
pub fn running_coupling_2loop(alpha_s_ref: f64, q_sq: f64, mu_sq: f64, n_iter: u32) -> f64 {
    // Start with one-loop estimate
    let mut alpha = running_coupling_1loop(alpha_s_ref, q_sq, mu_sq);
    let t = (q_sq / mu_sq).ln();

    for _ in 0..n_iter {
        let b0 = (11.0 * N_C - 2.0 * N_F) / (12.0 * PI);
        let b1 = (102.0 * N_C - 38.0 * N_F / 3.0) / (24.0 * PI * PI);
        // RGE: d alpha / d ln(Q^2) = -b0 * alpha^2 - b1 * alpha^3
        let beta = -b0 * alpha * alpha - b1 * alpha * alpha * alpha;
        // Newton step for implicit equation
        alpha = alpha + beta * t * 0.1;
        if alpha < 0.0 { alpha = 0.01; }
    }
    alpha
}

/// QCD potential (Cornell potential): V(r) = -4/3 * alpha_s/r + sigma*r
pub struct QcdPotential {
    pub alpha_s: f64,
    pub string_tension: f64, // sigma ~ 0.18 GeV^2 ≈ 0.9 GeV/fm
}

impl QcdPotential {
    pub fn new(alpha_s: f64, sigma: f64) -> Self {
        QcdPotential { alpha_s, string_tension: sigma }
    }

    /// V(r) in units where r is in GeV^{-1}
    pub fn potential(&self, r: f64) -> f64 {
        if r < 1e-10 { return f64::INFINITY; }
        -C_F * self.alpha_s / r + self.string_tension * r
    }

    /// r where confinement kicks in (dV/dr = 0 implies confinement for r > r_conf)
    /// Minimum at r_min = sqrt(C_F * alpha_s / sigma)
    pub fn confinement_radius(&self) -> f64 {
        (C_F * self.alpha_s / self.string_tension).sqrt()
    }

    /// Flux tube string tension in MeV/fm units (conversion factor)
    pub fn string_tension_mev_fm(&self) -> f64 {
        self.string_tension * 197.3 // ℏc = 197.3 MeV·fm
    }
}

/// Asymptotic freedom: alpha_s → 0 as Q → ∞
pub fn is_asymptotically_free(n_colors: u32, n_flavors: u32) -> bool {
    // Condition: 11*N_c > 2*N_f
    11 * n_colors > 2 * n_flavors
}

/// Color factor for quark-gluon vertex
pub fn quark_gluon_vertex_factor() -> f64 {
    C_F
}

/// DIS structure function F2 at leading order (quark-parton model)
/// F2(x) = sum_q e_q^2 * x * (q(x) + qbar(x))
/// Using simple toy PDF: q(x) = A * x^{-1/2} * (1-x)^3
pub fn toy_pdf(x: f64, a: f64) -> f64 {
    if x <= 0.0 || x >= 1.0 { return 0.0; }
    a * x.powf(-0.5) * (1.0 - x).powi(3)
}

pub fn f2_structure_function(x: f64) -> f64 {
    // Up quark: charge 2/3, down quark: charge 1/3
    let e_u_sq = 4.0 / 9.0;
    let e_d_sq = 1.0 / 9.0;
    let q_u = toy_pdf(x, 1.8);
    let q_d = toy_pdf(x, 2.5);
    x * (e_u_sq * (q_u + q_u * 0.5) + e_d_sq * (q_d + q_d * 0.3))
}
