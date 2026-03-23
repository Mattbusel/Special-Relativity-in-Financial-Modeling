//! Supersymmetry: SUSY algebra, superpartner spectrum, soft breaking terms

use std::f64::consts::PI;

/// SUSY algebra: {Q_alpha, Q_bar_beta} = 2 sigma^mu_{alpha beta} P_mu
/// Supercharge squares to momentum
pub struct SusyAlgebra {
    pub n_supercharges: u32, // N=1, N=2, N=4 SUSY
}

impl SusyAlgebra {
    pub fn new(n: u32) -> Self { SusyAlgebra { n_supercharges: n } }

    /// Mass degeneracy: bosons and fermions in same supermultiplet have equal mass
    pub fn mass_degeneracy(&self, boson_mass: f64) -> f64 { boson_mass }

    /// Number of bosonic vs fermionic degrees of freedom (must be equal in unbroken SUSY)
    pub fn degrees_of_freedom(&self) -> (u32, u32) {
        // N=1 chiral: 2 bosonic (Re, Im of scalar), 2 fermionic (Weyl fermion)
        // N=1 vector: 2 bosonic (gaugino spin-1/2 = 2 d.o.f.), 2 fermionic
        (2 * self.n_supercharges, 2 * self.n_supercharges)
    }

    /// Witten index: Tr[(-1)^F] — nonzero implies unbroken SUSY
    pub fn witten_index(&self, n_boson_states: i32, n_fermion_states: i32) -> i32 {
        n_boson_states - n_fermion_states
    }
}

/// Standard Model superpartner spectrum (MSSM)
#[derive(Debug, Clone)]
pub struct Superpartner {
    pub sm_particle: String,
    pub susy_partner: String,
    pub spin_sm: f64,
    pub spin_susy: f64,
    pub mass_lower_bound_gev: f64, // from LEP/LHC constraints
}

pub fn mssm_spectrum() -> Vec<Superpartner> {
    vec![
        Superpartner { sm_particle: "electron".into(), susy_partner: "selectron".into(), spin_sm: 0.5, spin_susy: 0.0, mass_lower_bound_gev: 107.0 },
        Superpartner { sm_particle: "muon".into(), susy_partner: "smuon".into(), spin_sm: 0.5, spin_susy: 0.0, mass_lower_bound_gev: 94.0 },
        Superpartner { sm_particle: "up quark".into(), susy_partner: "sup squark".into(), spin_sm: 0.5, spin_susy: 0.0, mass_lower_bound_gev: 800.0 },
        Superpartner { sm_particle: "gluon".into(), susy_partner: "gluino".into(), spin_sm: 1.0, spin_susy: 0.5, mass_lower_bound_gev: 1800.0 },
        Superpartner { sm_particle: "W boson".into(), susy_partner: "wino".into(), spin_sm: 1.0, spin_susy: 0.5, mass_lower_bound_gev: 103.5 },
        Superpartner { sm_particle: "Z boson".into(), susy_partner: "zino/bino".into(), spin_sm: 1.0, spin_susy: 0.5, mass_lower_bound_gev: 46.0 },
        Superpartner { sm_particle: "Higgs h".into(), susy_partner: "higgsino".into(), spin_sm: 0.0, spin_susy: 0.5, mass_lower_bound_gev: 103.5 },
        Superpartner { sm_particle: "graviton".into(), susy_partner: "gravitino".into(), spin_sm: 2.0, spin_susy: 1.5, mass_lower_bound_gev: 1e-12 },
    ]
}

/// Soft SUSY breaking mass terms (preserve SUSY at loop level)
pub struct SoftBreakingTerms {
    pub m_half: f64,  // gaugino mass
    pub m_0: f64,     // scalar mass
    pub a_0: f64,     // trilinear coupling
    pub tan_beta: f64, // ratio of Higgs vevs
    pub sign_mu: i32,  // sign of mu parameter
}

impl SoftBreakingTerms {
    pub fn new(m_half: f64, m_0: f64, a_0: f64, tan_beta: f64) -> Self {
        SoftBreakingTerms { m_half, m_0, a_0, tan_beta, sign_mu: 1 }
    }

    /// mSUGRA/CMSSM: all soft terms parameterized by (m_0, m_1/2, A_0, tan_beta, sign_mu)
    /// Gaugino masses at low scale: M_1 ~ 0.4*m_half, M_2 ~ 0.8*m_half, M_3 ~ 2.7*m_half
    pub fn gaugino_masses(&self) -> (f64, f64, f64) {
        (0.41 * self.m_half, 0.82 * self.m_half, 2.7 * self.m_half)
    }

    /// LSP (lightest SUSY particle) mass estimate — usually bino-like neutralino
    pub fn lsp_mass_estimate(&self) -> f64 {
        let (m1, m2, _) = self.gaugino_masses();
        // Lightest neutralino ~ bino-like for M_1 < M_2, mu
        m1.min(m2) * 0.9
    }

    /// SUSY breaking scale: F^{1/2} ~ sqrt(m_soft * M_pl)
    pub fn susy_breaking_scale_gev(&self) -> f64 {
        let m_pl = 2.4e18_f64; // Planck mass in GeV
        (self.m_0 * m_pl).sqrt()
    }

    /// Fine-tuning measure: Delta = max |d ln M_Z^2 / d ln param^2|
    pub fn fine_tuning(&self) -> f64 {
        let m_z = 91.2_f64;
        // Simplified: main contribution from stop mass to Higgs mass
        let m_stop = self.m_0; // rough estimate
        2.0 * m_stop * m_stop / (m_z * m_z) // naturalness: want < ~10-100
    }
}

/// Dark matter relic density estimate (bino-like LSP)
pub fn relic_density_estimate(lsp_mass_gev: f64) -> f64 {
    // Omega_chi * h^2 ~ 0.1 * (M_chi / 100 GeV)^2 / (sigma_ann / 3e-26 cm^3/s)
    // For bino: sigma_ann ~ alpha^2 / M_chi^2 * (M_chi^2/m_sfermion^4)
    // Rough estimate: Omega h^2 ~ 0.1 * (M_chi / 100)^2
    let _ = PI; // suppress unused import warning
    0.1 * (lsp_mass_gev / 100.0).powi(2)
}
