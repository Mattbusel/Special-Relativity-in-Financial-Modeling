//! Topological defects: cosmic strings, domain walls, monopoles

use std::f64::consts::PI;

/// Kibble mechanism: defect density from phase transition
/// n_defect ~ 1 / xi^D where xi = correlation length at formation
pub fn kibble_defect_density(correlation_length_mpc: f64, dimension: u32) -> f64 {
    correlation_length_mpc.powi(-(dimension as i32))
}

/// Cosmic string network
pub struct CosmicString {
    pub mass_per_length: f64, // Gmu (string tension in Planck units * G)
    pub velocity: f64,         // typical velocity ~ 0.3c (units of c)
}

impl CosmicString {
    pub fn new(gmu: f64) -> Self {
        CosmicString { mass_per_length: gmu, velocity: 0.3 }
    }

    /// Angle deficit around a cosmic string: delta = 8 pi G mu
    pub fn angle_deficit(&self) -> f64 {
        8.0 * PI * self.mass_per_length
    }

    /// CMB temperature anisotropy from string: delta_T/T ~ 4 pi G mu v
    pub fn cmb_anisotropy(&self) -> f64 {
        4.0 * PI * self.mass_per_length * self.velocity
    }

    /// String loop oscillation frequency: f ~ c / L (L = loop length)
    pub fn loop_frequency(&self, loop_length_pc: f64) -> f64 {
        // In natural units, c = 1, f ~ 1/L
        1.0 / loop_length_pc
    }

    /// Gravitational wave power from loop: P_GW = Gamma * G mu^2 * c (Gamma ~ 50)
    pub fn gw_power(&self) -> f64 {
        let gamma = 50.0;
        gamma * self.mass_per_length * self.mass_per_length
    }

    /// One-scale model network density: rho_string ~ mu / t^2
    pub fn network_energy_density(&self, time_gyr: f64) -> f64 {
        // Hubble time in appropriate units
        let t_sec = time_gyr * 3.156e16_f64;
        self.mass_per_length / (t_sec * t_sec)
    }
}

/// Domain wall from discrete symmetry breaking
pub struct DomainWall {
    pub surface_tension: f64, // sigma (energy per area)
    pub thickness_mpc: f64,
}

impl DomainWall {
    pub fn new(eta_gev: f64, lambda: f64) -> Self {
        // sigma ~ eta^3 / m_phi^3 * m_phi ~ lambda^{1/2} eta^3
        let surface_tension = lambda.sqrt() * eta_gev.powi(3);
        let thickness = 1.0 / (lambda.sqrt() * eta_gev);
        DomainWall { surface_tension, thickness_mpc: thickness * 6.24e24 } // convert to cosmological units
    }

    /// Domain wall domination condition: rho_wall > rho_crit
    /// Dangerous if wall forms before matter domination
    pub fn dominates_universe(&self, hubble_h: f64) -> bool {
        // rho_wall ~ sigma / t, rho_crit ~ H^2 M_pl^2
        let rho_crit = 3.0 * hubble_h * hubble_h * (2.4e18_f64).powi(2);
        self.surface_tension > rho_crit
    }

    /// Pressure on wall from temperature gradient: P = sigma * kappa (curvature)
    pub fn pressure(&self, curvature_radius_mpc: f64) -> f64 {
        self.surface_tension / curvature_radius_mpc
    }
}

/// Magnetic monopole ('t Hooft-Polyakov)
pub struct MagneticMonopole {
    pub mass_gev: f64,
    pub magnetic_charge: i32, // in units of Dirac charge g = 2pi/e
}

impl MagneticMonopole {
    pub fn new(mw_gev: f64, g_coupling: f64) -> Self {
        // Mass ~ M_W / alpha_GUT ~ M_W * (4pi/g^2)
        let mass = mw_gev * 4.0 * PI / (g_coupling * g_coupling);
        MagneticMonopole { mass_gev: mass, magnetic_charge: 1 }
    }

    /// Dirac quantization: e * g = 2 pi n
    pub fn dirac_charge(&self, electric_charge: f64) -> f64 {
        2.0 * PI * self.magnetic_charge as f64 / electric_charge
    }

    /// Parker bound: monopole flux from not disrupting galactic magnetic field
    /// F_max ~ 10^{-15} cm^{-2} s^{-1} sr^{-1}
    pub fn parker_bound() -> f64 { 1e-15 }

    /// Catalysis cross section (Callan-Rubakov effect): sigma ~ pi r_core^2
    pub fn catalysis_cross_section(&self) -> f64 {
        let r_core = 1.0 / self.mass_gev; // in GeV^{-1}
        PI * r_core * r_core
    }

    /// Inflation dilutes monopoles: rho_monopole / rho_photon ~ exp(-2*N_e)
    pub fn post_inflation_density(initial_density: f64, n_efolds: f64) -> f64 {
        initial_density * (-3.0 * n_efolds).exp()
    }
}

/// Texture: global topological defect (no gauge field)
pub struct Texture {
    pub symmetry_breaking_scale_gev: f64,
}

impl Texture {
    pub fn new(eta: f64) -> Self { Texture { symmetry_breaking_scale_gev: eta } }

    /// Textures collapse and form "unwinding" events
    /// Fraction of energy released as gravitational waves: f_gw ~ G mu_eff
    pub fn gw_fraction(&self) -> f64 {
        let g_newton = 6.674e-39_f64; // in GeV^{-2}
        g_newton * self.symmetry_breaking_scale_gev.powi(2)
    }

    pub fn cmb_hot_spot_amplitude(&self) -> f64 {
        // delta_T/T ~ 4 pi G sigma ~ G eta^2
        let g_newton = 6.674e-39_f64;
        4.0 * PI * g_newton * self.symmetry_breaking_scale_gev.powi(2)
    }
}
