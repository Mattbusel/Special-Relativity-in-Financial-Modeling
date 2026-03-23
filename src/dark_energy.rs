//! Dark energy analogy for hidden forces expanding market volatility.
//!
//! In cosmology, dark energy is the mysterious force causing the accelerating
//! expansion of the universe. Here we model analogous hidden forces that drive
//! the sustained and accelerating expansion of market volatility regimes.

// ---------------------------------------------------------------------------
// DarkEnergyField
// ---------------------------------------------------------------------------

/// A dark energy field characterised by its density and equation-of-state
/// parameter w.
///
/// - w = -1: cosmological constant (constant dark energy)
/// - w < -1/3: causes accelerating expansion
/// - w < -1: phantom dark energy (diverges in finite time — "big rip")
#[derive(Debug, Clone)]
pub struct DarkEnergyField {
    /// Energy density (ρ > 0).
    pub energy_density: f64,
    /// Equation-of-state parameter w = P / ρ.
    pub equation_of_state: f64,
}

impl DarkEnergyField {
    /// Create a cosmological-constant-style dark energy field (w = -1).
    pub fn cosmological_constant(density: f64) -> Self {
        Self {
            energy_density: density,
            equation_of_state: -1.0,
        }
    }

    /// Create a quintessence field with time-varying w (w > -1).
    pub fn quintessence(density: f64, w: f64) -> Self {
        assert!(w > -1.0, "quintessence requires w > -1");
        Self {
            energy_density: density,
            equation_of_state: w,
        }
    }

    /// Create a phantom field (w < -1 — leads to big rip).
    pub fn phantom(density: f64, w: f64) -> Self {
        assert!(w < -1.0, "phantom field requires w < -1");
        Self {
            energy_density: density,
            equation_of_state: w,
        }
    }
}

// ---------------------------------------------------------------------------
// CosmologicalConstant
// ---------------------------------------------------------------------------

/// A fixed cosmological constant Λ driving uniform expansion.
#[derive(Debug, Clone)]
pub struct CosmologicalConstant {
    /// Λ > 0 = repulsive (drives expansion).
    pub lambda: f64,
}

impl CosmologicalConstant {
    pub fn new(lambda: f64) -> Self {
        Self { lambda }
    }

    /// Check whether this constant causes accelerating expansion.
    pub fn is_accelerating(&self) -> bool {
        self.lambda > 0.0
    }
}

// ---------------------------------------------------------------------------
// VolatilityExpansion
// ---------------------------------------------------------------------------

/// Hubble-like expansion model for volatility regimes.
pub struct VolatilityExpansion;

impl VolatilityExpansion {
    /// Compute the "Hubble" expansion rate of volatility.
    ///
    /// Analogous to H = sqrt(Λ / 3) in de Sitter space:
    /// H = sqrt(vol * lambda)
    pub fn expansion_rate(dark_energy: &DarkEnergyField, current_vol: f64) -> f64 {
        let lambda = dark_energy.energy_density * (-dark_energy.equation_of_state);
        (current_vol * lambda).abs().sqrt()
    }

    /// Estimate the "age" of this volatility regime using the analogy
    /// to the Hubble time: T = 1 / H * ln(vol_current / vol_initial).
    ///
    /// Returns 0.0 if the regime has not expanded (vol_current <= vol_initial).
    pub fn age_of_volatility_regime(
        dark_energy: &DarkEnergyField,
        current_vol: f64,
        initial_vol: f64,
    ) -> f64 {
        if current_vol <= initial_vol || initial_vol <= 0.0 {
            return 0.0;
        }
        let h = Self::expansion_rate(dark_energy, current_vol);
        if h < 1e-12 {
            return 0.0;
        }
        (current_vol / initial_vol).ln() / h
    }

    /// True if the equation-of-state parameter crosses -1 (phantom crossing).
    /// Phantom dark energy (w < -1) leads to a volatility "big rip".
    pub fn phantom_crossing(w: f64) -> bool {
        w < -1.0
    }

    /// Model quintessence decay: dark energy density decays over time as
    /// ρ(t) = ρ₀ * exp((1 + w) * t).
    ///
    /// For w = -1 this is constant; for w > -1 it decays; for w < -1 it grows.
    pub fn quintessence_decay(field: &DarkEnergyField, time: f64) -> f64 {
        field.energy_density * ((1.0 + field.equation_of_state) * time).exp()
    }
}

// ---------------------------------------------------------------------------
// DarkMatter
// ---------------------------------------------------------------------------

/// Hidden market factor analogous to dark matter — influences behaviour
/// through gravity-like effects but is not directly observable.
#[derive(Debug, Clone)]
pub struct DarkMatter {
    /// Effective density of the hidden factor.
    pub density: f64,
    /// Interaction cross-section with visible market flows.
    pub interaction_cross_section: f64,
}

impl DarkMatter {
    pub fn new(density: f64, interaction_cross_section: f64) -> Self {
        Self {
            density,
            interaction_cross_section,
        }
    }

    /// Gravitational lensing analogy: the hidden factor amplifies
    /// observed volatility.
    ///
    /// lensed_vol = observed_vol * (1 + κ * density)
    /// where κ = interaction_cross_section (convergence factor).
    pub fn gravitational_lensing_effect(
        dark_matter: &DarkMatter,
        observed_vol: f64,
    ) -> f64 {
        let convergence = dark_matter.interaction_cross_section * dark_matter.density;
        observed_vol * (1.0 + convergence)
    }

    /// Estimate the implied hidden correlation factor from visible returns.
    ///
    /// Uses the dark matter density as a scaling factor on the mean-squared
    /// return to approximate the fraction of variance attributable to hidden
    /// systemic risk.
    ///
    /// Returns a value in [0, 1].
    pub fn hidden_correlation(visible_returns: &[f64], dark_matter: &DarkMatter) -> f64 {
        if visible_returns.is_empty() {
            return 0.0;
        }
        let mean = visible_returns.iter().sum::<f64>() / visible_returns.len() as f64;
        let variance = visible_returns
            .iter()
            .map(|r| (r - mean).powi(2))
            .sum::<f64>()
            / visible_returns.len() as f64;

        let hidden_variance = dark_matter.density * dark_matter.interaction_cross_section;
        let total = variance + hidden_variance;
        if total < 1e-12 {
            return 0.0;
        }
        (hidden_variance / total).clamp(0.0, 1.0)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_expansion_rate_positive() {
        let field = DarkEnergyField::cosmological_constant(1.0);
        let rate = VolatilityExpansion::expansion_rate(&field, 0.2);
        assert!(rate > 0.0);
    }

    #[test]
    fn test_expansion_rate_zero_vol() {
        let field = DarkEnergyField::cosmological_constant(1.0);
        let rate = VolatilityExpansion::expansion_rate(&field, 0.0);
        assert_eq!(rate, 0.0);
    }

    #[test]
    fn test_age_of_volatility_regime_positive() {
        let field = DarkEnergyField::cosmological_constant(1.0);
        let age = VolatilityExpansion::age_of_volatility_regime(&field, 0.4, 0.1);
        assert!(age > 0.0);
    }

    #[test]
    fn test_age_of_volatility_no_expansion() {
        let field = DarkEnergyField::cosmological_constant(1.0);
        let age = VolatilityExpansion::age_of_volatility_regime(&field, 0.1, 0.4);
        assert_eq!(age, 0.0);
    }

    #[test]
    fn test_phantom_crossing_true() {
        assert!(VolatilityExpansion::phantom_crossing(-1.5));
    }

    #[test]
    fn test_phantom_crossing_false_at_minus_one() {
        assert!(!VolatilityExpansion::phantom_crossing(-1.0));
    }

    #[test]
    fn test_phantom_crossing_false_quintessence() {
        assert!(!VolatilityExpansion::phantom_crossing(-0.7));
    }

    #[test]
    fn test_quintessence_decay_constant_at_minus_one() {
        let field = DarkEnergyField::cosmological_constant(2.0);
        // w = -1 => decay factor = exp(0) = 1
        let rho = VolatilityExpansion::quintessence_decay(&field, 100.0);
        assert!((rho - 2.0).abs() < 1e-9);
    }

    #[test]
    fn test_quintessence_decay_decreases_for_w_gt_minus_one() {
        let field = DarkEnergyField::quintessence(1.0, -0.5);
        let rho_late = VolatilityExpansion::quintessence_decay(&field, 10.0);
        let rho_early = VolatilityExpansion::quintessence_decay(&field, 0.0);
        assert!(rho_late > rho_early, "quintessence with w>-1 grows in density");
    }

    #[test]
    fn test_cosmological_constant_is_accelerating() {
        let cc = CosmologicalConstant::new(0.7);
        assert!(cc.is_accelerating());
    }

    #[test]
    fn test_cosmological_constant_negative_not_accelerating() {
        let cc = CosmologicalConstant::new(-0.1);
        assert!(!cc.is_accelerating());
    }

    #[test]
    fn test_gravitational_lensing_amplifies() {
        let dm = DarkMatter::new(1.0, 0.5);
        let lensed = DarkMatter::gravitational_lensing_effect(&dm, 0.2);
        assert!(lensed > 0.2);
    }

    #[test]
    fn test_hidden_correlation_empty() {
        let dm = DarkMatter::new(1.0, 0.5);
        assert_eq!(DarkMatter::hidden_correlation(&[], &dm), 0.0);
    }

    #[test]
    fn test_hidden_correlation_in_range() {
        let dm = DarkMatter::new(0.5, 0.3);
        let returns = vec![0.01, -0.02, 0.015, -0.005, 0.02];
        let corr = DarkMatter::hidden_correlation(&returns, &dm);
        assert!(corr >= 0.0 && corr <= 1.0);
    }

    #[test]
    fn test_phantom_field_density_grows() {
        let field = DarkEnergyField::phantom(1.0, -1.5);
        let rho_late = VolatilityExpansion::quintessence_decay(&field, 5.0);
        let rho_now = VolatilityExpansion::quintessence_decay(&field, 0.0);
        assert!(rho_late > rho_now, "phantom energy density grows over time");
    }
}
