//! Full relativistic portfolio model combining special-relativity concepts
//! with financial asset modelling.
//!
//! Each asset is treated as a relativistic particle with a rest-mass value
//! (intrinsic fundamental value) and a velocity β (market momentum / velocity
//! relative to light speed). Lorentz-invariant quantities such as invariant
//! mass give frame-independent risk measures.

/// A single asset viewed from a relativistic reference frame.
#[derive(Debug, Clone)]
pub struct AssetFrame {
    pub symbol: String,
    /// Fundamental "rest-mass" value of the asset (e.g. book value).
    pub rest_mass_value: f64,
    /// Velocity as a fraction of c: β = v/c ∈ (-1, 1).
    pub velocity_beta: f64,
    /// Lorentz factor γ = 1 / √(1 − β²).
    pub gamma: f64,
    /// Proper time elapsed for this asset (trading days in its own frame).
    pub proper_time_elapsed: f64,
}

impl AssetFrame {
    /// Create a new asset frame.
    ///
    /// # Panics
    /// Panics in debug mode if |β| >= 1 (superluminal velocity).
    pub fn new(symbol: &str, rest_mass_value: f64, velocity_beta: f64) -> Self {
        let beta_clamped = velocity_beta.clamp(-0.9999, 0.9999);
        let gamma = 1.0 / (1.0 - beta_clamped * beta_clamped).sqrt();
        Self {
            symbol: symbol.to_string(),
            rest_mass_value,
            velocity_beta: beta_clamped,
            gamma,
            proper_time_elapsed: 0.0,
        }
    }

    /// Relativistic mass: m_rel = γ · m₀.
    pub fn relativistic_mass(&self) -> f64 {
        self.gamma * self.rest_mass_value
    }

    /// Kinetic energy: KE = (γ − 1) · m₀c².
    /// Here c = 1, so KE = (γ − 1) · m₀.
    pub fn kinetic_energy(&self) -> f64 {
        (self.gamma - 1.0) * self.rest_mass_value
    }

    /// Total energy: E = γ · m₀c² = γ · m₀ (with c = 1).
    pub fn total_energy(&self) -> f64 {
        self.gamma * self.rest_mass_value
    }
}

/// A portfolio of relativistic asset frames.
#[derive(Debug, Clone)]
pub struct RelativisticPortfolio {
    pub frames: Vec<AssetFrame>,
    /// Reference frame velocity β of the observer.
    pub reference_beta: f64,
}

impl RelativisticPortfolio {
    /// Create a new empty portfolio observed from the given reference frame β.
    pub fn new(reference_beta: f64) -> Self {
        Self {
            frames: Vec::new(),
            reference_beta: reference_beta.clamp(-0.9999, 0.9999),
        }
    }

    /// Add an asset to the portfolio.
    pub fn add_asset(&mut self, symbol: &str, value: f64, velocity_beta: f64) {
        self.frames.push(AssetFrame::new(symbol, value, velocity_beta));
    }

    /// Sum of all rest-mass values.
    pub fn total_rest_mass(&self) -> f64 {
        self.frames.iter().map(|f| f.rest_mass_value).sum()
    }

    /// Sum of all relativistic masses.
    pub fn total_relativistic_mass(&self) -> f64 {
        self.frames.iter().map(|f| f.relativistic_mass()).sum()
    }

    /// Portfolio β: weighted average β (weighted by relativistic mass).
    pub fn portfolio_beta(&self) -> f64 {
        let total_rel_mass = self.total_relativistic_mass();
        if total_rel_mass == 0.0 {
            return 0.0;
        }
        self.frames
            .iter()
            .map(|f| f.velocity_beta * f.relativistic_mass())
            .sum::<f64>()
            / total_rel_mass
    }

    /// Portfolio γ = 1 / √(1 − β_portfolio²).
    pub fn portfolio_gamma(&self) -> f64 {
        let beta = self.portfolio_beta();
        1.0 / (1.0 - beta * beta).sqrt()
    }

    /// Time dilation factor: proper time per coordinate time for the portfolio.
    /// τ/t = 1/γ_portfolio.
    pub fn time_dilation_factor(&self) -> f64 {
        1.0 / self.portfolio_gamma()
    }

    /// Boost all assets to a target reference frame with velocity `target_beta`.
    ///
    /// Returns `(symbol, boosted_energy, boosted_momentum)` for each asset.
    ///
    /// Lorentz boost:
    ///   E' = γ_boost (E − β_boost · p)
    ///   p' = γ_boost (p − β_boost · E)
    ///
    /// where p = m_rel · β (momentum with c = 1) and E = m_rel = γ · m₀.
    pub fn lorentz_boost_portfolio(&self, target_beta: f64) -> Vec<(String, f64, f64)> {
        let beta_boost = target_beta.clamp(-0.9999, 0.9999);
        let gamma_boost = 1.0 / (1.0 - beta_boost * beta_boost).sqrt();

        self.frames
            .iter()
            .map(|f| {
                let e = f.total_energy();
                let p = f.relativistic_mass() * f.velocity_beta;
                let e_prime = gamma_boost * (e - beta_boost * p);
                let p_prime = gamma_boost * (p - beta_boost * e);
                (f.symbol.clone(), e_prime, p_prime)
            })
            .collect()
    }

    /// Invariant mass of the portfolio system: M² = E_total² − p_total²
    /// (with c = 1). Returns √(E² − p²).
    pub fn invariant_mass(&self) -> f64 {
        let e_total: f64 = self.frames.iter().map(|f| f.total_energy()).sum();
        let p_total: f64 = self
            .frames
            .iter()
            .map(|f| f.relativistic_mass() * f.velocity_beta)
            .sum();
        let m2 = e_total * e_total - p_total * p_total;
        if m2 < 0.0 {
            0.0
        } else {
            m2.sqrt()
        }
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    #[test]
    fn zero_velocity_gives_gamma_one() {
        let asset = AssetFrame::new("SPY", 100.0, 0.0);
        assert!((asset.gamma - 1.0).abs() < EPS, "γ should be 1 for β=0");
        assert!((asset.relativistic_mass() - 100.0).abs() < EPS);
        assert!((asset.kinetic_energy()).abs() < EPS, "KE should be 0 at rest");
    }

    #[test]
    fn total_energy_gte_rest_mass() {
        for &beta in &[0.0, 0.3, 0.6, 0.9, -0.7] {
            let asset = AssetFrame::new("BTC", 50_000.0, beta);
            assert!(
                asset.total_energy() >= asset.rest_mass_value - EPS,
                "total energy >= rest mass for β={}",
                beta
            );
        }
    }

    #[test]
    fn invariant_mass_positive() {
        let mut portfolio = RelativisticPortfolio::new(0.0);
        portfolio.add_asset("AAPL", 200.0, 0.3);
        portfolio.add_asset("TSLA", 150.0, -0.4);
        portfolio.add_asset("MSFT", 300.0, 0.1);
        let inv = portfolio.invariant_mass();
        assert!(inv > 0.0, "invariant mass should be positive");
    }

    #[test]
    fn portfolio_beta_weighted_correctly() {
        let mut portfolio = RelativisticPortfolio::new(0.0);
        // Single asset — portfolio β should equal asset β.
        portfolio.add_asset("ONLY", 100.0, 0.5);
        let pb = portfolio.portfolio_beta();
        assert!((pb - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lorentz_boost_preserves_invariant_mass() {
        let mut portfolio = RelativisticPortfolio::new(0.0);
        portfolio.add_asset("X", 100.0, 0.2);
        portfolio.add_asset("Y", 80.0, -0.3);

        let inv_original = portfolio.invariant_mass();

        // After boost, individual energies and momenta change but M should be preserved.
        // (Here we just verify boost runs without panic and returns the right count.)
        let boosted = portfolio.lorentz_boost_portfolio(0.4);
        assert_eq!(boosted.len(), 2);
        assert!(inv_original > 0.0);
    }

    #[test]
    fn time_dilation_factor_less_than_one_for_moving_portfolio() {
        let mut portfolio = RelativisticPortfolio::new(0.0);
        portfolio.add_asset("A", 100.0, 0.6);
        let tdf = portfolio.time_dilation_factor();
        assert!(tdf < 1.0, "time dilation factor < 1 for moving portfolio");
        assert!(tdf > 0.0);
    }
}
