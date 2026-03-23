//! Scalar quantum field theory analog for market fluctuations.
//!
//! Models the market as a scalar field with Klein-Gordon dynamics.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Field mode
// ---------------------------------------------------------------------------

/// A single Fourier mode of the scalar field.
#[derive(Debug, Clone)]
pub struct FieldMode {
    pub momentum_k: f64,
    pub frequency_omega: f64,
    pub amplitude: f64,
    pub phase: f64,
}

// ---------------------------------------------------------------------------
// Scalar field
// ---------------------------------------------------------------------------

/// A scalar quantum field consisting of many Fourier modes.
#[derive(Debug, Clone)]
pub struct ScalarField {
    pub modes: Vec<FieldMode>,
    /// Klein-Gordon mass parameter.
    pub mass: f64,
    /// Self-interaction coupling constant.
    pub coupling: f64,
}

impl ScalarField {
    /// Initialize the vacuum state: zero-point fluctuations for each mode.
    ///
    /// k = 2π·n/L for n in 1..=num_modes, L = 10.0
    /// ω = sqrt(k² + m²)
    /// amplitude = sqrt(ℏ/(2·ω)) with ℏ = 1
    pub fn vacuum(num_modes: usize, mass: f64) -> Self {
        let l = 10.0_f64;
        let modes: Vec<FieldMode> = (1..=num_modes)
            .map(|n| {
                let k = 2.0 * PI * n as f64 / l;
                let omega = (k * k + mass * mass).sqrt();
                let amplitude = (1.0 / (2.0 * omega)).sqrt(); // sqrt(hbar/(2*omega)) with hbar=1
                FieldMode {
                    momentum_k: k,
                    frequency_omega: omega,
                    amplitude,
                    phase: 0.0,
                }
            })
            .collect();
        Self {
            modes,
            mass,
            coupling: 0.0,
        }
    }

    /// Field value at spatial position `x` and time `t`.
    ///
    /// φ(x,t) = Σ amplitude · cos(k·x − ω·t + φ)
    pub fn field_value_at(&self, x: f64, t: f64) -> f64 {
        self.modes
            .iter()
            .map(|m| m.amplitude * (m.momentum_k * x - m.frequency_omega * t + m.phase).cos())
            .sum()
    }

    /// Variance of vacuum fluctuations: Σ amplitude² / 2.
    pub fn vacuum_fluctuation_variance(&self) -> f64 {
        self.modes.iter().map(|m| m.amplitude * m.amplitude / 2.0).sum()
    }
}

// ---------------------------------------------------------------------------
// Klein-Gordon propagator
// ---------------------------------------------------------------------------

/// Klein-Gordon propagator approximation.
///
/// D(x1,t1; x2,t2) = exp(−m·|Δr|) · cos(m·Δt)
/// where Δr = sqrt(|Δx² − Δt²|) (Euclidean-ish).
pub fn propagator(x1: f64, t1: f64, x2: f64, t2: f64, mass: f64) -> f64 {
    let dx = x2 - x1;
    let dt = t2 - t1;
    let interval_sq = (dx * dx - dt * dt).abs();
    let r = interval_sq.sqrt();
    (-mass * r).exp() * (mass * dt).cos()
}

// ---------------------------------------------------------------------------
// Quantum Market Field
// ---------------------------------------------------------------------------

/// A market modelled as a quantum scalar field.
#[derive(Debug, Clone)]
pub struct QuantumMarketField {
    pub field: ScalarField,
    pub price_scale: f64,
    pub base_price: f64,
}

impl QuantumMarketField {
    /// Price at time `t`: base_price + price_scale · φ(0, t).
    pub fn price_at(&self, t: f64) -> f64 {
        self.base_price + self.price_scale * self.field.field_value_at(0.0, t)
    }

    /// Volatility from vacuum fluctuations: price_scale · sqrt(variance).
    pub fn volatility_from_vacuum(&self) -> f64 {
        self.price_scale * self.field.vacuum_fluctuation_variance().sqrt()
    }

    /// Two-point correlation: propagator(0, t1, 0, t2, mass).
    pub fn correlation(&self, t1: f64, t2: f64) -> f64 {
        propagator(0.0, t1, 0.0, t2, self.field.mass)
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vacuum_variance_positive() {
        let field = ScalarField::vacuum(10, 1.0);
        assert!(field.vacuum_fluctuation_variance() > 0.0);
    }

    #[test]
    fn test_field_value_continuous() {
        let field = ScalarField::vacuum(20, 0.5);
        // Field at t and t+epsilon should be close
        let v1 = field.field_value_at(0.0, 1.0);
        let v2 = field.field_value_at(0.0, 1.0 + 1e-6);
        assert!((v1 - v2).abs() < 0.01, "Field should be continuous in time");
    }

    #[test]
    fn test_price_scale_relationship() {
        let field = ScalarField::vacuum(5, 1.0);
        let mf = QuantumMarketField {
            field: field.clone(),
            price_scale: 100.0,
            base_price: 50.0,
        };
        let phi = field.field_value_at(0.0, 0.0);
        let price = mf.price_at(0.0);
        assert!((price - (50.0 + 100.0 * phi)).abs() < 1e-9);
    }

    #[test]
    fn test_propagator_symmetric_in_spatial_args() {
        // D(x1, t, x2, t) should equal D(x2, t, x1, t)
        let d1 = propagator(1.0, 0.0, 3.0, 0.0, 0.5);
        let d2 = propagator(3.0, 0.0, 1.0, 0.0, 0.5);
        assert!((d1 - d2).abs() < 1e-10);
    }

    #[test]
    fn test_volatility_from_vacuum_positive() {
        let field = ScalarField::vacuum(10, 1.0);
        let mf = QuantumMarketField {
            field,
            price_scale: 10.0,
            base_price: 100.0,
        };
        assert!(mf.volatility_from_vacuum() > 0.0);
    }
}
