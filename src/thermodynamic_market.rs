//! Market thermodynamics: partition function, free energy, entropy of returns.
//!
//! Treats absolute return magnitudes as "energies" in a Boltzmann ensemble
//! where inverse temperature β = 1/σ (σ = volatility).  High β (low volatility)
//! = cold, ordered market; low β (high volatility) = hot, disordered market.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Core thermodynamic functions
// ---------------------------------------------------------------------------

/// A snapshot of returns at a given inverse temperature.
#[derive(Debug, Clone)]
pub struct ReturnDistribution {
    pub returns: Vec<f64>,
    /// Inverse temperature: β = 1/volatility.
    pub beta: f64,
}

/// Partition function Z = Σ exp(−β |r_i|).
pub fn partition_function(returns: &[f64], beta: f64) -> f64 {
    returns.iter().map(|r| (-beta * r.abs()).exp()).sum()
}

/// Helmholtz free energy F = −ln(Z) / β.
pub fn free_energy(returns: &[f64], beta: f64) -> f64 {
    let z = partition_function(returns, beta);
    if z <= 0.0 || beta == 0.0 {
        return 0.0;
    }
    -z.ln() / beta
}

/// Internal energy U = Σ |r_i| exp(−β|r_i|) / Z.
pub fn internal_energy(returns: &[f64], beta: f64) -> f64 {
    let z = partition_function(returns, beta);
    if z <= 0.0 {
        return 0.0;
    }
    let numerator: f64 = returns.iter()
        .map(|r| r.abs() * (-beta * r.abs()).exp())
        .sum();
    numerator / z
}

/// Entropy S = β·U + ln(Z)  (thermodynamic identity S = β(U − F)).
pub fn entropy(returns: &[f64], beta: f64) -> f64 {
    let z = partition_function(returns, beta);
    let u = internal_energy(returns, beta);
    beta * u + z.ln()
}

/// Heat capacity C = β² · Var(|r|) under Boltzmann weights.
pub fn heat_capacity(returns: &[f64], beta: f64) -> f64 {
    let z = partition_function(returns, beta);
    if z <= 0.0 {
        return 0.0;
    }
    // Compute <|r|> and <|r|²> under the Boltzmann distribution.
    let mean_r: f64 = returns.iter()
        .map(|r| r.abs() * (-beta * r.abs()).exp())
        .sum::<f64>() / z;
    let mean_r2: f64 = returns.iter()
        .map(|r| r.abs().powi(2) * (-beta * r.abs()).exp())
        .sum::<f64>() / z;
    let variance = mean_r2 - mean_r.powi(2);
    beta.powi(2) * variance
}

// ---------------------------------------------------------------------------
// MarketThermodynamics: rolling windows
// ---------------------------------------------------------------------------

/// Accumulates thermodynamic snapshots of rolling return windows.
#[derive(Debug, Default)]
pub struct MarketThermodynamics {
    pub history: Vec<ReturnDistribution>,
}

impl MarketThermodynamics {
    pub fn new() -> Self {
        Self { history: Vec::new() }
    }

    /// Add a return window; volatility is converted to β = 1/volatility.
    pub fn add_window(&mut self, returns: Vec<f64>, volatility: f64) {
        let beta = if volatility == 0.0 { f64::INFINITY } else { 1.0 / volatility };
        self.history.push(ReturnDistribution { returns, beta });
    }

    /// Temperature (1/β) for each recorded window.
    pub fn temperature_series(&self) -> Vec<f64> {
        self.history.iter().map(|rd| 1.0 / rd.beta).collect()
    }

    /// Entropy for each recorded window.
    pub fn entropy_series(&self) -> Vec<f64> {
        self.history.iter()
            .map(|rd| entropy(&rd.returns, rd.beta))
            .collect()
    }

    /// Indices where entropy changes by more than 50 % relative to the prior window.
    pub fn phase_transition_points(&self) -> Vec<usize> {
        let series = self.entropy_series();
        let mut pts = Vec::new();
        for i in 1..series.len() {
            let prev = series[i - 1].abs();
            if prev == 0.0 {
                continue;
            }
            let change = (series[i] - series[i - 1]).abs() / prev;
            if change > 0.5 {
                pts.push(i);
            }
        }
        pts
    }
}

// ---------------------------------------------------------------------------
// Maxwell-Boltzmann return distribution
// ---------------------------------------------------------------------------

/// A Maxwell-Boltzmann distribution parameterised by σ, used for return magnitudes.
pub struct MaxwellBoltzmannReturn {
    pub sigma: f64,
}

impl MaxwellBoltzmannReturn {
    /// PDF: f(r) = sqrt(2/π) · r² / σ³ · exp(−r²/(2σ²))  for r ≥ 0.
    pub fn pdf(r: f64, sigma: f64) -> f64 {
        if r < 0.0 || sigma <= 0.0 {
            return 0.0;
        }
        let s2 = sigma * sigma;
        let s3 = s2 * sigma;
        (2.0 / PI).sqrt() * r * r / s3 * (-r * r / (2.0 * s2)).exp()
    }

    /// Mean: 2σ·sqrt(2/π).
    pub fn mean_return(sigma: f64) -> f64 {
        2.0 * sigma * (2.0 / PI).sqrt()
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    fn sample_returns() -> Vec<f64> {
        vec![0.01, -0.02, 0.03, -0.015, 0.005, 0.04, -0.01]
    }

    #[test]
    fn partition_function_positive() {
        let z = partition_function(&sample_returns(), 10.0);
        assert!(z > 0.0, "partition function must be positive");
    }

    #[test]
    fn entropy_positive() {
        let s = entropy(&sample_returns(), 10.0);
        assert!(s > 0.0, "entropy should be positive for this dataset");
    }

    #[test]
    fn free_energy_negative_or_finite() {
        let returns: Vec<f64> = vec![0.1, 0.2, 0.15];
        let f = free_energy(&returns, 2.0);
        // F = -ln(Z)/beta; Z > 1 for these values so F < 0.
        assert!(f.is_finite(), "free energy should be finite");
    }

    #[test]
    fn heat_capacity_non_negative() {
        let c = heat_capacity(&sample_returns(), 10.0);
        assert!(c >= 0.0, "heat capacity must be non-negative");
    }

    #[test]
    fn maxwell_boltzmann_pdf_integrates_approx_one() {
        let sigma = 0.02_f64;
        // Numerical integration via rectangle rule over [0, 10σ].
        let n = 10_000;
        let upper = 10.0 * sigma;
        let dx = upper / n as f64;
        let integral: f64 = (0..n)
            .map(|i| MaxwellBoltzmannReturn::pdf(i as f64 * dx, sigma) * dx)
            .sum();
        assert!(
            (integral - 1.0).abs() < 0.01,
            "Maxwell-Boltzmann PDF should integrate to ≈1, got {integral}"
        );
    }
}
