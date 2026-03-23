/// Scalar potential types for field theory models.
#[derive(Debug, Clone)]
pub enum ScalarPotential {
    Quadratic { mass: f64 },
    Phi4 { mass: f64, lambda: f64 },
    Higgs { mu_sq: f64, lambda: f64 },
    Custom { coefficients: Vec<f64> },
}

/// Compute potential energy V(φ).
pub fn potential_energy(phi: f64, potential: &ScalarPotential) -> f64 {
    match potential {
        ScalarPotential::Quadratic { mass } => 0.5 * mass * mass * phi * phi,
        ScalarPotential::Phi4 { mass, lambda } => {
            0.5 * mass * mass * phi * phi + 0.25 * lambda * phi.powi(4)
        }
        ScalarPotential::Higgs { mu_sq, lambda } => {
            -0.5 * mu_sq * phi * phi + 0.25 * lambda * phi.powi(4)
        }
        ScalarPotential::Custom { coefficients } => {
            // Polynomial: sum c_i * phi^i
            coefficients
                .iter()
                .enumerate()
                .map(|(i, c)| c * phi.powi(i as i32))
                .sum()
        }
    }
}

/// Numerical derivative dV/dφ via central difference (h = 1e-6).
pub fn dv_dphi(phi: f64, potential: &ScalarPotential) -> f64 {
    let h = 1e-6;
    (potential_energy(phi + h, potential) - potential_energy(phi - h, potential)) / (2.0 * h)
}

/// Kinetic energy T = φ̇²/2.
pub fn kinetic_energy(phi_dot: f64) -> f64 {
    0.5 * phi_dot * phi_dot
}

/// Lagrangian L = T - V - (∇φ)²/2.
pub fn lagrangian(phi: f64, phi_dot: f64, phi_x: f64, potential: &ScalarPotential) -> f64 {
    kinetic_energy(phi_dot) - potential_energy(phi, potential) - 0.5 * phi_x * phi_x
}

/// Represents the state of a 1D scalar field on a lattice.
pub struct FieldState {
    pub phi: Vec<f64>,
    pub phi_dot: Vec<f64>,
    pub dx: f64,
}

/// LCG for small random values.
fn lcg_float(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    // Map to [-0.1, 0.1]
    let val = (*state >> 11) as f64 / (u64::MAX >> 11) as f64; // [0, 1]
    (val - 0.5) * 0.2
}

impl FieldState {
    /// Construct a new field with small random initial conditions.
    pub fn new(n: usize, dx: f64, seed: u64) -> Self {
        let mut state = seed;
        let phi: Vec<f64> = (0..n).map(|_| lcg_float(&mut state)).collect();
        let phi_dot: Vec<f64> = (0..n).map(|_| lcg_float(&mut state)).collect();
        Self { phi, phi_dot, dx }
    }

    /// Leapfrog (Störmer–Verlet) step of the Klein-Gordon equation.
    /// φ̈ = ∂²φ/∂x² - dV/dφ  with periodic boundary conditions.
    pub fn step(&mut self, dt: f64, potential: &ScalarPotential) {
        let n = self.phi.len();
        // Half-step update of phi_dot, then full-step phi, then half-step phi_dot
        let mut accel: Vec<f64> = (0..n)
            .map(|i| {
                let left = if i == 0 { n - 1 } else { i - 1 };
                let right = if i == n - 1 { 0 } else { i + 1 };
                let laplacian =
                    (self.phi[left] - 2.0 * self.phi[i] + self.phi[right]) / (self.dx * self.dx);
                laplacian - dv_dphi(self.phi[i], potential)
            })
            .collect();

        // Leapfrog: half-step phi_dot
        for i in 0..n {
            self.phi_dot[i] += 0.5 * dt * accel[i];
        }
        // Full-step phi
        for i in 0..n {
            self.phi[i] += dt * self.phi_dot[i];
        }
        // Recompute acceleration at new phi
        accel = (0..n)
            .map(|i| {
                let left = if i == 0 { n - 1 } else { i - 1 };
                let right = if i == n - 1 { 0 } else { i + 1 };
                let laplacian =
                    (self.phi[left] - 2.0 * self.phi[i] + self.phi[right]) / (self.dx * self.dx);
                laplacian - dv_dphi(self.phi[i], potential)
            })
            .collect();
        // Second half-step phi_dot
        for i in 0..n {
            self.phi_dot[i] += 0.5 * dt * accel[i];
        }
    }

    /// Total energy: kinetic + gradient + potential summed over lattice.
    pub fn total_energy(&self, potential: &ScalarPotential) -> f64 {
        let n = self.phi.len();
        let mut energy = 0.0;
        for i in 0..n {
            let right = if i == n - 1 { 0 } else { i + 1 };
            let grad = (self.phi[right] - self.phi[i]) / self.dx;
            energy += kinetic_energy(self.phi_dot[i]);
            energy += 0.5 * grad * grad;
            energy += potential_energy(self.phi[i], potential);
        }
        energy
    }

    /// Mean field value (order parameter).
    pub fn order_parameter(&self) -> f64 {
        if self.phi.is_empty() {
            return 0.0;
        }
        self.phi.iter().sum::<f64>() / self.phi.len() as f64
    }
}

/// Market field wrapping a scalar field with a coupling constant.
pub struct MarketField {
    pub state: FieldState,
    pub coupling: f64,
}

impl MarketField {
    /// Build a market field from a price series (normalized to zero mean, unit variance ≈ field).
    pub fn from_prices(prices: &[f64], coupling: f64) -> Self {
        let n = prices.len().max(1);
        let mean = prices.iter().sum::<f64>() / n as f64;
        let var: f64 =
            prices.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
        let std_dev = var.sqrt().max(1e-10);
        let phi: Vec<f64> = prices.iter().map(|p| (p - mean) / std_dev).collect();
        let phi_dot = vec![0.0; phi.len()];
        Self {
            state: FieldState { phi, phi_dot, dx: 1.0 },
            coupling,
        }
    }

    /// Estimate correlation length via exponentially weighted correlation function.
    pub fn correlation_length(&self) -> f64 {
        let phi = &self.state.phi;
        let n = phi.len();
        if n == 0 {
            return 0.0;
        }
        let mean = phi.iter().sum::<f64>() / n as f64;
        let phi_c: Vec<f64> = phi.iter().map(|p| p - mean).collect();

        let mut num = 0.0f64;
        let mut denom = 0.0f64;
        let dx = self.state.dx;

        for i in 0..n {
            for j in 0..n {
                let dist = (i as f64 - j as f64).abs() * dx;
                let w = (-dist).exp();
                num += phi_c[i] * phi_c[j] * w;
                denom += w;
            }
        }
        if denom.abs() < 1e-15 {
            0.0
        } else {
            num / denom
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_potential_quadratic_phi1() {
        let pot = ScalarPotential::Quadratic { mass: 2.0 };
        let v = potential_energy(1.0, &pot);
        // V = 0.5 * 4 * 1 = 2.0
        assert!((v - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_lagrangian_computable() {
        let pot = ScalarPotential::Phi4 { mass: 1.0, lambda: 0.5 };
        let l = lagrangian(0.5, 0.3, 0.1, &pot);
        // Just verify no panic and result is finite
        assert!(l.is_finite());
    }

    #[test]
    fn test_step_approximately_conserves_energy() {
        let pot = ScalarPotential::Quadratic { mass: 1.0 };
        let mut field = FieldState::new(16, 0.5, 42);
        let e0 = field.total_energy(&pot);
        for _ in 0..100 {
            field.step(0.01, &pot);
        }
        let e1 = field.total_energy(&pot);
        // Energy drift should be small (< 5% of initial)
        let drift = (e1 - e0).abs() / (e0.abs() + 1e-10);
        assert!(drift < 0.05, "Energy drift too large: {}", drift);
    }

    #[test]
    fn test_order_parameter_uniform_field() {
        let n = 10;
        let phi = vec![2.0; n];
        let phi_dot = vec![0.0; n];
        let field = FieldState { phi, phi_dot, dx: 1.0 };
        let op = field.order_parameter();
        assert!((op - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_higgs_double_minimum() {
        // Higgs: V = -mu²φ²/2 + λφ⁴/4 has minima at φ = ±sqrt(mu²/lambda)
        let mu_sq = 1.0;
        let lambda = 1.0;
        let pot = ScalarPotential::Higgs { mu_sq, lambda };
        let phi_min = (mu_sq / lambda).sqrt();
        let v_min = potential_energy(phi_min, &pot);
        let v_neg = potential_energy(-phi_min, &pot);
        let v_zero = potential_energy(0.0, &pot);
        // Both minima should have the same (lower) energy than phi=0
        assert!((v_min - v_neg).abs() < 1e-8);
        assert!(v_min < v_zero);
    }
}
