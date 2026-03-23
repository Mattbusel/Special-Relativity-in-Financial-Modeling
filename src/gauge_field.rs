use std::f64::consts::PI;

/// Complex number as (re, im).
pub type Complex = (f64, f64);

pub struct GaugeField {
    pub phases: Vec<f64>,
    pub coupling: f64,
    pub lattice_size: usize,
}

fn lcg_next(state: &mut u64) -> f64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as f64 / (u32::MAX as f64 + 1.0)
}

impl GaugeField {
    pub fn new(size: usize, coupling: f64, seed: u64) -> Self {
        let mut state = seed;
        let phases: Vec<f64> = (0..size)
            .map(|_| lcg_next(&mut state) * 2.0 * PI)
            .collect();
        GaugeField { phases, coupling, lattice_size: size }
    }

    /// exp(i*(theta_i - theta_j))
    pub fn link_variable(theta_i: f64, theta_j: f64) -> Complex {
        let diff = theta_i - theta_j;
        (diff.cos(), diff.sin())
    }

    /// Re(U_ij * U_jk * conj(U_ik)) for triangular plaquette
    pub fn plaquette_action(field: &GaugeField, i: usize, j: usize) -> f64 {
        let n = field.lattice_size;
        if n < 3 {
            return 1.0;
        }
        let k = (j + 1) % n;
        let theta_i = field.phases[i % n];
        let theta_j = field.phases[j % n];
        let theta_k = field.phases[k];

        // U_ij * U_jk * conj(U_ik)
        let u_ij = Self::link_variable(theta_i, theta_j);
        let u_jk = Self::link_variable(theta_j, theta_k);
        let u_ik_conj = (Self::link_variable(theta_i, theta_k).0, -Self::link_variable(theta_i, theta_k).1);

        // Multiply: u_ij * u_jk
        let mid = (
            u_ij.0 * u_jk.0 - u_ij.1 * u_jk.1,
            u_ij.0 * u_jk.1 + u_ij.1 * u_jk.0,
        );
        // Multiply by conj(u_ik)
        let result = (
            mid.0 * u_ik_conj.0 - mid.1 * u_ik_conj.1,
            mid.0 * u_ik_conj.1 + mid.1 * u_ik_conj.0,
        );
        // Return real part
        result.0
    }

    /// Sum of (1 - cos(theta_i - theta_j)) over adjacent pairs
    pub fn total_action(&self) -> f64 {
        let mut sum = 0.0;
        for i in 0..self.lattice_size.saturating_sub(1) {
            let diff = self.phases[i] - self.phases[i + 1];
            sum += 1.0 - diff.cos();
        }
        sum
    }

    pub fn update_phase(&mut self, i: usize, delta: f64) {
        if i < self.phases.len() {
            self.phases[i] = (self.phases[i] + delta).rem_euclid(2.0 * PI);
        }
    }

    pub fn metropolis_sweep(&mut self, beta: f64, seed: &mut u64) {
        let n = self.lattice_size;
        for i in 0..n {
            let s_before = self.local_action(i);
            let noise = lcg_next(seed);
            let delta = (noise - 0.5) * PI / 2.0; // [-π/4, π/4]
            let old_phase = self.phases[i];
            self.phases[i] = (old_phase + delta).rem_euclid(2.0 * PI);
            let s_after = self.local_action(i);
            let ds = s_after - s_before;
            if ds < 0.0 {
                // Accept
            } else {
                let r = lcg_next(seed);
                if r > (-beta * ds).exp() {
                    // Reject — restore
                    self.phases[i] = old_phase;
                }
            }
        }
    }

    fn local_action(&self, i: usize) -> f64 {
        let mut s = 0.0;
        if i > 0 {
            let diff = self.phases[i] - self.phases[i - 1];
            s += 1.0 - diff.cos();
        }
        if i + 1 < self.lattice_size {
            let diff = self.phases[i] - self.phases[i + 1];
            s += 1.0 - diff.cos();
        }
        s
    }

    /// |1/N * sum(exp(i*theta_j))| — magnitude of mean phase vector
    pub fn order_parameter(&self) -> f64 {
        if self.phases.is_empty() {
            return 0.0;
        }
        let n = self.phases.len() as f64;
        let re: f64 = self.phases.iter().map(|&t| t.cos()).sum::<f64>() / n;
        let im: f64 = self.phases.iter().map(|&t| t.sin()).sum::<f64>() / n;
        (re * re + im * im).sqrt()
    }
}

pub struct MarketPhaseCoherence {
    pub field: GaugeField,
    pub price_phases: Vec<f64>,
}

impl MarketPhaseCoherence {
    pub fn from_prices(prices: &[f64], coupling: f64) -> Self {
        let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let range = if (max - min).abs() < 1e-12 { 1.0 } else { max - min };

        let price_phases: Vec<f64> = prices
            .iter()
            .map(|&p| 2.0 * PI * (p - min) / range)
            .collect();

        let n = price_phases.len();
        let mut field = GaugeField {
            phases: price_phases.clone(),
            coupling,
            lattice_size: n,
        };
        field.lattice_size = n;

        MarketPhaseCoherence { field, price_phases }
    }

    pub fn coherence(&self) -> f64 {
        self.field.order_parameter()
    }

    pub fn phase_transition_beta(&self) -> f64 {
        1.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_action_nonnegative() {
        let field = GaugeField::new(20, 1.0, 42);
        assert!(field.total_action() >= 0.0);
    }

    #[test]
    fn test_order_parameter_in_range() {
        let field = GaugeField::new(50, 1.0, 99);
        let op = field.order_parameter();
        assert!(op >= 0.0 && op <= 1.0 + 1e-9, "op={}", op);
    }

    #[test]
    fn test_phase_wraps_correctly() {
        let mut field = GaugeField::new(5, 1.0, 1);
        field.phases[0] = 0.1;
        field.update_phase(0, 7.0); // 7.0 > 2π
        assert!(field.phases[0] >= 0.0 && field.phases[0] < 2.0 * PI + 1e-9);
    }

    #[test]
    fn test_metropolis_changes_phases() {
        let mut field = GaugeField::new(10, 1.0, 7);
        let initial: Vec<f64> = field.phases.clone();
        let mut seed = 12345u64;
        field.metropolis_sweep(1.0, &mut seed);
        // At least some phases should change
        let changed = field.phases.iter().zip(initial.iter()).any(|(a, b)| (a - b).abs() > 1e-10);
        assert!(changed);
    }

    #[test]
    fn test_coherence_from_uniform_prices() {
        // All same price → phases all equal → order parameter = 1
        let prices = vec![100.0f64; 20];
        let mpc = MarketPhaseCoherence::from_prices(&prices, 1.0);
        let coh = mpc.coherence();
        // All same price → all phases = 0 (min=max, use range=1 fallback)
        assert!((coh - 1.0).abs() < 1e-6, "coh={}", coh);
    }
}
