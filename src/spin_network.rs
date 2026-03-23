//! Spin network analogy for market correlation structure.
//!
//! Markets are modelled as an Ising spin system where each asset node
//! carries a spin ±1 (bull/bear) and edges represent coupling strengths
//! derived from empirical pairwise correlations.

/// A node in the spin network representing a single asset.
#[derive(Debug, Clone)]
pub struct SpinNode {
    pub id: String,
    /// Spin value: +1.0 (bullish) or −1.0 (bearish).
    pub spin: f64,
}

/// A weighted edge connecting two asset nodes.
#[derive(Debug, Clone)]
pub struct SpinEdge {
    pub from: String,
    pub to: String,
    /// Coupling strength J_ij (positive = ferromagnetic / correlated).
    pub coupling: f64,
}

/// An Ising-like spin network for market correlation analysis.
pub struct SpinNetwork {
    pub nodes: Vec<SpinNode>,
    pub edges: Vec<SpinEdge>,
}

/// Seeded LCG returning a pseudo-uniform f64 in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / (1u64 << 53) as f64
}

/// Seeded LCG returning an index in [0, n).
fn lcg_index(state: &mut u64, n: usize) -> usize {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 33) as usize) % n
}

impl SpinNetwork {
    /// Build a spin network from a correlation matrix.
    ///
    /// Each symbol becomes a node. An edge is added for every pair where
    /// `correlations[i][j]` is non-zero (using the upper triangle only to
    /// avoid duplicate edges). The coupling equals the correlation value.
    /// Initial spins are +1.
    pub fn from_correlation_matrix(
        symbols: &[String],
        correlations: &[Vec<f64>],
    ) -> SpinNetwork {
        let n = symbols.len();
        let nodes = symbols
            .iter()
            .map(|s| SpinNode { id: s.clone(), spin: 1.0 })
            .collect();

        let mut edges = Vec::new();
        for i in 0..n {
            for j in (i + 1)..n {
                let c = if i < correlations.len() && j < correlations[i].len() {
                    correlations[i][j]
                } else {
                    0.0
                };
                if c.abs() > 1e-12 {
                    edges.push(SpinEdge {
                        from: symbols[i].clone(),
                        to: symbols[j].clone(),
                        coupling: c,
                    });
                }
            }
        }

        SpinNetwork { nodes, edges }
    }

    /// Ising Hamiltonian: H = −Σ J_ij · s_i · s_j.
    pub fn hamiltonian(&self) -> f64 {
        let spin_of = |id: &str| -> f64 {
            self.nodes
                .iter()
                .find(|n| n.id == id)
                .map(|n| n.spin)
                .unwrap_or(0.0)
        };
        -self.edges.iter().map(|e| e.coupling * spin_of(&e.from) * spin_of(&e.to)).sum::<f64>()
    }

    /// Mean spin value — proxy for aggregate market sentiment.
    pub fn magnetization(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.nodes.iter().map(|n| n.spin).sum::<f64>() / self.nodes.len() as f64
    }

    /// One Metropolis step: attempt to flip a randomly chosen spin.
    ///
    /// Returns `true` if the flip was accepted.
    pub fn metropolis_step(&mut self, temperature: f64, seed: u64) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let mut rng = seed;
        let idx = lcg_index(&mut rng, self.nodes.len());

        // Compute ΔH for flipping node `idx`.
        let node_id = self.nodes[idx].id.clone();
        let s_i = self.nodes[idx].spin;

        let delta_h: f64 = self
            .edges
            .iter()
            .filter_map(|e| {
                if e.from == node_id {
                    let s_j = self.nodes.iter().find(|n| n.id == e.to)?.spin;
                    Some(2.0 * e.coupling * s_i * s_j)
                } else if e.to == node_id {
                    let s_j = self.nodes.iter().find(|n| n.id == e.from)?.spin;
                    Some(2.0 * e.coupling * s_i * s_j)
                } else {
                    None
                }
            })
            .sum();

        let accept = if delta_h <= 0.0 {
            true
        } else if temperature <= 0.0 {
            false
        } else {
            let r = lcg_f64(&mut rng);
            r < (-delta_h / temperature).exp()
        };

        if accept {
            self.nodes[idx].spin = -s_i;
        }
        accept
    }

    /// Run a Metropolis simulation for `steps` steps.
    ///
    /// Returns the magnetization recorded after each step.
    pub fn run_simulation(
        &mut self,
        steps: usize,
        temperature: f64,
        seed: u64,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(steps);
        for step in 0..steps {
            self.metropolis_step(temperature, seed.wrapping_add(step as u64 * 7919));
            history.push(self.magnetization());
        }
        history
    }

    /// Mean-field critical temperature: T_c = z · |J̄| where z is average degree.
    pub fn critical_temperature(n_nodes: usize, avg_coupling: f64) -> f64 {
        if n_nodes == 0 {
            return 0.0;
        }
        // Average degree estimate from fully connected graph: z = n - 1.
        let z = (n_nodes as f64) - 1.0;
        z * avg_coupling.abs()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_asset_network() -> SpinNetwork {
        let syms: Vec<String> = vec!["AAPL".to_string(), "MSFT".to_string()];
        let corr = vec![vec![1.0, 0.8], vec![0.8, 1.0]];
        SpinNetwork::from_correlation_matrix(&syms, &corr)
    }

    #[test]
    fn test_from_correlation_matrix_nodes() {
        let net = two_asset_network();
        assert_eq!(net.nodes.len(), 2);
    }

    #[test]
    fn test_from_correlation_matrix_edges() {
        let net = two_asset_network();
        // Upper triangle only: one edge.
        assert_eq!(net.edges.len(), 1);
        assert!((net.edges[0].coupling - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_initial_spins_positive() {
        let net = two_asset_network();
        for node in &net.nodes {
            assert_eq!(node.spin, 1.0);
        }
    }

    #[test]
    fn test_hamiltonian_all_aligned() {
        let net = two_asset_network();
        // H = −0.8 * 1 * 1 = −0.8
        let h = net.hamiltonian();
        assert!((h + 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_magnetization_all_up() {
        let net = two_asset_network();
        assert!((net.magnetization() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_magnetization_empty() {
        let net = SpinNetwork { nodes: vec![], edges: vec![] };
        assert_eq!(net.magnetization(), 0.0);
    }

    #[test]
    fn test_metropolis_step_returns_bool() {
        let mut net = two_asset_network();
        let result = net.metropolis_step(1.0, 42);
        assert!(result == true || result == false);
    }

    #[test]
    fn test_run_simulation_length() {
        let mut net = two_asset_network();
        let history = net.run_simulation(10, 2.0, 99);
        assert_eq!(history.len(), 10);
    }

    #[test]
    fn test_run_simulation_values_bounded() {
        let mut net = two_asset_network();
        let history = net.run_simulation(20, 1.0, 7);
        for &m in &history {
            assert!(m >= -1.0 && m <= 1.0, "magnetization out of range: {m}");
        }
    }

    #[test]
    fn test_critical_temperature_formula() {
        // z=9, J=0.5 -> T_c = 4.5
        let tc = SpinNetwork::critical_temperature(10, 0.5);
        assert!((tc - 4.5).abs() < 1e-9);
    }

    #[test]
    fn test_critical_temperature_zero_nodes() {
        assert_eq!(SpinNetwork::critical_temperature(0, 1.0), 0.0);
    }

    #[test]
    fn test_zero_coupling_no_edges() {
        let syms: Vec<String> = vec!["X".to_string(), "Y".to_string()];
        let corr = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let net = SpinNetwork::from_correlation_matrix(&syms, &corr);
        assert_eq!(net.edges.len(), 0);
    }

    #[test]
    fn test_hamiltonian_opposite_spins() {
        let mut net = two_asset_network();
        net.nodes[1].spin = -1.0;
        // H = −0.8 * 1 * (−1) = 0.8
        let h = net.hamiltonian();
        assert!((h - 0.8).abs() < 1e-9);
    }

    #[test]
    fn test_three_asset_edges() {
        let syms: Vec<String> = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        let corr = vec![
            vec![1.0, 0.5, 0.3],
            vec![0.5, 1.0, 0.4],
            vec![0.3, 0.4, 1.0],
        ];
        let net = SpinNetwork::from_correlation_matrix(&syms, &corr);
        // 3 choose 2 = 3 edges
        assert_eq!(net.edges.len(), 3);
    }
}
