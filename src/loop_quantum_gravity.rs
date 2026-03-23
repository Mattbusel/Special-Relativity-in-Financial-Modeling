//! Loop Quantum Gravity (LQG) concepts and their financial market analogues.
//!
//! Implements spin networks, LQG cosmology (bounce scenario), spin foam
//! amplitudes, and quantized market dynamics inspired by LQG discreteness.

use std::f64::consts::PI;

// ─── CONSTANTS ───────────────────────────────────────────────────────────────

/// Barbero–Immirzi parameter (commonly used value).
const BARBERO_IMMIRZI: f64 = 0.2375;

// ─── SPIN NETWORK ────────────────────────────────────────────────────────────

/// A combinatorial spin network: nodes carry half-integer spins (stored as 2j),
/// edges carry spin labels connecting pairs of nodes.
///
/// Spin j is stored as the integer `2j` to avoid floating-point bookkeeping.
#[derive(Debug, Clone)]
pub struct SpinNetwork2 {
    /// `(node_id, 2j)` — node index and its spin label (2j).
    pub nodes: Vec<(usize, u32)>,
    /// `(from_node, to_node, 2j_edge)` — directed edge with spin label.
    pub edges: Vec<(usize, usize, u32)>,
}

impl SpinNetwork2 {
    /// Creates a new empty spin network.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }

    /// Area eigenvalue for a given edge.
    ///
    /// A = 8π γ l_P² √(j(j+1))  where j = edge_2j / 2.
    pub fn area_eigenvalue(&self, edge_idx: usize, planck_area: f64) -> f64 {
        let two_j = self.edges[edge_idx].2;
        let j = two_j as f64 / 2.0;
        8.0 * PI * BARBERO_IMMIRZI * planck_area * (j * (j + 1.0)).sqrt()
    }

    /// Volume eigenvalue for a given node (simplified Thiemann formula).
    ///
    /// V ~ l_P³ √|det(J)|  approximated as √(j₁ j₂ j₃) for a trivalent node.
    pub fn volume_eigenvalue(&self, node_idx: usize, planck_vol: f64) -> f64 {
        // Gather edge spins incident on this node
        let node_id = self.nodes[node_idx].0;
        let incident: Vec<f64> = self.edges.iter()
            .filter(|(f, t, _)| *f == node_id || *t == node_id)
            .map(|(_, _, two_j)| *two_j as f64 / 2.0)
            .collect();

        if incident.is_empty() {
            return 0.0;
        }

        // Simplified: geometric mean of incident spins, scaled by l_P³
        let product: f64 = incident.iter().product();
        planck_vol * product.powf(1.0 / incident.len() as f64).sqrt()
    }

    /// Checks whether triangle inequalities hold at every node.
    ///
    /// For each node, the spins on its incident edges must satisfy the
    /// SU(2) admissibility condition: each triple (j₁, j₂, j₃) must satisfy
    /// |j₁ - j₂| ≤ j₃ ≤ j₁ + j₂ and the sum j₁+j₂+j₃ must be an integer.
    pub fn admissible(&self) -> bool {
        for &(node_id, _) in &self.nodes {
            let incident: Vec<u32> = self.edges.iter()
                .filter(|(f, t, _)| *f == node_id || *t == node_id)
                .map(|(_, _, two_j)| *two_j)
                .collect();

            // Check each consecutive triple (only meaningful for ≥3 edges)
            for i in 0..incident.len().saturating_sub(2) {
                let (a, b, c) = (incident[i], incident[i + 1], incident[i + 2]);
                // Triangle inequality in terms of 2j: |a-b| <= c <= a+b
                if c < a.abs_diff(b) || c > a + b {
                    return false;
                }
                // Sum must be even (so j₁+j₂+j₃ is integer)
                if (a + b + c) % 2 != 0 {
                    return false;
                }
            }
        }
        true
    }
}

impl Default for SpinNetwork2 {
    fn default() -> Self {
        Self::new()
    }
}

// ─── LQG COSMOLOGY ───────────────────────────────────────────────────────────

/// Loop quantum cosmology: flat FLRW universe with holonomy corrections.
#[derive(Debug, Clone)]
pub struct LqgCosmology {
    /// Current volume of the universe (in Planck units or SI m³).
    pub volume: f64,
    /// Current Hubble parameter H (s⁻¹).
    pub hubble: f64,
    /// Planck volume (m³) — sets the quantum-of-volume scale.
    pub planck_volume: f64,
}

impl LqgCosmology {
    /// Creates a new LQG cosmology model.
    pub fn new(volume: f64, hubble: f64, planck_volume: f64) -> Self {
        Self { volume, hubble, planck_volume }
    }

    /// Critical (bounce) density.
    ///
    /// ρ_c = √3 / (16π² γ³ G² ħ) ≈ 0.41 ρ_Planck.
    /// In natural units (G = ħ = 1), ρ_Planck = 1, so ρ_c ≈ 0.41.
    pub fn bounce_density(&self) -> f64 {
        let gamma = BARBERO_IMMIRZI;
        3_f64.sqrt() / (16.0 * PI * PI * gamma.powi(3))
    }

    /// Holonomy quantum correction factor: 1 − ρ / ρ_c.
    ///
    /// This modifies the Friedmann equation so the Big Bang is replaced
    /// by a Big Bounce.
    pub fn quantum_correction(&self, rho: f64) -> f64 {
        let rho_c = self.bounce_density();
        1.0 - rho / rho_c
    }

    /// Modified Friedmann equation with LQC holonomy corrections.
    ///
    /// H² = (8π G / 3) ρ (1 − ρ/ρ_c)
    ///
    /// In natural units G = 1/(8π):  H² = ρ(1 − ρ/ρ_c)/3.
    pub fn modified_friedmann(&self, rho: f64) -> f64 {
        let rho_c = self.bounce_density();
        (rho / 3.0) * (1.0 - rho / rho_c)
    }

    /// Characteristic time scale for the bounce: t_bounce ~ γ^(3/2) t_Planck.
    pub fn big_bounce_time(&self, planck_time: f64) -> f64 {
        BARBERO_IMMIRZI.powf(1.5) * planck_time
    }

    /// Quantum-corrected slow-roll parameter for inflation.
    ///
    /// δV = V(φ) * (1 − ρ/ρ_c)  — the potential is suppressed near the bounce.
    pub fn loop_corrections(&self, inflation_potential: f64, phi: f64) -> f64 {
        // Simple quadratic potential V ~ φ²; correction from quantum geometry.
        let v = inflation_potential * phi * phi;
        let rho = v; // slow-roll approximation: ρ ≈ V(φ)
        v * self.quantum_correction(rho)
    }
}

// ─── SPIN FOAM AMPLITUDE ─────────────────────────────────────────────────────

/// Spin foam transition amplitudes.
pub struct SpinFoamAmplitude;

impl SpinFoamAmplitude {
    /// Simplified EPRL vertex amplitude.
    ///
    /// A_v ~ Π_f d_{j_f}  × Σ_ι Π_e ι_e   (symbolic; full amplitude requires
    /// 6j/15j symbols).  Here we return a dimensionless product of spin dimensions.
    pub fn eprl_vertex(spins: &[u32], intertwiners: &[u32]) -> f64 {
        // dim(j) = 2j + 1
        let face_factor: f64 = spins.iter().map(|&two_j| (two_j + 1) as f64).product();
        let intertwiner_factor: f64 = if intertwiners.is_empty() {
            1.0
        } else {
            intertwiners.iter().map(|&i| i as f64 + 1.0).sum::<f64>()
                / intertwiners.len() as f64
        };
        face_factor * intertwiner_factor
    }

    /// Partition function: sum of EPRL vertex amplitudes over a set of networks.
    pub fn partition_function_lqg(networks: &[SpinNetwork2], planck_area: f64) -> f64 {
        networks.iter().map(|net| {
            let spins: Vec<u32> = net.edges.iter().map(|(_, _, two_j)| *two_j).collect();
            let intertwiners: Vec<u32> = net.nodes.iter().map(|(_, two_j)| *two_j).collect();
            let amp = Self::eprl_vertex(&spins, &intertwiners);
            // Weight by area eigenvalue of first edge (if present)
            let area_weight = if net.edges.is_empty() {
                planck_area
            } else {
                net.area_eigenvalue(0, planck_area)
            };
            amp * area_weight.exp().min(1e10)
        }).sum()
    }

    /// Minimum area gap (smallest non-zero LQG area eigenvalue).
    ///
    /// Δ_min = 4π √3 γ l_P²  (corresponds to j = 1/2, i.e. two_j = 1)
    pub fn area_gap(planck_area: f64) -> f64 {
        4.0 * PI * 3_f64.sqrt() * BARBERO_IMMIRZI * planck_area
    }
}

// ─── MARKET LQG ANALOGUE ─────────────────────────────────────────────────────

/// Financial market analogues of LQG concepts.
///
/// Treats prices as quantized on a discrete spectrum analogous to LQG area
/// eigenvalues, and identifies "bounce" patterns as market reversals.
pub struct MarketLQG;

impl MarketLQG {
    /// Generates a discrete price spectrum by analogy with LQG area eigenvalues.
    ///
    /// p_n = base_price × √(n(n+1)) / √(1×2)   for n = 1, 2, …, n_levels
    pub fn discrete_price_spectrum(base_price: f64, n_levels: u32) -> Vec<f64> {
        let norm = (1.0_f64 * 2.0_f64).sqrt();
        (1..=n_levels).map(|n| {
            let j = n as f64;
            base_price * (j * (j + 1.0)).sqrt() / norm
        }).collect()
    }

    /// Finds a "bounce" pattern in a price history.
    ///
    /// A bounce is a local minimum (prices decline then recover) analogous to
    /// the LQC Big Bounce — the universe collapses but then re-expands.
    ///
    /// Returns the index of the bounce (minimum) if found, else `None`.
    pub fn bounce_scenario(price_history: &[f64]) -> Option<usize> {
        if price_history.len() < 3 {
            return None;
        }
        for i in 1..price_history.len() - 1 {
            let prev = price_history[i - 1];
            let curr = price_history[i];
            let next = price_history[i + 1];
            if curr < prev && curr < next {
                return Some(i);
            }
        }
        None
    }

    /// Discrete momentum: first-order finite difference over a sliding window.
    ///
    /// Analogous to the LQC discrete time derivative of the scale factor.
    pub fn quantum_momentum(prices: &[f64], window: usize) -> f64 {
        if prices.len() < window + 1 || window == 0 {
            return 0.0;
        }
        let n = prices.len();
        let recent = &prices[n - window..];
        // First-order difference: (last - first) / window
        (recent[recent.len() - 1] - recent[0]) / window as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn area_eigenvalue_j_half() {
        let mut net = SpinNetwork2::new();
        net.nodes.push((0, 1));
        net.nodes.push((1, 1));
        net.edges.push((0, 1, 1)); // 2j=1 → j=0.5
        let l_p2 = 2.612e-70_f64; // m²
        let area = net.area_eigenvalue(0, l_p2);
        assert!(area > 0.0);
    }

    #[test]
    fn bounce_density_positive() {
        let cosmo = LqgCosmology::new(1.0, 1.0, 1.0);
        assert!(cosmo.bounce_density() > 0.0);
    }

    #[test]
    fn modified_friedmann_zero_at_bounce() {
        let cosmo = LqgCosmology::new(1.0, 1.0, 1.0);
        let rho_c = cosmo.bounce_density();
        let h_sq = cosmo.modified_friedmann(rho_c);
        assert!(h_sq.abs() < 1e-10, "H² should be ~0 at the bounce density");
    }

    #[test]
    fn area_gap_positive() {
        let gap = SpinFoamAmplitude::area_gap(1.0);
        assert!(gap > 0.0);
    }

    #[test]
    fn discrete_price_spectrum_monotone() {
        let prices = MarketLQG::discrete_price_spectrum(100.0, 5);
        assert_eq!(prices.len(), 5);
        for w in prices.windows(2) {
            assert!(w[1] > w[0]);
        }
    }

    #[test]
    fn bounce_scenario_finds_minimum() {
        let history = vec![100.0, 95.0, 90.0, 85.0, 88.0, 95.0];
        let idx = MarketLQG::bounce_scenario(&history);
        assert_eq!(idx, Some(3));
    }

    #[test]
    fn quantum_momentum_positive_trend() {
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0];
        let mom = MarketLQG::quantum_momentum(&prices, 4);
        assert!(mom > 0.0);
    }
}
