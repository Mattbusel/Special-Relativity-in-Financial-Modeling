//! Spin foam and loop quantum gravity analog for discrete market structure.
//!
//! Provides spin foam network representation and quantized market price levels.

use std::f64::consts::PI;

// ── SpinLabel ─────────────────────────────────────────────────────────────

pub type SpinLabel = f64;

// ── SpinFoamVertex ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpinFoamVertex {
    pub id: u32,
    pub spins: Vec<SpinLabel>,
    pub amplitude: f64,
}

// ── SpinFoamEdge ──────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct SpinFoamEdge {
    pub from_vertex: u32,
    pub to_vertex: u32,
    pub spin: SpinLabel,
}

// ── SpinFoam ──────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct SpinFoam {
    pub vertices: Vec<SpinFoamVertex>,
    pub edges: Vec<SpinFoamEdge>,
}

impl SpinFoam {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_vertex(&mut self, spins: Vec<SpinLabel>) -> u32 {
        let id = self.vertices.len() as u32;
        let amplitude = vertex_amplitude_from_spins(&spins);
        self.vertices.push(SpinFoamVertex {
            id,
            spins,
            amplitude,
        });
        id
    }

    pub fn add_edge(&mut self, from: u32, to: u32, spin: SpinLabel) -> usize {
        let idx = self.edges.len();
        self.edges.push(SpinFoamEdge {
            from_vertex: from,
            to_vertex: to,
            spin,
        });
        idx
    }
}

fn vertex_amplitude_from_spins(spins: &[SpinLabel]) -> f64 {
    spins.iter().map(|&j| 2.0 * j + 1.0).product()
}

/// Vertex amplitude: product of (2*j+1) for each spin j.
pub fn vertex_amplitude(v: &SpinFoamVertex) -> f64 {
    vertex_amplitude_from_spins(&v.spins)
}

/// Path amplitude: product of vertex amplitudes along a path of vertex ids.
pub fn path_amplitude(foam: &SpinFoam, path: &[u32]) -> f64 {
    path.iter()
        .filter_map(|&id| foam.vertices.iter().find(|v| v.id == id))
        .map(vertex_amplitude)
        .product()
}

/// Area eigenvalue in Planck units (G*hbar=1):
/// A = 8*pi*G*hbar * sqrt(j*(j+1))
pub fn area_eigenvalue(spin: SpinLabel) -> f64 {
    8.0 * PI * (spin * (spin + 1.0)).sqrt()
}

/// Simplified volume eigenvalue: (spin + 0.5)^(3/2)
pub fn volume_eigenvalue(spin: SpinLabel) -> f64 {
    (spin + 0.5_f64).powf(1.5)
}

// ── DiscreteMarket ────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct DiscreteMarket {
    pub foam: SpinFoam,
    pub tick_size: f64,
}

impl DiscreteMarket {
    pub fn new(tick_size: f64) -> Self {
        DiscreteMarket {
            foam: SpinFoam::new(),
            tick_size,
        }
    }

    /// Generate quantized price levels: base ± n*tick_size*sqrt(spin_label)
    /// Uses spin labels from foam vertices (defaults to 0.5 if none).
    pub fn quantized_price_levels(&self, base_price: f64, n_levels: usize) -> Vec<f64> {
        let spin = if self.foam.vertices.is_empty() {
            0.5
        } else {
            // Use first spin of first vertex as representative
            *self.foam.vertices[0].spins.first().unwrap_or(&0.5)
        };

        let mut levels = Vec::with_capacity(2 * n_levels + 1);
        let step = self.tick_size * spin.sqrt();

        // Generate from base - n*step to base + n*step
        for i in 0..=(2 * n_levels) {
            let offset = (i as f64 - n_levels as f64) * step;
            levels.push(base_price + offset);
        }
        levels.sort_by(|a, b| a.partial_cmp(b).unwrap());
        levels
    }

    /// Market foam entropy: sum of log(vertex_amplitude) over all vertices.
    pub fn market_foam_entropy(&self) -> f64 {
        self.foam
            .vertices
            .iter()
            .map(|v| vertex_amplitude(v).ln())
            .sum()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vertex_amplitude_for_half_spin() {
        // j=0.5: (2*0.5+1) = 2
        let v = SpinFoamVertex {
            id: 0,
            spins: vec![0.5],
            amplitude: 2.0,
        };
        assert!((vertex_amplitude(&v) - 2.0).abs() < 1e-9);
    }

    #[test]
    fn vertex_amplitude_for_j1() {
        // j=1.0: (2*1+1) = 3
        let v = SpinFoamVertex {
            id: 0,
            spins: vec![1.0],
            amplitude: 3.0,
        };
        assert!((vertex_amplitude(&v) - 3.0).abs() < 1e-9);
    }

    #[test]
    fn area_eigenvalue_positive() {
        assert!(area_eigenvalue(0.5) > 0.0);
        assert!(area_eigenvalue(1.0) > 0.0);
        assert!(area_eigenvalue(1.5) > 0.0);
    }

    #[test]
    fn path_amplitude_product() {
        let mut foam = SpinFoam::new();
        let v0 = foam.add_vertex(vec![0.5]); // amp = 2
        let v1 = foam.add_vertex(vec![1.0]); // amp = 3
        foam.add_edge(v0, v1, 0.5);
        let amp = path_amplitude(&foam, &[v0, v1]);
        assert!((amp - 6.0).abs() < 1e-9);
    }

    #[test]
    fn quantized_levels_monotone() {
        let mut market = DiscreteMarket::new(1.0);
        market.foam.add_vertex(vec![1.0]); // spin=1.0
        let levels = market.quantized_price_levels(100.0, 3);
        for i in 1..levels.len() {
            assert!(levels[i] >= levels[i - 1]);
        }
    }

    #[test]
    fn quantized_levels_count() {
        let mut market = DiscreteMarket::new(1.0);
        market.foam.add_vertex(vec![0.5]);
        let levels = market.quantized_price_levels(100.0, 3);
        assert_eq!(levels.len(), 7); // 2*3 + 1
    }

    #[test]
    fn entropy_positive() {
        let mut market = DiscreteMarket::new(1.0);
        market.foam.add_vertex(vec![0.5]); // amp=2, ln(2)>0
        market.foam.add_vertex(vec![1.0]); // amp=3, ln(3)>0
        let entropy = market.market_foam_entropy();
        assert!(entropy > 0.0);
    }

    #[test]
    fn add_vertex_returns_sequential_ids() {
        let mut foam = SpinFoam::new();
        let id0 = foam.add_vertex(vec![0.5]);
        let id1 = foam.add_vertex(vec![1.0]);
        assert_eq!(id0, 0);
        assert_eq!(id1, 1);
    }

    #[test]
    fn volume_eigenvalue_positive() {
        assert!(volume_eigenvalue(0.5) > 0.0);
        assert!(volume_eigenvalue(1.0) > 0.0);
    }
}
