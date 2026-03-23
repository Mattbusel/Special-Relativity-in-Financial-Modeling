//! Spin foam / spin network quantum gravity for the SRFM pipeline.
//!
//! Spin networks are combinatorial structures that describe the quantum
//! geometry of space in loop quantum gravity. Spin foams extend these to
//! spacetime histories.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// HalfInteger
// ---------------------------------------------------------------------------

/// Represents a half-integer spin quantum number j = twice_j / 2.
///
/// This covers j ∈ {0, 1/2, 1, 3/2, 2, ...}.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HalfInteger {
    /// Twice the spin value (so j = twice_j / 2).
    pub twice_j: u32,
}

impl HalfInteger {
    /// Create from the doubled value (e.g., `twice_j = 1` means j = 1/2).
    pub const fn new(twice_j: u32) -> Self {
        Self { twice_j }
    }

    /// The spin value as a floating-point number.
    pub fn value(&self) -> f64 {
        self.twice_j as f64 / 2.0
    }

    /// The Hilbert space dimension for this spin: 2j + 1.
    pub fn dimension(&self) -> u32 {
        self.twice_j + 1
    }
}

impl std::fmt::Display for HalfInteger {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.twice_j % 2 == 0 {
            write!(f, "{}", self.twice_j / 2)
        } else {
            write!(f, "{}/2", self.twice_j)
        }
    }
}

// ---------------------------------------------------------------------------
// SpinNode
// ---------------------------------------------------------------------------

/// A node (vertex) in a spin network, carrying a spin label and an
/// intertwiner index that selects the invariant subspace.
#[derive(Debug, Clone)]
pub struct SpinNode {
    /// Unique node identifier.
    pub id: usize,
    /// Spin label at this node.
    pub spin: HalfInteger,
    /// Intertwiner index (selects invariant tensor at the node).
    pub intertwiner: u32,
}

// ---------------------------------------------------------------------------
// SpinEdge
// ---------------------------------------------------------------------------

/// A directed edge in a spin network, labelled with a spin.
#[derive(Debug, Clone)]
pub struct SpinEdge {
    /// Source node id.
    pub from: usize,
    /// Target node id.
    pub to: usize,
    /// Spin label on this edge (represents area eigenvalue).
    pub spin: HalfInteger,
}

// ---------------------------------------------------------------------------
// SpinNetwork
// ---------------------------------------------------------------------------

/// A spin network — a graph with spin-labelled edges and intertwiner nodes.
#[derive(Debug, Clone)]
pub struct SpinNetwork {
    pub nodes: Vec<SpinNode>,
    pub edges: Vec<SpinEdge>,
}

impl SpinNetwork {
    /// Create an empty spin network.
    pub fn new() -> Self {
        Self { nodes: Vec::new(), edges: Vec::new() }
    }

    /// Add a node with the given spin; returns the node's id.
    pub fn add_node(&mut self, spin: HalfInteger) -> usize {
        let id = self.nodes.len();
        self.nodes.push(SpinNode { id, spin, intertwiner: 0 });
        id
    }

    /// Add a directed edge between two nodes.
    pub fn add_edge(&mut self, from: usize, to: usize, spin: HalfInteger) {
        self.edges.push(SpinEdge { from, to, spin });
    }

    /// Check admissibility: at every node the triangle inequality must hold
    /// for all pairs of incident edge spins.
    ///
    /// Triangle inequality: |j1 - j2| ≤ j3 ≤ j1 + j2 (using twice_j arithmetic).
    pub fn admissible(&self) -> bool {
        for node in &self.nodes {
            let incident: Vec<HalfInteger> = self.edges.iter()
                .filter(|e| e.from == node.id || e.to == node.id)
                .map(|e| e.spin)
                .collect();

            if incident.len() < 2 {
                continue;
            }
            // Check all triples.
            for i in 0..incident.len() {
                for j in (i + 1)..incident.len() {
                    let a = incident[i].twice_j;
                    let b = incident[j].twice_j;
                    // Need at least one other edge to form a triple; if only
                    // two edges, treat the node as a 2-valent coupling.
                    let diff = a.abs_diff(b);
                    let sum  = a + b;
                    // There must exist some c in [diff, sum] with same parity.
                    if diff > sum {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compute the area eigenvalue for an edge in Planck units.
    ///
    /// A = 8πγ * l_P² * √(j(j+1))
    ///
    /// where γ is the Barbero-Immirzi parameter (fixed at 0.2375 here).
    pub fn area_eigenvalue(&self, edge_id: usize, planck_area: f64) -> f64 {
        let gamma = 0.2375_f64; // Barbero-Immirzi parameter
        let j = self.edges[edge_id].spin.value();
        8.0 * PI * gamma * planck_area * (j * (j + 1.0)).sqrt()
    }

    /// Compute a simplified volume eigenvalue for a node.
    ///
    /// V ~ l_P³ * √(product of j_i(j_i+1) for incident edges) (simplified).
    pub fn volume_eigenvalue(&self, node_id: usize, planck_volume: f64) -> f64 {
        let incident_spins: Vec<f64> = self.edges.iter()
            .filter(|e| e.from == node_id || e.to == node_id)
            .map(|e| { let j = e.spin.value(); j * (j + 1.0) })
            .collect();

        if incident_spins.is_empty() {
            return 0.0;
        }
        let product: f64 = incident_spins.iter().product();
        planck_volume * product.sqrt()
    }

    /// Build a spin network from asset correlations.
    ///
    /// Each asset becomes a node; the absolute correlation between pairs
    /// becomes (rounded to nearest half-integer) the edge spin.
    pub fn market_spin_network(assets: &[(&str, f64)]) -> SpinNetwork {
        let mut net = SpinNetwork::new();
        let n = assets.len();

        // Add one node per asset, spin proportional to (normalised) weight.
        for &(_, weight) in assets {
            let twice_j = ((weight.abs() * 4.0).round() as u32).max(1);
            net.add_node(HalfInteger::new(twice_j));
        }

        // Add edges between all pairs using correlation proxy.
        for i in 0..n {
            for j in (i + 1)..n {
                let corr = (assets[i].1 * assets[j].1).abs().sqrt();
                let twice_j = ((corr * 4.0).round() as u32).max(1);
                net.add_edge(i, j, HalfInteger::new(twice_j));
            }
        }

        net
    }
}

impl Default for SpinNetwork {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// VertexAmplitude
// ---------------------------------------------------------------------------

/// Amplitude associated with a vertex (4-simplex) in a spin foam.
#[derive(Debug, Clone)]
pub struct VertexAmplitude {
    /// Index of the vertex.
    pub vertex_id: usize,
    /// Numerical amplitude value (simplified EPRL-like).
    pub amplitude: f64,
}

// ---------------------------------------------------------------------------
// SpinFoam
// ---------------------------------------------------------------------------

/// A spin foam — a sequence of spin network slices representing a spacetime
/// history (sum-over-histories path integral for quantum gravity).
pub struct SpinFoam {
    slices: Vec<SpinNetwork>,
}

impl SpinFoam {
    /// Create an empty spin foam.
    pub fn new() -> Self {
        Self { slices: Vec::new() }
    }

    /// Append a spatial slice (spin network) to the foam.
    pub fn add_slice(&mut self, network: SpinNetwork) {
        self.slices.push(network);
    }

    /// Compute the total amplitude as the product of per-vertex contributions.
    ///
    /// Each vertex amplitude is approximated as the product of
    /// √(2j+1) for all edges incident to nodes in that slice.
    pub fn amplitude(&self) -> f64 {
        if self.slices.is_empty() {
            return 0.0;
        }
        let mut total = 1.0_f64;
        for (vi, slice) in self.slices.iter().enumerate() {
            let va = self.vertex_amplitude_for_slice(vi, slice);
            total *= va.amplitude;
        }
        total
    }

    fn vertex_amplitude_for_slice(&self, vertex_id: usize, slice: &SpinNetwork) -> VertexAmplitude {
        let amp: f64 = slice.edges.iter()
            .map(|e| (e.spin.dimension() as f64).sqrt())
            .product();
        VertexAmplitude { vertex_id, amplitude: amp.max(1e-300) }
    }

    /// Compute a simplified partition function as a weighted sum of amplitude
    /// powers scaled by the coupling constant.
    ///
    /// Z(g) = Σ_history  exp(-g * |amplitude|)
    pub fn partition_function(&self, coupling: f64) -> f64 {
        if self.slices.is_empty() {
            return 1.0;
        }
        let amp = self.amplitude();
        (-coupling * amp.abs()).exp()
    }
}

impl Default for SpinFoam {
    fn default() -> Self {
        Self::new()
    }
}
