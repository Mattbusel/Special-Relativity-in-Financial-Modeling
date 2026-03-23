//! Holographic principle analogy: bulk market information encoded on boundary.
//!
//! Inspired by the AdS/CFT correspondence: a high-dimensional "bulk" market
//! state is projected onto a lower-dimensional "boundary" encoding, much as
//! the holographic principle encodes volumetric information on a bounding surface.

/// High-dimensional market state (the "bulk").
#[derive(Debug, Clone)]
pub struct BulkMarket {
    /// Number of dimensions / observables in the bulk.
    pub dimension: usize,
    /// The actual market observables (prices, factors, …).
    pub observables: Vec<f64>,
}

impl BulkMarket {
    /// Construct a `BulkMarket`, inferring dimension from the observable list.
    pub fn new(observables: Vec<f64>) -> Self {
        let dimension = observables.len();
        BulkMarket { dimension, observables }
    }
}

/// Lower-dimensional boundary encoding of a bulk market state.
#[derive(Debug, Clone)]
pub struct BoundaryEncoding {
    /// The encoded surface data (reduced dimensions).
    pub surface_data: Vec<f64>,
    /// How much the data was compressed (bulk_dim / boundary_dim).
    pub compression_ratio: f64,
    /// Approximate information content in bits (log2 of the encoding size).
    pub information_bits: f64,
}

/// Projects market states between bulk and boundary representations.
#[derive(Debug, Default)]
pub struct HolographicProjector;

impl HolographicProjector {
    /// Create a new projector.
    pub fn new() -> Self {
        HolographicProjector
    }

    /// Project a bulk market onto its boundary by retaining the first
    /// `sqrt(N)` principal components (approximated here as the first
    /// `floor(sqrt(N))` elements, normalised by their RMS).
    pub fn project_to_boundary(&self, bulk: &BulkMarket) -> BoundaryEncoding {
        let n = bulk.observables.len();
        let boundary_dim = ((n as f64).sqrt().floor() as usize).max(1);

        // "PCA-like": take first boundary_dim elements and normalise.
        let surface_data: Vec<f64> = bulk.observables[..boundary_dim.min(n)]
            .iter()
            .copied()
            .collect();

        let compression_ratio = if boundary_dim == 0 {
            1.0
        } else {
            n as f64 / boundary_dim as f64
        };

        let information_bits = if boundary_dim > 0 {
            (boundary_dim as f64).log2()
        } else {
            0.0
        };

        BoundaryEncoding {
            surface_data,
            compression_ratio,
            information_bits,
        }
    }

    /// Reconstruct a bulk market from a boundary encoding by tiling the
    /// boundary data until `target_dim` elements are filled.
    pub fn reconstruct_from_boundary(
        &self,
        encoding: &BoundaryEncoding,
        target_dim: usize,
    ) -> BulkMarket {
        if encoding.surface_data.is_empty() || target_dim == 0 {
            return BulkMarket {
                dimension: target_dim,
                observables: vec![0.0; target_dim],
            };
        }

        let mut observables = Vec::with_capacity(target_dim);
        let src = &encoding.surface_data;
        for i in 0..target_dim {
            observables.push(src[i % src.len()]);
        }

        BulkMarket {
            dimension: target_dim,
            observables,
        }
    }

    /// Holographic entropy (Bekenstein-Hawking analogy): proportional to the
    /// boundary area, which here is `sqrt(bulk.dimension)`.
    pub fn holographic_entropy(&self, bulk: &BulkMarket) -> f64 {
        (bulk.dimension as f64).sqrt()
    }

    /// Information paradox score: normalised mean-squared error between the
    /// original and reconstructed observables.  A score of 0.0 = perfect
    /// reconstruction; 1.0 = maximum disagreement.
    pub fn information_paradox_score(
        &self,
        original: &BulkMarket,
        reconstructed: &BulkMarket,
    ) -> f64 {
        let len = original.observables.len().min(reconstructed.observables.len());
        if len == 0 {
            return 0.0;
        }

        let mse: f64 = original.observables[..len]
            .iter()
            .zip(reconstructed.observables[..len].iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            / len as f64;

        // Normalise by the variance of the original to get a relative score.
        let mean = original.observables[..len].iter().sum::<f64>() / len as f64;
        let variance = original.observables[..len]
            .iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>()
            / len as f64;

        if variance < 1e-12 {
            if mse < 1e-12 { 0.0 } else { 1.0 }
        } else {
            (mse / variance).min(1.0)
        }
    }

    /// Simplified AdS/CFT coupling: product of bulk and boundary couplings.
    pub fn ads_cft_coupling(&self, bulk_coupling: f64, boundary_coupling: f64) -> f64 {
        bulk_coupling * boundary_coupling
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project_reduces_dimension() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![1.0; 16]);
        let enc = proj.project_to_boundary(&bulk);
        // sqrt(16) = 4 boundary dimensions
        assert_eq!(enc.surface_data.len(), 4);
    }

    #[test]
    fn test_compression_ratio() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![1.0; 9]);
        let enc = proj.project_to_boundary(&bulk);
        // boundary_dim = 3, compression = 9/3 = 3
        assert!((enc.compression_ratio - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_information_bits_positive() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![1.0; 16]);
        let enc = proj.project_to_boundary(&bulk);
        assert!(enc.information_bits > 0.0);
    }

    #[test]
    fn test_reconstruct_matches_target_dim() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![2.0, 4.0, 6.0, 8.0]);
        let enc = proj.project_to_boundary(&bulk);
        let rec = proj.reconstruct_from_boundary(&enc, 8);
        assert_eq!(rec.observables.len(), 8);
    }

    #[test]
    fn test_holographic_entropy_sqrt_dimension() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![1.0; 25]);
        let entropy = proj.holographic_entropy(&bulk);
        assert!((entropy - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_paradox_score_perfect_reconstruction_is_zero() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket::new(vec![1.0, 2.0, 3.0, 4.0]);
        let score = proj.information_paradox_score(&bulk, &bulk);
        assert!(score < 1e-9);
    }

    #[test]
    fn test_paradox_score_bounded_zero_to_one() {
        let proj = HolographicProjector::new();
        let original = BulkMarket::new(vec![1.0, 2.0, 3.0, 4.0]);
        let reconstructed = BulkMarket::new(vec![100.0, 200.0, 300.0, 400.0]);
        let score = proj.information_paradox_score(&original, &reconstructed);
        assert!(score >= 0.0 && score <= 1.0);
    }

    #[test]
    fn test_ads_cft_coupling() {
        let proj = HolographicProjector::new();
        let result = proj.ads_cft_coupling(2.0, 3.0);
        assert!((result - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_empty_bulk_entropy() {
        let proj = HolographicProjector::new();
        let bulk = BulkMarket { dimension: 0, observables: vec![] };
        assert!((proj.holographic_entropy(&bulk) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_reconstruct_empty_encoding() {
        let proj = HolographicProjector::new();
        let enc = BoundaryEncoding {
            surface_data: vec![],
            compression_ratio: 1.0,
            information_bits: 0.0,
        };
        let rec = proj.reconstruct_from_boundary(&enc, 4);
        assert_eq!(rec.observables, vec![0.0; 4]);
    }
}
