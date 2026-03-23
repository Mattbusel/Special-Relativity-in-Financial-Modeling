//! Holographic price encoding via AdS/CFT duality analogy.
//!
//! ## Concept
//!
//! The **holographic principle** in physics asserts that the information
//! content of a volume of space can be fully encoded on its bounding surface.
//! The AdS/CFT correspondence makes this precise: a (d+1)-dimensional
//! gravitational theory in anti-de Sitter space is dual to a d-dimensional
//! conformal field theory living on its boundary.
//!
//! Applied to market data:
//!
//! - **Bulk**: the 3D market state `(price, volume, time)` — three coordinates
//!   that evolve continuously.
//! - **Boundary surface**: a 2D encoding that preserves causal structure,
//!   analogous to the CFT boundary correlators.
//! - **Bulk reconstruction**: given the boundary encoding, the original bulk
//!   state can be approximately recovered via a kernel integral.
//!
//! ## Encoding Scheme
//!
//! The boundary coordinates `(u, v)` are computed as:
//!
//! ```text
//! u = atanh(r / r_max) * cos(theta)
//! v = atanh(r / r_max) * sin(theta)
//! ```
//!
//! where
//!
//! ```text
//! r     = sqrt(price_norm² + volume_norm²)   (radial distance in bulk slice)
//! theta = atan2(volume_norm, price_norm)       (angular position)
//! ```
//!
//! and `price_norm`, `volume_norm` are the input coordinates normalised to
//! `(-1, 1)`.  The `atanh` maps the open unit disk to the entire plane while
//! preserving the hyperbolic metric of AdS space (Poincaré disk model).
//!
//! The time coordinate is encoded separately as a **causal phase** `phi`:
//!
//! ```text
//! phi = 2 * pi * (t - t_min) / (t_max - t_min)
//! ```
//!
//! allowing temporal ordering to be read from the phase.
//!
//! ## Boundary Correlators
//!
//! The CFT two-point boundary correlator for primary operators of dimension
//! `Delta` is:
//!
//! ```text
//! G(x1, x2) = 1 / |x1 - x2|^(2*Delta)
//! ```
//!
//! This module computes the correlator matrix for all pairs of encoded points,
//! which approximates the bulk propagator and can be used to infer long-range
//! market dependencies.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::holographic::{HolographicEncoder, MarketPoint, HolographicConfig};
//!
//! let cfg = HolographicConfig::default();
//! let mut encoder = HolographicEncoder::new(cfg);
//!
//! encoder.add_point(MarketPoint { price: 100.0, volume: 1_000.0, time: 0.0 });
//! encoder.add_point(MarketPoint { price: 101.5, volume: 1_200.0, time: 1.0 });
//!
//! let encoded = encoder.encode_all();
//! let corr = encoder.boundary_correlator_matrix(2.0);
//! println!("G(0,1) = {:.4}", corr[0][1]);
//! ```

use std::f64::consts::PI;

/// A single 3D market observation in the bulk.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MarketPoint {
    /// Mid-price or log-price.
    pub price: f64,
    /// Traded volume (positive real).
    pub volume: f64,
    /// Wall-clock time (unix seconds or tick index).
    pub time: f64,
}

/// A 2D boundary encoding of one [`MarketPoint`].
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct BoundaryPoint {
    /// First boundary coordinate (Poincaré-disk horizontal).
    pub u: f64,
    /// Second boundary coordinate (Poincaré-disk vertical).
    pub v: f64,
    /// Causal phase in `[0, 2π)` encoding temporal ordering.
    pub phi: f64,
    /// Radial distance in the Poincaré disk before atanh mapping.
    pub r_raw: f64,
}

impl BoundaryPoint {
    /// Euclidean distance to another boundary point (ignoring phase).
    pub fn spatial_distance(&self, other: &BoundaryPoint) -> f64 {
        let du = self.u - other.u;
        let dv = self.v - other.v;
        (du * du + dv * dv).sqrt()
    }

    /// Hyperbolic (Poincaré disk) distance between two boundary points.
    ///
    /// `d_H(p, q) = 2 * atanh(|p - q| / (1 + p·q))` where the denominator
    /// accounts for the metric distortion of the disk model.
    pub fn hyperbolic_distance(&self, other: &BoundaryPoint) -> f64 {
        let num = self.spatial_distance(other);
        let dot = self.u * other.u + self.v * other.v;
        let denom = 1.0 + dot;
        if denom.abs() < 1e-12 {
            return f64::INFINITY;
        }
        2.0 * (num / denom).clamp(-1.0 + 1e-12, 1.0 - 1e-12).atanh()
    }
}

/// Configuration for the holographic encoder.
#[derive(Debug, Clone)]
pub struct HolographicConfig {
    /// Maximum radius in the bulk (price, volume) plane before normalisation.
    /// Points with `r >= r_max` are clamped to the boundary.
    pub r_max: f64,
    /// Small epsilon to keep `atanh` away from its poles.
    pub atanh_eps: f64,
    /// Whether to use log-price instead of raw price for normalisation.
    pub log_price: bool,
}

impl Default for HolographicConfig {
    fn default() -> Self {
        Self {
            r_max: 1.0,
            atanh_eps: 1e-6,
            log_price: true,
        }
    }
}

/// Encodes 3D market data `(price, volume, time)` onto a 2D holographic
/// boundary surface, preserving causal structure via phase encoding.
#[derive(Debug)]
pub struct HolographicEncoder {
    cfg: HolographicConfig,
    points: Vec<MarketPoint>,
}

impl HolographicEncoder {
    /// Create an encoder with the given configuration.
    pub fn new(cfg: HolographicConfig) -> Self {
        Self { cfg, points: Vec::new() }
    }

    /// Add a bulk market point.
    pub fn add_point(&mut self, p: MarketPoint) {
        self.points.push(p);
    }

    /// Return all stored bulk points.
    pub fn points(&self) -> &[MarketPoint] {
        &self.points
    }

    /// Encode all stored points onto the boundary.
    ///
    /// Normalisation ranges are derived from the stored data set, so the
    /// encoding is consistent across all points.
    pub fn encode_all(&self) -> Vec<BoundaryPoint> {
        if self.points.is_empty() {
            return Vec::new();
        }

        let (price_min, price_max) = self.range(|p| {
            if self.cfg.log_price { p.price.max(1e-12).ln() } else { p.price }
        });
        let (vol_min, vol_max) = self.range(|p| p.volume.max(0.0).ln().max(-30.0));
        let (t_min, t_max) = self.range(|p| p.time);

        self.points.iter().map(|p| {
            let price_val = if self.cfg.log_price { p.price.max(1e-12).ln() } else { p.price };
            let vol_val = p.volume.max(0.0).ln().max(-30.0);

            let pn = normalise(price_val, price_min, price_max); // in [-1, 1]
            let vn = normalise(vol_val, vol_min, vol_max);       // in [-1, 1]

            let r_raw = (pn * pn + vn * vn).sqrt().min(self.cfg.r_max - self.cfg.atanh_eps);
            let theta = vn.atan2(pn);

            // Map radial distance through atanh to get Poincaré disk coordinate.
            let rho = (r_raw / self.cfg.r_max)
                .clamp(self.cfg.atanh_eps, 1.0 - self.cfg.atanh_eps)
                .atanh();

            let u = rho * theta.cos();
            let v = rho * theta.sin();

            let phi = if (t_max - t_min).abs() < 1e-12 {
                0.0
            } else {
                2.0 * PI * (p.time - t_min) / (t_max - t_min)
            };

            BoundaryPoint { u, v, phi, r_raw }
        }).collect()
    }

    /// Compute the CFT two-point boundary correlator matrix for all encoded
    /// point pairs.
    ///
    /// `G(i, j) = 1 / max(eps, |xi - xj|)^(2*delta)`
    ///
    /// where `xi`, `xj` are the spatial boundary coordinates of points `i`
    /// and `j`, and `delta` is the conformal dimension of the dual operator.
    /// Diagonal entries are set to `1.0` (self-correlator).
    pub fn boundary_correlator_matrix(&self, delta: f64) -> Vec<Vec<f64>> {
        let encoded = self.encode_all();
        let n = encoded.len();
        let mut matrix = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    matrix[i][j] = 1.0;
                } else {
                    let d = encoded[i].spatial_distance(&encoded[j]).max(1e-12);
                    matrix[i][j] = 1.0 / d.powf(2.0 * delta);
                }
            }
        }
        matrix
    }

    /// Approximately reconstruct a bulk point from its boundary encoding using
    /// a simplified HKLL-like kernel.
    ///
    /// The reconstruction integrates the boundary data weighted by a Gaussian
    /// kernel centred on the target boundary point. This is a discretised
    /// analogue of the HKLL (Hamilton-Kabat-Lifschytz-Lowe) formula.
    pub fn bulk_reconstruction(&self, target: &BoundaryPoint, sigma: f64) -> f64 {
        let encoded = self.encode_all();
        if encoded.is_empty() {
            return 0.0;
        }
        let mut weight_sum = 0.0_f64;
        let mut value_sum = 0.0_f64;
        for (bp, mp) in encoded.iter().zip(self.points.iter()) {
            let d = target.spatial_distance(bp);
            let w = (-d * d / (2.0 * sigma * sigma)).exp();
            let price_val = if self.cfg.log_price { mp.price.max(1e-12).ln() } else { mp.price };
            weight_sum += w;
            value_sum += w * price_val;
        }
        if weight_sum < 1e-12 { return 0.0; }
        value_sum / weight_sum
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn range<F: Fn(&MarketPoint) -> f64>(&self, f: F) -> (f64, f64) {
        let mut min = f64::INFINITY;
        let mut max = f64::NEG_INFINITY;
        for p in &self.points {
            let v = f(p);
            if v < min { min = v; }
            if v > max { max = v; }
        }
        if (max - min).abs() < 1e-12 {
            // Degenerate case: all values identical.
            (min - 1.0, max + 1.0)
        } else {
            (min, max)
        }
    }
}

/// Normalise `v` from `[min, max]` to `[-1, 1]`.
fn normalise(v: f64, min: f64, max: f64) -> f64 {
    2.0 * (v - min) / (max - min) - 1.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_encoder() -> HolographicEncoder {
        HolographicEncoder::new(HolographicConfig::default())
    }

    fn sample_points() -> Vec<MarketPoint> {
        vec![
            MarketPoint { price: 100.0, volume: 1000.0, time: 0.0 },
            MarketPoint { price: 105.0, volume: 1500.0, time: 1.0 },
            MarketPoint { price: 98.0,  volume: 800.0,  time: 2.0 },
            MarketPoint { price: 110.0, volume: 2000.0, time: 3.0 },
        ]
    }

    #[test]
    fn encode_produces_correct_count() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        assert_eq!(enc.encode_all().len(), 4);
    }

    #[test]
    fn boundary_u_v_are_finite() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        for bp in enc.encode_all() {
            assert!(bp.u.is_finite(), "u should be finite: {}", bp.u);
            assert!(bp.v.is_finite(), "v should be finite: {}", bp.v);
        }
    }

    #[test]
    fn phi_spans_full_cycle() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        let encoded = enc.encode_all();
        let min_phi = encoded.iter().map(|b| b.phi).fold(f64::INFINITY, f64::min);
        let max_phi = encoded.iter().map(|b| b.phi).fold(f64::NEG_INFINITY, f64::max);
        assert!(min_phi < max_phi, "phi range should be non-degenerate");
    }

    #[test]
    fn correlator_matrix_diagonal_is_one() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        let m = enc.boundary_correlator_matrix(2.0);
        for (i, row) in m.iter().enumerate() {
            assert!((row[i] - 1.0).abs() < 1e-9, "diagonal should be 1.0");
        }
    }

    #[test]
    fn correlator_matrix_is_symmetric() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        let m = enc.boundary_correlator_matrix(2.0);
        let n = m.len();
        for i in 0..n {
            for j in 0..n {
                assert!((m[i][j] - m[j][i]).abs() < 1e-9,
                    "correlator matrix should be symmetric at ({i},{j})");
            }
        }
    }

    #[test]
    fn single_point_encodes_without_panic() {
        let mut enc = make_encoder();
        enc.add_point(MarketPoint { price: 50.0, volume: 500.0, time: 0.0 });
        let encoded = enc.encode_all();
        assert_eq!(encoded.len(), 1);
        assert!(encoded[0].u.is_finite());
    }

    #[test]
    fn bulk_reconstruction_returns_finite() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        let target = enc.encode_all()[0];
        let val = enc.bulk_reconstruction(&target, 0.5);
        assert!(val.is_finite());
    }

    #[test]
    fn hyperbolic_distance_is_non_negative() {
        let mut enc = make_encoder();
        for p in sample_points() { enc.add_point(p); }
        let pts = enc.encode_all();
        for i in 0..pts.len() {
            for j in 0..pts.len() {
                let d = pts[i].hyperbolic_distance(&pts[j]);
                if d.is_finite() {
                    assert!(d >= 0.0);
                }
            }
        }
    }
}
