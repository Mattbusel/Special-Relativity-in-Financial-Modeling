//! Information geometry metrics for market regime detection.
//!
//! ## Concept
//!
//! **Information geometry** treats the set of probability distributions as a
//! Riemannian manifold where the metric tensor is the **Fisher information
//! matrix**.  The geodesic distance between two distributions on this manifold
//! — the Fisher-Rao distance — measures the statistical distinguishability of
//! the two distributions.
//!
//! Applied to financial markets:
//!
//! - Each rolling window of returns defines a probability distribution (modelled
//!   here as a Gaussian `N(μ, σ²)`).
//! - The Fisher information metric on the Gaussian family is:
//!
//!   ```text
//!   ds² = (dμ)² / σ² + 2 (dσ)² / σ²
//!   ```
//!
//!   giving the Fisher-Rao geodesic distance:
//!
//!   ```text
//!   d_FR((μ1,σ1), (μ2,σ2)) = sqrt(2) * |ln(σ2/σ1)| ... (exact formula below)
//!   ```
//!
//! - A large jump in Fisher-Rao distance signals a **regime change**: the
//!   distribution of returns has shifted in a statistically meaningful way.
//!
//! ## Fisher-Rao Distance (Gaussian Family)
//!
//! For two Gaussians `N(μ1, σ1²)` and `N(μ2, σ2²)`, the exact geodesic
//! distance on the statistical manifold is:
//!
//! ```text
//! d = sqrt(2) * acosh(1 + ((μ1-μ2)² + 2(σ1²+σ2²)) / (4*σ1*σ2))
//! ```
//!
//! This is the Siegel metric on the upper half-plane `H² = {(μ, σ) : σ > 0}`.
//!
//! ## Kullback-Leibler Divergence
//!
//! The KL divergence from `P` to `Q` for Gaussians is:
//!
//! ```text
//! KL(P||Q) = ln(σ2/σ1) + (σ1² + (μ1-μ2)²) / (2*σ2²) - 1/2
//! ```
//!
//! The symmetric divergence `(KL(P||Q) + KL(Q||P)) / 2` is used as an
//! alternative dissimilarity measure.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::information_geometry::{
//!     MarketDistribution, InformationGeometry, GeometryConfig,
//! };
//!
//! let cfg = GeometryConfig::default();
//! let geo = InformationGeometry::new(cfg);
//!
//! let returns: Vec<f64> = vec![0.01, -0.02, 0.015, 0.005, -0.01, 0.02, -0.005, 0.008];
//! let windows = geo.rolling_distributions(&returns, 4);
//! let distances = geo.geodesic_distances(&windows);
//! println!("Regime change distance: {:.4}", distances[0]);
//! ```

/// Configuration for information geometry computations.
#[derive(Debug, Clone)]
pub struct GeometryConfig {
    /// Minimum standard deviation to avoid degenerate distributions.
    pub min_sigma: f64,
    /// Threshold Fisher-Rao distance above which a regime change is flagged.
    pub regime_change_threshold: f64,
}

impl Default for GeometryConfig {
    fn default() -> Self {
        Self {
            min_sigma: 1e-8,
            regime_change_threshold: 0.5,
        }
    }
}

/// A Gaussian distribution `N(μ, σ²)` representing market return statistics
/// over one rolling window.
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub struct MarketDistribution {
    /// Sample mean of returns.
    pub mean: f64,
    /// Sample standard deviation of returns (> 0).
    pub sigma: f64,
    /// Start index of the window in the original returns series.
    pub window_start: usize,
    /// End index (exclusive) of the window.
    pub window_end: usize,
}

impl MarketDistribution {
    /// Create a market distribution from a slice of returns.
    ///
    /// `min_sigma` guards against degenerate (zero-variance) windows.
    pub fn from_returns(returns: &[f64], window_start: usize, min_sigma: f64) -> Self {
        let n = returns.len();
        let mean = if n == 0 { 0.0 } else { returns.iter().sum::<f64>() / n as f64 };
        let variance = if n < 2 {
            min_sigma * min_sigma
        } else {
            let v = returns.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
            v.max(min_sigma * min_sigma)
        };
        Self {
            mean,
            sigma: variance.sqrt(),
            window_start,
            window_end: window_start + n,
        }
    }
}

/// A detected regime change event.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RegimeChange {
    /// Index of the distribution pair (between windows `index` and `index+1`).
    pub index: usize,
    /// Fisher-Rao geodesic distance between the two distributions.
    pub fisher_rao_distance: f64,
    /// Symmetric KL divergence at this transition.
    pub symmetric_kl: f64,
    /// The "from" distribution.
    pub from: MarketDistribution,
    /// The "to" distribution.
    pub to: MarketDistribution,
}

/// Information geometry engine for computing geodesic distances on the
/// statistical manifold of Gaussian market distributions.
pub struct InformationGeometry {
    cfg: GeometryConfig,
}

impl InformationGeometry {
    /// Create a new engine.
    pub fn new(cfg: GeometryConfig) -> Self {
        Self { cfg }
    }

    /// Partition `returns` into overlapping rolling windows of size `window`
    /// and fit a Gaussian to each window.
    ///
    /// Returns an empty vector when `returns.len() < window`.
    pub fn rolling_distributions(&self, returns: &[f64], window: usize) -> Vec<MarketDistribution> {
        if returns.len() < window || window == 0 {
            return Vec::new();
        }
        (0..=(returns.len() - window))
            .map(|i| {
                MarketDistribution::from_returns(&returns[i..i + window], i, self.cfg.min_sigma)
            })
            .collect()
    }

    /// Compute the Fisher-Rao geodesic distance between all consecutive pairs
    /// of distributions.
    ///
    /// Returns a vector of length `distributions.len() - 1`.
    pub fn geodesic_distances(&self, distributions: &[MarketDistribution]) -> Vec<f64> {
        distributions
            .windows(2)
            .map(|w| self.fisher_rao_distance(&w[0], &w[1]))
            .collect()
    }

    /// Fisher-Rao geodesic distance between two Gaussians on the statistical
    /// manifold (Siegel half-plane metric).
    ///
    /// Formula:
    /// ```text
    /// d = sqrt(2) * acosh(1 + ((μ1-μ2)² + 2(σ1²+σ2²)) / (4*σ1*σ2))
    /// ```
    pub fn fisher_rao_distance(&self, p: &MarketDistribution, q: &MarketDistribution) -> f64 {
        let s1 = p.sigma.max(self.cfg.min_sigma);
        let s2 = q.sigma.max(self.cfg.min_sigma);
        let dm = p.mean - q.mean;
        let numer = dm * dm + 2.0 * (s1 * s1 + s2 * s2);
        let denom = 4.0 * s1 * s2;
        let arg = 1.0 + numer / denom;
        // arg is always >= 1, so acosh is real.
        2.0_f64.sqrt() * arg.max(1.0).acosh()
    }

    /// KL divergence from `p` to `q`: `KL(P||Q)`.
    pub fn kl_divergence(&self, p: &MarketDistribution, q: &MarketDistribution) -> f64 {
        let s1 = p.sigma.max(self.cfg.min_sigma);
        let s2 = q.sigma.max(self.cfg.min_sigma);
        let dm = p.mean - q.mean;
        (s2 / s1).ln() + (s1 * s1 + dm * dm) / (2.0 * s2 * s2) - 0.5
    }

    /// Symmetric KL divergence: `(KL(P||Q) + KL(Q||P)) / 2`.
    pub fn symmetric_kl(&self, p: &MarketDistribution, q: &MarketDistribution) -> f64 {
        (self.kl_divergence(p, q) + self.kl_divergence(q, p)) / 2.0
    }

    /// Detect regime changes in a return series.
    ///
    /// Returns all windows where the Fisher-Rao distance exceeds
    /// `cfg.regime_change_threshold`.
    pub fn detect_regimes(
        &self,
        returns: &[f64],
        window: usize,
    ) -> Vec<RegimeChange> {
        let dists = self.rolling_distributions(returns, window);
        dists
            .windows(2)
            .enumerate()
            .filter_map(|(i, w)| {
                let fr = self.fisher_rao_distance(&w[0], &w[1]);
                if fr > self.cfg.regime_change_threshold {
                    Some(RegimeChange {
                        index: i,
                        fisher_rao_distance: fr,
                        symmetric_kl: self.symmetric_kl(&w[0], &w[1]),
                        from: w[0],
                        to: w[1],
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Compute the full pairwise geodesic distance matrix for a set of
    /// distributions.  The matrix is symmetric with zeros on the diagonal.
    pub fn geodesic_matrix(&self, distributions: &[MarketDistribution]) -> Vec<Vec<f64>> {
        let n = distributions.len();
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    m[i][j] = self.fisher_rao_distance(&distributions[i], &distributions[j]);
                }
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn geo() -> InformationGeometry {
        InformationGeometry::new(GeometryConfig::default())
    }

    fn sample_returns() -> Vec<f64> {
        vec![
            0.01, -0.02, 0.015, 0.005, -0.01,
            0.10, -0.12, 0.11,  0.09, -0.10, // Regime shift here
            0.01,  0.00, 0.005,-0.005, 0.002,
        ]
    }

    #[test]
    fn rolling_distributions_correct_count() {
        let returns = sample_returns();
        let window = 5;
        let dists = geo().rolling_distributions(&returns, window);
        assert_eq!(dists.len(), returns.len() - window + 1);
    }

    #[test]
    fn rolling_distributions_empty_when_window_too_large() {
        let returns = vec![0.01, 0.02];
        let dists = geo().rolling_distributions(&returns, 10);
        assert!(dists.is_empty());
    }

    #[test]
    fn fisher_rao_distance_is_zero_for_identical_distributions() {
        let p = MarketDistribution { mean: 0.01, sigma: 0.05, window_start: 0, window_end: 10 };
        let d = geo().fisher_rao_distance(&p, &p);
        assert!(d < 1e-9, "distance to self should be ~0, got {d}");
    }

    #[test]
    fn fisher_rao_distance_is_positive_and_finite() {
        let p = MarketDistribution { mean: 0.01, sigma: 0.02, window_start: 0, window_end: 5 };
        let q = MarketDistribution { mean: 0.05, sigma: 0.10, window_start: 5, window_end: 10 };
        let d = geo().fisher_rao_distance(&p, &q);
        assert!(d > 0.0 && d.is_finite());
    }

    #[test]
    fn fisher_rao_distance_is_symmetric() {
        let p = MarketDistribution { mean: 0.01, sigma: 0.02, window_start: 0, window_end: 5 };
        let q = MarketDistribution { mean: 0.05, sigma: 0.10, window_start: 5, window_end: 10 };
        let g = geo();
        let d_pq = g.fisher_rao_distance(&p, &q);
        let d_qp = g.fisher_rao_distance(&q, &p);
        assert!((d_pq - d_qp).abs() < 1e-9);
    }

    #[test]
    fn kl_divergence_identical_is_zero() {
        let p = MarketDistribution { mean: 0.01, sigma: 0.05, window_start: 0, window_end: 10 };
        let d = geo().kl_divergence(&p, &p);
        assert!(d.abs() < 1e-9);
    }

    #[test]
    fn kl_divergence_is_non_negative() {
        let p = MarketDistribution { mean: 0.0, sigma: 0.01, window_start: 0, window_end: 5 };
        let q = MarketDistribution { mean: 0.05, sigma: 0.05, window_start: 5, window_end: 10 };
        let g = geo();
        assert!(g.kl_divergence(&p, &q) >= 0.0);
        assert!(g.kl_divergence(&q, &p) >= 0.0);
    }

    #[test]
    fn detect_regimes_returns_results() {
        let returns = sample_returns();
        // With a low threshold, at least one regime change should be found.
        let cfg = GeometryConfig { regime_change_threshold: 0.0, ..GeometryConfig::default() };
        let g = InformationGeometry::new(cfg);
        let changes = g.detect_regimes(&returns, 5);
        assert!(!changes.is_empty());
    }

    #[test]
    fn geodesic_matrix_diagonal_is_zero() {
        let dists: Vec<MarketDistribution> = (0..4).map(|i| MarketDistribution {
            mean: i as f64 * 0.01,
            sigma: 0.02 + i as f64 * 0.005,
            window_start: i,
            window_end: i + 5,
        }).collect();
        let m = geo().geodesic_matrix(&dists);
        for i in 0..m.len() {
            assert!(m[i][i].abs() < 1e-9);
        }
    }

    #[test]
    fn geodesic_matrix_is_symmetric() {
        let dists: Vec<MarketDistribution> = (0..3).map(|i| MarketDistribution {
            mean: i as f64 * 0.02,
            sigma: 0.01 + i as f64 * 0.01,
            window_start: i,
            window_end: i + 5,
        }).collect();
        let m = geo().geodesic_matrix(&dists);
        let n = m.len();
        for i in 0..n {
            for j in 0..n {
                assert!((m[i][j] - m[j][i]).abs() < 1e-9);
            }
        }
    }
}
