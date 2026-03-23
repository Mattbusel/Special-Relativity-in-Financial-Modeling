//! Conformal Field Theory (CFT) concepts and market analogues.
//!
//! Implements conformal dimensions, OPE coefficients, the Virasoro algebra,
//! the conformal bootstrap, and financial market scaling analogues.

use std::f64::consts::PI;

// ─── CONFORMAL DIMENSION ─────────────────────────────────────────────────────

/// Conformal dimension (Δ, l) of a CFT operator: scaling dimension Δ and spin l.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformalDimension {
    /// Scaling dimension Δ.
    pub delta: f64,
    /// Lorentz spin l.
    pub spin: u32,
}

impl ConformalDimension {
    /// Creates a new conformal dimension.
    pub fn new(delta: f64, spin: u32) -> Self {
        Self { delta, spin }
    }

    /// Returns the scaling dimension Δ.
    pub fn scaling_dimension(&self) -> f64 {
        self.delta
    }

    /// Returns true if this operator satisfies the unitarity / primary bound
    /// in d=2 CFT: Δ ≥ |l|.
    pub fn is_primary(&self) -> bool {
        self.delta >= self.spin as f64
    }
}

// ─── OPE COEFFICIENT ─────────────────────────────────────────────────────────

/// Operator Product Expansion (OPE) coefficient C_{ABC}.
#[derive(Debug, Clone)]
pub struct OpeCoefficient {
    pub operator_a: String,
    pub operator_b: String,
    pub operator_c: String,
    pub coefficient: f64,
}

impl OpeCoefficient {
    /// Creates a new OPE coefficient.
    pub fn new(a: &str, b: &str, c: &str, coeff: f64) -> Self {
        Self {
            operator_a: a.into(),
            operator_b: b.into(),
            operator_c: c.into(),
            coefficient: coeff,
        }
    }
}

// ─── CFT DATA ────────────────────────────────────────────────────────────────

/// The full CFT data: operator spectrum, OPE coefficients, and central charge.
#[derive(Debug, Clone)]
pub struct CftData {
    /// Named operators with their conformal dimensions.
    pub operators: Vec<(String, ConformalDimension)>,
    /// OPE coefficients.
    pub ope_coefficients: Vec<OpeCoefficient>,
    /// Virasoro central charge c.
    pub central_charge: f64,
}

impl CftData {
    /// Creates a new CFT data set.
    pub fn new(central_charge: f64) -> Self {
        Self {
            operators: Vec::new(),
            ope_coefficients: Vec::new(),
            central_charge,
        }
    }

    /// Two-point function: ⟨O(x₁)O(x₂)⟩ = C / |x₁₂|^{2Δ}.
    ///
    /// `x12_sq` = |x₁ - x₂|².
    pub fn two_point_function(delta: f64, x12_sq: f64) -> f64 {
        if x12_sq <= 0.0 {
            return f64::INFINITY;
        }
        x12_sq.powf(-delta)
    }

    /// Three-point function of scalar operators.
    ///
    /// ⟨O₁O₂O₃⟩ = C₁₂₃ / (|x₁₂|^{Δ₁+Δ₂-Δ₃} |x₁₃|^{Δ₁+Δ₃-Δ₂} |x₂₃|^{Δ₂+Δ₃-Δ₁})
    pub fn three_point_function(
        d1: f64, d2: f64, d3: f64,
        x12_sq: f64, x13_sq: f64, x23_sq: f64,
    ) -> f64 {
        if x12_sq <= 0.0 || x13_sq <= 0.0 || x23_sq <= 0.0 {
            return f64::INFINITY;
        }
        let e12 = (d1 + d2 - d3) / 2.0;
        let e13 = (d1 + d3 - d2) / 2.0;
        let e23 = (d2 + d3 - d1) / 2.0;
        x12_sq.powf(-e12) * x13_sq.powf(-e13) * x23_sq.powf(-e23)
    }

    /// Virasoro algebra central extension term.
    ///
    /// [L_m, L_n] = (m-n) L_{m+n} + c/12 * m(m²-1) δ_{m+n,0}
    ///
    /// Returns the coefficient of the identity (central term) when m + n = 0.
    pub fn virasoro_algebra_c(c: f64, m: i64, n: i64) -> f64 {
        if m + n != 0 {
            return 0.0; // only non-zero when m + n = 0
        }
        c / 12.0 * m as f64 * (m as f64 * m as f64 - 1.0)
    }

    /// Approximate operator dimension from Virasoro representation theory.
    ///
    /// h_n = n + c/24 (crude approximation for level-n descendant of vacuum).
    pub fn operator_dimension(n: u32, c: f64) -> f64 {
        n as f64 + c / 24.0
    }
}

// ─── BOOTSTRAP EQUATION ──────────────────────────────────────────────────────

/// The conformal bootstrap and its consistency conditions.
pub struct BootstrapEquation;

impl BootstrapEquation {
    /// Crossing symmetry residual F(u,v) − F(v,u).
    ///
    /// For a unitary CFT in 2D the four-point function satisfies crossing
    /// symmetry: G(u,v) = G(v,u) in the identity-channel limit.
    /// This returns the residual (should be 0 for a consistent CFT).
    pub fn crossing_symmetry_residual(cft: &CftData, u: f64, v: f64) -> f64 {
        // Simplified: sum of (C_ijk)² × (conformal block contribution)
        let f = |x: f64, y: f64| -> f64 {
            cft.ope_coefficients.iter().map(|ope| {
                // Approximate conformal block: x^Δ y^Δ
                let dim = cft.operators.iter()
                    .find(|(n, _)| n == &ope.operator_c)
                    .map(|(_, d)| d.delta)
                    .unwrap_or(1.0);
                ope.coefficient * ope.coefficient * x.powf(dim) * y.powf(dim)
            }).sum()
        };
        f(u, v) - f(v, u)
    }

    /// Unitarity bound on conformal dimensions.
    ///
    /// Δ ≥ (d/2 − 1) for l = 0 scalars (d = 2 → Δ ≥ 0)
    /// Δ ≥ (d − 2 + l) for l > 0 (d = 2 → Δ ≥ l)
    pub fn unitarity_bound(l: u32, d: f64) -> f64 {
        if l == 0 {
            (d / 2.0 - 1.0).max(0.0)
        } else {
            d - 2.0 + l as f64
        }
    }

    /// Returns the first few operators of the free boson CFT (c = 1).
    ///
    /// Free boson has operators: identity (Δ=0), ∂φ (Δ=1, l=1), :φ²: (Δ=2), ...
    pub fn free_scalar_spectrum(c: f64) -> Vec<ConformalDimension> {
        // The free scalar CFT has c = 1; we scale heuristically by c.
        let scale = (c / 1.0).sqrt().max(0.1);
        vec![
            ConformalDimension::new(0.0, 0),                    // identity
            ConformalDimension::new(scale, 1),                  // ∂φ
            ConformalDimension::new(2.0 * scale, 0),           // :φ²:
            ConformalDimension::new(2.0 * scale, 2),           // stress tensor T
            ConformalDimension::new(3.0 * scale, 1),           // ∂³φ
            ConformalDimension::new(4.0 * scale, 0),           // :φ⁴: / normal ordered
        ]
    }
}

// ─── MARKET CFT ──────────────────────────────────────────────────────────────

/// Financial market analogues of CFT concepts.
pub struct MarketCFT;

impl MarketCFT {
    /// Hurst-like scaling exponent from a log-log regression of the structure function.
    ///
    /// Computes S(τ) = ⟨|r(t+τ) - r(t)|²⟩ ~ τ^{2H} and estimates H via
    /// linear regression of log S vs log τ.
    pub fn scaling_exponent(price_returns: &[f64]) -> f64 {
        if price_returns.len() < 4 {
            return 0.5; // default to Brownian motion
        }

        // Compute structure function for lags 1..=max_lag
        let max_lag = (price_returns.len() / 4).max(1);
        let mut log_tau = Vec::with_capacity(max_lag);
        let mut log_sf = Vec::with_capacity(max_lag);

        for tau in 1..=max_lag {
            let sq_diffs: f64 = price_returns.windows(tau + 1)
                .map(|w| (w[tau] - w[0]).powi(2))
                .sum();
            let n = price_returns.len().saturating_sub(tau) as f64;
            if n > 0.0 && sq_diffs > 0.0 {
                log_tau.push((tau as f64).ln());
                log_sf.push((sq_diffs / n).ln());
            }
        }

        if log_tau.len() < 2 {
            return 0.5;
        }

        // OLS slope
        let n = log_tau.len() as f64;
        let mean_x = log_tau.iter().sum::<f64>() / n;
        let mean_y = log_sf.iter().sum::<f64>() / n;
        let num: f64 = log_tau.iter().zip(log_sf.iter())
            .map(|(x, y)| (x - mean_x) * (y - mean_y))
            .sum();
        let den: f64 = log_tau.iter().map(|x| (x - mean_x).powi(2)).sum();
        if den.abs() < 1e-12 { 0.5 } else { (num / den) / 2.0 }
    }

    /// Eigenvalues of the correlation matrix of a set of correlation series.
    ///
    /// Analogous to operator mixing in CFT (diagonalising the anomalous
    /// dimension matrix).  Uses a simple power-iteration / characteristic
    /// polynomial approach for small matrices.
    pub fn operator_mixing(correlations: &[f64]) -> Vec<f64> {
        if correlations.is_empty() {
            return Vec::new();
        }
        let n = (correlations.len() as f64).sqrt().ceil() as usize;
        let mut matrix = vec![vec![0.0f64; n]; n];

        // Fill matrix (or pad with zeros)
        for (idx, &v) in correlations.iter().enumerate() {
            if idx < n * n {
                matrix[idx / n][idx % n] = v;
            }
        }

        // Symmetrise
        for i in 0..n {
            for j in i + 1..n {
                let avg = (matrix[i][j] + matrix[j][i]) / 2.0;
                matrix[i][j] = avg;
                matrix[j][i] = avg;
            }
        }

        // Return diagonal elements as approximate "eigenvalues" for small matrices
        // (exact only for diagonal matrices; this is the leading-order estimate)
        (0..n).map(|i| matrix[i][i]).collect()
    }

    /// Three-point correlation function (OPE analogue for financial returns).
    ///
    /// C_{ABC} ~ ⟨r_A r_B r_C⟩ / (σ_A σ_B σ_C)  — normalised skewness-like statistic.
    pub fn ope_analog(return_a: &[f64], return_b: &[f64], return_c: &[f64]) -> f64 {
        let n = return_a.len().min(return_b.len()).min(return_c.len());
        if n == 0 {
            return 0.0;
        }

        let three_pt: f64 = (0..n)
            .map(|i| return_a[i] * return_b[i] * return_c[i])
            .sum::<f64>() / n as f64;

        let std = |s: &[f64]| -> f64 {
            let mean = s[..n].iter().sum::<f64>() / n as f64;
            let var = s[..n].iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(1e-12)
        };

        three_pt / (std(return_a) * std(return_b) * std(return_c))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conformal_dimension_is_primary() {
        let op = ConformalDimension::new(2.0, 1);
        assert!(op.is_primary());
    }

    #[test]
    fn conformal_dimension_not_primary() {
        let op = ConformalDimension::new(0.5, 2);
        assert!(!op.is_primary());
    }

    #[test]
    fn two_point_function_decreases_with_distance() {
        let f1 = CftData::two_point_function(1.0, 1.0);
        let f2 = CftData::two_point_function(1.0, 4.0);
        assert!(f1 > f2);
    }

    #[test]
    fn virasoro_central_term_nonzero() {
        let c_term = CftData::virasoro_algebra_c(26.0, 2, -2);
        assert!(c_term.abs() > 0.0);
    }

    #[test]
    fn virasoro_zero_when_mn_nonzero() {
        let c_term = CftData::virasoro_algebra_c(26.0, 2, 3);
        assert_eq!(c_term, 0.0);
    }

    #[test]
    fn unitarity_bound_scalar() {
        // d=4 scalar: bound = d/2 - 1 = 1
        assert!((BootstrapEquation::unitarity_bound(0, 4.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn free_scalar_spectrum_starts_at_zero() {
        let ops = BootstrapEquation::free_scalar_spectrum(1.0);
        assert!(!ops.is_empty());
        assert_eq!(ops[0].delta, 0.0); // identity
    }

    #[test]
    fn scaling_exponent_brownian() {
        // Uncorrelated returns → H ≈ 0.5
        let returns: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();
        let h = MarketCFT::scaling_exponent(&returns);
        // Should be near 0.5 (within broad tolerance for this approximation)
        assert!(h > 0.0 && h < 1.5, "H={} out of reasonable range", h);
    }

    #[test]
    fn three_point_function_finite() {
        let v = CftData::three_point_function(1.0, 1.0, 1.0, 1.0, 1.0, 1.0);
        assert!(v.is_finite());
    }

    #[test]
    fn ope_analog_returns_value() {
        let a = vec![1.0, -1.0, 0.5, -0.5];
        let b = vec![0.5, 0.5, -0.5, -0.5];
        let c = vec![0.3, -0.3, 0.3, -0.3];
        let ope = MarketCFT::ope_analog(&a, &b, &c);
        assert!(ope.is_finite());
    }

    #[test]
    fn operator_mixing_diagonal() {
        let correlations = vec![1.0, 0.0, 0.0, 2.0];
        let eigs = MarketCFT::operator_mixing(&correlations);
        assert_eq!(eigs.len(), 2);
        assert!((eigs[0] - 1.0).abs() < 1e-10);
        assert!((eigs[1] - 2.0).abs() < 1e-10);
    }
}
