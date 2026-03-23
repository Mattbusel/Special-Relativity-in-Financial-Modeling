//! Causal structure inference between assets.
//!
//! ## Methods Implemented
//!
//! ### 1. Granger Causality
//!
//! **X Granger-causes Y** if lagged values of X improve the prediction of Y
//! beyond what Y's own lags can provide.  The test compares the residual sum
//! of squares (RSS) of two VAR models:
//!
//! - **Restricted**: `Y_t = a0 + Σ_{k=1}^{p} a_k * Y_{t-k} + ε_t`
//! - **Unrestricted**: `Y_t = a0 + Σ_{k=1}^{p} a_k * Y_{t-k} + Σ_{k=1}^{p} b_k * X_{t-k} + ε_t`
//!
//! The F-statistic:
//! ```text
//! F = ((RSS_r - RSS_u) / p) / (RSS_u / (n - 2p - 1))
//! ```
//! is compared against an approximate critical value. A simplified OLS is used.
//!
//! ### 2. Transfer Entropy
//!
//! **Transfer entropy** from X to Y measures the directional information flow:
//!
//! ```text
//! TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) * ln(p(y_{t+1} | y_t, x_t) / p(y_{t+1} | y_t))
//! ```
//!
//! This module approximates TE via a discrete binning of the continuous return
//! distributions into `n_bins` quantile bins.
//!
//! ### 3. Peters-Janzing-Schölkopf (PJS) Causal Direction
//!
//! The PJS heuristic infers causal direction by comparing the independence of
//! the residuals from the two possible regression directions:
//!
//! - Fit `Y ~ f(X)` and compute `residuals_YX`
//! - Fit `X ~ f(Y)` and compute `residuals_XY`
//! - The causal direction is `X → Y` if `residuals_YX` is more independent
//!   of X than `residuals_XY` is of Y (measured by correlation).
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::causal_inference::{CausalInference, CausalConfig};
//!
//! let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
//! let y: Vec<f64> = x.iter().map(|v| v * 0.8 + 0.01).collect(); // Y driven by X
//!
//! let ci = CausalInference::new(CausalConfig::default());
//! let gc = ci.granger_causality(&x, &y, 2);
//! println!("X Granger-causes Y: {} (F={:.3})", gc.x_causes_y, gc.f_stat_xy);
//!
//! let te = ci.transfer_entropy(&x, &y, 4);
//! println!("TE(X→Y)={:.4}, TE(Y→X)={:.4}", te.te_x_to_y, te.te_y_to_x);
//! ```

use std::collections::HashMap;

/// Configuration for causal inference algorithms.
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Number of quantile bins for transfer entropy estimation.
    pub n_bins: usize,
    /// Approximate critical F-value for Granger causality rejection (5% level,
    /// large sample).
    pub f_critical: f64,
    /// Minimum series length required for causal tests.
    pub min_length: usize,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            n_bins: 5,
            f_critical: 3.84, // approximate chi²(1)/1 at 5% level (large n)
            min_length: 20,
        }
    }
}

/// Result of a Granger causality test between two series.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrangerResult {
    /// F-statistic for the test `X → Y` (does X predict Y?).
    pub f_stat_xy: f64,
    /// F-statistic for the reverse test `Y → X`.
    pub f_stat_yx: f64,
    /// `true` when F_xy exceeds the critical threshold.
    pub x_causes_y: bool,
    /// `true` when F_yx exceeds the critical threshold.
    pub y_causes_x: bool,
    /// Number of lags used.
    pub lags: usize,
    /// Effective sample size used.
    pub n_obs: usize,
}

/// Result of a transfer entropy computation between two series.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct TransferEntropyResult {
    /// Transfer entropy from X to Y (bits, using natural log → nats).
    pub te_x_to_y: f64,
    /// Transfer entropy from Y to X.
    pub te_y_to_x: f64,
    /// Net directional information flow: `te_x_to_y - te_y_to_x`.
    pub net_flow: f64,
    /// Dominant causal direction: `"X→Y"`, `"Y→X"`, or `"bidirectional"`.
    pub dominant_direction: String,
    /// Number of bins used in probability estimation.
    pub n_bins: usize,
}

/// Result of the Peters-Janzing-Schölkopf causal direction test.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PjsResult {
    /// Residual-cause correlation for the `X → Y` model.
    pub corr_yx_residuals: f64,
    /// Residual-cause correlation for the `Y → X` model.
    pub corr_xy_residuals: f64,
    /// Inferred causal direction: `"X→Y"`, `"Y→X"`, or `"indeterminate"`.
    pub direction: String,
    /// Confidence as `|corr_xy_residuals| - |corr_yx_residuals|` (positive = X→Y).
    pub confidence: f64,
}

/// Aggregated causal summary for an asset pair.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CausalSummary {
    /// Granger causality test results.
    pub granger: GrangerResult,
    /// Transfer entropy results.
    pub transfer_entropy: TransferEntropyResult,
    /// PJS causal direction inference.
    pub pjs: PjsResult,
    /// Consensus causal direction (majority vote across methods).
    pub consensus: String,
}

/// Causal inference engine implementing Granger causality, transfer entropy,
/// and the Peters-Janzing-Schölkopf causal model.
pub struct CausalInference {
    cfg: CausalConfig,
}

impl CausalInference {
    /// Create a new engine.
    pub fn new(cfg: CausalConfig) -> Self {
        Self { cfg }
    }

    /// Run all three causal inference methods and return a consensus summary.
    pub fn infer(&self, x: &[f64], y: &[f64], lags: usize) -> CausalSummary {
        let granger = self.granger_causality(x, y, lags);
        let transfer_entropy = self.transfer_entropy(x, y, lags);
        let pjs = self.pjs_direction(x, y);

        // Majority vote among the three methods.
        let mut xy_votes = 0i32;
        let mut yx_votes = 0i32;

        if granger.x_causes_y { xy_votes += 1; }
        if granger.y_causes_x { yx_votes += 1; }
        if transfer_entropy.dominant_direction == "X→Y" { xy_votes += 1; }
        if transfer_entropy.dominant_direction == "Y→X" { yx_votes += 1; }
        if pjs.direction == "X→Y" { xy_votes += 1; }
        if pjs.direction == "Y→X" { yx_votes += 1; }

        let consensus = if xy_votes > yx_votes {
            "X→Y".to_string()
        } else if yx_votes > xy_votes {
            "Y→X".to_string()
        } else {
            "indeterminate".to_string()
        };

        CausalSummary { granger, transfer_entropy, pjs, consensus }
    }

    /// Granger causality test: does knowing X's past improve prediction of Y,
    /// and vice versa?
    ///
    /// Uses OLS regression with `lags` lagged values. Clamped to `lags >= 1`.
    pub fn granger_causality(&self, x: &[f64], y: &[f64], lags: usize) -> GrangerResult {
        let lags = lags.max(1);
        let n = x.len().min(y.len());

        let f_xy = self.granger_f_stat(y, x, lags, n);
        let f_yx = self.granger_f_stat(x, y, lags, n);

        GrangerResult {
            f_stat_xy: f_xy,
            f_stat_yx: f_yx,
            x_causes_y: f_xy > self.cfg.f_critical,
            y_causes_x: f_yx > self.cfg.f_critical,
            lags,
            n_obs: n.saturating_sub(lags),
        }
    }

    /// Transfer entropy from X to Y and Y to X via discrete binning.
    ///
    /// `lag` is the prediction horizon (default 1 is standard TE).
    pub fn transfer_entropy(&self, x: &[f64], y: &[f64], lag: usize) -> TransferEntropyResult {
        let lag = lag.max(1);
        let n = x.len().min(y.len());
        let bins = self.cfg.n_bins;

        let xb = quantile_bin(x, bins);
        let yb = quantile_bin(y, bins);

        let te_xy = compute_te(&xb, &yb, lag, bins);
        let te_yx = compute_te(&yb, &xb, lag, bins);
        let net_flow = te_xy - te_yx;

        let dominant_direction = if (net_flow).abs() < 1e-9 {
            "bidirectional".to_string()
        } else if net_flow > 0.0 {
            "X→Y".to_string()
        } else {
            "Y→X".to_string()
        };

        TransferEntropyResult {
            te_x_to_y: te_xy,
            te_y_to_x: te_yx,
            net_flow,
            dominant_direction,
            n_bins: bins,
        }
    }

    /// Peters-Janzing-Schölkopf causal direction heuristic.
    ///
    /// Compares the independence of residuals from `Y ~ X` vs `X ~ Y`.
    /// The direction is `X → Y` when residuals from `Y ~ X` are more
    /// independent of X (lower absolute correlation).
    pub fn pjs_direction(&self, x: &[f64], y: &[f64]) -> PjsResult {
        let n = x.len().min(y.len());
        let xn = &x[..n];
        let yn = &y[..n];

        // Fit Y ~ linear(X), compute residuals, correlate with X.
        let (_, residuals_yx) = ols_residuals(xn, yn);
        let corr_yx = pearson(xn, &residuals_yx);

        // Fit X ~ linear(Y), compute residuals, correlate with Y.
        let (_, residuals_xy) = ols_residuals(yn, xn);
        let corr_xy = pearson(yn, &residuals_xy);

        // Causal direction: lower |corr| of residuals with the cause → that is the cause.
        let confidence = corr_xy.abs() - corr_yx.abs();
        let direction = if confidence.abs() < 1e-6 {
            "indeterminate".to_string()
        } else if confidence > 0.0 {
            "X→Y".to_string() // residuals of Y~X are less correlated with X
        } else {
            "Y→X".to_string()
        };

        PjsResult { corr_yx_residuals: corr_yx, corr_xy_residuals: corr_xy, direction, confidence }
    }

    // ── private ───────────────────────────────────────────────────────────────

    /// Compute the Granger F-statistic for "does `cause` predict `target`?"
    fn granger_f_stat(&self, target: &[f64], cause: &[f64], lags: usize, n: usize) -> f64 {
        if n <= 2 * lags + 1 { return 0.0; }

        // Build restricted model: target ~ target_lags only.
        let (rss_r, _) = ols_lag_rss(target, &[], lags, n);
        // Build unrestricted model: target ~ target_lags + cause_lags.
        let (rss_u, _) = ols_lag_rss(target, cause, lags, n);

        let df_num = lags as f64;
        let df_denom = (n - 2 * lags - 1) as f64;
        if df_denom <= 0.0 || rss_u < 1e-14 { return 0.0; }

        ((rss_r - rss_u).max(0.0) / df_num) / (rss_u / df_denom)
    }
}

// ── OLS helpers ───────────────────────────────────────────────────────────────

/// Simple OLS: fit `y ~ intercept + slope * x`, return (fitted, residuals).
fn ols_residuals(x: &[f64], y: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let n = x.len().min(y.len());
    if n < 2 {
        return (vec![0.0; n], y[..n].to_vec());
    }
    let nf = n as f64;
    let mx = x.iter().sum::<f64>() / nf;
    let my = y[..n].iter().sum::<f64>() / nf;
    let mut cov = 0.0f64;
    let mut var = 0.0f64;
    for i in 0..n {
        cov += (x[i] - mx) * (y[i] - my);
        var += (x[i] - mx).powi(2);
    }
    let slope = if var.abs() < 1e-12 { 0.0 } else { cov / var };
    let intercept = my - slope * mx;
    let fitted: Vec<f64> = x[..n].iter().map(|xi| intercept + slope * xi).collect();
    let residuals: Vec<f64> = fitted.iter().zip(y[..n].iter()).map(|(f, yi)| yi - f).collect();
    (fitted, residuals)
}

/// OLS with lagged predictors: target_lags (always included) + optional cause_lags.
/// Returns (RSS_restricted_or_unrestricted, residuals).
///
/// Uses a simplified approach: mean-subtract each variable and regress.
/// For p lags the design matrix has dimension `(n-p) × (p or 2p)`.
fn ols_lag_rss(target: &[f64], cause: &[f64], lags: usize, n: usize) -> (f64, Vec<f64>) {
    let t = n.min(target.len());
    if t <= lags { return (0.0, Vec::new()); }
    let rows = t - lags;

    // Build response vector y.
    let y: Vec<f64> = (lags..t).map(|i| target[i]).collect();

    // Build design matrix X (rows × cols).
    // Columns: [1 (intercept), target_{t-1} .. target_{t-p}, cause_{t-1} .. cause_{t-p}]
    let has_cause = !cause.is_empty() && cause.len() >= t;
    let cols = 1 + lags + if has_cause { lags } else { 0 };
    let mut design = vec![vec![0.0f64; cols]; rows];
    for row in 0..rows {
        design[row][0] = 1.0; // intercept
        for lag in 1..=lags {
            let col = lag;
            let src_idx = lags + row - lag;
            design[row][col] = if src_idx < target.len() { target[src_idx] } else { 0.0 };
        }
        if has_cause {
            for lag in 1..=lags {
                let col = lags + lag;
                let src_idx = lags + row - lag;
                design[row][col] = if src_idx < cause.len() { cause[src_idx] } else { 0.0 };
            }
        }
    }

    // Normal equations: β = (X'X)^{-1} X'y  (Gram-Schmidt via Gram matrix inversion).
    // For simplicity use the closed-form for the restricted OLS via column-by-column regression.
    // This is an approximation for the multi-variate case but sufficient for the F-stat ratio.
    let beta = solve_normal_equations(&design, &y, cols, rows);
    let fitted: Vec<f64> = (0..rows).map(|r| {
        design[r].iter().zip(beta.iter()).map(|(d, b)| d * b).sum()
    }).collect();
    let rss: f64 = y.iter().zip(fitted.iter()).map(|(yi, fi)| (yi - fi).powi(2)).sum();
    let residuals: Vec<f64> = y.iter().zip(fitted.iter()).map(|(yi, fi)| yi - fi).collect();
    (rss, residuals)
}

/// Solve normal equations X'X β = X'y via Cholesky-free Gaussian elimination.
fn solve_normal_equations(design: &[Vec<f64>], y: &[f64], cols: usize, rows: usize) -> Vec<f64> {
    // Build X'X and X'y.
    let mut xtx = vec![vec![0.0f64; cols]; cols];
    let mut xty = vec![0.0f64; cols];
    for r in 0..rows {
        for i in 0..cols {
            xty[i] += design[r][i] * y[r];
            for j in 0..cols {
                xtx[i][j] += design[r][i] * design[r][j];
            }
        }
    }
    // Gaussian elimination with partial pivoting.
    gauss_solve(&xtx, &xty, cols)
}

/// Simple Gaussian elimination returning β or zero vector if singular.
fn gauss_solve(a: &[Vec<f64>], b: &[f64], n: usize) -> Vec<f64> {
    let mut mat: Vec<Vec<f64>> = a.to_vec();
    let mut rhs: Vec<f64> = b.to_vec();

    for col in 0..n {
        // Partial pivot.
        let max_row = (col..n)
            .max_by(|&r1, &r2| mat[r1][col].abs().partial_cmp(&mat[r2][col].abs()).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap_or(col);
        mat.swap(col, max_row);
        rhs.swap(col, max_row);

        let pivot = mat[col][col];
        if pivot.abs() < 1e-12 { continue; }
        for row in (col + 1)..n {
            let factor = mat[row][col] / pivot;
            for c in col..n {
                let v = mat[col][c] * factor;
                mat[row][c] -= v;
            }
            rhs[row] -= rhs[col] * factor;
        }
    }

    // Back substitution.
    let mut x = vec![0.0f64; n];
    for i in (0..n).rev() {
        if mat[i][i].abs() < 1e-12 { x[i] = 0.0; continue; }
        let mut s = rhs[i];
        for j in (i + 1)..n {
            s -= mat[i][j] * x[j];
        }
        x[i] = s / mat[i][i];
    }
    x
}

// ── Transfer entropy helpers ──────────────────────────────────────────────────

/// Bin continuous values into discrete quantile bins in `0..n_bins`.
fn quantile_bin(data: &[f64], n_bins: usize) -> Vec<usize> {
    if data.is_empty() || n_bins == 0 { return Vec::new(); }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = data.len();
    data.iter().map(|&v| {
        // Count how many sorted values are <= v → gives rank.
        let rank = sorted.partition_point(|&s| s < v);
        (rank * n_bins / n).min(n_bins - 1)
    }).collect()
}

/// Compute transfer entropy TE(source → target) using discrete symbols.
///
/// TE(X→Y) = Σ p(y_{t+1}, y_t, x_t) * log(p(y_{t+1}|y_t, x_t) / p(y_{t+1}|y_t))
fn compute_te(source: &[usize], target: &[usize], lag: usize, n_bins: usize) -> f64 {
    let n = source.len().min(target.len());
    if n <= lag { return 0.0; }

    // Count joint occurrences.
    let mut joint3: HashMap<(usize, usize, usize), f64> = HashMap::new(); // (y_next, y_t, x_t)
    let mut joint2: HashMap<(usize, usize), f64> = HashMap::new();         // (y_next, y_t)
    let mut cond2: HashMap<(usize, usize), f64> = HashMap::new();           // (y_t, x_t)
    let mut cond1: HashMap<usize, f64> = HashMap::new();                    // y_t

    let count = (n - lag) as f64;
    for t in 0..(n - lag) {
        let y_next = target[t + lag];
        let y_t = target[t];
        let x_t = source[t];
        *joint3.entry((y_next, y_t, x_t)).or_insert(0.0) += 1.0;
        *joint2.entry((y_next, y_t)).or_insert(0.0) += 1.0;
        *cond2.entry((y_t, x_t)).or_insert(0.0) += 1.0;
        *cond1.entry(y_t).or_insert(0.0) += 1.0;
    }

    let mut te = 0.0f64;
    for (&(y_next, y_t, x_t), &cnt) in &joint3 {
        let p_yyx = cnt / count;
        let p_yx = *cond2.get(&(y_t, x_t)).unwrap_or(&1.0) / count;
        let p_yy = *joint2.get(&(y_next, y_t)).unwrap_or(&1.0) / count;
        let p_y = *cond1.get(&y_t).unwrap_or(&1.0) / count;

        // p(y_next | y_t, x_t) / p(y_next | y_t)
        let cond_full = p_yyx / p_yx.max(1e-12);
        let cond_marg = p_yy / p_y.max(1e-12);
        let ratio = cond_full / cond_marg.max(1e-12);
        if ratio > 1e-12 {
            te += p_yyx * ratio.ln();
        }
    }
    let _ = n_bins; // used only for binning, suppress unused warning
    te
}

// ── Shared statistics ─────────────────────────────────────────────────────────

fn pearson(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    if n < 2 { return 0.0; }
    let nf = n as f64;
    let ma = a[..n].iter().sum::<f64>() / nf;
    let mb = b[..n].iter().sum::<f64>() / nf;
    let mut cov = 0.0f64;
    let mut va = 0.0f64;
    let mut vb = 0.0f64;
    for i in 0..n {
        let da = a[i] - ma;
        let db = b[i] - mb;
        cov += da * db;
        va += da * da;
        vb += db * db;
    }
    let denom = (va * vb).sqrt();
    if denom < 1e-12 { return 0.0; }
    cov / denom
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ci() -> CausalInference {
        CausalInference::new(CausalConfig::default())
    }

    fn linear_series(n: usize, slope: f64) -> Vec<f64> {
        (0..n).map(|i| i as f64 * slope).collect()
    }

    /// A series that causes Y (Y is a lagged version of X).
    fn causal_pair(n: usize) -> (Vec<f64>, Vec<f64>) {
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| if i == 0 { 0.0 } else { x[i - 1] * 0.9 }).collect();
        (x, y)
    }

    #[test]
    fn granger_result_fields_are_finite() {
        let (x, y) = causal_pair(50);
        let r = ci().granger_causality(&x, &y, 2);
        assert!(r.f_stat_xy.is_finite());
        assert!(r.f_stat_yx.is_finite());
    }

    #[test]
    fn granger_lags_clamped_to_one() {
        let x = linear_series(30, 0.01);
        let y = linear_series(30, 0.02);
        let r = ci().granger_causality(&x, &y, 0);
        assert_eq!(r.lags, 1);
    }

    #[test]
    fn transfer_entropy_non_negative() {
        let (x, y) = causal_pair(60);
        let te = ci().transfer_entropy(&x, &y, 1);
        assert!(te.te_x_to_y >= 0.0, "TE should be non-negative");
        assert!(te.te_y_to_x >= 0.0, "TE should be non-negative");
    }

    #[test]
    fn transfer_entropy_net_flow_direction() {
        let (x, y) = causal_pair(80);
        let te = ci().transfer_entropy(&x, &y, 1);
        // Net flow direction should match dominant_direction label.
        if te.net_flow > 1e-9 {
            assert_eq!(te.dominant_direction, "X→Y");
        } else if te.net_flow < -1e-9 {
            assert_eq!(te.dominant_direction, "Y→X");
        } else {
            assert_eq!(te.dominant_direction, "bidirectional");
        }
    }

    #[test]
    fn pjs_direction_returns_valid_label() {
        let (x, y) = causal_pair(50);
        let r = ci().pjs_direction(&x, &y);
        assert!(["X→Y", "Y→X", "indeterminate"].contains(&r.direction.as_str()));
    }

    #[test]
    fn infer_returns_valid_consensus() {
        let (x, y) = causal_pair(50);
        let summary = ci().infer(&x, &y, 1);
        assert!(["X→Y", "Y→X", "indeterminate"].contains(&summary.consensus.as_str()));
    }

    #[test]
    fn quantile_bin_produces_valid_bins() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let bins = quantile_bin(&data, 5);
        for b in &bins {
            assert!(*b < 5, "bin index should be < n_bins");
        }
    }

    #[test]
    fn pearson_identical_is_one() {
        let a: Vec<f64> = (0..20).map(|i| i as f64).collect();
        assert!((pearson(&a, &a) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn granger_short_series_does_not_panic() {
        let x = vec![0.01, 0.02];
        let y = vec![0.01, 0.02];
        let r = ci().granger_causality(&x, &y, 2);
        assert!(r.f_stat_xy.is_finite());
    }
}
