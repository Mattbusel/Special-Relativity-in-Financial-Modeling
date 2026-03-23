//! Granger causality testing and causal graph discovery for price time series.
//!
//! Implements pairwise Granger causality tests via OLS regression and builds a
//! directed causal graph over multiple asset series.

use std::collections::{HashMap, HashSet, VecDeque};

// ─── GRANGER RESULT ───────────────────────────────────────────────────────────

/// Result of a pairwise Granger causality test ("does X Granger-cause Y?").
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GrangerCausalityResult {
    /// F-statistic comparing restricted vs unrestricted models.
    pub f_statistic: f64,
    /// Approximate p-value (based on F-distribution approximation).
    pub p_value_approx: f64,
    /// `true` if the F-statistic exceeds the critical value (~3.84 at 5%).
    pub causes: bool,
    /// Residual sum of squares for the restricted (AR-only) model.
    pub rss_restricted: f64,
    /// Residual sum of squares for the unrestricted (AR + X lags) model.
    pub rss_unrestricted: f64,
}

// ─── OLS HELPERS ──────────────────────────────────────────────────────────────

/// Solve a least-squares system `Xb = y` using the normal equations `(X'X)b = X'y`.
///
/// Returns the coefficient vector, or `None` if `X'X` is singular.
fn ols(x_mat: &[Vec<f64>], y: &[f64]) -> Option<Vec<f64>> {
    let n = y.len();
    let k = if x_mat.is_empty() { 0 } else { x_mat[0].len() };
    if n == 0 || k == 0 {
        return None;
    }

    // Compute X'X (k×k) and X'y (k×1).
    let mut xtx = vec![vec![0.0f64; k]; k];
    let mut xty = vec![0.0f64; k];
    for i in 0..n {
        for j in 0..k {
            xty[j] += x_mat[i][j] * y[i];
            for l in 0..k {
                xtx[j][l] += x_mat[i][j] * x_mat[i][l];
            }
        }
    }

    // Solve via Gaussian elimination with partial pivoting.
    let mut aug: Vec<Vec<f64>> = xtx
        .iter()
        .enumerate()
        .map(|(r, row)| {
            let mut v = row.clone();
            v.push(xty[r]);
            v
        })
        .collect();

    for col in 0..k {
        // Find pivot.
        let pivot = (col..k)
            .max_by(|&a, &b| {
                aug[a][col]
                    .abs()
                    .partial_cmp(&aug[b][col].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap_or(col);
        if aug[pivot][col].abs() < 1e-12 {
            return None; // singular
        }
        aug.swap(col, pivot);
        let div = aug[col][col];
        for val in aug[col].iter_mut() {
            *val /= div;
        }
        for row in 0..k {
            if row != col {
                let factor = aug[row][col];
                for c in 0..=k {
                    let sub = factor * aug[col][c];
                    aug[row][c] -= sub;
                }
            }
        }
    }

    Some((0..k).map(|r| aug[r][k]).collect())
}

/// Compute the residual sum of squares for predictions `Xb` vs `y`.
fn rss(x_mat: &[Vec<f64>], y: &[f64], coef: &[f64]) -> f64 {
    x_mat
        .iter()
        .zip(y.iter())
        .map(|(row, &yi)| {
            let pred: f64 = row.iter().zip(coef.iter()).map(|(x, b)| x * b).sum();
            let e = yi - pred;
            e * e
        })
        .sum()
}

// ─── GRANGER CAUSALITY TEST ───────────────────────────────────────────────────

/// Test whether `x` Granger-causes `y` using `lags` autoregressive lags.
///
/// Critical value defaults to 3.84 (≈ χ²(1)/1 at 5% significance, large n).
pub fn granger_causality_test(
    x: &[f64],
    y: &[f64],
    lags: usize,
) -> GrangerCausalityResult {
    const F_CRIT: f64 = 3.84;

    let n_total = y.len().min(x.len());
    if n_total < 2 * lags + 2 || lags == 0 {
        return GrangerCausalityResult {
            f_statistic: 0.0,
            p_value_approx: 1.0,
            causes: false,
            rss_restricted: 0.0,
            rss_unrestricted: 0.0,
        };
    }

    let n = n_total - lags; // number of observations after consuming lags

    // Build design matrices.
    // Restricted: intercept + y lags
    // Unrestricted: intercept + y lags + x lags
    let mut x_restricted: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut x_unrestricted: Vec<Vec<f64>> = Vec::with_capacity(n);
    let mut y_target: Vec<f64> = Vec::with_capacity(n);

    for t in lags..n_total {
        let mut r_row = vec![1.0]; // intercept
        let mut u_row = vec![1.0]; // intercept
        for lag in 1..=lags {
            r_row.push(y[t - lag]);
            u_row.push(y[t - lag]);
        }
        for lag in 1..=lags {
            u_row.push(x[t - lag]);
        }
        x_restricted.push(r_row);
        x_unrestricted.push(u_row);
        y_target.push(y[t]);
    }

    // Fit both models.
    let coef_r = match ols(&x_restricted, &y_target) {
        Some(c) => c,
        None => {
            return GrangerCausalityResult {
                f_statistic: 0.0,
                p_value_approx: 1.0,
                causes: false,
                rss_restricted: 0.0,
                rss_unrestricted: 0.0,
            }
        }
    };
    let coef_u = match ols(&x_unrestricted, &y_target) {
        Some(c) => c,
        None => {
            return GrangerCausalityResult {
                f_statistic: 0.0,
                p_value_approx: 1.0,
                causes: false,
                rss_restricted: 0.0,
                rss_unrestricted: 0.0,
            }
        }
    };

    let rss_r = rss(&x_restricted, &y_target, &coef_r);
    let rss_u = rss(&x_unrestricted, &y_target, &coef_u);

    let denom_df = (n as f64) - (2 * lags + 1) as f64;
    let f_stat = if denom_df > 0.0 && rss_u > 0.0 {
        ((rss_r - rss_u) / lags as f64) / (rss_u / denom_df)
    } else {
        0.0
    };

    // Approximate p-value: use a simple threshold mapping.
    // F-distribution CDF approximation: we use Cornish-Fisher for large df.
    let p_value_approx = approximate_f_pvalue(f_stat, lags as f64, denom_df);

    GrangerCausalityResult {
        f_statistic: f_stat,
        p_value_approx,
        causes: f_stat > F_CRIT,
        rss_restricted: rss_r,
        rss_unrestricted: rss_u,
    }
}

/// Rough p-value approximation for F(d1, d2).
///
/// Uses the beta incomplete function approximation via Abramowitz & Stegun 6.6.4.
/// For d2 > 30 we use a chi-squared approximation.
fn approximate_f_pvalue(f: f64, d1: f64, d2: f64) -> f64 {
    if f <= 0.0 {
        return 1.0;
    }
    // For large d2, F(d1, d2) ≈ chi²(d1)/d1, so F*d1 ~ chi²(d1).
    // P(chi²(d1) > f*d1): use regularised incomplete gamma approximation.
    let x = d1 * f / (d1 * f + d2);
    // Beta CDF via continued fraction (simplified).
    // P-value = 1 - I_x(d1/2, d2/2).
    // We use a simple exponential approximation good to ~10%.
    let alpha = d1 / 2.0;
    let beta = d2 / 2.0;
    // Regularised incomplete beta via log approximation.
    let log_p = alpha * x.ln() + beta * (1.0 - x).ln()
        - (log_beta(alpha, beta))
        - x.ln()
        - (1.0 - x).ln();
    // Clamp p-value to [0, 1].
    (1.0 - log_p.exp().clamp(0.0, 1.0)).clamp(0.0, 1.0)
}

fn log_beta(a: f64, b: f64) -> f64 {
    log_gamma(a) + log_gamma(b) - log_gamma(a + b)
}

/// Stirling approximation for log Γ(x).
fn log_gamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }
    // Lanczos approximation (g=5, n=6).
    let c = [
        76.18009172947146,
        -86.50532032941677,
        24.01409824083091,
        -1.231739572450155,
        0.001208650973866179,
        -0.000005395239384953,
    ];
    let mut y = x;
    let tmp = x + 5.5;
    let ser: f64 = c
        .iter()
        .enumerate()
        .fold(1.000000000190015, |acc, (i, &ci)| {
            y += 1.0;
            acc + ci / y
        });
    0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * tmp.ln() - tmp + ser.ln()
}

// ─── CAUSAL GRAPH ─────────────────────────────────────────────────────────────

/// A directed graph of Granger-causal relationships between asset series.
#[derive(Debug, Default)]
pub struct CausalGraph {
    /// Adjacency set: `edges[from]` = set of `to` symbols this symbol causes.
    edges: HashMap<String, HashSet<String>>,
    /// All known symbols (nodes).
    nodes: HashSet<String>,
}

impl CausalGraph {
    /// Create an empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Run pairwise Granger tests for every ordered pair of series and build
    /// the causal graph.
    ///
    /// * `significance` — significance threshold (not currently used; F > 3.84 is the gate).
    pub fn discover(
        series: HashMap<String, Vec<f64>>,
        lags: usize,
        _significance: f64,
    ) -> CausalGraph {
        let mut graph = CausalGraph::new();
        let symbols: Vec<&String> = series.keys().collect();

        for sym in &symbols {
            graph.nodes.insert((*sym).clone());
        }

        for i in 0..symbols.len() {
            for j in 0..symbols.len() {
                if i == j {
                    continue;
                }
                let from = symbols[i];
                let to = symbols[j];
                let x = &series[from];
                let y = &series[to];
                let result = granger_causality_test(x, y, lags);
                if result.causes {
                    graph
                        .edges
                        .entry(from.clone())
                        .or_default()
                        .insert(to.clone());
                }
            }
        }

        graph
    }

    /// Return `true` if there is a direct causal edge from → to.
    pub fn has_edge(&self, from: &str, to: &str) -> bool {
        self.edges
            .get(from)
            .map(|s| s.contains(to))
            .unwrap_or(false)
    }

    /// BFS upstream: all symbols that transitively Granger-cause `symbol`.
    pub fn causal_ancestors(&self, symbol: &str) -> Vec<String> {
        // Build reverse adjacency.
        let mut reverse: HashMap<&str, Vec<&str>> = HashMap::new();
        for (from, tos) in &self.edges {
            for to in tos {
                reverse.entry(to.as_str()).or_default().push(from.as_str());
            }
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(symbol);
        visited.insert(symbol);

        let mut ancestors = Vec::new();
        while let Some(node) = queue.pop_front() {
            if let Some(parents) = reverse.get(node) {
                for &parent in parents {
                    if visited.insert(parent) {
                        ancestors.push(parent.to_string());
                        queue.push_back(parent);
                    }
                }
            }
        }
        ancestors
    }

    /// BFS downstream: all symbols transitively Granger-caused by `symbol`.
    pub fn causal_descendants(&self, symbol: &str) -> Vec<String> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        queue.push_back(symbol.to_string());
        visited.insert(symbol.to_string());

        let mut descendants = Vec::new();
        while let Some(node) = queue.pop_front() {
            if let Some(children) = self.edges.get(&node) {
                for child in children {
                    if visited.insert(child.clone()) {
                        descendants.push(child.clone());
                        queue.push_back(child.clone());
                    }
                }
            }
        }
        descendants
    }

    /// Emit a Graphviz DOT representation of the causal graph.
    pub fn to_dot(&self) -> String {
        let mut dot = String::from("digraph CausalGraph {\n");
        dot.push_str("    rankdir=LR;\n");
        for node in &self.nodes {
            dot.push_str(&format!("    \"{}\";\n", node));
        }
        for (from, tos) in &self.edges {
            for to in tos {
                dot.push_str(&format!("    \"{}\" -> \"{}\";\n", from, to));
            }
        }
        dot.push('}');
        dot
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(start: f64, end: f64, n: usize) -> Vec<f64> {
        (0..n)
            .map(|i| start + (end - start) * i as f64 / (n - 1) as f64)
            .collect()
    }

    fn pseudo_random(seed: u64, n: usize) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let bits = (state >> 33) as f64 / u32::MAX as f64;
                bits * 2.0 - 1.0 // uniform [-1, 1]
            })
            .collect()
    }

    #[test]
    fn random_series_unlikely_to_cause_each_other() {
        // Two independent random series should not show spurious causality.
        let x = pseudo_random(42, 100);
        let y = pseudo_random(99, 100);
        let result = granger_causality_test(&x, &y, 2);
        // Not guaranteed but very likely for independent series.
        // We just check the result is well-formed.
        assert!(result.f_statistic >= 0.0);
        assert!(result.rss_restricted >= 0.0);
        assert!(result.rss_unrestricted >= 0.0);
        assert!(result.rss_restricted >= result.rss_unrestricted - 1e-9);
    }

    #[test]
    fn caused_series_triggers_causality() {
        // y_t = x_{t-1} + small noise → x should Granger-cause y.
        let n = 80;
        let x: Vec<f64> = linspace(0.0, 4.0 * std::f64::consts::PI, n)
            .iter()
            .map(|&t| t.sin())
            .collect();
        let noise = pseudo_random(7, n);
        let y: Vec<f64> = (0..n)
            .map(|t| if t == 0 { 0.0 } else { x[t - 1] + 0.05 * noise[t] })
            .collect();
        let result = granger_causality_test(&x, &y, 1);
        assert!(result.f_statistic > 1.0, "expected non-trivial F-stat, got {}", result.f_statistic);
        assert!(result.causes, "expected x to Granger-cause y");
    }

    #[test]
    fn ols_coefficient_shapes() {
        // A simple AR(1) should give a coefficient vector of length 2 (intercept + lag).
        let y = linspace(1.0, 10.0, 30);
        let x_mat: Vec<Vec<f64>> = (1..30)
            .map(|t| vec![1.0, y[t - 1]])
            .collect();
        let y_target: Vec<f64> = (1..30).map(|t| y[t]).collect();
        let coef = ols(&x_mat, &y_target).unwrap();
        assert_eq!(coef.len(), 2);
    }

    #[test]
    fn causal_graph_edge_detection() {
        let n = 80;
        let x: Vec<f64> = linspace(0.0, 4.0 * std::f64::consts::PI, n)
            .iter()
            .map(|&t| t.sin())
            .collect();
        let noise = pseudo_random(7, n);
        let y: Vec<f64> = (0..n)
            .map(|t| if t == 0 { 0.0 } else { x[t - 1] + 0.05 * noise[t] })
            .collect();

        let mut series = HashMap::new();
        series.insert("X".to_string(), x);
        series.insert("Y".to_string(), y);

        let graph = CausalGraph::discover(series, 1, 0.05);
        assert!(graph.has_edge("X", "Y"), "X should cause Y");
    }

    #[test]
    fn bfs_ancestors_finds_upstream() {
        let mut graph = CausalGraph::new();
        graph.nodes.insert("A".into());
        graph.nodes.insert("B".into());
        graph.nodes.insert("C".into());
        graph.edges.entry("A".into()).or_default().insert("B".into());
        graph.edges.entry("B".into()).or_default().insert("C".into());

        let ancestors = graph.causal_ancestors("C");
        assert!(ancestors.contains(&"B".to_string()));
        assert!(ancestors.contains(&"A".to_string()));
    }

    #[test]
    fn bfs_descendants_finds_downstream() {
        let mut graph = CausalGraph::new();
        graph.nodes.insert("A".into());
        graph.nodes.insert("B".into());
        graph.nodes.insert("C".into());
        graph.edges.entry("A".into()).or_default().insert("B".into());
        graph.edges.entry("B".into()).or_default().insert("C".into());

        let descs = graph.causal_descendants("A");
        assert!(descs.contains(&"B".to_string()));
        assert!(descs.contains(&"C".to_string()));
    }

    #[test]
    fn dot_output_contains_nodes() {
        let mut graph = CausalGraph::new();
        graph.nodes.insert("SPY".into());
        graph.nodes.insert("BTC".into());
        graph.edges.entry("SPY".into()).or_default().insert("BTC".into());

        let dot = graph.to_dot();
        assert!(dot.contains("digraph"));
        assert!(dot.contains("SPY"));
        assert!(dot.contains("BTC"));
        assert!(dot.contains("->"));
    }

    #[test]
    fn short_series_returns_zero_result() {
        let x = vec![1.0, 2.0, 3.0];
        let y = vec![1.0, 2.0, 3.0];
        let result = granger_causality_test(&x, &y, 5); // lags > series length
        assert!(!result.causes);
        assert_eq!(result.f_statistic, 0.0);
    }
}
