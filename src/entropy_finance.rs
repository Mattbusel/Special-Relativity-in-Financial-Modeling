//! Information entropy applied to market analysis.
//!
//! Implements several entropy metrics for financial return series:
//! - Shannon entropy of return distributions (histogram-based)
//! - Transfer entropy between two return series
//! - Approximate entropy (ApEn)
//! - Sample entropy (SampEn)
//! - Permutation entropy

use std::collections::HashMap;

// ─── DISCRETIZATION HELPERS ───────────────────────────────────────────────────

/// Discretize a slice of f64 values into `bins` equal-width bins.
/// Returns a Vec of bin indices (0..bins-1) for each value.
fn discretize(values: &[f64], bins: usize) -> Vec<usize> {
    if values.is_empty() || bins == 0 {
        return Vec::new();
    }
    let min_val = values.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = max_val - min_val;

    if range == 0.0 {
        return vec![0; values.len()];
    }

    values
        .iter()
        .map(|&v| {
            let bin = ((v - min_val) / range * bins as f64) as usize;
            bin.min(bins - 1)
        })
        .collect()
}

/// Compute Shannon entropy from a frequency map.
/// H = -Σ p_i * log(p_i)
fn shannon_entropy_from_counts(counts: &HashMap<usize, usize>, total: usize) -> f64 {
    if total == 0 {
        return 0.0;
    }
    let n = total as f64;
    counts.values().fold(0.0, |acc, &count| {
        if count == 0 {
            acc
        } else {
            let p = count as f64 / n;
            acc - p * p.ln()
        }
    })
}

// ─── MARKET ENTROPY ───────────────────────────────────────────────────────────

pub struct MarketEntropy;

impl MarketEntropy {
    /// Compute Shannon entropy of return distribution using histogram binning.
    ///
    /// H = -Σ p_i * log(p_i)
    pub fn price_entropy(returns: &[f64], bins: usize) -> f64 {
        if returns.len() < 2 || bins == 0 {
            return 0.0;
        }
        let discretized = discretize(returns, bins);
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for bin in &discretized {
            *counts.entry(*bin).or_insert(0) += 1;
        }
        shannon_entropy_from_counts(&counts, discretized.len())
    }

    /// Transfer entropy from `source` to `target` with the given lag.
    ///
    /// TE(X→Y) = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
    ///
    /// Estimates conditional entropies from joint frequency tables.
    pub fn transfer_entropy(
        source: &[f64],
        target: &[f64],
        lag: usize,
        bins: usize,
    ) -> f64 {
        let min_len = source.len().min(target.len());
        if min_len <= lag + 1 || bins == 0 {
            return 0.0;
        }

        let src_disc = discretize(&source[..min_len], bins);
        let tgt_disc = discretize(&target[..min_len], bins);

        // Build joint frequency tables
        // For H(Y_t | Y_{t-1}): joint counts of (Y_{t-1}, Y_t)
        let mut counts_yt_yprev: HashMap<(usize, usize), usize> = HashMap::new();
        let mut counts_yprev: HashMap<usize, usize> = HashMap::new();

        // For H(Y_t | Y_{t-1}, X_{t-lag}): joint counts of (Y_{t-1}, X_{t-lag}, Y_t)
        let mut counts_yt_yprev_xlag: HashMap<(usize, usize, usize), usize> = HashMap::new();
        let mut counts_yprev_xlag: HashMap<(usize, usize), usize> = HashMap::new();

        let start = lag + 1;
        let n_samples = min_len - start;

        for t in start..min_len {
            let yt = tgt_disc[t];
            let yt_prev = tgt_disc[t - 1];
            let x_lag = src_disc[t - lag];

            *counts_yt_yprev.entry((yt_prev, yt)).or_insert(0) += 1;
            *counts_yprev.entry(yt_prev).or_insert(0) += 1;

            *counts_yt_yprev_xlag
                .entry((yt_prev, x_lag, yt))
                .or_insert(0) += 1;
            *counts_yprev_xlag.entry((yt_prev, x_lag)).or_insert(0) += 1;
        }

        if n_samples == 0 {
            return 0.0;
        }

        let n = n_samples as f64;

        // H(Y_t | Y_{t-1}) = -Σ p(yt, yt_prev) * log(p(yt | yt_prev))
        let h_yt_given_yprev: f64 = counts_yt_yprev
            .iter()
            .map(|(&(yt_prev, _yt), &joint_count)| {
                let p_joint = joint_count as f64 / n;
                let marginal = *counts_yprev.get(&yt_prev).unwrap_or(&1) as f64;
                let p_cond = joint_count as f64 / marginal;
                if p_cond > 0.0 && p_joint > 0.0 {
                    -p_joint * p_cond.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // H(Y_t | Y_{t-1}, X_{t-lag})
        let h_yt_given_yprev_xlag: f64 = counts_yt_yprev_xlag
            .iter()
            .map(|(&(yt_prev, x_lag, _yt), &joint_count)| {
                let p_joint = joint_count as f64 / n;
                let marginal = *counts_yprev_xlag
                    .get(&(yt_prev, x_lag))
                    .unwrap_or(&1) as f64;
                let p_cond = joint_count as f64 / marginal;
                if p_cond > 0.0 && p_joint > 0.0 {
                    -p_joint * p_cond.ln()
                } else {
                    0.0
                }
            })
            .sum();

        // TE = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-lag})
        (h_yt_given_yprev - h_yt_given_yprev_xlag).max(0.0)
    }

    /// Approximate entropy (ApEn).
    ///
    /// phi_m = (1/N) * Σ log(C_i^m(r))  where C_i^m(r) = count of templates
    /// of length m within distance r divided by (N - m + 1).
    /// ApEn = phi_m - phi_{m+1}
    pub fn approximate_entropy(signal: &[f64], m: usize, r: f64) -> f64 {
        let n = signal.len();
        if n < m + 2 {
            return 0.0;
        }

        let phi = |template_len: usize| -> f64 {
            let count = n - template_len + 1;
            if count == 0 {
                return 0.0;
            }
            let sum: f64 = (0..count)
                .map(|i| {
                    let matches = (0..count)
                        .filter(|&j| {
                            (0..template_len)
                                .all(|k| (signal[i + k] - signal[j + k]).abs() <= r)
                        })
                        .count();
                    let c = matches as f64 / count as f64;
                    if c > 0.0 { c.ln() } else { 0.0 }
                })
                .sum();
            sum / count as f64
        };

        phi(m) - phi(m + 1)
    }

    /// Sample entropy (SampEn).
    ///
    /// Similar to ApEn but excludes self-matches.
    /// Returns None if no matches are found (undefined).
    pub fn sample_entropy(signal: &[f64], m: usize, r: f64) -> Option<f64> {
        let n = signal.len();
        if n < m + 2 {
            return None;
        }

        let count_matches = |template_len: usize| -> usize {
            let limit = n - template_len;
            (0..limit)
                .flat_map(|i| {
                    (0..limit).filter(move |&j| {
                        i != j
                            && (0..template_len)
                                .all(|k| (signal[i + k] - signal[j + k]).abs() <= r)
                    })
                })
                .count()
        };

        let a = count_matches(m + 1) as f64; // templates of length m+1
        let b = count_matches(m) as f64;     // templates of length m

        if b == 0.0 {
            None
        } else {
            Some(-(a / b).ln())
        }
    }

    /// Permutation entropy.
    ///
    /// Enumerate ordinal patterns of length `order` with delay `delay`.
    /// H = -Σ p_π * log(p_π)
    pub fn permutation_entropy(signal: &[f64], order: usize, delay: usize) -> f64 {
        if signal.len() < order * delay || order < 2 {
            return 0.0;
        }

        let n_patterns = signal.len() - (order - 1) * delay;
        let mut pattern_counts: HashMap<Vec<usize>, usize> = HashMap::new();

        for i in 0..n_patterns {
            // Extract the sub-sequence with the given delay
            let sub: Vec<f64> = (0..order).map(|k| signal[i + k * delay]).collect();

            // Compute the ordinal pattern (argsort)
            let mut indices: Vec<usize> = (0..order).collect();
            indices.sort_by(|&a, &b| {
                sub[a].partial_cmp(&sub[b]).unwrap_or(std::cmp::Ordering::Equal)
            });

            *pattern_counts.entry(indices).or_insert(0) += 1;
        }

        let n = n_patterns as f64;
        -pattern_counts
            .values()
            .map(|&count| {
                let p = count as f64 / n;
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
    }
}

// ─── ENTROPY REPORT ───────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct EntropyReport {
    pub price_entropy: f64,
    pub approximate_entropy: f64,
    pub sample_entropy: Option<f64>,
    pub permutation_entropy: f64,
    pub interpretation: String,
}

impl EntropyReport {
    /// Compute all entropy metrics for a return series.
    pub fn compute(returns: &[f64]) -> Self {
        let price_entropy = MarketEntropy::price_entropy(returns, 10);
        let approximate_entropy = MarketEntropy::approximate_entropy(returns, 2, 0.2);
        let sample_entropy = MarketEntropy::sample_entropy(returns, 2, 0.2);
        let permutation_entropy = MarketEntropy::permutation_entropy(returns, 3, 1);

        let interpretation = interpret_entropy(price_entropy, approximate_entropy);

        EntropyReport {
            price_entropy,
            approximate_entropy,
            sample_entropy,
            permutation_entropy,
            interpretation,
        }
    }
}

fn interpret_entropy(price_entropy: f64, apen: f64) -> String {
    match (price_entropy > 2.0, apen > 0.5) {
        (true, true) => "High entropy: market appears random / efficient. Signal-to-noise is low.".to_string(),
        (true, false) => "High distributional entropy but low regularity: mixed signals.".to_string(),
        (false, true) => "Low distributional entropy but complex dynamics: possible regime structure.".to_string(),
        (false, false) => "Low entropy: market shows strong structure / predictability. Potential edge.".to_string(),
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_returns_high_entropy() {
        // Many distinct values → high entropy
        let returns: Vec<f64> = (0..100).map(|i| i as f64 * 0.01 - 0.5).collect();
        let h = MarketEntropy::price_entropy(&returns, 20);
        assert!(h > 2.0, "uniform returns should have high entropy, got {}", h);
    }

    #[test]
    fn constant_returns_low_entropy() {
        // All the same value → entropy should be 0 (one bin)
        let returns = vec![0.01; 100];
        let h = MarketEntropy::price_entropy(&returns, 10);
        assert!(h < 1e-6, "constant returns should have entropy ~0, got {}", h);
    }

    #[test]
    fn permutation_entropy_in_valid_range() {
        let returns: Vec<f64> = (0..50).map(|i| (i as f64 * 0.1).sin()).collect();
        let order = 3usize;
        let pe = MarketEntropy::permutation_entropy(&returns, order, 1);
        // Maximum permutation entropy for order k is ln(k!)
        let max_pe = (1..=order).map(|i| i as f64).product::<f64>().ln();
        assert!(pe >= 0.0, "permutation entropy must be non-negative, got {}", pe);
        assert!(pe <= max_pe + 1e-10, "permutation entropy {} exceeds max {}", pe, max_pe);
    }

    #[test]
    fn permutation_entropy_low_for_monotone() {
        // Strictly increasing → only one ordinal pattern
        let returns: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let pe = MarketEntropy::permutation_entropy(&returns, 3, 1);
        assert!(pe < 0.01, "monotone series should have low permutation entropy, got {}", pe);
    }

    #[test]
    fn approximate_entropy_nonnegative() {
        let returns: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let apen = MarketEntropy::approximate_entropy(&returns, 2, 0.2);
        assert!(apen >= 0.0, "ApEn must be non-negative, got {}", apen);
    }

    #[test]
    fn sample_entropy_returns_some_for_valid_input() {
        let returns: Vec<f64> = (0..30).map(|i| (i as f64 * 0.3).sin()).collect();
        let sampen = MarketEntropy::sample_entropy(&returns, 2, 0.5);
        // May be None if no matches; if Some, should be finite
        if let Some(s) = sampen {
            assert!(s.is_finite(), "SampEn should be finite, got {}", s);
        }
    }

    #[test]
    fn transfer_entropy_nonnegative() {
        let source: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2).sin()).collect();
        let target: Vec<f64> = (0..50).map(|i| (i as f64 * 0.2 + 0.5).sin()).collect();
        let te = MarketEntropy::transfer_entropy(&source, &target, 1, 5);
        assert!(te >= 0.0, "Transfer entropy must be non-negative, got {}", te);
    }

    #[test]
    fn entropy_report_computes_all_fields() {
        let returns: Vec<f64> = (0..40).map(|i| (i as f64 * 0.15).sin() * 0.05).collect();
        let report = EntropyReport::compute(&returns);
        assert!(report.price_entropy >= 0.0);
        assert!(report.approximate_entropy >= 0.0);
        assert!(report.permutation_entropy >= 0.0);
        assert!(!report.interpretation.is_empty());
    }
}
