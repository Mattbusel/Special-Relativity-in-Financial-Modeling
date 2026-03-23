//! Quantum-inspired financial models.
//!
//! A playful extension of the special-relativity financial modeling framework
//! into quantum mechanics. Assets exist in superpositions of price states;
//! portfolios collapse probabilistically at measurement time.
//!
//! ## Disclaimer
//! This is a mathematical toy model — not real quantum mechanics and not
//! investment advice.

use std::collections::HashMap;

// ─── WAVE FUNCTION ────────────────────────────────────────────────────────────

/// A discrete quantum state with amplitudes, phases, and named states.
#[derive(Debug, Clone)]
pub struct WaveFunction {
    /// Raw (unnormalized) amplitudes for each state.
    pub amplitudes: Vec<f64>,
    /// Phase angles in radians for each state.
    pub phases: Vec<f64>,
    /// Named labels for each state.
    pub states: Vec<String>,
}

impl WaveFunction {
    /// Create a new WaveFunction with the given amplitudes, phases and state labels.
    /// All three vectors must have the same length.
    pub fn new(amplitudes: Vec<f64>, phases: Vec<f64>, states: Vec<String>) -> Self {
        assert_eq!(amplitudes.len(), phases.len());
        assert_eq!(amplitudes.len(), states.len());
        WaveFunction {
            amplitudes,
            phases,
            states,
        }
    }

    /// Compute |amplitude_i|² normalized over all states: the Born probability.
    pub fn probability(&self, state_index: usize) -> f64 {
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a * a).sum();
        if norm_sq == 0.0 {
            return 0.0;
        }
        (self.amplitudes[state_index] * self.amplitudes[state_index]) / norm_sq
    }

    /// Probabilistically collapse the wave function to a single state using a
    /// linear congruential generator seeded by `rng_seed`.
    pub fn collapse(&self, rng_seed: u64) -> String {
        // Build probability distribution
        let norm_sq: f64 = self.amplitudes.iter().map(|a| a * a).sum();
        if norm_sq == 0.0 || self.states.is_empty() {
            return self.states.first().cloned().unwrap_or_default();
        }

        // LCG to produce a value in [0, 1)
        let lcg_value = lcg_random(rng_seed);
        let target = lcg_value;

        // Select state by cumulative probability
        let mut cumulative = 0.0;
        for (i, amp) in self.amplitudes.iter().enumerate() {
            cumulative += (amp * amp) / norm_sq;
            if target <= cumulative {
                return self.states[i].clone();
            }
        }

        // Fallback to last state (floating point rounding)
        self.states.last().cloned().unwrap_or_default()
    }

    /// Compute the superposition α|a⟩ + √(1−α²)|b⟩, then normalize amplitudes.
    pub fn superposition(a: &WaveFunction, b: &WaveFunction, alpha: f64) -> WaveFunction {
        let beta = (1.0 - alpha * alpha).max(0.0).sqrt();

        // Combine: a-states with weight alpha, b-states with weight beta
        let mut amplitudes: Vec<f64> = a.amplitudes.iter().map(|amp| alpha * amp).collect();
        let mut phases: Vec<f64> = a.phases.clone();
        let mut states: Vec<String> = a.states.clone();

        for (i, amp) in b.amplitudes.iter().enumerate() {
            amplitudes.push(beta * amp);
            phases.push(b.phases[i]);
            states.push(b.states[i].clone());
        }

        // Normalize amplitudes so sum(amp²) = 1
        let norm_sq: f64 = amplitudes.iter().map(|x| x * x).sum();
        if norm_sq > 0.0 {
            let norm = norm_sq.sqrt();
            for amp in &mut amplitudes {
                *amp /= norm;
            }
        }

        WaveFunction {
            amplitudes,
            phases,
            states,
        }
    }
}

// ─── QUANTUM PORTFOLIO ────────────────────────────────────────────────────────

/// An asset in the quantum portfolio exists in superposition of price states.
#[derive(Debug, Clone)]
struct QuantumAsset {
    wave_function: WaveFunction,
    price_states: Vec<f64>,
}

/// A portfolio where each asset's price is modeled as a quantum superposition.
#[derive(Debug, Default)]
pub struct QuantumPortfolio {
    assets: HashMap<String, QuantumAsset>,
    /// Correlation pairs: (sym_a, sym_b) → correlation coefficient
    correlations: HashMap<(String, String), f64>,
}

impl QuantumPortfolio {
    pub fn new() -> Self {
        QuantumPortfolio::default()
    }

    /// Add an asset with given price states and (unnormalized) amplitudes.
    pub fn add_asset(&mut self, symbol: &str, price_states: Vec<f64>, amplitudes: Vec<f64>) {
        assert_eq!(price_states.len(), amplitudes.len(), "price_states and amplitudes must have the same length");
        let n = price_states.len();
        let state_labels: Vec<String> = price_states
            .iter()
            .map(|p| format!("{:.2}", p))
            .collect();
        let phases = vec![0.0; n];
        let wf = WaveFunction::new(amplitudes, phases, state_labels);
        self.assets.insert(
            symbol.to_string(),
            QuantumAsset {
                wave_function: wf,
                price_states,
            },
        );
    }

    fn get_probs(&self, symbol: &str) -> Option<Vec<f64>> {
        let asset = self.assets.get(symbol)?;
        let norm_sq: f64 = asset.wave_function.amplitudes.iter().map(|a| a * a).sum();
        if norm_sq == 0.0 {
            return None;
        }
        Some(
            asset
                .wave_function
                .amplitudes
                .iter()
                .map(|a| (a * a) / norm_sq)
                .collect(),
        )
    }

    /// Expectation value = Σ prob_i × price_i
    pub fn expected_value(&self, symbol: &str) -> f64 {
        let asset = match self.assets.get(symbol) {
            Some(a) => a,
            None => return 0.0,
        };
        let probs = match self.get_probs(symbol) {
            Some(p) => p,
            None => return 0.0,
        };
        probs
            .iter()
            .zip(asset.price_states.iter())
            .map(|(p, price)| p * price)
            .sum()
    }

    /// Quantum uncertainty = Σ prob_i × (price_i − E[price])²
    pub fn variance(&self, symbol: &str) -> f64 {
        let ev = self.expected_value(symbol);
        let asset = match self.assets.get(symbol) {
            Some(a) => a,
            None => return 0.0,
        };
        let probs = match self.get_probs(symbol) {
            Some(p) => p,
            None => return 0.0,
        };
        probs
            .iter()
            .zip(asset.price_states.iter())
            .map(|(p, price)| p * (price - ev).powi(2))
            .sum()
    }

    /// Entangle two assets by correlating their amplitude products.
    ///
    /// Simplified model: scale the amplitudes of each asset so they are
    /// jointly biased towards aligned outcomes when `correlation > 0`
    /// (both high or both low) or anti-aligned when `correlation < 0`.
    pub fn entangle(&mut self, sym_a: &str, sym_b: &str, correlation: f64) {
        self.correlations.insert(
            (sym_a.to_string(), sym_b.to_string()),
            correlation.clamp(-1.0, 1.0),
        );
        self.correlations.insert(
            (sym_b.to_string(), sym_a.to_string()),
            correlation.clamp(-1.0, 1.0),
        );

        // Adjust amplitudes to reflect correlation: boost high-price states of
        // sym_b when sym_a tends toward high prices (positive correlation).
        let n_a = self.assets.get(sym_a).map(|a| a.price_states.len()).unwrap_or(0);
        let n_b = self.assets.get(sym_b).map(|a| a.price_states.len()).unwrap_or(0);
        if n_a == 0 || n_b == 0 {
            return;
        }

        // We apply a simple amplitude tilt for the higher-index states (higher prices)
        let boost = 1.0 + 0.1 * correlation;
        let decay = 1.0 - 0.1 * correlation.abs();

        if let Some(asset_b) = self.assets.get_mut(sym_b) {
            let n = asset_b.wave_function.amplitudes.len();
            if n > 1 {
                let mid = n / 2;
                for (i, amp) in asset_b.wave_function.amplitudes.iter_mut().enumerate() {
                    if correlation > 0.0 && i >= mid {
                        *amp *= boost;
                    } else if correlation < 0.0 && i < mid {
                        *amp *= boost;
                    } else {
                        *amp *= decay;
                    }
                }
                // Re-normalize
                let norm_sq: f64 = asset_b.wave_function.amplitudes.iter().map(|a| a * a).sum();
                if norm_sq > 0.0 {
                    let norm = norm_sq.sqrt();
                    for amp in &mut asset_b.wave_function.amplitudes {
                        *amp /= norm;
                    }
                }
            }
        }
    }

    /// Collapse each asset's wave function to a definite price.
    pub fn measure_portfolio(&self, rng_seed: u64) -> HashMap<String, f64> {
        let mut result = HashMap::new();
        let mut seed = rng_seed;
        for (symbol, asset) in &self.assets {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let collapsed_state = asset.wave_function.collapse(seed);
            // Parse the price from the collapsed state label
            let price: f64 = collapsed_state.parse().unwrap_or(0.0);
            // Find closest matching price state
            let closest = asset
                .price_states
                .iter()
                .copied()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (a - price).abs().partial_cmp(&(b - price).abs()).unwrap()
                })
                .map(|(_, p)| p)
                .unwrap_or(price);
            result.insert(symbol.clone(), closest);
        }
        result
    }
}

// ─── LCG HELPER ───────────────────────────────────────────────────────────────

/// Linear congruential generator returning a value in [0, 1).
fn lcg_random(seed: u64) -> f64 {
    let next = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    // Use upper bits for better quality
    let upper = (next >> 33) as f64;
    upper / (u32::MAX as f64 + 1.0)
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn uniform_wf(n: usize) -> WaveFunction {
        let amp = vec![1.0; n];
        let phases = vec![0.0; n];
        let states: Vec<String> = (0..n).map(|i| format!("state_{}", i)).collect();
        WaveFunction::new(amp, phases, states)
    }

    #[test]
    fn probabilities_sum_to_one() {
        let wf = uniform_wf(4);
        let total: f64 = (0..4).map(|i| wf.probability(i)).sum();
        assert!((total - 1.0).abs() < 1e-10, "probabilities must sum to 1, got {}", total);
    }

    #[test]
    fn uniform_probabilities_are_equal() {
        let wf = uniform_wf(5);
        for i in 0..5 {
            let p = wf.probability(i);
            assert!((p - 0.2).abs() < 1e-10, "expected 0.2, got {}", p);
        }
    }

    #[test]
    fn collapse_returns_valid_state() {
        let wf = uniform_wf(4);
        let state = wf.collapse(42);
        assert!(
            wf.states.contains(&state),
            "collapsed state '{}' not in wave function states",
            state
        );
    }

    #[test]
    fn expected_value_correct() {
        let mut portfolio = QuantumPortfolio::new();
        // Two equally probable price states: 100 and 200 → E = 150
        portfolio.add_asset("AAPL", vec![100.0, 200.0], vec![1.0, 1.0]);
        let ev = portfolio.expected_value("AAPL");
        assert!((ev - 150.0).abs() < 1e-6, "expected 150, got {}", ev);
    }

    #[test]
    fn expected_value_weighted() {
        let mut portfolio = QuantumPortfolio::new();
        // Amplitude 1 for 100, amplitude 0 for 200 → E = 100
        portfolio.add_asset("BTC", vec![100.0, 200.0], vec![1.0, 0.0]);
        let ev = portfolio.expected_value("BTC");
        assert!((ev - 100.0).abs() < 1e-6, "expected 100, got {}", ev);
    }

    #[test]
    fn variance_is_nonnegative() {
        let mut portfolio = QuantumPortfolio::new();
        portfolio.add_asset("ETH", vec![1000.0, 2000.0, 3000.0], vec![1.0, 2.0, 1.0]);
        let var = portfolio.variance("ETH");
        assert!(var >= 0.0, "variance must be non-negative, got {}", var);
    }

    #[test]
    fn variance_is_zero_for_certain_state() {
        let mut portfolio = QuantumPortfolio::new();
        // Only one state possible
        portfolio.add_asset("USD", vec![1.0], vec![1.0]);
        let var = portfolio.variance("USD");
        assert!(var.abs() < 1e-10, "variance of certain state should be 0, got {}", var);
    }

    #[test]
    fn measure_portfolio_returns_all_symbols() {
        let mut portfolio = QuantumPortfolio::new();
        portfolio.add_asset("AAPL", vec![150.0, 160.0], vec![1.0, 1.0]);
        portfolio.add_asset("GOOG", vec![100.0, 110.0], vec![1.0, 1.0]);
        let measurement = portfolio.measure_portfolio(99);
        assert!(measurement.contains_key("AAPL"));
        assert!(measurement.contains_key("GOOG"));
    }

    #[test]
    fn superposition_probabilities_sum_to_one() {
        let wf_a = uniform_wf(2);
        let wf_b = uniform_wf(3);
        let combined = WaveFunction::superposition(&wf_a, &wf_b, 0.6);
        let total: f64 = (0..combined.amplitudes.len())
            .map(|i| combined.probability(i))
            .sum();
        assert!((total - 1.0).abs() < 1e-10, "superposition probs must sum to 1, got {}", total);
    }
}
