//! Quantum decoherence analogy for information loss in financial markets.
//!
//! Just as quantum superpositions decohere (lose quantum interference) due to
//! environmental entanglement, market price superpositions (multiple potential
//! outcomes) collapse to a single realised price as information enters the
//! market.  This module models that process.

use std::collections::HashMap;

// ─── QUANTUM STATE ────────────────────────────────────────────────────────────

/// A single coherent amplitude-phase quantum-analogue state.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantumState {
    /// Probability amplitude (|amplitude|² = probability).
    pub amplitude: f64,
    /// Phase in radians.
    pub phase: f64,
}

impl QuantumState {
    pub fn probability(&self) -> f64 {
        self.amplitude * self.amplitude
    }
}

// ─── MARKET SUPERPOSITION ─────────────────────────────────────────────────────

/// A superposition of labelled price scenarios.
#[derive(Debug, Clone)]
pub struct MarketSuperposition {
    /// Each entry: (quantum_state, price_scenario_label).
    pub states: Vec<(QuantumState, String)>,
}

impl MarketSuperposition {
    pub fn new(states: Vec<(QuantumState, String)>) -> Self {
        Self { states }
    }

    /// Probabilistic collapse to one scenario, probabilities ∝ amplitude².
    ///
    /// Uses a simple LCG seeded from `seed` for deterministic selection.
    pub fn collapse(&self, seed: u64) -> String {
        if self.states.is_empty() {
            return String::new();
        }
        let total: f64 = self.states.iter().map(|(s, _)| s.probability()).sum();
        if total <= 0.0 {
            return self.states[0].1.clone();
        }
        // LCG random number in [0, 1).
        let rng = lcg_f64(seed);
        let mut cumulative = 0.0;
        for (state, label) in &self.states {
            cumulative += state.probability() / total;
            if rng < cumulative {
                return label.clone();
            }
        }
        self.states.last().unwrap().1.clone()
    }

    /// Coherence: sum of all off-diagonal interference terms |ρ_ij| for i≠j.
    ///
    /// For a density matrix ρ with ρ_ii = p_i and ρ_ij = a_i * a_j (amplitude product):
    /// coherence = Σ_{i≠j} |a_i * a_j|
    pub fn coherence(&self) -> f64 {
        let n = self.states.len();
        let mut sum = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    sum += (self.states[i].0.amplitude * self.states[j].0.amplitude).abs();
                }
            }
        }
        sum
    }
}

// ─── DECOHERENCE MODEL ────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct DecoherenceModel;

impl DecoherenceModel {
    pub fn new() -> Self {
        Self
    }

    /// Decoherence rate γ = environment_coupling * temperature (simple proportional model).
    pub fn decoherence_rate(&self, environment_coupling: f64, temperature: f64) -> f64 {
        environment_coupling * temperature
    }

    /// Coherence decays exponentially: C(t) = C₀ · exp(-γ · t).
    pub fn coherence_decay(&self, initial_coherence: f64, rate: f64, time: f64) -> f64 {
        initial_coherence * (-rate * time).exp()
    }

    /// Apply decoherence to a superposition: reduce off-diagonal elements by
    /// the decay factor exp(-rate * elapsed_ms / 1000).
    ///
    /// Amplitudes are multiplied by sqrt(decay_factor) so that probabilities (amplitude²)
    /// scale correctly, while the diagonal is preserved.
    pub fn decohere(
        &self,
        state: &MarketSuperposition,
        elapsed_ms: u64,
        rate: f64,
    ) -> MarketSuperposition {
        let time_s = elapsed_ms as f64 / 1000.0;
        // Off-diagonal elements decay as exp(-rate*t); amplitudes decay as exp(-rate*t/2).
        let amp_factor = (-rate * time_s / 2.0).exp();
        let new_states = state
            .states
            .iter()
            .map(|(qs, label)| {
                (
                    QuantumState {
                        amplitude: qs.amplitude * amp_factor,
                        phase: qs.phase,
                    },
                    label.clone(),
                )
            })
            .collect();
        MarketSuperposition { states: new_states }
    }
}

// ─── INFORMATION LOSS ─────────────────────────────────────────────────────────

/// Quantifies information loss during decoherence via von Neumann entropy.
#[derive(Debug, Clone, PartialEq)]
pub struct InformationLoss {
    pub initial_entropy: f64,
    pub final_entropy: f64,
    pub bits_lost: f64,
}

impl InformationLoss {
    /// Create from two superposition states.
    pub fn from_states(initial: &MarketSuperposition, final_state: &MarketSuperposition) -> Self {
        let h_i = Self::von_neumann_entropy(initial);
        let h_f = Self::von_neumann_entropy(final_state);
        Self {
            initial_entropy: h_i,
            final_entropy: h_f,
            bits_lost: (h_f - h_i).abs() / std::f64::consts::LN_2,
        }
    }

    /// Von Neumann entropy: S = -Σ p_i · ln(p_i) where p_i = amplitude_i² / Z.
    pub fn von_neumann_entropy(state: &MarketSuperposition) -> f64 {
        let total: f64 = state.states.iter().map(|(s, _)| s.probability()).sum();
        if total <= 0.0 {
            return 0.0;
        }
        -state
            .states
            .iter()
            .map(|(s, _)| {
                let p = s.probability() / total;
                if p > 0.0 { p * p.ln() } else { 0.0 }
            })
            .sum::<f64>()
    }
}

// ─── HELPERS ──────────────────────────────────────────────────────────────────

/// Simple LCG for deterministic pseudo-random float in [0, 1).
fn lcg_f64(seed: u64) -> f64 {
    const A: u64 = 6_364_136_223_846_793_005;
    const C: u64 = 1_442_695_040_888_963_407;
    let next = seed.wrapping_mul(A).wrapping_add(C);
    (next >> 11) as f64 / (1u64 << 53) as f64
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn equal_superposition(labels: &[&str]) -> MarketSuperposition {
        let amp = 1.0 / (labels.len() as f64).sqrt();
        let states = labels
            .iter()
            .map(|&l| (QuantumState { amplitude: amp, phase: 0.0 }, l.to_owned()))
            .collect();
        MarketSuperposition { states }
    }

    #[test]
    fn test_quantum_state_probability() {
        let qs = QuantumState { amplitude: 0.5_f64.sqrt(), phase: 0.0 };
        assert!((qs.probability() - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_collapse_returns_valid_scenario() {
        let sup = equal_superposition(&["bull", "bear", "sideways"]);
        let scenario = sup.collapse(42);
        assert!(["bull", "bear", "sideways"].contains(&scenario.as_str()));
    }

    #[test]
    fn test_collapse_empty_superposition() {
        let sup = MarketSuperposition::new(vec![]);
        let result = sup.collapse(0);
        assert!(result.is_empty());
    }

    #[test]
    fn test_collapse_single_state_always_returns_it() {
        let sup = MarketSuperposition::new(vec![(
            QuantumState { amplitude: 1.0, phase: 0.0 },
            "certain".to_owned(),
        )]);
        for seed in 0..20u64 {
            assert_eq!(sup.collapse(seed), "certain");
        }
    }

    #[test]
    fn test_coherence_two_states() {
        // For two states with amplitude 0.707 each:
        // coherence = 2 * |a1 * a2| = 2 * 0.5 = 1.0
        let sup = equal_superposition(&["up", "down"]);
        let c = sup.coherence();
        assert!((c - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_coherence_single_state_is_zero() {
        let sup = MarketSuperposition::new(vec![(
            QuantumState { amplitude: 1.0, phase: 0.0 },
            "only".to_owned(),
        )]);
        assert_eq!(sup.coherence(), 0.0);
    }

    #[test]
    fn test_decoherence_rate() {
        let model = DecoherenceModel::new();
        let rate = model.decoherence_rate(2.0, 3.0);
        assert!((rate - 6.0).abs() < 1e-9);
    }

    #[test]
    fn test_coherence_decay_formula() {
        let model = DecoherenceModel::new();
        let decayed = model.coherence_decay(1.0, 1.0, 1.0);
        assert!((decayed - std::f64::consts::E.recip()).abs() < 1e-9);
    }

    #[test]
    fn test_coherence_decay_zero_time() {
        let model = DecoherenceModel::new();
        let decayed = model.coherence_decay(5.0, 10.0, 0.0);
        assert!((decayed - 5.0).abs() < 1e-9);
    }

    #[test]
    fn test_decohere_reduces_amplitudes() {
        let model = DecoherenceModel::new();
        let sup = equal_superposition(&["a", "b"]);
        let initial_amp = sup.states[0].0.amplitude;
        let decohered = model.decohere(&sup, 1000, 1.0); // 1 second, rate=1
        let final_amp = decohered.states[0].0.amplitude;
        assert!(final_amp < initial_amp);
        // amp_factor = exp(-0.5) ≈ 0.6065
        assert!((final_amp / initial_amp - (-0.5f64).exp()).abs() < 1e-6);
    }

    #[test]
    fn test_von_neumann_entropy_uniform() {
        // Uniform distribution over N=4 states: S = ln(4)
        let sup = equal_superposition(&["a", "b", "c", "d"]);
        let entropy = InformationLoss::von_neumann_entropy(&sup);
        assert!((entropy - (4.0f64).ln()).abs() < 1e-9);
    }

    #[test]
    fn test_von_neumann_entropy_pure_state() {
        // Pure state (one outcome certain) has S = 0.
        let sup = MarketSuperposition::new(vec![(
            QuantumState { amplitude: 1.0, phase: 0.0 },
            "x".to_owned(),
        )]);
        let entropy = InformationLoss::von_neumann_entropy(&sup);
        assert!((entropy - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_information_loss_from_states() {
        let initial = equal_superposition(&["a", "b", "c"]);
        let model = DecoherenceModel::new();
        let decohered = model.decohere(&initial, 5000, 2.0);
        let loss = InformationLoss::from_states(&initial, &decohered);
        // Both states have the same probability distribution (amplitudes scaled uniformly),
        // so entropy should be the same (loss ≈ 0 bits).
        assert!(loss.bits_lost.abs() < 1e-9);
    }

    #[test]
    fn test_decohere_preserves_relative_probabilities() {
        let model = DecoherenceModel::new();
        let states = vec![
            (QuantumState { amplitude: 0.8, phase: 0.0 }, "likely".to_owned()),
            (QuantumState { amplitude: 0.6, phase: 0.0 }, "less_likely".to_owned()),
        ];
        let sup = MarketSuperposition { states };
        let decohered = model.decohere(&sup, 500, 1.0);
        let ratio_before = sup.states[0].0.amplitude / sup.states[1].0.amplitude;
        let ratio_after = decohered.states[0].0.amplitude / decohered.states[1].0.amplitude;
        assert!((ratio_before - ratio_after).abs() < 1e-9);
    }
}
