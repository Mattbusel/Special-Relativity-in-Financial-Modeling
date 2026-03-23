//! Quantum-inspired superposition model for price uncertainty.
//!
//! ## Concept
//!
//! A price state is modeled as a superposition of three orthogonal market
//! regimes — **Bullish**, **Bearish**, and **Sideways** — analogous to a
//! quantum wavefunction in a 3-dimensional Hilbert space. The amplitudes
//! (complex probability weights) evolve continuously as new price data arrives.
//!
//! "Wavefunction collapse" occurs when the price makes a decisive move that
//! pushes the dominant regime's probability above a configurable threshold.
//! After collapse, the state is reset with a small residual amplitude on the
//! non-dominant basis states (decoherence memory), reflecting the empirical
//! fact that markets retain some memory of previous regimes.
//!
//! ## Decoherence Rate
//!
//! The decoherence rate `γ` is a dimensionless measure of how quickly the
//! superposition loses coherence. It is computed as the Shannon entropy of the
//! probability distribution over the three basis states, normalised to [0, 1]:
//!
//! ```text
//! γ = -Σ p_i · log₃(p_i)   (base-3 logarithm so max = 1 when uniform)
//! ```
//!
//! A high `γ` means the three regimes are nearly equally probable — the market
//! is in maximum uncertainty — and serves as a direct volatility proxy.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::quantum::{PriceQuantumState, QuantumConfig, MarketRegime};
//!
//! let cfg = QuantumConfig::default();
//! let mut state = PriceQuantumState::new(cfg);
//!
//! // Feed price returns; observe decoherence rate.
//! state.update(0.003);   // +0.3 %
//! state.update(-0.001);
//! state.update(0.005);
//!
//! println!("dominant regime: {:?}", state.dominant_regime());
//! println!("decoherence rate (vol proxy): {:.4}", state.decoherence_rate());
//! ```

/// Configuration for the quantum superposition model.
#[derive(Debug, Clone)]
pub struct QuantumConfig {
    /// Return threshold (absolute) that biases the wavefunction toward
    /// Bullish (positive) or Bearish (negative).
    pub return_sensitivity: f64,

    /// Probability above which the wavefunction is considered "collapsed"
    /// into a single regime.  Must be in (0.5, 1.0].
    pub collapse_threshold: f64,

    /// After collapse, the residual amplitude left on each non-dominant
    /// basis state.  Reflects decoherence memory.  Must be < 1.0.
    pub residual_amplitude: f64,

    /// Decay factor applied to all amplitudes each tick toward the
    /// uniform superposition (entropy increase over time).  Must be in
    /// (0.0, 1.0].
    pub coherence_decay: f64,
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            return_sensitivity: 0.002, // 0.2 % per tick triggers meaningful bias
            collapse_threshold: 0.75,
            residual_amplitude: 0.05,
            coherence_decay: 0.92,
        }
    }
}

/// The three orthogonal market-regime basis states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum MarketRegime {
    /// Price is trending upward.
    Bullish,
    /// Price is trending downward.
    Bearish,
    /// Price is range-bound.
    Sideways,
}

impl std::fmt::Display for MarketRegime {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MarketRegime::Bullish => write!(f, "Bullish"),
            MarketRegime::Bearish => write!(f, "Bearish"),
            MarketRegime::Sideways => write!(f, "Sideways"),
        }
    }
}

/// Recorded collapse event: regime, probability at collapse, and tick index.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollapseEvent {
    /// Which regime the wavefunction collapsed into.
    pub regime: MarketRegime,
    /// Probability of the dominant regime at the moment of collapse.
    pub probability: f64,
    /// Tick index (zero-based count of `update` calls) when collapse occurred.
    pub tick_index: usize,
}

/// Quantum-inspired price state living in a 3-state Hilbert space.
///
/// Amplitudes are stored as real-valued (non-negative) probability amplitudes
/// ψ_i ∈ [0, 1], normalised so that `Σ ψ_i² = 1`.  The Born-rule probability
/// for regime i is `p_i = ψ_i²`.
#[derive(Debug, Clone)]
pub struct PriceQuantumState {
    cfg: QuantumConfig,
    /// Probability amplitudes [Bullish, Bearish, Sideways].
    amplitudes: [f64; 3],
    /// Number of ticks processed.
    tick_count: usize,
    /// All collapse events recorded so far.
    collapses: Vec<CollapseEvent>,
}

impl PriceQuantumState {
    /// Construct a new state in the maximally uncertain superposition
    /// (equal amplitudes across all three regimes).
    pub fn new(cfg: QuantumConfig) -> Self {
        let a = (1.0_f64 / 3.0).sqrt();
        Self {
            cfg,
            amplitudes: [a, a, a],
            tick_count: 0,
            collapses: Vec::new(),
        }
    }

    /// Ingest a single price return `r = (price_t - price_{t-1}) / price_{t-1}`.
    ///
    /// The wavefunction is updated in three steps:
    /// 1. **Drive**: the return biases the Bullish/Bearish amplitudes.
    /// 2. **Decay**: all amplitudes decay toward the uniform superposition.
    /// 3. **Renormalise**: amplitudes are rescaled so `Σ ψ_i² = 1`.
    ///
    /// If the dominant probability exceeds `collapse_threshold`, a collapse
    /// event is recorded and the state is re-initialised with residual memory.
    pub fn update(&mut self, r: f64) {
        self.tick_count += 1;

        // Step 1 — drive: return biases bullish / bearish amplitudes.
        let bias = (r / self.cfg.return_sensitivity).tanh();
        // Positive bias increases bullish, negative increases bearish.
        self.amplitudes[0] += bias.max(0.0) * 0.15;
        self.amplitudes[1] += (-bias).max(0.0) * 0.15;
        // Sideways amplitude decays when there is a strong directional move.
        self.amplitudes[2] *= 1.0 - bias.abs() * 0.1;

        // Step 2 — coherence decay: drift toward uniform superposition.
        let uniform = (1.0_f64 / 3.0).sqrt();
        for a in &mut self.amplitudes {
            *a = self.cfg.coherence_decay * (*a) + (1.0 - self.cfg.coherence_decay) * uniform;
        }

        // Clamp to non-negative.
        for a in &mut self.amplitudes {
            *a = a.max(0.0);
        }

        // Step 3 — renormalise.
        self.renormalise();

        // Check for collapse.
        let probs = self.probabilities();
        let (dominant_idx, dominant_prob) = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, &p)| (i, p))
            .unwrap_or((2, 0.0));

        if dominant_prob >= self.cfg.collapse_threshold {
            let regime = Self::idx_to_regime(dominant_idx);
            self.collapses.push(CollapseEvent {
                regime,
                probability: dominant_prob,
                tick_index: self.tick_count,
            });
            // Re-initialise with residual amplitude on non-dominant states.
            let res = self.cfg.residual_amplitude;
            for (i, a) in self.amplitudes.iter_mut().enumerate() {
                if i == dominant_idx {
                    *a = (1.0 - 2.0 * res * res).sqrt().max(0.0);
                } else {
                    *a = res;
                }
            }
            self.renormalise();
        }
    }

    /// Return the Born-rule probabilities `[p_Bullish, p_Bearish, p_Sideways]`.
    pub fn probabilities(&self) -> [f64; 3] {
        [
            self.amplitudes[0].powi(2),
            self.amplitudes[1].powi(2),
            self.amplitudes[2].powi(2),
        ]
    }

    /// The currently dominant regime (highest probability basis state).
    pub fn dominant_regime(&self) -> MarketRegime {
        let probs = self.probabilities();
        let idx = probs
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(2);
        Self::idx_to_regime(idx)
    }

    /// Decoherence rate `γ ∈ [0, 1]` — the base-3 Shannon entropy of the
    /// probability distribution.  High γ signals maximum regime uncertainty
    /// and acts as a volatility proxy.
    pub fn decoherence_rate(&self) -> f64 {
        let probs = self.probabilities();
        let entropy: f64 = probs
            .iter()
            .filter(|&&p| p > 1e-12)
            .map(|&p| -p * p.log(3.0))
            .sum();
        entropy.clamp(0.0, 1.0)
    }

    /// All wavefunction collapse events recorded since construction.
    pub fn collapse_history(&self) -> &[CollapseEvent] {
        &self.collapses
    }

    /// Total number of `update` calls processed.
    pub fn tick_count(&self) -> usize {
        self.tick_count
    }

    // ── private helpers ───────────────────────────────────────────────────────

    fn renormalise(&mut self) {
        let norm: f64 = self.amplitudes.iter().map(|a| a.powi(2)).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for a in &mut self.amplitudes {
                *a /= norm;
            }
        } else {
            // Degenerate: reset to uniform.
            let u = (1.0_f64 / 3.0).sqrt();
            self.amplitudes = [u, u, u];
        }
    }

    fn idx_to_regime(idx: usize) -> MarketRegime {
        match idx {
            0 => MarketRegime::Bullish,
            1 => MarketRegime::Bearish,
            _ => MarketRegime::Sideways,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn initial_state_is_maximally_uncertain() {
        let state = PriceQuantumState::new(QuantumConfig::default());
        let gamma = state.decoherence_rate();
        // Uniform distribution → maximum entropy → γ ≈ 1.
        assert!(gamma > 0.95, "expected near-unity decoherence, got {gamma}");
    }

    #[test]
    fn sustained_positive_returns_favour_bullish() {
        let mut state = PriceQuantumState::new(QuantumConfig::default());
        for _ in 0..40 {
            state.update(0.01); // strong bullish signal
        }
        assert_eq!(state.dominant_regime(), MarketRegime::Bullish);
    }

    #[test]
    fn probabilities_sum_to_one() {
        let mut state = PriceQuantumState::new(QuantumConfig::default());
        state.update(0.005);
        state.update(-0.002);
        let sum: f64 = state.probabilities().iter().sum();
        assert!((sum - 1.0).abs() < 1e-10, "probabilities must sum to 1, got {sum}");
    }

    #[test]
    fn collapse_event_recorded_after_strong_trend() {
        let mut state = PriceQuantumState::new(QuantumConfig::default());
        for _ in 0..60 {
            state.update(0.02);
        }
        assert!(
            !state.collapse_history().is_empty(),
            "expected at least one collapse event after 60 strong bullish ticks"
        );
    }

    #[test]
    fn decoherence_decreases_after_directional_move() {
        let mut state = PriceQuantumState::new(QuantumConfig::default());
        let gamma_initial = state.decoherence_rate();
        for _ in 0..20 {
            state.update(0.015);
        }
        let gamma_after = state.decoherence_rate();
        assert!(
            gamma_after < gamma_initial,
            "decoherence should decrease as dominant regime emerges: {gamma_initial} -> {gamma_after}"
        );
    }
}
