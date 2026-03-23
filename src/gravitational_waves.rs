//! Gravitational-wave analogy for market shock detection.
//!
//! ## Concept
//!
//! A large market shock (e.g. a flash crash, a Fed announcement, a
//! liquidation cascade) propagates through the market analogously to a
//! gravitational wave radiating from a massive event: it arrives at
//! different assets at different times, with an amplitude that decays
//! with distance in correlation space.
//!
//! This module adapts **matched filtering** — the core technique used by
//! LIGO to detect gravitational waves — to identify shock propagation
//! patterns:
//!
//! 1. A **template waveform** `h(t)` is constructed from the returns of the
//!    *source* asset around the time of a large shock.  The template is the
//!    characteristic "chirp": a sharp spike followed by damped oscillation.
//!
//! 2. The **matched filter SNR** at lag `τ` for a target asset is
//!
//!    ```text
//!    SNR(τ) = |⟨d, h(t-τ)⟩| / √⟨h, h⟩
//!    ```
//!
//!    where `⟨·,·⟩` is the inner product (sum of element-wise products).
//!    A high SNR at lag `τ` means the shock template matches the target
//!    asset's returns `τ` ticks later.
//!
//! 3. When SNR exceeds a threshold, a [`ShockPropagationEvent`] is emitted
//!    recording the source asset, target asset, estimated lag, and SNR.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::gravitational_waves::{ShockDetector, ShockConfig};
//!
//! let cfg = ShockConfig::default();
//! let mut detector = ShockDetector::new(cfg);
//!
//! // Register a shock template from source asset returns.
//! let template: Vec<f64> = vec![0.0, 0.0, 0.08, -0.03, 0.01, -0.005, 0.001];
//! detector.set_template("SPY".to_string(), template);
//!
//! // Feed returns for a target asset.
//! for r in [0.0, 0.0, 0.0, 0.075, -0.028, 0.009, -0.004] {
//!     detector.push_target_return("QQQ".to_string(), r);
//! }
//!
//! let events = detector.detect("QQQ");
//! for e in &events {
//!     println!("shock from {} -> {} lag={} SNR={:.2}", e.source, e.target, e.lag_ticks, e.snr);
//! }
//! ```

/// Configuration for the gravitational-wave shock detector.
#[derive(Debug, Clone)]
pub struct ShockConfig {
    /// SNR threshold above which a propagation event is declared.
    pub snr_threshold: f64,

    /// Maximum lag (in ticks) to search when sliding the template over the
    /// target return stream.
    pub max_lag: usize,

    /// Minimum number of target returns before running matched filtering.
    pub min_returns: usize,
}

impl Default for ShockConfig {
    fn default() -> Self {
        Self {
            snr_threshold: 3.0,
            max_lag: 20,
            min_returns: 10,
        }
    }
}

/// A detected shock propagation from source to target asset.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ShockPropagationEvent {
    /// Asset where the shock originated.
    pub source: String,
    /// Asset where the propagated shock was detected.
    pub target: String,
    /// Estimated lag in ticks between source shock and target response.
    pub lag_ticks: usize,
    /// Matched-filter signal-to-noise ratio at the best lag.
    pub snr: f64,
    /// Peak absolute return in the matched window of the target.
    pub target_peak_return: f64,
}

/// Streaming shock detector using matched filtering.
#[derive(Debug, Clone)]
pub struct ShockDetector {
    cfg: ShockConfig,
    /// Registered shock templates keyed by source asset symbol.
    templates: std::collections::HashMap<String, Vec<f64>>,
    /// Rolling return history per target asset.
    target_returns: std::collections::HashMap<String, std::collections::VecDeque<f64>>,
}

impl ShockDetector {
    /// Create a new detector with the given configuration.
    pub fn new(cfg: ShockConfig) -> Self {
        Self {
            cfg,
            templates: std::collections::HashMap::new(),
            target_returns: std::collections::HashMap::new(),
        }
    }

    /// Register or update the shock template for `source`.
    ///
    /// The template should be a short vector of log-returns centred on the
    /// shock event (a few ticks before and after the peak).
    pub fn set_template(&mut self, source: String, template: Vec<f64>) {
        self.templates.insert(source, template);
    }

    /// Append a log-return for `target`.  The internal history is capped at
    /// `max_lag + template_len` to keep memory bounded.
    pub fn push_target_return(&mut self, target: String, log_return: f64) {
        let history = self.target_returns.entry(target).or_default();
        history.push_back(log_return);
        // Keep at most max_lag + longest_template returns.
        let max_template_len = self.templates.values().map(|v| v.len()).max().unwrap_or(0);
        let cap = self.cfg.max_lag + max_template_len + 1;
        while history.len() > cap {
            history.pop_front();
        }
    }

    /// Run matched filtering of all registered templates against `target`.
    ///
    /// Returns a list of [`ShockPropagationEvent`]s for every source whose
    /// template produced an SNR exceeding the threshold.
    pub fn detect(&self, target: &str) -> Vec<ShockPropagationEvent> {
        let mut events = Vec::new();
        let Some(history) = self.target_returns.get(target) else {
            return events;
        };
        if history.len() < self.cfg.min_returns {
            return events;
        }
        let data: Vec<f64> = history.iter().cloned().collect();

        for (source, template) in &self.templates {
            if template.is_empty() {
                continue;
            }

            // Normalise template energy.
            let template_energy: f64 = template.iter().map(|x| x * x).sum();
            if template_energy < 1e-12 {
                continue;
            }
            let h_norm = template_energy.sqrt();

            // Slide the template over the data at each lag.
            let mut best_snr = 0.0f64;
            let mut best_lag = 0;
            let mut best_peak = 0.0f64;

            let max_lag = self.cfg.max_lag.min(data.len().saturating_sub(template.len()));

            for lag in 0..=max_lag {
                if lag + template.len() > data.len() {
                    break;
                }
                let window = &data[lag..lag + template.len()];

                // Inner product ⟨d, h⟩.
                let inner: f64 = window.iter().zip(template.iter()).map(|(d, h)| d * h).sum();
                let snr = inner.abs() / h_norm;

                if snr > best_snr {
                    best_snr = snr;
                    best_lag = lag;
                    best_peak = window.iter().cloned().fold(0.0_f64, |acc, v| acc.max(v.abs()));
                }
            }

            if best_snr >= self.cfg.snr_threshold {
                events.push(ShockPropagationEvent {
                    source: source.clone(),
                    target: target.to_string(),
                    lag_ticks: best_lag,
                    snr: best_snr,
                    target_peak_return: best_peak,
                });
            }
        }

        events
    }

    /// Registered source symbols.
    pub fn sources(&self) -> Vec<&str> {
        self.templates.keys().map(String::as_str).collect()
    }

    /// Registered target symbols.
    pub fn targets(&self) -> Vec<&str> {
        self.target_returns.keys().map(String::as_str).collect()
    }
}

/// Build a simple chirp-like shock template from a vector of large returns.
///
/// The template is centred on the peak return, with an exponential envelope.
/// Useful for auto-generating a template from an observed shock event.
pub fn build_template_from_shock(returns: &[f64], half_width: usize) -> Vec<f64> {
    if returns.is_empty() {
        return Vec::new();
    }
    // Find the index of the peak absolute return.
    let peak_idx = returns
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i)
        .unwrap_or(0);

    let start = peak_idx.saturating_sub(half_width);
    let end = (peak_idx + half_width + 1).min(returns.len());
    returns[start..end].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_chirp(amplitude: f64, len: usize) -> Vec<f64> {
        // Simple spike at centre with exponential decay.
        let mid = len / 2;
        (0..len)
            .map(|i| {
                let dist = (i as f64 - mid as f64).abs();
                amplitude * (-dist * 0.5).exp() * if i == mid { 1.0 } else { -0.1 }
            })
            .collect()
    }

    #[test]
    fn detects_exact_template_with_zero_lag() {
        let cfg = ShockConfig {
            snr_threshold: 2.0,
            max_lag: 10,
            min_returns: 3,
        };
        let mut det = ShockDetector::new(cfg);
        let template = make_chirp(0.05, 7);
        det.set_template("SPY".to_string(), template.clone());

        for &r in &template {
            det.push_target_return("QQQ".to_string(), r);
        }
        // Pad with zeros so there is enough history.
        for _ in 0..5 {
            det.push_target_return("QQQ".to_string(), 0.0);
        }

        let events = det.detect("QQQ");
        assert!(!events.is_empty(), "should detect template in matching data");
    }

    #[test]
    fn no_detection_for_flat_target() {
        let cfg = ShockConfig::default();
        let mut det = ShockDetector::new(cfg);
        det.set_template("SPY".to_string(), vec![0.0, 0.08, -0.03, 0.01]);
        for _ in 0..25 {
            det.push_target_return("FLAT".to_string(), 0.0);
        }
        let events = det.detect("FLAT");
        assert!(events.is_empty(), "flat target should not trigger detection");
    }

    #[test]
    fn build_template_centred_on_peak() {
        let returns = vec![0.001, 0.002, 0.08, 0.003, 0.001];
        let tpl = build_template_from_shock(&returns, 1);
        // Peak is at index 2; half_width=1 → [idx 1..4].
        assert_eq!(tpl, vec![0.002, 0.08, 0.003]);
    }

    #[test]
    fn no_detection_below_min_returns() {
        let cfg = ShockConfig { min_returns: 20, ..ShockConfig::default() };
        let mut det = ShockDetector::new(cfg);
        det.set_template("SPY".to_string(), vec![0.0, 0.05, -0.02]);
        det.push_target_return("QQQ".to_string(), 0.05);
        assert!(det.detect("QQQ").is_empty());
    }
}
