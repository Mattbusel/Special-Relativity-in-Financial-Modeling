//! Geodesic trading signals from Minkowski-space price trajectories.
//!
//! ## Concept
//!
//! A price time-series is embedded as a worldline in 1+1 Minkowski spacetime:
//! each data point becomes a spacetime event `(t, x)` where `t` is coordinate
//! time (tick index normalised to the speed-of-light scale) and `x` is
//! log-price.  The geodesic curvature of that worldline measures how sharply
//! the trajectory is bending away from a straight-line (constant-velocity)
//! path.
//!
//! In Minkowski geometry the signed curvature of a timelike curve is the
//! proper acceleration (rate of change of proper velocity):
//!
//! ```text
//! κ = dα/dτ   where α = rapidity = atanh(v/c)
//! ```
//!
//! For a discretised worldline we approximate:
//!
//! ```text
//! κ_n ≈ Δα / Δτ
//!     where Δα = α_n - α_{n-1}
//!           α  = atanh(β)   β = dx/dt   (normalised velocity)
//!           Δτ = proper-time interval between ticks
//! ```
//!
//! **High curvature signals trend exhaustion**: a worldline that is bending
//! sharply is decelerating in Minkowski space, analogous to a particle
//! approaching its turning point.
//!
//! ## Signal Generation
//!
//! When `|κ_n|` exceeds a configurable threshold a [`GeodesicSignal`] is
//! emitted.  The sign of the curvature indicates the direction:
//!
//! - `κ > threshold` → upward exhaustion → potential short / exit long
//! - `κ < -threshold` → downward exhaustion → potential long / exit short
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::geodesic_signals::{GeodesicDetector, GeodesicConfig};
//!
//! let mut detector = GeodesicDetector::new(GeodesicConfig::default());
//! // Feed (tick_index, log_price) pairs.
//! for (t, lp) in [(0.0, 10.0), (1.0, 10.01), (2.0, 10.025), (3.0, 10.02)] {
//!     if let Some(signal) = detector.update(t, lp) {
//!         println!("geodesic signal: {signal:?}");
//!     }
//! }
//! ```

/// Configuration for geodesic curvature signal detection.
#[derive(Debug, Clone)]
pub struct GeodesicConfig {
    /// Speed-of-light scale `c` used to normalise the velocity `β = v/c`.
    /// Set to a value larger than the maximum expected log-return per tick.
    pub c_scale: f64,

    /// Curvature magnitude threshold above which a signal is generated.
    pub curvature_threshold: f64,

    /// Minimum proper-time interval required to compute a valid curvature.
    /// Prevents division-by-zero on degenerate segments.
    pub min_proper_time: f64,

    /// Rolling window length for smoothing the curvature series before
    /// signal detection.
    pub smooth_window: usize,
}

impl Default for GeodesicConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,          // 5 % log-return per tick is "speed of light"
            curvature_threshold: 1.5,
            min_proper_time: 1e-6,
            smooth_window: 5,
        }
    }
}

/// Direction of a geodesic exhaustion signal.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SignalDirection {
    /// Upward trend exhaustion — consider exiting longs / initiating shorts.
    UpwardExhaustion,
    /// Downward trend exhaustion — consider exiting shorts / initiating longs.
    DownwardExhaustion,
}

impl std::fmt::Display for SignalDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SignalDirection::UpwardExhaustion => write!(f, "UpwardExhaustion"),
            SignalDirection::DownwardExhaustion => write!(f, "DownwardExhaustion"),
        }
    }
}

/// A trading signal generated when worldline curvature exceeds the threshold.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GeodesicSignal {
    /// Coordinate time of the triggering tick.
    pub coord_time: f64,
    /// Log-price at the triggering tick.
    pub log_price: f64,
    /// Signed geodesic curvature (proper acceleration) at this point.
    pub curvature: f64,
    /// Smoothed curvature (average over the rolling window).
    pub smoothed_curvature: f64,
    /// Direction of the exhaustion signal.
    pub direction: SignalDirection,
}

/// Internal per-tick state for curvature computation.
#[derive(Debug, Clone)]
struct TickState {
    t: f64,
    x: f64,
    rapidity: f64,
    proper_time: f64,
}

/// Streaming geodesic curvature detector.
///
/// Maintains a minimal rolling history of worldline states and emits
/// [`GeodesicSignal`]s whenever the curvature magnitude exceeds the threshold.
#[derive(Debug, Clone)]
pub struct GeodesicDetector {
    cfg: GeodesicConfig,
    /// Last two tick states (we need exactly two consecutive segments).
    prev: Option<TickState>,
    /// Rolling buffer of recent curvature values for smoothing.
    curvature_window: std::collections::VecDeque<f64>,
    /// All signals emitted so far.
    signals: Vec<GeodesicSignal>,
}

impl GeodesicDetector {
    /// Create a new detector with the given configuration.
    pub fn new(cfg: GeodesicConfig) -> Self {
        Self {
            cfg,
            prev: None,
            curvature_window: std::collections::VecDeque::new(),
            signals: Vec::new(),
        }
    }

    /// Ingest a new `(coord_time, log_price)` event.
    ///
    /// Returns `Some(signal)` if a curvature threshold crossing was detected,
    /// `None` otherwise.  The detector requires at least two prior events
    /// before any curvature can be computed.
    pub fn update(&mut self, coord_time: f64, log_price: f64) -> Option<GeodesicSignal> {
        if let Some(prev) = self.prev.take() {
            let dt = coord_time - prev.t;
            let dx = log_price - prev.x;

            if dt.abs() < 1e-12 {
                // Degenerate segment — skip but keep state.
                self.prev = Some(prev);
                return None;
            }

            // Coordinate velocity β = (dx/dt) / c  clamped to (-1, 1).
            let beta = (dx / dt / self.cfg.c_scale).clamp(-0.9999, 0.9999);

            // Proper time interval: Δτ = Δt · √(1 - β²).
            let gamma_inv = (1.0 - beta * beta).sqrt();
            let proper_time = dt.abs() * gamma_inv;

            // Rapidity α = atanh(β).
            let rapidity = beta.atanh();

            // Curvature (proper acceleration) = Δα / Δτ.
            let curvature = if proper_time > self.cfg.min_proper_time {
                (rapidity - prev.rapidity) / proper_time
            } else {
                0.0
            };

            // Update smoothing window.
            self.curvature_window.push_back(curvature);
            if self.curvature_window.len() > self.cfg.smooth_window {
                self.curvature_window.pop_front();
            }
            let smoothed = self.curvature_window.iter().sum::<f64>()
                / self.curvature_window.len() as f64;

            // Advance state.
            self.prev = Some(TickState {
                t: coord_time,
                x: log_price,
                rapidity,
                proper_time,
            });

            // Signal check on smoothed curvature.
            if smoothed.abs() >= self.cfg.curvature_threshold {
                let direction = if smoothed > 0.0 {
                    SignalDirection::UpwardExhaustion
                } else {
                    SignalDirection::DownwardExhaustion
                };
                let sig = GeodesicSignal {
                    coord_time,
                    log_price,
                    curvature,
                    smoothed_curvature: smoothed,
                    direction,
                };
                self.signals.push(sig.clone());
                return Some(sig);
            }

            None
        } else {
            // First tick: initialise state.
            self.prev = Some(TickState {
                t: coord_time,
                x: log_price,
                rapidity: 0.0,
                proper_time: 0.0,
            });
            None
        }
    }

    /// All signals emitted since construction.
    pub fn signal_history(&self) -> &[GeodesicSignal] {
        &self.signals
    }

    /// Current smoothed curvature (most recent window average).
    pub fn current_smoothed_curvature(&self) -> f64 {
        if self.curvature_window.is_empty() {
            return 0.0;
        }
        self.curvature_window.iter().sum::<f64>() / self.curvature_window.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_signal_on_first_tick() {
        let mut d = GeodesicDetector::new(GeodesicConfig::default());
        assert!(d.update(0.0, 10.0).is_none());
    }

    #[test]
    fn no_signal_on_linear_trajectory() {
        // A perfectly linear trajectory has zero curvature.
        let mut d = GeodesicDetector::new(GeodesicConfig::default());
        for i in 0..20 {
            let sig = d.update(i as f64, 10.0 + i as f64 * 0.001);
            // Linear trajectory should never exceed threshold.
            assert!(sig.is_none(), "unexpected signal on linear trajectory at tick {i}");
        }
    }

    #[test]
    fn upward_exhaustion_after_sharp_deceleration() {
        let cfg = GeodesicConfig {
            curvature_threshold: 0.01,
            smooth_window: 1,
            ..GeodesicConfig::default()
        };
        let mut d = GeodesicDetector::new(cfg);
        // Accelerate upward then sharply decelerate.
        let prices = [0.0, 0.01, 0.025, 0.04, 0.045, 0.0449, 0.0448];
        let mut found = false;
        for (i, &p) in prices.iter().enumerate() {
            if let Some(sig) = d.update(i as f64, p) {
                if sig.direction == SignalDirection::UpwardExhaustion {
                    found = true;
                }
            }
        }
        assert!(found, "expected upward exhaustion signal");
    }

    #[test]
    fn signal_history_grows() {
        let cfg = GeodesicConfig {
            curvature_threshold: 0.001,
            smooth_window: 1,
            ..GeodesicConfig::default()
        };
        let mut d = GeodesicDetector::new(cfg);
        for i in 0..30 {
            let price = if i % 6 < 3 { i as f64 * 0.01 } else { -(i as f64) * 0.01 };
            d.update(i as f64, price);
        }
        assert!(!d.signal_history().is_empty());
    }
}
