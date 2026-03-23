//! Minkowski light-cone analysis for market event causality.
//!
//! ## Motivation
//!
//! In special relativity, the **light cone** around a spacetime event `E`
//! divides the rest of spacetime into three regions:
//!
//! - **Causal past** (past light cone): events that *could* have caused `E`.
//! - **Causal future** (future light cone): events that `E` *could* cause.
//! - **Spacelike separated**: events that have *no causal relation* to `E`
//!   — no signal could travel between them, even at light speed.
//!
//! Applied to financial markets:
//!
//! - Replace "light speed" with `c_scale`: the maximum meaningful
//!   log-return-per-tick (e.g. 5% per minute for a liquid equity).
//! - A price shock at tick `t₀` with magnitude `|Δp₀|` defines a light cone.
//!   Any subsequent price move at tick `t₁ > t₀` lies in the **causal future**
//!   of the shock only if `|Δp₁| / (t₁ - t₀) ≤ c_scale` — i.e. the
//!   information could have propagated that fast.
//! - Events *outside* the cone are spacelike-separated: the later price move
//!   was driven by an independent cause, not the earlier shock.
//!
//! ## Spacetime Interval
//!
//! For two events `A = (t_A, x_A)` and `B = (t_B, x_B)`:
//!
//! ```text
//! Δs² = c² Δt² - Δx²
//!      where Δt = t_B - t_A
//!            Δx = x_B - x_A   (log-price difference)
//!            c  = c_scale
//! ```
//!
//! | `Δs²` | Region | Interpretation |
//! |--------|--------|----------------|
//! | > 0   | **Timelike** | B is in A's causal light cone |
//! | = 0   | **Lightlike** | B lies exactly on the light cone boundary |
//! | < 0   | **Spacelike** | B is causally disconnected from A |
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::light_cone::{LightConeAnalyzer, MarketEvent, LightConeConfig};
//!
//! let cfg = LightConeConfig::default();
//! let analyzer = LightConeAnalyzer::new(cfg);
//!
//! let shock = MarketEvent { tick: 0.0, log_price: 4.605, label: "shock".to_string() };
//! let later = MarketEvent { tick: 5.0, log_price: 4.620, label: "follow".to_string() };
//!
//! let relation = analyzer.classify_pair(&shock, &later);
//! println!("{:?}", relation.region);
//! ```

use serde::{Deserialize, Serialize};

// ─── CONFIGURATION ───────────────────────────────────────────────────────────

/// Configuration for light-cone analysis.
#[derive(Debug, Clone)]
pub struct LightConeConfig {
    /// Speed-of-light scale `c` — maximum log-return per unit tick time.
    /// Typical value: 0.05 (5% per tick for intraday equities).
    pub c_scale: f64,
    /// Minimum `|Δt|` to avoid numerical singularities.
    pub min_dt: f64,
    /// When `|Δs²| < epsilon_threshold` classify as lightlike.
    pub epsilon_threshold: f64,
}

impl Default for LightConeConfig {
    fn default() -> Self {
        Self {
            c_scale: 0.05,
            min_dt: 1e-6,
            epsilon_threshold: 1e-8,
        }
    }
}

// ─── MARKET EVENT ────────────────────────────────────────────────────────────

/// A single market event represented as a point in Minkowski spacetime.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEvent {
    /// Coordinate time (tick index, seconds, minutes, etc.).
    pub tick: f64,
    /// Log-price at this event (ln of price).
    pub log_price: f64,
    /// Human-readable label (ticker, event name, etc.).
    pub label: String,
}

impl MarketEvent {
    pub fn new(tick: f64, log_price: f64, label: impl Into<String>) -> Self {
        Self { tick, log_price, label: label.into() }
    }
}

// ─── CAUSAL REGION ───────────────────────────────────────────────────────────

/// The causal relationship between two market events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CausalRegion {
    /// `Δs² > 0`: B is in A's causal future or past.
    Timelike,
    /// `|Δs²| < ε`: B lies on A's light cone boundary.
    Lightlike,
    /// `Δs² < 0`: B is causally disconnected from A.
    Spacelike,
}

impl CausalRegion {
    pub fn label(self) -> &'static str {
        match self {
            CausalRegion::Timelike => "timelike (causal)",
            CausalRegion::Lightlike => "lightlike (boundary)",
            CausalRegion::Spacelike => "spacelike (independent)",
        }
    }
}

/// The temporal ordering from A's perspective.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalOrder {
    /// B is in A's future (`t_B > t_A`).
    Future,
    /// B is in A's past (`t_B < t_A`).
    Past,
    /// Same tick — simultaneous.
    Simultaneous,
}

// ─── PAIR RESULT ─────────────────────────────────────────────────────────────

/// Full analysis result for a pair of market events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PairCausalityResult {
    pub event_a: String,
    pub event_b: String,
    /// Coordinate-time separation `Δt = t_B - t_A`.
    pub delta_t: f64,
    /// Log-price separation `Δx = x_B - x_A`.
    pub delta_x: f64,
    /// Spacetime interval squared: `c²Δt² - Δx²`.
    pub interval_sq: f64,
    /// Spacetime interval: `sqrt(|Δs²|)` with sign of `Δs²`.
    pub interval_signed: f64,
    /// Causal region classification.
    pub region: CausalRegion,
    /// Temporal ordering of B relative to A.
    pub temporal_order: TemporalOrder,
    /// `true` when B is in A's forward light cone (A could have caused B).
    pub a_could_cause_b: bool,
    /// Signal speed required for A to influence B: `|Δx| / |Δt|` in units of `c`.
    /// Values > 1.0 mean superluminal (spacelike).
    pub required_speed_fraction: f64,
}

// ─── CONE SUMMARY ────────────────────────────────────────────────────────────

/// Summary of a full cone analysis: one anchor event vs a set of probe events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConeSummary {
    pub anchor_label: String,
    pub anchor_tick: f64,
    pub anchor_log_price: f64,
    pub c_scale: f64,
    pub total_probes: usize,
    pub timelike_count: usize,
    pub lightlike_count: usize,
    pub spacelike_count: usize,
    pub causal_future_count: usize,
    pub causal_past_count: usize,
    /// Fraction of probe events that are causally connected to the anchor.
    pub causal_fraction: f64,
    /// Average required signal speed (in units of c) across all probes.
    pub mean_required_speed: f64,
    /// Individual pair results.
    pub pairs: Vec<PairCausalityResult>,
}

impl ConeSummary {
    /// Short human-readable summary.
    pub fn summary_line(&self) -> String {
        format!(
            "Anchor '{}' @ t={:.1}: {}/{} probes causally connected ({:.1}%), \
             mean speed {:.3}c, future={}, past={}",
            self.anchor_label,
            self.anchor_tick,
            self.timelike_count + self.lightlike_count,
            self.total_probes,
            self.causal_fraction * 100.0,
            self.mean_required_speed,
            self.causal_future_count,
            self.causal_past_count,
        )
    }
}

// ─── SHOCK DETECTION ─────────────────────────────────────────────────────────

/// A detected price shock (candidate anchor event for cone analysis).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PriceShock {
    pub tick: f64,
    pub log_price: f64,
    /// Absolute log-return magnitude at this tick.
    pub magnitude: f64,
    /// Z-score of the return relative to the rolling window.
    pub z_score: f64,
}

// ─── ANALYZER ────────────────────────────────────────────────────────────────

/// The light-cone analysis engine.
pub struct LightConeAnalyzer {
    cfg: LightConeConfig,
}

impl LightConeAnalyzer {
    pub fn new(cfg: LightConeConfig) -> Self {
        Self { cfg }
    }

    // ── Pair classification ───────────────────────────────────────────────────

    /// Classify the causal relationship between two market events.
    pub fn classify_pair(&self, a: &MarketEvent, b: &MarketEvent) -> PairCausalityResult {
        let dt = b.tick - a.tick;
        let dx = b.log_price - a.log_price;
        let c = self.cfg.c_scale;

        let interval_sq = c * c * dt * dt - dx * dx;
        let interval_signed = if interval_sq >= 0.0 {
            interval_sq.sqrt()
        } else {
            -(-interval_sq).sqrt()
        };

        let region = if interval_sq.abs() < self.cfg.epsilon_threshold {
            CausalRegion::Lightlike
        } else if interval_sq > 0.0 {
            CausalRegion::Timelike
        } else {
            CausalRegion::Spacelike
        };

        let temporal_order = if dt.abs() < self.cfg.min_dt {
            TemporalOrder::Simultaneous
        } else if dt > 0.0 {
            TemporalOrder::Future
        } else {
            TemporalOrder::Past
        };

        // A can cause B only if B is timelike/lightlike AND in A's future.
        let a_could_cause_b = matches!(region, CausalRegion::Timelike | CausalRegion::Lightlike)
            && temporal_order == TemporalOrder::Future;

        let required_speed_fraction = if dt.abs() < self.cfg.min_dt {
            f64::INFINITY
        } else {
            dx.abs() / (c * dt.abs())
        };

        PairCausalityResult {
            event_a: a.label.clone(),
            event_b: b.label.clone(),
            delta_t: dt,
            delta_x: dx,
            interval_sq,
            interval_signed,
            region,
            temporal_order,
            a_could_cause_b,
            required_speed_fraction,
        }
    }

    /// Compute the spacetime interval squared between two raw `(tick, log_price)` points.
    pub fn interval_sq(&self, t_a: f64, x_a: f64, t_b: f64, x_b: f64) -> f64 {
        let c = self.cfg.c_scale;
        let dt = t_b - t_a;
        let dx = x_b - x_a;
        c * c * dt * dt - dx * dx
    }

    // ── Full cone analysis ────────────────────────────────────────────────────

    /// Analyse how many probe events lie inside the anchor event's light cone.
    pub fn analyze_cone(&self, anchor: &MarketEvent, probes: &[MarketEvent]) -> ConeSummary {
        let pairs: Vec<PairCausalityResult> = probes
            .iter()
            .map(|p| self.classify_pair(anchor, p))
            .collect();

        let timelike_count = pairs.iter().filter(|p| p.region == CausalRegion::Timelike).count();
        let lightlike_count = pairs.iter().filter(|p| p.region == CausalRegion::Lightlike).count();
        let spacelike_count = pairs.iter().filter(|p| p.region == CausalRegion::Spacelike).count();
        let causal_future_count = pairs.iter().filter(|p| p.a_could_cause_b).count();
        let causal_past_count = pairs
            .iter()
            .filter(|p| {
                matches!(p.region, CausalRegion::Timelike | CausalRegion::Lightlike)
                    && p.temporal_order == TemporalOrder::Past
            })
            .count();

        let total = probes.len();
        let causal_fraction = if total == 0 {
            0.0
        } else {
            (timelike_count + lightlike_count) as f64 / total as f64
        };

        let mean_required_speed = if pairs.is_empty() {
            0.0
        } else {
            let finite_speeds: Vec<f64> = pairs
                .iter()
                .filter(|p| p.required_speed_fraction.is_finite())
                .map(|p| p.required_speed_fraction)
                .collect();
            if finite_speeds.is_empty() {
                f64::INFINITY
            } else {
                finite_speeds.iter().sum::<f64>() / finite_speeds.len() as f64
            }
        };

        ConeSummary {
            anchor_label: anchor.label.clone(),
            anchor_tick: anchor.tick,
            anchor_log_price: anchor.log_price,
            c_scale: self.cfg.c_scale,
            total_probes: total,
            timelike_count,
            lightlike_count,
            spacelike_count,
            causal_future_count,
            causal_past_count,
            causal_fraction,
            mean_required_speed,
            pairs,
        }
    }

    // ── Price shock detection ─────────────────────────────────────────────────

    /// Detect significant price shocks in a time-series of `(tick, log_price)` pairs.
    ///
    /// A shock is any return whose absolute value exceeds `z_threshold` standard
    /// deviations from the rolling mean over `window` observations.
    pub fn detect_shocks(
        &self,
        series: &[(f64, f64)],
        window: usize,
        z_threshold: f64,
    ) -> Vec<PriceShock> {
        if series.len() < 2 {
            return Vec::new();
        }

        let window = window.max(2);
        let mut shocks = Vec::new();

        // Compute log-returns.
        let returns: Vec<f64> = series
            .windows(2)
            .map(|w| w[1].1 - w[0].1)
            .collect();

        for i in 0..returns.len() {
            let ret = returns[i];
            // Rolling window ending at i.
            let lo = i.saturating_sub(window - 1);
            let hi = i + 1;
            let window_slice = &returns[lo..hi];

            let n = window_slice.len() as f64;
            let mean = window_slice.iter().sum::<f64>() / n;
            let variance = window_slice.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / n;
            let std_dev = variance.sqrt();

            let z = if std_dev < 1e-12 { 0.0 } else { (ret - mean) / std_dev };

            if z.abs() >= z_threshold {
                shocks.push(PriceShock {
                    tick: series[i + 1].0,
                    log_price: series[i + 1].1,
                    magnitude: ret.abs(),
                    z_score: z,
                });
            }
        }

        shocks
    }

    /// Build `MarketEvent` list from shock detection results and run a full
    /// multi-cone analysis: for each shock, count how many subsequent events
    /// fall inside its forward light cone.
    pub fn multi_cone_analysis(
        &self,
        series: &[(f64, f64)],
        window: usize,
        z_threshold: f64,
    ) -> Vec<ConeSummary> {
        let shocks = self.detect_shocks(series, window, z_threshold);
        let all_events: Vec<MarketEvent> = series
            .iter()
            .enumerate()
            .map(|(i, &(t, lp))| MarketEvent::new(t, lp, format!("t{}", i)))
            .collect();

        shocks
            .iter()
            .map(|shock| {
                let anchor = MarketEvent::new(shock.tick, shock.log_price, "shock");
                // Only probe events *after* the shock.
                let probes: Vec<MarketEvent> = all_events
                    .iter()
                    .filter(|e| e.tick > shock.tick)
                    .cloned()
                    .collect();
                self.analyze_cone(&anchor, &probes)
            })
            .collect()
    }
}

// ─── TESTS ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn analyzer() -> LightConeAnalyzer {
        LightConeAnalyzer::new(LightConeConfig::default())
    }

    fn ev(tick: f64, lp: f64, label: &str) -> MarketEvent {
        MarketEvent::new(tick, lp, label)
    }

    #[test]
    fn timelike_slow_move() {
        // Small Δx over large Δt → inside the cone.
        let a = ev(0.0, 4.600, "A");
        let b = ev(10.0, 4.601, "B"); // Δx=0.001 << c*Δt=0.5
        let r = analyzer().classify_pair(&a, &b);
        assert_eq!(r.region, CausalRegion::Timelike);
        assert!(r.a_could_cause_b);
    }

    #[test]
    fn spacelike_fast_move() {
        // Large Δx over small Δt → outside the cone.
        let a = ev(0.0, 4.600, "A");
        let b = ev(1.0, 5.200, "B"); // Δx=0.6 >> c*Δt=0.05
        let r = analyzer().classify_pair(&a, &b);
        assert_eq!(r.region, CausalRegion::Spacelike);
        assert!(!r.a_could_cause_b);
    }

    #[test]
    fn past_event_not_causal_future() {
        let a = ev(5.0, 4.600, "A");
        let b = ev(0.0, 4.595, "B"); // B is before A
        let r = analyzer().classify_pair(&a, &b);
        assert!(!r.a_could_cause_b);
        assert_eq!(r.temporal_order, TemporalOrder::Past);
    }

    #[test]
    fn interval_sq_formula() {
        let cfg = LightConeConfig { c_scale: 1.0, ..Default::default() };
        let ana = LightConeAnalyzer::new(cfg);
        // Δt=3, Δx=4 → Δs²=9-16=-7 (spacelike)
        let s2 = ana.interval_sq(0.0, 0.0, 3.0, 4.0);
        assert!((s2 - (-7.0)).abs() < 1e-9);
    }

    #[test]
    fn cone_summary_counts_consistent() {
        let anchor = ev(0.0, 4.600, "anchor");
        let probes = vec![
            ev(10.0, 4.601, "p1"),   // slow → timelike
            ev(1.0, 5.500, "p2"),    // fast → spacelike
            ev(20.0, 4.605, "p3"),   // slow → timelike
        ];
        let summary = analyzer().analyze_cone(&anchor, &probes);
        assert_eq!(
            summary.timelike_count + summary.lightlike_count + summary.spacelike_count,
            3
        );
        assert_eq!(summary.total_probes, 3);
        assert!((summary.causal_fraction >= 0.0) && (summary.causal_fraction <= 1.0));
    }

    #[test]
    fn detect_shocks_finds_outliers() {
        // Flat series with one spike.
        let mut series: Vec<(f64, f64)> = (0..20).map(|i| (i as f64, 4.600)).collect();
        series[10].1 = 4.700; // 10% spike
        let shocks = analyzer().detect_shocks(&series, 10, 2.0);
        assert!(!shocks.is_empty(), "should detect at least one shock");
    }

    #[test]
    fn required_speed_finite_and_positive() {
        let a = ev(0.0, 4.600, "A");
        let b = ev(2.0, 4.610, "B");
        let r = analyzer().classify_pair(&a, &b);
        assert!(r.required_speed_fraction.is_finite());
        assert!(r.required_speed_fraction >= 0.0);
    }

    #[test]
    fn summary_line_does_not_panic() {
        let anchor = ev(0.0, 4.0, "anchor");
        let probes = vec![ev(1.0, 4.001, "p")];
        let summary = analyzer().analyze_cone(&anchor, &probes);
        let _s = summary.summary_line();
    }
}
