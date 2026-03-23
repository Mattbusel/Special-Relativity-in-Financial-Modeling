//! Penrose diagram for market microstructure causal structure.
//!
//! ## Concept
//!
//! A **Penrose diagram** (conformal diagram) compresses the infinite extent of
//! Minkowski spacetime into a finite square, preserving causal relationships:
//! light cones always appear at 45° and it becomes trivial to read off which
//! events can causally influence which others.
//!
//! Applied to market microstructure:
//!
//! - **Events**: individual trades / quotes / order-book updates represented
//!   as spacetime points `(t, x)` where `t` is time and `x` is log-price.
//! - **Light cones**: the maximum speed at which price information propagates
//!   (parameterised by `c`, the "market speed of light" set to the maximum
//!   physically plausible tick-to-tick log-return).
//! - **Event horizons**: a price level beyond which a trajectory cannot
//!   return given its current velocity — the analogue of a black-hole horizon.
//! - **Causal disconnection**: two events are causally disconnected when they
//!   lie outside each other's light cones; their correlation is therefore
//!   spurious.
//!
//! ## Conformal Mapping
//!
//! The standard Penrose coordinates `(T, X)` are obtained via:
//!
//! ```text
//! u = t - x   (retarded null coordinate)
//! v = t + x   (advanced null coordinate)
//! U = arctan(u)
//! V = arctan(v)
//! T = U + V   (conformal time)
//! X = V - U   (conformal space)
//! ```
//!
//! In this module `t` and `x` are normalised before the transform so that
//! the diagram fits neatly in `(-π, π)²`.
//!
//! ## Usage
//!
//! ```rust
//! use tokio_prompt_orchestrator::penrose::{PenroseDiagram, MarketEvent, PenroseConfig};
//!
//! let mut diagram = PenroseDiagram::new(PenroseConfig::default());
//! diagram.add_event(MarketEvent::new("trade_1", 0.0, 10.0));
//! diagram.add_event(MarketEvent::new("trade_2", 1.0, 10.01));
//!
//! let connected = diagram.are_causally_connected("trade_1", "trade_2");
//! println!("causally connected: {connected}");
//! ```

use std::collections::HashMap;

/// Configuration for the Penrose diagram mapper.
#[derive(Debug, Clone)]
pub struct PenroseConfig {
    /// Market speed of light: maximum log-return per unit time.
    /// Events separated by `|Δx/Δt| > c` are causally disconnected.
    pub c: f64,

    /// Normalisation scale for the time axis before conformal mapping.
    /// Set to the expected total duration of the event window.
    pub time_scale: f64,

    /// Normalisation scale for the price axis before conformal mapping.
    /// Set to the expected log-price range of the event window.
    pub price_scale: f64,

    /// Minimum speed fraction of `c` at which an event horizon is flagged.
    /// When `|v| / c >= horizon_fraction`, the trajectory is near-horizon.
    pub horizon_fraction: f64,
}

impl Default for PenroseConfig {
    fn default() -> Self {
        Self {
            c: 0.05,
            time_scale: 100.0,
            price_scale: 1.0,
            horizon_fraction: 0.90,
        }
    }
}

/// A single microstructure event in the Penrose diagram.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MarketEvent {
    /// Unique label for this event.
    pub id: String,
    /// Coordinate time (e.g. tick index or unix seconds).
    pub coord_time: f64,
    /// Log-price at the time of this event.
    pub log_price: f64,
    /// Conformal time coordinate `T` after Penrose mapping.
    pub penrose_t: f64,
    /// Conformal space coordinate `X` after Penrose mapping.
    pub penrose_x: f64,
}

impl MarketEvent {
    /// Create an event; Penrose coordinates are computed lazily by the diagram.
    pub fn new(id: impl Into<String>, coord_time: f64, log_price: f64) -> Self {
        Self {
            id: id.into(),
            coord_time,
            log_price,
            penrose_t: 0.0,
            penrose_x: 0.0,
        }
    }
}

/// Causal relationship between two events.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CausalRelation {
    /// Event A is in the past light cone of event B (A can influence B).
    PastLightCone,
    /// Event A is in the future light cone of event B.
    FutureLightCone,
    /// Events are spacelike separated — no causal link possible.
    SpacelikeSeparated,
    /// Events are lightlike separated (exactly on the light cone).
    LightlikeSeparated,
}

impl std::fmt::Display for CausalRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CausalRelation::PastLightCone => write!(f, "PastLightCone"),
            CausalRelation::FutureLightCone => write!(f, "FutureLightCone"),
            CausalRelation::SpacelikeSeparated => write!(f, "SpacelikeSeparated"),
            CausalRelation::LightlikeSeparated => write!(f, "LightlikeSeparated"),
        }
    }
}

/// An identified event horizon: a trajectory point beyond which the price
/// cannot return without superluminal velocity.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EventHorizon {
    /// Event ID at which the horizon was identified.
    pub event_id: String,
    /// Estimated "escape velocity" as a fraction of `c`.
    pub velocity_fraction: f64,
    /// Penrose-space coordinates of the horizon point.
    pub penrose_t: f64,
    pub penrose_x: f64,
}

/// The Penrose diagram for a sequence of market microstructure events.
#[derive(Debug, Clone)]
pub struct PenroseDiagram {
    cfg: PenroseConfig,
    events: Vec<MarketEvent>,
    index: HashMap<String, usize>,
}

impl PenroseDiagram {
    /// Create an empty diagram.
    pub fn new(cfg: PenroseConfig) -> Self {
        Self {
            cfg,
            events: Vec::new(),
            index: HashMap::new(),
        }
    }

    /// Add an event to the diagram and compute its Penrose coordinates.
    pub fn add_event(&mut self, mut event: MarketEvent) {
        let (pt, px) = self.penrose_coords(event.coord_time, event.log_price);
        event.penrose_t = pt;
        event.penrose_x = px;
        let idx = self.events.len();
        self.index.insert(event.id.clone(), idx);
        self.events.push(event);
    }

    /// Retrieve a stored event by ID.
    pub fn get_event(&self, id: &str) -> Option<&MarketEvent> {
        self.index.get(id).and_then(|&i| self.events.get(i))
    }

    /// Determine the causal relationship between two events `a` and `b`.
    ///
    /// Uses the Minkowski interval with speed of light `c`.
    pub fn causal_relation(&self, id_a: &str, id_b: &str) -> Option<CausalRelation> {
        let a = self.get_event(id_a)?;
        let b = self.get_event(id_b)?;
        let dt = b.coord_time - a.coord_time;
        let dx = b.log_price - a.log_price;
        // Interval² = (c·Δt)² - Δx²
        let interval_sq = (self.cfg.c * dt).powi(2) - dx.powi(2);
        const LIGHTLIKE_TOL: f64 = 1e-10;
        let relation = if interval_sq.abs() < LIGHTLIKE_TOL {
            CausalRelation::LightlikeSeparated
        } else if interval_sq > 0.0 {
            // Timelike: causally connected.
            if dt >= 0.0 {
                CausalRelation::PastLightCone // a is in the past of b
            } else {
                CausalRelation::FutureLightCone
            }
        } else {
            CausalRelation::SpacelikeSeparated
        };
        Some(relation)
    }

    /// Return `true` if event `a` can causally influence event `b`
    /// (i.e. `a` is in `b`'s past light cone).
    pub fn are_causally_connected(&self, id_a: &str, id_b: &str) -> bool {
        matches!(
            self.causal_relation(id_a, id_b),
            Some(CausalRelation::PastLightCone | CausalRelation::LightlikeSeparated)
        )
    }

    /// Identify all events that are near an event horizon — where the implied
    /// velocity between consecutive events approaches `c`.
    pub fn event_horizons(&self) -> Vec<EventHorizon> {
        let mut horizons = Vec::new();
        if self.events.len() < 2 {
            return horizons;
        }
        for i in 1..self.events.len() {
            let prev = &self.events[i - 1];
            let curr = &self.events[i];
            let dt = curr.coord_time - prev.coord_time;
            if dt.abs() < 1e-12 {
                continue;
            }
            let velocity = (curr.log_price - prev.log_price) / dt;
            let velocity_fraction = (velocity / self.cfg.c).abs();
            if velocity_fraction >= self.cfg.horizon_fraction {
                horizons.push(EventHorizon {
                    event_id: curr.id.clone(),
                    velocity_fraction,
                    penrose_t: curr.penrose_t,
                    penrose_x: curr.penrose_x,
                });
            }
        }
        horizons
    }

    /// Find all pairs of events that are spacelike separated (causally
    /// disconnected) — correlations between these events are therefore
    /// not transmitted by market information flow.
    pub fn spacelike_pairs(&self) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        for i in 0..self.events.len() {
            for j in (i + 1)..self.events.len() {
                let a = &self.events[i];
                let b = &self.events[j];
                let dt = b.coord_time - a.coord_time;
                let dx = b.log_price - a.log_price;
                let interval_sq = (self.cfg.c * dt).powi(2) - dx.powi(2);
                if interval_sq < 0.0 {
                    pairs.push((a.id.clone(), b.id.clone()));
                }
            }
        }
        pairs
    }

    /// All events in insertion order.
    pub fn events(&self) -> &[MarketEvent] {
        &self.events
    }

    // ── private ───────────────────────────────────────────────────────────────

    fn penrose_coords(&self, t: f64, x: f64) -> (f64, f64) {
        let t_norm = t / self.cfg.time_scale;
        let x_norm = x / self.cfg.price_scale;
        let u = t_norm - x_norm;
        let v = t_norm + x_norm;
        let cap_u = u.atan();
        let cap_v = v.atan();
        (cap_u + cap_v, cap_v - cap_u) // (T, X)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_diagram() -> PenroseDiagram {
        PenroseDiagram::new(PenroseConfig::default())
    }

    #[test]
    fn sequential_slow_events_are_causally_connected() {
        let mut d = make_diagram();
        d.add_event(MarketEvent::new("e1", 0.0, 10.0));
        d.add_event(MarketEvent::new("e2", 10.0, 10.1)); // Δx/Δt = 0.01/10 << c
        assert!(d.are_causally_connected("e1", "e2"));
    }

    #[test]
    fn simultaneous_distant_events_are_spacelike() {
        let mut d = make_diagram();
        // Same time, different price => spacelike separated.
        d.add_event(MarketEvent::new("e1", 0.0, 10.0));
        d.add_event(MarketEvent::new("e2", 0.0, 11.0)); // Δt=0 → interval² = -Δx² < 0
        assert_eq!(
            d.causal_relation("e1", "e2"),
            Some(CausalRelation::SpacelikeSeparated)
        );
    }

    #[test]
    fn penrose_coords_are_finite_for_extreme_inputs() {
        let mut d = make_diagram();
        d.add_event(MarketEvent::new("e1", 1e9, 1e6));
        let e = d.get_event("e1").expect("event should be stored");
        // arctan is always finite.
        assert!(e.penrose_t.is_finite());
        assert!(e.penrose_x.is_finite());
    }

    #[test]
    fn event_horizons_detected_for_superluminal_move() {
        let cfg = PenroseConfig {
            c: 0.01,
            horizon_fraction: 0.90,
            ..PenroseConfig::default()
        };
        let mut d = PenroseDiagram::new(cfg);
        d.add_event(MarketEvent::new("e1", 0.0, 0.0));
        // Δx/Δt = 0.05 >> c=0.01 → velocity_fraction >> 1 → horizon.
        d.add_event(MarketEvent::new("e2", 1.0, 0.05));
        let horizons = d.event_horizons();
        assert!(!horizons.is_empty(), "expected horizon at superluminal event");
    }

    #[test]
    fn spacelike_pairs_list_matches_manual_check() {
        let mut d = make_diagram();
        // e1 @ (0, 0), e2 @ (0, 0.5): same time → spacelike.
        d.add_event(MarketEvent::new("e1", 0.0, 0.0));
        d.add_event(MarketEvent::new("e2", 0.0, 0.5));
        let pairs = d.spacelike_pairs();
        assert!(
            pairs.iter().any(|(a, b)| a == "e1" && b == "e2"),
            "expected (e1, e2) in spacelike pairs"
        );
    }

    #[test]
    fn unknown_event_returns_none() {
        let d = make_diagram();
        assert!(d.causal_relation("ghost", "nobody").is_none());
    }
}
