//! Causal sets and Alexandrov intervals for market event ordering.
//!
//! Uses Minkowski spacetime structure to classify causal relationships
//! between market events.

use std::collections::HashMap;

// ── Event ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Event {
    pub id: u64,
    pub time: f64,
    pub space: f64,
    pub label: String,
}

// ── CausalRelation ────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum CausalRelation {
    CausallyConnected,
    Spacelike,
    Lightlike,
}

// ── causal_relation ───────────────────────────────────────────────────────

/// Classify the causal relation between two events using Minkowski interval.
/// s² = -c²*(t2-t1)² + (x2-x1)²
/// s² < 0 => causally connected, s² > 0 => spacelike, s² == 0 => lightlike
pub fn causal_relation(a: &Event, b: &Event, c: f64) -> CausalRelation {
    let dt = b.time - a.time;
    let dx = b.space - a.space;
    let s_sq = -(c * c * dt * dt) + (dx * dx);

    if s_sq.abs() < 1e-12 {
        CausalRelation::Lightlike
    } else if s_sq < 0.0 {
        CausalRelation::CausallyConnected
    } else {
        CausalRelation::Spacelike
    }
}

/// Check if event `a` causally precedes event `b` (a is in past light cone of b).
fn causally_precedes(a: &Event, b: &Event, c: f64) -> bool {
    if b.time <= a.time {
        return false;
    }
    matches!(
        causal_relation(a, b, c),
        CausalRelation::CausallyConnected | CausalRelation::Lightlike
    )
}

// ── alexandrov_interval ───────────────────────────────────────────────────

/// Returns ids of events causally between a and b:
/// in the future light cone of a AND past light cone of b.
pub fn alexandrov_interval(a: &Event, b: &Event, events: &[Event], c: f64) -> Vec<u64> {
    events
        .iter()
        .filter(|e| e.id != a.id && e.id != b.id)
        .filter(|e| causally_precedes(a, e, c) && causally_precedes(e, b, c))
        .map(|e| e.id)
        .collect()
}

// ── CausalSet ────────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct CausalSet {
    pub events: Vec<Event>,
    /// causal_matrix[i][j] = true means event i causally precedes event j
    pub causal_matrix: Vec<Vec<bool>>,
}

impl CausalSet {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_event(&mut self, event: Event, c: f64) {
        let n = self.events.len();
        // Expand matrix: add new row and column
        for row in &mut self.causal_matrix {
            row.push(false);
        }
        self.causal_matrix.push(vec![false; n + 1]);

        // Compute causal relations with existing events
        for i in 0..n {
            let existing = &self.events[i];
            if causally_precedes(existing, &event, c) {
                self.causal_matrix[i][n] = true;
            }
            if causally_precedes(&event, existing, c) {
                self.causal_matrix[n][i] = true;
            }
        }

        self.events.push(event);
    }

    fn index_of(&self, event_id: u64) -> Option<usize> {
        self.events.iter().position(|e| e.id == event_id)
    }

    pub fn precedes(&self, a: u64, b: u64) -> bool {
        if let (Some(ai), Some(bi)) = (self.index_of(a), self.index_of(b)) {
            self.causal_matrix[ai][bi]
        } else {
            false
        }
    }

    pub fn causal_future(&self, event_id: u64) -> Vec<u64> {
        if let Some(idx) = self.index_of(event_id) {
            self.causal_matrix[idx]
                .iter()
                .enumerate()
                .filter(|(_, &v)| v)
                .map(|(i, _)| self.events[i].id)
                .collect()
        } else {
            vec![]
        }
    }

    pub fn causal_past(&self, event_id: u64) -> Vec<u64> {
        if let Some(idx) = self.index_of(event_id) {
            self.causal_matrix
                .iter()
                .enumerate()
                .filter(|(_, row)| row[idx])
                .map(|(i, _)| self.events[i].id)
                .collect()
        } else {
            vec![]
        }
    }

    pub fn causet_volume(&self) -> usize {
        self.causal_matrix
            .iter()
            .flat_map(|row| row.iter())
            .filter(|&&v| v)
            .count()
    }
}

// ── MarketCausalChain ─────────────────────────────────────────────────────

#[derive(Debug)]
pub struct MarketCausalChain {
    pub events: CausalSet,
    pub labels: HashMap<u64, String>,
}

impl MarketCausalChain {
    pub fn from_price_events(timestamps: &[f64], prices: &[f64]) -> Self {
        let c = 1.0; // information propagates at c=1
        let mut chain = MarketCausalChain {
            events: CausalSet::new(),
            labels: HashMap::new(),
        };

        for (i, (&t, &p)) in timestamps.iter().zip(prices.iter()).enumerate() {
            let id = i as u64;
            let event = Event {
                id,
                time: t,
                space: p,
                label: format!("price@{:.2}", t),
            };
            chain.labels.insert(id, format!("price={:.2}", p));
            chain.events.add_event(event, c);
        }

        chain
    }
}

/// Longest causal chain ending at event_id (causal depth).
pub fn causal_depth(chain: &MarketCausalChain, event_id: u64) -> usize {
    // BFS/DFS: find longest path ending at event_id using causal_past
    fn depth_recursive(chain: &MarketCausalChain, event_id: u64, memo: &mut HashMap<u64, usize>) -> usize {
        if let Some(&cached) = memo.get(&event_id) {
            return cached;
        }
        let past = chain.events.causal_past(event_id);
        let d = if past.is_empty() {
            0
        } else {
            past.iter()
                .map(|&pid| depth_recursive(chain, pid, memo) + 1)
                .max()
                .unwrap_or(0)
        };
        memo.insert(event_id, d);
        d
    }

    let mut memo = HashMap::new();
    depth_recursive(chain, event_id, &mut memo)
}

// ── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn causal_relation_classification() {
        let a = Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() };
        let b = Event { id: 1, time: 2.0, space: 1.0, label: "B".to_string() };
        let c_speed = 1.0;
        // s² = -(1*4) + 1 = -3 < 0 => CausallyConnected
        assert_eq!(causal_relation(&a, &b, c_speed), CausalRelation::CausallyConnected);
    }

    #[test]
    fn spacelike_classification() {
        let a = Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() };
        // dt=1, dx=5 => s² = -(1) + 25 = 24 > 0 => spacelike
        let b = Event { id: 1, time: 1.0, space: 5.0, label: "B".to_string() };
        assert_eq!(causal_relation(&a, &b, 1.0), CausalRelation::Spacelike);
    }

    #[test]
    fn lightlike_boundary() {
        let a = Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() };
        // dt=1, dx=1, c=1 => s² = -1 + 1 = 0 => Lightlike
        let b = Event { id: 1, time: 1.0, space: 1.0, label: "B".to_string() };
        assert_eq!(causal_relation(&a, &b, 1.0), CausalRelation::Lightlike);
    }

    #[test]
    fn alexandrov_interval_contains_intermediate() {
        let a = Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() };
        let mid = Event { id: 1, time: 1.0, space: 0.0, label: "Mid".to_string() };
        let b = Event { id: 2, time: 2.0, space: 0.0, label: "B".to_string() };
        let events = vec![a.clone(), mid.clone(), b.clone()];
        let interval = alexandrov_interval(&a, &b, &events, 1.0);
        assert!(interval.contains(&1));
    }

    #[test]
    fn causal_future_non_empty() {
        let mut cs = CausalSet::new();
        let c = 1.0;
        cs.add_event(Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() }, c);
        cs.add_event(Event { id: 1, time: 2.0, space: 0.5, label: "B".to_string() }, c);
        let future = cs.causal_future(0);
        assert!(!future.is_empty());
        assert!(future.contains(&1));
    }

    #[test]
    fn causal_past() {
        let mut cs = CausalSet::new();
        let c = 1.0;
        cs.add_event(Event { id: 0, time: 0.0, space: 0.0, label: "A".to_string() }, c);
        cs.add_event(Event { id: 1, time: 2.0, space: 0.5, label: "B".to_string() }, c);
        let past = cs.causal_past(1);
        assert!(past.contains(&0));
    }
}
