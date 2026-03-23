//! Causal set theory — discrete spacetime for the SRFM pipeline.
//!
//! A causal set is a locally finite partially ordered set that models
//! spacetime at the Planck scale. Each element is an event; the partial
//! order encodes the causal precedence relation.

use std::collections::VecDeque;

// ---------------------------------------------------------------------------
// Event
// ---------------------------------------------------------------------------

/// A point in a discrete causal spacetime.
#[derive(Debug, Clone)]
pub struct Event {
    /// Unique event identifier (index into the causal set's event list).
    pub id: usize,
    /// Spacetime coordinates (may be embedding coordinates or abstract labels).
    pub coordinates: Vec<f64>,
    /// Optional human-readable label.
    pub label: Option<String>,
}

impl Event {
    /// Create a new event.
    pub fn new(id: usize, coordinates: Vec<f64>, label: Option<String>) -> Self {
        Self { id, coordinates, label }
    }
}

// ---------------------------------------------------------------------------
// CausalRelation
// ---------------------------------------------------------------------------

/// The causal relationship between two events.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CausalRelation {
    /// Event i is in the causal past of event j (i ≺ j).
    Precedes,
    /// Event i is in the causal future of event j (i ≻ j).
    Succeeds,
    /// Events i and j are causally disconnected (spacelike).
    Spacelike,
}

// ---------------------------------------------------------------------------
// CausalSet
// ---------------------------------------------------------------------------

/// A finite causal set with a stored adjacency matrix.
///
/// `adjacency[i][j] == true` means event `i` directly precedes event `j`
/// (irreducible link, not necessarily transitive closure).
pub struct CausalSet {
    events: Vec<Event>,
    /// `adjacency[i][j]` — direct causal link from i to j.
    adjacency: Vec<Vec<bool>>,
}

impl CausalSet {
    /// Create an empty causal set.
    pub fn new() -> Self {
        Self {
            events: Vec::new(),
            adjacency: Vec::new(),
        }
    }

    /// Add an event; the adjacency matrix is extended accordingly.
    pub fn add_event(&mut self, event: Event) {
        let n = self.events.len();
        // Extend existing rows.
        for row in &mut self.adjacency {
            row.push(false);
        }
        // Add new row of length n+1.
        self.adjacency.push(vec![false; n + 1]);
        self.events.push(event);
    }

    /// Add a direct causal link from event `from` to event `to`.
    ///
    /// Returns `Err` if:
    /// - Either index is out of range.
    /// - The link would introduce a cycle (i.e., `to` already precedes `from`).
    pub fn add_causal_link(&mut self, from: usize, to: usize) -> Result<(), String> {
        let n = self.events.len();
        if from >= n {
            return Err(format!("Event {} does not exist (len = {})", from, n));
        }
        if to >= n {
            return Err(format!("Event {} does not exist (len = {})", to, n));
        }
        if from == to {
            return Err(format!("Self-link on event {} would create a trivial cycle", from));
        }
        // Check for cycle: if `to` already (transitively) precedes `from`, adding
        // from→to would create a cycle.
        if self.past(from).contains(&to) {
            return Err(format!(
                "Adding link {} → {} would create a cycle (event {} already precedes {})",
                from, to, to, from
            ));
        }
        self.adjacency[from][to] = true;
        Ok(())
    }

    /// Return the causal relation between events `i` and `j`.
    pub fn causal_relation(&self, i: usize, j: usize) -> CausalRelation {
        if self.past(j).contains(&i) {
            CausalRelation::Precedes
        } else if self.past(i).contains(&j) {
            CausalRelation::Succeeds
        } else {
            CausalRelation::Spacelike
        }
    }

    /// Return the Alexandrov set — all events `c` such that `from ≺ c ≺ to`.
    pub fn alexandrov_set(&self, from: usize, to: usize) -> Vec<usize> {
        let future_of_from = self.future(from);
        let past_of_to     = self.past(to);
        future_of_from.into_iter()
            .filter(|&x| x != from && x != to && past_of_to.contains(&x))
            .collect()
    }

    /// Compute the transitive past of `event_id` via BFS on reversed links.
    pub fn past(&self, event_id: usize) -> Vec<usize> {
        let n = self.events.len();
        let mut visited = vec![false; n];
        let mut queue   = VecDeque::new();
        queue.push_back(event_id);
        visited[event_id] = true;

        let mut result = Vec::new();
        while let Some(current) = queue.pop_front() {
            // Find all predecessors: j such that adjacency[j][current] == true.
            for j in 0..n {
                if !visited[j] && self.adjacency[j][current] {
                    visited[j] = true;
                    result.push(j);
                    queue.push_back(j);
                }
            }
        }
        result
    }

    /// Compute the transitive future of `event_id` via BFS on forward links.
    pub fn future(&self, event_id: usize) -> Vec<usize> {
        let n = self.events.len();
        let mut visited = vec![false; n];
        let mut queue   = VecDeque::new();
        queue.push_back(event_id);
        visited[event_id] = true;

        let mut result = Vec::new();
        while let Some(current) = queue.pop_front() {
            for j in 0..n {
                if !visited[j] && self.adjacency[current][j] {
                    visited[j] = true;
                    result.push(j);
                    queue.push_back(j);
                }
            }
        }
        result
    }

    /// Count the number of direct (irreducible) causal links.
    pub fn link_count(&self) -> usize {
        let n = self.events.len();
        let mut count = 0;
        for i in 0..n {
            for j in 0..n {
                if self.adjacency[i][j] {
                    count += 1;
                }
            }
        }
        count
    }

    /// Estimate the effective spacetime dimension using the Myrheim-Meyer relation.
    ///
    /// For a causal set of N elements embedded in a d-dimensional flat spacetime,
    /// the expected number of causal pairs scales as C ~ N^2 * xi(d), where
    /// xi(d) is a dimension-dependent constant. We invert this to estimate d.
    ///
    /// Returns a real-valued estimate (may be non-integer).
    pub fn estimate_dimension(&self) -> f64 {
        let n = self.events.len() as f64;
        if n < 2.0 {
            return 0.0;
        }
        // Count ordered causal pairs (transitive closure).
        let mut causal_pairs = 0usize;
        let len = self.events.len();
        for i in 0..len {
            for j in 0..len {
                if i != j && self.past(j).contains(&i) {
                    causal_pairs += 1;
                }
            }
        }
        let c = causal_pairs as f64 / (n * n);
        if c <= 0.0 {
            return 1.0;
        }
        // xi(d) ≈ 1/(d*(d-1)) for d ≥ 2; invert to estimate d.
        // c = 1/(d*(d-1)) → d*(d-1) = 1/c → d ≈ (1 + sqrt(1 + 4/c)) / 2
        let discriminant = 1.0 + 4.0 / c;
        (1.0 + discriminant.sqrt()) / 2.0
    }

    /// Return a maximal antichain — a set of pairwise spacelike events.
    pub fn antichain(&self) -> Vec<usize> {
        let n = self.events.len();
        let mut chain: Vec<usize> = Vec::new();

        'outer: for i in 0..n {
            for &j in &chain {
                if self.causal_relation(i, j) != CausalRelation::Spacelike {
                    continue 'outer;
                }
            }
            chain.push(i);
        }
        chain
    }

    /// Describe the causal structure in terms of market events (analogy).
    pub fn to_market_analogy(&self) -> String {
        let n = self.events.len();
        let links = self.link_count();
        let dim = self.estimate_dimension();
        format!(
            "Market causal structure: {} trading events, {} direct causal links \
             (information flows), effective dimension ≈ {:.2}. \
             Events in the past light-cone represent confirmed market signals; \
             spacelike events represent simultaneous but causally independent price moves.",
            n, links, dim
        )
    }

    /// Number of events in the set.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// True if the set contains no events.
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Access all events.
    pub fn events(&self) -> &[Event] {
        &self.events
    }
}

impl Default for CausalSet {
    fn default() -> Self {
        Self::new()
    }
}
