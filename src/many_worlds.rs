//! Many-worlds interpretation: parallel market scenario branching.

use serde::{Deserialize, Serialize};

// ─── STRUCTS ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldState {
    pub world_id: u64,
    pub probability: f64,
    pub price_vector: Vec<f64>,
    pub time_step: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BranchEvent {
    pub world_id: u64,
    pub branch_probability: f64,
    pub event_description: String,
}

// ─── LCG HELPER ──────────────────────────────────────────────────────────────

fn lcg(state: u64) -> u64 {
    state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407)
}

/// Sample a U(0,1) from an LCG state.
fn lcg_uniform(state: u64) -> (f64, u64) {
    let next = lcg(state);
    let u = (next >> 11) as f64 / (1u64 << 53) as f64;
    (u, next)
}

/// Sample a standard normal via Box-Muller (uses two LCG steps).
fn lcg_normal(state: u64) -> (f64, u64) {
    let (u1, s1) = lcg_uniform(state);
    let (u2, s2) = lcg_uniform(s1);
    let u1 = u1.max(1e-12);
    let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    (z, s2)
}

// ─── MANY WORLDS SIMULATOR ───────────────────────────────────────────────────

pub struct ManyWorldsSimulator {
    pub worlds: Vec<WorldState>,
    next_id: u64,
}

impl ManyWorldsSimulator {
    pub fn new(initial_prices: Vec<f64>, initial_prob: f64) -> Self {
        let world = WorldState {
            world_id: 0,
            probability: initial_prob,
            price_vector: initial_prices,
            time_step: 0,
        };
        ManyWorldsSimulator {
            worlds: vec![world],
            next_id: 1,
        }
    }

    /// Split a world into branches according to the provided events.
    /// Each branch gets probability = world.probability * event.branch_probability.
    pub fn branch(&mut self, world_id: u64, events: Vec<BranchEvent>) -> Vec<WorldState> {
        let parent = match self.worlds.iter().find(|w| w.world_id == world_id) {
            Some(w) => w.clone(),
            None => return Vec::new(),
        };

        let mut new_worlds: Vec<WorldState> = Vec::new();
        for event in &events {
            let branch_prob = parent.probability * event.branch_probability;
            let new_id = self.next_id;
            self.next_id += 1;
            let w = WorldState {
                world_id: new_id,
                probability: branch_prob,
                price_vector: parent.price_vector.clone(),
                time_step: parent.time_step,
            };
            new_worlds.push(w.clone());
            self.worlds.push(w);
        }
        new_worlds
    }

    /// Evolve a world by one time step using Geometric Brownian Motion.
    /// Returns a new WorldState (does not mutate in-place).
    pub fn evolve(&self, world: &WorldState, volatility: f64, dt: f64, seed: u64) -> WorldState {
        let mut state = seed;
        let mut new_prices = Vec::with_capacity(world.price_vector.len());
        for &p in &world.price_vector {
            let (z, s) = lcg_normal(state);
            state = s;
            // GBM: S(t+dt) = S(t) * exp((mu - 0.5*vol^2)*dt + vol*sqrt(dt)*z)
            // Using mu=0 (risk-neutral drift component aside from vol adjustment)
            let drift = -0.5 * volatility * volatility * dt;
            let diffusion = volatility * dt.sqrt() * z;
            let new_p = p * (drift + diffusion).exp();
            new_prices.push(new_p.max(0.0));
        }
        WorldState {
            world_id: world.world_id,
            probability: world.probability,
            price_vector: new_prices,
            time_step: world.time_step + 1,
        }
    }

    /// Sample one world according to probability weights.
    pub fn collapse<'a>(&self, worlds: &'a [WorldState], seed: u64) -> &'a WorldState {
        if worlds.is_empty() {
            panic!("Cannot collapse empty world set");
        }
        let total: f64 = worlds.iter().map(|w| w.probability).sum();
        let (u, _) = lcg_uniform(seed);
        let target = u * total;
        let mut cumulative = 0.0;
        for world in worlds {
            cumulative += world.probability;
            if cumulative >= target {
                return world;
            }
        }
        worlds.last().unwrap()
    }

    /// Probability-weighted average price vector across worlds.
    pub fn expected_value(&self, worlds: &[WorldState]) -> Vec<f64> {
        if worlds.is_empty() {
            return Vec::new();
        }
        let n = worlds[0].price_vector.len();
        let total_prob: f64 = worlds.iter().map(|w| w.probability).sum();
        let mut ev = vec![0.0f64; n];
        for world in worlds {
            let w = if total_prob > 0.0 { world.probability / total_prob } else { 0.0 };
            for (i, &p) in world.price_vector.iter().enumerate() {
                if i < n {
                    ev[i] += w * p;
                }
            }
        }
        ev
    }

    /// Variance of prices across worlds (probability-weighted).
    pub fn world_variance(&self, worlds: &[WorldState]) -> Vec<f64> {
        if worlds.is_empty() {
            return Vec::new();
        }
        let ev = self.expected_value(worlds);
        let n = ev.len();
        let total_prob: f64 = worlds.iter().map(|w| w.probability).sum();
        let mut var = vec![0.0f64; n];
        for world in worlds {
            let w = if total_prob > 0.0 { world.probability / total_prob } else { 0.0 };
            for (i, &p) in world.price_vector.iter().enumerate() {
                if i < n {
                    let diff = p - ev[i];
                    var[i] += w * diff * diff;
                }
            }
        }
        var
    }

    /// Remove worlds with probability below threshold.
    pub fn prune_low_probability(
        &self,
        worlds: Vec<WorldState>,
        threshold: f64,
    ) -> Vec<WorldState> {
        worlds.into_iter().filter(|w| w.probability >= threshold).collect()
    }
}

// ─── TESTS ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_sim() -> ManyWorldsSimulator {
        ManyWorldsSimulator::new(vec![100.0, 200.0, 50.0], 1.0)
    }

    #[test]
    fn test_new_creates_one_world() {
        let sim = make_sim();
        assert_eq!(sim.worlds.len(), 1);
        assert_eq!(sim.worlds[0].world_id, 0);
        assert!((sim.worlds[0].probability - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_branch_creates_worlds() {
        let mut sim = make_sim();
        let events = vec![
            BranchEvent { world_id: 0, branch_probability: 0.6, event_description: "Bull".into() },
            BranchEvent { world_id: 0, branch_probability: 0.4, event_description: "Bear".into() },
        ];
        let branches = sim.branch(0, events);
        assert_eq!(branches.len(), 2);
        assert!((branches[0].probability - 0.6).abs() < 1e-9);
        assert!((branches[1].probability - 0.4).abs() < 1e-9);
    }

    #[test]
    fn test_branch_unknown_world() {
        let mut sim = make_sim();
        let events = vec![
            BranchEvent { world_id: 999, branch_probability: 1.0, event_description: "X".into() },
        ];
        let branches = sim.branch(999, events);
        assert!(branches.is_empty());
    }

    #[test]
    fn test_evolve_deterministic() {
        let sim = make_sim();
        let w = &sim.worlds[0];
        let r1 = sim.evolve(w, 0.2, 1.0 / 252.0, 42);
        let r2 = sim.evolve(w, 0.2, 1.0 / 252.0, 42);
        for (a, b) in r1.price_vector.iter().zip(r2.price_vector.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_evolve_prices_positive() {
        let sim = make_sim();
        for seed in 0..20u64 {
            let r = sim.evolve(&sim.worlds[0], 0.3, 1.0 / 52.0, seed * 1234567);
            for p in &r.price_vector {
                assert!(*p >= 0.0);
            }
        }
    }

    #[test]
    fn test_evolve_time_step_increments() {
        let sim = make_sim();
        let r = sim.evolve(&sim.worlds[0], 0.1, 0.01, 1);
        assert_eq!(r.time_step, 1);
    }

    #[test]
    fn test_collapse_returns_valid_world() {
        let sim = make_sim();
        let worlds = sim.worlds.clone();
        let chosen = sim.collapse(&worlds, 99);
        assert_eq!(chosen.world_id, 0);
    }

    #[test]
    fn test_expected_value_single_world() {
        let sim = make_sim();
        let ev = sim.expected_value(&sim.worlds);
        assert!((ev[0] - 100.0).abs() < 1e-9);
        assert!((ev[1] - 200.0).abs() < 1e-9);
    }

    #[test]
    fn test_expected_value_weighted() {
        let worlds = vec![
            WorldState { world_id: 0, probability: 0.7, price_vector: vec![100.0], time_step: 0 },
            WorldState { world_id: 1, probability: 0.3, price_vector: vec![200.0], time_step: 0 },
        ];
        let sim = ManyWorldsSimulator::new(vec![], 1.0);
        let ev = sim.expected_value(&worlds);
        let expected = 0.7 * 100.0 + 0.3 * 200.0;
        assert!((ev[0] - expected).abs() < 1e-9);
    }

    #[test]
    fn test_world_variance_single_world_zero() {
        let sim = make_sim();
        let var = sim.world_variance(&sim.worlds);
        for v in &var {
            assert!(v.abs() < 1e-9);
        }
    }

    #[test]
    fn test_prune_removes_low_prob() {
        let worlds = vec![
            WorldState { world_id: 0, probability: 0.5, price_vector: vec![100.0], time_step: 0 },
            WorldState { world_id: 1, probability: 0.001, price_vector: vec![50.0], time_step: 0 },
            WorldState { world_id: 2, probability: 0.499, price_vector: vec![120.0], time_step: 0 },
        ];
        let sim = ManyWorldsSimulator::new(vec![], 1.0);
        let pruned = sim.prune_low_probability(worlds, 0.01);
        assert_eq!(pruned.len(), 2);
    }

    #[test]
    fn test_expected_value_empty() {
        let sim = make_sim();
        let ev = sim.expected_value(&[]);
        assert!(ev.is_empty());
    }
}
