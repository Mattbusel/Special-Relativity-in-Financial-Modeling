//! Wormhole traversal analogy: instantaneous arbitrage paths in curved market
//! space.
//!
//! A wormhole connects two otherwise distant market regions, allowing
//! near-instantaneous arbitrage. The cost of traversal maps to:
//!   - time dilation  → execution latency drag
//!   - energy cost    → transaction fees
//!   - exotic matter  → required short-selling capacity

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// Wormhole
// ---------------------------------------------------------------------------

/// Represents a shortcut between two market regions.
///
/// * `throat_radius`  – Minimum cross-section; smaller = harder to pass orders.
/// * `length`         – Effective path length through the wormhole.
/// * `traversability` – 0.0 (impassable) to 1.0 (frictionless).
#[derive(Debug, Clone)]
pub struct Wormhole {
    pub throat_radius: f64,
    pub length: f64,
    pub traversability: f64,
}

impl Wormhole {
    pub fn new(throat_radius: f64, length: f64, traversability: f64) -> Self {
        Wormhole {
            throat_radius,
            length,
            traversability: traversability.clamp(0.0, 1.0),
        }
    }
}

// ---------------------------------------------------------------------------
// TraversalCost
// ---------------------------------------------------------------------------

/// Cost components of traversing a wormhole.
#[derive(Debug, Clone)]
pub struct TraversalCost {
    /// Fraction of signal speed lost to time dilation (0 = none, 1 = total loss).
    pub time_dilation: f64,
    /// Energy (fee) cost in arbitrary units.
    pub energy_cost: f64,
    /// Exotic matter required — proxy for short-selling capacity needed.
    pub exotic_matter_required: f64,
}

// ---------------------------------------------------------------------------
// compute_traversal_cost
// ---------------------------------------------------------------------------

/// Compute the traversal cost through a wormhole at a given signal speed.
///
/// * `time_dilation`         ∝ length / (traversability * signal_speed)
/// * `energy_cost`           ∝ length / throat_radius²
/// * `exotic_matter_required`∝ (1 - traversability) * length
pub fn compute_traversal_cost(wormhole: &Wormhole, signal_speed: f64) -> TraversalCost {
    let trav = wormhole.traversability.max(1e-9);
    let eff_speed = (signal_speed * trav).max(1e-12);

    let time_dilation = (wormhole.length / eff_speed).min(1.0);
    let energy_cost = wormhole.length / (wormhole.throat_radius * wormhole.throat_radius).max(1e-12);
    let exotic_matter_required = (1.0 - trav) * wormhole.length;

    TraversalCost {
        time_dilation,
        energy_cost,
        exotic_matter_required,
    }
}

// ---------------------------------------------------------------------------
// ArbitragePath
// ---------------------------------------------------------------------------

/// A sequence of markets connected by wormhole-style arbitrage links.
#[derive(Debug, Clone)]
pub struct ArbitragePath {
    /// Ordered list of market identifiers.
    pub markets: Vec<String>,
    /// Price differential at each hop (positive = profit opportunity).
    pub price_differentials: Vec<f64>,
    /// Traversal cost at each hop.
    pub traversal_costs: Vec<f64>,
}

// ---------------------------------------------------------------------------
// ArbitrageFinder
// ---------------------------------------------------------------------------

/// Identifies wormhole-style arbitrage paths across markets.
pub struct ArbitrageFinder;

impl ArbitrageFinder {
    /// Find all market pairs where the price differential exceeds `threshold`.
    ///
    /// `markets` maps market_id → current price.
    pub fn find_wormhole_paths(
        markets: &HashMap<String, f64>,
        threshold: f64,
    ) -> Vec<ArbitragePath> {
        let mut paths = Vec::new();
        let keys: Vec<&String> = markets.keys().collect();

        for i in 0..keys.len() {
            for j in (i + 1)..keys.len() {
                let price_a = markets[keys[i]];
                let price_b = markets[keys[j]];
                let diff = (price_b - price_a).abs();

                if diff > threshold {
                    // Model traversal cost proportional to distance (price gap).
                    let wormhole = Wormhole::new(
                        1.0,             // unit throat
                        diff / 100.0,    // length proportional to gap
                        0.8,             // 80% traversable by default
                    );
                    let cost = compute_traversal_cost(&wormhole, 1.0);
                    let cost_total = cost.energy_cost + cost.time_dilation * diff * 0.01;

                    paths.push(ArbitragePath {
                        markets: vec![keys[i].clone(), keys[j].clone()],
                        price_differentials: vec![diff],
                        traversal_costs: vec![cost_total],
                    });
                }
            }
        }

        paths
    }

    /// Net profit of a path: sum(price_differentials) - sum(traversal_costs).
    pub fn net_profit(path: &ArbitragePath) -> f64 {
        let gain: f64 = path.price_differentials.iter().sum();
        let cost: f64 = path.traversal_costs.iter().sum();
        gain - cost
    }

    /// Return the path with the highest net profit, or `None` if empty.
    pub fn optimal_path<'a>(paths: &'a [ArbitragePath]) -> Option<&'a ArbitragePath> {
        paths
            .iter()
            .max_by(|a, b| {
                Self::net_profit(a)
                    .partial_cmp(&Self::net_profit(b))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_wormhole() -> Wormhole {
        Wormhole::new(1.0, 10.0, 0.9)
    }

    #[test]
    fn test_traversal_cost_positive_energy() {
        let w = simple_wormhole();
        let cost = compute_traversal_cost(&w, 1.0);
        assert!(cost.energy_cost > 0.0);
    }

    #[test]
    fn test_traversal_cost_time_dilation_clamped_to_one() {
        // Very slow signal → time dilation should be clamped to 1.
        let w = Wormhole::new(1.0, 1000.0, 0.001);
        let cost = compute_traversal_cost(&w, 0.0001);
        assert!(cost.time_dilation <= 1.0);
    }

    #[test]
    fn test_traversal_cost_exotic_matter_zero_for_fully_traversable() {
        let w = Wormhole::new(1.0, 10.0, 1.0);
        let cost = compute_traversal_cost(&w, 1.0);
        assert!(cost.exotic_matter_required.abs() < 1e-10);
    }

    #[test]
    fn test_traversal_cost_exotic_matter_nonzero_for_partial() {
        let w = Wormhole::new(1.0, 10.0, 0.5);
        let cost = compute_traversal_cost(&w, 1.0);
        assert!(cost.exotic_matter_required > 0.0);
    }

    #[test]
    fn test_traversability_clamped_to_0_1() {
        let w = Wormhole::new(1.0, 1.0, 1.5);
        assert!((w.traversability - 1.0).abs() < 1e-10);
        let w2 = Wormhole::new(1.0, 1.0, -0.5);
        assert!(w2.traversability.abs() < 1e-10);
    }

    #[test]
    fn test_find_wormhole_paths_detects_large_differentials() {
        let mut markets = HashMap::new();
        markets.insert("NYSE".to_string(), 100.0);
        markets.insert("LSE".to_string(), 150.0);
        markets.insert("TSE".to_string(), 105.0);

        let paths = ArbitrageFinder::find_wormhole_paths(&markets, 10.0);
        // NYSE<->LSE diff=50, NYSE<->TSE diff=5 (below threshold), LSE<->TSE diff=45
        assert_eq!(paths.len(), 2, "paths={:?}", paths.iter().map(|p| &p.markets).collect::<Vec<_>>());
    }

    #[test]
    fn test_net_profit_basic() {
        let path = ArbitragePath {
            markets: vec!["A".into(), "B".into()],
            price_differentials: vec![50.0],
            traversal_costs: vec![5.0],
        };
        assert!((ArbitrageFinder::net_profit(&path) - 45.0).abs() < 1e-10);
    }

    #[test]
    fn test_net_profit_multi_hop() {
        let path = ArbitragePath {
            markets: vec!["A".into(), "B".into(), "C".into()],
            price_differentials: vec![30.0, 20.0],
            traversal_costs: vec![5.0, 3.0],
        };
        assert!((ArbitrageFinder::net_profit(&path) - 42.0).abs() < 1e-10);
    }

    #[test]
    fn test_optimal_path_picks_highest_profit() {
        let paths = vec![
            ArbitragePath {
                markets: vec!["A".into(), "B".into()],
                price_differentials: vec![10.0],
                traversal_costs: vec![2.0],
            },
            ArbitragePath {
                markets: vec!["C".into(), "D".into()],
                price_differentials: vec![30.0],
                traversal_costs: vec![2.0],
            },
        ];
        let opt = ArbitrageFinder::optimal_path(&paths).unwrap();
        assert_eq!(opt.markets[0], "C");
    }

    #[test]
    fn test_optimal_path_empty_returns_none() {
        let paths: Vec<ArbitragePath> = Vec::new();
        assert!(ArbitrageFinder::optimal_path(&paths).is_none());
    }

    #[test]
    fn test_find_wormhole_paths_no_paths_when_prices_equal() {
        let mut markets = HashMap::new();
        markets.insert("A".to_string(), 100.0);
        markets.insert("B".to_string(), 100.0);
        let paths = ArbitrageFinder::find_wormhole_paths(&markets, 1.0);
        assert!(paths.is_empty());
    }

    #[test]
    fn test_traversal_cost_higher_throat_lowers_energy() {
        let wide = Wormhole::new(10.0, 10.0, 0.8);
        let narrow = Wormhole::new(1.0, 10.0, 0.8);
        let cost_wide   = compute_traversal_cost(&wide, 1.0);
        let cost_narrow = compute_traversal_cost(&narrow, 1.0);
        assert!(cost_wide.energy_cost < cost_narrow.energy_cost);
    }
}
