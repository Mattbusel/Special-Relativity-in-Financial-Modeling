//! Feynman path integral analogy for market pricing.
//!
//! Market paths are treated as trajectories in price-space, weighted by
//! a Boltzmann factor exp(-S/T) where S is the Wiener action (sum of
//! squared log-returns) and T is a temperature-like parameter.

/// A single simulated market price path.
#[derive(Debug, Clone)]
pub struct MarketPath {
    /// Price at each time step.
    pub prices: Vec<f64>,
    /// Wall-clock timestamp (milliseconds since epoch) for each step.
    pub timestamps: Vec<u64>,
    /// Wiener action S = Σ (ln p[i+1]/p[i])^2.
    pub action: f64,
}

/// Functions for computing path actions and Boltzmann weights.
pub struct PathAction;

impl PathAction {
    /// Wiener action: Σ (ln(p[i+1] / p[i]))^2 over consecutive price pairs.
    pub fn wiener_action(path: &[f64]) -> f64 {
        if path.len() < 2 {
            return 0.0;
        }
        path.windows(2)
            .map(|w| {
                let log_ret = (w[1] / w[0]).ln();
                log_ret * log_ret
            })
            .sum()
    }

    /// Boltzmann weight: exp(-S / T).
    pub fn weight(action: f64, temperature: f64) -> f64 {
        if temperature <= 0.0 {
            return 0.0;
        }
        (-action / temperature).exp()
    }
}

/// Feynman path integral pricer using Monte Carlo paths.
pub struct PathIntegralPricer;

/// Simple multiplicative LCG — returns a pseudo-uniform f64 in [0, 1).
fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    ((*state >> 11) as f64) / (1u64 << 53) as f64
}

/// Box-Muller transform: produce a standard-normal sample.
fn box_muller(state: &mut u64) -> f64 {
    let u1 = lcg_f64(state).max(1e-300);
    let u2 = lcg_f64(state);
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

impl PathIntegralPricer {
    /// Generate GBM price paths using Euler-Maruyama discretization.
    ///
    /// # Parameters
    /// * `s0`      – Initial price.
    /// * `sigma`   – Annual volatility.
    /// * `dt`      – Time step (in years).
    /// * `steps`   – Number of steps per path.
    /// * `n_paths` – Number of paths to generate.
    /// * `seed`    – RNG seed for reproducibility.
    pub fn generate_paths(
        s0: f64,
        sigma: f64,
        dt: f64,
        steps: usize,
        n_paths: usize,
        seed: u64,
    ) -> Vec<MarketPath> {
        let mut rng = seed;
        let mut paths = Vec::with_capacity(n_paths);

        for _ in 0..n_paths {
            let mut prices = Vec::with_capacity(steps + 1);
            let mut timestamps = Vec::with_capacity(steps + 1);
            prices.push(s0);
            timestamps.push(0u64);

            let mut s = s0;
            for step in 1..=steps {
                let z = box_muller(&mut rng);
                // GBM: S_{t+dt} = S_t * exp((-sigma^2/2)*dt + sigma*sqrt(dt)*Z)
                s *= ((-sigma * sigma / 2.0) * dt + sigma * dt.sqrt() * z).exp();
                prices.push(s);
                timestamps.push((step as u64) * (dt * 1_000_000.0) as u64);
            }

            let action = PathAction::wiener_action(&prices);
            paths.push(MarketPath {
                prices,
                timestamps,
                action,
            });
        }
        paths
    }

    /// Price a European call using path-integral (importance-sampled) pricing.
    ///
    /// Weighted average of max(S_T - K, 0) using Boltzmann weights.
    pub fn price_european_call(paths: &[MarketPath], strike: f64, temperature: f64) -> f64 {
        if paths.is_empty() {
            return 0.0;
        }
        let z = Self::partition_function(paths, temperature);
        if z <= 0.0 {
            return 0.0;
        }
        paths.iter().map(|p| {
            let s_t = p.prices.last().copied().unwrap_or(0.0);
            let payoff = (s_t - strike).max(0.0);
            let w = PathAction::weight(p.action, temperature);
            payoff * w
        }).sum::<f64>() / z
    }

    /// Price a European put using path-integral pricing.
    pub fn price_european_put(paths: &[MarketPath], strike: f64, temperature: f64) -> f64 {
        if paths.is_empty() {
            return 0.0;
        }
        let z = Self::partition_function(paths, temperature);
        if z <= 0.0 {
            return 0.0;
        }
        paths.iter().map(|p| {
            let s_t = p.prices.last().copied().unwrap_or(0.0);
            let payoff = (strike - s_t).max(0.0);
            let w = PathAction::weight(p.action, temperature);
            payoff * w
        }).sum::<f64>() / z
    }

    /// Partition function Z = Σ exp(-S_i / T).
    pub fn partition_function(paths: &[MarketPath], temperature: f64) -> f64 {
        paths.iter().map(|p| PathAction::weight(p.action, temperature)).sum()
    }

    /// Shannon entropy of the normalized Boltzmann weight distribution.
    ///
    /// H = -Σ p_i ln(p_i) where p_i = w_i / Z.
    pub fn entropy_of_paths(paths: &[MarketPath], temperature: f64) -> f64 {
        let z = Self::partition_function(paths, temperature);
        if z <= 0.0 {
            return 0.0;
        }
        paths.iter().map(|p| {
            let prob = PathAction::weight(p.action, temperature) / z;
            if prob > 0.0 {
                -prob * prob.ln()
            } else {
                0.0
            }
        }).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wiener_action_empty() {
        assert_eq!(PathAction::wiener_action(&[]), 0.0);
    }

    #[test]
    fn test_wiener_action_single() {
        assert_eq!(PathAction::wiener_action(&[100.0]), 0.0);
    }

    #[test]
    fn test_wiener_action_flat_path() {
        let prices = vec![100.0, 100.0, 100.0, 100.0];
        assert!((PathAction::wiener_action(&prices)).abs() < 1e-12);
    }

    #[test]
    fn test_wiener_action_nonnegative() {
        let prices = vec![100.0, 101.0, 99.0, 102.0, 98.0];
        assert!(PathAction::wiener_action(&prices) >= 0.0);
    }

    #[test]
    fn test_boltzmann_weight_positive() {
        let w = PathAction::weight(1.0, 1.0);
        assert!((w - (-1.0f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn test_boltzmann_weight_zero_temp() {
        assert_eq!(PathAction::weight(1.0, 0.0), 0.0);
    }

    #[test]
    fn test_generate_paths_count() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 10, 50, 42);
        assert_eq!(paths.len(), 50);
    }

    #[test]
    fn test_generate_paths_length() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 20, 5, 7);
        assert_eq!(paths[0].prices.len(), 21); // steps + 1
    }

    #[test]
    fn test_generate_paths_positive_prices() {
        let paths = PathIntegralPricer::generate_paths(50.0, 0.3, 1.0 / 252.0, 30, 20, 99);
        for p in &paths {
            for &price in &p.prices {
                assert!(price > 0.0, "negative price: {price}");
            }
        }
    }

    #[test]
    fn test_partition_function_positive() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 10, 20, 1);
        let z = PathIntegralPricer::partition_function(&paths, 1.0);
        assert!(z > 0.0);
    }

    #[test]
    fn test_call_price_nonnegative() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 252, 100, 5);
        let price = PathIntegralPricer::price_european_call(&paths, 100.0, 1.0);
        assert!(price >= 0.0);
    }

    #[test]
    fn test_put_price_nonnegative() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 252, 100, 5);
        let price = PathIntegralPricer::price_european_put(&paths, 100.0, 1.0);
        assert!(price >= 0.0);
    }

    #[test]
    fn test_entropy_nonnegative() {
        let paths = PathIntegralPricer::generate_paths(100.0, 0.2, 1.0 / 252.0, 10, 50, 13);
        let ent = PathIntegralPricer::entropy_of_paths(&paths, 1.0);
        assert!(ent >= 0.0);
    }

    #[test]
    fn test_empty_paths_price_zero() {
        assert_eq!(PathIntegralPricer::price_european_call(&[], 100.0, 1.0), 0.0);
        assert_eq!(PathIntegralPricer::price_european_put(&[], 100.0, 1.0), 0.0);
    }

    #[test]
    fn test_deep_itm_call_positive() {
        // Very low strike means nearly all paths have positive payoff.
        let paths = PathIntegralPricer::generate_paths(100.0, 0.05, 1.0 / 252.0, 252, 200, 77);
        let price = PathIntegralPricer::price_european_call(&paths, 1.0, 1.0);
        assert!(price > 0.0);
    }
}
