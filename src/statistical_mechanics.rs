/// A point in phase space.
#[derive(Debug, Clone)]
pub struct PhasePoint {
    pub position: Vec<f64>,
    pub momentum: Vec<f64>,
}

impl PhasePoint {
    pub fn norm(&self) -> f64 {
        let sum_sq: f64 = self
            .position
            .iter()
            .chain(self.momentum.iter())
            .map(|x| x * x)
            .sum();
        sum_sq.sqrt()
    }
}

/// A generic Hamiltonian system on a phase space of dimension `dim`.
pub struct HamiltonianSystem {
    pub positions: Vec<f64>,
    pub momenta: Vec<f64>,
    pub dim: usize,
}

fn lcg_f64(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    let bits = *state >> 11;
    bits as f64 / (u64::MAX >> 11) as f64 // [0, 1)
}

impl HamiltonianSystem {
    /// Initialise with random positions and momenta in [-1, 1].
    pub fn new(dim: usize, seed: u64) -> Self {
        let mut state = seed;
        let positions: Vec<f64> = (0..dim).map(|_| lcg_f64(&mut state) * 2.0 - 1.0).collect();
        let momenta: Vec<f64> = (0..dim).map(|_| lcg_f64(&mut state) * 2.0 - 1.0).collect();
        Self { positions, momenta, dim }
    }

    /// Symplectic Euler step: p_new = p - dt*grad_V(q); q_new = q + dt*p_new.
    pub fn step(&mut self, dt: f64, potential_gradient: &dyn Fn(&[f64]) -> Vec<f64>) {
        let grad = potential_gradient(&self.positions);
        for i in 0..self.dim {
            self.momenta[i] -= dt * grad[i];
        }
        for i in 0..self.dim {
            self.positions[i] += dt * self.momenta[i];
        }
    }
}

/// Lyapunov exponent estimate from a 1D trajectory.
/// Uses average of log|x_{n+1} - x_n| / dt.
pub fn lyapunov_exponent(trajectory: &[f64], dt: f64) -> f64 {
    if trajectory.len() < 2 || dt == 0.0 {
        return 0.0;
    }
    let n = trajectory.len() - 1;
    let sum: f64 = (0..n)
        .map(|i| {
            let diff = (trajectory[i + 1] - trajectory[i]).abs();
            if diff > 0.0 {
                diff.ln() / dt
            } else {
                f64::NEG_INFINITY
            }
        })
        .filter(|v| v.is_finite())
        .sum();
    let count = (0..n)
        .filter(|&i| (trajectory[i + 1] - trajectory[i]).abs() > 0.0)
        .count();
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// Check approximate ergodicity by comparing time-average to spatial uniformity.
/// Returns true if std(bin_counts) / mean(bin_counts) < 0.5.
pub fn is_ergodic_approx(trajectory: &[f64], n_bins: usize) -> bool {
    if trajectory.is_empty() || n_bins == 0 {
        return false;
    }
    let min = trajectory.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = trajectory.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-15 {
        return true; // all points identical — trivially uniform
    }
    let mut counts = vec![0usize; n_bins];
    for &x in trajectory {
        let bin = ((x - min) / (max - min) * n_bins as f64) as usize;
        let bin = bin.min(n_bins - 1);
        counts[bin] += 1;
    }
    let mean = counts.iter().sum::<usize>() as f64 / n_bins as f64;
    let std_dev = (counts
        .iter()
        .map(|&c| (c as f64 - mean).powi(2))
        .sum::<f64>()
        / n_bins as f64)
        .sqrt();
    if mean < 1e-15 {
        return true;
    }
    std_dev / mean < 0.5
}

/// Pearson autocorrelation between a series and its lag-k copy.
pub fn autocorrelation(series: &[f64], lag: usize) -> f64 {
    let n = series.len();
    if lag >= n || n == 0 {
        return 0.0;
    }
    let m = n - lag;
    let mean_x: f64 = series[..m].iter().sum::<f64>() / m as f64;
    let mean_y: f64 = series[lag..].iter().sum::<f64>() / m as f64;

    let num: f64 = (0..m).map(|i| (series[i] - mean_x) * (series[i + lag] - mean_y)).sum();
    let den_x: f64 = (0..m).map(|i| (series[i] - mean_x).powi(2)).sum::<f64>().sqrt();
    let den_y: f64 = (0..m).map(|i| (series[i + lag] - mean_y).powi(2)).sum::<f64>().sqrt();

    let denom = den_x * den_y;
    if denom < 1e-15 { 0.0 } else { num / denom }
}

/// First lag where autocorrelation drops below 1/e.
pub fn mixing_time(series: &[f64]) -> usize {
    let threshold = 1.0_f64 / std::f64::consts::E;
    for lag in 1..series.len() {
        if autocorrelation(series, lag).abs() < threshold {
            return lag;
        }
    }
    series.len()
}

/// Recurrence analysis of a time series.
pub struct RecurrenceAnalysis {
    pub series: Vec<f64>,
    pub threshold: f64,
}

impl RecurrenceAnalysis {
    pub fn new(series: Vec<f64>, threshold: f64) -> Self {
        Self { series, threshold }
    }

    /// Fraction of (i, j) pairs where |s_i - s_j| < threshold.
    pub fn recurrence_rate(&self) -> f64 {
        let n = self.series.len();
        if n == 0 {
            return 0.0;
        }
        let mut count = 0usize;
        for i in 0..n {
            for j in 0..n {
                if (self.series[i] - self.series[j]).abs() < self.threshold {
                    count += 1;
                }
            }
        }
        count as f64 / (n * n) as f64
    }

    /// Fraction of recurrent points forming diagonal lines of length >= 2.
    pub fn determinism_rate(&self) -> f64 {
        let n = self.series.len();
        if n < 2 {
            return 0.0;
        }
        // Build recurrence matrix
        let rec: Vec<Vec<bool>> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| (self.series[i] - self.series[j]).abs() < self.threshold)
                    .collect()
            })
            .collect();

        let total_recurrent: usize = rec.iter().flatten().filter(|&&b| b).count();
        if total_recurrent == 0 {
            return 0.0;
        }

        // Count points on diagonal lines of length >= 2
        let mut in_diag = 0usize;
        for start_col in 0..n {
            let mut len = 0usize;
            let mut col_pts = 0usize;
            for k in 0..(n - start_col) {
                if rec[k][k + start_col] {
                    len += 1;
                } else {
                    if len >= 2 {
                        col_pts += len;
                    }
                    len = 0;
                }
            }
            if len >= 2 {
                col_pts += len;
            }
            in_diag += col_pts;
        }
        // Count off-diagonal (negative diagonals)
        for start_row in 1..n {
            let mut len = 0usize;
            let mut row_pts = 0usize;
            for k in 0..(n - start_row) {
                if rec[k + start_row][k] {
                    len += 1;
                } else {
                    if len >= 2 {
                        row_pts += len;
                    }
                    len = 0;
                }
            }
            if len >= 2 {
                row_pts += len;
            }
            in_diag += row_pts;
        }

        in_diag as f64 / total_recurrent as f64
    }
}

/// Ergodicity analysis for market return series.
pub struct MarketErgodicity {
    pub returns: Vec<f64>,
}

impl MarketErgodicity {
    pub fn new(returns: Vec<f64>) -> Self {
        Self { returns }
    }

    /// Time-average of the return series.
    pub fn time_average(&self) -> f64 {
        if self.returns.is_empty() {
            return 0.0;
        }
        self.returns.iter().sum::<f64>() / self.returns.len() as f64
    }

    /// Ensemble average: approximate as mean of n_paths normal samples ~ N(0, sigma²).
    /// For log-returns E[r] ≈ 0.
    pub fn ensemble_average(sigma: f64, n_paths: usize, seed: u64) -> f64 {
        if n_paths == 0 {
            return 0.0;
        }
        let mut state = seed;
        // Box-Muller transform using LCG
        let mut sum = 0.0f64;
        let mut count = 0usize;
        for _ in 0..n_paths {
            let u1 = lcg_f64(&mut state).max(1e-15);
            let u2 = lcg_f64(&mut state);
            let z = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
            sum += z * sigma;
            count += 1;
        }
        sum / count as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lyapunov_on_growing_sequence_positive() {
        // Exponentially growing: x_n = e^n => differences grow
        let traj: Vec<f64> = (0..20).map(|i| (i as f64).exp()).collect();
        let lyap = lyapunov_exponent(&traj, 1.0);
        assert!(lyap > 0.0, "Expected positive Lyapunov exponent, got {}", lyap);
    }

    #[test]
    fn test_autocorrelation_lag0_is_1() {
        let series: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ac = autocorrelation(&series, 0);
        assert!((ac - 1.0).abs() < 1e-10, "autocorrelation at lag 0 should be 1, got {}", ac);
    }

    #[test]
    fn test_mixing_time_finite() {
        // White noise-ish series should decorrelate quickly
        let mut state = 12345u64;
        let series: Vec<f64> = (0..200)
            .map(|_| {
                state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (state >> 11) as f64 / (u64::MAX >> 11) as f64
            })
            .collect();
        let mt = mixing_time(&series);
        assert!(mt < series.len(), "mixing_time should be finite");
    }

    #[test]
    fn test_recurrence_rate_in_0_1() {
        let series: Vec<f64> = (0..10).map(|i| i as f64).collect();
        let ra = RecurrenceAnalysis::new(series, 1.5);
        let rr = ra.recurrence_rate();
        assert!((0.0..=1.0).contains(&rr), "recurrence_rate out of [0,1]: {}", rr);
    }

    #[test]
    fn test_ergodic_bins_uniform() {
        // Uniform coverage => should be ergodic
        let series: Vec<f64> = (0..1000).map(|i| (i % 10) as f64).collect();
        assert!(is_ergodic_approx(&series, 10));
    }

    #[test]
    fn test_phase_point_norm() {
        let p = PhasePoint {
            position: vec![3.0, 0.0],
            momentum: vec![4.0, 0.0],
        };
        assert!((p.norm() - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_hamiltonian_step_runs() {
        let mut sys = HamiltonianSystem::new(3, 42);
        let grad = |q: &[f64]| q.to_vec(); // V = sum(q²/2), grad = q
        sys.step(0.01, &grad);
        assert_eq!(sys.dim, 3);
    }

    #[test]
    fn test_determinism_rate_in_0_1() {
        let series = vec![1.0, 1.0, 2.0, 2.0, 1.0, 1.0];
        let ra = RecurrenceAnalysis::new(series, 0.5);
        let dr = ra.determinism_rate();
        assert!((0.0..=1.0).contains(&dr));
    }

    #[test]
    fn test_market_ergodicity_time_average() {
        let me = MarketErgodicity::new(vec![1.0, 2.0, 3.0]);
        assert!((me.time_average() - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_ensemble_average_near_zero() {
        // With many paths, ensemble average of log-returns should be near 0
        let avg = MarketErgodicity::ensemble_average(0.01, 10000, 777);
        assert!(avg.abs() < 0.005, "ensemble average too far from 0: {}", avg);
    }
}
