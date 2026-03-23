//! String theory analogy for market vibrational modes.
//!
//! Market price series are treated as vibrating strings.  The DFT decomposes
//! them into vibrational modes, each with a frequency, amplitude, and mode
//! number — directly mirroring the harmonic excitations of an open string.

use std::f64::consts::PI;

/// A single vibrational mode of a market "string".
#[derive(Debug, Clone)]
pub struct StringMode {
    /// Oscillation frequency of this mode.
    pub frequency: f64,
    /// Amplitude of this mode.
    pub amplitude: f64,
    /// Mode number (1 = fundamental, 2 = first harmonic, …).
    pub mode_number: u32,
}

/// A vibrating market string defined by tension and length.
#[derive(Debug, Clone)]
pub struct MarketString {
    /// String tension parameter.
    pub tension: f64,
    /// String length parameter.
    pub length: f64,
    /// Active vibrational modes.
    pub modes: Vec<StringMode>,
}

impl MarketString {
    /// Fundamental frequency: `sqrt(tension / length) / 2`.
    pub fn fundamental_frequency(&self) -> f64 {
        if self.length <= 0.0 {
            return 0.0;
        }
        (self.tension / self.length).sqrt() / 2.0
    }

    /// Return the frequencies of the first `n_modes` modes (n * fundamental).
    pub fn mode_frequencies(&self, n_modes: u32) -> Vec<f64> {
        let f0 = self.fundamental_frequency();
        (1..=n_modes).map(|n| n as f64 * f0).collect()
    }

    /// Total energy: `sum(mode_number * amplitude^2 * frequency)`.
    pub fn energy(&self) -> f64 {
        self.modes
            .iter()
            .map(|m| m.mode_number as f64 * m.amplitude.powi(2) * m.frequency)
            .sum()
    }
}

/// Decomposes price series into string modes and reconstructs them.
#[derive(Debug, Default)]
pub struct StringSpectrum;

impl StringSpectrum {
    /// Create a new spectrum analyser.
    pub fn new() -> Self {
        StringSpectrum
    }

    /// Decompose `price_series` into the top `n_modes` DFT modes by amplitude.
    ///
    /// Computes the real DFT, ranks bins by amplitude, and returns the top
    /// `n_modes` as `StringMode` values.
    pub fn decompose(&self, price_series: &[f64], n_modes: u32) -> Vec<StringMode> {
        let n = price_series.len();
        if n == 0 || n_modes == 0 {
            return vec![];
        }

        // Compute DFT magnitudes for positive-frequency bins.
        let half = n / 2 + 1;
        let mut bins: Vec<(usize, f64)> = (0..half)
            .map(|k| {
                let (re, im): (f64, f64) = price_series
                    .iter()
                    .enumerate()
                    .map(|(t, &x)| {
                        let angle = 2.0 * PI * k as f64 * t as f64 / n as f64;
                        (x * angle.cos(), -x * angle.sin())
                    })
                    .fold((0.0, 0.0), |(ar, ai), (r, i)| (ar + r, ai + i));
                let amp = (re * re + im * im).sqrt() / n as f64;
                (k, amp)
            })
            .collect();

        // Sort by amplitude descending to find top modes.
        bins.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        bins.iter()
            .take(n_modes as usize)
            .enumerate()
            .map(|(rank, &(k, amp))| StringMode {
                frequency: k as f64 / n as f64,
                amplitude: amp,
                mode_number: (rank + 1) as u32,
            })
            .collect()
    }

    /// Reconstruct a price series from a list of modes using additive sinusoids.
    ///
    /// Each mode contributes `amplitude * cos(2π * frequency * t)`.
    pub fn reconstruct(&self, modes: &[StringMode], length: usize) -> Vec<f64> {
        (0..length)
            .map(|t| {
                modes
                    .iter()
                    .map(|m| m.amplitude * (2.0 * PI * m.frequency * t as f64).cos())
                    .sum()
            })
            .collect()
    }

    /// Spectral gap: frequency difference between the first and second mode
    /// (sorted by mode_number).  Returns 0.0 if fewer than two modes exist.
    pub fn spectral_gap(&self, modes: &[StringMode]) -> f64 {
        let mut sorted: Vec<&StringMode> = modes.iter().collect();
        sorted.sort_by_key(|m| m.mode_number);
        if sorted.len() < 2 {
            return 0.0;
        }
        (sorted[1].frequency - sorted[0].frequency).abs()
    }

    /// Open-string mass level: `m^2 = (n - 1) / alpha_prime`.
    pub fn mass_level(&self, mode_number: u32, alpha_prime: f64) -> f64 {
        if alpha_prime.abs() < 1e-12 {
            return f64::INFINITY;
        }
        (mode_number as f64 - 1.0) / alpha_prime
    }
}

// ---------------------------------------------------------------------------
// Unit Tests
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fundamental_frequency() {
        let s = MarketString {
            tension: 4.0,
            length: 1.0,
            modes: vec![],
        };
        // sqrt(4/1) / 2 = 1.0
        assert!((s.fundamental_frequency() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_mode_frequencies_count() {
        let s = MarketString { tension: 1.0, length: 1.0, modes: vec![] };
        let freqs = s.mode_frequencies(4);
        assert_eq!(freqs.len(), 4);
    }

    #[test]
    fn test_mode_frequencies_harmonics() {
        let s = MarketString { tension: 1.0, length: 1.0, modes: vec![] };
        let f0 = s.fundamental_frequency();
        let freqs = s.mode_frequencies(3);
        assert!((freqs[0] - f0).abs() < 1e-9);
        assert!((freqs[1] - 2.0 * f0).abs() < 1e-9);
        assert!((freqs[2] - 3.0 * f0).abs() < 1e-9);
    }

    #[test]
    fn test_energy_sum() {
        let modes = vec![
            StringMode { frequency: 1.0, amplitude: 2.0, mode_number: 1 },
            StringMode { frequency: 2.0, amplitude: 1.0, mode_number: 2 },
        ];
        let s = MarketString { tension: 1.0, length: 1.0, modes };
        // 1 * 4 * 1 + 2 * 1 * 2 = 4 + 4 = 8
        assert!((s.energy() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn test_decompose_returns_n_modes() {
        let sp = StringSpectrum::new();
        let series: Vec<f64> = (0..32).map(|i| (i as f64).sin()).collect();
        let modes = sp.decompose(&series, 4);
        assert_eq!(modes.len(), 4);
    }

    #[test]
    fn test_decompose_amplitudes_non_negative() {
        let sp = StringSpectrum::new();
        let series: Vec<f64> = (0..64).map(|i| (i as f64 * 0.1).cos()).collect();
        let modes = sp.decompose(&series, 5);
        for m in &modes {
            assert!(m.amplitude >= 0.0);
        }
    }

    #[test]
    fn test_reconstruct_correct_length() {
        let sp = StringSpectrum::new();
        let modes = vec![
            StringMode { frequency: 0.1, amplitude: 1.0, mode_number: 1 },
            StringMode { frequency: 0.2, amplitude: 0.5, mode_number: 2 },
        ];
        let rec = sp.reconstruct(&modes, 50);
        assert_eq!(rec.len(), 50);
    }

    #[test]
    fn test_reconstruct_empty_modes_is_zeros() {
        let sp = StringSpectrum::new();
        let rec = sp.reconstruct(&[], 10);
        assert!(rec.iter().all(|&x| x.abs() < 1e-9));
    }

    #[test]
    fn test_spectral_gap_two_modes() {
        let sp = StringSpectrum::new();
        let modes = vec![
            StringMode { frequency: 0.1, amplitude: 1.0, mode_number: 1 },
            StringMode { frequency: 0.3, amplitude: 0.5, mode_number: 2 },
        ];
        let gap = sp.spectral_gap(&modes);
        assert!((gap - 0.2).abs() < 1e-9);
    }

    #[test]
    fn test_spectral_gap_single_mode() {
        let sp = StringSpectrum::new();
        let modes = vec![StringMode { frequency: 0.1, amplitude: 1.0, mode_number: 1 }];
        assert!((sp.spectral_gap(&modes) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mass_level_ground_state_is_zero() {
        let sp = StringSpectrum::new();
        // Mode 1: (1-1)/alpha' = 0
        assert!((sp.mass_level(1, 0.5) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_mass_level_higher_modes() {
        let sp = StringSpectrum::new();
        // Mode 3, alpha'=0.5: (3-1)/0.5 = 4
        assert!((sp.mass_level(3, 0.5) - 4.0).abs() < 1e-9);
    }

    #[test]
    fn test_decompose_empty_series() {
        let sp = StringSpectrum::new();
        let modes = sp.decompose(&[], 3);
        assert!(modes.is_empty());
    }
}
