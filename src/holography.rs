//! Holographic principle and AdS/CFT analogy for financial markets.
//!
//! This module provides structs and functions modelling the holographic
//! principle: bulk market information encoded on a lower-dimensional boundary,
//! inspired by the AdS/CFT correspondence.

/// A point in the AdS bulk space (radial coordinate z, time t, spatial x).
/// z → 0 corresponds to the CFT boundary.
#[derive(Debug, Clone, Copy)]
pub struct AdsBulkPoint {
    /// Radial coordinate (z → 0 is boundary).
    pub radial: f64,
    pub time: f64,
    pub x: f64,
}

/// A point on the CFT (boundary) side of the duality.
#[derive(Debug, Clone, Copy)]
pub struct CftBoundaryPoint {
    pub time: f64,
    pub x: f64,
}

/// A holographic screen described by area, entropy, and temperature.
#[derive(Debug, Clone, Copy)]
pub struct HolographicScreen {
    pub area: f64,
    pub entropy: f64,
    pub temperature: f64,
}

impl HolographicScreen {
    /// Bekenstein–Hawking entropy: S = A / (4 l_P²).
    pub fn bekenstein_hawking_entropy(area: f64, planck_area: f64) -> f64 {
        area / (4.0 * planck_area)
    }

    /// Unruh temperature (dimensionless): T = a / (2π).
    pub fn unruh_temperature(acceleration: f64) -> f64 {
        acceleration / (2.0 * std::f64::consts::PI)
    }

    /// Hawking temperature in Planck units: T = 1 / (8πM).
    pub fn hawking_temperature(mass: f64) -> f64 {
        if mass.abs() < 1e-300 {
            return f64::INFINITY;
        }
        1.0 / (8.0 * std::f64::consts::PI * mass)
    }
}

/// A bulk scalar field on a radial × time grid.
#[derive(Debug, Clone)]
pub struct BulkField {
    /// values[radial_idx][time_idx]
    pub values: Vec<Vec<f64>>,
    pub dz: f64,
    pub dt: f64,
}

impl BulkField {
    /// Extrapolate to the boundary (z → 0) using a linear fit of the last two
    /// radial slices (smallest z values).
    pub fn propagate_to_boundary(&self) -> Vec<f64> {
        let nr = self.values.len();
        if nr == 0 {
            return vec![];
        }
        if nr == 1 {
            return self.values[0].clone();
        }
        // Index 0 is smallest z (closest to boundary).
        let v0 = &self.values[0];
        let v1 = &self.values[1];
        v0.iter()
            .zip(v1.iter())
            .map(|(&a, &b)| {
                let slope = (a - b) / self.dz;
                a + slope * self.dz * 0.5
            })
            .collect()
    }

    /// Boundary–boundary two-point function between time indices t1 and t2.
    pub fn two_point_function(&self, t1: usize, t2: usize) -> f64 {
        let nr = self.values.len();
        if nr == 0 {
            return 0.0;
        }
        let nt = self.values[0].len();
        if t1 >= nt || t2 >= nt {
            return 0.0;
        }
        let col1: Vec<f64> =
            (0..nr).map(|r| self.values[r].get(t1).copied().unwrap_or(0.0)).collect();
        let col2: Vec<f64> =
            (0..nr).map(|r| self.values[r].get(t2).copied().unwrap_or(0.0)).collect();
        col1.iter().zip(col2.iter()).map(|(a, b)| a * b).sum::<f64>() * self.dz
    }
}

/// A Ryu–Takayanagi minimal surface (semicircle arc in AdS₂).
#[derive(Debug, Clone, Copy)]
pub struct RyuTakayanagiSurface {
    /// (x_left, x_right) boundary endpoints.
    pub endpoints: (f64, f64),
    /// Bulk depth / UV cutoff ε.
    pub bulk_depth: f64,
}

impl RyuTakayanagiSurface {
    /// Arc length: L = 2 ln(l / ε).
    pub fn area(&self) -> f64 {
        let l = (self.endpoints.1 - self.endpoints.0).abs();
        let eps = self.bulk_depth.max(1e-15);
        2.0 * (l / eps).max(1.0).ln()
    }

    /// Holographic entanglement entropy: S = (c/3) ln(l / ε).
    pub fn entanglement_entropy(&self, central_charge: f64) -> f64 {
        let l = (self.endpoints.1 - self.endpoints.0).abs();
        let eps = self.bulk_depth.max(1e-15);
        (central_charge / 3.0) * (l / eps).max(1.0).ln()
    }
}

/// Market holography utilities.
pub struct MarketHolography;

impl MarketHolography {
    /// Encode a price series into a bulk field by Gaussian smearing at each
    /// radial depth. Deeper layers receive more smearing.
    pub fn encode_prices_to_bulk(prices: &[f64], radial_depth: usize) -> BulkField {
        let nt = prices.len();
        let nr = radial_depth.max(1);
        let mut values = vec![vec![0.0f64; nt]; nr];
        let dz = 1.0 / nr as f64;

        for (r, row) in values.iter_mut().enumerate() {
            let sigma = (r + 1) as f64 * dz;
            let sigma2 = 2.0 * sigma * sigma;
            for t in 0..nt {
                let mut acc = 0.0f64;
                let mut norm = 0.0f64;
                for (s, &p) in prices.iter().enumerate() {
                    let d = (t as f64 - s as f64).powi(2);
                    let w = (-d / sigma2).exp();
                    acc += w * p;
                    norm += w;
                }
                row[t] = if norm > 1e-15 { acc / norm } else { 0.0 };
            }
        }

        BulkField { values, dz, dt: 1.0 }
    }

    /// Holographic complexity: L1 norm of all bulk values × dz × dt.
    pub fn holographic_complexity(bulk: &BulkField) -> f64 {
        bulk.values
            .iter()
            .flat_map(|row| row.iter())
            .map(|v| v.abs())
            .sum::<f64>()
            * bulk.dz
            * bulk.dt
    }

    /// Info loss rate: average increase in boundary entropy per time step.
    pub fn info_loss_rate(bulk: &BulkField, steps: usize) -> f64 {
        let boundary = bulk.propagate_to_boundary();
        let nt = boundary.len();
        if nt < 2 || steps == 0 {
            return 0.0;
        }
        let actual_steps = steps.min(nt - 1);
        let mut total_increase = 0.0f64;
        let mut count = 0usize;
        for t in 0..actual_steps {
            let s_before = boundary.get(t).copied().unwrap_or(0.0).abs().ln().max(0.0);
            let s_after = boundary.get(t + 1).copied().unwrap_or(0.0).abs().ln().max(0.0);
            if s_after > s_before {
                total_increase += s_after - s_before;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { total_increase / count as f64 }
    }

    /// Rolling entanglement entropy of the price series using a sliding window.
    pub fn entanglement_structure(prices: &[f64], window: usize) -> Vec<f64> {
        let n = prices.len();
        if window == 0 || n < window {
            return vec![];
        }
        let eps = 1e-3_f64;
        let mut result = Vec::with_capacity(n - window + 1);
        for i in 0..=(n - window) {
            let slice = &prices[i..i + window];
            let surf = RyuTakayanagiSurface {
                endpoints: (slice[0], slice[window - 1]),
                bulk_depth: eps,
            };
            result.push(surf.entanglement_entropy(1.0).max(0.0));
        }
        result
    }
}
