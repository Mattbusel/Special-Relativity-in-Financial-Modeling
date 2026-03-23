//! Relativistic tensor operations for portfolio analysis.
//!
//! Implements basic 2×2 tensor arithmetic, the Minkowski metric in 1+1 dimensions,
//! Lorentz boost matrices, and a stress-energy tensor analog for portfolio dynamics.

// ─── TENSOR 2×2 ───────────────────────────────────────────────────────────────

/// A general 2×2 matrix / rank-2 tensor.
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Tensor2x2 {
    /// Row-major storage: `data[row][col]`.
    pub data: [[f64; 2]; 2],
}

impl Tensor2x2 {
    /// Construct from explicit values (row-major).
    pub fn new(a00: f64, a01: f64, a10: f64, a11: f64) -> Self {
        Self {
            data: [[a00, a01], [a10, a11]],
        }
    }

    /// 2×2 identity matrix.
    pub fn identity() -> Self {
        Self::new(1.0, 0.0, 0.0, 1.0)
    }

    /// 2×2 zero matrix.
    pub fn zero() -> Self {
        Self::new(0.0, 0.0, 0.0, 0.0)
    }

    /// Component-wise addition.
    pub fn add(&self, other: &Self) -> Self {
        Self {
            data: [
                [
                    self.data[0][0] + other.data[0][0],
                    self.data[0][1] + other.data[0][1],
                ],
                [
                    self.data[1][0] + other.data[1][0],
                    self.data[1][1] + other.data[1][1],
                ],
            ],
        }
    }

    /// Matrix multiplication (`self * other`).
    pub fn mul(&self, other: &Self) -> Self {
        let a = &self.data;
        let b = &other.data;
        Self {
            data: [
                [
                    a[0][0] * b[0][0] + a[0][1] * b[1][0],
                    a[0][0] * b[0][1] + a[0][1] * b[1][1],
                ],
                [
                    a[1][0] * b[0][0] + a[1][1] * b[1][0],
                    a[1][0] * b[0][1] + a[1][1] * b[1][1],
                ],
            ],
        }
    }

    /// Transpose.
    pub fn transpose(&self) -> Self {
        Self {
            data: [
                [self.data[0][0], self.data[1][0]],
                [self.data[0][1], self.data[1][1]],
            ],
        }
    }

    /// Determinant.
    pub fn det(&self) -> f64 {
        self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
    }

    /// Trace (sum of diagonal elements).
    pub fn trace(&self) -> f64 {
        self.data[0][0] + self.data[1][1]
    }

    /// Inverse, or `None` if the determinant is approximately zero.
    pub fn inv(&self) -> Option<Tensor2x2> {
        let d = self.det();
        if d.abs() < 1e-12 {
            return None;
        }
        Some(Self::new(
            self.data[1][1] / d,
            -self.data[0][1] / d,
            -self.data[1][0] / d,
            self.data[0][0] / d,
        ))
    }
}

impl std::ops::Add for Tensor2x2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Tensor2x2::add(&self, &rhs)
    }
}

impl std::ops::Mul for Tensor2x2 {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Tensor2x2::mul(&self, &rhs)
    }
}

// ─── MINKOWSKI METRIC ─────────────────────────────────────────────────────────

/// Minkowski metric in 1+1 dimensions with signature (−, +).
///
/// The metric tensor is `η = diag(-c², +1)` where `c` is the speed of light.
pub struct MinkowskiMetric {
    /// Speed of light (or analogous constant).
    pub c: f64,
}

impl MinkowskiMetric {
    /// Create with the standard speed of light (`c = 1.0` in natural units).
    pub fn new(c: f64) -> Self {
        Self { c }
    }

    /// Natural-units metric (`c = 1`).
    pub fn natural() -> Self {
        Self::new(1.0)
    }

    /// Minkowski inner product: `a^μ η_μν b^ν = -c²·a[0]·b[0] + a[1]·b[1]`.
    pub fn inner_product(&self, a: &[f64; 2], b: &[f64; 2]) -> f64 {
        -(self.c * self.c) * a[0] * b[0] + a[1] * b[1]
    }

    /// Raise a covariant index by multiplying by the inverse metric.
    ///
    /// Inverse metric for `diag(-c², 1)` is `diag(-1/c², 1)`.
    pub fn raise_index(&self, covariant: &[f64; 2]) -> [f64; 2] {
        [
            -covariant[0] / (self.c * self.c),
            covariant[1],
        ]
    }
}

// ─── LORENTZ TRANSFORM MATRIX ─────────────────────────────────────────────────

/// A 2×2 Lorentz boost matrix along one spatial direction.
///
/// The matrix is:
/// ```text
/// Λ = | γ        -γβ |
///     | -γβ       γ  |
/// ```
/// where `β = v/c` and `γ = 1/√(1-β²)`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LorentzTransformMatrix {
    /// Velocity parameter `β = v/c ∈ (-1, 1)`.
    pub beta: f64,
    /// Lorentz factor `γ`.
    pub gamma: f64,
    /// The underlying 2×2 matrix.
    pub matrix: Tensor2x2,
}

impl LorentzTransformMatrix {
    /// Build the boost matrix for a given `beta ∈ (-1, 1)`.
    ///
    /// # Panics
    /// Panics if `|beta| >= 1`.
    pub fn new(beta: f64) -> Self {
        assert!(
            beta.abs() < 1.0,
            "beta must satisfy |beta| < 1, got {}",
            beta
        );
        let gamma = 1.0 / (1.0 - beta * beta).sqrt();
        let matrix = Tensor2x2::new(gamma, -gamma * beta, -gamma * beta, gamma);
        Self { beta, gamma, matrix }
    }

    /// Apply the boost to a 2-vector `[t, x]`.
    pub fn apply(&self, event: &[f64; 2]) -> [f64; 2] {
        let m = &self.matrix.data;
        [
            m[0][0] * event[0] + m[0][1] * event[1],
            m[1][0] * event[0] + m[1][1] * event[1],
        ]
    }

    /// Compose two boosts using relativistic velocity addition.
    ///
    /// β_combined = (β₁ + β₂) / (1 + β₁·β₂)
    pub fn compose(l1: &Self, l2: &Self) -> Self {
        let beta_combined = (l1.beta + l2.beta) / (1.0 + l1.beta * l2.beta);
        Self::new(beta_combined)
    }
}

// ─── STRESS-ENERGY TENSOR ─────────────────────────────────────────────────────

/// Financial analog of the stress-energy tensor T^μν.
///
/// | Component         | Financial interpretation             |
/// |-------------------|--------------------------------------|
/// | `energy_density`  | Portfolio value density              |
/// | `momentum_density`| Rate of value change (∂V/∂t)        |
/// | `pressure`        | Market impact / volatility pressure  |
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct StressEnergyTensor {
    /// T^00: energy (value) density.
    pub energy_density: f64,
    /// T^01 / T^10: momentum (rate of change).
    pub momentum_density: f64,
    /// T^11: spatial pressure (market impact).
    pub pressure: f64,
}

impl StressEnergyTensor {
    /// Invariant scalar trace: T^μ_μ = -energy_density + pressure (1+1D).
    pub fn trace(&self) -> f64 {
        -self.energy_density + self.pressure
    }

    /// Measure how well "energy" is conserved across a time series of tensors.
    ///
    /// Computes the discrete divergence (finite difference) of the energy
    /// density over time and returns the sum of absolute increments divided by
    /// `dt`.
    ///
    /// A value near zero implies conservation; large values indicate rapid
    /// value flow.
    pub fn conservation_check(tensors: &[StressEnergyTensor], dt: f64) -> f64 {
        if tensors.len() < 2 || dt == 0.0 {
            return 0.0;
        }
        tensors
            .windows(2)
            .map(|w| (w[1].energy_density - w[0].energy_density) / dt)
            .sum::<f64>()
    }
}

// ─── TESTS ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-10;

    // ── Tensor2x2 ──────────────────────────────────────────────────────────

    #[test]
    fn identity_det_is_one() {
        let id = Tensor2x2::identity();
        assert!((id.det() - 1.0).abs() < EPS);
    }

    #[test]
    fn identity_trace_is_two() {
        let id = Tensor2x2::identity();
        assert!((id.trace() - 2.0).abs() < EPS);
    }

    #[test]
    fn mul_by_identity_is_self() {
        let m = Tensor2x2::new(1.0, 2.0, 3.0, 4.0);
        let id = Tensor2x2::identity();
        let result = m.mul(&id);
        assert!((result.data[0][0] - m.data[0][0]).abs() < EPS);
        assert!((result.data[1][1] - m.data[1][1]).abs() < EPS);
    }

    #[test]
    fn inv_of_identity_is_identity() {
        let id = Tensor2x2::identity();
        let inv = id.inv().unwrap();
        assert!((inv.data[0][0] - 1.0).abs() < EPS);
        assert!((inv.data[0][1]).abs() < EPS);
    }

    #[test]
    fn singular_matrix_returns_none() {
        let singular = Tensor2x2::new(1.0, 2.0, 2.0, 4.0);
        assert!(singular.inv().is_none());
    }

    // ── LorentzTransformMatrix ─────────────────────────────────────────────

    #[test]
    fn boost_at_beta_zero_is_identity() {
        let boost = LorentzTransformMatrix::new(0.0);
        assert!((boost.gamma - 1.0).abs() < EPS);
        let event = [2.0, 3.0];
        let boosted = boost.apply(&event);
        assert!((boosted[0] - event[0]).abs() < EPS);
        assert!((boosted[1] - event[1]).abs() < EPS);
    }

    #[test]
    fn boost_composition_uses_relativistic_velocity_addition() {
        let l1 = LorentzTransformMatrix::new(0.5);
        let l2 = LorentzTransformMatrix::new(0.5);
        let composed = LorentzTransformMatrix::compose(&l1, &l2);
        // (0.5 + 0.5) / (1 + 0.25) = 1.0 / 1.25 = 0.8
        assert!((composed.beta - 0.8).abs() < EPS);
    }

    #[test]
    fn boost_matrix_det_is_one() {
        // All Lorentz boosts have det = 1.
        let boost = LorentzTransformMatrix::new(0.6);
        assert!((boost.matrix.det() - 1.0).abs() < 1e-9);
    }

    // ── MinkowskiMetric ────────────────────────────────────────────────────

    #[test]
    fn minkowski_inner_product_is_lorentz_invariant() {
        // A timelike separation (ct, x) = (1, 0) should give -1 in natural units.
        let metric = MinkowskiMetric::natural();
        let a = [1.0, 0.0];
        let ip = metric.inner_product(&a, &a);
        assert!((ip - (-1.0)).abs() < EPS);
    }

    #[test]
    fn spacelike_separation_positive_inner_product() {
        let metric = MinkowskiMetric::natural();
        let a = [0.0, 1.0]; // purely spatial
        let ip = metric.inner_product(&a, &a);
        assert!((ip - 1.0).abs() < EPS);
    }

    #[test]
    fn inner_product_preserved_under_boost() {
        // Apply a boost to a 4-vector and check the inner product is invariant.
        let metric = MinkowskiMetric::natural();
        let event = [2.0, 1.0];
        let ip_before = metric.inner_product(&event, &event);

        let boost = LorentzTransformMatrix::new(0.5);
        let boosted = boost.apply(&event);
        let ip_after = metric.inner_product(&boosted, &boosted);

        assert!((ip_before - ip_after).abs() < 1e-9);
    }

    // ── StressEnergyTensor ─────────────────────────────────────────────────

    #[test]
    fn trace_is_negative_energy_plus_pressure() {
        let t = StressEnergyTensor {
            energy_density: 10.0,
            momentum_density: 0.0,
            pressure: 2.0,
        };
        assert!((t.trace() - (-8.0)).abs() < EPS);
    }

    #[test]
    fn conservation_check_constant_energy() {
        let tensors: Vec<StressEnergyTensor> = (0..5)
            .map(|_| StressEnergyTensor {
                energy_density: 100.0,
                momentum_density: 0.0,
                pressure: 0.0,
            })
            .collect();
        assert!((StressEnergyTensor::conservation_check(&tensors, 1.0)).abs() < EPS);
    }

    #[test]
    fn conservation_check_linearly_increasing_energy() {
        let tensors: Vec<StressEnergyTensor> = (0..5)
            .map(|i| StressEnergyTensor {
                energy_density: i as f64,
                momentum_density: 0.0,
                pressure: 0.0,
            })
            .collect();
        // Each step delta = 1, dt = 1, sum = 4 steps × (1/1)
        let result = StressEnergyTensor::conservation_check(&tensors, 1.0);
        assert!((result - 4.0).abs() < EPS);
    }
}
