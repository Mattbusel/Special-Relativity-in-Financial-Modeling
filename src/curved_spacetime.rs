//! Curved spacetime analogies for financial markets.
//!
//! The Schwarzschild metric is used to model market concentration (whale
//! positions) and its effect on price signals (gravitational redshift).

// ---------------------------------------------------------------------------
// SchwarzschildMetric
// ---------------------------------------------------------------------------

/// Schwarzschild metric parameters.
///
/// `mass` is analogous to total whale position concentration — the heavier the
/// position, the stronger the curvature around that price level.
#[derive(Debug, Clone, Copy)]
pub struct SchwarzschildMetric {
    /// Effective gravitational mass (market concentration), M > 0.
    pub mass: f64,
}

impl SchwarzschildMetric {
    pub fn new(mass: f64) -> Self {
        assert!(mass > 0.0, "mass must be positive");
        SchwarzschildMetric { mass }
    }

    /// Schwarzschild radius r_s = 2M (using G=c=1 units).
    pub fn event_horizon_radius(&self) -> f64 {
        2.0 * self.mass
    }

    /// Time-time metric component g_tt = -(1 - 2M/r).
    ///
    /// Returns positive magnitude (|g_tt|) to avoid sign confusion in
    /// downstream finance calculations.
    pub fn metric_component_tt(&self, r: f64) -> f64 {
        let rs = self.event_horizon_radius();
        1.0 - rs / r
    }

    /// Radial-radial metric component g_rr = 1 / (1 - 2M/r).
    pub fn metric_component_rr(&self, r: f64) -> f64 {
        let denom = self.metric_component_tt(r);
        if denom.abs() < 1e-12 {
            f64::INFINITY
        } else {
            1.0 / denom
        }
    }

    /// Gravitational redshift: ratio of observed frequency to emitted frequency.
    ///
    /// f_obs / f_emit = sqrt(g_tt(r_emit) / g_tt(r_obs))
    ///
    /// Values < 1 mean the signal is redshifted (slowed) near a whale position.
    pub fn gravitational_redshift(&self, r_emitter: f64, r_observer: f64) -> f64 {
        let g_emit = self.metric_component_tt(r_emitter);
        let g_obs  = self.metric_component_tt(r_observer);
        if g_emit <= 0.0 || g_obs <= 0.0 {
            return 0.0;
        }
        (g_emit / g_obs).sqrt()
    }

    /// Escape velocity at radius r: v_esc = sqrt(2M/r), clamped to 1.0 (c).
    pub fn escape_velocity(&self, r: f64) -> f64 {
        let rs = self.event_horizon_radius();
        ((rs / r).sqrt()).min(1.0)
    }
}

// ---------------------------------------------------------------------------
// GeodesicPath
// ---------------------------------------------------------------------------

/// Result of a radial geodesic integration.
#[derive(Debug, Clone)]
pub struct GeodesicPath {
    /// Coordinate radii at each time step.
    pub r_values: Vec<f64>,
    /// Coordinate time at each step.
    pub t_values: Vec<f64>,
    /// Accumulated proper time at each step.
    pub proper_times: Vec<f64>,
}

// ---------------------------------------------------------------------------
// integrate_radial_geodesic
// ---------------------------------------------------------------------------

/// Euler integration of radial free-fall in the Schwarzschild metric.
///
/// Uses coordinate time `t` as the integration parameter with step `dt`.
/// The radial velocity `v0` is the initial dr/dt (negative for infall).
///
/// # Arguments
/// * `metric`  – Schwarzschild metric.
/// * `r0`      – Initial radius (must be > r_s).
/// * `v0`      – Initial dr/dt (negative for inward trajectory).
/// * `steps`   – Number of Euler steps.
/// * `dt`      – Coordinate time step size.
pub fn integrate_radial_geodesic(
    metric: &SchwarzschildMetric,
    r0: f64,
    v0: f64,
    steps: usize,
    dt: f64,
) -> GeodesicPath {
    let mut r = r0;
    let mut v = v0;
    let mut t = 0.0;
    let mut tau = 0.0;

    let mut r_values    = Vec::with_capacity(steps + 1);
    let mut t_values    = Vec::with_capacity(steps + 1);
    let mut proper_times = Vec::with_capacity(steps + 1);

    r_values.push(r);
    t_values.push(t);
    proper_times.push(tau);

    let rs = metric.event_horizon_radius();

    for _ in 0..steps {
        if r <= rs * 1.001 {
            // Stop integration at or inside the horizon.
            break;
        }
        // Acceleration: d2r/dt2 = -M/r^2 * (1 - rs/r)^2   [radial geodesic approx]
        let factor = 1.0 - rs / r;
        let accel = -metric.mass / (r * r) * factor * factor;

        v += accel * dt;
        r += v * dt;
        t += dt;

        // Proper time increment: d_tau = dt * sqrt(|g_tt| - g_rr * v^2 / c^2)
        // In G=c=1 units, c=1.
        let g_tt = factor.abs();
        let g_rr = if factor.abs() > 1e-12 { 1.0 / factor } else { f64::INFINITY };
        let inside = g_tt - g_rr * v * v;
        let d_tau = if inside > 0.0 { inside.sqrt() * dt } else { 0.0 };
        tau += d_tau;

        r_values.push(r);
        t_values.push(t);
        proper_times.push(tau);
    }

    GeodesicPath { r_values, t_values, proper_times }
}

// ---------------------------------------------------------------------------
// FinancialCurvature
// ---------------------------------------------------------------------------

/// Maps financial market quantities to curved-spacetime analogues.
pub struct FinancialCurvature {
    pub metric: SchwarzschildMetric,
}

impl FinancialCurvature {
    /// Create from whale trade concentration (total position / market cap).
    ///
    /// `concentration` in (0, 1] is rescaled to an effective mass.
    pub fn from_concentration(concentration: f64) -> Self {
        let mass = (concentration * 100.0).max(0.001);
        FinancialCurvature {
            metric: SchwarzschildMetric::new(mass),
        }
    }

    /// Price levels closer than `horizon_radius()` are inaccessible to
    /// traders without exceeding the whale's liquidity wall.
    pub fn horizon_radius(&self) -> f64 {
        self.metric.event_horizon_radius()
    }

    /// Signal redshift: how much a price signal emitted at `price_level` is
    /// distorted by the time it reaches observers at `observer_level`.
    pub fn signal_redshift(&self, price_level: f64, observer_level: f64) -> f64 {
        self.metric.gravitational_redshift(price_level, observer_level)
    }

    /// Probability of a price escaping the whale's gravity well.
    ///
    /// Returns a value in [0, 1]: 0 = trapped, 1 = free.
    pub fn escape_probability(&self, price_level: f64) -> f64 {
        let v_esc = self.metric.escape_velocity(price_level);
        1.0 - v_esc
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const M: f64 = 10.0;

    #[test]
    fn test_event_horizon_radius() {
        let m = SchwarzschildMetric::new(M);
        assert!((m.event_horizon_radius() - 2.0 * M).abs() < 1e-10);
    }

    #[test]
    fn test_metric_tt_at_infinity_is_one() {
        let m = SchwarzschildMetric::new(M);
        let g = m.metric_component_tt(1e12);
        assert!((g - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_metric_tt_at_horizon_is_zero() {
        let m = SchwarzschildMetric::new(M);
        let rs = m.event_horizon_radius();
        let g = m.metric_component_tt(rs);
        assert!(g.abs() < 1e-10);
    }

    #[test]
    fn test_metric_rr_inverse_of_tt() {
        let m = SchwarzschildMetric::new(M);
        let r = 50.0;
        let tt = m.metric_component_tt(r);
        let rr = m.metric_component_rr(r);
        assert!((tt * rr - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gravitational_redshift_same_radius_is_one() {
        let m = SchwarzschildMetric::new(M);
        let ratio = m.gravitational_redshift(100.0, 100.0);
        assert!((ratio - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_gravitational_redshift_inner_emitter_is_blueshifted_at_outer() {
        // Observer farther out: signal emitted from closer in is redshifted.
        let m = SchwarzschildMetric::new(M);
        let ratio = m.gravitational_redshift(30.0, 100.0);
        // g_tt(30) < g_tt(100) => ratio < 1 (redshift)
        assert!(ratio < 1.0, "ratio={}", ratio);
    }

    #[test]
    fn test_escape_velocity_at_horizon_is_one() {
        let m = SchwarzschildMetric::new(M);
        let rs = m.event_horizon_radius();
        let v = m.escape_velocity(rs);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_escape_velocity_far_field_approaches_zero() {
        let m = SchwarzschildMetric::new(M);
        let v = m.escape_velocity(1e10);
        assert!(v < 0.001);
    }

    #[test]
    fn test_geodesic_integration_starts_at_r0() {
        let m = SchwarzschildMetric::new(M);
        let path = integrate_radial_geodesic(&m, 100.0, -0.1, 100, 0.1);
        assert!((path.r_values[0] - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_geodesic_integration_inward_motion() {
        let m = SchwarzschildMetric::new(M);
        let path = integrate_radial_geodesic(&m, 100.0, -0.5, 50, 0.1);
        assert!(path.r_values.last().unwrap() < &100.0);
    }

    #[test]
    fn test_geodesic_proper_time_monotone() {
        let m = SchwarzschildMetric::new(M);
        let path = integrate_radial_geodesic(&m, 100.0, -0.3, 30, 0.1);
        for i in 1..path.proper_times.len() {
            assert!(path.proper_times[i] >= path.proper_times[i - 1]);
        }
    }

    #[test]
    fn test_financial_curvature_from_concentration() {
        let fc = FinancialCurvature::from_concentration(0.1);
        // mass = 0.1 * 100 = 10, horizon = 20
        assert!((fc.horizon_radius() - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_escape_probability_far_from_horizon() {
        let fc = FinancialCurvature::from_concentration(0.01);
        let p = fc.escape_probability(1e9);
        assert!(p > 0.99);
    }

    #[test]
    fn test_signal_redshift_same_level() {
        let fc = FinancialCurvature::from_concentration(0.1);
        let r = fc.signal_redshift(100.0, 100.0);
        assert!((r - 1.0).abs() < 1e-9);
    }
}
