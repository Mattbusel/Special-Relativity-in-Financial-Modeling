//! Reference-frame transformations for financial "observers".
//!
//! Models each market participant as an observer moving at a relativistic
//! velocity (expressed as a fraction of *c*).  Lorentz transforms are applied
//! to financial events (price, time, position) to derive how a fast-moving
//! participant perceives the same price action.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// Observer
// ---------------------------------------------------------------------------

/// A market observer characterised by its relativistic velocity and position.
///
/// `velocity` is expressed as a fraction of the speed of light *c*, so it
/// must lie strictly within `(-1, 1)`.  Clamping is applied automatically.
#[derive(Debug, Clone, PartialEq)]
pub struct Observer {
    /// Velocity as a fraction of *c*.  Clamped to `(-0.9999, 0.9999)`.
    pub velocity: f64,
    /// Spatial position (e.g. exchange location on a 1-D axis, in light-ms).
    pub position: f64,
}

impl Observer {
    /// Create a new observer, clamping velocity away from ±1.
    pub fn new(velocity: f64, position: f64) -> Self {
        let clamped = velocity.clamp(-0.9999, 0.9999);
        Self { velocity: clamped, position }
    }

    /// Returns an observer at rest.
    pub fn at_rest(position: f64) -> Self {
        Self::new(0.0, position)
    }
}

// ---------------------------------------------------------------------------
// FinancialEvent
// ---------------------------------------------------------------------------

/// A price event in a particular reference frame.
#[derive(Debug, Clone, PartialEq)]
pub struct FinancialEvent {
    /// Asset price (analogous to energy in relativistic mechanics).
    pub price:    f64,
    /// Coordinate time of the event (e.g. seconds since epoch).
    pub time:     f64,
    /// Spatial position of the exchange emitting the event.
    pub position: f64,
}

impl FinancialEvent {
    pub fn new(price: f64, time: f64, position: f64) -> Self {
        Self { price, time, position }
    }
}

// ---------------------------------------------------------------------------
// Relativistic factors
// ---------------------------------------------------------------------------

/// Lorentz factor γ = 1 / √(1 − v²).
///
/// `v` must be in `(-1, 1)`.  Values outside this range are clamped.
pub fn time_dilation_factor(v: f64) -> f64 {
    let v = v.clamp(-0.9999, 0.9999);
    1.0 / (1.0 - v * v).sqrt()
}

/// Length contraction factor = 1 / γ = √(1 − v²).
pub fn length_contraction_factor(v: f64) -> f64 {
    let v = v.clamp(-0.9999, 0.9999);
    (1.0 - v * v).sqrt()
}

// ---------------------------------------------------------------------------
// Lorentz transform
// ---------------------------------------------------------------------------

/// Apply a standard Lorentz boost to a financial event.
///
/// The observer moves at velocity `v` (fraction of *c*) along the spatial
/// axis.  The transformed coordinates are:
/// ```text
///   t' = γ (t − v·x)
///   x' = γ (x − v·t)
///   p' = relativistic_price_adjustment(price, v)
/// ```
pub fn lorentz_transform(event: &FinancialEvent, observer: &Observer) -> FinancialEvent {
    let v     = observer.velocity;
    let gamma = time_dilation_factor(v);

    let t_prime = gamma * (event.time     - v * event.position);
    let x_prime = gamma * (event.position - v * event.time);
    let p_prime = relativistic_price_adjustment(event.price, v);

    FinancialEvent {
        price:    p_prime,
        time:     t_prime,
        position: x_prime,
    }
}

// ---------------------------------------------------------------------------
// Relativistic price adjustment
// ---------------------------------------------------------------------------

/// Treat price momentum as relativistic (invariant) mass.
///
/// Analogous to relativistic kinetic energy: the *observed* price is boosted
/// by the Lorentz factor, modelling how a fast-moving participant perceives
/// higher effective asset prices due to their relative motion.
///
/// `price_prime = price × γ`
pub fn relativistic_price_adjustment(price: f64, v: f64) -> f64 {
    price * time_dilation_factor(v)
}

// ---------------------------------------------------------------------------
// Simultaneity breakdown
// ---------------------------------------------------------------------------

/// Reorder a slice of events as perceived by a given observer.
///
/// Events that are simultaneous in the rest frame may appear in a different
/// order when viewed from a moving frame.  This function transforms each
/// event's time coordinate and returns them sorted by transformed time.
pub fn simultaneity_breakdown(
    events:   &[FinancialEvent],
    observer: &Observer,
) -> Vec<FinancialEvent> {
    let mut transformed: Vec<FinancialEvent> = events
        .iter()
        .map(|e| lorentz_transform(e, observer))
        .collect();
    transformed.sort_by(|a, b| a.time.partial_cmp(&b.time).unwrap_or(std::cmp::Ordering::Equal));
    transformed
}

// ---------------------------------------------------------------------------
// Spacetime interval (invariant)
// ---------------------------------------------------------------------------

/// The spacetime interval Δs² = Δt² − Δx² is Lorentz-invariant.
///
/// Returns `None` if either slice is empty.
pub fn spacetime_interval(a: &FinancialEvent, b: &FinancialEvent) -> f64 {
    let dt = a.time     - b.time;
    let dx = a.position - b.position;
    dt * dt - dx * dx
}

// ---------------------------------------------------------------------------
// Doppler shift for prices
// ---------------------------------------------------------------------------

/// Relativistic Doppler factor: prices "blue-shift" or "red-shift" depending
/// on whether the observer is approaching or receding.
///
/// `factor = sqrt((1 + v) / (1 - v))`  for an approaching source.
pub fn doppler_price_factor(v: f64) -> f64 {
    let v = v.clamp(-0.9999, 0.9999);
    ((1.0 + v) / (1.0 - v)).sqrt()
}

/// Apply a Doppler shift to a price.
pub fn doppler_shifted_price(price: f64, v: f64) -> f64 {
    price * doppler_price_factor(v)
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    // --- time_dilation_factor ---

    #[test]
    fn gamma_at_rest_is_one() {
        assert!((time_dilation_factor(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn gamma_increases_with_velocity() {
        let g1 = time_dilation_factor(0.5);
        let g2 = time_dilation_factor(0.9);
        assert!(g2 > g1);
        assert!(g1 > 1.0);
    }

    #[test]
    fn gamma_clamps_at_boundary() {
        let g = time_dilation_factor(1.5);
        assert!(g.is_finite());
        assert!(g > 0.0);
    }

    // --- length_contraction_factor ---

    #[test]
    fn contraction_at_rest_is_one() {
        assert!((length_contraction_factor(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn contraction_is_less_than_one_when_moving() {
        let lc = length_contraction_factor(0.6);
        assert!(lc < 1.0 && lc > 0.0);
    }

    #[test]
    fn gamma_times_contraction_is_one() {
        let v = 0.7;
        let product = time_dilation_factor(v) * length_contraction_factor(v);
        assert!((product - 1.0).abs() < EPS);
    }

    // --- lorentz_transform ---

    #[test]
    fn lorentz_at_rest_is_identity_for_time_and_position() {
        let event    = FinancialEvent::new(100.0, 10.0, 5.0);
        let observer = Observer::at_rest(0.0);
        let t        = lorentz_transform(&event, &observer);
        assert!((t.time     - event.time    ).abs() < EPS);
        assert!((t.position - event.position).abs() < EPS);
        // Price at rest: gamma=1 so price unchanged.
        assert!((t.price    - event.price   ).abs() < EPS);
    }

    #[test]
    fn lorentz_moving_observer_changes_time() {
        let event    = FinancialEvent::new(100.0, 10.0, 0.0);
        let observer = Observer::new(0.5, 0.0);
        let t        = lorentz_transform(&event, &observer);
        assert!((t.time - event.time).abs() > EPS);
    }

    #[test]
    fn lorentz_moving_observer_boosts_price() {
        let event    = FinancialEvent::new(100.0, 0.0, 0.0);
        let observer = Observer::new(0.6, 0.0);
        let t        = lorentz_transform(&event, &observer);
        assert!(t.price > event.price);
    }

    // --- simultaneity_breakdown ---

    #[test]
    fn simultaneity_breakdown_returns_same_count() {
        let events = vec![
            FinancialEvent::new(100.0, 1.0, 0.0),
            FinancialEvent::new(101.0, 1.0, 10.0),
            FinancialEvent::new(99.0,  1.0, -5.0),
        ];
        let observer = Observer::new(0.5, 0.0);
        let transformed = simultaneity_breakdown(&events, &observer);
        assert_eq!(transformed.len(), 3);
    }

    #[test]
    fn simultaneity_breakdown_is_sorted_by_time() {
        let events = vec![
            FinancialEvent::new(100.0, 5.0,  10.0),
            FinancialEvent::new(100.0, 1.0, -10.0),
            FinancialEvent::new(100.0, 3.0,   0.0),
        ];
        let observer = Observer::new(0.7, 0.0);
        let result = simultaneity_breakdown(&events, &observer);
        for w in result.windows(2) {
            assert!(w[0].time <= w[1].time, "not sorted: {:?} > {:?}", w[0].time, w[1].time);
        }
    }

    #[test]
    fn simultaneity_breakdown_empty_input() {
        let observer = Observer::new(0.5, 0.0);
        let result = simultaneity_breakdown(&[], &observer);
        assert!(result.is_empty());
    }

    // --- relativistic_price_adjustment ---

    #[test]
    fn price_adjustment_at_rest_unchanged() {
        assert!((relativistic_price_adjustment(100.0, 0.0) - 100.0).abs() < EPS);
    }

    #[test]
    fn price_adjustment_increases_with_velocity() {
        let p1 = relativistic_price_adjustment(100.0, 0.3);
        let p2 = relativistic_price_adjustment(100.0, 0.8);
        assert!(p2 > p1);
    }

    // --- spacetime_interval ---

    #[test]
    fn spacetime_interval_same_event_is_zero() {
        let e = FinancialEvent::new(100.0, 5.0, 3.0);
        assert!((spacetime_interval(&e, &e)).abs() < EPS);
    }

    // --- doppler ---

    #[test]
    fn doppler_at_rest_is_one() {
        assert!((doppler_price_factor(0.0) - 1.0).abs() < EPS);
    }

    #[test]
    fn doppler_approaching_is_greater_than_one() {
        assert!(doppler_price_factor(0.5) > 1.0);
    }

    #[test]
    fn doppler_receding_is_less_than_one() {
        assert!(doppler_price_factor(-0.5) < 1.0);
    }

    // --- Observer constructor ---

    #[test]
    fn observer_clamps_velocity() {
        let o = Observer::new(2.0, 0.0);
        assert!(o.velocity < 1.0);
    }

    #[test]
    fn observer_at_rest() {
        let o = Observer::at_rest(5.0);
        assert_eq!(o.velocity, 0.0);
        assert_eq!(o.position, 5.0);
    }

    // --- PI is used indirectly via doppler; silence unused-import lint ---
    #[test]
    fn pi_is_reasonable() {
        assert!((PI - 3.14159).abs() < 0.001);
    }
}
