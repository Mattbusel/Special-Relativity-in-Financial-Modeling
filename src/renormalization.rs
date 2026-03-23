use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub enum BetaFunction {
    QED { alpha: f64 },
    Phi4 { lambda: f64, n: u32 },
    YangMills { g: f64 },
    Custom { coefficients: Vec<f64> },
}

pub fn beta_value(func: &BetaFunction, coupling: f64) -> f64 {
    match func {
        BetaFunction::QED { alpha } => {
            alpha / (3.0 * PI) * coupling * coupling
        }
        BetaFunction::Phi4 { lambda, n: _ } => {
            3.0 * lambda / (16.0 * PI * PI) * coupling * coupling
        }
        BetaFunction::YangMills { g } => {
            // n here refers to number of flavors stored in g field metadata
            // Use g as n_flavors placeholder; the formula is:
            // -(11/3 - 2*n/3) / (16π²) * coupling^3
            let n_f = *g; // reuse g as n_flavors for this variant
            -(11.0 / 3.0 - 2.0 * n_f / 3.0) / (16.0 * PI * PI) * coupling.powi(3)
        }
        BetaFunction::Custom { coefficients } => {
            coefficients
                .iter()
                .enumerate()
                .map(|(i, &c)| c * coupling.powi(i as i32))
                .sum()
        }
    }
}

pub struct RGFlow {
    pub beta: BetaFunction,
    pub coupling_0: f64,
    pub scale_0: f64,
}

impl RGFlow {
    pub fn new(beta: BetaFunction, coupling_0: f64, scale_0: f64) -> Self {
        RGFlow { beta, coupling_0, scale_0 }
    }

    /// Euler integration: dc/d(ln μ) = beta(c)
    pub fn run_coupling(&self, target_scale: f64, steps: usize) -> f64 {
        if steps == 0 || self.scale_0 <= 0.0 || target_scale <= 0.0 {
            return self.coupling_0;
        }
        let ln_ratio = (target_scale / self.scale_0).ln();
        let dt = ln_ratio / steps as f64;
        let mut coupling = self.coupling_0;
        for _ in 0..steps {
            let beta = beta_value(&self.beta, coupling);
            coupling += beta * dt;
        }
        coupling
    }

    /// Scan for zeros of beta_value via sign changes and bisection
    pub fn fixed_points(&self, coupling_range: (f64, f64), n_points: usize) -> Vec<f64> {
        let mut fixed = Vec::new();
        if n_points < 2 {
            return fixed;
        }
        let (lo, hi) = coupling_range;
        let step = (hi - lo) / (n_points - 1) as f64;

        let mut prev_c = lo;
        let mut prev_b = beta_value(&self.beta, lo);

        for i in 1..n_points {
            let c = lo + i as f64 * step;
            let b = beta_value(&self.beta, c);
            if prev_b * b < 0.0 {
                // Sign change — bisect
                let mut a = prev_c;
                let mut bnd = c;
                let mut fa = prev_b;
                for _ in 0..50 {
                    let mid = (a + bnd) / 2.0;
                    let fmid = beta_value(&self.beta, mid);
                    if fmid.abs() < 1e-12 {
                        a = mid;
                        break;
                    }
                    if fa * fmid < 0.0 {
                        bnd = mid;
                    } else {
                        a = mid;
                        fa = fmid;
                    }
                }
                fixed.push((a + bnd) / 2.0);
            }
            prev_c = c;
            prev_b = b;
        }
        fixed
    }
}

pub struct VolatilityRG {
    pub vol_0: f64,
    pub time_scale: f64,
}

impl VolatilityRG {
    /// vol evolves as dσ/d(ln μ) = beta_coeff * σ^2; Euler integration
    pub fn run(vol_0: f64, beta_coeff: f64, scales: &[f64]) -> Vec<f64> {
        if scales.is_empty() {
            return Vec::new();
        }
        let mut vols = Vec::with_capacity(scales.len());
        let mut vol = vol_0;
        let mut prev_ln = scales[0].ln();
        vols.push(vol);

        for &s in &scales[1..] {
            let ln_s = s.ln();
            let d_ln = ln_s - prev_ln;
            vol += beta_coeff * vol * vol * d_ln;
            vols.push(vol);
            prev_ln = ln_s;
        }
        vols
    }
}

/// Mean-field critical exponents: (nu, eta, gamma)
pub fn mean_field_critical_exponents() -> (f64, f64, f64) {
    (0.5, 0.0, 1.0)
}

pub fn scaling_hypothesis(observable: f64, length_scale: f64, exponent: f64) -> f64 {
    observable * length_scale.powf(-exponent)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qed_beta_at_coupling_one() {
        let alpha = 1.0 / 137.0;
        let beta = BetaFunction::QED { alpha };
        let val = beta_value(&beta, 1.0);
        assert!((val - alpha / (3.0 * PI)).abs() < 1e-10);
    }

    #[test]
    fn test_phi4_beta_at_coupling_one() {
        let lambda = 0.1;
        let beta = BetaFunction::Phi4 { lambda, n: 0 };
        let val = beta_value(&beta, 1.0);
        let expected = 3.0 * lambda / (16.0 * PI * PI);
        assert!((val - expected).abs() < 1e-10);
    }

    #[test]
    fn test_qed_rg_coupling_increases_with_scale() {
        // QED beta function is positive → coupling increases at higher scale
        let flow = RGFlow::new(
            BetaFunction::QED { alpha: 1.0 / 137.0 },
            1.0 / 137.0,
            1.0,
        );
        let c_high = flow.run_coupling(10.0, 1000);
        assert!(c_high > flow.coupling_0, "coupling should increase: {} vs {}", c_high, flow.coupling_0);
    }

    #[test]
    fn test_fixed_point_detection() {
        // Custom beta with a zero at coupling=1.0: c*(c-1)
        let beta = BetaFunction::Custom {
            coefficients: vec![0.0, -1.0, 1.0], // -c + c^2
        };
        let flow = RGFlow::new(beta, 0.5, 1.0);
        let fps = flow.fixed_points((0.0, 2.0), 200);
        // Should find fixed points near 0 and 1
        assert!(!fps.is_empty(), "No fixed points found");
        let has_near_one = fps.iter().any(|&fp| (fp - 1.0).abs() < 0.05);
        assert!(has_near_one, "No fixed point near 1.0: {:?}", fps);
    }

    #[test]
    fn test_vol_scaling() {
        let scales = vec![1.0, 2.0, 4.0, 8.0];
        let vols = VolatilityRG::run(0.2, 0.1, &scales);
        assert_eq!(vols.len(), 4);
        // vol should increase with positive beta_coeff and scales
        assert!(vols.last().unwrap() >= &0.2);
    }
}
