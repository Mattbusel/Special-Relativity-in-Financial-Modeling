//! Quantum entanglement and Bell inequalities applied to market correlations.

/// A single qubit state: |ψ⟩ = α|0⟩ + β|1⟩ where α and β are complex numbers
/// represented as (re, im) tuples.
#[derive(Debug, Clone)]
pub struct QubitState {
    /// α = (re, im)
    pub alpha: (f64, f64),
    /// β = (re, im)
    pub beta: (f64, f64),
}

impl QubitState {
    /// Normalize so |α|² + |β|² = 1.
    pub fn normalize(&mut self) {
        let norm2 = self.alpha.0.powi(2)
            + self.alpha.1.powi(2)
            + self.beta.0.powi(2)
            + self.beta.1.powi(2);
        if norm2 < 1e-300 {
            return;
        }
        let norm = norm2.sqrt();
        self.alpha.0 /= norm;
        self.alpha.1 /= norm;
        self.beta.0 /= norm;
        self.beta.1 /= norm;
    }

    /// Probabilistic measurement in the Z basis using a simple LCG seed.
    /// Returns 0 with probability |α|² and 1 with probability |β|².
    pub fn measure(&self, seed: u64) -> u8 {
        let prob0 = self.alpha.0.powi(2) + self.alpha.1.powi(2);
        // LCG to get a value in [0, 1)
        let r = lcg_f64(seed);
        if r < prob0 { 0 } else { 1 }
    }

    /// ⟨Z⟩ = |α|² - |β|².
    pub fn expectation_z(&self) -> f64 {
        let p0 = self.alpha.0.powi(2) + self.alpha.1.powi(2);
        let p1 = self.beta.0.powi(2) + self.beta.1.powi(2);
        p0 - p1
    }

    /// ⟨X⟩ = 2 Re(α β*) = 2(α_re·β_re + α_im·β_im).
    pub fn expectation_x(&self) -> f64 {
        2.0 * (self.alpha.0 * self.beta.0 + self.alpha.1 * self.beta.1)
    }
}

fn lcg_f64(seed: u64) -> f64 {
    let s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (s >> 11) as f64 / (1u64 << 53) as f64
}

/// The four maximally entangled Bell states.
#[derive(Debug, Clone, PartialEq)]
pub enum BellState {
    /// |Φ+⟩ = (|00⟩ + |11⟩)/√2
    PhiPlus,
    /// |Φ-⟩ = (|00⟩ - |11⟩)/√2
    PhiMinus,
    /// |Ψ+⟩ = (|01⟩ + |10⟩)/√2
    PsiPlus,
    /// |Ψ-⟩ = (|01⟩ - |10⟩)/√2
    PsiMinus,
}

impl BellState {
    /// Quantum correlator: ⟨A⊗B⟩ = -cos(angle_a - angle_b).
    pub fn correlator(&self, angle_a: f64, angle_b: f64) -> f64 {
        let delta = angle_a - angle_b;
        match self {
            BellState::PhiPlus => -delta.cos(),
            BellState::PhiMinus => delta.cos(),
            BellState::PsiPlus => -delta.cos(),
            BellState::PsiMinus => delta.cos(),
        }
    }
}

/// CHSH Bell inequality test.
pub struct ChshTest;

impl ChshTest {
    /// CHSH value: |E(a,b) - E(a,b') + E(a',b) + E(a',b')|.
    /// correlators = [E(a,b), E(a,b'), E(a',b), E(a',b')].
    pub fn chsh_value(correlators: [f64; 4]) -> f64 {
        (correlators[0] - correlators[1] + correlators[2] + correlators[3]).abs()
    }

    /// True if the CHSH value exceeds the classical bound of 2.
    pub fn violates_bell(chsh: f64) -> bool {
        chsh > 2.0
    }

    /// Tsirelson bound: 2√2.
    pub fn quantum_bound() -> f64 {
        2.0 * 2.0_f64.sqrt()
    }
}

/// Entanglement entropy calculators.
pub struct EntanglementEntropy;

impl EntanglementEntropy {
    /// Von Neumann entropy: -Σ p ln(p), skipping p ≤ 0.
    pub fn von_neumann(probs: &[f64]) -> f64 {
        probs
            .iter()
            .filter(|&&p| p > 0.0)
            .map(|&p| -p * p.ln())
            .sum()
    }

    /// Rényi entropy of order α: S_α = (1/(1-α)) ln(Σ p^α).
    /// For α = 1, returns von Neumann entropy.
    pub fn renyi(probs: &[f64], alpha: f64) -> f64 {
        if (alpha - 1.0).abs() < 1e-10 {
            return Self::von_neumann(probs);
        }
        let sum_pa: f64 = probs.iter().filter(|&&p| p > 0.0).map(|&p| p.powf(alpha)).sum();
        if sum_pa <= 0.0 {
            return 0.0;
        }
        sum_pa.ln() / (1.0 - alpha)
    }

    /// Mutual information: I(A:B) = S(A) + S(B) - S(AB).
    /// joint[i][j] = P(A=i, B=j).
    pub fn mutual_information(joint: &[Vec<f64>]) -> f64 {
        let na = joint.len();
        if na == 0 {
            return 0.0;
        }
        let nb = joint[0].len();

        // Marginal P(A=i)
        let mut p_a = vec![0.0f64; na];
        // Marginal P(B=j)
        let mut p_b = vec![0.0f64; nb];
        let mut p_ab = Vec::new();

        for (i, row) in joint.iter().enumerate() {
            for (j, &p) in row.iter().enumerate() {
                p_a[i] += p;
                if j < nb {
                    p_b[j] += p;
                }
                p_ab.push(p);
            }
        }

        let s_a = Self::von_neumann(&p_a);
        let s_b = Self::von_neumann(&p_b);
        let s_ab = Self::von_neumann(&p_ab);
        (s_a + s_b - s_ab).max(0.0)
    }
}

/// Market entanglement analysis.
pub struct MarketEntanglement;

impl MarketEntanglement {
    /// Map a Pearson correlation in [-1, 1] to an entanglement entropy in [0, 1].
    /// Uses the formula: E = -((1+|ρ|)/2) ln((1+|ρ|)/2) - ((1-|ρ|)/2) ln((1-|ρ|)/2)
    /// (two-qubit Werner state analogy), normalized to [0, 1].
    pub fn correlation_to_entanglement(corr: f64) -> f64 {
        let r = corr.abs().min(1.0);
        let p = (1.0 + r) / 2.0;
        let q = 1.0 - p;
        let entropy = if p > 1e-15 && q > 1e-15 {
            -p * p.ln() - q * q.ln()
        } else {
            0.0
        };
        // Max entropy of a 2-state system is ln(2)
        entropy / std::f64::consts::LN_2
    }

    /// Compute a Bell-violation proxy from two return series.
    ///
    /// Uses sign correlations at lags 0, 1, 2, 3 as "measurement angles"
    /// and returns a CHSH-style value.
    pub fn compute_market_bell_violation(returns_a: &[f64], returns_b: &[f64]) -> f64 {
        let n = returns_a.len().min(returns_b.len());
        if n < 4 {
            return 0.0;
        }

        // Treat each lag as a "measurement setting"
        let e = |lag_a: usize, lag_b: usize| -> f64 {
            let mut sum = 0.0;
            let mut cnt = 0usize;
            let end = n.saturating_sub(lag_a.max(lag_b));
            for i in 0..end {
                let a_sign = returns_a[i + lag_a].signum();
                let b_sign = returns_b[i + lag_b].signum();
                sum += a_sign * b_sign;
                cnt += 1;
            }
            if cnt == 0 { 0.0 } else { sum / cnt as f64 }
        };

        // CHSH settings: (a=lag0, a'=lag1, b=lag0, b'=lag2)
        let e_ab = e(0, 0);
        let e_abp = e(0, 2);
        let e_apb = e(1, 0);
        let e_apbp = e(1, 2);

        ChshTest::chsh_value([e_ab, e_abp, e_apb, e_apbp])
    }

    /// Compute pairwise entanglement entropy matrix from a matrix of return series.
    /// return_matrix[asset_idx][time_idx].
    pub fn entanglement_network(return_matrix: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = return_matrix.len();
        let mut result = vec![vec![0.0f64; n]; n];

        for i in 0..n {
            for j in 0..n {
                if i == j {
                    result[i][j] = 0.0;
                    continue;
                }
                let ri = &return_matrix[i];
                let rj = &return_matrix[j];
                let len = ri.len().min(rj.len());
                if len == 0 {
                    continue;
                }
                // Pearson correlation
                let mean_i = ri.iter().take(len).sum::<f64>() / len as f64;
                let mean_j = rj.iter().take(len).sum::<f64>() / len as f64;
                let mut cov = 0.0f64;
                let mut var_i = 0.0f64;
                let mut var_j = 0.0f64;
                for k in 0..len {
                    let di = ri[k] - mean_i;
                    let dj = rj[k] - mean_j;
                    cov += di * dj;
                    var_i += di * di;
                    var_j += dj * dj;
                }
                let denom = (var_i * var_j).sqrt();
                let corr = if denom > 1e-15 { cov / denom } else { 0.0 };
                result[i][j] = Self::correlation_to_entanglement(corr);
            }
        }
        result
    }
}
