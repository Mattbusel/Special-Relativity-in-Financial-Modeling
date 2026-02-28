#include <gtest/gtest.h>
#include "../../src/lorentz/lorentz_transform.hpp"
#include "srfm/constants.hpp"
#include <cmath>
#include <limits>

using namespace srfm;
using namespace srfm::lorentz;
using namespace srfm::constants;

// ─── isValidBeta ──────────────────────────────────────────────────────────────

TEST(LorentzValidBeta, Zero_IsValid) {
    EXPECT_TRUE(LorentzTransform::isValidBeta(0.0));
}

TEST(LorentzValidBeta, SmallPositive_IsValid) {
    EXPECT_TRUE(LorentzTransform::isValidBeta(0.5));
}

TEST(LorentzValidBeta, BetaMaxSafe_IsInvalid) {
    // BETA_MAX_SAFE itself is the boundary — must be excluded
    EXPECT_FALSE(LorentzTransform::isValidBeta(BETA_MAX_SAFE));
}

TEST(LorentzValidBeta, JustBelowMaxSafe_IsValid) {
    EXPECT_TRUE(LorentzTransform::isValidBeta(BETA_MAX_SAFE - 1e-10));
}

TEST(LorentzValidBeta, ExactlyOne_IsInvalid) {
    EXPECT_FALSE(LorentzTransform::isValidBeta(1.0));
}

TEST(LorentzValidBeta, GreaterThanOne_IsInvalid) {
    EXPECT_FALSE(LorentzTransform::isValidBeta(1.5));
}

TEST(LorentzValidBeta, NegativeBeta_IsValid) {
    // β can be negative (market moving in reverse direction)
    EXPECT_TRUE(LorentzTransform::isValidBeta(-0.5));
}

TEST(LorentzValidBeta, NaN_IsInvalid) {
    EXPECT_FALSE(LorentzTransform::isValidBeta(
        std::numeric_limits<double>::quiet_NaN()));
}

TEST(LorentzValidBeta, PosInfinity_IsInvalid) {
    EXPECT_FALSE(LorentzTransform::isValidBeta(
        std::numeric_limits<double>::infinity()));
}

TEST(LorentzValidBeta, NegInfinity_IsInvalid) {
    EXPECT_FALSE(LorentzTransform::isValidBeta(
        -std::numeric_limits<double>::infinity()));
}

// ─── gamma: Newtonian Limit ───────────────────────────────────────────────────

TEST(LorentzGamma, NewtonianLimit_BetaZero_GammaIsOne) {
    // γ(0) = 1/√(1−0) = 1 exactly
    auto result = LorentzTransform::gamma(BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.0, FLOAT_EPSILON);
}

TEST(LorentzGamma, NewtonianLimit_SmallBeta_NearOne) {
    // β = 0.001 → γ ≈ 1.0000005 (≪ 0.001% correction)
    auto result = LorentzTransform::gamma(BetaVelocity{0.001});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.0, 1e-5);
    EXPECT_GT(result->value, 1.0); // always strictly ≥ 1
}

TEST(LorentzGamma, NewtonianThreshold_StillAboveOne) {
    auto result = LorentzTransform::gamma(BetaVelocity{BETA_NEWTONIAN_THRESHOLD});
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value, 1.0);
    // Exact value: 1/√(1−0.01)
    EXPECT_NEAR(result->value, 1.0 / std::sqrt(1.0 - 0.01), FLOAT_EPSILON);
}

// ─── gamma: Relativistic Regime ──────────────────────────────────────────────

TEST(LorentzGamma, Beta06_ExactValue_1p25) {
    // β = 0.6 → γ = 1/√(1−0.36) = 1/0.8 = 1.25 exactly
    auto result = LorentzTransform::gamma(BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.25, 1e-10);
}

TEST(LorentzGamma, Beta08_ExactValue) {
    // β = 0.8 → γ = 1/√(1−0.64) = 1/0.6 = 5/3
    auto result = LorentzTransform::gamma(BetaVelocity{0.8});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 5.0 / 3.0, 1e-10);
}

TEST(LorentzGamma, Beta09999_VeryLarge) {
    // β = 0.9999 → γ ≈ 70.7
    auto result = LorentzTransform::gamma(BetaVelocity{0.9999});
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(result->value, 70.0);
    EXPECT_LT(result->value, 1000.0);
}

TEST(LorentzGamma, AlwaysAtLeastOne) {
    for (double b : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99, 0.999}) {
        auto result = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(result.has_value()) << "nullopt at beta=" << b;
        EXPECT_GE(result->value, 1.0) << "gamma < 1 at beta=" << b;
    }
}

TEST(LorentzGamma, SymmetricInBeta) {
    // γ depends only on β², so γ(β) = γ(−β)
    for (double b : {0.2, 0.5, 0.8}) {
        auto gpos = LorentzTransform::gamma(BetaVelocity{b});
        auto gneg = LorentzTransform::gamma(BetaVelocity{-b});
        ASSERT_TRUE(gpos.has_value());
        ASSERT_TRUE(gneg.has_value());
        EXPECT_NEAR(gpos->value, gneg->value, FLOAT_EPSILON)
            << "Asymmetry at beta=" << b;
    }
}

// ─── gamma: Invalid Inputs ────────────────────────────────────────────────────

TEST(LorentzGamma, BetaOne_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::gamma(BetaVelocity{1.0}).has_value());
}

TEST(LorentzGamma, BetaGreaterThanOne_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::gamma(BetaVelocity{1.5}).has_value());
}

TEST(LorentzGamma, BetaNaN_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::gamma(
        BetaVelocity{std::numeric_limits<double>::quiet_NaN()}).has_value());
}

TEST(LorentzGamma, BetaInfinity_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::gamma(
        BetaVelocity{std::numeric_limits<double>::infinity()}).has_value());
}

// ─── dilateTime ───────────────────────────────────────────────────────────────

TEST(LorentzTimeDilation, ZeroBeta_NoDilation) {
    // t = γ·τ, γ(0) = 1, so t = τ
    auto result = LorentzTransform::dilateTime(100.0, BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 100.0, FLOAT_EPSILON);
}

TEST(LorentzTimeDilation, Beta06_DilatesByGamma) {
    // γ(0.6) = 1.25, so t = 100 × 1.25 = 125
    auto result = LorentzTransform::dilateTime(100.0, BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 125.0, 1e-8);
}

TEST(LorentzTimeDilation, ZeroProperTime_StaysZero) {
    auto result = LorentzTransform::dilateTime(0.0, BetaVelocity{0.9});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(LorentzTimeDilation, NegativeProperTime_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::dilateTime(-1.0, BetaVelocity{0.5}).has_value());
}

TEST(LorentzTimeDilation, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::dilateTime(10.0, BetaVelocity{2.0}).has_value());
}

TEST(LorentzTimeDilation, AlwaysAtLeastProperTime) {
    // Dilation never compresses time (γ ≥ 1)
    double tau = 42.0;
    for (double b : {0.0, 0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto result = LorentzTransform::dilateTime(tau, BetaVelocity{b});
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(*result, tau) << "Dilation compressed time at beta=" << b;
    }
}

TEST(LorentzTimeDilation, ScalesLinearly) {
    // dilateTime(k·τ, β) = k · dilateTime(τ, β)
    BetaVelocity beta{0.5};
    double tau = 10.0;
    auto t1 = LorentzTransform::dilateTime(tau, beta);
    auto t2 = LorentzTransform::dilateTime(2.0 * tau, beta);
    ASSERT_TRUE(t1.has_value());
    ASSERT_TRUE(t2.has_value());
    EXPECT_NEAR(*t2, 2.0 * (*t1), 1e-10);
}

// ─── applyMomentumCorrection ──────────────────────────────────────────────────

TEST(LorentzMomentum, ZeroBeta_NewtonianLimit) {
    // At β = 0, p = 1 · m · v = m · raw
    auto result = LorentzTransform::applyMomentumCorrection(
        2.0, BetaVelocity{0.0}, 3.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_value, 6.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->gamma.value, 1.0, FLOAT_EPSILON);
    EXPECT_NEAR(result->raw_value, 2.0, FLOAT_EPSILON);
}

TEST(LorentzMomentum, Beta06_AmplifiedByGamma) {
    // γ(0.6) = 1.25; p = 1.25 × 1.0 × 1.0 = 1.25
    auto result = LorentzTransform::applyMomentumCorrection(
        1.0, BetaVelocity{0.6}, 1.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->adjusted_value, 1.25, 1e-8);
    EXPECT_NEAR(result->gamma.value, 1.25, 1e-8);
}

TEST(LorentzMomentum, ZeroMass_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::applyMomentumCorrection(
        1.0, BetaVelocity{0.5}, 0.0).has_value());
}

TEST(LorentzMomentum, NegativeMass_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::applyMomentumCorrection(
        1.0, BetaVelocity{0.5}, -1.0).has_value());
}

TEST(LorentzMomentum, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::applyMomentumCorrection(
        1.0, BetaVelocity{1.5}, 1.0).has_value());
}

TEST(LorentzMomentum, AdjustedAtLeastNewtonianMomentum) {
    // γ ≥ 1, so p_rel ≥ p_classical for positive signal and mass
    double raw = 1.0, mass = 2.0;
    double classical = raw * mass;
    for (double b : {0.0, 0.1, 0.5, 0.9}) {
        auto result = LorentzTransform::applyMomentumCorrection(
            raw, BetaVelocity{b}, mass);
        ASSERT_TRUE(result.has_value());
        EXPECT_GE(result->adjusted_value, classical)
            << "Relativistic momentum below classical at beta=" << b;
    }
}

// ─── composeVelocities ────────────────────────────────────────────────────────

TEST(LorentzVelocityComposition, HalfPlusHalf_IsPointEight) {
    // 0.5 ⊕ 0.5 = 1.0/1.25 = 0.8, not 1.0
    auto result = LorentzTransform::composeVelocities(
        BetaVelocity{0.5}, BetaVelocity{0.5});
    EXPECT_NEAR(result.value, 0.8, 1e-10);
}

TEST(LorentzVelocityComposition, ZeroPlusAny_EqualsAny) {
    auto result = LorentzTransform::composeVelocities(
        BetaVelocity{0.0}, BetaVelocity{0.7});
    EXPECT_NEAR(result.value, 0.7, FLOAT_EPSILON);
}

TEST(LorentzVelocityComposition, AlwaysSubLuminal) {
    for (double b1 : {0.3, 0.5, 0.7, 0.9}) {
        for (double b2 : {0.3, 0.5, 0.7, 0.9}) {
            auto r = LorentzTransform::composeVelocities(
                BetaVelocity{b1}, BetaVelocity{b2});
            EXPECT_LT(r.value, 1.0)
                << "Superluminal at b1=" << b1 << " b2=" << b2;
        }
    }
}

TEST(LorentzVelocityComposition, IsCommutative) {
    auto r1 = LorentzTransform::composeVelocities(
        BetaVelocity{0.3}, BetaVelocity{0.6});
    auto r2 = LorentzTransform::composeVelocities(
        BetaVelocity{0.6}, BetaVelocity{0.3});
    EXPECT_NEAR(r1.value, r2.value, FLOAT_EPSILON);
}

TEST(LorentzVelocityComposition, NegativeBeta_SubLuminal) {
    // Opposite-direction velocities should partially cancel
    auto result = LorentzTransform::composeVelocities(
        BetaVelocity{0.7}, BetaVelocity{-0.7});
    EXPECT_LT(std::abs(result.value), 1.0);
}

// ─── inverseTransform ────────────────────────────────────────────────────────

TEST(LorentzInverse, RoundTrip_DilateAndRecover) {
    double proper = 42.0;
    BetaVelocity beta{0.6};

    auto dilated = LorentzTransform::dilateTime(proper, beta);
    ASSERT_TRUE(dilated.has_value());

    auto recovered = LorentzTransform::inverseTransform(*dilated, beta);
    ASSERT_TRUE(recovered.has_value());
    EXPECT_NEAR(*recovered, proper, 1e-9);
}

TEST(LorentzInverse, ZeroBeta_NoDivision) {
    // γ(0) = 1, so inverse returns value unchanged
    auto result = LorentzTransform::inverseTransform(50.0, BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 50.0, FLOAT_EPSILON);
}

TEST(LorentzInverse, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::inverseTransform(
        100.0, BetaVelocity{1.5}).has_value());
}

TEST(LorentzInverse, InverseCompressesTime) {
    // Inverse of dilation must give back a value ≤ dilated
    double proper = 10.0;
    BetaVelocity beta{0.8};
    auto dilated  = LorentzTransform::dilateTime(proper, beta);
    ASSERT_TRUE(dilated.has_value());
    auto recovered = LorentzTransform::inverseTransform(*dilated, beta);
    ASSERT_TRUE(recovered.has_value());
    EXPECT_LE(*recovered, *dilated);
}

// ─── contractLength ──────────────────────────────────────────────────────────

TEST(LorentzContractLength, ZeroBeta_NoContraction) {
    auto result = LorentzTransform::contractLength(10.0, BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 10.0, FLOAT_EPSILON);
}

TEST(LorentzContractLength, Beta06_ContractsByGamma) {
    // L = L₀/γ, γ(0.6) = 1.25, so L = 10/1.25 = 8
    auto result = LorentzTransform::contractLength(10.0, BetaVelocity{0.6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 8.0, 1e-8);
}

TEST(LorentzContractLength, AlwaysAtMostProperLength) {
    double L0 = 5.0;
    for (double b : {0.0, 0.1, 0.5, 0.9}) {
        auto result = LorentzTransform::contractLength(L0, BetaVelocity{b});
        ASSERT_TRUE(result.has_value());
        EXPECT_LE(*result, L0) << "Length exceeded proper length at beta=" << b;
    }
}

TEST(LorentzContractLength, AlwaysPositive) {
    auto result = LorentzTransform::contractLength(3.0, BetaVelocity{0.99});
    ASSERT_TRUE(result.has_value());
    EXPECT_GT(*result, 0.0);
}

TEST(LorentzContractLength, ZeroLength_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::contractLength(0.0, BetaVelocity{0.5}).has_value());
}

TEST(LorentzContractLength, NegativeLength_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::contractLength(-1.0, BetaVelocity{0.5}).has_value());
}

TEST(LorentzContractLength, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::contractLength(5.0, BetaVelocity{2.0}).has_value());
}

// ─── rapidity ────────────────────────────────────────────────────────────────

TEST(LorentzRapidity, ZeroBeta_ZeroRapidity) {
    auto result = LorentzTransform::rapidity(BetaVelocity{0.0});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 0.0, FLOAT_EPSILON);
}

TEST(LorentzRapidity, KnownValue_Beta_tanh1) {
    // β = tanh(1) ≈ 0.7616 → rapidity = 1.0
    double b = std::tanh(1.0);
    auto result = LorentzTransform::rapidity(BetaVelocity{b});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 1.0, 1e-10);
}

TEST(LorentzRapidity, NegativeBeta_NegativeRapidity) {
    auto rpos = LorentzTransform::rapidity(BetaVelocity{0.5});
    auto rneg = LorentzTransform::rapidity(BetaVelocity{-0.5});
    ASSERT_TRUE(rpos.has_value());
    ASSERT_TRUE(rneg.has_value());
    EXPECT_NEAR(*rneg, -*rpos, FLOAT_EPSILON);
}

TEST(LorentzRapidity, IsAdditiveUnderComposition) {
    // Key property: φ(β₁ ⊕ β₂) = φ(β₁) + φ(β₂)
    BetaVelocity b1{0.3}, b2{0.4};
    auto phi1     = LorentzTransform::rapidity(b1);
    auto phi2     = LorentzTransform::rapidity(b2);
    auto composed = LorentzTransform::composeVelocities(b1, b2);
    auto phi_comp = LorentzTransform::rapidity(composed);

    ASSERT_TRUE(phi1.has_value());
    ASSERT_TRUE(phi2.has_value());
    ASSERT_TRUE(phi_comp.has_value());
    EXPECT_NEAR(*phi_comp, *phi1 + *phi2, 1e-10);
}

TEST(LorentzRapidity, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::rapidity(BetaVelocity{1.0}).has_value());
    EXPECT_FALSE(LorentzTransform::rapidity(BetaVelocity{1.5}).has_value());
}

// ─── totalEnergy ─────────────────────────────────────────────────────────────

TEST(LorentzTotalEnergy, ZeroBeta_RestEnergyOnly) {
    // E = γ·m·c² = 1·m·1² = m at β=0
    auto result = LorentzTransform::totalEnergy(BetaVelocity{0.0}, 5.0);
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(*result, 5.0, FLOAT_EPSILON);
}

TEST(LorentzTotalEnergy, HighBeta_ExceedsRestEnergy) {
    double mass = 2.0;
    auto result = LorentzTransform::totalEnergy(BetaVelocity{0.6}, mass);
    ASSERT_TRUE(result.has_value());
    // E = γ · m · c² = 1.25 × 2.0 × 1.0 = 2.5
    EXPECT_NEAR(*result, 2.5, 1e-8);
    EXPECT_GT(*result, mass); // more energy than at rest
}

TEST(LorentzTotalEnergy, ZeroMass_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::totalEnergy(BetaVelocity{0.5}, 0.0).has_value());
}

TEST(LorentzTotalEnergy, NegativeMass_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::totalEnergy(BetaVelocity{0.5}, -1.0).has_value());
}

TEST(LorentzTotalEnergy, InvalidBeta_ReturnsNullopt) {
    EXPECT_FALSE(LorentzTransform::totalEnergy(BetaVelocity{2.0}, 1.0).has_value());
}

// ─── Mathematical Identities ──────────────────────────────────────────────────

TEST(LorentzIdentity, GammaSquared_Is_OneOverOneMinusBetaSquared) {
    for (double b : {0.1, 0.3, 0.5, 0.7, 0.9}) {
        auto g = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(g.has_value());
        double expected = 1.0 / (1.0 - b * b);
        EXPECT_NEAR(g->value * g->value, expected, 1e-8)
            << "γ² identity failed at beta=" << b;
    }
}

TEST(LorentzIdentity, GammaBeta_FourMomentumRelation) {
    // γβ = β/√(1−β²) — used in four-momentum calculations
    for (double b : {0.2, 0.4, 0.6, 0.8}) {
        auto g = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(g.has_value());
        double gamma_beta = g->value * b;
        double expected   = b / std::sqrt(1.0 - b * b);
        EXPECT_NEAR(gamma_beta, expected, 1e-10)
            << "γβ identity failed at beta=" << b;
    }
}

TEST(LorentzIdentity, DilationAndContraction_InversePair) {
    // dilateTime(τ, β) × contractLength(1, β) = τ/γ × γ = τ only if same γ
    // More precise: dilate * contract product = L₀ * τ (unchanged by γ²/γ²)
    double tau = 5.0, L0 = 3.0;
    BetaVelocity beta{0.8};
    auto dilated    = LorentzTransform::dilateTime(tau, beta);
    auto contracted = LorentzTransform::contractLength(L0, beta);
    auto g          = LorentzTransform::gamma(beta);
    ASSERT_TRUE(dilated.has_value());
    ASSERT_TRUE(contracted.has_value());
    ASSERT_TRUE(g.has_value());
    // dilated = γτ, contracted = L₀/γ → product = τL₀
    EXPECT_NEAR(*dilated * *contracted, tau * L0, 1e-8);
}

TEST(LorentzIdentity, EnergyMinusKineticIsRestEnergy) {
    // E_total - E_kinetic = m·c² (rest energy)
    // We'll verify through totalEnergy formula: E_total = γ·m·c²
    // and kineticEnergy is handled by BetaCalculator, but we can verify here
    // E = γ·m. Rest = m. E - Rest = (γ-1)·m.
    double mass = 4.0;
    BetaVelocity beta{0.6};
    auto E = LorentzTransform::totalEnergy(beta, mass);
    auto g = LorentzTransform::gamma(beta);
    ASSERT_TRUE(E.has_value());
    ASSERT_TRUE(g.has_value());
    double rest = mass; // c_market = 1 by default
    double kinetic_expected = (g->value - 1.0) * mass;
    EXPECT_NEAR(*E - rest, kinetic_expected, 1e-8);
}

// ─── Numerical Precision ──────────────────────────────────────────────────────

TEST(LorentzPrecision, VerySmallBeta_GammaCloseToOne) {
    // For β = 1e-6: γ = 1/√(1−1e-12) ≈ 1 + 5e-13
    auto result = LorentzTransform::gamma(BetaVelocity{1e-6});
    ASSERT_TRUE(result.has_value());
    EXPECT_NEAR(result->value, 1.0, 1e-9);
    EXPECT_GE(result->value, 1.0);
}

TEST(LorentzPrecision, GammaMonotonicallyIncreasing) {
    // γ must increase strictly as β increases
    std::vector<double> betas = {0.0, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99};
    double prev_gamma = 0.0;
    for (double b : betas) {
        auto g = LorentzTransform::gamma(BetaVelocity{b});
        ASSERT_TRUE(g.has_value());
        EXPECT_GT(g->value, prev_gamma)
            << "gamma not monotone at beta=" << b;
        prev_gamma = g->value;
    }
}
