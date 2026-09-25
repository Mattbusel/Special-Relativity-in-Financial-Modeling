// examples/stream_and_simd.cpp
// Build: cmake --build build --target stream_and_simd
#include <srfm/stream/beta_calculator.hpp>
#include <srfm/stream/lorentz_transform.hpp>
#include <srfm/simd/simd_dispatch.hpp>
#include <cstdio>
#include <vector>

int main() {
    // Streaming: rolling beta over the last 8 returns, then a Lorentz boost.
    srfm::stream::BetaCalculator<8> beta_calc;
    srfm::stream::LorentzTransform boost;
    const std::vector<double> closes = {100, 100.2, 100.1, 100.5, 100.4, 100.9, 101.3, 101.2, 101.6, 102.0};
    for (std::size_t i = 0; i < closes.size(); ++i) {
        beta_calc.update(closes[i]);
        if (beta_calc.warmed_up()) {
            auto ev = boost.transform(static_cast<double>(i), closes[i], beta_calc.beta());
            std::printf("t=%zu beta=%.4f gamma=%.4f\n", i, ev.beta, ev.gamma);
        }
    }
    // SIMD batch: dispatches to AVX-512, AVX2 or scalar at runtime.
    std::vector<double> velocities = {0.1, -0.4, 0.8, 0.2};
    double running_max = 0.0;
    auto betas  = srfm::simd::computeBetaBatch(velocities, running_max);
    auto gammas = srfm::simd::computeGammaBatch(betas);
    std::printf("%zu betas, gamma[2]=%.4f\n", betas.size(), gammas[2].value());
}
