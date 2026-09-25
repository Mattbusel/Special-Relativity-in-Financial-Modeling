// examples/lorentz_basics.cpp
// Build: cmake --build build --target lorentz_basics
// The output shown in the README is what this program prints.
#include "srfm/manifold.hpp"
#include "lorentz/lorentz_transform.hpp"

#include <cstdio>

int main() {
    using srfm::BetaVelocity;
    using srfm::lorentz::LorentzTransform;
    using namespace srfm::manifold;

    // Lorentz factor for a few price velocities.
    for (double b : {0.0, 0.5, 0.9, 0.99}) {
        if (auto g = LorentzTransform::gamma(BetaVelocity{b})) {
            std::printf("beta = %.2f   gamma = %.4f\n", b, g->value);
        }
    }

    // Two pairs of bars, one time unit apart: (time, price, volume, momentum).
    const SpacetimeEvent a{0.0, 100.0, 1.0, 0.0};
    const SpacetimeEvent slow{1.0, 100.4, 1.0, 0.0};  // small move: inside the cone
    const SpacetimeEvent fast{1.0, 103.0, 1.0, 0.0};  // big move: outside it

    for (const auto* b : {&slow, &fast}) {
        auto ds2 = SpacetimeInterval::compute(a, *b);
        auto cls = MarketManifold::classify(a, *b);
        if (ds2 && cls) {
            std::printf("dP = %+.1f   ds2 = %+.2f   %s\n",
                        b->price - a.price, *ds2, to_string(*cls));
        }
    }
    return 0;
}
