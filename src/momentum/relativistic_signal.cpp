/// @file src/momentum/relativistic_signal.cpp
/// @brief RelativisticMomentum signal utilities — AGT-06 stub.
///
/// This translation unit is intentionally minimal: the RelativisticMomentum
/// struct and all core logic live in momentum_processor.cpp and the
/// LorentzTransform engine.  This file provides the free-function helpers
/// that operate on RelativisticMomentum values after they have been produced
/// by MomentumProcessor::process().

#include "srfm/momentum.hpp"

#include <cmath>
#include <numeric>

namespace srfm::momentum {

// There are no additional free functions to define in this translation unit.
// The translation unit exists to satisfy the CMakeLists.txt target and to
// serve as the location for future signal-level helpers (e.g. serialisation,
// statistics over a RelativisticMomentum series).
//
// This intentional stub follows the AGT-06 integration mandate:
//   "AGT-02 and AGT-03 source files are absent; implement the manifold-interval
//    math and momentum pipeline inline within the core engine, wiring against
//    the existing srfm_lorentz and srfm_tensor libraries."

}  // namespace srfm::momentum
