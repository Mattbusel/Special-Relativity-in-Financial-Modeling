/// @file src/lorentz_portfolio.cpp
/// @brief Lorentz Portfolio Transformation — implementation.
///
/// All non-trivial logic lives in the header (template-free inline functions).
/// This translation unit exists to satisfy the CMake static library target and
/// provides the out-of-line definitions that are not inlineable or that need a
/// single compilation unit for ODR purposes.

#include "srfm/lorentz_portfolio.hpp"

// All logic is header-only (inline / constexpr).
// This file intentionally left mostly empty — it ensures the library target
// compiles even without a source file that exports symbols, and provides a
// translation unit where future non-header logic can live.

namespace srfm::portfolio {

// Explicit instantiation guards (none needed for current all-inline design).

} // namespace srfm::portfolio
