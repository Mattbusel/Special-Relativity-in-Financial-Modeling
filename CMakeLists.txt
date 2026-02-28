cmake_minimum_required(VERSION 3.20)
project(SpecialRelativityFinance VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ─── Compiler Flags ───────────────────────────────────────────────────────────
add_compile_options(
    -Wall -Wextra -Werror
    -Wpedantic
    -Wno-unused-parameter
)

# Sanitizers for Debug builds
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    add_compile_options(-fsanitize=address,undefined)
    add_link_options(-fsanitize=address,undefined)
endif()

# ─── Dependencies ─────────────────────────────────────────────────────────────
find_package(Eigen3 3.4 REQUIRED NO_MODULE)
find_package(GTest REQUIRED)
find_package(fmt REQUIRED)

# ─── Include Directories ──────────────────────────────────────────────────────
include_directories(include)

# ─── Module Libraries ─────────────────────────────────────────────────────────

# AGT-01: Lorentz Transform Engine
add_library(srfm_lorentz
    src/lorentz/lorentz_transform.cpp
    src/lorentz/beta_calculator.cpp
)
target_include_directories(srfm_lorentz PUBLIC include)
target_link_libraries(srfm_lorentz PUBLIC Eigen3::Eigen fmt::fmt)

# AGT-02: Spacetime Market Manifold
add_library(srfm_manifold
    src/manifold/market_manifold.cpp
    src/manifold/spacetime_interval.cpp
)
target_include_directories(srfm_manifold PUBLIC include)
target_link_libraries(srfm_manifold PUBLIC srfm_lorentz Eigen3::Eigen)

# AGT-03: Momentum-Velocity Signal Processor
add_library(srfm_momentum
    src/momentum/momentum_processor.cpp
    src/momentum/relativistic_signal.cpp
)
target_include_directories(srfm_momentum PUBLIC include)
target_link_libraries(srfm_momentum PUBLIC srfm_manifold)

# AGT-04: Tensor Calculus & Covariance Engine
add_library(srfm_tensor
    src/tensor/metric_tensor.cpp
    src/tensor/christoffel.cpp
    src/tensor/geodesic.cpp
)
target_include_directories(srfm_tensor PUBLIC include)
target_link_libraries(srfm_tensor PUBLIC srfm_manifold Eigen3::Eigen)

# AGT-05: Relativistic Backtester
add_library(srfm_backtest
    src/backtest/backtester.cpp
    src/backtest/performance_metrics.cpp
)
target_include_directories(srfm_backtest PUBLIC include)
target_link_libraries(srfm_backtest PUBLIC srfm_momentum srfm_tensor)

# AGT-06: Core / Integration
add_library(srfm_core
    src/core/engine.cpp
    src/core/data_loader.cpp
)
target_include_directories(srfm_core PUBLIC include)
target_link_libraries(srfm_core PUBLIC
    srfm_lorentz srfm_manifold srfm_momentum srfm_tensor srfm_backtest
)

# ─── Main Executable ──────────────────────────────────────────────────────────
add_executable(srfm src/main.cpp)
target_link_libraries(srfm PRIVATE srfm_core fmt::fmt)

# ─── Tests ────────────────────────────────────────────────────────────────────
enable_testing()

macro(add_srfm_test name sources libs)
    add_executable(${name} ${sources})
    target_link_libraries(${name} PRIVATE ${libs} GTest::gtest GTest::gtest_main)
    add_test(NAME ${name} COMMAND ${name})
endmacro()

# AGT-01 tests
add_srfm_test(test_lorentz
    "tests/lorentz/test_lorentz_transform.cpp;tests/lorentz/test_beta_calculator.cpp"
    srfm_lorentz)

# AGT-02 tests
add_srfm_test(test_manifold
    "tests/manifold/test_market_manifold.cpp;tests/manifold/test_spacetime_interval.cpp"
    srfm_manifold)

# AGT-03 tests
add_srfm_test(test_momentum
    "tests/momentum/test_momentum_processor.cpp;tests/momentum/test_relativistic_signal.cpp"
    srfm_momentum)

# AGT-04 tests
add_srfm_test(test_tensor
    "tests/tensor/test_metric_tensor.cpp;tests/tensor/test_christoffel.cpp;tests/tensor/test_geodesic.cpp"
    srfm_tensor)

# AGT-05 tests
add_srfm_test(test_backtest
    "tests/backtest/test_backtester.cpp;tests/backtest/test_performance_metrics.cpp;tests/backtest/test_metrics_precision.cpp"
    srfm_backtest)

# AGT-06 integration tests
add_srfm_test(test_integration
    "tests/integration/test_full_pipeline.cpp"
    srfm_core)
