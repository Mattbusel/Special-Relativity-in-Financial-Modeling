# cmake/tests.cmake
# All test executables and CTest registrations.

# ── Existing momentum unit tests ──────────────────────────────────────────────
add_executable(test_momentum
    tests/momentum/test_momentum.cpp
)
target_include_directories(test_momentum PRIVATE src tests)
target_link_libraries(test_momentum PRIVATE srfm_momentum)
add_test(NAME MomentumUnitTests COMMAND test_momentum)

# ── Edge-case and boundary tests (require Eigen3 via lorentz headers) ─────────
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
add_executable(test_edge_cases
    tests/momentum/test_edge_cases.cpp
)
target_include_directories(test_edge_cases PRIVATE src include tests/momentum)
target_link_libraries(test_edge_cases PRIVATE
    srfm_momentum
    srfm_beta_calculator
    srfm_manifold
    srfm_geodesic
    srfm_lorentz
    Eigen3::Eigen
)
add_test(NAME EdgeCaseBoundaryTests COMMAND test_edge_cases)
endif()

# ── Lorentz invariants and numerical edge-case tests ─────────────────────────
add_executable(test_lorentz_invariants
    tests/momentum/test_lorentz_invariants.cpp
)
target_include_directories(test_lorentz_invariants PRIVATE src include tests/momentum)
target_link_libraries(test_lorentz_invariants PRIVATE
    srfm_momentum
    srfm_manifold
    srfm_geodesic
)
add_test(NAME LorentzInvariantTests COMMAND test_lorentz_invariants)

# ── Lorentz transform and beta calculator unit tests ─────────────────────────
add_executable(test_lorentz_transform
    tests/lorentz/test_lorentz_transform.cpp
)
target_include_directories(test_lorentz_transform PRIVATE src include)
target_link_libraries(test_lorentz_transform PRIVATE
    srfm_lorentz
    GTest::gtest_main
)
add_test(NAME LorentzTransformTests COMMAND test_lorentz_transform)

add_executable(test_beta_calculator
    tests/lorentz/test_beta_calculator.cpp
)
target_include_directories(test_beta_calculator PRIVATE src include)
target_link_libraries(test_beta_calculator PRIVATE
    srfm_lorentz
    GTest::gtest_main
)
add_test(NAME BetaCalculatorTests COMMAND test_beta_calculator)

add_executable(test_online_beta
    tests/lorentz/test_online_beta.cpp
)
target_include_directories(test_online_beta PRIVATE src include)
target_link_libraries(test_online_beta PRIVATE
    srfm_lorentz
    GTest::gtest_main
)
add_test(NAME OnlineBetaTests COMMAND test_online_beta)

# ── Event-Driven Backtester unit tests (Round 3) ──────────────────────────────
add_executable(test_event_backtester
    tests/backtest/test_event_backtester.cpp
)
target_include_directories(test_event_backtester PRIVATE src include)
target_link_libraries(test_event_backtester PRIVATE
    srfm_event_backtest
    GTest::gtest_main
)
add_test(NAME EventBacktesterTests COMMAND test_event_backtester)

# ── Backtest unit tests ───────────────────────────────────────────────────────
add_executable(test_backtester
    tests/backtest/test_backtester.cpp
)
target_include_directories(test_backtester PRIVATE src include)
target_link_libraries(test_backtester PRIVATE
    srfm_backtest
    GTest::gtest_main
)
add_test(NAME BacktesterTests COMMAND test_backtester)

add_executable(test_performance_metrics
    tests/backtest/test_performance_metrics.cpp
)
target_include_directories(test_performance_metrics PRIVATE src include)
target_link_libraries(test_performance_metrics PRIVATE
    srfm_backtest
    GTest::gtest_main
)
add_test(NAME PerformanceMetricsTests COMMAND test_performance_metrics)

add_executable(test_gamma_sizing
    tests/backtest/test_gamma_sizing.cpp
)
target_include_directories(test_gamma_sizing PRIVATE src include)
target_link_libraries(test_gamma_sizing PRIVATE
    srfm_backtest
    GTest::gtest_main
)
add_test(NAME GammaSizingTests COMMAND test_gamma_sizing)

add_executable(test_metrics_precision
    tests/backtest/test_metrics_precision.cpp
)
target_include_directories(test_metrics_precision PRIVATE src include)
target_link_libraries(test_metrics_precision PRIVATE
    srfm_backtest
    GTest::gtest_main
)
add_test(NAME MetricsPrecisionTests COMMAND test_metrics_precision)

# ── Integration tests ─────────────────────────────────────────────────────────
# NOTE: test_full_pipeline and test_error_handling reference srfm::core::Engine
# (declared in include/srfm/engine.hpp) which currently has no implementation
# .cpp file. These tests are excluded from the default build until the core
# engine implementation is provided in src/core/engine.cpp.
# To include them: cmake -DSRFM_BUILD_INTEGRATION_TESTS=ON ...
option(SRFM_BUILD_INTEGRATION_TESTS
    "Build integration tests that require srfm::core::Engine implementation" ON)

if(SRFM_BUILD_INTEGRATION_TESTS)
    add_executable(test_full_pipeline
        tests/integration/test_full_pipeline.cpp
    )
    target_include_directories(test_full_pipeline PRIVATE src include)
    target_link_libraries(test_full_pipeline PRIVATE
        srfm_backtest
        srfm_lorentz
        srfm_tensor
        GTest::gtest_main
    )
    add_test(NAME FullPipelineIntegrationTests COMMAND test_full_pipeline)

    add_executable(test_error_handling
        tests/integration/test_error_handling.cpp
    )
    target_include_directories(test_error_handling PRIVATE src include)
    target_link_libraries(test_error_handling PRIVATE
        srfm_backtest
        srfm_lorentz
        srfm_tensor
        GTest::gtest_main
    )
    add_test(NAME ErrorHandlingIntegrationTests COMMAND test_error_handling)
endif()

# ── Tensor / geodesic unit tests ──────────────────────────────────────────────
add_executable(test_metric_tensor
    tests/tensor/test_metric_tensor.cpp
)
target_include_directories(test_metric_tensor PRIVATE src include)
target_link_libraries(test_metric_tensor PRIVATE
    srfm_tensor
    GTest::gtest_main
)
add_test(NAME MetricTensorTests COMMAND test_metric_tensor)

add_executable(test_christoffel
    tests/tensor/test_christoffel.cpp
)
target_include_directories(test_christoffel PRIVATE src include)
target_link_libraries(test_christoffel PRIVATE
    srfm_tensor
    GTest::gtest_main
)
add_test(NAME ChristoffelTests COMMAND test_christoffel)

add_executable(test_geodesic
    tests/tensor/test_geodesic.cpp
)
target_include_directories(test_geodesic PRIVATE src include)
target_link_libraries(test_geodesic PRIVATE
    srfm_tensor
    GTest::gtest_main
)
add_test(NAME GeodesicTests COMMAND test_geodesic)

# ── Geodesic solver branch-coverage tests ─────────────────────────────────────
add_executable(test_geodesic_solver_branches
    tests/geodesic/test_geodesic_solver_branches.cpp
)
target_include_directories(test_geodesic_solver_branches PRIVATE src include)
target_link_libraries(test_geodesic_solver_branches PRIVATE
    srfm_geodesic
    srfm_backtest
    GTest::gtest_main
)
add_test(NAME GeodesicSolverBranchTests COMMAND test_geodesic_solver_branches)

# ── N-asset tests (each file has its own main()) ─────────────────────────────
set(_NASSET_TESTS
    test_n_asset_interval
    test_n_asset_manifold
    test_christoffel_n
    test_geodesic_n
    test_n_asset_engine
)
foreach(_nat IN LISTS _NASSET_TESTS)
    add_executable(${_nat} tests/n_asset/${_nat}.cpp)
    target_include_directories(${_nat} PRIVATE src include tests/n_asset)
    target_link_libraries(${_nat} PRIVATE srfm_tensor srfm_engine)
    add_test(NAME ${_nat} COMMAND ${_nat})
endforeach()

# ── Stream processing tests (each file has its own main()) ───────────────────
set(_STREAM_TESTS
    test_beta_calculator
    test_coordinate_normalizer
    test_lorentz_manifold
    test_pipeline_invariants
    test_signal_chain
    test_spsc_ring
    test_spsc_stress
    test_stream_engine
    test_tick_validation
)
foreach(_st IN LISTS _STREAM_TESTS)
    add_executable(stream_${_st} tests/stream/${_st}.cpp)
    target_include_directories(stream_${_st} PRIVATE src include tests/stream)
    target_link_libraries(stream_${_st} PRIVATE srfm_stream)
    # Enable QueueTickSource (test-only class guarded by SRFM_TESTING).
    target_compile_definitions(stream_${_st} PRIVATE SRFM_TESTING=1)
    add_test(NAME stream_${_st} COMMAND stream_${_st})
endforeach()

# ── Interval gap tests (symmetry, numerical stability, Lorentz invariance) ───
if(GTest_FOUND)
    add_executable(test_interval_gaps
        tests/manifold/test_interval_gaps.cpp
    )
    target_include_directories(test_interval_gaps PRIVATE src include)
    target_link_libraries(test_interval_gaps PRIVATE
        srfm_manifold
        GTest::gtest_main
    )
    add_test(NAME IntervalGapTests COMMAND test_interval_gaps)
else()
    message(STATUS
        "GTest not found -- test_interval_gaps will not be built. "
        "Install via vcpkg: vcpkg install gtest"
    )
endif()

# ── Property-based tests (RapidCheck) ────────────────────────────────────────
if(RapidCheck_FOUND)
    set(PROPERTY_TESTS
        prop_lorentz_identity
        prop_rapidity_additivity
        prop_doppler_inverse
        prop_velocity_subluminal
        prop_christoffel_flat
        prop_geodesic_flat
        prop_interval_invariant
        prop_metric_positive_definite
        prop_christoffel_symmetry
        prop_lorentz_invariance
    )

    foreach(test_name IN LISTS PROPERTY_TESTS)
        add_executable(${test_name}
            test/property/${test_name}.cpp
        )
        target_include_directories(${test_name} PRIVATE src)
        target_link_libraries(${test_name} PRIVATE
            srfm_momentum
            srfm_beta_calculator
            srfm_manifold
            srfm_geodesic
            rapidcheck
        )
        # 10,000 inputs per property via RC_PARAMS
        add_test(NAME ${test_name}
            COMMAND ${test_name}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
        set_tests_properties(${test_name} PROPERTIES
            ENVIRONMENT "RC_PARAMS=max_success=10000"
        )
    endforeach()
else()
    message(WARNING
        "RapidCheck not found — property tests will not be built.\n"
        "Install via vcpkg: vcpkg install rapidcheck"
    )
endif()

# ── Fuzz targets (libFuzzer, Clang only) ─────────────────────────────────────
option(SRFM_FUZZ "Build libFuzzer fuzz targets (requires Clang with -fsanitize=fuzzer)" OFF)

if(SRFM_FUZZ)
    if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        message(FATAL_ERROR "Fuzzing requires Clang (CMAKE_CXX_COMPILER_ID=Clang)")
    endif()

    set(FUZZ_FLAGS -fsanitize=fuzzer,address,undefined)

    set(FUZZ_TARGETS
        fuzz_beta_calculator
        fuzz_manifold
        fuzz_geodesic
        fuzz_engine
    )

    foreach(fuzz_name IN LISTS FUZZ_TARGETS)
        add_executable(${fuzz_name}
            fuzz/${fuzz_name}.cpp
        )
        target_include_directories(${fuzz_name} PRIVATE src)
        target_compile_options(${fuzz_name} PRIVATE ${FUZZ_FLAGS})
        target_link_options(${fuzz_name} PRIVATE ${FUZZ_FLAGS})
        target_link_libraries(${fuzz_name} PRIVATE
            srfm_momentum
            srfm_beta_calculator
            srfm_manifold
            srfm_geodesic
            srfm_engine
        )
    endforeach()
endif()

# ── SIMD unit tests (AGT-08) ──────────────────────────────────────────────────

add_executable(test_simd
    tests/momentum/test_simd.cpp
)
target_include_directories(test_simd PRIVATE
    ${SRFM_SIMD_INCLUDE_DIRS}
    tests/momentum          # for srfm_test.hpp
)
target_link_libraries(test_simd PRIVATE srfm_simd_dispatch)
add_test(NAME SimdAccelerationTests COMMAND test_simd)

# ── Dual-number Christoffel symbols (Task 3) ──────────────────────────────────

add_executable(test_christoffel_dual
    tests/tensor/test_christoffel_dual.cpp
)
target_include_directories(test_christoffel_dual PRIVATE src include)
target_link_libraries(test_christoffel_dual PRIVATE
    srfm_tensor
    GTest::gtest_main
)
add_test(NAME ChristoffelDualTests COMMAND test_christoffel_dual)

# ── Metric singularity / Tikhonov regularization (Task 4) ────────────────────

add_executable(test_metric_singularity
    tests/tensor/test_metric_singularity.cpp
)
target_include_directories(test_metric_singularity PRIVATE src include)
target_link_libraries(test_metric_singularity PRIVATE
    srfm_tensor
    GTest::gtest_main
)
add_test(NAME MetricSingularityTests COMMAND test_metric_singularity)

# ── Portfolio Manifold & Relativistic Optimizer (Task 1 + Task 2) ────────────

add_executable(test_portfolio_manifold
    tests/portfolio/test_portfolio_manifold.cpp
)
target_include_directories(test_portfolio_manifold PRIVATE src include)
target_link_libraries(test_portfolio_manifold PRIVATE
    srfm_portfolio
    GTest::gtest_main
)
add_test(NAME PortfolioManifoldTests COMMAND test_portfolio_manifold)

add_executable(test_relativistic_optimizer
    tests/portfolio/test_relativistic_optimizer.cpp
)
target_include_directories(test_relativistic_optimizer PRIVATE src include)
target_link_libraries(test_relativistic_optimizer PRIVATE
    srfm_portfolio
    GTest::gtest_main
)
add_test(NAME RelativisticOptimizerTests COMMAND test_relativistic_optimizer)
