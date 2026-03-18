cmake_minimum_required(VERSION 3.25)
project(srfm_cpp
    VERSION 1.1.0
    DESCRIPTION "Special Relativity in Financial Modelling — C++ library"
    LANGUAGES CXX
)

# ── Standard and compiler options ─────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Warnings as errors (matches the existing build_momentum.* scripts)
if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror -Wno-unused-parameter)
endif()

# ── Optional dependencies ─────────────────────────────────────────────────────
find_package(RapidCheck CONFIG QUIET)
find_package(Eigen3 QUIET)

# If Eigen3 was not found via find_package, fall back to the vendored copy.
if(NOT (Eigen3_FOUND OR TARGET Eigen3::Eigen))
    set(_EIGEN_VENDOR_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
    if(EXISTS "${_EIGEN_VENDOR_DIR}/Eigen/Dense")
        message(STATUS "Using vendored Eigen3 from third_party/eigen")
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${_EIGEN_VENDOR_DIR};${CMAKE_SOURCE_DIR}"
        )
        set(Eigen3_FOUND TRUE)
    endif()
endif()

# GTest: used by test_interval_gaps and any other GTest-based suites.
# Try system package first; fall back to FetchContent so the build works
# without vcpkg or a pre-installed libgtest-dev.
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
    set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    # Alias so the downstream targets can use the same GTest::gtest_main name.
    if(NOT TARGET GTest::gtest_main)
        add_library(GTest::gtest_main ALIAS gtest_main)
    endif()
endif()
find_package(GTest CONFIG QUIET)

# ── Source library targets ────────────────────────────────────────────────────

add_library(srfm_momentum STATIC
    src/momentum/momentum.cpp
)
target_include_directories(srfm_momentum PUBLIC src)

add_library(srfm_beta_calculator STATIC
    src/beta_calculator/beta_calculator.cpp
)
target_include_directories(srfm_beta_calculator PUBLIC src include)
target_link_libraries(srfm_beta_calculator PUBLIC srfm_momentum)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_beta_calculator PUBLIC Eigen3::Eigen)
endif()

add_library(srfm_manifold STATIC
    src/manifold/spacetime_manifold.cpp
)
target_include_directories(srfm_manifold PUBLIC src include)
target_link_libraries(srfm_manifold PUBLIC srfm_momentum)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_manifold PUBLIC Eigen3::Eigen)
endif()

add_library(srfm_geodesic STATIC
    src/geodesic/geodesic_solver.cpp
)
target_include_directories(srfm_geodesic PUBLIC src include)
target_link_libraries(srfm_geodesic PUBLIC srfm_manifold)

add_library(srfm_engine STATIC
    src/engine/engine.cpp
)
target_include_directories(srfm_engine PUBLIC src include)
target_link_libraries(srfm_engine PUBLIC
    srfm_beta_calculator
    srfm_manifold
    srfm_geodesic
    srfm_momentum
)

add_library(srfm_lorentz STATIC
    src/lorentz/beta_calculator.cpp
    src/lorentz/lorentz_transform.cpp
)
target_include_directories(srfm_lorentz PUBLIC src include)
target_link_libraries(srfm_lorentz PUBLIC srfm_momentum)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_lorentz PUBLIC Eigen3::Eigen)
endif()

add_library(srfm_backtest STATIC
    src/backtest/backtester.cpp
    src/backtest/geodesic_strategy.cpp
    src/backtest/performance_metrics.cpp
)
target_include_directories(srfm_backtest PUBLIC src include)
target_link_libraries(srfm_backtest PUBLIC srfm_engine srfm_lorentz)

add_library(srfm_stream STATIC
    src/engine/stream_engine.cpp
    src/stream/signal_consumer.cpp
    src/stream/signal_processor.cpp
    src/stream/tick_ingester.cpp
)
target_include_directories(srfm_stream PUBLIC src include)
target_link_libraries(srfm_stream PUBLIC srfm_engine)

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
add_executable(test_full_pipeline
    tests/integration/test_full_pipeline.cpp
)
target_include_directories(test_full_pipeline PRIVATE src include)
target_link_libraries(test_full_pipeline PRIVATE
    srfm_backtest
    srfm_lorentz
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
    GTest::gtest_main
)
add_test(NAME ErrorHandlingIntegrationTests COMMAND test_error_handling)

# ── Tensor / geodesic unit tests ──────────────────────────────────────────────
add_executable(test_metric_tensor
    tests/tensor/test_metric_tensor.cpp
)
target_include_directories(test_metric_tensor PRIVATE src include)
target_link_libraries(test_metric_tensor PRIVATE
    srfm_manifold
    GTest::gtest_main
)
add_test(NAME MetricTensorTests COMMAND test_metric_tensor)

add_executable(test_christoffel
    tests/tensor/test_christoffel.cpp
)
target_include_directories(test_christoffel PRIVATE src include)
target_link_libraries(test_christoffel PRIVATE
    srfm_manifold
    srfm_geodesic
    GTest::gtest_main
)
add_test(NAME ChristoffelTests COMMAND test_christoffel)

add_executable(test_geodesic
    tests/tensor/test_geodesic.cpp
)
target_include_directories(test_geodesic PRIVATE src include)
target_link_libraries(test_geodesic PRIVATE
    srfm_manifold
    srfm_geodesic
    GTest::gtest_main
)
add_test(NAME GeodesicTests COMMAND test_geodesic)

# ── N-asset tests ─────────────────────────────────────────────────────────────
add_executable(test_n_asset
    tests/n_asset/test_n_asset_interval.cpp
    tests/n_asset/test_n_asset_manifold.cpp
    tests/n_asset/test_christoffel_n.cpp
    tests/n_asset/test_geodesic_n.cpp
    tests/n_asset/test_n_asset_engine.cpp
)
target_include_directories(test_n_asset PRIVATE src include tests/n_asset
    ${CMAKE_SOURCE_DIR}  # needed for <third_party/eigen/Eigen/Dense>
)
target_link_libraries(test_n_asset PRIVATE
    srfm_engine
    GTest::gtest_main
)
add_test(NAME NAssetTests COMMAND test_n_asset)

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

# ── CTest ─────────────────────────────────────────────────────────────────────
enable_testing()

# ═══════════════════════════════════════════════════════════════════════════════
# AGT-08: AVX-512 SIMD Acceleration
# Stage 2 — hardware-fast β and γ batch computation
# ═══════════════════════════════════════════════════════════════════════════════

# ── SIMD compiler flags ───────────────────────────────────────────────────────

if(MSVC)
    set(SRFM_AVX2_FLAGS   "/arch:AVX2")
    set(SRFM_AVX512_FLAGS "/arch:AVX512")
else()
    set(SRFM_AVX2_FLAGS   -mavx2 -mfma)
    set(SRFM_AVX512_FLAGS -mavx512f -mfma)
endif()

# Include roots:
#   src/    — existing C++ sources (momentum, beta_calculator, …)
#   include/ — new public SIMD headers  (srfm/simd/cpu_features.hpp, …)
#   .        — project root, so bench/ and tests/ can use relative paths
set(SRFM_SIMD_INCLUDE_DIRS
    ${CMAKE_SOURCE_DIR}/src/simd
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}
)

# ── Scalar reference kernels ──────────────────────────────────────────────────

add_library(srfm_simd_scalar STATIC
    src/simd/beta_scalar.cpp
    src/simd/gamma_scalar.cpp
)
target_include_directories(srfm_simd_scalar PUBLIC ${SRFM_SIMD_INCLUDE_DIRS})
target_link_libraries(srfm_simd_scalar PUBLIC srfm_momentum)

# ── AVX2 kernels (4-wide) ─────────────────────────────────────────────────────

add_library(srfm_simd_avx2 STATIC
    src/simd/beta_avx2.cpp
    src/simd/gamma_avx2.cpp
)
target_include_directories(srfm_simd_avx2 PUBLIC ${SRFM_SIMD_INCLUDE_DIRS})
target_link_libraries(srfm_simd_avx2 PUBLIC srfm_momentum)
# These translation units MUST be compiled with AVX2 so that intrinsics resolve.
target_compile_options(srfm_simd_avx2 PRIVATE ${SRFM_AVX2_FLAGS})

# ── AVX-512F kernels (8-wide) ─────────────────────────────────────────────────

add_library(srfm_simd_avx512 STATIC
    src/simd/beta_avx512.cpp
    src/simd/gamma_avx512.cpp
)
target_include_directories(srfm_simd_avx512 PUBLIC ${SRFM_SIMD_INCLUDE_DIRS})
target_link_libraries(srfm_simd_avx512 PUBLIC srfm_momentum)
target_compile_options(srfm_simd_avx512 PRIVATE ${SRFM_AVX512_FLAGS})

# ── Runtime dispatch layer ────────────────────────────────────────────────────

add_library(srfm_simd_dispatch STATIC
    src/simd/simd_dispatch.cpp
)
target_include_directories(srfm_simd_dispatch PUBLIC ${SRFM_SIMD_INCLUDE_DIRS})
target_link_libraries(srfm_simd_dispatch PUBLIC
    srfm_momentum
    srfm_simd_scalar
    srfm_simd_avx2
    srfm_simd_avx512
)

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

# ── Google Benchmark ──────────────────────────────────────────────────────────

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googlebenchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.8.3
    )
    set(BENCHMARK_ENABLE_TESTING         OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS     OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_DOWNLOAD_DEPENDENCIES  OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googlebenchmark)
endif()

# ── β/γ benchmark suite ───────────────────────────────────────────────────────

add_executable(bench_beta_gamma
    bench/bench_beta_gamma.cpp
)
target_include_directories(bench_beta_gamma PRIVATE ${SRFM_SIMD_INCLUDE_DIRS})
target_link_libraries(bench_beta_gamma PRIVATE
    srfm_simd_dispatch
    benchmark::benchmark
)
# Ensure benchmarks are always compiled with optimisations.
target_compile_options(bench_beta_gamma PRIVATE
    $<IF:$<CXX_COMPILER_ID:MSVC>,/O2,-O3>
)

# Convenience target: cmake --build build --target bench
add_custom_target(bench
    COMMAND bench_beta_gamma
    DEPENDS bench_beta_gamma
    COMMENT "Running SRFM AGT-08 beta/gamma SIMD benchmarks"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# ── Install targets and package config ────────────────────────────────────────
# Only the core API libraries (momentum, beta_calculator, manifold, geodesic,
# engine) are exported. The SIMD implementation targets use source-relative
# include paths that are not suitable for export and are considered internal.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Patch include dirs on core targets to use generator expressions so that
# the same target works both in the build tree and after installation.
foreach(_tgt srfm_momentum srfm_beta_calculator srfm_manifold srfm_geodesic srfm_engine)
    get_target_property(_cur_incs ${_tgt} INTERFACE_INCLUDE_DIRECTORIES)
    set_target_properties(${_tgt} PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES
            "$<BUILD_INTERFACE:${CMAKE_SOURCE_DIR}/src>;$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
    )
endforeach()

install(TARGETS
        srfm_momentum
        srfm_beta_calculator
        srfm_manifold
        srfm_geodesic
        srfm_engine
    EXPORT srfmTargets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(DIRECTORY src/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/srfm
    FILES_MATCHING PATTERN "*.hpp" PATTERN "*.h"
)

install(EXPORT srfmTargets
    FILE srfmTargets.cmake
    NAMESPACE srfm::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srfm
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/srfmConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/srfmConfig.cmake"
    INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srfm
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/srfmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/srfmConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/srfmConfigVersion.cmake"
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/srfm
)