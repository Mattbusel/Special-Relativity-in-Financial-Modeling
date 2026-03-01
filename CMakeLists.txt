cmake_minimum_required(VERSION 3.25)
project(srfm_cpp
    VERSION 0.1.0
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

# ── vcpkg / RapidCheck ────────────────────────────────────────────────────────
find_package(RapidCheck CONFIG QUIET)

# ── Source library targets ────────────────────────────────────────────────────

add_library(srfm_momentum STATIC
    src/momentum/momentum.cpp
)
target_include_directories(srfm_momentum PUBLIC src)

add_library(srfm_beta_calculator STATIC
    src/beta_calculator/beta_calculator.cpp
)
target_include_directories(srfm_beta_calculator PUBLIC src)
target_link_libraries(srfm_beta_calculator PUBLIC srfm_momentum)

add_library(srfm_manifold STATIC
    src/manifold/spacetime_manifold.cpp
)
target_include_directories(srfm_manifold PUBLIC src)
target_link_libraries(srfm_manifold PUBLIC srfm_momentum)

add_library(srfm_geodesic STATIC
    src/geodesic/geodesic_solver.cpp
)
target_include_directories(srfm_geodesic PUBLIC src)
target_link_libraries(srfm_geodesic PUBLIC srfm_manifold)

add_library(srfm_engine STATIC
    src/engine/engine.cpp
)
target_include_directories(srfm_engine PUBLIC src)
target_link_libraries(srfm_engine PUBLIC
    srfm_beta_calculator
    srfm_manifold
    srfm_geodesic
    srfm_momentum
)

# ── Existing momentum unit tests ──────────────────────────────────────────────
add_executable(test_momentum
    tests/momentum/test_momentum.cpp
)
target_include_directories(test_momentum PRIVATE src tests)
target_link_libraries(test_momentum PRIVATE srfm_momentum)
add_test(NAME MomentumUnitTests COMMAND test_momentum)

# ── Property-based tests (RapidCheck) ────────────────────────────────────────
if(RapidCheck_FOUND)
    set(PROPERTY_TESTS
        prop_lorentz_identity
        prop_rapidity_additivity
        prop_doppler_inverse
        prop_velocity_subluminal
        prop_christoffel_flat
        prop_geodesic_flat
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
    COMMENT "Running SRFM AGT-08 β/γ SIMD benchmarks"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
