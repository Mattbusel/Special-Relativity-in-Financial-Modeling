# cmake/tensor.cmake
# Tensor library, SIMD acceleration targets, and benchmark targets.

add_library(srfm_tensor STATIC
    src/tensor/metric_tensor.cpp
    src/tensor/christoffel.cpp
    src/tensor/christoffel_n.cpp
    src/tensor/geodesic.cpp
    src/tensor/geodesic_n.cpp
    src/tensor/geodesic_signal.cpp
    src/tensor/n_asset_manifold.cpp
)
# Note: ${CMAKE_SOURCE_DIR} is NOT added here to avoid a bare 'version' file
# in the project root shadowing <version> from C++20 standard library on MSVC.
# Targets that include <third_party/eigen/Eigen/Dense> must add ${CMAKE_SOURCE_DIR}
# to their own include directories (see test_n_asset below).
target_include_directories(srfm_tensor PUBLIC src include)
target_link_libraries(srfm_tensor PUBLIC srfm_manifold srfm_geodesic)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_tensor PUBLIC Eigen3::Eigen)
endif()

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
