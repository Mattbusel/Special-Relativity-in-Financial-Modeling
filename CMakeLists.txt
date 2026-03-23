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
find_package(fmt QUIET)
find_package(spdlog QUIET)

# If Eigen3 was not found via find_package, fall back to the vendored copy.
# If the vendored copy is incomplete (missing Eigen/Core), use FetchContent.
if(NOT (Eigen3_FOUND OR TARGET Eigen3::Eigen))
    set(_EIGEN_VENDOR_DIR "${CMAKE_SOURCE_DIR}/third_party/eigen")
    if(EXISTS "${_EIGEN_VENDOR_DIR}/Eigen/Dense" AND EXISTS "${_EIGEN_VENDOR_DIR}/Eigen/Core")
        message(STATUS "Using vendored Eigen3 from third_party/eigen")
        add_library(Eigen3::Eigen INTERFACE IMPORTED GLOBAL)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES
                "${_EIGEN_VENDOR_DIR};${CMAKE_SOURCE_DIR}"
        )
        set(Eigen3_FOUND TRUE)
    else()
        message(STATUS "Vendored Eigen3 incomplete or missing — fetching via FetchContent")
        include(FetchContent)
        FetchContent_Declare(
            Eigen3
            GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
            GIT_TAG        3.4.0
            GIT_SHALLOW    TRUE
        )
        set(EIGEN_BUILD_DOC OFF CACHE BOOL "" FORCE)
        set(BUILD_TESTING OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(Eigen3)
        set(Eigen3_FOUND TRUE)
    endif()
endif()

if(NOT TARGET Eigen3::Eigen)
    message(FATAL_ERROR "Eigen3 not found. Install it (e.g. 'apt install libeigen3-dev') or ensure FetchContent has network access.")
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

# ── Per-module build definitions ──────────────────────────────────────────────
include(cmake/momentum.cmake)
include(cmake/manifold.cmake)
include(cmake/tensor.cmake)
include(cmake/tests.cmake)

# ── CTest ─────────────────────────────────────────────────────────────────────
enable_testing()

# ── Install targets and package config ────────────────────────────────────────
# Only the core API libraries (momentum, beta_calculator, manifold, geodesic,
# engine) are exported. The SIMD implementation targets use source-relative
# include paths that are not suitable for export and are considered internal.
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

# Patch include dirs on core targets to use generator expressions so that
# the same target works both in the build tree and after installation.
foreach(_tgt srfm_momentum srfm_beta_calculator srfm_manifold srfm_geodesic srfm_engine srfm_portfolio srfm_geodesic_path)
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
        srfm_portfolio
        srfm_geodesic_path
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
