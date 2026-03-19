# cmake/momentum.cmake
# Momentum, BetaCalculator, and Lorentz library targets.

add_library(srfm_momentum STATIC
    src/momentum/momentum.cpp
    src/momentum/momentum_processor.cpp
    src/momentum/relativistic_signal.cpp
)
target_include_directories(srfm_momentum PUBLIC src include)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_momentum PUBLIC Eigen3::Eigen)
endif()

add_library(srfm_beta_calculator STATIC
    src/beta_calculator/beta_calculator.cpp
)
target_include_directories(srfm_beta_calculator PUBLIC src include)
target_link_libraries(srfm_beta_calculator PUBLIC srfm_momentum)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_beta_calculator PUBLIC Eigen3::Eigen)
endif()

add_library(srfm_lorentz STATIC
    src/lorentz/beta_calculator.cpp
    src/lorentz/lorentz_transform.cpp
)
target_include_directories(srfm_lorentz PUBLIC src include)
target_link_libraries(srfm_lorentz PUBLIC srfm_momentum)
if(Eigen3_FOUND OR TARGET Eigen3::Eigen)
    target_link_libraries(srfm_lorentz PUBLIC Eigen3::Eigen)
endif()
