# cmake/manifold.cmake
# Manifold, geodesic, engine, backtest, and stream library targets.

add_library(srfm_manifold STATIC
    src/manifold/spacetime_manifold.cpp
    src/manifold/spacetime_interval.cpp
    src/manifold/market_manifold.cpp
    src/manifold/n_asset_interval.cpp
    src/manifold/normalizer.cpp
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
if(spdlog_FOUND OR TARGET spdlog::spdlog)
    target_link_libraries(srfm_geodesic PUBLIC spdlog::spdlog)
    target_compile_definitions(srfm_geodesic PUBLIC SRFM_HAS_SPDLOG=1)
endif()

add_library(srfm_engine STATIC
    src/engine/engine.cpp
    src/engine/n_asset_engine.cpp
    src/validation/backtest_runner.cpp
    src/validation/regime_validator.cpp
)
target_include_directories(srfm_engine PUBLIC src include)
target_link_libraries(srfm_engine PUBLIC
    srfm_beta_calculator
    srfm_manifold
    srfm_geodesic
    srfm_momentum
)

add_library(srfm_backtest STATIC
    src/backtest/backtester.cpp
    src/backtest/geodesic_strategy.cpp
    src/backtest/performance_metrics.cpp
)
target_include_directories(srfm_backtest PUBLIC src include)
target_link_libraries(srfm_backtest PUBLIC srfm_engine srfm_lorentz)
if(fmt_FOUND OR TARGET fmt::fmt)
    target_link_libraries(srfm_backtest PUBLIC fmt::fmt)
endif()

add_library(srfm_stream STATIC
    src/engine/stream_engine.cpp
    src/stream/signal_consumer.cpp
    src/stream/signal_processor.cpp
    src/stream/tick_ingester.cpp
)
target_include_directories(srfm_stream PUBLIC src include)
target_link_libraries(srfm_stream PUBLIC srfm_engine)
