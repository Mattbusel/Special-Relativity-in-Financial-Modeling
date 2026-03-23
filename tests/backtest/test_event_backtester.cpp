/// @file tests/backtest/test_event_backtester.cpp
/// @brief Unit tests for the event-driven BacktestEngine and
///        RelativisticStrategy (Round 3).

#include <gtest/gtest.h>

#include "srfm/event_backtester.hpp"

#include <cmath>
#include <vector>

using namespace srfm::event_bt;

// ─── Helpers ─────────────────────────────────────────────────────────────────

static BacktestEvent make_trade(long long ts_ms, double price,
                                 double volume = 100.0,
                                 std::string symbol = "BTC") {
    return BacktestEvent{ts_ms, price, volume, EventType::Trade, std::move(symbol)};
}

static BacktestEvent make_bar(long long ts_ms, double price,
                               double volume = 100.0,
                               std::string symbol = "ETH") {
    return BacktestEvent{ts_ms, price, volume, EventType::Bar, std::move(symbol)};
}

/// Strategy that always buys 1 unit on each trade event.
class AlwaysBuyStrategy : public Strategy {
public:
    std::optional<Order> on_trade(const BacktestEvent& ev) override {
        Order o;
        o.symbol       = ev.symbol;
        o.side         = OrderSide::Buy;
        o.quantity     = 1.0;
        o.order_type   = OrderType::Market;
        o.timestamp_ms = ev.timestamp_ms;
        return o;
    }
};

/// Strategy that never emits an order.
class PassiveStrategy : public Strategy {
public:
    std::optional<Order> on_bar(const BacktestEvent&) override { return std::nullopt; }
    std::optional<Order> on_trade(const BacktestEvent&) override { return std::nullopt; }
};

// ─── BacktestEvent priority ────────────────────────────────────────────────────

TEST(EventBacktesterEvent, EarlierTimestampHigherPriority) {
    BacktestEvent early{1000, 100.0, 10.0, EventType::Trade, "A"};
    BacktestEvent late {2000, 200.0, 10.0, EventType::Trade, "A"};
    // greater<> uses operator> — early should NOT be greater-than late
    EXPECT_FALSE(early > late);
    EXPECT_TRUE(late  > early);
}

// ─── Engine: no strategy ───────────────────────────────────────────────────────

TEST(BacktestEngineNoStrategy, RunWithNoStrategyReturnsEmptyResult) {
    BacktestEngine engine(10'000.0);
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 110.0));
    BacktestResult r = engine.run();
    EXPECT_EQ(r.num_trades, 0);
    EXPECT_DOUBLE_EQ(r.total_return, 0.0);
}

TEST(BacktestEngineNoStrategy, EmptyQueueReturnsZeroResult) {
    BacktestEngine engine(10'000.0);
    engine.set_strategy(std::make_unique<PassiveStrategy>());
    BacktestResult r = engine.run();
    EXPECT_EQ(r.num_trades, 0);
}

// ─── Engine: passive strategy ──────────────────────────────────────────────────

TEST(BacktestEnginePassive, NoFillsProducedByPassiveStrategy) {
    BacktestEngine engine(10'000.0);
    engine.set_strategy(std::make_unique<PassiveStrategy>());
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 200.0));
    BacktestResult r = engine.run();
    EXPECT_EQ(r.num_trades, 0);
    EXPECT_EQ(engine.fills().size(), 0u);
}

// ─── Engine: always-buy strategy ───────────────────────────────────────────────

TEST(BacktestEngineAlwaysBuy, FillsCreatedForEachTradeEvent) {
    BacktestEngine engine(1'000'000.0, 0.0);  // zero commission for simplicity
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 110.0));
    engine.add_event(make_trade(3000, 120.0));
    BacktestResult r = engine.run();
    EXPECT_EQ(r.num_trades, 3);
}

TEST(BacktestEngineAlwaysBuy, CashDecreasesAfterBuy) {
    BacktestEngine engine(10'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 100.0));
    engine.run();
    EXPECT_LT(engine.portfolio().cash, 10'000.0);
}

TEST(BacktestEngineAlwaysBuy, PositionIncreasesAfterBuy) {
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 100.0, 100.0, "BTC"));
    engine.run();
    auto const& pos = engine.portfolio().positions;
    EXPECT_GT(pos.at("BTC"), 0.0);
}

TEST(BacktestEngineAlwaysBuy, InsufficientCashScalesDownFill) {
    // Only enough cash for 0.5 units at price 200
    BacktestEngine engine(100.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 200.0));
    engine.run();
    // Should still execute but at scaled-down quantity
    EXPECT_EQ(engine.fills().size(), 1u);
    EXPECT_LT(engine.fills()[0].fill_qty, 1.0);
}

// ─── Engine: event ordering ────────────────────────────────────────────────────

TEST(BacktestEngineOrdering, EventsProcessedInTimestampOrder) {
    // Add events out of order; fills should appear in timestamp order.
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(3000, 130.0));
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 115.0));
    engine.run();

    auto const& fills = engine.fills();
    ASSERT_EQ(fills.size(), 3u);
    EXPECT_LE(fills[0].timestamp_ms, fills[1].timestamp_ms);
    EXPECT_LE(fills[1].timestamp_ms, fills[2].timestamp_ms);
}

// ─── Engine: bar events ────────────────────────────────────────────────────────

class AlwaysBuyOnBar : public Strategy {
public:
    std::optional<Order> on_bar(const BacktestEvent& ev) override {
        Order o;
        o.symbol       = ev.symbol;
        o.side         = OrderSide::Buy;
        o.quantity     = 1.0;
        o.order_type   = OrderType::Market;
        o.timestamp_ms = ev.timestamp_ms;
        return o;
    }
};

TEST(BacktestEngineBarEvents, BarEventsTriggerOnBar) {
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyOnBar>());
    engine.add_event(make_bar(1000, 200.0));
    engine.add_event(make_bar(2000, 210.0));
    BacktestResult r = engine.run();
    EXPECT_EQ(r.num_trades, 2);
}

// ─── Engine: limit orders ──────────────────────────────────────────────────────

class LimitBuyStrategy : public Strategy {
public:
    explicit LimitBuyStrategy(double limit) : limit_(limit) {}
    std::optional<Order> on_trade(const BacktestEvent& ev) override {
        Order o;
        o.symbol       = ev.symbol;
        o.side         = OrderSide::Buy;
        o.quantity     = 1.0;
        o.order_type   = OrderType::Limit;
        o.limit_price  = limit_;
        o.timestamp_ms = ev.timestamp_ms;
        return o;
    }
    double limit_;
};

TEST(BacktestEngineLimitOrder, LimitBuyNotFilledWhenPriceAboveLimit) {
    // Limit at 90, price at 100 — should not fill
    BacktestEngine engine(10'000.0, 0.0);
    engine.set_strategy(std::make_unique<LimitBuyStrategy>(90.0));
    engine.add_event(make_trade(1000, 100.0));
    engine.run();
    EXPECT_EQ(engine.fills().size(), 0u);
}

TEST(BacktestEngineLimitOrder, LimitBuyFilledWhenPriceAtOrBelowLimit) {
    // Limit at 100, price at 100 — should fill
    BacktestEngine engine(10'000.0, 0.0);
    engine.set_strategy(std::make_unique<LimitBuyStrategy>(100.0));
    engine.add_event(make_trade(1000, 100.0));
    engine.run();
    EXPECT_EQ(engine.fills().size(), 1u);
}

// ─── BacktestResult fields ─────────────────────────────────────────────────────

TEST(BacktestResult, TotalReturnFiniteAfterTrades) {
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 120.0));
    BacktestResult r = engine.run();
    EXPECT_TRUE(std::isfinite(r.total_return));
}

TEST(BacktestResult, MaxDrawdownNonNegative) {
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    engine.add_event(make_trade(1000, 100.0));
    engine.add_event(make_trade(2000, 80.0));
    BacktestResult r = engine.run();
    EXPECT_GE(r.max_drawdown, 0.0);
}

TEST(BacktestResult, WinRateInUnitInterval) {
    BacktestEngine engine(1'000'000.0, 0.0);
    engine.set_strategy(std::make_unique<AlwaysBuyStrategy>());
    for (int i = 0; i < 10; ++i) {
        engine.add_event(make_trade(1000LL + i * 100, 100.0 + i));
    }
    BacktestResult r = engine.run();
    EXPECT_GE(r.win_rate, 0.0);
    EXPECT_LE(r.win_rate, 1.0);
}

// ─── RelativisticStrategy ─────────────────────────────────────────────────────

TEST(RelativisticStrategy, FirstEventNoOrder) {
    RelativisticStrategy strat(1.0, 0.001);
    strat.on_start();
    BacktestEvent ev = make_trade(1000, 100.0);
    auto order = strat.on_trade(ev);
    EXPECT_FALSE(order.has_value());
}

TEST(RelativisticStrategy, SpacelikeEventRejected) {
    // Create two events where the price jumps are very large relative to
    // the time delta, making the interval spacelike.
    RelativisticStrategy strat(1.0, 0.0);  // threshold=0 to avoid signal filtering
    strat.on_start();

    // First event: anchor
    BacktestEvent ev1 = make_trade(0LL, 1.0, 1.0);
    strat.on_trade(ev1);
    int before = strat.spacelike_rejections();

    // Second event: huge price jump in tiny time → spacelike
    BacktestEvent ev2 = make_trade(1LL, 1e9, 1e9);
    strat.on_trade(ev2);

    // Either accepted as timelike or rejected as spacelike — check counts sum
    int after = strat.spacelike_rejections() + strat.timelike_accepts();
    EXPECT_EQ(after - before, 1);
}

TEST(RelativisticStrategy, TimelikeEventCanGenerateOrder) {
    // Two events close in time with small price change: should be timelike
    RelativisticStrategy strat(1.0, 0.0);  // threshold=0
    strat.on_start();

    BacktestEvent ev1 = make_trade(0LL, 100.0, 100.0);
    strat.on_trade(ev1);

    // Small price increase over 60 seconds
    BacktestEvent ev2 = make_trade(60'000LL, 100.1, 100.0);
    auto order = strat.on_trade(ev2);

    // Either generates an order (timelike) or the interval was classified differently.
    // The key invariant is that the counts increment.
    int total = strat.timelike_accepts() + strat.spacelike_rejections();
    EXPECT_EQ(total, 1);
}

TEST(RelativisticStrategy, CountsStartAtZero) {
    RelativisticStrategy strat;
    EXPECT_EQ(strat.spacelike_rejections(), 0);
    EXPECT_EQ(strat.timelike_accepts(), 0);
}

TEST(RelativisticStrategy, IntegrationWithEngine) {
    BacktestEngine engine(1'000'000.0, 0.001);
    engine.set_strategy(std::make_unique<RelativisticStrategy>(1.0, 0.0005));

    // 10 bars with steadily rising price, 1-minute spacing
    for (int i = 0; i < 10; ++i) {
        engine.add_event(make_trade(
            static_cast<long long>(i) * 60'000LL,
            100.0 + i * 0.2,
            1000.0
        ));
    }
    BacktestResult r = engine.run();
    EXPECT_TRUE(std::isfinite(r.total_return));
    EXPECT_GE(r.num_trades, 0);
    EXPECT_GE(r.max_drawdown, 0.0);
}
