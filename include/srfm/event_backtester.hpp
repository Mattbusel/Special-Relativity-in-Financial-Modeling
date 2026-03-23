#pragma once

/// @file include/srfm/event_backtester.hpp
/// @brief Event-Driven Backtester — Round 3 addition.
///
/// # Module: Event-Driven Backtester
///
/// ## Responsibility
/// Provide a lightweight event queue-based simulation engine that replays
/// market events in strict timestamp order and passes them to a pluggable
/// Strategy.  All trade fills, portfolio accounting, and performance metrics
/// are tracked internally.
///
/// ## Architecture
///   BacktestEvent → priority_queue → BacktestEngine → Strategy::on_trade /
///   on_bar → optional<Order> → simulated Fill → Portfolio update →
///   BacktestResult
///
/// ## RelativisticStrategy
/// A concrete Strategy subclass that uses `SpacetimeInterval::classify()` to
/// filter trades.  Only trades that are TIMELIKE relative to the previous
/// filled trade are accepted — spacelike jumps are treated as noise and
/// rejected.
///
/// ## Guarantees
/// - No raw pointers; ownership by value
/// - All fallible operations return std::optional or bool
/// - Thread-safe reads: const member functions on BacktestResult are safe

#include "srfm/manifold.hpp"
#include "srfm/types.hpp"
#include "srfm/constants.hpp"

#include <cmath>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <queue>
#include <string>
#include <vector>

namespace srfm::event_bt {

// ─── EventType ────────────────────────────────────────────────────────────────

enum class EventType {
    Trade,  ///< Individual trade execution tick
    Quote,  ///< Bid-ask quote update
    Bar,    ///< OHLCV bar (1-min, 5-min, daily, etc.)
};

// ─── BacktestEvent ────────────────────────────────────────────────────────────

/// A single market event with a timestamp.
struct BacktestEvent {
    long long   timestamp_ms;  ///< Unix epoch milliseconds
    double      price;         ///< Trade price / bar close / mid-quote
    double      volume;        ///< Volume for this event (0 for quotes)
    EventType   type;          ///< Trade, Quote, or Bar
    std::string symbol;        ///< Instrument identifier

    /// Priority comparison — earlier events have higher priority.
    bool operator>(const BacktestEvent& rhs) const noexcept {
        return timestamp_ms > rhs.timestamp_ms;
    }
};

// ─── OrderSide ────────────────────────────────────────────────────────────────

enum class OrderSide { Buy, Sell };

// ─── OrderType ────────────────────────────────────────────────────────────────

enum class OrderType {
    Market,          ///< Fill immediately at current price
    Limit,           ///< Fill only when price <= limit_price (buy) or >= (sell)
};

// ─── Order ────────────────────────────────────────────────────────────────────

/// An order emitted by a Strategy in response to a market event.
struct Order {
    std::string symbol;
    OrderSide   side          = OrderSide::Buy;
    double      quantity      = 0.0;
    OrderType   order_type    = OrderType::Market;
    double      limit_price   = 0.0;   ///< Only used for Limit orders
    long long   timestamp_ms  = 0;
};

// ─── Fill ─────────────────────────────────────────────────────────────────────

/// Confirmation of an executed order.
struct Fill {
    Order       order;
    double      fill_price = 0.0;
    double      fill_qty   = 0.0;
    long long   timestamp_ms = 0;
    double      commission = 0.0;
};

// ─── Portfolio ────────────────────────────────────────────────────────────────

/// Running portfolio state tracked by BacktestEngine.
struct Portfolio {
    double                      cash         = 0.0;
    std::map<std::string,double> positions;     ///< symbol -> net quantity
    std::vector<double>         equity_curve;  ///< Equity snapshot after each fill

    /// Current gross notional exposure.
    double gross_notional(const std::map<std::string,double>& last_prices) const {
        double gn = 0.0;
        for (auto const& [sym, qty] : positions) {
            auto it = last_prices.find(sym);
            if (it != last_prices.end()) {
                gn += std::abs(qty) * it->second;
            }
        }
        return gn;
    }
};

// ─── BacktestResult ───────────────────────────────────────────────────────────

/// Aggregate performance statistics from a completed backtest.
struct BacktestResult {
    double total_return    = 0.0;  ///< (final_equity - initial_equity) / initial
    double sharpe_ratio    = 0.0;  ///< Annualised Sharpe (assuming 252 days)
    double max_drawdown    = 0.0;  ///< Peak-to-trough equity drawdown fraction
    int    num_trades      = 0;    ///< Total fills executed
    double win_rate        = 0.0;  ///< Fraction of profitable trades
    double profit_factor   = 0.0;  ///< Gross gains / gross losses
};

// ─── Strategy (abstract base) ─────────────────────────────────────────────────

/// Abstract strategy interface.  Subclass and override the virtual methods.
class Strategy {
public:
    virtual ~Strategy() = default;

    /// Called at the start of the simulation before any events are dispatched.
    virtual void on_start() {}

    /// Called at the end of the simulation after all events are processed.
    virtual void on_end() {}

    /// Called for every Trade-type event.
    ///
    /// @return An Order to execute, or std::nullopt for no action.
    virtual std::optional<Order> on_trade(const BacktestEvent& /*event*/) {
        return std::nullopt;
    }

    /// Called for every Bar-type event.
    ///
    /// @return An Order to execute, or std::nullopt for no action.
    virtual std::optional<Order> on_bar(const BacktestEvent& /*event*/) {
        return std::nullopt;
    }
};

// ─── BacktestEngine ───────────────────────────────────────────────────────────

/// Event-driven backtesting engine.
///
/// Events are inserted via `add_event()` and processed in strict
/// timestamp order by `run()`.  The engine simulates market fills,
/// tracks a Portfolio, and returns a BacktestResult.
///
/// Usage:
/// @code
/// BacktestEngine engine(10'000.0, 0.001);
/// engine.set_strategy(std::make_unique<MyStrategy>());
/// for (auto& ev : raw_events) engine.add_event(ev);
/// auto result = engine.run();
/// @endcode
class BacktestEngine {
public:
    /// @param initial_cash  Starting cash in quote currency.
    /// @param commission    Commission fraction per trade notional.
    explicit BacktestEngine(double initial_cash = 100'000.0,
                            double commission   = 0.001) noexcept
        : initial_cash_(initial_cash)
        , commission_(commission)
    {
        portfolio_.cash = initial_cash;
    }

    /// Set the strategy to drive order generation.
    void set_strategy(std::unique_ptr<Strategy> strat) noexcept {
        strategy_ = std::move(strat);
    }

    /// Insert a market event into the priority queue.
    void add_event(BacktestEvent event) {
        queue_.push(std::move(event));
    }

    /// Process all queued events and return aggregate performance.
    ///
    /// @return BacktestResult populated after all events are consumed.
    [[nodiscard]] BacktestResult run();

    /// Const accessor for the fill history.
    [[nodiscard]] const std::vector<Fill>& fills() const noexcept { return fills_; }

    /// Const accessor for the portfolio state after `run()`.
    [[nodiscard]] const Portfolio& portfolio() const noexcept { return portfolio_; }

private:
    // Process a single event; may append to fills_ and update portfolio_.
    void process_event(const BacktestEvent& event);

    // Attempt to execute an Order against the current event price.
    std::optional<Fill> execute_order(const Order& order,
                                       const BacktestEvent& event);

    // Compute BacktestResult from fills_ and portfolio_ equity curve.
    [[nodiscard]] BacktestResult compute_result() const noexcept;

    double initial_cash_;
    double commission_;
    std::unique_ptr<Strategy> strategy_;

    using EventQueue = std::priority_queue<
        BacktestEvent,
        std::vector<BacktestEvent>,
        std::greater<BacktestEvent>>;
    EventQueue queue_;

    Portfolio                    portfolio_;
    std::vector<Fill>            fills_;
    std::map<std::string,double> last_prices_;  // symbol -> latest price
};

// ─── RelativisticStrategy ─────────────────────────────────────────────────────

/// A concrete Strategy that filters trades using SpacetimeInterval::classify().
///
/// On every Trade event the engine converts the current and previous event
/// into SpacetimeEvents and computes their interval.  Only TIMELIKE intervals
/// (ds² < 0) are allowed to generate orders — SPACELIKE events are treated as
/// causally disconnected noise and ignored.
///
/// This implements the causal-cone hypothesis: only market moves that respect
/// the relativistic causal structure of financial spacetime are acted upon.
class RelativisticStrategy : public Strategy {
public:
    /// @param base_qty  Quantity to buy/sell on each qualifying signal.
    /// @param threshold  Fraction of equity change required to generate a signal.
    explicit RelativisticStrategy(double base_qty    = 1.0,
                                   double threshold   = 0.001) noexcept
        : base_qty_(base_qty)
        , threshold_(threshold)
        , has_prev_(false)
    {}

    void on_start() override { has_prev_ = false; }

    std::optional<Order> on_trade(const BacktestEvent& event) override;
    std::optional<Order> on_bar  (const BacktestEvent& event) override;

    /// Number of events rejected due to spacelike interval.
    [[nodiscard]] int spacelike_rejections() const noexcept {
        return spacelike_count_;
    }

    /// Number of events accepted due to timelike interval.
    [[nodiscard]] int timelike_accepts() const noexcept {
        return timelike_count_;
    }

private:
    std::optional<Order> evaluate(const BacktestEvent& event);

    double base_qty_;
    double threshold_;
    bool   has_prev_;
    BacktestEvent prev_event_{};
    int    spacelike_count_ = 0;
    int    timelike_count_  = 0;
};

} // namespace srfm::event_bt
