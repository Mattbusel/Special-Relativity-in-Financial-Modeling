/// @file src/event_backtester.cpp
/// @brief Implementation of the event-driven BacktestEngine and
///        RelativisticStrategy.

#include "srfm/event_backtester.hpp"
#include "srfm/manifold.hpp"
#include "srfm/constants.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>

namespace srfm::event_bt {

// ─── BacktestEngine::process_event ────────────────────────────────────────────

void BacktestEngine::process_event(const BacktestEvent& event) {
    last_prices_[event.symbol] = event.price;

    if (!strategy_) return;

    std::optional<Order> order_opt;
    if (event.type == EventType::Trade || event.type == EventType::Quote) {
        order_opt = strategy_->on_trade(event);
    } else if (event.type == EventType::Bar) {
        order_opt = strategy_->on_bar(event);
    }

    if (!order_opt.has_value()) return;

    auto fill_opt = execute_order(*order_opt, event);
    if (!fill_opt.has_value()) return;

    const Fill& fill = *fill_opt;
    fills_.push_back(fill);

    // Update cash
    double cost = fill.fill_price * fill.fill_qty + fill.commission;
    if (fill.order.side == OrderSide::Buy) {
        portfolio_.cash -= cost;
    } else {
        portfolio_.cash += fill.fill_price * fill.fill_qty - fill.commission;
    }

    // Update position
    double signed_qty = (fill.order.side == OrderSide::Buy)
                        ? fill.fill_qty
                        : -fill.fill_qty;
    portfolio_.positions[fill.order.symbol] += signed_qty;

    // Record equity snapshot
    double equity = portfolio_.cash;
    for (auto const& [sym, qty] : portfolio_.positions) {
        auto it = last_prices_.find(sym);
        if (it != last_prices_.end()) {
            equity += qty * it->second;
        }
    }
    portfolio_.equity_curve.push_back(equity);
}

// ─── BacktestEngine::execute_order ────────────────────────────────────────────

std::optional<Fill> BacktestEngine::execute_order(const Order& order,
                                                    const BacktestEvent& event) {
    if (order.quantity <= 0.0) return std::nullopt;

    double fill_price = event.price;

    // Limit order check
    if (order.order_type == OrderType::Limit) {
        if (order.side == OrderSide::Buy  && event.price > order.limit_price) return std::nullopt;
        if (order.side == OrderSide::Sell && event.price < order.limit_price) return std::nullopt;
        fill_price = order.limit_price;
    }

    // Insufficient cash guard for buy orders
    double notional = fill_price * order.quantity;
    double comm     = notional * commission_;
    if (order.side == OrderSide::Buy && (notional + comm) > portfolio_.cash) {
        // Scale down to affordable quantity
        double affordable = portfolio_.cash / (fill_price * (1.0 + commission_));
        if (affordable <= 0.0) return std::nullopt;
        notional = fill_price * affordable;
        comm     = notional * commission_;
        return Fill{order, fill_price, affordable, event.timestamp_ms, comm};
    }

    return Fill{order, fill_price, order.quantity, event.timestamp_ms, comm};
}

// ─── BacktestEngine::run ──────────────────────────────────────────────────────

BacktestResult BacktestEngine::run() {
    if (strategy_) strategy_->on_start();

    while (!queue_.empty()) {
        BacktestEvent event = queue_.top();
        queue_.pop();
        process_event(event);
    }

    if (strategy_) strategy_->on_end();

    return compute_result();
}

// ─── BacktestEngine::compute_result ───────────────────────────────────────────

BacktestResult BacktestEngine::compute_result() const noexcept {
    BacktestResult result;
    result.num_trades = static_cast<int>(fills_.size());

    const auto& eq = portfolio_.equity_curve;
    if (eq.empty()) return result;

    double final_equity = eq.back();
    result.total_return = (final_equity - initial_cash_) / initial_cash_;

    // Max drawdown
    double peak = initial_cash_;
    double max_dd = 0.0;
    double running = initial_cash_;
    for (double e : eq) {
        if (e > peak) peak = e;
        double dd = (peak - e) / peak;
        if (dd > max_dd) max_dd = dd;
        running = e;
    }
    result.max_drawdown = max_dd;

    // Daily returns proxy: per-fill equity changes
    std::vector<double> returns;
    returns.reserve(eq.size());
    double prev = initial_cash_;
    for (double e : eq) {
        if (prev != 0.0) returns.push_back((e - prev) / prev);
        prev = e;
    }

    if (!returns.empty()) {
        double mean_r = std::accumulate(returns.begin(), returns.end(), 0.0)
                        / static_cast<double>(returns.size());
        double var = 0.0;
        for (double r : returns) var += (r - mean_r) * (r - mean_r);
        var /= static_cast<double>(returns.size());
        double std_r = std::sqrt(var);
        // Annualise assuming ~252 trading days
        constexpr double ann = 252.0;
        result.sharpe_ratio = (std_r > 0.0)
                              ? (mean_r / std_r) * std::sqrt(ann)
                              : 0.0;
    }

    // Win rate and profit factor from fills
    if (!fills_.empty()) {
        double gross_win  = 0.0;
        double gross_loss = 0.0;
        int    wins       = 0;

        for (std::size_t i = 1; i < fills_.size(); ++i) {
            const Fill& prev_f = fills_[i - 1];
            const Fill& curr_f = fills_[i];
            if (curr_f.order.symbol != prev_f.order.symbol) continue;

            double pnl = 0.0;
            if (prev_f.order.side == OrderSide::Buy
                && curr_f.order.side == OrderSide::Sell) {
                pnl = (curr_f.fill_price - prev_f.fill_price) * curr_f.fill_qty
                      - curr_f.commission - prev_f.commission;
            } else if (prev_f.order.side == OrderSide::Sell
                       && curr_f.order.side == OrderSide::Buy) {
                pnl = (prev_f.fill_price - curr_f.fill_price) * curr_f.fill_qty
                      - curr_f.commission - prev_f.commission;
            }

            if (pnl > 0.0) { gross_win += pnl; ++wins; }
            else if (pnl < 0.0) { gross_loss += std::abs(pnl); }
        }

        int round_trips = static_cast<int>(fills_.size()) / 2;
        result.win_rate      = (round_trips > 0)
                               ? static_cast<double>(wins) / round_trips
                               : 0.0;
        result.profit_factor = (gross_loss > 0.0) ? gross_win / gross_loss : 0.0;
    }

    return result;
}

// ─── RelativisticStrategy ─────────────────────────────────────────────────────

std::optional<Order> RelativisticStrategy::evaluate(const BacktestEvent& event) {
    if (!has_prev_) {
        prev_event_ = event;
        has_prev_   = true;
        return std::nullopt;
    }

    // Build SpacetimeEvents from the two consecutive market events.
    // Map: time → timestamp_ms in seconds, price → price,
    //      volume → volume, momentum → Δprice / prev_price (rate of change)
    double dt_s     = static_cast<double>(event.timestamp_ms - prev_event_.timestamp_ms) / 1000.0;
    double dp       = event.price - prev_event_.price;
    double momentum = (prev_event_.price > 0.0) ? dp / prev_event_.price : 0.0;

    srfm::manifold::SpacetimeEvent A{
        /* time     */ static_cast<double>(prev_event_.timestamp_ms) / 1000.0,
        /* price    */ prev_event_.price,
        /* volume   */ prev_event_.volume,
        /* momentum */ 0.0
    };
    srfm::manifold::SpacetimeEvent B{
        /* time     */ static_cast<double>(event.timestamp_ms) / 1000.0,
        /* price    */ event.price,
        /* volume   */ event.volume,
        /* momentum */ momentum
    };

    auto interval_opt = srfm::manifold::SpacetimeInterval::compute(A, B);
    if (!interval_opt.has_value()) {
        prev_event_ = event;
        return std::nullopt;
    }

    auto itype = srfm::manifold::SpacetimeInterval::classify(*interval_opt);

    if (itype != srfm::manifold::IntervalType::Timelike) {
        ++spacelike_count_;
        prev_event_ = event;
        return std::nullopt;
    }

    ++timelike_count_;
    prev_event_ = event;

    // Simple momentum signal: price increased → buy, decreased → sell.
    if (std::abs(momentum) < threshold_) return std::nullopt;

    Order order;
    order.symbol       = event.symbol;
    order.quantity     = base_qty_;
    order.order_type   = OrderType::Market;
    order.timestamp_ms = event.timestamp_ms;
    order.side         = (momentum > 0.0) ? OrderSide::Buy : OrderSide::Sell;

    return order;
}

std::optional<Order> RelativisticStrategy::on_trade(const BacktestEvent& event) {
    return evaluate(event);
}

std::optional<Order> RelativisticStrategy::on_bar(const BacktestEvent& event) {
    return evaluate(event);
}

} // namespace srfm::event_bt
