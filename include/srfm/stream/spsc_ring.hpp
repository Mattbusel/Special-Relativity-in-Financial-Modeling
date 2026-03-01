#pragma once
/**
 * @file  spsc_ring.hpp
 * @brief Lock-free Single-Producer / Single-Consumer ring buffer.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Provide a zero-heap-allocation, lock-free SPSC queue with sub-microsecond
 * throughput.  This is the sole inter-thread communication primitive in the
 * streaming pipeline.
 *
 * Guarantees
 * ----------
 *   • Deterministic: push()/pop() never allocate heap memory.
 *   • Lock-free: no mutexes, no condition variables, no spin-locks.
 *   • Cache-friendly: producer and consumer indices live on separate
 *     cache lines (alignas(64)) to eliminate false sharing.
 *   • Correct: all cross-thread visibility relies on C++11 acquire/release
 *     atomic ordering — no relaxed cross-thread loads.
 *   • Non-blocking: push() returns false when full; pop() returns nullopt when
 *     empty.  Neither ever blocks the calling thread.
 *   • noexcept: both push() and pop() are unconditionally noexcept, provided
 *     T's move-constructor and move-assignment are noexcept.
 *
 * Template Parameters
 * -------------------
 *   T    — Element type.  Must be noexcept move-constructible and
 *           noexcept move-assignable.
 *   SIZE — Capacity in elements.  Must be an exact power of two.
 *           Valid range: [2, 2^30].
 *
 * Usage
 * -----
 * @code
 *   SPSCRing<OHLCVTick, 65536> ring;
 *
 *   // Producer thread:
 *   OHLCVTick tick = ...;
 *   if (!ring.push(std::move(tick))) { ++drop_count; }
 *
 *   // Consumer thread:
 *   while (auto t = ring.pop()) { process(*t); }
 * @endcode
 *
 * NOT Responsible For
 * -------------------
 *   • Multi-producer or multi-consumer safety — use one producer, one consumer.
 *   • Blocking wait — callers must implement their own spin or yield logic.
 *   • Persistence — in-memory only.
 *
 * Memory Layout
 * -------------
 * The buffer is stored inline (std::array<T, SIZE>), so the total object size
 * is approximately SIZE * sizeof(T) + 128 bytes (two padded indices).
 * For SIZE=65536 and OHLCVTick (48 bytes): ~3 MiB per ring.
 */

#include <array>
#include <atomic>
#include <cstddef>
#include <optional>
#include <type_traits>

// Suppress C4324 (structure was padded due to alignment specifier) — padding
// between tail_ and head_ is intentional; it eliminates false sharing.
#ifdef _MSC_VER
#  pragma warning(push)
#  pragma warning(disable: 4324)
#endif

namespace srfm::stream {

// ── SPSCRing ──────────────────────────────────────────────────────────────────

/**
 * @brief Lock-free single-producer / single-consumer ring buffer.
 *
 * @tparam T    Element type (noexcept move-constructible + move-assignable).
 * @tparam SIZE Capacity; must be a power of two.
 */
template<typename T, std::size_t SIZE>
class SPSCRing {
    // ── Compile-time invariants ────────────────────────────────────────────────
    static_assert(SIZE >= 2,
        "SPSCRing: SIZE must be at least 2");
    static_assert((SIZE & (SIZE - 1u)) == 0u,
        "SPSCRing: SIZE must be a power of two");
    static_assert(SIZE <= (1u << 30u),
        "SPSCRing: SIZE exceeds safe limit (2^30)");
    static_assert(std::is_nothrow_move_constructible_v<T>,
        "SPSCRing: T must be noexcept move-constructible");
    static_assert(std::is_nothrow_move_assignable_v<T>,
        "SPSCRing: T must be noexcept move-assignable");

    static constexpr std::size_t MASK = SIZE - 1u;

public:
    // ── Capacity query ─────────────────────────────────────────────────────────

    /// Maximum number of elements the ring can hold simultaneously.
    [[nodiscard]] static constexpr std::size_t capacity() noexcept {
        return SIZE - 1u; // One slot reserved to distinguish full from empty.
    }

    // ── Producer API (call from a single producer thread only) ─────────────────

    /**
     * @brief Try to enqueue one element.
     *
     * Moves @p item into the ring.  Returns immediately without blocking.
     *
     * @param item  Element to enqueue (moved from).
     * @return true  on success.
     * @return false if the ring is full; @p item is left in a moved-from state.
     *
     * @note noexcept — never throws, never allocates.
     * @note Call from the producer thread only.
     */
    [[nodiscard]] bool push(T&& item) noexcept {
        const std::size_t tail = tail_.load(std::memory_order_relaxed);
        const std::size_t next = (tail + 1u) & MASK;

        // If the next slot equals head_, the ring is full.
        if (next == head_.load(std::memory_order_acquire)) {
            return false;
        }

        buffer_[tail] = std::move(item);

        // Release: make the written element visible to the consumer.
        tail_.store(next, std::memory_order_release);
        return true;
    }

    /**
     * @brief Try to enqueue a copy of one element.
     *
     * Convenience overload for copy-constructible types.
     *
     * @param item  Element to copy-enqueue.
     * @return true on success, false if full.
     */
    [[nodiscard]] bool push_copy(const T& item) noexcept(
            std::is_nothrow_copy_constructible_v<T>) {
        T copy{item};
        return push(std::move(copy));
    }

    // ── Consumer API (call from a single consumer thread only) ─────────────────

    /**
     * @brief Try to dequeue one element.
     *
     * Returns immediately without blocking.
     *
     * @return std::optional<T> containing the dequeued element, or
     *         std::nullopt if the ring is empty.
     *
     * @note noexcept — never throws, never allocates.
     * @note Call from the consumer thread only.
     */
    [[nodiscard]] std::optional<T> pop() noexcept {
        const std::size_t head = head_.load(std::memory_order_relaxed);

        // If head == tail_, the ring is empty.
        if (head == tail_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T item{std::move(buffer_[head])};

        // Release: make the freed slot visible to the producer.
        head_.store((head + 1u) & MASK, std::memory_order_release);
        return item;
    }

    // ── Diagnostic queries (approximate — non-atomic snapshot) ─────────────────

    /**
     * @brief Approximate number of elements currently in the ring.
     *
     * Not guaranteed to be exact under concurrent use (non-atomic snapshot of
     * two separate atomics).  Suitable for monitoring / metrics only.
     */
    [[nodiscard]] std::size_t size_approx() const noexcept {
        const std::size_t t = tail_.load(std::memory_order_acquire);
        const std::size_t h = head_.load(std::memory_order_acquire);
        return (t - h + SIZE) & MASK;
    }

    /**
     * @brief Approximate check for emptiness.
     *
     * Non-atomic snapshot — use for monitoring only.
     */
    [[nodiscard]] bool empty_approx() const noexcept {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    /**
     * @brief Approximate check for fullness.
     *
     * Non-atomic snapshot — use for monitoring only.
     */
    [[nodiscard]] bool full_approx() const noexcept {
        const std::size_t t = tail_.load(std::memory_order_acquire);
        const std::size_t h = head_.load(std::memory_order_acquire);
        return ((t + 1u) & MASK) == h;
    }

private:
    // ── Storage ────────────────────────────────────────────────────────────────

    // Producer index: next slot to write.
    // alignas(64) places this on its own cache line.
    alignas(64) std::atomic<std::size_t> tail_{0};

    // Consumer index: next slot to read.
    // alignas(64) separates from tail_ — eliminates false sharing.
    alignas(64) std::atomic<std::size_t> head_{0};

    // Element storage.  Inline array — zero heap allocation.
    std::array<T, SIZE> buffer_{};
};

} // namespace srfm::stream

#ifdef _MSC_VER
#  pragma warning(pop)
#endif
