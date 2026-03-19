#pragma once
/**
 * @file  tick_source.hpp
 * @brief Abstract tick source interface + concrete implementations.
 *
 * Module:  include/srfm/stream/
 * Owner:   AGT-10  (Builder)  —  2026-03-01
 *
 * Responsibility
 * --------------
 * Abstract the byte-level transport from the TickIngester's validation and
 * ring-push logic.  The TickIngester accepts any TickSource* and calls read()
 * in a tight loop.
 *
 * Concrete implementations provided here
 * ---------------------------------------
 *   PipeTickSource   — reads raw OHLCVTick bytes from a named pipe (Windows)
 *                      or POSIX FIFO.
 *   QueueTickSource  — in-process queue for unit-testing without I/O.
 *
 * NOT Responsible For
 * -------------------
 *   • Tick validation  (TickIngester::push_tick)
 *   • Ring-buffer push (TickIngester)
 *   • Connection retry (caller manages reconnect policy)
 */

#include "tick.hpp"

#include <chrono>
#include <deque>
#include <optional>
#include <string>
#include <thread>

#ifdef _WIN32
#  define WIN32_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#else
#  include <cerrno>
#  include <cstring>
#  include <fcntl.h>
#  include <unistd.h>
#endif

namespace srfm::stream {

// ── TickSource (abstract) ─────────────────────────────────────────────────────

/**
 * @brief Abstract source of raw OHLCVTick values.
 *
 * TickIngester calls read() in its hot loop.  Implementors must be thread-safe
 * if called from the ingestion thread while other threads interact with the
 * underlying resource (this is not required by the default implementations).
 */
class TickSource {
public:
    virtual ~TickSource() noexcept = default;

    /**
     * @brief Read the next OHLCVTick from the source.
     *
     * @return std::optional<OHLCVTick>:
     *   - has_value()  → a tick was read; it may or may not pass validation.
     *   - nullopt      → no data available right now (transient) or source
     *                    closed (permanent — caller checks is_open()).
     *
     * @note Implementations must not throw.
     */
    [[nodiscard]] virtual std::optional<OHLCVTick> read() noexcept = 0;

    /**
     * @brief Whether the source is still usable.
     *
     * Returns false after a permanent error or after close() is called.
     */
    [[nodiscard]] virtual bool is_open() const noexcept = 0;

    /**
     * @brief Close the source and release any OS resources.
     *
     * Safe to call multiple times.
     */
    virtual void close() noexcept = 0;
};

// ── QueueTickSource ───────────────────────────────────────────────────────────

#ifdef SRFM_TESTING
// QueueTickSource is for unit tests only. Include this class only when
// SRFM_TESTING is defined (set by CMake for test targets).

/**
 * @brief In-process tick source backed by a std::deque.
 *
 * Designed for unit tests and the StreamEngine::inject() path.  Thread-safe
 * enough for single-producer / single-consumer usage when the producer uses
 * inject() from the test thread and the consumer calls read() from the
 * ingestion thread.  For concurrent push/pop correctness in tests the caller
 * is responsible for ensuring only one side runs at a time.
 */
class QueueTickSource : public TickSource {
public:
    QueueTickSource() noexcept = default;

    /**
     * @brief Push one tick for future reads.
     *
     * Called from the test / injection thread.
     */
    void push(OHLCVTick tick) noexcept {
        queue_.push_back(tick);
    }

    /**
     * @brief Push multiple ticks at once.
     */
    template<typename InputIt>
    void push_range(InputIt first, InputIt last) noexcept {
        for (auto it = first; it != last; ++it) {
            queue_.push_back(*it);
        }
    }

    [[nodiscard]] std::optional<OHLCVTick> read() noexcept override {
        if (queue_.empty()) return std::nullopt;
        OHLCVTick t = queue_.front();
        queue_.pop_front();
        return t;
    }

    [[nodiscard]] bool is_open() const noexcept override { return open_; }

    void close() noexcept override { open_ = false; }

    /// Number of ticks waiting in the queue.
    [[nodiscard]] std::size_t pending() const noexcept { return queue_.size(); }

private:
    std::deque<OHLCVTick> queue_;
    bool                  open_{true};
};

#endif // SRFM_TESTING

// ── PipeTickSource ────────────────────────────────────────────────────────────

/**
 * @brief Tick source that reads raw OHLCVTick bytes from a named pipe / FIFO.
 *
 * Wire protocol: little-endian binary, exactly sizeof(OHLCVTick) bytes per tick.
 * No framing header — the ingester detects malformed ticks via tick_is_valid().
 *
 * On Windows: uses a named pipe (\\\\.\\pipe\\<name>).
 * On POSIX:   uses a named FIFO at path <name>.
 */
class PipeTickSource : public TickSource {
public:
    /**
     * @brief Open the named pipe / FIFO for reading.
     *
     * @param name  Pipe name (Windows) or FIFO path (POSIX).
     *              Windows: omit the \\.\pipe\ prefix — it is added automatically.
     *              POSIX:   full path, e.g. "/tmp/srfm_ticks".
     *
     * On Windows, waits up to @p timeout_ms for the server to create the pipe.
     * On POSIX, the FIFO must already exist (call mkfifo before constructing).
     */
    explicit PipeTickSource(const std::string& name,
                            unsigned int timeout_ms = 5000) noexcept {
        open_pipe(name, timeout_ms);
    }

    ~PipeTickSource() noexcept override { close(); }

    [[nodiscard]] std::optional<OHLCVTick> read() noexcept override {
        if (!is_open()) return std::nullopt;

        OHLCVTick tick{};
        const bool ok = read_exact(reinterpret_cast<char*>(&tick),
                                   sizeof(OHLCVTick));
        if (!ok) {
            close();
            return std::nullopt;
        }
        return tick;
    }

    [[nodiscard]] bool is_open() const noexcept override {
#ifdef _WIN32
        return handle_ != INVALID_HANDLE_VALUE;
#else
        return fd_ >= 0;
#endif
    }

    void close() noexcept override {
#ifdef _WIN32
        if (handle_ != INVALID_HANDLE_VALUE) {
            CloseHandle(handle_);
            handle_ = INVALID_HANDLE_VALUE;
        }
#else
        if (fd_ >= 0) {
            ::close(fd_);
            fd_ = -1;
        }
#endif
    }

private:
    void open_pipe(const std::string& name, unsigned int timeout_ms) noexcept {
#ifdef _WIN32
        const std::string full = "\\\\.\\pipe\\" + name;
        const DWORD deadline   = GetTickCount() + timeout_ms;
        while (true) {
            handle_ = CreateFileA(full.c_str(), GENERIC_READ, 0, nullptr,
                                  OPEN_EXISTING, 0, nullptr);
            if (handle_ != INVALID_HANDLE_VALUE) break;
            if (GetLastError() != ERROR_PIPE_BUSY) break;
            if (GetTickCount() >= deadline)        break;
            WaitNamedPipeA(full.c_str(), 100);
        }
#else
        fd_ = ::open(name.c_str(), O_RDONLY | O_NONBLOCK);
        (void)timeout_ms;
#endif
    }

    bool read_exact(char* buf, std::size_t n) noexcept {
#ifdef _WIN32
        DWORD total = 0;
        while (total < static_cast<DWORD>(n)) {
            DWORD got = 0;
            if (!ReadFile(handle_, buf + total,
                          static_cast<DWORD>(n) - total, &got, nullptr)) {
                return false;
            }
            total += got;
        }
        return true;
#else
        std::size_t total = 0;
        while (total < n) {
            ssize_t got = ::read(fd_, buf + total, n - total);
            if (got <= 0) return false;
            total += static_cast<std::size_t>(got);
        }
        return true;
#endif
    }

#ifdef _WIN32
    HANDLE handle_{INVALID_HANDLE_VALUE};
#else
    int    fd_{-1};
#endif
};

// ── ReconnectingTickSource ────────────────────────────────────────────────────

/**
 * @brief Wraps any TickSource with automatic reconnection on read failure.
 *
 * Uses exponential backoff starting at @p initial_delay (default 100 ms),
 * doubling each attempt up to @p max_delay (default 30 000 ms = 30 s).
 * After a successful read the delay resets to @p initial_delay.
 *
 * Thread safety: same as the wrapped Source type.
 *
 * @par Example
 * @code{.cpp}
 * // Create a QueueTickSource wrapped with reconnect logic.
 * // (Useful in tests to simulate transient read failures.)
 * QueueTickSource inner;
 * inner.push(OHLCVTick{...});
 *
 * ReconnectingTickSource<QueueTickSource> src(
 *     std::move(inner),
 *     std::chrono::milliseconds{50},   // initial backoff
 *     std::chrono::milliseconds{5000}  // max backoff
 * );
 *
 * // Returns nullopt (and sleeps briefly) when the queue is empty;
 * // returns the tick and resets backoff on success.
 * while (true) {
 *     auto tick = src.read();
 *     if (tick) process(*tick);
 * }
 * @endcode
 */
template<typename Source>
class ReconnectingTickSource {
public:
    explicit ReconnectingTickSource(Source source,
                                    std::chrono::milliseconds initial_delay = std::chrono::milliseconds{100},
                                    std::chrono::milliseconds max_delay     = std::chrono::milliseconds{30'000})
        : source_(std::move(source))
        , initial_delay_(initial_delay)
        , max_delay_(max_delay)
        , current_delay_(initial_delay)
    {}

    /// Read the next tick. Returns std::nullopt on transient failure (after
    /// sleeping current_delay_). Reconnect is attempted automatically.
    std::optional<OHLCVTick> read() {
        auto result = source_.read();
        if (result) {
            current_delay_ = initial_delay_;  // reset on success
            return result;
        }
        // Back off and signal caller to retry
        std::this_thread::sleep_for(current_delay_);
        current_delay_ = std::min(current_delay_ * 2, max_delay_);
        return std::nullopt;
    }

    /// Reset the backoff state without reconnecting the underlying source.
    void reset_backoff() noexcept { current_delay_ = initial_delay_; }

    /// Access the underlying source (e.g. to re-open a pipe).
    Source& source() noexcept { return source_; }
    const Source& source() const noexcept { return source_; }

private:
    Source                    source_;
    std::chrono::milliseconds initial_delay_;
    std::chrono::milliseconds max_delay_;
    std::chrono::milliseconds current_delay_;
};

} // namespace srfm::stream
