#pragma once

// Exception capture for parlay tasks. parlay's scheduler does not propagate
// an exception out of a task: one escaping a stolen job terminates the
// process, one escaping an inline job leaves the scheduler mid-fork. Every
// parallel region in pmabv / trie therefore runs its task bodies through a
// ParGuard: the FIRST exception is captured, later tasks see `failed` and
// return early, and the owner rethrows AFTER the region has joined -- so a
// std::bad_alloc inside a batch merge surfaces to the caller as an ordinary
// exception (and the trie's transaction discards its candidate), instead of
// aborting.

#include <atomic>
#include <exception>
#include <mutex>
#include <utility>

namespace pmabv {

struct ParGuard {
    std::atomic<bool>  failed{false};
    std::mutex         m;
    std::exception_ptr ep;

    template <class F>
    void run(F&& f) noexcept {
        if (failed.load(std::memory_order_relaxed)) return;
        try { f(); }
        catch (...) {
            std::lock_guard<std::mutex> g(m);
            if (!failed.exchange(true)) ep = std::current_exception();
        }
    }
    void rethrow() {
        if (ep) { std::exception_ptr e = ep; ep = nullptr; std::rethrow_exception(e); }
    }
};

}  // namespace pmabv
