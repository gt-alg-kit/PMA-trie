// REGRESSION: parlay's block_allocator::allocate_blocks allocates an aligned buffer and then records it in
// allocated_buffers (a hazptr_stack push that allocates a node with plain operator new). If that bookkeeping
// allocation throws, the buffer was previously leaked (found by LeakSanitizer in the DFUDS double-trie
// transactional allocation-failure test, docs/id_capacity_terminal_ratio_progress_2026-10-22.md 5.2). The vendored
// allocator now releases the buffer before rethrowing.
//
// Deterministic injection: once armed, the FIRST plain operator new that follows an aligned operator new on the
// arming thread throws std::bad_alloc -- in alloc() on an empty allocator that is exactly the bookkeeping push after
// allocate_blocks' buffer. The test counts live aligned allocations itself, so it detects the leak without relying on
// LeakSanitizer (and under ASan, LeakSanitizer checks it as well): after the failure, and again after the allocator
// is destroyed, no aligned allocation may remain live. A second alloc() then succeeds (no state was corrupted), and
// num_allocated_blocks() counts only the recorded buffer.
#include <parlay/internal/block_allocator.h>

#include <atomic>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <string>
#include <thread>

namespace {
std::atomic<bool> g_armed{false};
std::atomic<bool> g_after_aligned{false};
std::atomic<std::thread::id> g_tid{};
std::atomic<long> g_aligned_live{0};          // live aligned allocations of block-buffer size (>= kBufferBytes)
constexpr std::size_t kBufferBytes = 1 << 16;
// Unsized aligned delete cannot tell a buffer from a small object: buffers are recognised by a registry of the
// addresses handed out for buffer-sized requests.
constexpr int kMaxBuf = 64;
std::atomic<void*> g_bufs[kMaxBuf];   // the allocator's lists are ~256 KiB; its other aligned objects are small
std::atomic<long> g_injected{0};
std::size_t g_checks = 0, g_fail = 0;
void check(bool ok, const std::string& what) {
    ++g_checks;
    if (!ok) { ++g_fail; std::printf("FAIL: %s\n", what.c_str()); }
}
bool on_arming_thread() { return g_armed.load(std::memory_order_acquire) && std::this_thread::get_id() == g_tid.load(); }
}  // namespace

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    if (on_arming_thread() && g_after_aligned.exchange(false)) { g_armed = false; ++g_injected; throw std::bad_alloc{}; }
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { return ::operator new(n); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept { try { return ::operator new(n); } catch (...) { return nullptr; } }
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept { try { return ::operator new(n); } catch (...) { return nullptr; } }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
void* operator new(std::size_t n, std::align_val_t a) {
    const std::size_t al = static_cast<std::size_t>(a);
    void* p = std::aligned_alloc(al, (n + al - 1) / al * al);
    if (!p) throw std::bad_alloc{};
    if (n >= kBufferBytes) {
        ++g_aligned_live;
        for (auto& b : g_bufs) { void* z = nullptr; if (b.compare_exchange_strong(z, p)) break; }
    }
    if (on_arming_thread()) g_after_aligned = true;
    return p;
}
void* operator new[](std::size_t n, std::align_val_t a) { return ::operator new(n, a); }
void operator delete(void* p, std::align_val_t) noexcept {
    if (!p) return;
    for (auto& b : g_bufs) { void* q = p; if (b.compare_exchange_strong(q, nullptr)) { --g_aligned_live; break; } }
    std::free(p);
}
void operator delete[](void* p, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete(void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
void operator delete[](void* p, std::size_t, std::align_val_t a) noexcept { ::operator delete(p, a); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

int main() {
    const long base = g_aligned_live.load();
    {
        parlay::internal::block_allocator ba(64);
        g_tid = std::this_thread::get_id();
        g_after_aligned = false;
        g_armed.store(true, std::memory_order_release);
        bool threw = false;
        try { (void)ba.alloc(); } catch (const std::bad_alloc&) { threw = true; }
        g_armed = false;
        check(g_injected.load() == 1, "the bookkeeping allocation after the aligned buffer was injected exactly once");
        check(threw, "alloc() propagates the injected std::bad_alloc");
        check(g_aligned_live.load() == base, "the aligned buffer is released when recording it throws (live aligned "
              "allocations " + std::to_string(g_aligned_live.load() - base) + ")");
        check(ba.num_allocated_blocks() == 0, "an unrecorded buffer is not counted as allocated");
        void* p = ba.alloc();                                   // nothing injected: the allocator still works
        check(p != nullptr, "a later alloc() succeeds");
        check(ba.num_allocated_blocks() > 0, "the recorded buffer is counted");
        ba.free(p);
    }
    check(g_aligned_live.load() == base, "every aligned buffer is released when the allocator is destroyed");
    std::printf("parlay_alloc_failure: %zu checks, %zu failures\n", g_checks, g_fail);
    return g_fail ? 1 : 0;
}
