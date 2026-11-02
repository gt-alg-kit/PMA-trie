#pragma once
// Allocation-failure injection for the DFUDS tests (test_dfuds_exceptions:
// the dictionary; test_dfuds_double_trie_exceptions: the DFUDS-front double
// trie). Include in exactly ONE translation unit of a test executable: it
// DEFINES the replacement allocation functions.
//
// The replacement operator new families count the allocations made while the
// injector is ARMED -- the PLAIN family (operator new(size_t) and its array /
// nothrow forms) and the ALIGNED one (std::align_val_t: parlay's internal
// scratch allocators) separately, across all threads -- and throw
// std::bad_alloc at one chosen ordinal of one family.
//
// Every ordinal is injected in a FRESH child process (posix_spawn of the test
// binary itself), so a failure that ends the process is an observed outcome,
// not a crash of the test. The protocol between the parent and the test's
// child entry points:
//   <self> --count <tag>           the counting child: prints "<plain> <aligned>"
//                                  allocations of the armed operation, exit 0;
//   <self> --child <kind> <k> <tag> the injecting child (kind 1 = plain, 2 =
//                                  aligned, k = the ordinal), exit codes
//                                  kCompleted (0): the armed ordinal was never
//                                    reached and the result is the expected one;
//                                  kAbsorbed (11): the injected allocation
//                                    failed, the operation still produced the
//                                    expected result (a nothrow allocation with a
//                                    fallback);
//                                  kRecovered (10): std::bad_alloc reached the
//                                    caller and the state is unchanged and valid;
//                                  kViolation (3): anything else the child could
//                                    check (a changed state after a throw, a wrong
//                                    final state);
//                                  a fatal failure ends in std::terminate (SIGABRT).
// Both children must run the SAME history before arming (a fresh process, the
// fixture, its builds, nothing else): lazily created state -- parlay's pools,
// per-thread state first touched by a parallel loop -- would otherwise exist
// in one child and not the other, and its allocations would not recur (an
// ordinal below the counted total is then never reached; sweep() reports it).

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <spawn.h>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <utility>
#include <vector>

extern char** environ;

namespace inj {
inline constexpr int kCompleted = 0, kRecovered = 10, kAbsorbed = 11, kViolation = 3;
inline std::atomic<bool> g_armed{false};
inline std::atomic<int> g_kind{1};
inline std::atomic<std::size_t> g_count[3];
inline std::atomic<std::size_t> g_fail_at{SIZE_MAX};
// Arm family `kind` (1 plain, 2 aligned) to fail its allocation number `at`
// (0-based, counted from now); both counters restart.
inline void arm(int kind, std::size_t at) {
    g_count[1].store(0); g_count[2].store(0);
    g_kind.store(kind); g_fail_at.store(at); g_armed.store(true);
}
inline void disarm() { g_armed.store(false); g_fail_at.store(SIZE_MAX); }
// Allocations of family `kind` since arm() (read after the armed operation).
inline std::size_t count(int kind) { return g_count[kind].load(); }
// Was ordinal `k` of family `kind` reached by the armed operation?
inline bool reached(int kind, std::size_t k) { return count(kind) > k; }
inline void maybe_fail(int kind) {
    if (!g_armed.load(std::memory_order_relaxed)) return;
    const std::size_t k = g_count[kind].fetch_add(1, std::memory_order_relaxed);
    if (kind == g_kind.load(std::memory_order_relaxed) && k == g_fail_at.load(std::memory_order_relaxed))
        throw std::bad_alloc{};
}
// The array forms below reject sizes no object can have before forwarding: GCC passes SIZE_MAX for an overflowing
// array new-expression, and once these definitions are inlined into a caller (parlay's ThreadSpecific allocates
// aligned arrays) the forwarded call would carry that constant (-Walloc-size-larger-than). The outcome is the
// library's: std::bad_alloc.
inline void check_array_size(std::size_t n) {
    if (n > static_cast<std::size_t>(PTRDIFF_MAX)) throw std::bad_alloc{};
}
}  // namespace inj

// LeakSanitizer (ASan builds): parlay's make_unique_array (parlay/utilities.h) is not exception-safe -- when
// an injected failure hits the lazy construction of parlay's global pool allocator (get_default_allocator ->
// pool_allocator -> make_unique_array<block_allocator>), the array allocated before it leaks (once, 4,592 bytes;
// the next use retries the construction). Third-party code, outside the dictionaries' failure contract (the
// child recovered with the structure unchanged): suppressed by name, every other leak still fails a child.
// Found 2026-10-16 (docs/dfuds_library_integration_progress_2026-10-16.md, section 15).
extern "C" const char* __lsan_default_suppressions() { return "leak:parlay::make_unique_array\n"; }

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmismatched-new-delete"
#endif
void* operator new(std::size_t n) {
    inj::maybe_fail(1);
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n) { inj::check_array_size(n); return ::operator new(n); }
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t) noexcept { std::free(p); }
void* operator new(std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new(n); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n); } catch (...) { return nullptr; }
}
void operator delete(void* p, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, const std::nothrow_t&) noexcept { std::free(p); }
// the aligned family
void* operator new(std::size_t n, std::align_val_t al) {
    inj::maybe_fail(2);
    std::size_t a = static_cast<std::size_t>(al);
    if (a < sizeof(void*)) a = sizeof(void*);
    void* p = nullptr;
    if (posix_memalign(&p, a, n ? n : 1) == 0) return p;
    throw std::bad_alloc{};
}
void* operator new[](std::size_t n, std::align_val_t al) { inj::check_array_size(n); return ::operator new(n, al); }
void* operator new(std::size_t n, std::align_val_t al, const std::nothrow_t&) noexcept {
    try { return ::operator new(n, al); } catch (...) { return nullptr; }
}
void* operator new[](std::size_t n, std::align_val_t al, const std::nothrow_t&) noexcept {
    try { return ::operator new[](n, al); } catch (...) { return nullptr; }
}
void operator delete(void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t) noexcept { std::free(p); }
void operator delete[](void* p, std::size_t, std::align_val_t) noexcept { std::free(p); }
void operator delete(void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
void operator delete[](void* p, std::align_val_t, const std::nothrow_t&) noexcept { std::free(p); }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

namespace inj {

// The absolute path of the running test binary (the children re-execute it).
inline std::string self_path(const char* argv0) {
    char buf[4096];
    const ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n > 0) { buf[n] = 0; return buf; }
    return argv0;
}

// Allocations of each family made by the armed operation of a fresh counting
// child (`<self> --count <tag>`). Throws std::runtime_error when the child fails.
inline std::pair<std::size_t, std::size_t> count_allocations(const std::string& self, const std::string& tag) {
    int fd[2];
    if (pipe(fd) != 0) throw std::runtime_error("pipe failed");
    char* argv[] = {const_cast<char*>(self.c_str()), const_cast<char*>("--count"), const_cast<char*>(tag.c_str()), nullptr};
    posix_spawn_file_actions_t fa;
    posix_spawn_file_actions_init(&fa);
    posix_spawn_file_actions_adddup2(&fa, fd[1], 1);
    posix_spawn_file_actions_addclose(&fa, fd[0]);
    pid_t pid;
    const int e = posix_spawn(&pid, self.c_str(), &fa, nullptr, argv, environ);
    posix_spawn_file_actions_destroy(&fa);
    if (e != 0) throw std::runtime_error(std::string("posix_spawn: ") + std::strerror(e));
    close(fd[1]);
    std::string out;
    char buf[256];
    for (ssize_t r; (r = read(fd[0], buf, sizeof buf)) > 0;) out.append(buf, static_cast<std::size_t>(r));
    close(fd[0]);
    int status = 0;
    if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        throw std::runtime_error("counting child failed (tag " + tag + ")");
    std::size_t plain = 0, aligned = 0;
    if (std::sscanf(out.c_str(), "%zu %zu", &plain, &aligned) != 2) throw std::runtime_error("counting child output: " + out);
    return {plain, aligned};
}

struct Outcomes {
    std::size_t ordinals = 0, completed = 0, absorbed = 0, recovered = 0, fatal = 0;
    std::size_t unreached_below_total = 0;   // completed below the counted total: the histories differ
    std::string violations;                  // one line per violating ordinal (empty: none)
};

// Inject family `kind` at sampled ordinals of `total` counted allocations
// (every (total / per)-th, the last 25, and one beyond the total, which must
// complete), each in a fresh child `<self> --child <kind> <k> <tag>` with its
// stderr silenced (a fatal child's terminate message is expected).
inline Outcomes sweep(const std::string& self, const std::string& tag, int kind, std::size_t total, std::size_t per) {
    std::vector<std::size_t> ords;
    if (total) {
        const std::size_t step = std::max<std::size_t>(1, total / per);
        for (std::size_t k = 0; k < total; k += step) ords.push_back(k);
        for (std::size_t k = total > 25 ? total - 25 : 0; k < total; ++k) ords.push_back(k);
        std::sort(ords.begin(), ords.end());
        ords.erase(std::unique(ords.begin(), ords.end()), ords.end());
    }
    ords.push_back(total + 2);
    Outcomes o;
    for (std::size_t k : ords) {
        const std::string ks = std::to_string(k), kinds = std::to_string(kind);
        char* argv[] = {const_cast<char*>(self.c_str()), const_cast<char*>("--child"), const_cast<char*>(kinds.c_str()),
                        const_cast<char*>(ks.c_str()), const_cast<char*>(tag.c_str()), nullptr};
        pid_t pid;
        posix_spawn_file_actions_t fa;
        posix_spawn_file_actions_init(&fa);
        posix_spawn_file_actions_addopen(&fa, 2, "/dev/null", O_WRONLY, 0);
        const int e = posix_spawn(&pid, self.c_str(), &fa, nullptr, argv, environ);
        posix_spawn_file_actions_destroy(&fa);
        if (e != 0) throw std::runtime_error(std::string("posix_spawn: ") + std::strerror(e));
        int status = 0;
        if (waitpid(pid, &status, 0) != pid) throw std::runtime_error("waitpid failed");
        ++o.ordinals;
        if (WIFEXITED(status) && WEXITSTATUS(status) == kCompleted) { ++o.completed; if (k < total) ++o.unreached_below_total; }
        else if (WIFEXITED(status) && WEXITSTATUS(status) == kAbsorbed) ++o.absorbed;
        else if (WIFEXITED(status) && WEXITSTATUS(status) == kRecovered) ++o.recovered;
        else if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) ++o.fatal;
        else o.violations += "tag " + tag + " kind " + kinds + " ordinal " + ks + ": status " + std::to_string(status) + "\n";
    }
    return o;
}

}  // namespace inj
