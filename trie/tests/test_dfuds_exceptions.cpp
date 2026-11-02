// DFUDS dictionary: the mutation failure contract (trie/dfuds/config.hpp),
// built once per mode (test_dfuds_exceptions_direct / _txn).
//   1  argument errors: an unsorted batch (the violation early, in the middle,
//      on the old preflight's 1,024-key chunk boundaries, at the end) and a
//      batch holding a key with byte 0 (the NUL-free domain; the same
//      positions) are std::invalid_argument and leave the dictionary unchanged
//      (logical encoding, key count, invariants); a zero-length batch never
//      touches its pointers;
//   2  allocation failures, injected at sampled allocation ordinals (counted
//      across all threads) of a batch insert that also forces global growth,
//      separately for the PLAIN operator new family and the ALIGNED one
//      (std::align_val_t: parlay's internal scratch allocators). The
//      allocations are counted in a fresh child process and every ordinal is
//      injected in another fresh child with the same history before the
//      armed insert, so a failure that ends the process is an observed
//      outcome, not a crash of the test:
//        completed  the armed ordinal was never reached and the insert
//                   produced the expected state (only the ordinal beyond the
//                   counted total may do this: below it, the children's
//                   histories diverged -- a test defect, reported as such);
//        absorbed   the injected allocation failed, the insert still succeeded
//                   with the expected state (a nothrow allocation with a
//                   fallback);
//        recovered  std::bad_alloc reached the caller, the state is unchanged
//                   and valid;
//        fatal      the process terminated (std::terminate / SIGABRT);
//        violation  anything else (a changed state after a throw, a wrong
//                   final state, another signal or exit code) -- never allowed.
//      Required: every ordinal below the counted total is reached (no
//      completed outcome there), and every counted kind fires at least once;
//      transactional mode, plain allocations: no fatal outcome (the strong
//      guarantee); direct mode, plain allocations: both recovered (pre-apply)
//      and fatal (apply-phase) outcomes observed. Aligned allocations: every
//      outcome is reported; a fatal one there is an allocation inside one of
//      parlay's parallel primitives, which cannot propagate an exception
//      (stated in the progress log).
#include "dfuds_test_common.hpp"
#include "dfuds_alloc_injection.hpp"   // defines the replacement operator new families

#include <cstring>

namespace {
using D = trie::dfuds::Dictionary;

// The fixture: a base set at a small block capacity and a batch that grows
// the PMA globally, with END-ADDs (prefixes of stored keys), duplicates and
// present keys.
struct Fixture {
    std::vector<std::string> base, batch;
    std::vector<std::string_view> bv;
    Fixture() {
        std::mt19937_64 g(77);
        for (int i = 0; i < 1500; ++i) base.push_back(tst::rnd(g, 26, 'a', 1, 10));
        base = tst::sorted_unique(base);
        for (int i = 0; i < 2500; ++i) batch.push_back(tst::rnd(g, 26, 'a', 1, 12));
        for (int i = 0; i < 50; ++i) batch.push_back(base[g() % base.size()].substr(0, 2));   // END-ADDs or present
        for (int i = 0; i < 50; ++i) batch.push_back(base[g() % base.size()]);                // present
        batch.push_back("");
        bv.assign(batch.begin(), batch.end());
        std::sort(bv.begin(), bv.end());
    }
    template <class Dict>
    std::unique_ptr<Dict> build() const { return Dict::from_sorted_parallel(base, 256); }
    template <class Dict>
    trie::dfuds::Logical expected() const {
        auto d = build<Dict>();
        d->insert_sorted(bv.data(), bv.size());
        return d->logical();
    }
};

template <class Dict>
void argument_errors() {
    Fixture f;
    auto d = f.build<Dict>();
    const auto before = d->logical();
    const std::size_t kc = d->key_count();
    auto unchanged = [&](const std::string& what) {
        CHECK(d->logical() == before && d->key_count() == kc, what + ": dictionary changed");
        d->check_invariants();
    };
    for (std::size_t at : {std::size_t{2}, std::size_t{500}, std::size_t{1024}, std::size_t{2048}, f.bv.size() - 1}) {
        auto v = f.bv;
        v[at] = std::string_view("\x01");                 // smaller than its predecessor
        CHECK(tst::throws_invalid([&] { d->insert_sorted(v.data(), v.size()); }),
              "unsorted batch (violation at " + std::to_string(at) + ") must throw");
        unchanged("unsorted batch at " + std::to_string(at));
        auto w = f.bv;                                     // byte 0 appended to one key: the order is kept
        const std::string nul = std::string(w[at]) + std::string(1, '\0');
        w[at] = nul;
        if (at + 1 < w.size() && !(w[at] < w[at + 1])) continue;
        CHECK(tst::throws_invalid([&] { d->insert_sorted(w.data(), w.size()); }),
              "batch with a key holding byte 0 (at " + std::to_string(at) + ") must throw");
        unchanged("NUL key at " + std::to_string(at));
    }
    CHECK(d->insert_sorted(nullptr, 0) == 0, "zero-length batch");
    unchanged("zero-length batch");
    d->contains_sorted_parallel(nullptr, 0, nullptr);
    std::vector<std::string_view> q{"b", "a"};
    std::vector<std::uint8_t> o(2);
    CHECK(tst::throws_invalid([&] { d->contains_sorted_parallel(q.data(), 2, o.data()); }), "unsorted query must throw");
    ++tst::g_cases;
}

// Child processes. Every child runs the SAME history before arming -- a fresh
// process, the fixture, one build, nothing else -- so the armed insert
// performs the same allocations in the counting child and in every injecting
// child: lazily created state (parlay's internal pools, per-thread state first
// touched by a parallel loop) would otherwise already exist in one child and
// not in the other, and its allocations would not recur. (An earlier version
// read the dictionary's logical state before arming in the injecting child
// only; in the assertion and sanitizer builds that warmed such state, so 17
// plain and all 25 aligned ordinals were never reached there.) The reference
// states -- the state before the insert, the expected state after it -- are
// computed only after the armed insert, from fresh builds of the same
// fixture (the build is deterministic).
// --count: prints "<plain> <aligned>" allocations of the armed insert, exit 0.
// --child: exit 0 = completed with the armed ordinal never reached, 11 =
// completed although the injected allocation failed (absorbed), 10 =
// bad_alloc with the state unchanged, 3 = a contract violation; a fatal
// failure ends in std::terminate (SIGABRT).
template <class Dict>
int child(int kind, std::size_t k) {
    Fixture f;
    auto d = f.build<Dict>();
    inj::arm(kind, k);
    try {
        d->insert_sorted(f.bv.data(), f.bv.size());
    } catch (const std::bad_alloc&) {
        inj::disarm();
        const auto fresh = f.build<Dict>();
        if (!(d->logical() == fresh->logical()) || d->key_count() != fresh->key_count()) return inj::kViolation;
        d->check_invariants();
        return inj::kRecovered;
    }
    const bool reached = inj::reached(kind, k);
    inj::disarm();
    if (!(d->logical() == f.expected<Dict>())) return inj::kViolation;
    d->check_invariants();
    return reached ? inj::kAbsorbed : inj::kCompleted;
}
template <class Dict>
int count_child() {
    Fixture f;
    auto d = f.build<Dict>();
    inj::arm(1, SIZE_MAX);
    d->insert_sorted(f.bv.data(), f.bv.size());
    inj::disarm();
    std::printf("%zu %zu\n", inj::count(1), inj::count(2));
    return 0;
}

template <class Dict>
void injection(const std::string& self, const char* rt) {   // rt: the child-process tag (one dictionary type)
    const auto [plain, aligned] = inj::count_allocations(self, rt);
    CHECK(plain > 10, "the insert allocates");
    const inj::Outcomes p = inj::sweep(self, rt, 1, plain, 90);
    const inj::Outcomes a = inj::sweep(self, rt, 2, aligned, 60);
    CHECK(p.violations.empty() && a.violations.empty(), std::string("contract violations:\n") + p.violations + a.violations);
    CHECK(p.completed > 0 && a.completed > 0, "an ordinal beyond the last allocation must complete");
    CHECK(p.unreached_below_total == 0 && a.unreached_below_total == 0,
          std::string("an ordinal below the counted total was never reached (the children's histories differ): plain ") +
              std::to_string(p.unreached_below_total) + ", aligned " + std::to_string(a.unreached_below_total));
    CHECK(p.recovered + p.fatal + p.absorbed > 0, "no plain injection fired");
    CHECK(aligned == 0 || a.recovered + a.fatal + a.absorbed > 0, "aligned allocations were counted but no injection fired");
    if constexpr (Dict::transactional) {
        CHECK(p.fatal == 0, "transactional: a plain allocation failure terminated the process");
        CHECK(p.recovered > 0, "transactional: no injection fired");
    } else {
        CHECK(p.recovered > 0, "direct: no recoverable (pre-apply) failure observed");
        CHECK(p.fatal > 0, "direct: no fatal (apply-phase) failure observed");
    }
    std::printf("ok injection transactional=%d tag=%s plain: %zu ordinals of %zu allocations: %zu completed, %zu absorbed, "
                "%zu recovered (unchanged), %zu fatal; aligned: %zu ordinals of %zu: %zu completed, %zu absorbed, %zu recovered, "
                "%zu fatal\n",
                int(Dict::transactional), rt, p.ordinals, plain, p.completed, p.absorbed, p.recovered, p.fatal, a.ordinals, aligned,
                a.completed, a.absorbed, a.recovered, a.fatal);
    std::fflush(stdout);
    ++tst::g_cases;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::strcmp(argv[1], "--child") == 0) return child<D>(std::atoi(argv[2]), std::stoull(argv[3]));
        if (argc == 3 && std::strcmp(argv[1], "--count") == 0) return count_child<D>();
        argument_errors<D>();
        const std::string self = inj::self_path(argv[0]);
        injection<D>(self, "dict");
        std::printf("PASS test_dfuds_exceptions transactional=%d cases=%zu checks=%zu\n", int(D::transactional),
                    tst::g_cases, tst::g_checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 2;
    }
}
