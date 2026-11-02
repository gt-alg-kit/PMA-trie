// DFUDS-front DoubleTrie: the mutation failure contract (trie/dfuds/config.hpp
// via trie/dfuds_double_trie.hpp), built once per mode
// (test_dfuds_double_trie_exceptions_direct / _txn):
//   1  argument errors: an unsorted batch (the violation early, in the middle,
//      at the end) is std::invalid_argument and leaves the trie unchanged
//      (front, rear, ids, key count, invariants);
//   2  allocation failures (tests/dfuds_alloc_injection.hpp), injected at
//      sampled ordinals of the PLAIN and the ALIGNED operator new families of
//      one batch insert that exercises every phase: the front's insertion
//      walk and planner, rear additions (fresh ids, in-place flags, a rear CSR, an id
//      width growth of both tries), front flag / has-id writes and the front
//      CSR, rear releases (affected leaves drop their old remainders, some to
//      zero owners), and global growth of both PMAs (block capacity 64).
//      Outcomes per ordinal: completed (never reached), absorbed, recovered
//      (std::bad_alloc, the trie unchanged and valid), fatal (std::terminate),
//      violation (never allowed). The unchanged / expected states are checked
//      position by position against fresh builds of the same fixture, made
//      after the armed insert (the builds are deterministic).
//      Required: no ordinal below the counted total unreached; every counted
//      family fires at least once; transactional mode, plain allocations: no
//      fatal outcome (the strong guarantee over front, rear, ids, handles and
//      owner counts together); direct mode, plain allocations: both recovered
//      (preflight) and fatal (apply) outcomes. Aligned allocations: every
//      outcome reported (a fatal one = an allocation inside a parlay parallel
//      primitive, which cannot propagate an exception).
#include "dfuds_dt_test_common.hpp"
#include "dfuds_alloc_injection.hpp"   // defines the replacement operator new families

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

namespace {
std::size_t g_checks = 0;
[[noreturn]] void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::fflush(stderr);
    std::exit(1);
}
void check(bool ok, const std::string& what) { ++g_checks; if (!ok) fail(what); }

using DD = dtt::DD;

// The fixture: few distinct remainders at the start (a narrow id width), a
// batch that adds many (width growth), lands on remainder leaves (affected
// leaves, keep-flag cases), flips internal nodes, repeats present keys, adds
// the empty key and remainders shared by several keys.
struct Fixture {
    std::vector<std::string> base, batch;
    std::vector<std::string_view> bv;
    Fixture() {
        std::mt19937_64 g(91);
        for (int i = 0; i < 6; ++i) base.push_back("host" + std::to_string(i) + "/path/to/item_" + std::to_string(i));
        for (int i = 0; i < 40; ++i) base.push_back("w" + std::to_string(g() % 50) + "x");
        std::sort(base.begin(), base.end());
        base.erase(std::unique(base.begin(), base.end()), base.end());
        for (int i = 0; i < 700; ++i)
            batch.push_back("k" + std::to_string(g() % 100000) + (g() % 3 ? "_tail" + std::to_string(g() % 400) : std::string()));
        for (int i = 0; i < 6; ++i) batch.push_back("host" + std::to_string(i) + "/path");          // prefixes of stored keys
        for (int i = 0; i < 6; ++i) batch.push_back("host" + std::to_string(i) + "/path/to/item_X"); // diverge inside remainders
        for (int i = 0; i < 20; ++i) batch.push_back(base[g() % base.size()]);                        // present
        for (int i = 0; i < 20; ++i) batch.push_back("q" + std::to_string(i) + "/shared/ending.txt");  // shared remainders
        batch.push_back("w");                                                                          // flips
        batch.push_back("");
        bv.assign(batch.begin(), batch.end());
        std::sort(bv.begin(), bv.end());
    }
    std::unique_ptr<DD> build() const { return DD::from_sorted_parallel(base, 64); }
    std::unique_ptr<DD> expected() const {
        auto d = build();
        d->insert_sorted(bv.data(), bv.size());
        return d;
    }
};

void argument_errors() {
    Fixture f;
    auto d = f.build();
    const auto ref = f.build();
    for (std::size_t at : {std::size_t{1}, f.bv.size() / 2, f.bv.size() - 1}) {
        auto v = f.bv;
        v[at - 1] = std::string_view("\xff\xff");          // greater than its successor (the fixture's keys are ASCII)
        bool threw = false;
        try { d->insert_sorted(v.data(), v.size()); } catch (const std::invalid_argument&) { threw = true; }
        check(threw, "unsorted batch (violation at " + std::to_string(at) + ") must throw std::invalid_argument");
        std::string why;
        const bool same = dtt::same_state(*d, *ref, why);   // before the message reads `why`
        check(same, "unsorted batch at " + std::to_string(at) + ": trie changed: " + why);
        d->check_invariants();
    }
}

// --child kind k: see tests/dfuds_alloc_injection.hpp (the same history as the
// counting child: the fixture and one build, then the armed insert).
int child(int kind, std::size_t k) {
    Fixture f;
    auto d = f.build();
    inj::arm(kind, k);
    try {
        d->insert_sorted(f.bv.data(), f.bv.size());
    } catch (const std::bad_alloc&) {
        inj::disarm();
        std::string why;
        if (!dtt::same_state(*d, *f.build(), why)) return inj::kViolation;
        d->check_invariants();
        return inj::kRecovered;
    }
    const bool reached = inj::reached(kind, k);
    inj::disarm();
    std::string why;
    if (!dtt::same_state(*d, *f.expected(), why)) return inj::kViolation;
    d->check_invariants();
    return reached ? inj::kAbsorbed : inj::kCompleted;
}
int count_child() {
    Fixture f;
    auto d = f.build();
    inj::arm(1, SIZE_MAX);
    d->insert_sorted(f.bv.data(), f.bv.size());
    inj::disarm();
    std::printf("%zu %zu\n", inj::count(1), inj::count(2));
    return 0;
}

void injection(const std::string& self) {
    {   // the fixture's insert does what the header says
        Fixture f;
        auto d = f.build();
        const unsigned w0 = d->id_width();
        const std::size_t r0 = d->distinct_remainders();
        d->insert_sorted(f.bv.data(), f.bv.size());
        check(d->id_width() > w0, "the fixture's batch widens the ids");
        check(d->distinct_remainders() > r0, "the fixture's batch adds remainders");
        d->check_invariants();
    }
    const auto [plain, aligned] = inj::count_allocations(self, "dt");
    check(plain > 10, "the insert allocates");
    const inj::Outcomes p = inj::sweep(self, "dt", 1, plain, 120);
    const inj::Outcomes a = inj::sweep(self, "dt", 2, aligned, 60);
    check(p.violations.empty() && a.violations.empty(), "contract violations:\n" + p.violations + a.violations);
    check(p.completed > 0 && a.completed > 0, "an ordinal beyond the last allocation must complete");
    check(p.unreached_below_total == 0 && a.unreached_below_total == 0,
          "an ordinal below the counted total was never reached (the children's histories differ): plain " +
              std::to_string(p.unreached_below_total) + ", aligned " + std::to_string(a.unreached_below_total));
    check(p.recovered + p.fatal + p.absorbed > 0, "no plain injection fired");
    check(aligned == 0 || a.recovered + a.fatal + a.absorbed > 0, "aligned allocations were counted but no injection fired");
    if constexpr (DD::transactional) {
        check(p.fatal == 0, "transactional: a plain allocation failure terminated the process");
        check(p.recovered > 0, "transactional: no injection fired");
    } else {
        check(p.recovered > 0, "direct: no recoverable (preflight) failure observed");
        check(p.fatal > 0, "direct: no fatal (apply-phase) failure observed");
    }
    std::printf("ok injection transactional=%d plain: %zu ordinals of %zu allocations: %zu completed, %zu absorbed, "
                "%zu recovered (unchanged), %zu fatal; aligned: %zu ordinals of %zu: %zu completed, %zu absorbed, "
                "%zu recovered, %zu fatal\n",
                int(DD::transactional), p.ordinals, plain, p.completed, p.absorbed, p.recovered, p.fatal, a.ordinals,
                aligned, a.completed, a.absorbed, a.recovered, a.fatal);
    std::fflush(stdout);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 5 && std::strcmp(argv[1], "--child") == 0) return child(std::atoi(argv[2]), std::stoull(argv[3]));
        if (argc == 3 && std::strcmp(argv[1], "--count") == 0) return count_child();
        argument_errors();
        injection(inj::self_path(argv[0]));
        std::printf("PASS test_dfuds_double_trie_exceptions transactional=%d checks=%zu\n", int(DD::transactional), g_checks);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "ERROR %s\n", e.what());
        return 2;
    }
}
