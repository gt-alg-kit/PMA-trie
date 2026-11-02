// Global DT handles / owner counts follow their growth policy: the rear
// terminal-capacity ratio (default) or an explicit reserve percentage
// (trie/dt/table_growth.hpp). Check actual allocated capacity, reserve
// consumption and exhaustion, copy and width changes, ID reuse, policy
// switches, and the public sequential / parallel builds of the double trie.
#include "trie/dfuds_double_trie.hpp"
#include "dfuds_dt_test_common.hpp"

#include <algorithm>
#include <cstdio>
#include <functional>
#include <limits>
#include <span>
#include <string_view>
#include <random>
#include <set>
#include <string>
#include <vector>

namespace {
namespace dd = trie::dt::detail;
using Growth = trie::dt::TableGrowth;
using DfDT = trie::dfuds::DoubleTrie;
std::size_t checks = 0, failures = 0;
void check(bool ok, const std::string& what) {
    ++checks;
    if (!ok) { ++failures; std::printf("FAIL: %s\n", what.c_str()); }
}
std::size_t bytes_for(std::size_t n, unsigned w) { return ((n * w + 63) / 64 + 1) * 8; }

void packed_fields() {
    for (unsigned w : {1U, 7U, 14U, 16U, 25U, 64U}) {
        dd::PackedFields p(w);
        p.reserve_fields(2000);
        p.resize(1000);
        for (std::size_t i = 0; i < p.size(); ++i) p.set(i, i % 2);
        const auto bytes = p.bytes(), cap = p.capacity_fields();
        dd::PackedFields copy(p), assigned;
        assigned = p;
        check(copy.bytes() == bytes && assigned.bytes() == bytes, "copy and assignment keep reserve");
        copy.resize(cap);
        check(copy.bytes() == bytes, "reserve is consumed without reallocating");
        bool equal = true;
        for (std::size_t i = 0; i < cap; ++i) equal &= copy.get(i) == (i < 1000 ? i % 2 : 0);
        check(equal, "old values and new zero fields survive growth");
        copy.resize(11);
        copy.resize(cap);
        equal = true;
        for (std::size_t i = 11; i < cap; ++i) equal &= copy.get(i) == 0;
        check(equal, "shrink then regrow never exposes stale fields");
        if (w < 64) {
            p.rewiden(w + 1);
            check(p.capacity_fields() >= cap && p.size() == 1000, "rewiden retains field reserve and live size");
            equal = true;
            for (std::size_t i = 0; i < p.size(); ++i) equal &= p.get(i) == i % 2;
            check(equal, "rewiden preserves every live field");
        }
    }
    dd::PackedFields p(64);
    p.resize(1); p.set(0, 123);
    const auto before = p.bytes();
    bool threw = false;
    try { p.reserve_fields(std::numeric_limits<std::size_t>::max()); }
    catch (const std::length_error&) { threw = true; }
    check(threw && p.get(0) == 123 && p.size() == 1 && p.bytes() == before,
          "overflow is rejected without changing the table");
}

struct RearFixture {
    std::vector<std::size_t> n, c;
    std::size_t block_count() const { return n.size(); }
    std::size_t block_label_count(std::size_t b) const { return n[b]; }
    std::size_t label_capacity_of(std::size_t b) const { return c[b]; }
};
void target_and_reuse() {
    // Independent arithmetic, including a consumed reserve, excess actual
    // capacity, an empty block, and the sentinel that is not a remainder ID.
    RearFixture rear{{8192, 4000, 0}, {8192, 4352, 0}};
    const std::size_t c0 = ((8192 + block::kLabelGrowthSlack + 127) / 128) * 128;
    const std::size_t c1 = std::max(std::size_t{4352}, ((4000 + block::kLabelGrowthSlack + 127) / 128) * 128);
    const std::size_t target = 1 + (10000 * (c0 + c1) + 12191) / 12192;
    check(dd::id_table_capacity_target(rear, 10001) == target, "terminal ratio is weighted by entries, with replenishment");
    check(dd::id_table_capacity_target(rear, 1) == 1, "no reserve for an empty ID namespace");
    const std::size_t next_n = 12192 + 3000;
    const std::size_t next_c = ((8192 + 3000 + block::kLabelGrowthSlack + 127) / 128) * 128 + c1;
    const std::size_t next_target = 1 + (12000 * next_c + next_n - 1) / next_n;
    check(dd::id_table_capacity_target(rear, 12001, {{0, 3000}}) == next_target,
          "preflight uses planned terminal entries and existing capacity before redistribution");
    struct State { RearFixture rear; dd::IdTable ids; dd::PackedFields hblock{14}; Growth growth; } s{rear, {}, dd::PackedFields(14), {}};
    dd::reserve_id_tables(s, 10001);
    s.ids.assign(std::vector<std::uint64_t>(10001, 1));
    s.hblock.resize(10001);
    const auto owner_bytes = s.ids.owner_bytes(), handle_bytes = s.hblock.bytes();
    check(owner_bytes == bytes_for(target, 16) && handle_bytes == bytes_for(target, 14), "explicit capacity has no vector doubling");
    bool grew = false;
    const auto id = s.ids.acquire(grew);
    s.hblock.resize(s.ids.size());
    check(grew && id == 10001 && s.ids.owner_bytes() == owner_bytes && s.hblock.bytes() == handle_bytes,
          "first new ID consumes preallocated slots in both tables");
    s.ids.set_owners(id, 2);
    s.ids.release(id);
    bool planned = false;
    dd::prepare_id_tables(s, 1, [&] { planned = true; return std::vector<dd::TerminalGrowth>{}; });
    check(!planned && s.ids.free_ids().back() == id && s.ids.owner_bytes() == owner_bytes,
          "preparation counts reusable IDs without scanning or consuming the free list");
    const auto reused = s.ids.acquire(grew);
    check(!grew && reused == id && s.ids.owners(id) == 0 && s.ids.size() == 10002,
          "released IDs are reused without growing the namespace");
    auto copy = s;
    check(copy.ids.owner_bytes() == owner_bytes && copy.hblock.bytes() == handle_bytes,
          "a state copy preserves both global reserves");
    // Consume the reserve and replenish repeatedly. Every old field must survive.
    s.hblock.set(7, 777);
    std::size_t growths = 0;
    for (std::size_t k = 0; k < 3000; ++k) {
        const auto slots = s.ids.size() + 1;
        if (slots > s.ids.capacity_slots() || slots > s.hblock.capacity_fields()) {
            dd::reserve_id_tables(s, slots);
            ++growths;
            check(s.ids.owner_bytes() == bytes_for(dd::id_table_capacity_target(rear, slots), 16),
                  "each replenishment uses the terminal ratio");
        }
        s.ids.acquire(grew);
        s.hblock.resize(s.ids.size());
    }
    check(growths >= 2 && s.hblock.get(7) == 777 && s.ids.owners(7) == 1,
          "multiple replenishments keep handles and ownership intact");
}

template <class DT> std::pair<std::size_t, std::size_t> table_bytes(const DT& d) {
    const auto s = d.space();
    std::size_t handles = 0, owners = 0;
    for (const auto& row : s.rows) {
        if (row.component == "handle_table") handles = row.value;
        if (row.component == "owner_counts") owners = row.value;
    }
    return {handles, owners};
}
std::vector<std::string> keys(std::size_t n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::set<std::string> all;
    while (all.size() < n) {
        std::string key = "http://example.org/";
        for (int j = 0; j < 28; ++j) key += static_cast<char>('a' + rng() % 26);
        all.insert(std::move(key));
    }
    return {all.begin(), all.end()};
}
template <class DT> void public_builds(const char* name) {
    const auto base = keys(5000, 17), more = keys(3000, 99);
    {
        DT d(16384);
        d.insert_sorted(base);
        // The empty rear has one receiving block. Its preflight reference
        // contains all inserted entries, before PMA splitting into new blocks.
        const auto n = d.rear().label_count();
        const auto c = ((n + block::kLabelGrowthSlack + 127) / 128) * 128;
        const auto target = 1 + (d.distinct_remainders() * c + n - 1) / n;
        const auto actual = table_bytes(d);
        check(actual.second == bytes_for(target, 16), std::string(name) + ": empty-tree batch keeps its proportional preflight reserve");
        bool found = true;
        for (const auto& k : base) found &= d.contains(k);
        check(found, std::string(name) + ": empty-tree batch preserves every key");
        d.check_invariants();
    }
    for (bool parallel : {false, true}) {
        auto d = parallel ? DT::from_sorted_parallel(base, 16384) : DT::from_sorted(base, 16384);
        const std::string tag = std::string(name) + (parallel ? " parallel" : " sequential");
        std::size_t n = 0, c = 0;
        for (std::size_t b = 0; b < d->rear().block_count(); ++b) {
            n += d->rear().block_label_count(b);
            c += d->rear().label_capacity_of(b);
        }
        const auto f = d->distinct_remainders();
        const auto target = 1 + (f * c + n - 1) / n;
        const auto w = dd::bits_for(d->rear().block_count());
        const auto before = table_bytes(*d);
        check(before.second == bytes_for(target, 16), tag + ": build reserves owners at terminal ratio");
        check(before.first == bytes_for(target, w), tag + ": build reserves handles at terminal ratio and final width");
        { const std::string_view first_key = more[0]; d->insert_sorted(&first_key, 1); }
        const auto first = table_bytes(*d);
        check(first.second == before.second, tag + ": first insertion does not double owner capacity");
        if (dd::bits_for(d->rear().block_count()) == w)
            check(first.first == before.first, tag + ": first insertion consumes handle reserve");
        d->insert_sorted(std::vector<std::string>(more.begin() + 1, more.end()));
        bool found = true;
        for (const auto& k : base) found &= d->contains(k);
        for (const auto& k : more) found &= d->contains(k);
        check(found, tag + ": all old and new keys survive growth");
        try { d->check_invariants(); check(true, tag + ": invariants"); }
        catch (const std::exception& e) { check(false, tag + ": " + e.what()); }
    }
}

template <class DT> std::unique_ptr<DT> seq_build(const std::vector<std::string>& k, std::size_t bc, Growth g) {
    return DT::from_sorted(k, bc, g);
}
template <class DT> void stable_ids_and_reuse(const char* label, Growth g) {
    const std::string name = std::string(label) + " growth=" + g.to_string();
    auto d = seq_build<DT>(std::vector<std::string>{"aONE", "bTWO", "cTHREE"}, 64, g);
    auto add = [&](const std::vector<std::string>& ks) { d->insert_sorted(ks); };
    const auto original = dtt::rear_map(d->rear());
    const auto freed = original.at("ONE");
    add({"a"});                    // ONE loses its last owner; NE remains live.
    const auto before = dtt::rear_map(d->rear());
    check(!before.contains("ONE") && before.at("TWO") == original.at("TWO") &&
          before.at("THREE") == original.at("THREE"), std::string(name) + ": unrelated live IDs survive release");
    const auto tables = table_bytes(*d);
    const auto slots = d->table_capacity().slots;
    check(d->table_capacity().free_ids == 1, name + ": the release leaves one reusable ID");
    add({"dFOUR"});
    check(d->table_capacity().slots == slots && d->table_capacity().free_ids == 0,
          name + ": reuse pops the free list without extending the namespace");
    const auto after = dtt::rear_map(d->rear());
    check(after.at("FOUR") == freed, std::string(name) + ": previously released ID is reused first");
    check(table_bytes(*d) == tables, std::string(name) + ": ID reuse does not grow either table");
    add({"eFIVE", "fSIX", "gSEVEN"});
    const auto final = dtt::rear_map(d->rear());
    bool stable = true;
    for (const auto& [rem, id] : after) stable &= final.at(rem) == id;
    check(stable, std::string(name) + ": later one-key and multi-key batches never renumber live IDs");
    for (const auto& k : {"a", "aONE", "bTWO", "cTHREE", "dFOUR", "eFIVE", "fSIX", "gSEVEN"})
        stable &= d->contains(k);
    check(stable, std::string(name) + ": reuse and growth preserve all keys");
    d->check_invariants();
}

// ---- explicit-percentage policy -------------------------------------------------------
std::size_t ref_percent(std::size_t h, std::uint64_t p) { return h <= 1 ? h : 1 + (h - 1) + ((h - 1) * p + 99) / 100; }
template <class F> bool throws_as(F&& f, const std::function<bool(const std::exception&)>& kind) {
    try { f(); } catch (const std::exception& e) { return kind(e); }
    return false;
}
bool is_invalid(const std::exception& e) { return dynamic_cast<const std::invalid_argument*>(&e) != nullptr; }
bool is_length(const std::exception& e) { return dynamic_cast<const std::length_error*>(&e) != nullptr; }

void growth_policy_values() {
    check(Growth().is_terminal() && Growth::terminal().is_terminal() && Growth::parse("terminal").is_terminal() &&
          Growth().to_string() == "terminal", "terminal is the default and parses");
    for (const auto& [text, p] : std::vector<std::pair<std::string, std::uint64_t>>{{"2", 2}, {"10", 10}, {"100", 100}, {"007", 7},
                                                                                   {"18446744073709551615", ~0ULL}}) {
        const Growth g = Growth::parse(text);
        check(!g.is_terminal() && g.reserve_percent() == p && g == Growth::percent(p) && g.to_string() == std::to_string(p),
              "parse '" + text + "'");
    }
    for (const char* bad : {"", "0", "00", "-1", "+5", "10%", " 10", "10 ", "1.5", "1e2", "Terminal", "terminal ",
                            "18446744073709551616"})
        check(throws_as([&] { Growth::parse(bad); }, is_invalid), std::string("reject '") + bad + "'");
    check(throws_as([] { Growth::percent(0); }, is_invalid), "reject percent(0)");
    check(throws_as([] { dd::percent_capacity_target(10, 0); }, is_invalid), "target rejects a zero percentage");
    bool same = true;
    for (std::size_t h : {0UL, 1UL, 2UL, 3UL, 101UL, 102UL, 10001UL, 123457UL})
        for (std::uint64_t p : {1ULL, 2ULL, 10ULL, 33ULL, 100ULL, 250ULL})
            same &= dd::percent_capacity_target(h, p) == ref_percent(h, p);
    check(same, "percentage target = 1 + (H - 1) + ceil((H - 1) * p / 100)");
    check(dd::percent_capacity_target(101, 10) == 111 && dd::percent_capacity_target(102, 10) == 113 &&
          dd::percent_capacity_target(101, 100) == 201 && dd::percent_capacity_target(2, 2) == 3 &&
          dd::percent_capacity_target(1, 100) == 1, "percentage examples");
    check(dd::percent_capacity_target(2, ~0ULL) == 2 + 184467440737095517ULL, "128-bit product of a huge percentage");
    const std::size_t big = std::numeric_limits<std::size_t>::max();
    check(throws_as([&] { dd::percent_capacity_target(big, 1); }, is_length) &&
          throws_as([&] { dd::percent_capacity_target(big / 2 + 2, 100); }, is_length), "target overflow is a length_error");
}

// A rear that counts the scans of the terminal-ratio reference.
struct CountingRear : RearFixture {
    mutable std::size_t scans = 0;
    std::size_t block_count() const { ++scans; return n.size(); }
};
void percent_tables() {
    struct State { CountingRear rear; dd::IdTable ids; dd::PackedFields hblock{14}; Growth growth; } s;
    s.rear.n = {100, 50}; s.rear.c = {256, 128};
    s.growth = Growth::percent(10);
    dd::reserve_id_tables(s, 101);                          // the build's reserve
    s.ids.assign(std::vector<std::uint64_t>(101, 1));
    s.hblock.resize(101);
    check(s.ids.owner_bytes() == bytes_for(111, 16) && s.hblock.bytes() == bytes_for(111, 14) && s.rear.scans == 0,
          "percent build reserve: exact target in both tables, no rear scan");
    bool planned = false;
    auto plan = [&] { planned = true; return std::vector<dd::TerminalGrowth>{{0, 5}}; };
    std::size_t growths = 0;
    bool exact = true;
    s.hblock.set(7, 777);
    for (std::size_t k = 0; k < 400; ++k) {
        const std::size_t before = s.ids.owner_bytes();
        dd::prepare_id_tables(s, 1, plan);
        if (s.ids.owner_bytes() != before) {
            ++growths;
            const std::size_t t = ref_percent(s.ids.size() + 1, 10);
            exact &= s.ids.owner_bytes() == bytes_for(t, 16) && s.hblock.bytes() == bytes_for(t, 14);
        }
        bool grew = false;
        s.ids.acquire(grew);
        s.hblock.resize(s.ids.size());
        s.ids.set_owners(s.ids.size() - 1, 1);
    }
    check(growths >= 3 && exact, "every percent exhaustion reserves exactly 1 + (H - 1) + ceil((H - 1) p / 100)");
    check(!planned && s.rear.scans == 0, "percent exhaustion computes no planned entries and scans no rear block");
    check(s.hblock.get(7) == 777 && s.ids.owners(7) == 1 && s.ids.size() == 501, "growth keeps old fields");
    // A multi-ID demand after reuse: fill both tables to capacity, then 3
    // released IDs cover 3 of 7 new ones.
    while (s.ids.size() < s.ids.capacity_slots() && s.ids.size() < s.hblock.capacity_fields()) {
        bool grew = false;
        s.ids.acquire(grew);
        s.hblock.resize(s.ids.size());
    }
    for (std::uint64_t id : {11, 12, 13}) s.ids.release(id);
    const std::size_t need = s.ids.size() + 4;
    dd::prepare_id_tables(s, 7, plan);
    check(s.ids.owner_bytes() == bytes_for(ref_percent(need, 10), 16) && s.ids.free_ids().size() == 3,
          "the percentage applies to the slots required after reusable IDs, without consuming the free list");
    // Copies keep the policy and both reserves.
    auto copy = s;
    check(copy.growth == s.growth && copy.ids.owner_bytes() == s.ids.owner_bytes() && copy.hblock.bytes() == s.hblock.bytes(),
          "a state copy preserves the policy and both reserves");
    // Handle-width repacking keeps the slot reserve.
    const std::size_t hcap = s.hblock.capacity_fields();
    s.hblock.rewiden(20);
    check(s.hblock.capacity_fields() >= hcap && s.hblock.get(7) == 777, "handle repacking preserves slot reserve and values");
    // Switching reallocates nothing; a smaller percentage never shrinks.
    const auto ob = s.ids.owner_bytes(), hb = s.hblock.bytes();
    s.growth = Growth::percent(2);
    dd::reserve_id_tables(s, s.ids.size());
    check(s.ids.owner_bytes() == ob && s.hblock.bytes() == hb, "a smaller percentage does not shrink existing tables");
    s.growth = Growth::terminal();
    dd::prepare_id_tables(s, 1, [] { return std::vector<dd::TerminalGrowth>{}; });
    check(s.ids.owner_bytes() == ob && s.rear.scans == 0, "a switch to terminal neither reallocates nor scans while reserve lasts");
    // Rejected growth leaves both tables unchanged (checked arithmetic first; then the vector limit).
    s.growth = Growth::percent(~0ULL);
    const std::size_t size0 = s.ids.size();
    const bool t1 = throws_as([&] { dd::prepare_id_tables(s, std::numeric_limits<std::size_t>::max() / 2, plan); }, is_length);
    s.growth = Growth::percent(1);
    const bool t2 = throws_as([&] { dd::reserve_id_tables(s, std::numeric_limits<std::size_t>::max() / 8); }, is_length);
    const bool t3 = throws_as([&] { dd::prepare_id_tables(s, std::numeric_limits<std::size_t>::max(), plan); }, is_length);
    check(t1 && t2 && t3 && s.ids.size() == size0 && s.ids.owner_bytes() == ob && s.hblock.bytes() == hb &&
          s.hblock.get(7) == 777 && s.ids.free_ids().size() == 3, "overflow and oversize growth are rejected without changes");
}

// ---- public DTs under every policy ---------------------------------------------------------
template <class DT> std::size_t terminal_build_target(const DT& d, std::size_t slots) {
    std::size_t n = 0, c = 0;
    for (std::size_t b = 0; b < d.rear().block_count(); ++b) {
        if (d.rear().block_label_count(b) == 0) continue;
        n += d.rear().block_label_count(b);
        c += std::max(d.rear().label_capacity_of(b),
                      ((d.rear().block_label_count(b) + block::kLabelGrowthSlack + 127) / 128) * 128);
    }
    return slots <= 1 ? slots : 1 + ((slots - 1) * c + n - 1) / n;
}
template <class DT> std::vector<std::pair<std::string, std::function<std::unique_ptr<DT>(const std::vector<std::string>&)>>>
build_routes(std::size_t bc, Growth g) {
    using Fn = std::function<std::unique_ptr<DT>(const std::vector<std::string>&)>;
    std::vector<std::pair<std::string, Fn>> r;
    r.emplace_back("sequential", [=](const std::vector<std::string>& k) { return seq_build<DT>(k, bc, g); });
    r.emplace_back("parallel", [=](const std::vector<std::string>& k) { return DT::from_sorted_parallel(k, bc, 0, false, g); });
    r.emplace_back("parallel-views", [=](const std::vector<std::string>& k) {
        std::vector<std::string_view> v(k.begin(), k.end());
        return DT::from_sorted_parallel(std::span<const std::string_view>(v), bc, 0, false, g);
    });
    return r;
}
template <class DT> void insert_one(DT& d, const std::string& k) { const std::string_view v = k; d.insert_sorted(&v, 1); }
// One insertion step: a growth happens only when the required slots exceed a
// table's capacity; under a percentage it is exactly the target of the slots
// after the step (required post-insertion slots, after reuse).
struct StepCheck {
    std::size_t growths = 0, rewidens = 0;
    bool ok_nogrow = true, ok_cap = true, ok_exact = true, ok_rewiden = true;
    template <class DT, class Op> void step(DT& d, Growth g, Op&& op) {
        const auto b = d.table_capacity();
        const auto bb = table_bytes(d);
        op();
        const auto a = d.table_capacity();
        const auto ab = table_bytes(d);
        ok_cap &= a.owner_capacity >= a.slots && a.handle_capacity >= a.slots;
        if (a.handle_width != b.handle_width) { ++rewidens; ok_rewiden &= a.handle_capacity >= b.handle_capacity; }
        if (a.slots <= b.owner_capacity && a.slots <= b.handle_capacity) {
            ok_nogrow &= ab.second == bb.second && (a.handle_width != b.handle_width || ab.first == bb.first);
        } else if (ab.second != bb.second || ab.first != bb.first) {
            ++growths;
            if (!g.is_terminal()) {
                const std::size_t t = ref_percent(a.slots, g.reserve_percent());
                ok_exact &= ab.second == bytes_for(t, 16) && a.handle_capacity >= t;
                if (a.handle_width == b.handle_width) ok_exact &= ab.first == bytes_for(t, a.handle_width);
            }
        }
    }
};
template <class DT> void policy_lifecycle(const char* name) {
    const auto base = keys(500, 7), more = keys(1200, 8), batch = keys(3000, 9);
    for (Growth g : {Growth::terminal(), Growth::percent(2), Growth::percent(10), Growth::percent(100)}) {
        for (auto& [route, build] : build_routes<DT>(256, g)) {
            const std::string tag = std::string(name) + " " + route + " growth=" + g.to_string();
            auto d = build(base);
            const auto tc = d->table_capacity();
            const std::size_t t = g.is_terminal() ? terminal_build_target(*d, tc.slots) : ref_percent(tc.slots, g.reserve_percent());
            const auto tb = table_bytes(*d);
            check(d->table_growth() == g && tc.slots == d->distinct_remainders() + 1 && tc.free_ids == 0, tag + ": build policy and slots");
            check(tb.second == bytes_for(t, 16) && tb.first == bytes_for(t, tc.handle_width),
                  tag + ": the initial build allocates exactly the policy's target");
            const auto ids0 = dtt::rear_map(d->rear());
            StepCheck sc;
            for (const auto& k : more) sc.step(*d, g, [&] { insert_one(*d, k); });
            const auto ids1 = dtt::rear_map(d->rear());
            sc.step(*d, g, [&] { d->insert_sorted(batch); });
            const auto ids2 = dtt::rear_map(d->rear());
            check(sc.ok_cap && sc.ok_nogrow, tag + ": tables grow only on exhaustion and always cover the slots");
            check(sc.ok_exact, tag + ": every percentage growth is exactly its target");
            check(sc.ok_rewiden, tag + ": handle repacking never loses reserve");
            if (!g.is_terminal()) check(sc.growths >= 2, tag + ": the reserve was exhausted repeatedly (" + std::to_string(sc.growths) + ")");
            bool stable = true;
            for (const auto& [rem, id] : ids0) if (ids1.contains(rem)) stable &= ids1.at(rem) == id;
            for (const auto& [rem, id] : ids1) if (ids2.contains(rem)) stable &= ids2.at(rem) == id;
            check(stable, tag + ": no live ID is renumbered by single or batch growth");
            bool found = true;
            for (const auto* v : {&base, &more, &batch})
                for (const auto& k : *v) found &= d->contains(k);
            for (const auto& k : keys(200, 1234)) {
                const bool stored = std::binary_search(base.begin(), base.end(), k) || std::binary_search(more.begin(), more.end(), k) ||
                                    std::binary_search(batch.begin(), batch.end(), k);
                found &= d->contains(k) == stored;
            }
            check(found && d->key_count() == base.size() + more.size() + batch.size(), tag + ": query results unchanged");
            try { d->check_invariants(); check(true, tag + ": invariants"); }
            catch (const std::exception& e) { check(false, tag + ": " + e.what()); }
            if (route == "sequential") check(sc.rewidens >= 1, tag + ": the run covers handle-width growth");
        }
        // The empty constructor takes the policy before its first allocation.
        DT e(256, g);
        e.insert_sorted(base);
        const auto ec = e.table_capacity();
        check(e.table_growth() == g && ec.owner_capacity >= ec.slots, std::string(name) + " empty growth=" + g.to_string() + ": policy");
        if (!g.is_terminal())
            check(table_bytes(e).second == bytes_for(ref_percent(ec.slots, g.reserve_percent()), 16),
                  std::string(name) + " empty growth=" + g.to_string() + ": first batch reserves exactly its target");
    }
    // Switching: no immediate reallocation; the next exhaustion follows the new policy.
    auto d = seq_build<DT>(base, 256, Growth::percent(100));
    const auto b0 = table_bytes(*d);
    d->set_table_growth(Growth::percent(2));
    check(table_bytes(*d) == b0 && d->table_growth() == Growth::percent(2), std::string(name) + ": switching reallocates nothing");
    StepCheck sc;
    for (std::size_t i = 0; i < more.size() && sc.growths == 0; ++i) sc.step(*d, Growth::percent(2), [&] { insert_one(*d, more[i]); });
    check(sc.growths == 1 && sc.ok_exact && sc.ok_nogrow && table_bytes(*d).second > b0.second,
          std::string(name) + ": after 100 -> 2 the old reserve is consumed first, then 2 % applies, never shrinking");
    d->set_table_growth(Growth::terminal());
    const auto b1 = table_bytes(*d);
    check(table_bytes(*d) == b1 && d->table_growth().is_terminal(), std::string(name) + ": switching back to terminal");
    d->check_invariants();
}
}  // namespace
int main() {
    packed_fields();
    target_and_reuse();
    public_builds<DfDT>("DFUDS DT");
    for (Growth g : {Growth::terminal(), Growth::percent(2), Growth::percent(10), Growth::percent(100)})
        stable_ids_and_reuse<DfDT>("DFUDS DT", g);
    growth_policy_values();
    percent_tables();
    policy_lifecycle<DfDT>("DFUDS DT");
    std::printf("dt_global_capacity: %zu checks, %zu failures\n", checks, failures);
    return failures ? 1 : 0;
}
