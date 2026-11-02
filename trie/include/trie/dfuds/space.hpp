#pragma once
// Additive, reconciling space breakdown of a DFUDS dictionary.
//
// ROWS. in_total = true: an additive byte part of `parent` (the additive rows
// under one parent sum to it exactly; the top-level rows, parent "total", sum
// to the dictionary's resident total). in_total = false: a diagnostic (a
// count, a bit count, or bytes that are a subset of an additive row), never
// summed. A residual is never clamped silently: a negative one is reported
// and fails reconcile().
//
// total = pma + object:
//   pma                  BitPMAImpl::byte_size(): every allocation of the
//                        backend once, at its capacity:
//     pma.topology       the DFUDS bit sequence words incl. PMA gaps/slack;
//     pma.labels         per-block label vectors (one byte per one bit);
//     pma.flags          0: the END dictionary stores no flag channel (key
//                        ends are END leaves; trie/dfuds/detail/backend.hpp);
//     pma.rank_index     the rank-sample slab;
//     pma.excess_index   the excess-record slab;
//     pma.summary_counts the summary tree's live/one counts;
//     pma.minmax_aug     the BpAug min-excess array;
//     pma.other          the rest of byte_size(): the BitPMAImpl object, the
//                        per-block live counts, the retained redistribute
//                        scratch bitsets;
//   object               the dictionary object and its state header outside
//                        the backend (key count, pointer, mutex, padding), by
//                        sizeof.
// Input keys, query answers and operation scratch (released before an
// operation returns) are outside every total.

#include <cstddef>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace trie::dfuds {

struct SpaceRow {
    std::string component;   // unique; "<prefix>.<category>[.<part>]"
    std::string parent;      // "total" at the top level
    std::size_t value;       // in `unit`
    std::string unit;        // "bytes", "count" or "bits"
    bool in_total;           // an additive byte part of `parent`
    std::string basis;       // how the value was obtained
};

struct SpaceReport {
    std::size_t total = 0;          // resident bytes
    std::vector<SpaceRow> rows;
};

struct Reconcile {
    bool ok;
    long long residual;
    std::string detail;
};

// The top-level additive rows sum to expected_total, every row with additive
// parts equals their sum, parents exist, names are unique, additive rows are
// bytes under additive rows only. Diagnostics are never summed.
inline Reconcile reconcile(const std::vector<SpaceRow>& rows, std::size_t expected_total) {
    Reconcile r{true, 0, {}};
    auto fail = [&](long long res, const std::string& what) {
        if (r.ok) r.residual = res;
        r.ok = false;
        r.detail += (r.detail.empty() ? "" : "; ") + what;
    };
    auto is_top = [](const std::string& name) { return name.empty() || name == "total"; };
    std::map<std::string, const SpaceRow*> by_name;
    for (const SpaceRow& x : rows) {
        if (is_top(x.component) || !by_name.emplace(x.component, &x).second)
            fail(0, "component '" + x.component + "' is reserved or duplicated");
        if (x.in_total && x.unit != "bytes") fail(0, x.component + ": an additive row in " + x.unit);
        if (x.basis.find("NEGATIVE") != std::string::npos) fail(0, x.component + ": " + x.basis);
    }
    std::map<std::string, unsigned long long> parts;
    unsigned long long top = 0;
    std::size_t additive = 0;
    for (const SpaceRow& x : rows) {
        const auto it = by_name.find(x.parent);
        if (!is_top(x.parent) && it == by_name.end()) { fail(0, x.component + ": unknown parent '" + x.parent + "'"); continue; }
        if (!x.in_total) continue;
        ++additive;
        if (is_top(x.parent)) { top += x.value; continue; }
        if (!it->second->in_total) { fail(0, x.component + ": an additive part of the diagnostic " + x.parent); continue; }
        parts[x.parent] += x.value;
    }
    auto diff = [](unsigned long long whole, unsigned long long sum) {
        return static_cast<long long>(whole) - static_cast<long long>(sum);
    };
    if (const long long d = diff(expected_total, top); d != 0)
        fail(d, "total " + std::to_string(expected_total) + " != top-level parts " + std::to_string(top));
    for (const auto& [name, sum] : parts)
        if (const long long d = diff(by_name.at(name)->value, sum); d != 0)
            fail(d, name + " " + std::to_string(by_name.at(name)->value) + " != its parts " + std::to_string(sum));
    if (r.ok)
        r.detail = "ok: " + std::to_string(additive) + " additive rows, " + std::to_string(parts.size() + 1) + " sums";
    return r;
}

}  // namespace trie::dfuds
