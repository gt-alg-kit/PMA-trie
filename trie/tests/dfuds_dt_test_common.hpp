#pragma once
// Helpers of the DFUDS-front DoubleTrie tests (test_dfuds_double_trie, test_dfuds_double_trie_exceptions,
// test_dfuds_dt_adaptive, test_dt_global_capacity): logical and positional views of a
// trie::dfuds::DoubleTrie -- the END DFUDS front (one has-id flag per entry, ids in flag rank) and the END BP rear
// (leaf-rank ids, no flag channel).
#include "trie/dfuds_double_trie.hpp"

#include <algorithm>
#include <cstdint>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace dtt {

using DD = trie::dfuds::DoubleTrie;            // END DFUDS front, END BP rear (LeafRank)

// One front node: leaf or not, an END first child or not, the has-id flag of its incoming edge, the id.
struct FNode {
    bool leaf = false, end_child = false, hid = false;
    std::uint64_t id = 0;
    bool operator==(const FNode&) const = default;
};
// path -> node; an END leaf's path is its parent's path + '\0' (keys are NUL-free, so it names no other node)
using FrontMap = std::map<std::string, FNode>;
using RearMap = std::map<std::string, std::uint64_t>;   // remainder (forward) -> id

// The DFUDS front as path -> node: a preorder scan pairing every run with its incoming edge (the top of the stack is
// the next child). `ok` false: the sequence is unbalanced.
template <class DT>
FrontMap dfuds_front(const DT& dt, bool* ok = nullptr) {
    const auto& b = dt.front();
    FrontMap m;
    std::vector<std::pair<std::size_t, std::string>> st;
    const std::size_t total = b.size();
    bool root = true;
    if (ok) *ok = true;
    for (std::size_t p = 1; p < total;) {
        std::size_t d = 0;
        while (p + d < total && b.get(p + d) == 1) ++d;
        std::string path;
        FNode node;
        if (!root) {
            if (st.empty()) { if (ok) *ok = false; return m; }
            auto [ep, pa] = std::move(st.back());
            st.pop_back();
            const auto e = b.record_at(ep);
            node.hid = b.flag_of(e);
            node.id = node.hid ? b.id_of(e) : 0;
            path = std::move(pa);
        }
        root = false;
        node.leaf = d == 0;
        node.end_child = d > 0 && b.record_at(p + d - 1).payload == 0;   // the ascending-first child is the run's last "("
        for (std::size_t j = 0; j < d; ++j)
            st.emplace_back(p + j, path + static_cast<char>(b.record_at(p + j).payload));
        m[path] = node;
        p += d + 1;
    }
    if (ok && !st.empty()) *ok = false;
    return m;
}

// The rear (trie::bp::DynamicBP, leaf-rank ids) as remainder -> id: every leaf but the lone root pair of an empty
// rear, an END leaf naming its parent's path.
template <class Bp>
RearMap rear_map(const Bp& rear) {
    static_assert(Bp::kLeafRank, "the double trie's rear carries leaf-rank ids");
    RearMap m;
    std::string path;
    for (std::size_t p = 1; p + 1 < rear.size(); ++p) {
        if (rear.get(p) == 1) {
            path.push_back(static_cast<char>(rear.label_at(p)));
            if (rear.get(p + 1) == 0) {
                const std::size_t n = path.size() - (path.back() == '\0' ? 1 : 0);
                m[std::string(path.rbegin() + static_cast<std::ptrdiff_t>(path.size() - n), path.rend())] = rear.id_at(p);
            }
        } else {
            path.pop_back();
        }
    }
    return m;
}

// Every stored key, reconstructed from the two tries: per non-root front leaf, an END leaf ends its parent's path, a
// has-id leaf its path + the remainder of its id (the rear's id -> remainder), any other leaf its path; the root leaf
// of {""} the empty key. `ok` false: a front id with no rear remainder, or an id on two rear leaves.
template <class DT>
std::vector<std::string> reconstruct_keys(const DT& dt, bool* ok = nullptr) {
    const FrontMap f = dfuds_front(dt);
    std::map<std::uint64_t, std::string> rem;
    if (ok) *ok = true;
    for (const auto& [r, id] : rear_map(dt.rear()))
        if (!rem.emplace(id, r).second && ok) *ok = false;
    std::vector<std::string> keys;
    for (const auto& [path, e] : f) {
        if (!e.leaf) continue;
        if (path.empty()) { if (dt.key_count() > 0) keys.push_back(path); continue; }
        if (path.back() == '\0') { keys.push_back(path.substr(0, path.size() - 1)); continue; }
        if (!e.hid) { keys.push_back(path); continue; }
        const auto it = rem.find(e.id);
        if (it == rem.end()) { if (ok) *ok = false; continue; }
        keys.push_back(path + it->second);
    }
    std::sort(keys.begin(), keys.end());
    return keys;
}

// Two rears of this double trie position by position: bits, labels, which entries own a field, the ids.
template <class R1, class R2>
bool same_rear(const R1& x, const R2& y, std::string& why) {
    if (x.size() != y.size()) { why = "rear size " + std::to_string(x.size()) + " vs " + std::to_string(y.size()); return false; }
    for (std::size_t p = 0; p < x.size(); ++p) {
        if (x.get(p) != y.get(p)) { why = "rear bit at " + std::to_string(p); return false; }
        if (x.get(p) == 0) continue;
        if (x.label_at(p) != y.label_at(p) || x.has_id(p) != y.has_id(p)) { why = "rear label / field at " + std::to_string(p); return false; }
        if (x.has_id(p) && x.id_at(p) != y.id_at(p)) { why = "rear id at " + std::to_string(p); return false; }
    }
    return true;
}

// Two double tries of one type with the same shape: front bits, labels and has-id flags, rear bits and labels, the
// key count and the stored keys (reconstructed through the ids). The id values and the id width may differ: they
// depend on the history (from_sorted numbers the remainders by rank, a batch into an empty trie in allocation order).
template <class DT>
bool same_structure(const DT& a, const DT& b, std::string& why) {
    const auto& x = a.front();
    const auto& y = b.front();
    if (x.size() != y.size()) { why = "front size " + std::to_string(x.size()) + " vs " + std::to_string(y.size()); return false; }
    for (std::size_t p = 0; p < x.size(); ++p) {
        if (x.get(p) != y.get(p)) { why = "front bit at " + std::to_string(p); return false; }
        if (x.get(p) == 0) continue;
        const auto rx = x.record_at(p), ry = y.record_at(p);
        if (rx.payload != ry.payload || x.flag_of(rx) != y.flag_of(ry)) { why = "front entry at " + std::to_string(p); return false; }
    }
    const auto& u = a.rear();
    const auto& v = b.rear();
    if (u.size() != v.size()) { why = "rear size " + std::to_string(u.size()) + " vs " + std::to_string(v.size()); return false; }
    for (std::size_t p = 0; p < u.size(); ++p) {
        if (u.get(p) != v.get(p) || (u.get(p) && u.label_at(p) != v.label_at(p))) { why = "rear at " + std::to_string(p); return false; }
    }
    if (a.key_count() != b.key_count()) { why = "key count"; return false; }
    if (reconstruct_keys(a) != reconstruct_keys(b)) { why = "reconstructed keys"; return false; }
    return true;
}

// Two double tries of one type position by position (front: bits, labels, has-id flags, ids; the rear: bits,
// labels, fields, ids), id width, key count, live remainders and the global tables' logical state (slots, free ids,
// live ids): equal histories (sequential / parallel builds, a failed operation's untouched state, policies that must
// not change the state).
template <class DT>
bool same_state(const DT& a, const DT& b, std::string& why) {
    const auto& x = a.front();
    const auto& y = b.front();
    if (x.size() != y.size()) { why = "front size " + std::to_string(x.size()) + " vs " + std::to_string(y.size()); return false; }
    for (std::size_t p = 0; p < x.size(); ++p) {
        if (x.get(p) != y.get(p)) { why = "front bit at " + std::to_string(p); return false; }
        if (x.get(p) == 0) continue;
        const auto rx = x.record_at(p), ry = y.record_at(p);
        if (rx.payload != ry.payload || x.flag_of(rx) != y.flag_of(ry) || (x.flag_of(rx) && x.id_of(rx) != y.id_of(ry))) {
            why = "front entry at " + std::to_string(p);
            return false;
        }
    }
    if (!same_rear(a.rear(), b.rear(), why)) return false;
    if (a.id_width() != b.id_width()) { why = "id width"; return false; }
    if (a.key_count() != b.key_count()) { why = "key count"; return false; }
    if (a.distinct_remainders() != b.distinct_remainders()) { why = "distinct remainders"; return false; }
    const auto ta = a.table_capacity(), tb = b.table_capacity();
    if (ta.slots != tb.slots || ta.free_ids != tb.free_ids || ta.live_ids != tb.live_ids) { why = "global table slots / free ids"; return false; }
    return true;
}

}  // namespace dtt
