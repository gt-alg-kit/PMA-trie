#pragma once
// Helpers of the DFUDS dictionary tests: the independent oracle (a pointer
// trie over byte strings and the canonical DFUDS encoding written straight
// from the definition by an explicit preorder traversal -- nothing shared with
// the library's encoders; ported from the encoding comparison's
// tests/reference_tree.hpp), checks that abort with a message, key-set
// families and probe sets.
//
// DFUDS with END leaves (the library's convention): a dummy "(" (label 0), then
// every node in preorder as "(" x degree + ")"; the "(" at offset j of v's run
// leads to child deg-j (children ascending by label), so a run's labels
// DECREASE; each "(" carries its child's label. A key ends at a leaf: a stored
// key that another stored key extends owns an END child (label 0), its
// ascending-first child (the run's last "("); the root is a leaf only in {} and
// {""} (Logical::key_count tells them apart). Keys are NUL-free. 2n bits for n
// entries (nodes and END leaves).

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "trie/dfuds_dictionary.hpp"

namespace tst {

inline std::size_t g_checks = 0, g_cases = 0;
[[noreturn]] inline void fail(const std::string& what) {
    std::fprintf(stderr, "FAIL %s\n", what.c_str());
    std::fflush(stderr);
    std::exit(1);
}
#define CHECK(cond, what) do { ++::tst::g_checks; if (!(cond)) ::tst::fail(what); } while (0)

inline std::vector<std::uint64_t> pack_bools(const std::vector<bool>& v) {
    std::vector<std::uint64_t> w((v.size() + 63) / 64, 0);
    for (std::size_t i = 0; i < v.size(); ++i) if (v[i]) w[i >> 6] |= 1ULL << (i & 63);
    return w;
}

struct RefNode {
    std::map<std::uint8_t, std::unique_ptr<RefNode>> kids;
    bool terminal = false;
};

class RefTree {
public:
    RefTree() = default;
    RefTree(const RefTree&) = delete;
    RefTree& operator=(const RefTree&) = delete;
    ~RefTree() {   // iterative teardown: a 100,000-byte key is a 100,000-deep chain
        std::vector<std::unique_ptr<RefNode>> st;
        for (auto& [c, kid] : root_.kids) st.push_back(std::move(kid));
        root_.kids.clear();
        while (!st.empty()) {
            std::unique_ptr<RefNode> v = std::move(st.back());
            st.pop_back();
            for (auto& [c, kid] : v->kids) st.push_back(std::move(kid));
            v->kids.clear();
        }
    }
    bool insert(std::string_view k) {
        RefNode* v = &root_;
        for (unsigned char c : k) {
            auto& p = v->kids[c];
            if (!p) { p = std::make_unique<RefNode>(); ++nodes_; }
            v = p.get();
        }
        if (v->terminal) return false;
        v->terminal = true;
        ++keys_;
        return true;
    }
    bool contains(std::string_view k) const {
        const RefNode* v = &root_;
        for (unsigned char c : k) {
            auto it = v->kids.find(c);
            if (it == v->kids.end()) return false;
            v = it->second.get();
        }
        return v->terminal;
    }
    std::size_t nodes() const { return nodes_; }   // root included
    std::size_t keys() const { return keys_; }
    // Keys that another key extends: terminal nodes with children (each owns one END leaf).
    std::size_t end_leaves() const {
        std::size_t n = 0;
        std::vector<const RefNode*> st{&root_};
        while (!st.empty()) {
            const RefNode* v = st.back(); st.pop_back();
            n += v->terminal && !v->kids.empty();
            for (const auto& [c, kid] : v->kids) st.push_back(kid.get());
        }
        return n;
    }

    trie::dfuds::Logical dfuds() const {
        std::vector<bool> bits{true};
        std::vector<std::uint8_t> labels{0};
        std::vector<const RefNode*> st{&root_};   // preorder: pop a node, emit its run, push children last-first
        while (!st.empty()) {
            const RefNode* v = st.back(); st.pop_back();
            if (v == nullptr) { bits.push_back(false); continue; }   // an END leaf: an empty run
            if (v->kids.count(0)) fail("RefTree::dfuds: a key holds byte 0 (outside the END domain)");
            const bool end = v->terminal && !v->kids.empty();
            for (auto it = v->kids.rbegin(); it != v->kids.rend(); ++it) {   // run order: child deg first
                bits.push_back(true); labels.push_back(it->first);
            }
            if (end) { bits.push_back(true); labels.push_back(0); }      // END: the ascending-first child, last "("
            bits.push_back(false);
            for (auto it = v->kids.rbegin(); it != v->kids.rend(); ++it) st.push_back(it->second.get());
            if (end) st.push_back(nullptr);                              // ... and the first child in preorder
        }
        trie::dfuds::Logical L;
        L.live = bits.size();
        L.words = pack_bools(bits);
        L.labels = labels;
        L.key_count = keys_;
        return L;
    }

private:
    RefNode root_;
    std::size_t nodes_ = 1, keys_ = 0;
};

inline std::string describe(const trie::dfuds::Logical& a, const trie::dfuds::Logical& b) {
    std::string s = "live " + std::to_string(a.live) + " vs " + std::to_string(b.live) + ", labels " +
                    std::to_string(a.labels.size()) + " vs " + std::to_string(b.labels.size()) + ", keys " +
                    std::to_string(a.key_count) + " vs " + std::to_string(b.key_count);
    for (std::size_t i = 0; i < std::min(a.live, b.live); ++i)
        if (a.bit(i) != b.bit(i)) { s += ", first bit difference at " + std::to_string(i); break; }
    for (std::size_t i = 0; i < std::min(a.labels.size(), b.labels.size()); ++i)
        if (a.labels[i] != b.labels[i]) { s += ", first label difference at " + std::to_string(i); break; }
    return s;
}

inline std::vector<std::string> sorted_unique(std::vector<std::string> v) {
    std::sort(v.begin(), v.end());
    v.erase(std::unique(v.begin(), v.end()), v.end());
    return v;
}
inline std::vector<std::string_view> views(const std::vector<std::string>& v) { return {v.begin(), v.end()}; }

inline std::string rnd(std::mt19937_64& g, int alpha, int base, std::size_t lo, std::size_t hi) {
    std::uniform_int_distribution<std::size_t> L(lo, hi);
    std::uniform_int_distribution<int> S(0, alpha - 1);
    std::string k(L(g), ' ');
    for (auto& c : k) c = static_cast<char>((base + S(g)) & 0xFF);
    return k;
}

// Every key, every prefix (a stride above 4,096 bytes plus the last 64),
// one-byte extensions with 'a' / 0x00 (outside the key domain: absent) /
// 0xff, a bumped last byte, + extra.
inline std::vector<std::string> probes(const std::vector<std::string>& keys, const std::vector<std::string>& extra = {}) {
    std::set<std::string> p;
    p.insert(std::string());
    for (const auto& k : keys) {
        const std::size_t step = k.size() > 4096 ? 997 : 1;
        for (std::size_t l = 0; l <= k.size(); l += step) p.insert(k.substr(0, l));
        for (std::size_t l = k.size() > 64 ? k.size() - 64 : 0; l <= k.size(); ++l) p.insert(k.substr(0, l));
        p.insert(k + "a"); p.insert(k + std::string(1, '\0')); p.insert(k + std::string(1, '\xff'));
        if (!k.empty()) { std::string m = k; m.back() = static_cast<char>(m.back() + 1); p.insert(m); }
    }
    for (const auto& e : extra) p.insert(e);
    return {p.begin(), p.end()};
}

struct KeySet { std::string name; std::vector<std::string> keys; };

inline std::vector<KeySet> families(std::mt19937_64& g, bool large = true) {
    std::vector<KeySet> F;
    F.push_back({"empty", {}});
    F.push_back({"empty_key", {""}});
    F.push_back({"empty_key_and_a", {"", "a"}});
    F.push_back({"one_key", {"abc"}});
    F.push_back({"prefix_chain", {"a", "ab", "abc", "abcd", "abcde", "abcdef"}});
    {   // every ordinary label at the root, plus the empty key: 256 root children (END + 255)
        std::vector<std::string> v{""};
        for (int c = 1; c < 256; ++c) v.push_back(std::string(1, static_cast<char>(c)));
        F.push_back({"all_255_root_labels_and_end", v});
    }
    {   // every ordinary label below "x", which is a key itself: 256 children (END + 255)
        std::vector<std::string> v;
        for (int c = 1; c < 256; ++c) v.push_back(std::string("x") + static_cast<char>(c));
        v.push_back("x"); v.push_back("y");
        F.push_back({"all_255_below_root_and_end", v});
    }
    {
        std::vector<std::string> v;
        for (int c = 1; c < 256; c += 3) v.push_back(std::string(1, static_cast<char>(c)) + "tail" + static_cast<char>(c));
        F.push_back({"root_children_with_tails", v});
    }
    // 0x01, the smallest ordinary label, sorts right after END in a run (the old NUL fixture with 0x00 -> 0x01)
    F.push_back({"high_bytes", {"\x01", "\x01\x01", "a\x01" "b", "\x7f", "\x80", "\x80\x80", "\xff", "\xff\xff", "\xfe\x01"}});
    {
        std::string longk(3000, 'q');
        for (std::size_t i = 0; i < longk.size(); i += 11) longk[i] = static_cast<char>('a' + i % 26);
        F.push_back({"long_paths", {longk.substr(0, 10), longk.substr(0, 1500), longk, longk.substr(0, 2000) + "x",
                                    longk.substr(0, 700) + "\xff"}});
    }
    for (int alpha : {2, 5, 26, 40, 255}) {   // 255: every ordinary byte
        std::vector<std::string> v;
        for (int i = 0; i < 700; ++i) v.push_back(rnd(g, alpha, alpha > 26 ? (alpha == 255 ? 1 : 0x60) : 'a', 0, 9));
        F.push_back({"random_alpha" + std::to_string(alpha), v});
    }
    if (large) {
        std::vector<std::string> v;
        for (int i = 0; i < 30000; ++i) v.push_back(rnd(g, 26, 'a', 1, 14));
        F.push_back({"random_large", v});
        std::vector<std::string> u;
        for (int i = 0; i < 4000; ++i) u.push_back("http://www.example" + std::to_string(i % 97) + ".com/" + rnd(g, 40, 0x41, 0, 30));
        F.push_back({"url_like", u});
    }
    for (auto& f : F) f.keys = sorted_unique(f.keys);
    return F;
}

template <class F>
bool throws_invalid(F&& f) {
    try { f(); } catch (const std::invalid_argument&) { return true; }
    return false;
}
template <class F>
bool throws_length(F&& f) {
    try { f(); } catch (const std::length_error&) { return true; }
    return false;
}

}  // namespace tst
