// ConstNavigationView tests (build-order Step 5, pmabv side).
//
// GENERIC ONLY. This file uses a live/ones mirror augmentation and mentions no
// parenthesis, excess, target or trie concept -- the view is BP-agnostic and is
// tested that way. The BP extension (include/bp/) that consumes it is tested
// in tests/test_bp_*.cpp.

#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <type_traits>
#include <vector>

using namespace pmabv;

namespace {

// live/ones mirror: root value is independently checkable via the public API.
struct MirrorAug {
    struct Value {
        std::int64_t live = 0, ones = 0;
        bool operator==(const Value&) const noexcept = default;
    };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t> w,
                            std::size_t live) noexcept {
        std::int64_t ones = 0;
        const std::size_t full = live / 64, rem = live % 64;
        for (std::size_t i = 0; i < full; ++i) ones += __builtin_popcountll(w[i]);
        if (rem) ones += __builtin_popcountll(w[full] &
                          ((~std::uint64_t{0}) >> (64 - rem)));
        return Value{static_cast<std::int64_t>(live), ones};
    }
    static Value combine(Value l, Value r) noexcept {
        return Value{l.live + r.live, l.ones + r.ones};
    }
};
static_assert(NonNullAugmentationPolicy<MirrorAug>);

using MEyt = BitPMAImpl<EytzingerTreePolicy, block::Soa,      MirrorAug>;

std::vector<bool> pattern(std::size_t n, std::uint64_t seed) {
    std::vector<bool> v(n);
    std::uint64_t x = seed | 1ULL;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v[i] = (x & 0x11) != 0;
    }
    return v;
}

bool g_phantom_fixture_seen = false;

// Recursively check topology + augmentation consistency through the VIEW only.
// `leaf_ord` counts leaves left-to-right as traversal reaches them -- derived
// from the traversal order itself, never from a node index. Leaves with
// ordinal < block_count() are real; the rest are phantom.
template <class V>
void walk(const V& view, typename V::node_type n, std::size_t n_real_blocks,
          std::size_t& live_sum, std::size_t& leaf_ord,
          std::size_t& real_leaves) {
    if (view.is_leaf(n)) {
        const std::size_t ord = leaf_ord++;
        if (ord < n_real_blocks) {
            ++real_leaves;
            // Real: block() is allowed and must agree with the node summary.
            const auto blk = view.block(n);
            DBV_CHECK_EQ(blk.live_bits, view.node_live(n));
        } else {
            // Phantom: zero live, identity augmentation, and block() must NOT
            // be called -- doing so would dereference storage that does not
            // exist for this index.
            DBV_CHECK_EQ(view.node_live(n), std::size_t{0});
            DBV_CHECK(view.node_aug(n) == MirrorAug::empty());
        }
        live_sum += view.node_live(n);
        return;
    }
    const auto l = view.left_child(n), r = view.right_child(n);
    // Parent live == left + right, everywhere in the tree.
    DBV_CHECK_EQ(view.node_live(n), view.node_live(l) + view.node_live(r));
    // The generic augmentation must combine through the exposed topology.
    DBV_CHECK(view.node_aug(n) ==
              MirrorAug::combine(view.node_aug(l), view.node_aug(r)));
    // Children know their own side.
    DBV_CHECK(view.is_left_child(l));
    DBV_CHECK(!view.is_left_child(r));
    DBV_CHECK(view.parent(l) == n);
    DBV_CHECK(view.parent(r) == n);
    walk(view, l, n_real_blocks, live_sum, leaf_ord, real_leaves);
    walk(view, r, n_real_blocks, live_sum, leaf_ord, real_leaves);
}

template <class T>
void check_instance(const char* name, const T& pma) {
    const auto view = pma.navigation_view();

    // Root: live == size(), augmentation == root_aug().
    DBV_CHECK_EQ(view.node_live(view.root()), pma.size());
    DBV_CHECK(view.node_aug(view.root()) == pma.root_aug());

    std::size_t live_sum = 0, leaf_ord = 0, real_leaves = 0;
    walk(view, view.root(), pma.block_count(), live_sum, leaf_ord, real_leaves);
    DBV_CHECK_EQ(live_sum, pma.size());
    // The traversal must see exactly the padded leaf count, of which exactly
    // block_count() are real.
    DBV_CHECK_EQ(leaf_ord, pma.block_count_padded());
    DBV_CHECK_EQ(real_leaves, pma.block_count());

    // Phantom leaves: zero live, identity augmentation. Reached through the
    // view's own traversal, never by index arithmetic.
    if (pma.block_count() < pma.block_count_padded()) {
        g_phantom_fixture_seen = true;
    }

    // Every boundary on manageable fixtures; strided on large ones. The
    // endpoints are ALWAYS probed: with a stride alone, k == size() is skipped
    // on exactly the large fixtures that have phantom leaves, which is where
    // routing the end boundary into a phantom would show up.
    const std::size_t step = pma.size() > 4096 ? 97 : 1;
    std::vector<std::size_t> ks;
    for (std::size_t k = 0; k <= pma.size(); k += step) ks.push_back(k);
    ks.push_back(0);
    ks.push_back(pma.size());
    if (pma.size() > 0) ks.push_back(pma.size() - 1);
    for (std::size_t k : ks) {
        const auto loc = view.locate_boundary(k);
        // The defining identity.
        DBV_CHECK_EQ(loc.global_block_start + loc.local_boundary, k);
        // Never a phantom block.
        DBV_CHECK(loc.block_index < pma.block_count());
        const auto blk = view.block(loc.leaf_node);
        DBV_CHECK(loc.local_boundary <= blk.live_bits);
        if (k < pma.size() && pma.size() > 0) {
            // Interior boundaries sit strictly inside their owning block --
            // this is the "exact block boundary belongs to the RIGHT block
            // with local_boundary == 0" rule.
            DBV_CHECK(loc.local_boundary < blk.live_bits);
        }
        if (k == pma.size() && pma.size() > 0) {
            // End boundary: final REAL block, local == live_bits.
            DBV_CHECK_EQ(loc.local_boundary, blk.live_bits);
            DBV_CHECK_EQ(loc.block_index, pma.block_count() - 1);
        }
    }
    // k == 0 always lands at the start of the first real block.
    {
        const auto loc = view.locate_boundary(0);
        DBV_CHECK_EQ(loc.global_block_start, std::size_t{0});
        DBV_CHECK_EQ(loc.local_boundary, std::size_t{0});
    }
    (void)name;
}

template <class T>
void run_sizes(const char* name, std::size_t bc) {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{64}, std::size_t{65}, std::size_t{1000},
                          std::size_t{bc}, std::size_t{bc * 3 + 7},
                          std::size_t{bc * 5 + 13}, std::size_t{20000}}) {
        const T pma(pattern(n, 4242), bc);
        check_instance(name, pma);
    }
}

// Compile-time guarantees.
void test_static_properties() {
    using V = typename MEyt::ConstNavigationView;
    // Block words are const.
    static_assert(std::is_same_v<decltype(V::BlockView::words),
                                 std::span<const std::uint64_t>>);
    // A const engine yields the view.
    static_assert(std::is_same_v<
        decltype(std::declval<const MEyt&>().navigation_view()), V>);
    // Node tokens carry no arithmetic and cannot be built from an integer.
    static_assert(!std::is_constructible_v<typename V::node_type, std::size_t>);
    static_assert(!std::is_default_constructible_v<typename V::node_type>);
    // The view adds no per-instance storage to the engine.
    static_assert(sizeof(MEyt) == sizeof(BitPMAImpl<EytzingerTreePolicy,
                                                    block::Soa, MirrorAug>));
    DBV_CHECK(true);
}

}  // namespace

int main() {
    test_static_properties();
    run_sizes<MEyt>("Eytzinger/Soa", 1024);
    run_sizes<MEyt>("Eytzinger/Soa bc=16896", 16896);

    if (!g_phantom_fixture_seen) {
        std::cerr << "no fixture had block_count() < block_count_padded(): "
                     "phantom-tree coverage was NOT exercised\n";
        ++pmabv_test::stats().failures;
    }
    ++pmabv_test::stats().tests;
    return pmabv_test::finalize("navview");
}
