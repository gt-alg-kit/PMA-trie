#pragma once
// DFUDS-front double trie with END leaves in BOTH tries: trie::dfuds::DoubleTrie. The front is a DFUDS trie (the
// dictionary's encoding, trie/dfuds/detail/core.hpp) over the minimal key prefixes; the rear is a BP trie
// (trie::bp::DynamicBP) over the distinct reversed remainders. There is no terminal channel in either trie.
//
// SPLIT (trie/dt/rear.hpp rank_parallel): key i is cut at c_i = min(|k_i|, 1 + max(lcp_prev, lcp_next));
// f_i = k_i[0, c_i) goes to the front; a non-empty remainder k_i[c_i, |k_i|) is reversed into the rear; equal reversed
// remainders share one id (g + 1 for the g-th distinct one in sorted order); owners[id] = its key count;
// W = bits_for(D) + 1 (D distinct remainders).
// FRONT (block::WithPayload<DefaultBlock, uint8_t, Flags, Ids, FlagRank>): the dictionary's END encoding
// (trie/dfuds/detail/encode.hpp) over the f_i, plus ONE flag per entry -- the HAS-ID marker of the edge into a
// remainder leaf -- whose W-bit id field is packed in flag rank. A key with an empty remainder ends at node(f_i), a
// leaf, or (when f_{i+1} extends f_i) a node whose ascending-first child is END. A remainder leaf is never extended
// (the cut rule), so END never sits under it (checked by the encoder).
// REAR (label-only trie::bp::DynamicBP on pmabv::IdKey::LeafRank): the distinct reversed remainders r_0 < r_1 < ...;
// r_j ends at a leaf, or, when r_{j+1} extends r_j (a remainder that is a suffix of another), at the END child of
// node(r_j) -- the parent's first child (label 0 sorts first), encoded as the string r_j + "\0" (keys are NUL-free, so
// the order is kept and every string ends at a leaf). Leaves in preorder are therefore exactly r_0, r_1, ...: the
// rear's ids are LEAF-RANK ids, W bits each, stored per PMA block in leaf order (a leaf belongs to the block holding
// its OPEN), with pmabv's per-block field counts and leaf-rank samples and its proportional id reserve
// (block/payload.hpp id_target_words). The lone root pair of an empty rear is a leaf with a zero field.
// A remainder's node: handle(id) -> rear block -> the field holding id -> that leaf of the block; an END leaf's
// remainder ends at its parent.
// GLOBAL TABLES (trie/dt/common.hpp, trie/dt/rear.hpp): id -> rear block handles (bits_for(rear blocks)
// bits), 16-bit owner counts, free list, overflow map, reserved and grown by the TableGrowth policy
// (trie/dt/table_growth.hpp).
// KEY DOMAIN: NUL-free keys (byte 0 is END in both tries): the builds and insert_sorted reject a key holding byte 0
// with std::invalid_argument before anything changes; a query holding it is absent.
//
// SUPPORTED: empty construction, from_sorted, from_sorted_parallel, contains, contains_sorted_parallel, insert_sorted
// (parallel batch insertion), ordered scans (scan_from, for_each_from), key_count,
// check_invariants, space, table_capacity. NOT SUPPORTED: erase, single-key insert, has_prefix, prefix enumeration.
// FRONT WALKS: contains() takes the reference DFUDS child step; contains_sorted_parallel's front walk is the
// reference shared walk; the batch insert's insertion walk is the run walk (detail::dt_insert_walk); the scans'
// descent is their own (the reference step plus the unary-stretch skip).
// MUTATION POLICY (trie/dfuds/config.hpp, TRIE_DFUDS_TRANSACTIONAL): the transaction covers front, rear, ids, handles
// and owner counts together -- a batch is applied to a copy, validated and swapped in (a failure leaves the trie
// unchanged); the direct mode validates its arguments before any change and applies in a noexcept call (a failure
// after that point terminates rather than leaving a partly applied batch).

#include <algorithm>
#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "bit_pma.hpp"
#include "entry_capacity.hpp"
#include "flag_bits.hpp"
#include "trie/dfuds/config.hpp"
#include "trie/dfuds/detail/scan.hpp"
#include "trie/dfuds/space.hpp"
#include "trie/dfuds/detail/backend.hpp"
#include "trie/dfuds/detail/core.hpp"
#include "trie/dfuds/detail/dt_batch.hpp"
#include "trie/dfuds/detail/encode.hpp"
#include "trie/dt/common.hpp"
#include "trie/dt/rear.hpp"
#include "trie/bp/dynamic_bp.hpp"
#include "trie/bp/encode.hpp"
#include "trie/detail/sorted_descent.hpp"
#include "trie/bp/sorted_walk.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN

class DoubleTrie {
public:
    using FrontBlock = ::block::WithPayload<pmabv::DefaultBlock, std::uint8_t, true, true, pmabv::IdKey::FlagRank>;
    using FrontBits = pmabv::BitPMAImpl<pmabv::EytzingerTreePolicy, FrontBlock, pmabv::bp::BpAug>;
    // the rear: labels + leaf-rank ids (pmabv::IdKey::LeafRank: no flag channel stored; every leaf owns a field)
    using RearBp = trie::bp::DynamicBP<pmabv::EytzingerTreePolicy, pmabv::DefaultBlock, true, true, pmabv::IdKey::LeafRank>;
    using FRef = FrontBits::RecordRef;
    using RRef = RearBp::RecordRef;
    using FrontRef = FRef;
    using RearRef = RRef;
    static constexpr bool transactional = TRIE_DFUDS_TRANSACTIONAL != 0;
    static constexpr std::size_t kDefaultBlockCapacity = 1024;
    static constexpr unsigned kInitialIdWidth = RearBp::kInitialIdWidth;
    static constexpr const char* kWho = "trie::dfuds::DoubleTrie";
    // The growth policy of the global handle and owner-count tables (trie/dt/table_growth.hpp): the rear's
    // terminal-entry ratio (default) or an explicit reserve percentage.
    using TableGrowth = trie::dt::TableGrowth;
    static constexpr const char* key_end_policy = "end_leaf_front_and_rear";
    static constexpr std::uint8_t kEnd = detail::kEnd;
    static constexpr std::size_t kWalkGrain = 512;   // the insertion walk's sequential grain (the BP insert walk's)
#ifdef NDEBUG
    static constexpr bool kValidateCsr = false;
#else
    static constexpr bool kValidateCsr = true;
#endif

    // The empty set: an empty front (the dummy and the root leaf), an empty rear (its root pair, a leaf with a zero
    // field) and the global tables reserved by `growth`. Both tries start at the rear's initial id width (as the BP
    // double trie and the double trie this one replaced), widened as ids grow.
    explicit DoubleTrie(std::size_t block_capacity = kDefaultBlockCapacity, TableGrowth growth = TableGrowth::terminal())
        : DoubleTrie(std::move(*build_parallel(std::vector<std::string_view>{}, block_capacity, 1, false, growth))) {
        W_ = kInitialIdWidth;
        front_.set_id_width(W_);
        trie::dt::detail::RawAccess::set_id_width(rear_, W_);
    }
    DoubleTrie& operator=(const DoubleTrie&) = delete;
    DoubleTrie& operator=(DoubleTrie&&) = delete;

    // ---- bulk builds ---------------------------------------------------------------------------------------------
    // `keys` strictly increasing in unsigned-byte order and NUL-free (checked: std::invalid_argument). Minimal
    // placement for every key (the unique-point cut), ids assigned in rear (reversed-remainder) order,
    // W = bits(distinct remainders) + 1; both tries laid out by pmabv's parallel build. from_sorted is the same build
    // with one chunk (the sequential encoding); from_sorted_parallel gives the same state for every chunk count.
    // `chunks` and `wide_records` (the remainder ranking's record width): test knobs. `growth`: the global tables'
    // policy, already applied to their initial reserve.
    static std::unique_ptr<DoubleTrie> from_sorted(const std::vector<std::string>& keys,
                                                   std::size_t block_capacity = kDefaultBlockCapacity,
                                                   TableGrowth growth = TableGrowth::terminal()) {
        return build_parallel(keys, block_capacity, 1, false, growth);
    }
    static std::unique_ptr<DoubleTrie> from_sorted_parallel(std::span<const std::string_view> keys,
                                                            std::size_t block_capacity = kDefaultBlockCapacity,
                                                            std::size_t chunks = 0, bool wide_records = false,
                                                            TableGrowth growth = TableGrowth::terminal()) {
        return build_parallel(keys, block_capacity, chunks, wide_records, growth);
    }
    static std::unique_ptr<DoubleTrie> from_sorted_parallel(const std::vector<std::string>& keys,
                                                            std::size_t block_capacity = kDefaultBlockCapacity,
                                                            std::size_t chunks = 0, bool wide_records = false,
                                                            TableGrowth growth = TableGrowth::terminal()) {
        return build_parallel(keys, block_capacity, chunks, wide_records, growth);
    }

    // Front descent along x. A leaf reached by the edge e: no flag -- a key iff x ends here; the has-id flag -- a key
    // iff bytes are left and they equal the remainder of id(e). An internal node reached with x consumed: a key iff
    // its ascending-first child is END. The root: a leaf with keys ({""}), or its END child.
    bool contains(std::string_view x) const {
        using S = detail::State;
        const std::size_t n = x.size();
        FRef cur = front_.record_at(1);
        if (n == 0) return cur.bit != 1 ? key_count_ > 0 : first_child_label(cur) == kEnd;
        for (std::size_t i = 0;; ++i) {
            if (cur.bit != 1) return false;
            const std::uint8_t c = static_cast<std::uint8_t>(x[i]);
            if (c == kEnd) return false;
            const std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), live = front_.block_live(b);
            const std::size_t pos = cur.slot.logical_pos();
            const std::size_t d = S::run_length(front_.block_words(b), local, live);
            FRef edge = cur, child = cur;
            if (local + d < live) {
                const std::size_t j = S::search(front_.block_payload(b) + cur.open_rank, d, c);
                if (j == d) return false;
                edge = front_.record_at_coordinate(b, local + j, pos + j, cur.open_rank + j);
                if (j + 1 == d) {
                    const std::size_t nl = local + d + 1;
                    child = nl < live ? front_.record_at_coordinate(b, nl, pos + d + 1, cur.open_rank + d) : front_.record_at(pos + d + 1);
                } else {
                    child = child_of(edge);
                }
            } else {
                S::RunBuf rb;
                read_run_slow(cur, rb);
                const std::size_t j = S::search(rb.label.data(), rb.d, c);
                if (j == rb.d) return false;
                edge = front_.record_at(pos + j);
                child = (j + 1 == rb.d) ? front_.record_at(pos + rb.d + 1) : child_of(edge);
            }
            if (child.bit != 1) {
                if (!front_.flag_of(edge)) return i + 1 == n;
                return i + 1 < n && remainder_equals(x.substr(i + 1), front_.id_of(edge));
            }
            if (i + 1 == n) return first_child_label(child) == kEnd;
            cur = child;
        }
    }

    // Membership of a SORTED batch (non-decreasing, duplicates allowed; std::invalid_argument otherwise),
    // out[i] = contains(keys[i]) -- two phases with END tests:
    //   F  the front's shared walk at the REFERENCE step (as the dictionary's shared lookup), every node carrying
    //      its incoming edge: keys ending at a node are keys iff the node is an unflagged leaf or its first child is
    //      END (the root: a leaf with keys, or its END child); keys going on at a has-id leaf go to R with its id;
    //      every other key is absent;
    //   R  the pending keys' rests REVERSED, sorted, walked down the rear once (trie::bp::detail::sorted_walk_par):
    //      present iff the rest's endpoint ends a remainder -- a leaf, or a node whose first child is END -- whose
    //      leaf-rank id is the front leaf's id.
    void contains_sorted_parallel(const std::string_view* keys, std::size_t n, std::uint8_t* out, std::size_t grain = 512) const {
        namespace sd = trie::sorted_descent;
        if (n == 0) return;
        sd::require_sorted(keys, n, "trie::dfuds::DoubleTrie::contains_sorted_parallel");
        constexpr std::uint8_t kPending = 2;
        const std::unique_ptr<std::uint64_t[]> pid(new std::uint64_t[n]);
        const std::unique_ptr<std::size_t[]> pd(new std::size_t[n]);
        auto finish = [&](const SNode& s, std::size_t lo, std::size_t hi) {
            struct Item { SNode s; std::size_t lo, hi; };
            std::vector<Item> st;
            st.push_back({s, lo, hi});
            while (!st.empty()) {
                const Item it = st.back(); st.pop_back();
                expand(it.s, keys, it.lo, it.hi, out, pid.get(), pd.get(), [&](const SNode& c, std::size_t i, std::size_t j) { st.push_back({c, i, j}); });
            }
        };
        auto visit = [&](const SNode& s, std::size_t lo, std::size_t hi, std::vector<sd::Child<SNode>>& children) {
            expand(s, keys, lo, hi, out, pid.get(), pd.get(), [&](const SNode& c, std::size_t i, std::size_t j) { children.push_back({c, i, j}); });
        };
        {
            pmabv::ParGuard guard;
            const FRef root = front_.record_at(1);
            sd::descend_sorted_parallel(SNode{root, root, 0}, 0, n, grain == 0 ? 1 : grain, guard, visit, finish);
            guard.rethrow();
        }
        const auto pend = parlay::filter(parlay::iota<std::size_t>(n), [&](std::size_t i) { return out[i] == kPending; });
        const std::size_t m = pend.size();
        if (m == 0) return;
        auto [off, total] = parlay::scan(parlay::tabulate(m, [&](std::size_t j) { return keys[pend[j]].size() - pd[pend[j]]; }));
        const std::unique_ptr<char[]> arena(new char[total ? total : 1]);
        auto rr = parlay::tabulate(m, [&](std::size_t j) {
            const std::size_t i = pend[j];
            const std::string_view rest = keys[i].substr(pd[i]);
            char* dst = arena.get() + off[j];
            std::reverse_copy(rest.begin(), rest.end(), dst);
            return std::pair<std::string_view, std::size_t>(std::string_view(dst, rest.size()), i);
        });
        parlay::sort_inplace(parlay::make_slice(rr), [](const auto& x, const auto& y) { return x.first < y.first; });
        const auto rk = parlay::tabulate(m, [&](std::size_t j) { return rr[j].first; });
        pmabv::ParGuard guard;
        trie::bp::detail::sorted_walk_par(rear_, rear_.record_at(0), 0, rk.data(), 0, m, grain == 0 ? 1 : grain, guard,
            [&](std::size_t j, bool ok, const RRef& node, std::size_t, std::size_t) {
                const std::size_t i = rr[j].second;
                out[i] = ok && rear_end_id(node) == pid[i];
            });
        guard.rethrow();
    }

    // ---- batch insert ---------------------------------------------------------------------------------------------
    // `keys` non-decreasing (checked) and NUL-free (checked; nothing changes on a violation); in-batch duplicates and
    // present keys are skipped; returns the number added. The double trie's pipeline with END transitions:
    //   P   P1 the END front's insertion walk (detail::dt_insert_walk, the run walk);
    //       P2-P4 the shared planner (dt::detail::plan_front_core); the owner deltas merged
    //       (dt::detail::merge_contribs); P5 the END insertion groups (plan_groups above);
    //   R1  rear additions with END (rear_add_end);
    //   F   front: the affected leaves lose their has-id flag and field, then ONE pmabv batch insert of the END CSR
    //       (detail::build_gap_csr_ids: END leaves, has-id flags + ids in entry order);
    //   R2  rear releases with END (rear_release_end).
    // P reads only; R1 / F / R2 run through mutate() (the mutation policy above).
    std::size_t insert_sorted(const std::string_view* keys, std::size_t n) {
        using namespace trie::dt::detail;
        if (n == 0) return 0;
        trie::sorted_descent::require_sorted(keys, n, "trie::dfuds::DoubleTrie::insert_sorted");
        const bool nul = parlay::reduce(parlay::delayed_tabulate(n, [&](std::size_t i) -> std::size_t {
            return !keys[i].empty() && std::memchr(keys[i].data(), 0, keys[i].size()) != nullptr; })) != 0;
        if (nul) throw std::invalid_argument(std::string(kWho) + "::insert_sorted: a key contains byte 0 (END)");
        const trie::detail::ForOverwrite<std::string_view> uk = distinct_keys(keys, n);
        trie::detail::ForOverwrite<Walk> wk(uk.size());
        trie::detail::ForOverwrite<detail::FPre> fp(uk.size());
        std::vector<std::uint8_t> term_leaf(uk.size(), 0);
        detail::dt_insert_walk(front_, key_count_, uk.data(), uk.size(), kWalkGrain, fp,
            [&](std::size_t i, const detail::DtReach& r) {
                wk[i] = Walk{r.edge, r.d, i, r.owns, r.id};
                term_leaf[i] = r.term_leaf;
            });
        FrontPlan pl;
        const bool any = plan_front_core(pl, uk, wk, [&](std::uint64_t id) { return remainder_of(id); }, [](auto) {});
        if (!any) return 0;
        trie::detail::ForOverwrite<Rec> recs;
        trie::detail::ForOverwrite<std::size_t> rec_of;
        merge_contribs(pl.contrib.size(), [&](std::size_t c) -> const Contrib& { return pl.contrib[c]; }, recs, rec_of);
        FrontGroups gr = plan_groups(pl, fp, term_leaf);
        const std::size_t added = pl.ks.size();
        mutate([&](DoubleTrie& t) { t.apply_batch(recs, rec_of, pl, gr, added); });
        return added;
    }
    std::size_t insert_sorted(const std::vector<std::string>& keys) {
        std::vector<std::string_view> v(keys.begin(), keys.end());
        return insert_sorted(v.data(), v.size());
    }

    // ---- ordered scans ----------------------------------------------------------------------------------------
    // The END dictionary's scan (trie/dfuds/detail/scan.hpp: keys at leaves in preorder, an END leaf ending its parent's
    // path, unary stretches crossed whole, a NUL bound byte never entering END) on this front, with the double trie's
    // key ends: a has-id leaf ends path + rem(id), the
    // remainder read through the scan's memo by id and the record walk up the rear that stops on the previous walk's
    // path (an END leaf's remainder from its parent); a descent stopped at a has-id leaf with bytes of lo left includes
    // it iff rem(id) >= those bytes. scan_from appends each key + '\n' to `out` and returns their number.
    using ScanScratch = trie::dfuds::detail::ScanScratch<FRef>;
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out, ScanScratch& sc) const {
        trie::dfuds::detail::AppendSink sink{out};
        sc.memo.begin();
        return DtScan(*this).scan_from(lo, k, sc, sink);
    }
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) const {
        ScanScratch sc;
        return scan_from(lo, k, out, sc);
    }
    // f(std::string_view key) per key, in order (a key with a remainder assembled in the scratch).
    template <class F>
    std::size_t for_each_from(std::string_view lo, std::size_t k, F&& f, ScanScratch& sc) const {
        trie::dfuds::detail::CallSink<std::remove_reference_t<F>> sink{f, sc.key};
        sc.memo.begin();
        return DtScan(*this).scan_from(lo, k, sc, sink);
    }
    template <class F>
    std::size_t for_each_from(std::string_view lo, std::size_t k, F&& f) const {
        ScanScratch sc;
        return for_each_from(lo, k, std::forward<F>(f), sc);
    }

    // The leaf-update policy of both tries' batch inserts (Adaptive, max_runs 4, movement factor 1 by default). The
    // LeafRank rear supports no shifted insertion: under Adaptive its leaves take the merge path,
    // counted in rear().last_batch().shift_rej_policy.
    using LeafUpdatePolicy = typename FrontBits::LeafUpdatePolicy;
    using LeafUpdateMode = typename FrontBits::LeafUpdateMode;
    void set_leaf_update_policy(const LeafUpdatePolicy& p) {
        front_.set_leaf_update_policy(p);
        typename RearBp::LeafUpdatePolicy rp;
        rp.mode = static_cast<decltype(rp.mode)>(p.mode);
        rp.max_runs = p.max_runs;
        rp.movement_factor = p.movement_factor;
        rear_.set_leaf_update_policy(rp);
    }
    const LeafUpdatePolicy& leaf_update_policy() const noexcept { return front_.leaf_update_policy(); }
    static LeafUpdatePolicy default_leaf_update_policy() noexcept { return {LeafUpdateMode::Adaptive, 4, 1.0}; }

    std::size_t key_count() const noexcept { return key_count_; }
    std::size_t distinct_remainders() const noexcept { return ids_.live(); }
    unsigned id_width() const noexcept { return W_; }
    const FrontBits& front() const noexcept { return front_; }
    const RearBp& rear() const noexcept { return rear_; }
    // The global tables' logical slots (sentinel and released IDs included) and allocated capacities.
    struct TableCapacity {
        std::size_t slots, free_ids, owner_capacity, handle_capacity; unsigned handle_width;
        std::size_t live_ids = 0, owner_bytes = 0, handle_bytes = 0;   // live remainders; allocated bytes per table
    };
    TableCapacity table_capacity() const noexcept {
        return {ids_.size(), ids_.free_ids().size(), ids_.capacity_slots(), hblock_.capacity_fields(), hblock_.width(),
                ids_.live(), ids_.owner_bytes(), hblock_.bytes()};
    }
    void set_table_growth(TableGrowth g) noexcept { growth_ = g; }   // applies from the next table growth on
    TableGrowth table_growth() const noexcept { return growth_; }

    struct Counts {
        std::size_t front_entries = 0, front_end_leaves = 0, front_id_entries = 0;
        std::size_t rear_entries = 0, rear_end_leaves = 0, rear_leaves = 0, rear_blocks = 0;
    };
    // Computed on demand (one parallel pass over both tries' labels and flags), never inside a build.
    Counts counts() const {
        namespace fb = pmabv::flag_bits;
        Counts c;
        const std::size_t FB = front_.block_count(), RB = rear_.block_count();
        std::vector<std::size_t> fe(FB, 0), fo(FB, 0), re(RB, 0);
        parlay::parallel_for(0, FB, [&](std::size_t b) {
            const std::size_t m = front_.block_label_count(b);
            const std::uint8_t* pl = front_.block_payload(b);
            std::size_t e = 0;
            for (std::size_t r = 0; r < m; ++r) e += pl[r] == kEnd;
            fe[b] = e;
            fo[b] = fb::count_range(front_.block_flags(b), 0, m);
        }, 1);
        parlay::parallel_for(0, RB, [&](std::size_t b) {
            const std::size_t m = rear_.block_label_count(b);
            const std::uint8_t* pl = rear_.block_labels(b);
            std::size_t e = 0;
            for (std::size_t r = 0; r < m; ++r) e += pl[r] == kEnd;
            re[b] = e;
        }, 1);
        c.front_entries = front_.label_count();
        c.rear_entries = rear_.label_count();
        for (std::size_t b = 0; b < FB; ++b) { c.front_end_leaves += fe[b]; c.front_id_entries += fo[b]; }
        for (std::size_t b = 0; b < RB; ++b) c.rear_end_leaves += re[b];
        c.front_end_leaves -= 1;   // the dummy "(" at position 0 carries label 0
        c.rear_end_leaves -= 1;    // the root OPEN carries label 0
        for (std::size_t b = 0; b < RB; ++b) c.rear_leaves += rear_.block_id_count(b);
        if (rear_.size() == 2) c.rear_leaves = 0;   // an empty rear: the root pair's zero field is no remainder
        c.rear_blocks = RB;
        return c;
    }

    // ---- space -----------------------------------------------------------------------------------------------------
    std::size_t object_bytes() const noexcept { return sizeof(DoubleTrie) - sizeof(FrontBits) - sizeof(RearBp); }
    std::size_t byte_size() const noexcept { return space().total; }
    trie::dfuds::SpaceReport space() const {
        const Counts cnt = counts();
        trie::dfuds::SpaceReport r;
        auto add = [&](const char* c, std::size_t v, const char* basis) { r.rows.push_back({c, "total", v, "bytes", true, basis}); r.total += v; };
        auto sub = [&](const char* c, const char* parent, std::size_t v, const char* unit, const char* basis) {
            r.rows.push_back({c, parent, v, unit, false, basis});
        };
        add("front", front_.byte_size(), "the DFUDS front's BitPMAImpl::byte_size() (bits, labels, has-id flags, ids, indexes, summary)");
        add("rear", rear_.byte_size(), "the BP rear's DynamicBP::byte_size() (bits, labels, leaf-rank ids with their per-block counts and samples, no flag channel, indexes, summary)");
        add("handle_table", hblock_.bytes(), "id -> rear block, packed");
        add("owner_counts", ids_.owner_bytes(), "16-bit owner counts, packed");
        add("free_list", ids_.free_list_bytes(), "freed ids (capacity)");
        add("overflow", ids_.overflow_bytes(), "owner counts >= 0xFFFF: estimate, 32 B per entry");
        add("object", object_bytes(), "sizeof(DoubleTrie) minus the two tries (counted by their byte_size)");
        sub("front.labels", "front", front_.label_bytes(), "bytes", "subset of front");
        sub("front.flags", "front", front_.flag_bytes(), "bytes", "subset of front: the has-id channel (no terminal channel)");
        sub("front.ids", "front", front_.id_bytes(), "bytes", "subset of front");
        sub("front.ids.live_bytes", "front.ids", front_.id_live_bytes(), "bytes", "words_for(F * W) per block");
        sub("front.ids.capacity_bytes", "front.ids", front_.id_capacity_bytes(), "bytes", "allocated id words");
        sub("rear.labels", "rear", rear_.label_bytes(), "bytes", "subset of rear");
        sub("rear.flags", "rear", rear_.flag_bytes(), "bytes", "subset of rear: 0 (LeafRank stores no flag channel)");
        sub("rear.ids", "rear", rear_.id_bytes(), "bytes", "subset of rear: fields (capacity) + descriptors + per-block counts + leaf-rank samples");
        sub("rear.ids.live_bytes", "rear.ids", rear_.id_live_bytes(), "bytes", "words_for(F * W) per block");
        sub("rear.ids.capacity_bytes", "rear.ids", rear_.id_capacity_bytes(), "bytes", "allocated id words");
        sub("front.nodes", "front", cnt.front_entries, "count", "front entries (nodes, END leaves included)");
        sub("front.end_leaves", "front", cnt.front_end_leaves, "count", "front END leaves");
        sub("front.id_entries", "front", cnt.front_id_entries, "count", "has-id entries (remainder leaves)");
        sub("rear.nodes", "rear", cnt.rear_entries, "count", "rear nodes (the root and END leaves included)");
        sub("rear.end_leaves", "rear", cnt.rear_end_leaves, "count", "rear END leaves");
        sub("rear.leaves", "rear", cnt.rear_leaves, "count", "rear leaves (= distinct remainders)");
        sub("rear.blocks", "rear", cnt.rear_blocks, "count", "rear PMA blocks");
        sub("keys", "total", key_count_, "count", "stored keys");
        sub("remainders", "total", ids_.live(), "count", "distinct live remainders");
        sub("id_width", "total", W_, "bits", "W (front and rear)");
        sub("handle_width", "total", hblock_.width(), "bits", "handle width");
        return r;
    }

    // Structural validation (parallel, block-wise): pmabv's own checks on both tries; FRONT: every END entry (label 0,
    // the dummy excluded) is the last "(" of a degree >= 2 run, unflagged, and its child is a leaf; flagged entries =
    // the keys with a remainder, their ids in 1..D, per-id references = owner counts; REAR: every END leaf a first
    // child with a next sibling, leaves = D, every field's handle names its block; KEYS: every non-root front leaf
    // (ordinary, END or has-id) ends exactly one key, the root leaf of {""} the empty key: key_count = those leaves.
    void check_invariants() const {
        namespace fb = pmabv::flag_bits;
        front_.check_invariants_parallel();
        rear_.check_invariants_parallel();
        const std::size_t D = ids_.size() - 1;
        std::vector<std::uint32_t> refs(D + 1, 0);
        std::atomic<int> bad{0};
        auto fail = [&](int code) { int z = 0; bad.compare_exchange_strong(z, code); };
        const std::size_t FB = front_.block_count();
        std::vector<std::size_t> fleaves(FB, 0);   // per block: ")" runs (leaves) at positions >= 2
        parlay::parallel_for(0, FB, [&](std::size_t b) {
            const std::size_t live = front_.block_live(b), base = front_.block_start(b);
            const std::uint64_t* w = front_.block_words(b);
            const std::uint8_t* pl = front_.block_payload(b);
            const std::uint64_t* fl = front_.block_flags(b);
            const std::uint64_t* iw = front_.block_ids(b);
            std::size_t r = 0, fi = 0;
            std::size_t lf = 0;
            for (std::size_t i = 0; i < live; ++i) {
                if (!bit(w, i)) {
                    const int prev = i > 0 ? int(bit(w, i - 1)) : (base > 0 ? front_bit(base - 1) : 1);
                    if (base + i >= 2 && prev == 0) ++lf;   // a run that is a lone ")" follows the previous run's ")"
                    continue;
                }
                const bool flagged = fb::get(fl, r);
                if (flagged) {
                    const std::uint64_t id = fb::get_field(iw, fi++, W_);
                    if (id == 0 || id > D) fail(1);
                    else std::atomic_ref<std::uint32_t>(refs[id]).fetch_add(1, std::memory_order_relaxed);
                }
                if (pl[r] == kEnd && base + i != 0) {
                    const int prev = i > 0 ? int(bit(w, i - 1)) : front_bit(base + i - 1);
                    const int next = i + 1 < live ? int(bit(w, i + 1)) : front_bit(base + i + 1);
                    const int next2 = i + 2 < live ? int(bit(w, i + 2)) : front_bit(base + i + 2);
                    if (flagged || prev != 1 || next != 0 || next2 != 0) fail(2);
                }
                ++r;
            }
            if (fi != front_.block_id_count(b)) fail(3);
            fleaves[b] = lf;
        }, 1);
        if (bad.load()) throw std::logic_error(std::string(kWho) + ": front invariant " + std::to_string(bad.load()));
        {
            std::size_t keys = 0;
            for (std::size_t x : fleaves) keys += x;
            const bool root_leaf = front_.get(1) == 0;   // the front "()" of {} and {""}
            if (root_leaf ? (keys != 0 || key_count_ > 1) : keys != key_count_)
                throw std::logic_error(std::string(kWho) + ": key count != the keys the front ends");
        }
        std::atomic<std::size_t> mism{0};
        parlay::parallel_for(1, D + 1, [&](std::size_t id) {
            if (refs[id] != ids_.owners(id)) mism.fetch_add(1, std::memory_order_relaxed);
        }, 4096);
        if (mism.load()) throw std::logic_error(std::string(kWho) + ": owner counts != front references");
        const std::size_t RB = rear_.block_count();
        std::vector<std::size_t> leaves(RB, 0);
        parlay::parallel_for(0, RB, [&](std::size_t b) {
            const std::size_t base = rear_.block_start(b);
            std::size_t k = 0;
            for_each_rear_leaf(b, [&](std::size_t g) {
                if (rear_.label_at(g) == kEnd && (rear_.get(g - 1) != 1 || g + 2 >= rear_.size() || rear_.get(g + 2) != 1)) fail(4);
                ++k;
            });
            (void)base;
            leaves[b] = k;
            const std::size_t root_leaf = (base == 0 && rear_.size() == 2) ? 1 : 0;   // an empty rear's root pair
            if (k + root_leaf != rear_.block_id_count(b)) fail(5);
            const std::uint64_t* f = rear_.block_ids(b);
            for (std::size_t x = 0; x < rear_.block_id_count(b); ++x) {
                const std::uint64_t id = fb::get_field(f, x, W_);
                if (id == 0 ? !root_leaf : (id >= ids_.size() || hblock_.get(id) != b)) fail(6);
            }
        }, 1);
        if (bad.load()) throw std::logic_error(std::string(kWho) + ": rear invariant " + std::to_string(bad.load()));
        std::size_t total = 0;
        for (std::size_t x : leaves) total += x;
        if (total != ids_.live()) throw std::logic_error(std::string(kWho) + ": rear leaves != live remainders");   // freed ids keep their slots
    }

private:
    struct FrontGroups;   // P5's insertion groups (defined with plan_groups below)
    // The bulk build (from_sorted / from_sorted_parallel / the empty construction).
    template <class Keys>
    static std::unique_ptr<DoubleTrie> build_parallel(const Keys& keys, std::size_t bc, std::size_t chunks, bool wide_records,
                                                      TableGrowth growth) {
        using namespace trie::dt::detail;
        const std::size_t n = keys.size();
        const bool nul = parlay::reduce(parlay::delayed_tabulate(n, [&](std::size_t i) -> std::size_t {
            const std::string_view k(keys[i]);
            return !k.empty() && std::memchr(k.data(), 0, k.size()) != nullptr; })) != 0;
        if (nul) throw std::invalid_argument(std::string(kWho) + ": a key contains byte 0 (END); the domain is NUL-free");
        ParallelRanks pr = rank_parallel(keys, chunks, wide_records, "trie::dfuds::DoubleTrie: keys must be strictly increasing");
        const unsigned W = pr.W;
        auto fseq = detail::encode_parallel<FrontBits::Seq>(
            n, [&](std::size_t i) { return std::string_view(keys[i]).substr(0, pr.cut[i]); }, chunks,
            detail::EndIds{pr.key_id.data()}, W);
        pr.cut = {};
        pr.key_id = {};
        FrontBits front(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, std::move(fseq), bc);
        RearBp rear = RawAccess::make<RearBp>(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, encode_rear(pr, W, chunks), bc);
        std::unique_ptr<DoubleTrie> d(new DoubleTrie(std::move(front), std::move(rear), W, n));
        d->growth_ = growth;
        d->hblock_ = PackedFields(bits_for(d->rear_.block_count()));
        {
            Tables t = d->tables();
            reserve_id_tables(t, std::max<std::size_t>(pr.owners.size(), 1));
            d->ids_.assign_parallel(pr.owners.data(), pr.owners.size());
            d->hblock_.resize(d->ids_.size());
        }
        pr.owners = {};
        parlay::parallel_for(0, d->rear_.block_count(), [&](std::size_t b) {
            const std::uint64_t* f = d->rear_.block_ids(b);
            for (std::size_t k = 0; k < d->rear_.block_id_count(b); ++k) {
                const std::uint64_t id = pmabv::flag_bits::get_field(f, k, W);
                if (id != 0) d->hblock_.set_atomic(id, b);
            }
        }, 4);
        RawAccess::clear_rewritten(d->rear_);
        d->front_.clear_rewritten();                 // the front keeps no handles
        if constexpr (transactional) d->check_invariants();
        return d;   // the node counts (counts(), space()) are computed on demand, outside the build
    }

    // The transaction's copy (mutate()) and the empty construction's move; not part of the public interface.
    DoubleTrie(const DoubleTrie&) = default;
    DoubleTrie(DoubleTrie&&) noexcept = default;
    DoubleTrie(FrontBits&& f, RearBp&& r, unsigned W, std::size_t n)
        : front_(std::move(f)), rear_(std::move(r)), hblock_(1), W_(W), key_count_(n) {
        set_leaf_update_policy(default_leaf_update_policy());
    }

    // R1 / F / R2 of a planned batch on this trie (mutate(): the trie itself, or the transaction's copy).
    void apply_batch(trie::detail::ForOverwrite<trie::dt::detail::Rec>& recs,
                     const trie::detail::ForOverwrite<std::size_t>& rec_of, const trie::dt::detail::FrontPlan& pl,
                     FrontGroups& gr, std::size_t added) {
        // ---- R1 ---------------------------------------------------------------------------------------------------
        rear_add_end(recs);
        // ---- F ----------------------------------------------------------------------------------------------------
        if (!pl.aff.empty()) {
            std::vector<FrontBits::FlagIdWriteAt> fw;
            fw.reserve(pl.aff.size());
            for (std::size_t a : pl.aff) fw.push_back({pl.leaves[a].node, false, 0});
            std::sort(fw.begin(), fw.end(), [](const auto& x, const auto& y) { return x.pos < y.pos; });
            front_.set_flags_ids_batch(fw);
        }
        if (!gr.g.empty()) {
            constexpr std::size_t npos = static_cast<std::size_t>(-1);
            parlay::parallel_for(0, gr.str.size(), [&](std::size_t x) {
                const std::size_t c = gr.str_c[x];
                gr.str_id[x] = c == npos ? 0 : recs[rec_of[c]].id;
            }, 2048);
            const auto csr = detail::build_gap_csr_ids<FrontBits::CSR>(
                gr.g.size(), [&](std::size_t g) { return gr.g[g].run_pos; }, [&](std::size_t g) { return gr.g[g].sub_pos; },
                [&](std::size_t g, auto&& f) {
                    for (std::size_t x = gr.g[g].s_lo; x < gr.g[g].s_hi; ++x) f(gr.str[x], gr.str_id[x]);
                }, front_.id_width(), front_.size(), kWho);
            front_.batch_insert_preprocessed(csr, nullptr, kValidateCsr);
        }
        // ---- R2 ---------------------------------------------------------------------------------------------------
        rear_release_end(recs);
        key_count_ += added;
    }
    template <class Apply>
    void mutate(Apply&& apply) {
        if constexpr (transactional) {
            DoubleTrie cand(*this);
            apply(cand);
            cand.check_invariants();
            swap_state(cand);
        } else {
            apply_direct(apply);
        }
    }
    template <class Apply>
    void apply_direct(Apply& apply) noexcept { apply(*this); }
    void swap_state(DoubleTrie& o) noexcept {
        using std::swap;
        swap(front_, o.front_);
        swap(rear_, o.rear_);
        swap(ids_, o.ids_);
        swap(hblock_, o.hblock_);
        swap(W_, o.W_);
        swap(key_count_, o.key_count_);
        swap(growth_, o.growth_);
    }

    static bool bit(const std::uint64_t* w, std::size_t i) noexcept { return (w[i >> 6] >> (i & 63)) & 1ULL; }
    int front_bit(std::size_t p) const { return p < front_.size() ? int(front_.get(p)) : 0; }

    // The rear's label-only packed run: the distinct reversed remainders, r_j + "\0" where r_{j+1} extends r_j (the
    // END leaf), through the BP encoder with terminal flags (every string now ends at a leaf); the flags are dropped.
    static RearBp::Bits::Seq encode_rear(trie::dt::detail::ParallelRanks& pr, unsigned W, std::size_t chunks) {
        namespace fb = pmabv::flag_bits;
        const std::size_t D = pr.rear_keys.size();
        std::vector<std::uint8_t> ext(D, 0);
        parlay::parallel_for(0, D, [&](std::size_t j) {
            if (j + 1 < D && pr.rear_keys[j + 1].size() > pr.rear_keys[j].size() &&
                pr.rear_keys[j + 1].compare(0, pr.rear_keys[j].size(), pr.rear_keys[j]) == 0) ext[j] = 1;
        });
        const auto ei = parlay::pack_index<std::size_t>(parlay::make_slice(ext));
        std::vector<std::string> aug(ei.size());
        std::vector<std::size_t> slot(ei.size() ? D : 0, 0);
        parlay::parallel_for(0, ei.size(), [&](std::size_t x) {
            aug[x].reserve(pr.rear_keys[ei[x]].size() + 1);
            aug[x].assign(pr.rear_keys[ei[x]]);
            aug[x].push_back('\0');
            slot[ei[x]] = x;
        });
        using FlagSeq = pmabv::PackedBitsT<std::uint8_t, true, false>;
        FlagSeq fq = trie::bp::detail::encode_sorted_parallel<FlagSeq>(
            D, [&](std::size_t j) -> std::string_view { return ext[j] ? std::string_view(aug[slot[j]]) : pr.rear_keys[j]; },
            true, "trie::dfuds::DoubleTrie: rear order", chunks);
        const std::size_t flagged = fb::count_range(fq.flg.data(), 0, fq.pay.size());
        if (flagged != D) throw std::logic_error(std::string(kWho) + ": the augmented rear's terminal count != D");
        // the LeafRank run: zero flag words, one W-bit field per leaf in preorder -- leaf j is r_j, id j + 1 (an
        // empty rear: the root pair, a leaf, with a zero field) -- then pmabv's samples
        RearBp::Bits::Seq rs;
        rs.live = fq.live;
        rs.words = std::move(fq.words);
        rs.pay = std::move(fq.pay);
        rs.flg.assign(fb::words_for(rs.pay.size()), 0ULL);
        const std::size_t nf = D == 0 ? 1 : D;
        rs.im.idw = W;
        rs.im.nf = nf;
        rs.im.ids.assign(fb::words_for(nf * W), 0ULL);
        std::uint64_t* const iw = rs.im.ids.data();
        if (D > 0)   // 64 fields per task: a task's fields start and end on word boundaries
            parlay::parallel_for(0, (D + 63) / 64, [&](std::size_t t) {
                for (std::size_t j = 64 * t; j < std::min(D, 64 * t + 64); ++j) fb::set_field(iw, j, W, j + 1);
            }, 16);
        rs.rebuild_fs();
        pr.rear_keys = {};
        pr.arena = {};
        return rs;
    }

    // Leaves of rear block b in order: an OPEN followed by a CLOSE (the next bit: in b, or the first bit of the next
    // nonempty block); global position 0 (the root) is not a leaf. f(global position).
    // The first bit of the next nonempty block after b (0 past the end): the successor of b's last bit.
    std::uint64_t rear_bit_after_block(std::size_t b) const {
        std::size_t nb = b + 1;
        while (nb < rear_.block_count() && rear_.block_live(nb) == 0) ++nb;
        return nb < rear_.block_count() ? (rear_.block_words(nb)[0] & 1ULL) : 0ULL;
    }
    // Word k of block b's LEAF MASK: bit i set iff local i is an OPEN followed by a CLOSE (the root excluded).
    std::uint64_t rear_leaf_word(std::size_t b, std::size_t k, std::size_t live, std::uint64_t after) const {
        const std::uint64_t* w = rear_.block_words(b);
        const std::size_t nw = (live + 63) / 64;
        std::uint64_t x = w[k];
        if (k + 1 == nw && (live & 63)) x &= (1ULL << (live & 63)) - 1ULL;
        std::uint64_t nxt = (x >> 1) | (k + 1 < nw ? (w[k + 1] << 63) : 0ULL);
        if (k + 1 == nw) {   // bit live-1's successor is `after`
            const std::size_t t = (live - 1) & 63;
            nxt = (nxt & ~(1ULL << t)) | (after << t);
        }
        std::uint64_t m = x & ~nxt;
        if (k == 0 && rear_.block_start(b) == 0) m &= ~1ULL;
        return m;
    }
    template <class F>
    void for_each_rear_leaf(std::size_t b, F&& f) const {
        const std::size_t live = rear_.block_live(b), base = rear_.block_start(b);
        if (live == 0) return;
        const std::uint64_t after = rear_bit_after_block(b);
        for (std::size_t k = 0; k < (live + 63) / 64; ++k)
            for (std::uint64_t m = rear_leaf_word(b, k, live, after); m; m &= m - 1)
                f(base + 64 * k + static_cast<std::size_t>(std::countr_zero(m)));
    }
    // The remainder id ending at the rear node `node` (an OPEN): its own id when it is a leaf, its END child's id when
    // its first child is END, 0 otherwise (no remainder ends there; the root never ends one). pmabv's leaf-rank read.
    std::uint64_t rear_end_id(const RRef& node) const {
        if (node.slot.logical_pos() == 0) return 0;
        const RRef nx = rear_.record_next(node);
        if (nx.bit == 0) return rear_.id_of(node);                             // a leaf
        if (nx.payload != kEnd) return 0;                                      // first child ordinary
        return rear_.id_of(nx);                                                 // the END child (a leaf by invariant)
    }
    // The rear OPEN of remainder `id`: its handle's block, the field holding id, that leaf of the block.
    std::size_t rear_node(std::uint64_t id) const { return rear_record(id).slot.logical_pos(); }
    // Is `rest` (non-empty) exactly the remainder of `id`? Walk up from its node (an END leaf: from its parent).
    bool remainder_equals(std::string_view rest, std::uint64_t id) const {
        std::size_t o = rear_node(id);
        if (rear_.label_at(o) == kEnd) o = trie::dt::detail::parent_of(rear_, o);
        for (std::size_t k = 0; k < rest.size(); ++k) {
            if (o == 0 || rear_.label_at(o) != static_cast<std::uint8_t>(rest[k])) return false;
            o = trie::dt::detail::parent_of(rear_, o);
        }
        return o == 0;
    }

    // ---- P5 with END: the front's insertion groups ------------------------------------------------------------------
    // The front's insertion groups with the END transitions as strings of the groups (the END
    // string is the one byte kEnd, contribution npos, no id; it sorts before every other string of its group):
    //   run keys (no leaf, bytes left) and flips (no leaf, the whole path present, not owned) form one stretch per gap
    //     (s(u), j): a flip contributes the END string -- END-ADD at the gap deg(u), merged with the new children of
    //     that gap; a stretch whose node is a terminal LEAF gets the END string first (END-PREPEND); a stretch that is
    //     only the END string at the root LEAF (the empty key into an empty set) is no group: the key count alone
    //     records it (a root leaf with keys is {""});
    //   affected leaves (a remainder leaf landed on): the END string first when a batch key equals path(v)
    //     (keep_flag: path(v) stays a key through END), then the members' strings below v and v's new prefix.
    struct FrontGroups {
        struct Group { std::size_t run_pos, sub_pos, s_lo, s_hi; };
        std::vector<Group> g;
        std::vector<std::string_view> str;
        std::vector<std::size_t> str_c;
        std::vector<std::uint64_t> str_id;
    };
    static constexpr char kEndStr[1] = {0};
    FrontGroups plan_groups(trie::dt::detail::FrontPlan& pl, const trie::detail::ForOverwrite<detail::FPre>& fp,
                            const std::vector<std::uint8_t>& term_leaf) const {
        using trie::dt::detail::KeyInfo;
        using trie::dt::detail::Leaf;
        namespace dd = trie::dt::detail;
        constexpr std::size_t npos = static_cast<std::size_t>(-1);
        const std::string_view end_s(kEndStr, 1);
        const auto& ks = pl.ks;
        const std::size_t n = ks.size();
        auto is_unit = [&](std::size_t i) { return ks[i].leaf == npos; };              // run keys and flips
        auto head = [&](std::size_t i) {
            const detail::FPre& f = fp[ks[i].stop];
            return std::pair<std::size_t, std::uint16_t>(f.s_u, f.j);
        };
        auto str_of = [&](std::size_t i) -> std::string_view {                          // relative to u
            return ks[i].d == ks[i].b.size() ? end_s : ks[i].b.substr(ks[i].d, ks[i].cut - ks[i].d);
        };
        const auto rs = dd::pack_indices_par(n, [&](std::size_t i) {
            return is_unit(i) && (i == 0 || !is_unit(i - 1) || head(i - 1) != head(i)); });
        const std::size_t R = rs.size();
        trie::detail::ForOverwrite<std::size_t> re(R);
        parlay::parallel_for(0, R, [&](std::size_t r) {
            const auto h = head(rs[r]);
            std::size_t e = rs[r] + 1;
            while (e < n && is_unit(e) && head(e) == h) ++e;
            re[r] = e;
        }, 256);
        // the END-PREPEND of a stretch at a terminal leaf, and the lone root-leaf END that is no group
        auto prepend = [&](std::size_t r) -> std::size_t {
            const std::size_t i0 = rs[r];
            return fp[ks[i0].stop].aux == detail::kLeaf && term_leaf[ks[i0].stop] && ks[i0].d < ks[i0].b.size() ? 1 : 0;
        };
        auto lone_root = [&](std::size_t r) {
            const std::size_t i0 = rs[r];
            return re[r] == i0 + 1 && ks[i0].d == 0 && ks[i0].b.empty() && fp[ks[i0].stop].aux == detail::kLeaf;
        };
        auto leaf_strings = [&](const Leaf& lf, auto&& emit) {
            if (lf.keep_flag) emit(end_s, npos);
            const std::string_view kr = std::string_view(lf.r).substr(0, lf.cut - lf.depth);
            bool k_done = false;
            for (std::size_t m = lf.m_lo; m < lf.m_hi; ++m) {
                const KeyInfo& ki = ks[m];
                if (ki.cut == lf.depth) continue;                    // == path(v): the END string above
                const std::string_view bf = ki.b.substr(lf.depth, ki.cut - lf.depth);
                if (!k_done && kr < bf) { emit(kr, lf.new_c); k_done = true; }
                emit(bf, ki.c);
            }
            if (!k_done) emit(kr, lf.new_c);
        };
        const std::size_t A = pl.aff.size();
        trie::detail::ForOverwrite<std::size_t> lstr(A);
        parlay::parallel_for(0, A, [&](std::size_t a) {
            std::size_t c = 0;
            leaf_strings(pl.leaves[pl.aff[a]], [&](std::string_view, std::size_t) { ++c; });
            lstr[a] = c;
        }, 64);
        struct Unit { std::size_t key, idx; bool leaf; };
        const auto units0 = parlay::merge(parlay::tabulate(R, [&](std::size_t r) { return Unit{rs[r], r, false}; }),
                                          parlay::tabulate(A, [&](std::size_t a) { return Unit{pl.leaves[pl.aff[a]].m_lo, a, true}; }),
                                          [](const Unit& x, const Unit& y) { return x.key < y.key; });
        const auto units = parlay::filter(units0, [&](const Unit& u) { return u.leaf || !lone_root(u.idx); });
        const std::size_t U = units.size();
        const auto soff = dd::prefix_sums_par(U, [&](std::size_t u) {
            return units[u].leaf ? lstr[units[u].idx] : re[units[u].idx] - rs[units[u].idx] + prepend(units[u].idx); });
        FrontGroups out;
        out.g.resize(U);
        out.str.resize(soff[U]);
        out.str_c.resize(soff[U]);
        out.str_id.assign(soff[U], 0);
        parlay::parallel_for(0, U, [&](std::size_t u) {
            std::size_t si = soff[u];
            FrontGroups::Group& g = out.g[u];
            g.s_lo = si;
            if (!units[u].leaf) {
                const std::size_t r = units[u].idx, i0 = rs[r];
                const detail::FPre& f = fp[ks[i0].stop];
                g.run_pos = f.s_u + f.j;
                g.sub_pos = trie::dfuds::detail::sub_pos_of(front_, f);
                if (prepend(r)) { out.str[si] = end_s; out.str_c[si] = npos; ++si; }
                for (std::size_t i = i0; i < re[r]; ++i, ++si) {
                    out.str[si] = str_of(i);
                    out.str_c[si] = ks[i].d == ks[i].b.size() ? npos : ks[i].c;
                }
            } else {
                const Leaf& lf = pl.leaves[pl.aff[units[u].idx]];
                const detail::FPre& f = fp[ks[lf.m_lo].stop];
                g.run_pos = f.s_u + f.j;
                g.sub_pos = trie::dfuds::detail::sub_pos_of(front_, f);
                leaf_strings(lf, [&](std::string_view x, std::size_t c) {
                    out.str[si] = x;
                    out.str_c[si] = c;
                    ++si;
                });
            }
            g.s_hi = si;
        }, 64);
        return out;
    }

    // ---- the rear side of a batch: R1 / R2 with END leaves -------------------------------------------------------------
    // The global tables' view the shared trie::dt::detail helpers expect (st.rear, st.ids, st.hblock, st.growth).
    struct Tables {
        RearBp& rear;
        trie::dt::detail::IdTable& ids;
        trie::dt::detail::PackedFields& hblock;
        TableGrowth growth;
    };
    Tables tables() { return Tables{rear_, ids_, hblock_, growth_}; }
    // rem(id) in forward order: labels from its rear node up to the root (an END leaf: from its parent).
    std::string remainder_of(std::uint64_t id) const {
        std::string out;
        std::size_t o = rear_node(id);
        if (rear_.label_at(o) == kEnd) o = trie::dt::detail::parent_of(rear_, o);
        for (; o != 0; o = trie::dt::detail::parent_of(rear_, o)) out.push_back(static_cast<char>(rear_.label_at(o)));
        return out;
    }
    // A new remainder id: its table slots (dt::detail::prepare_id_tables / acquire_id_slot), both tries widened when it
    // needs one more bit.
    std::uint64_t alloc_id() {
        Tables t = tables();
        trie::dt::detail::prepare_id_tables(t, 1, [] { return std::vector<trie::dt::detail::TerminalGrowth>{}; });
        const std::uint64_t id = trie::dt::detail::acquire_id_slot(t);
        if (!pmabv::flag_bits::fits_field(id, W_)) {
            ++W_;
            front_.set_id_width(W_);
            trie::dt::detail::RawAccess::set_id_width(rear_, W_);
        }
        return id;
    }
    // After a rear mutation: the handles of every field of a rewritten block (leaf-rank fields: block_id_count).
    void refresh_handles() {
        const unsigned Wb = trie::dt::detail::bits_for(rear_.block_count());
        if (Wb > hblock_.width()) { Tables t = tables(); trie::dt::detail::widen_handles(t, Wb); }
        auto one = [&](std::size_t b) {
            const std::size_t nf = rear_.block_id_count(b);
            const std::uint64_t* f = rear_.block_ids(b);
            for (std::size_t i = 0; i < nf; ++i) {
                const std::uint64_t id = pmabv::flag_bits::get_field(f, i, W_);
                if (id != 0) hblock_.set_atomic(id, b);
            }
        };
        if (rear_.rewritten_all()) parlay::parallel_for(0, rear_.block_count(), [&](std::size_t b) { one(b); }, 4);
        else {
            const auto& list = rear_.rewritten_blocks();
            parlay::parallel_for(0, list.size(), [&](std::size_t i) { one(list[i]); }, 4);
        }
        trie::dt::detail::RawAccess::clear_rewritten(rear_);
    }
    bool rear_is_leaf(std::size_t pos) const { return pos + 1 < rear_.size() && rear_.get(pos + 1) == 0; }

    // R1: every record with delta > 0: owners += delta when its id is known; otherwise the rear walk of the fresh
    // strings (sorted) decides -- a path that ends a live remainder (a leaf, or a node with END) reuses its id; an
    // existing path ending at an internal node without END gets the END child (END-ADD, a new id); a missing path
    // becomes a group string at its insertion slot with a new id, and when the deepest existing node is a remainder
    // LEAF its group starts with END carrying that leaf's id (END-PREPEND: the leaf turns internal and loses its
    // field). Inside a group a string that is a strict prefix of the next gets END under its node, so every string
    // ends at a leaf and the group's leaves in preorder are its strings: the leaf-rank fields are the strings' ids in
    // order. ONE pmabv batch insert, ONE handle refresh.
    void rear_add_end(trie::detail::ForOverwrite<trie::dt::detail::Rec>& recs) {
        using namespace trie::dt::detail;
        constexpr std::size_t npos = static_cast<std::size_t>(-1);
        const std::string_view end_s(kEndStr, 1);
        std::vector<std::size_t> fresh;
        std::vector<std::string_view> fresh_str;
        for (std::size_t u = 0; u < recs.size(); ++u)
            if (recs[u].delta > 0 && recs[u].id == 0) { fresh.push_back(u); fresh_str.push_back(recs[u].key); }
        struct RW { bool ok; std::size_t node, d, stop; std::uint64_t end_id; bool leaf; };
        std::vector<RW> rw(fresh.size());
        if (!fresh.empty()) {
            pmabv::ParGuard guard;
            trie::bp::detail::sorted_walk_par(rear_, rear_.record_at(0), 0, fresh_str.data(), 0, fresh_str.size(), kWalkGrain, guard,
                [&](std::size_t i, bool ok, const RRef& node, std::size_t d, std::size_t stop) {
                    const std::size_t p = node.slot.logical_pos();
                    const bool leaf = p != 0 && rear_is_leaf(p);
                    rw[i] = RW{ok, p, d, ok ? npos : stop, ok ? rear_end_id(node) : (leaf ? rear_end_id(node) : 0), leaf};
                });
            guard.rethrow();
        }
        std::size_t new_ids = 0;
        for (const RW& w : rw) new_ids += !(w.ok && w.end_id != 0);
        Tables t = tables();
        // The planned terminal entries per receiving block (dt::detail::TerminalGrowth):
        // exactly the entries of the CSR built below -- END-ADD and END-PREPEND one END leaf each; a group string its
        // OPENs beyond the previous string of its group, plus the END under that previous string when it extends it.
        prepare_id_tables(t, new_ids, [&] {
            std::vector<TerminalGrowth> pending;
            std::size_t gpos = npos;
            std::string_view prev;
            bool has_prev = false;
            auto put = [&](std::size_t pos, std::string_view x) {
                if (pos != gpos) {
                    const std::size_t b = rear_.record_at(pos).slot.block();
                    if (pending.empty() || pending.back().block != b) pending.push_back({b, 0});
                    gpos = pos;
                    has_prev = false;
                }
                std::size_t e = x.size();
                if (has_prev) {
                    const std::size_t l = common_prefix(prev, x);
                    e = e - l + (l == prev.size() && x.size() > prev.size());
                }
                if (e > std::numeric_limits<std::size_t>::max() - pending.back().entries)
                    throw std::length_error("DT batch terminal entries overflow");
                pending.back().entries += e;
                prev = x;
                has_prev = true;
            };
            for (std::size_t i = 0; i < fresh.size(); ++i) {
                const RW& w = rw[i];
                if (w.ok && w.end_id != 0) continue;
                if (w.ok) { put(w.node + 1, end_s); continue; }
                if (w.leaf && !(gpos == w.stop && has_prev)) put(w.stop, end_s);
                put(w.stop, fresh_str[i].substr(w.d));
            }
            return pending;
        });
        for (const Rec& rec : recs)
            if (rec.delta > 0 && rec.id != 0) ids_.set_owners(rec.id, ids_.owners(rec.id) + static_cast<std::uint64_t>(rec.delta));
        std::vector<CsrGroup> groups;
        auto group_at = [&](std::size_t pos) -> CsrGroup& {
            if (groups.empty() || groups.back().pos != pos) groups.push_back(CsrGroup{pos, {}});
            return groups.back();
        };
        for (std::size_t i = 0; i < fresh.size(); ++i) {
            Rec& rec = recs[fresh[i]];
            const RW& w = rw[i];
            if (w.ok && w.end_id != 0) {                                          // a live remainder: shared
                rec.id = w.end_id;
                ids_.set_owners(rec.id, ids_.owners(rec.id) + static_cast<std::uint64_t>(rec.delta));
                continue;
            }
            rec.id = alloc_id();
            ids_.set_owners(rec.id, static_cast<std::uint64_t>(rec.delta));
            if (w.ok) { group_at(w.node + 1).strs.push_back({end_s, rec.id}); continue; }   // END-ADD
            CsrGroup& g = group_at(w.stop);
            if (w.leaf && g.strs.empty()) g.strs.push_back({end_s, w.end_id});    // END-PREPEND (once per group)
            g.strs.push_back({fresh_str[i].substr(w.d), rec.id});
        }
        if (groups.empty()) { refresh_handles(); return; }
        // END under a string that the next string of its group extends: the augmented copies live in `aug`
        std::size_t extra = 0;
        for (const CsrGroup& g : groups)
            for (std::size_t k = 0; k + 1 < g.strs.size(); ++k)
                extra += strict_prefix(g.strs[k].suf, g.strs[k + 1].suf);
        std::vector<std::string> aug;
        aug.reserve(extra);
        for (CsrGroup& g : groups)
            for (std::size_t k = 0; k + 1 < g.strs.size(); ++k)
                if (strict_prefix(g.strs[k].suf, g.strs[k + 1].suf)) {
                    aug.emplace_back(g.strs[k].suf);
                    aug.back().push_back('\0');
                    g.strs[k].suf = aug.back();
                }
        const auto csr = make_rear_csr(groups);
        trie::dt::detail::RawAccess::batch_insert_csr(rear_, csr);
        refresh_handles();
    }
    // pmabv's LeafRank CSR for the groups: the BP forest encoder, every string's endpoint a leaf whose field is
    // the string's id (the groups were augmented so that no string is a prefix of the next).
    RearBp::Bits::CSR make_rear_csr(const std::vector<trie::dt::detail::CsrGroup>& groups) const {
        return trie::dt::detail::build_csr<RearBp::Bits::CSR>(groups, W_);
    }
    static bool strict_prefix(std::string_view a, std::string_view b) {
        return b.size() > a.size() && b.compare(0, a.size(), a) == 0;
    }

    // R2: records with delta < 0: owners -= ; the ids reaching zero are released together. A rear node is removable
    // iff it is not the root and every leaf of its subtree ends a released remainder (a node ending a live remainder
    // holds its live END leaf); maximal removable subtrees are erased as one batch. A surviving parent left with only
    // its END child collapses: the END pair is erased too and the parent, now a leaf, carries the END's id. The
    // climb from a released leaf stops at the first ancestor that ends a live remainder (above it nothing is
    // removable).
    std::size_t rear_release_end(const trie::detail::ForOverwrite<trie::dt::detail::Rec>& recs) {
        using namespace trie::dt::detail;
        constexpr std::size_t npos = static_cast<std::size_t>(-1);
        std::vector<std::uint64_t> released;
        for (const Rec& rec : recs) {
            if (rec.delta >= 0) continue;
            const std::uint64_t o = ids_.owners(rec.id), dec = static_cast<std::uint64_t>(-rec.delta);
            if (o > dec) { ids_.set_owners(rec.id, o - dec); continue; }
            if (o < dec) throw std::logic_error(std::string(kWho) + ": owners below zero");
            released.push_back(rec.id);
        }
        if (released.empty()) return 0;
        const std::size_t m = released.size();
        auto rel = parlay::tabulate(m, [&](std::size_t i) { return rear_node(released[i]); }, 64);
        parlay::sort_inplace(rel);
        auto is_rel = [&](std::size_t p) { return std::binary_search(rel.begin(), rel.end(), p); };
        // a node ends a LIVE remainder: a leaf not released, or its first child an END leaf not released
        auto live_end = [&](std::size_t x) {
            if (rear_is_leaf(x)) return !is_rel(x);
            return rear_.label_at(x + 1) == kEnd && rear_is_leaf(x + 1) && !is_rel(x + 1);
        };
        // candidates: every released leaf and its ancestors up to (and including) the first that ends a live
        // remainder, the root excluded -- two parallel passes (lengths, one flat array), a parallel sort, the distinct
        auto len = parlay::tabulate(m, [&](std::size_t i) {
            std::size_t l = 0;
            for (std::size_t x = rel[i]; x != 0; x = parent_of(rear_, x)) { ++l; if (x != rel[i] && live_end(x)) break; }
            return l;
        }, 16);
        const auto [off, tot] = parlay::scan(len);
        parlay::sequence<std::size_t> all(tot);
        parlay::parallel_for(0, m, [&](std::size_t i) {
            std::size_t k = off[i];
            for (std::size_t x = rel[i]; x != 0; x = parent_of(rear_, x)) { all[k++] = x; if (x != rel[i] && live_end(x)) break; }
        }, 16);
        parlay::sort_inplace(all);
        const auto cand = parlay::unique(all);
        all = parlay::sequence<std::size_t>();
        const std::size_t C = cand.size();
        // removable: no live leaf in the subtree (a scan in BP order with records, stopping at the first live leaf)
        parlay::sequence<unsigned char> rem(C, 0);
        parlay::sequence<std::size_t> close_of(C, 0);
        parlay::parallel_for(0, C, [&](std::size_t i) {
            const std::size_t x = cand[i];
            const std::size_t close = *rear_.find_close(x);
            close_of[i] = close;
            RRef c = rear_.record_at(x);
            bool clean = true;
            while (c.slot.logical_pos() < close) {
                const RRef nx = rear_.record_next(c);
                if (c.bit == 1 && nx.bit == 0 && !is_rel(c.slot.logical_pos())) { clean = false; break; }
                c = nx;
            }
            rem[i] = clean;
        }, 16);
        auto cand_idx = [&](std::size_t p) {
            const auto it = std::lower_bound(cand.begin(), cand.end(), p);
            return it != cand.end() && *it == p ? static_cast<std::size_t>(it - cand.begin()) : npos;
        };
        auto parent = parlay::tabulate(C, [&](std::size_t i) { return rem[i] ? parent_of(rear_, cand[i]) : std::size_t{0}; }, 64);
        const auto maximal = parlay::filter(parlay::iota<std::size_t>(C), [&](std::size_t i) {
            if (!rem[i]) return false;
            const std::size_t p = parent[i];
            if (p == 0) return true;
            const std::size_t j = cand_idx(p);
            return j == npos || !rem[j];
        });
        auto ranges = parlay::tabulate(maximal.size(), [&](std::size_t q) {
            const std::size_t i = maximal[q];
            return std::pair<std::size_t, std::size_t>{cand[i], close_of[i] + 1};
        });
        auto parents = parlay::filter(parlay::map(maximal, [&](std::size_t i) { return parent[i]; }), [](std::size_t p) { return p != 0; });
        parlay::sort_inplace(parents);
        parents = parlay::unique(parents);
        // collapses: a surviving parent whose remaining children are exactly its live END leaf (ranges sorted: lookups)
        auto in_ranges = [&](std::size_t c) {
            const auto it = std::lower_bound(ranges.begin(), ranges.end(), std::pair<std::size_t, std::size_t>{c, 0});
            return it != ranges.end() && it->first == c;
        };
        parlay::sequence<std::uint64_t> cid(parents.size(), 0);   // the END's id, 0: no collapse
        parlay::parallel_for(0, parents.size(), [&](std::size_t q) {
            const std::size_t p = parents[q];
            if (rear_.label_at(p + 1) != kEnd || !rear_is_leaf(p + 1) || is_rel(p + 1)) return;
            for (std::size_t c = *rear_.find_close(p + 1) + 1; c < rear_.size() && rear_.get(c) == 1; c = *rear_.find_close(c) + 1)
                if (!in_ranges(c)) return;                         // another child survives
            cid[q] = rear_end_id(rear_.record_at(p));
        }, 16);
        const auto cq = parlay::filter(parlay::iota<std::size_t>(parents.size()), [&](std::size_t q) { return cid[q] != 0; });
        auto all_ranges = parlay::append(ranges, parlay::map(cq, [&](std::size_t q) {
            return std::pair<std::size_t, std::size_t>{parents[q] + 1, parents[q] + 3}; }));
        parlay::sort_inplace(all_ranges);
        // a collapsed END pair next to a removed sibling subtree, or removed sibling subtrees side by side: one range
        // ([a, b) then [b, c) is [a, c)), the same erase with one predecessor re-key per chain
        std::vector<std::pair<std::size_t, std::size_t>> er;
        er.reserve(all_ranges.size());
        for (const auto& r : all_ranges) {
            if (!er.empty() && er.back().second == r.first) er.back().second = r.second;
            else er.push_back(r);
        }
        if (!er.empty()) RawAccess::erase_ranges(rear_, er);
        // a collapsed parent's position after the erase: minus the erased bits before it (prefix sums, binary search)
        auto lens = parlay::map(all_ranges, [](const auto& r) { return r.second - r.first; });
        const auto [pre, ptot] = parlay::scan(lens);
        (void)ptot;
        std::vector<typename RearBp::Bits::FlagIdWriteAt> w(cq.size());
        parlay::parallel_for(0, cq.size(), [&](std::size_t t) {
            const std::size_t p = parents[cq[t]];
            const std::size_t k = static_cast<std::size_t>(std::lower_bound(all_ranges.begin(), all_ranges.end(),
                std::pair<std::size_t, std::size_t>{p, 0}) - all_ranges.begin());   // ranges starting before p
            w[t] = {p - (k ? pre[k - 1] + lens[k - 1] : 0), false, cid[cq[t]]};
        }, 256);
        if (!w.empty()) RawAccess::set_flags_ids_batch(rear_, w);
        refresh_handles();
        for (const auto& x : w) hblock_.set(x.id, rear_.record_at(x.pos).slot.block());   // the field written after the erase
        for (std::uint64_t id : released) ids_.release(id);
        return released.size();
    }

    // ---- the rear's leaf-rank fields (pmabv's LeafRank channel: in leaf order per block) --------------------------------
    const std::uint64_t* rear_fields(std::size_t b) const noexcept { return rear_.block_ids(b); }
    std::size_t rear_field_count(std::size_t b) const noexcept { return rear_.block_id_count(b); }
    // The rear RECORD of remainder `id` (its leaf, an END leaf included): handle -> block -> field -> leaf select,
    // with the leaf's open rank for the record coordinates.
    RRef rear_record(std::uint64_t id) const {
        const std::size_t b = hblock_.get(id);
        const std::size_t nf = rear_field_count(b);
        const std::size_t k = find_field(rear_fields(b), nf, W_, id);
        if (k == nf) throw std::logic_error(std::string(kWho) + ": stale handle (id not in its block)");
        const std::size_t local = rear_.block_leaf_select(b, k);              // pmabv's sampled leaf select
        return rear_.record_at_coordinate(b, local, rear_.block_start(b) + local);
    }
    // The index of the W-bit field equal to `id` among the first nf (nf when absent): a linear search, the fields
    // read by one running bit offset.
    static std::size_t find_field(const std::uint64_t* f, std::size_t nf, unsigned W, std::uint64_t id) noexcept {
        const std::uint64_t mask = W == 64 ? ~0ULL : ((1ULL << W) - 1ULL);
        std::size_t bit = 0;
        for (std::size_t k = 0; k < nf; ++k, bit += W) {
            const std::size_t wi = bit >> 6, sh = bit & 63;
            std::uint64_t v = f[wi] >> sh;
            if (sh + W > 64) v |= f[wi + 1] << (64 - sh);
            if ((v & mask) == id) return k;
        }
        return nf;
    }
    // The record where rem(id)'s labels start: its leaf, or an END leaf's parent.
    RRef rear_start(std::uint64_t id) const {
        RRef r = rear_record(id);
        if (r.payload == kEnd) trie::dt::detail::rear_parent(rear_, r);
        return r;
    }
    // rem(id) appended by the record walk that stops on the previous walk's path (dt_rear.hpp
    // append_remainder_via_path, from rear_start).
    void append_remainder_via_path(std::uint64_t id, std::string& out, std::vector<std::size_t>& path_pos,
                                   std::string& path_labels, std::vector<std::size_t>& walk_pos) const {
        RRef r = rear_start(id);
        const std::size_t start = out.size();
        walk_pos.clear();
        std::size_t j = 0;
        bool hit = false;
        for (;;) {
            const std::size_t p = r.slot.logical_pos();
            while (j < path_pos.size() && path_pos[j] > p) ++j;
            if (j < path_pos.size() && path_pos[j] == p) { hit = true; break; }
            out.push_back(static_cast<char>(r.payload));
            walk_pos.push_back(p);
            if (!trie::dt::detail::rear_parent(rear_, r)) break;
        }
        if (hit) {
            out.append(path_labels, j, std::string::npos);
            walk_pos.insert(walk_pos.end(), path_pos.begin() + static_cast<std::ptrdiff_t>(j), path_pos.end());
        }
        path_pos.swap(walk_pos);
        path_labels.assign(out, start, std::string::npos);
    }
    // rem(id) >= rest (unsigned bytes), by the record walk (dt_rear.hpp remainder_at_least, from rear_start).
    bool remainder_at_least(std::string_view rest, std::uint64_t id) const {
        RRef r = rear_start(id);
        for (std::size_t k = 0;; ++k) {
            if (k == rest.size()) return true;
            const std::uint8_t c = r.payload, x = static_cast<std::uint8_t>(rest[k]);
            if (c != x) return c > x;
            if (!trie::dt::detail::rear_parent(rear_, r)) return k + 1 == rest.size();
        }
    }

    // The scan over the front (trie/dfuds/detail/scan.hpp's Scan on FrontBits, with the key ends above).
    class DtScan {
    public:
        using Ref = FRef;
        using Frame = trie::dfuds::detail::ScanFrame<Ref>;
        using Start = trie::dfuds::detail::ScanStart<Ref>;
        explicit DtScan(const DoubleTrie& d) noexcept : D_(d), b_(d.front_), ops_{d.front_} {}

        template <class Sink>
        std::size_t scan_from(std::string_view lo, std::size_t k, ScanScratch& sc, Sink& sink) const {
            if (k == 0) return 0;
            return enumerate(seek(lo, sc), k, sc, sink);
        }
        Start seek(std::string_view lo, ScanScratch& sc) const {
            using St = detail::State;
            auto& sf = sc.frames;
            sf.clear();
            std::size_t i = 0;
            Ref cur = b_.record_at(1);
            Ref edge = cur;
            enum { kAt, kPop, kChild } how = kAt;
            std::size_t x = 0, d = 0;
            bool in_block = true;
            St::RunBuf rb;
            for (;;) {
                if (i == lo.size()) break;
                if (cur.bit != 1) {                                            // a leaf with bytes of lo left
                    const bool in = i > 0 && b_.flag_of(edge) && D_.remainder_at_least(lo.substr(i), b_.id_of(edge));
                    how = in ? kAt : kPop;
                    break;
                }
                const std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), live = b_.block_live(blk);
                const std::size_t pos = cur.slot.logical_pos(), rk = cur.open_rank;
                const std::uint64_t* w = b_.block_words(blk);
                const std::uint8_t* pl = b_.block_payload(blk);
                const std::uint8_t c = static_cast<std::uint8_t>(lo[i]);
                const std::uint8_t* labels = nullptr;
                if (local + 1 < live && !St::bit_at(w, local + 1)) {
                    const std::size_t m = St::unary_pairs(w, local, live, lo.size() - i);
                    const std::size_t l = St::match_len(pl + rk, lo.data() + i, m);
                    if (l > 0) {
                        edge = pair_edge(blk, local, pos, rk, l - 1);
                        cur = pair_node(blk, local, live, pos, rk, l);
                        i += l;
                        continue;
                    }
                    labels = pl + rk;
                    d = 1;
                    in_block = true;
                } else {
                    const std::size_t d0 = St::run_length(w, local, live);
                    in_block = local + d0 < live;
                    if (in_block) { labels = pl + rk; d = d0; }
                    else { ops_.read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
                }
                x = St::lower_desc(labels, d, c);
                if (c != kEnd && x < d && labels[x] == c) {
                    if (x >= 1) sf.push_back(Frame{cur, i, x - 1, in_block});
                    const Ref e = edge_at(cur, x, in_block);
                    const Ref child = x + 1 == d ? after_run(cur, d, in_block) : ops_.child_of(e);
                    cur = child;
                    edge = e;
                    ++i;
                    continue;
                }
                how = x == 0 ? kPop : kChild;
                break;
            }
            sc.path.assign(lo.substr(0, i));
            if (how != kChild) return Start{how == kPop, cur, edge, i};
            if (x >= 2) sc.frames.push_back(Frame{cur, i, x - 2, in_block});
            const Ref e = edge_at(cur, x - 1, in_block);
            const Ref child = x == d ? after_run(cur, d, in_block) : ops_.child_of(e);
            sc.path.push_back(static_cast<char>(e.payload));
            return Start{false, child, e, i + 1};
        }
        template <class Sink>
        std::size_t enumerate(const Start& st, std::size_t k, ScanScratch& sc, Sink& sink) const {
            using St = detail::State;
            auto& fr = sc.frames;
            std::string& path = sc.path;
            Ref cur = st.node, edge = st.edge;
            std::size_t depth = st.depth, count = 0;
            if (k == 0) return 0;
            if (st.pop) {
                if (!pop_child(fr, path, edge, depth)) return 0;
                cur = ops_.child_of(edge);
            }
            if (cur.bit != 1) {
                if (depth == 0) {
                    if (D_.key_count_ > 0) { sink.key(path); if (++count == k) return count; }
                } else if (emit_leaf(edge, path, depth, sink, sc) && ++count == k) {
                    return count;
                }
            }
            for (;;) {
                const std::size_t blk = cur.slot.block(), local = cur.slot.local_slot(), live = b_.block_live(blk);
                const std::size_t pos = cur.slot.logical_pos(), rk = cur.open_rank;
                if (cur.bit == 1) {
                    const std::uint64_t* w = b_.block_words(blk);
                    if (local + 1 < live && !St::bit_at(w, local + 1)) {
                        const std::size_t m = St::unary_pairs(w, local, live, live);
                        path.append(reinterpret_cast<const char*>(b_.block_payload(blk) + rk), m);
                        edge = pair_edge(blk, local, pos, rk, m - 1);
                        cur = pair_node(blk, local, live, pos, rk, m);
                        depth += m;
                    } else {
                        const std::size_t d0 = St::run_length(w, local, live);
                        if (local + d0 < live) {
                            edge = b_.record_at_coordinate(blk, local + d0 - 1, pos + d0 - 1, rk + d0 - 1);
                            if (d0 >= 2) fr.push_back(Frame{cur, depth, d0 - 2, true});
                            cur = node_at(blk, local + d0 + 1, live, pos + d0 + 1, rk + d0);
                        } else {
                            St::RunBuf rb;
                            ops_.read_run_slow(cur, rb);
                            const std::size_t d = rb.d;
                            edge = b_.record_at(pos + d - 1);
                            if (d >= 2) fr.push_back(Frame{cur, depth, d - 2, false});
                            cur = b_.record_at(pos + d + 1);
                        }
                        path.push_back(static_cast<char>(edge.payload));
                        ++depth;
                    }
                } else {
                    if (!pop_child(fr, path, edge, depth)) return count;
                    cur = node_at(blk, local + 1, live, pos + 1, rk);
                }
                if (cur.bit != 1 && emit_leaf(edge, path, depth, sink, sc) && ++count == k) return count;
            }
        }

    private:
        // A leaf reached through `edge`: END ends its parent's path, a has-id leaf path + rem(id), any other its path.
        template <class Sink>
        bool emit_leaf(const Ref& edge, const std::string& path, std::size_t depth, Sink& sink, ScanScratch& sc) const {
            if (static_cast<std::uint8_t>(edge.payload) == kEnd) { sink.key(std::string_view(path).substr(0, depth - 1)); return true; }
            if (b_.flag_of(edge)) {
                const std::uint64_t id = b_.id_of(edge);
                auto& m = sc.memo;
                const std::string_view rem = m.get(id, [&](std::string& pool) {
                    D_.append_remainder_via_path(id, pool, m.path_pos, m.path_labels, m.walk_pos);
                });
                sink.key(std::string_view(path), [&](std::string& o) { o.append(rem); });
                return true;
            }
            sink.key(std::string_view(path));
            return true;
        }
        Ref node_at(std::size_t blk, std::size_t nl, std::size_t live, std::size_t p, std::size_t rank) const {
            if (nl < live) return b_.record_at_coordinate(blk, nl, p, rank);
            std::size_t nb = blk + 1;
            while (nb < b_.block_count() && b_.block_live(nb) == 0) ++nb;
            if (nb >= b_.block_count()) throw std::logic_error("trie::dfuds DT scan: no node after the last block");
            return b_.record_at_coordinate(nb, 0, p, 0);
        }
        Ref pair_node(std::size_t blk, std::size_t local, std::size_t live, std::size_t pos, std::size_t rk, std::size_t t) const {
            return node_at(blk, local + 2 * t, live, pos + 2 * t, rk + t);
        }
        Ref pair_edge(std::size_t blk, std::size_t local, std::size_t pos, std::size_t rk, std::size_t t) const {
            return b_.record_at_coordinate(blk, local + 2 * t, pos + 2 * t, rk + t);
        }
        Ref edge_at(const Ref& run, std::size_t j, bool in_block) const {
            if (!in_block) return b_.record_at(run.slot.logical_pos() + j);
            return b_.record_at_coordinate(run.slot.block(), run.slot.local_slot() + j, run.slot.logical_pos() + j, run.open_rank + j);
        }
        Ref after_run(const Ref& run, std::size_t d, bool in_block) const {
            const std::size_t p = run.slot.logical_pos() + d + 1;
            if (!in_block) return b_.record_at(p);
            const std::size_t blk = run.slot.block();
            return node_at(blk, run.slot.local_slot() + d + 1, b_.block_live(blk), p, run.open_rank + d);
        }
        bool pop_child(std::vector<Frame>& fr, std::string& path, Ref& edge, std::size_t& depth) const {
            if (fr.empty()) return false;
            Frame& f = fr.back();
            const std::size_t j = f.next;
            edge = edge_at(f.run, j, f.in_block);
            depth = f.depth + 1;
            if (j == 0) fr.pop_back();
            else f.next = j - 1;
            path.resize(depth - 1);
            path.push_back(static_cast<char>(edge.payload));
            return true;
        }
        const DoubleTrie& D_;
        const FrontBits& b_;
        detail::FrontOps<FrontBits> ops_;
    };

    // ---- the front's shared walk (reference step) -------------------------------------------------------------------
    // One node of the walk with its incoming edge (the root: edge = rec, depth 0).
    struct SNode { FRef rec; FRef edge; std::size_t depth; };
    // Answer the keys of [lo, hi) that end at s or stop at a leaf; emit(child, glo, ghi) for every byte group whose
    // child exists. out[i] = 2 marks a key pending in the rear (pid, pd: the leaf's id and the rest's start).
    template <class Emit>
    void expand(const SNode& s, const std::string_view* keys, std::size_t lo, std::size_t hi, std::uint8_t* out,
                std::uint64_t* pid, std::size_t* pd, Emit&& emit) const {
        namespace sd = trie::sorted_descent;
        using St = detail::State;
        const FRef& cur = s.rec;
        const bool leaf = cur.bit != 1;
        const std::size_t head = sd::prefix_key_end(keys, lo, hi, s.depth);
        if (head > lo) {
            std::uint8_t k;
            if (s.depth == 0) k = leaf ? (key_count_ > 0) : (first_child_label(cur) == kEnd);
            else k = leaf ? !front_.flag_of(s.edge) : (first_child_label(cur) == kEnd);
            std::fill(out + lo, out + head, k);
        }
        if (head == hi) return;
        if (leaf) {
            if (s.depth > 0 && front_.flag_of(s.edge)) {
                const std::uint64_t id = front_.id_of(s.edge);
                for (std::size_t i = head; i < hi; ++i) { out[i] = 2; pid[i] = id; pd[i] = s.depth; }
            } else {
                std::fill(out + head, out + hi, std::uint8_t{0});
            }
            return;
        }
        const std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), live = front_.block_live(b);
        const std::size_t pos = cur.slot.logical_pos(), r = cur.open_rank;
        const std::size_t d0 = St::run_length(front_.block_words(b), local, live);
        const bool in_block = local + d0 < live;
        St::RunBuf rb;
        const std::uint8_t* labels;
        std::size_t d;
        if (in_block) { labels = front_.block_payload(b) + r; d = d0; }
        else { read_run_slow(cur, rb); labels = rb.label.data(); d = rb.d; }
        sd::for_each_byte_group(keys, head, hi, s.depth, [&](std::uint8_t c, std::size_t i, std::size_t j) {
            const std::size_t x = c == kEnd ? d : St::search(labels, d, c);   // NUL never selects END
            if (x == d) { std::fill(out + i, out + j, std::uint8_t{0}); return; }
            const FRef edge = in_block ? front_.record_at_coordinate(b, local + x, pos + x, r + x) : front_.record_at(pos + x);
            FRef child = edge;
            if (x + 1 == d) {
                const std::size_t nl = local + d + 1;
                child = (in_block && nl < live) ? front_.record_at_coordinate(b, nl, pos + d + 1, r + d) : front_.record_at(pos + d + 1);
            } else {
                child = child_of(edge);
            }
            emit(SNode{child, edge, s.depth + 1}, i, j);
        });
    }

    // ---- the front's DFUDS helpers (trie/dfuds/detail/core.hpp, on this front's PMA) ----------------------------------
    void read_run_slow(const FRef& cur, detail::State::RunBuf& rb) const {
        std::size_t b = cur.slot.block(), local = cur.slot.local_slot(), li = cur.open_rank;
        rb.d = 0;
        for (;;) {
            const std::size_t live = front_.block_live(b);
            const std::uint64_t* w = front_.block_words(b);
            const std::uint8_t* pl = front_.block_payload(b);
            while (local < live) {
                if (!bit(w, local)) return;
                if (rb.d >= 256) throw std::logic_error(std::string(kWho) + ": run longer than 256");
                rb.label[rb.d++] = pl[li++];
                ++local;
            }
            do { ++b; } while (b < front_.block_count() && front_.block_live(b) == 0);
            if (b >= front_.block_count()) throw std::logic_error(std::string(kWho) + ": run without its )");
            local = 0; li = 0;
        }
    }
    std::uint8_t first_child_label(const FRef& node) const {
        const std::size_t b = node.slot.block(), local = node.slot.local_slot(), live = front_.block_live(b);
        const std::size_t d = detail::State::run_length(front_.block_words(b), local, live);
        if (local + d < live) return front_.block_payload(b)[node.open_rank + d - 1];
        detail::State::RunBuf rb;
        read_run_slow(node, rb);
        return rb.label[rb.d - 1];
    }
    FRef child_of(const FRef& e) const {
        const std::size_t u = e.slot.logical_pos();
        const auto hit = pmabv::bp::find_close_hit_from(front_, u, e.slot);
        const std::size_t v = hit->pos, w = v + 1;
        const std::size_t next_local = hit->local + 1;
        if (next_local < front_.block_live(hit->block)) {
            if (hit->block == e.slot.block())
                return front_.record_at_coordinate(hit->block, next_local, w, e.open_rank + (v - u + 1) / 2);
            return front_.record_at_coordinate(hit->block, next_local, w);
        }
        return front_.record_at(w);
    }

    FrontBits front_;
    RearBp rear_;
    trie::dt::detail::IdTable ids_;
    trie::dt::detail::PackedFields hblock_;
    unsigned W_ = 1;
    std::size_t key_count_ = 0;
    TableGrowth growth_ = TableGrowth::terminal();
};

TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
