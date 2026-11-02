// Builds, batch insert, validation, logical extraction and space of trie::dfuds::Dictionary (trie/dfuds_dictionary.hpp):
// the END encoding (trie/dfuds/detail/encode.hpp, trie/dfuds/detail/batch.hpp), the NUL-free key domain, no flag
// channel and no root flag.
//
// BATCH INSERT, all positions in OLD coordinates, one pmabv batch insert:
//   1 order + domain  the batch is non-decreasing and NUL-free (parallel checks; nothing changes on a throw);
//   2 walk            dfuds::detail::insert_walk: per key the deepest existing node u and its EndKind;
//   3 classify        duplicates, present keys, END-ADD keys (one END child each), the root-leaf empty key, and the
//                     INSERTION GROUPS (maximal stretches of New / END-ADD keys with the same (u, j, depth)); a group
//                     under a key leaf (or under the root leaf when "" is stored after the batch) starts with an END
//                     string (END-PREPEND);
//   4-6 the CSR       dfuds::detail::build_gap_csr (labels only; END under every new string that the next new
//                     string of its child extends);
//   7 apply           pmabv's batch_insert_preprocessed, the key count.

#include "trie/dfuds_dictionary.hpp"
#include "trie/dfuds/detail/batch.hpp"
#include "trie/dfuds/detail/encode.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstring>
#include <exception>
#include <limits>
#include <string>
#include <vector>

#include <parlay/delayed.h>
#include <parlay/monoid.h>
#include <parlay/primitives.h>

#include "trie/detail/for_overwrite.hpp"

namespace trie::dfuds {
TRIE_DFUDS_ABI_BEGIN
namespace detail {

namespace fb = pmabv::flag_bits;
using trie::detail::ForOverwrite;
using RecordRef = Bits::RecordRef;
using Seq = Bits::Seq;
using CSR = Bits::CSR;

#ifdef NDEBUG
constexpr bool kValidateCsr = false;
#else
constexpr bool kValidateCsr = true;
#endif
constexpr std::size_t kWalkGrain = 512;
constexpr char kEndString[1] = {'\0'};
const std::string_view kEndView(kEndString, 1);

// Whether any key holds byte 0 (parallel; the key domain is NUL-free).
bool any_nul(const std::string_view* keys, std::size_t n) {
    return parlay::reduce(parlay::delayed_tabulate(n, [&](std::size_t i) -> std::size_t {
        return keys[i].empty() ? 0 : (std::memchr(keys[i].data(), 0, keys[i].size()) != nullptr);
    })) != 0;
}

State State::build_parallel(std::span<const std::string_view> sorted, std::size_t bc, std::size_t chunks) {
    if (any_nul(sorted.data(), sorted.size()))
        throw std::invalid_argument("trie::dfuds::Dictionary: a key contains byte 0 (END); the domain is NUL-free");
    Seq seq = encode_parallel<Seq>(sorted.size(), [&](std::size_t i) -> std::string_view { return sorted[i]; }, chunks);
    State s(Bits(pmabv::ParallelBuildTag{}, pmabv::PackedSeqTag{}, std::move(seq), bc));
    s.key_count_ = sorted.size();
    return s;
}

State State::build_sequential(std::span<const std::string_view> sorted, std::size_t bc) {
    for (std::size_t i = 1; i < sorted.size(); ++i)
        if (!(sorted[i - 1] < sorted[i]))
            throw std::invalid_argument("trie::dfuds::Dictionary::from_sorted: keys must be strictly increasing");
    if (any_nul(sorted.data(), sorted.size()))
        throw std::invalid_argument("trie::dfuds::Dictionary: a key contains byte 0 (END); the domain is NUL-free");
    std::vector<bool> bits{true};
    std::vector<std::uint8_t> labels{0};
    encode_runs(sorted.size(), [&](std::size_t i) -> std::string_view { return sorted[i]; }, bits, labels);
    State s(Bits(pmabv::CompactPayloadTag{}, bits, labels, bc));
    s.key_count_ = sorted.size();
    return s;
}

InsertPlan State::plan_insert(const std::string_view* keys, std::size_t n, InsertStats* stats) const {
    using Clock = std::chrono::steady_clock;
    InsertPlan plan;
    if (n == 0) return plan;
    auto t = Clock::now();
    auto lap = [&](double InsertStats::* f) {
        const auto now = Clock::now();
        if (stats) stats->*f += std::chrono::duration<double, std::nano>(now - t).count();
        t = now;
    };
    // ---- 1 order and domain -------------------------------------------------------------------------------------
    if (!trie::sorted_descent::is_sorted(keys, n))
        throw std::invalid_argument("trie::dfuds::Dictionary::insert_sorted: keys not sorted");
    if (any_nul(keys, n))
        throw std::invalid_argument("trie::dfuds::Dictionary::insert_sorted: a key contains byte 0 (END)");
    lap(&InsertStats::ns_check);

    // ---- 2 walk -------------------------------------------------------------------------------------------------
    ForOverwrite<FPre> fp(n);
    ForOverwrite<Reach> rc(n);
    insert_walk(*this, keys, n, kWalkGrain, fp, [&](std::size_t i, const Reach& r) { rc[i] = r; });
    lap(&InsertStats::ns_walk);

    // ---- 3 classify ---------------------------------------------------------------------------------------------
    const auto dup = parlay::tabulate(n, [&](std::size_t i) -> std::uint8_t { return i > 0 && keys[i] == keys[i - 1]; });
    auto kind_count = [&](EndKind k) {
        return parlay::reduce(parlay::delayed_tabulate(n, [&](std::size_t i) -> std::size_t { return !dup[i] && rc[i].kind == k; }));
    };
    const std::size_t root_leaf = kind_count(EndKind::RootLeaf);   // 0 or 1 (the empty key)
    const std::size_t end_adds = kind_count(EndKind::EndAdd);
    const auto newi = parlay::pack_index(parlay::tabulate(n, [&](std::size_t i) {
        return !dup[i] && (rc[i].kind == EndKind::New || rc[i].kind == EndKind::EndAdd); }));
    const std::size_t nn = newi.size();
    const auto gstart = parlay::pack_index(parlay::tabulate(nn, [&](std::size_t x) {
        if (x == 0) return true;
        const std::size_t a = newi[x - 1], b = newi[x];
        return fp[a].s_u != fp[b].s_u || fp[a].j != fp[b].j || rc[a].m != rc[b].m;
    }));
    const std::size_t G = gstart.size();
    // END-PREPEND: a group under a leaf u keeps u's key -- a non-root leaf always ends a key; the root leaf iff the
    // empty key is stored after the batch.
    const bool root_key_after = key_count_ > 0 || root_leaf > 0;
    const auto prepend = parlay::tabulate(G, [&](std::size_t g) -> std::uint8_t {
        const std::size_t i = newi[gstart[g]];
        return fp[i].aux == kLeaf && (rc[i].m > 0 || root_key_after);
    });
    plan.added = nn + root_leaf;
    lap(&InsertStats::ns_classify);

    // ---- 4-6 positions, order, encoding -------------------------------------------------------------------------
    GapCsrStats cs;
    if (G) {
        plan.csr = build_gap_csr<CSR>(
            G, [&](std::size_t g) { const FPre& f = fp[newi[gstart[g]]]; return f.s_u + f.j; },
            [&](std::size_t g) { return sub_pos_of(bits_, fp[newi[gstart[g]]]); },
            [&](std::size_t g, auto&& f) {
                if (prepend[g]) f(kEndView);
                const std::size_t hi = g + 1 < G ? gstart[g + 1] : nn;
                for (std::size_t x = gstart[g]; x < hi; ++x) {
                    const std::size_t i = newi[x];
                    f(rc[i].kind == EndKind::EndAdd ? kEndView : keys[i].substr(rc[i].m));
                }
            }, bits_.size(), "trie::dfuds::Dictionary::insert_sorted", stats ? &cs : nullptr);
    }
    if (stats) {
        stats->ns_place += cs.ns_place;
        stats->ns_order += cs.ns_order;
        stats->ns_csr_index += cs.ns_csr_index;
        stats->ns_csr_init += cs.ns_csr_init;
        stats->ns_encode += cs.ns_encode;
        stats->keys = n; stats->added = plan.added;
        stats->end_adds = end_adds;
        stats->end_prepends = parlay::reduce(parlay::delayed_tabulate(G, [&](std::size_t g) -> std::size_t { return prepend[g]; }));
        stats->root_count = root_leaf;
        stats->groups = cs.groups; stats->children = cs.children; stats->csr_entries = cs.entries;
        stats->new_bits = cs.bits; stats->new_nodes = cs.labels;
        stats->dups = parlay::reduce(parlay::delayed_tabulate(n, [&](std::size_t i) { return std::size_t{dup[i]}; }));
        stats->present = n - stats->dups - plan.added;
        stats->blocks_before = bits_.block_count();
    }
    return plan;
}

void State::apply(const InsertPlan& plan, InsertStats* stats) {
    using Clock = std::chrono::steady_clock;
    const auto t1 = Clock::now();
    if (!plan.csr.indexes.empty()) bits_.batch_insert_preprocessed(plan.csr, nullptr, kValidateCsr);
    key_count_ += plan.added;
    if (stats) stats->ns_merge += std::chrono::duration<double, std::nano>(Clock::now() - t1).count();
}

// Structural invariants: the PMA's own, 2n bits for n ones, the dummy entry, a balanced sequence, and the END
// encoding on the canonical content (logical()): every kEnd entry but the dummy is the LAST "(" of its run (the
// ascending-first child), has a sibling (the previous bit is a "(") and leads to a leaf (the bit after the run's ")"
// is a ")"); the stored keys are the non-root leaves (a root leaf: 0 or 1 keys).
void State::validate() const {
    bits_.check_invariants_parallel();
    const std::size_t ones = bits_.label_count();
    if (bits_.size() != 2 * ones) throw std::logic_error("trie::dfuds validate: bits != 2 x ones");
    if (bits_.block_count() == 0 || bits_.size() < 2) throw std::logic_error("trie::dfuds validate: empty sequence");
    const RecordRef r0 = bits_.record_at(0);
    if (r0.bit != 1 || r0.payload != 0) throw std::logic_error("trie::dfuds validate: bad dummy");
    const Logical L = logical();
    long long e = 0;
    for (std::size_t i = 0; i < L.live; ++i) {
        e += L.bit(i) ? 1 : -1;
        if (i + 1 < L.live && e < 1) throw std::logic_error("trie::dfuds validate: unbalanced (excess reaches 0 before the end)");
    }
    if (e != 0) throw std::logic_error("trie::dfuds validate: unbalanced end");
    std::size_t rank = 0, leaves = 0;
    for (std::size_t p = 0; p < L.live; ++p) {
        if (!L.bit(p)) {
            if (p >= 2 && !L.bit(p - 1)) ++leaves;                              // a run ")" right after a run's ")"
            continue;
        }
        const std::uint8_t lab = L.labels[rank++];
        if (p == 0 || lab != kEnd) continue;
        if (p + 2 >= L.live || L.bit(p + 1) || L.bit(p + 2) || !L.bit(p - 1))
            throw std::logic_error("trie::dfuds validate: an END entry that is not a sibling-bearing first-child leaf");
    }
    if (rank != ones) throw std::logic_error("trie::dfuds validate: label count");
    const bool root_leaf = !L.bit(1);
    if (root_leaf ? key_count_ > 1 : key_count_ != leaves)
        throw std::logic_error("trie::dfuds validate: key count != stored leaves");
}

Logical State::logical() const {
    const Bits& b = bits_;
    const std::size_t B = b.block_count();
    std::vector<std::size_t> bit0(B + 1, 0), one0(B + 1, 0);
    for (std::size_t x = 0; x < B; ++x) {
        bit0[x + 1] = bit0[x] + b.block_live(x);
        one0[x + 1] = one0[x] + b.block_label_count(x);
    }
    Logical L;
    L.live = bit0[B];
    L.words.assign(fb::words_for(L.live), 0ULL);
    L.labels.assign(one0[B], 0);
    L.key_count = key_count_;
    parlay::parallel_for(0, B, [&](std::size_t x) {
        const std::size_t live = b.block_live(x), ones = b.block_label_count(x);
        if (live) fb::or_range(L.words.data(), bit0[x], b.block_words(x), 0, live, fb::Mode::BoundaryAtomic);
        const std::uint8_t* pl = b.block_payload(x);
        for (std::size_t i = 0; i < ones; ++i) L.labels[one0[x] + i] = pl[i];
    }, 1);
    return L;
}

}  // namespace detail

// ---- the public class -------------------------------------------------------------------------------------------

Dictionary::Dictionary(std::size_t block_capacity) {
    s_ = std::make_unique<State>(State::build_parallel({}, block_capacity, 0));
    s_->bits_.set_leaf_update_policy(default_leaf_update_policy());
}

Dictionary::Dictionary(std::unique_ptr<State> s) : s_(std::move(s)) {
    s_->bits_.set_leaf_update_policy(default_leaf_update_policy());
}

Dictionary::~Dictionary() {}

std::unique_ptr<Dictionary> Dictionary::from_sorted(std::span<const std::string_view> keys, std::size_t block_capacity) {
    auto s = std::make_unique<State>(State::build_sequential(keys, block_capacity));
    if constexpr (transactional) s->validate();
    return std::unique_ptr<Dictionary>(new Dictionary(std::move(s)));
}

std::unique_ptr<Dictionary> Dictionary::from_sorted_parallel(std::span<const std::string_view> keys,
                                                             std::size_t block_capacity, std::size_t chunks) {
    auto s = std::make_unique<State>(State::build_parallel(keys, block_capacity, chunks));
    if constexpr (transactional) s->validate();
    return std::unique_ptr<Dictionary>(new Dictionary(std::move(s)));
}

std::unique_ptr<Dictionary> Dictionary::from_sorted_parallel(const std::vector<std::string>& keys,
                                                             std::size_t block_capacity, std::size_t chunks) {
    std::vector<std::string_view> v(keys.size());
    parlay::parallel_for(0, keys.size(), [&](std::size_t i) { v[i] = keys[i]; }, 4096);
    return from_sorted_parallel(std::span<const std::string_view>(v), block_capacity, chunks);
}

std::size_t Dictionary::insert_sorted(const std::string_view* keys, std::size_t n, InsertStats* stats) {
    if (stats) *stats = InsertStats{};
    if (n == 0) return 0;
    detail::InsertPlan plan = s_->plan_insert(keys, n, stats);   // throws: unchanged
    if (plan.added != 0) {
        if constexpr (transactional) {
            using Clock = std::chrono::steady_clock;
            const auto t0 = Clock::now();
            auto cand = std::make_unique<State>(*s_);
            const auto t1 = Clock::now();
            cand->apply(plan, stats);
            const auto t2 = Clock::now();
            cand->validate();
            s_.swap(cand);
            if (stats) stats->ns_txn += std::chrono::duration<double, std::nano>((t1 - t0) + (Clock::now() - t2)).count();
        } else {
            apply_direct(plan, stats);
        }
    }
    if (stats) stats->blocks_after = s_->bits_.block_count();
    return plan.added;
}

void Dictionary::apply_direct(const detail::InsertPlan& plan, InsertStats* stats) noexcept { s_->apply(plan, stats); }

std::size_t Dictionary::end_leaves() const {
    const detail::Bits& b = s_->bits_;
    const std::size_t zeros = parlay::reduce(parlay::delayed_tabulate(b.block_count(), [&](std::size_t x) -> std::size_t {
        const std::uint8_t* pl = b.block_payload(x);
        const std::size_t ones = b.block_label_count(x);
        std::size_t c = 0;
        for (std::size_t i = 0; i < ones; ++i) c += pl[i] == detail::kEnd;
        return c;
    }));
    return zeros - 1;   // the dummy entry carries label 0 too
}

std::size_t Dictionary::byte_size() const noexcept {
    return sizeof(Dictionary) + sizeof(State) + s_->bits_.byte_size() - sizeof(detail::Bits);
}

SpaceReport Dictionary::space() const {
    const detail::Bits& b = s_->bits_;
    SpaceReport r;
    r.total = byte_size();
    auto& v = r.rows;
    v.push_back({"pma", "total", b.byte_size(), "bytes", true, "BitPMAImpl::byte_size(): the backend's allocations"});
    std::size_t listed = 0;
    auto cat = [&](const char* name, std::size_t x, std::string basis) {
        v.push_back({std::string("pma.") + name, "pma", x, "bytes", true, std::move(basis)});
        listed += x;
    };
    cat("topology", b.word_capacity_bytes(), "allocated capacity: sequence words incl. PMA gaps and slab slack");
    cat("labels", b.label_bytes(), "allocated capacity: per-block label vectors + descriptors");
    static_assert(!detail::Bits::kHasFlags, "no terminal flag channel");
    cat("flags", 0, "none: the block type stores no flag channel (block::WithPayload Flags = false, detail::NoFlagStore)");
    static_assert(detail::Bits::kSeparateIndexes, "the DFUDS backend keeps its indexes in separate slabs");
    cat("rank_index", b.rank_index_bytes(), "allocated capacity: rank-sample slab");
    cat("excess_index", b.excess_index_bytes(), "allocated capacity: excess-record slab");
    const std::size_t aug = b.aug_bytes();
    cat("summary_counts", b.tree_bytes() - aug, "allocated capacity: summary-tree live/one counts");
    cat("minmax_aug", aug, "allocated capacity: BpAug min-excess array");
    const long long residual = static_cast<long long>(b.byte_size()) - static_cast<long long>(listed);
    v.push_back({"pma.other", "pma", residual >= 0 ? static_cast<std::size_t>(residual) : 0, "bytes", true,
                 residual >= 0 ? std::string("residual: byte_size() minus the rows above")
                               : "residual NEGATIVE (" + std::to_string(residual) + " bytes): rows overlap"});
    v.push_back({"pma.other.object", "pma.other", sizeof(detail::Bits), "bytes", true, "sizeof(BitPMAImpl), counted by byte_size()"});
    v.push_back({"pma.other.block_live_counts", "pma.other", b.live_count_bytes(), "bytes", true,
                 "allocated capacity: per-block live counts"});
    v.push_back({"pma.other.scratch", "pma.other", b.scratch_bytes(), "bytes", true,
                 "allocated capacity: retained redistribute scratch bitsets"});
    v.push_back({"object", "total", r.total - b.byte_size(), "bytes", true,
                 "sizeof(Dictionary) + sizeof(State) - sizeof(BitPMAImpl): key count, pointer, mutex, padding"});
    v.push_back({"pma.ones", "pma", b.label_count(), "count", false, "diagnostic: one bits (== label entries == nodes incl. END leaves)"});
    v.push_back({"pma.end_leaves", "pma", end_leaves(), "count", false, "diagnostic: END leaves (label-0 entries but the dummy)"});
    v.push_back({"pma.blocks", "pma", b.block_count(), "count", false, "diagnostic: PMA blocks"});
    v.push_back({"pma.topology.logical_bits", "pma.topology", b.size(), "bits", false, "diagnostic: live sequence bits"});
    v.push_back({"pma.labels.slack", "pma.labels", b.label_slack_bytes(), "bytes", false,
                 "subset of labels: capacity beyond one byte per entry"});
    v.push_back({"pma.labels.compact_blocks", "pma.labels", b.label_compact_blocks(), "count", false,
                 "diagnostic: blocks whose label vector has capacity == size (reserve consumed)"});
    v.push_back({"pma.labels.reserved_blocks", "pma.labels", b.label_reserved_blocks(), "count", false,
                 "diagnostic: blocks whose label vector carries the full reserve (capacity >= size + slack)"});
    v.push_back({"pma.flags.storage_type", "pma.flags", 0, "count", false,
                 "diagnostic: 0 = the block type has no flag member (sizeof(NoFlagStore) is empty)"});
    v.push_back({"keys", "total", s_->key_count_, "count", false, "diagnostic: stored keys"});
    return r;
}

TRIE_DFUDS_ABI_END
}  // namespace trie::dfuds
