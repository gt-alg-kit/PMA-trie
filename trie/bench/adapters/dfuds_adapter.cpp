// The DFUDS structures behind the unified Index interface of common/dict_index.hpp
// (docs/dfuds_library_integration_progress_2026-10-16.md, sections 13 and 17):
//   --impl dfuds      trie::dfuds::Dictionary (library trie_dfuds);
//   --impl dfuds-dt   trie::dfuds::DoubleTrie, the DFUDS-front double trie with
//                     the BP rear (header-only; trie/dfuds_double_trie.hpp).
// Compiled into each driver that registers them, linked against the DFUDS
// library variant whose pmabv definitions match that driver.
//   build             the parallel bulk build of the sorted stored set
//                     (from_sorted_parallel), the counterpart of BP's and the BP
//                     double trie's --setup-build parallel;
//   direct queries    one Querier per worker, one virtual call into the inline
//                     descent (as the BP queriers);
//   tree queries      the shared traversal (contains_sorted_parallel, the
//                     driver's grain; the double trie walks the reversed rests
//                     down its rear in one batch);
//   batch insert      insert_sorted of the batch the driver sorted inside its
//                     timer (the BP boundary: sort + batch insert);
//   ordered scans     one Scanner per worker around the structure's
//                     scan_from, with the structure's ScanScratch (frame
//                     stack, path buffer) kept across scans
//                     (trie/dfuds/detail/scan.hpp);
//   key domain        NUL-free byte strings: byte 0 is the END label of both structures; a key holding it is
//                     reported by key_domain_violation (the driver refuses such a dataset; nothing is dropped);
//   space             the structure's reconciled breakdown: the top-level rows
//                     sum to byte_size(), the nested rows are informational.
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

#include "common/dict_index.hpp"
#include "trie/dfuds_dictionary.hpp"
#include "trie/dfuds_double_trie.hpp"

namespace dictcmp {
namespace {

// The library's space rows as Index components (top level = in the total).
std::vector<SpaceComponent> components_of(const trie::dfuds::SpaceReport& rep) {
    std::vector<SpaceComponent> v;
    for (const auto& r : rep.rows) {
        const bool top = r.in_total && r.parent == "total";
        v.push_back({(r.unit == "bytes" ? "" : r.unit + ":") + r.component, r.value, top});
    }
    return v;
}

using DfudsDict = trie::dfuds::Dictionary;

// One side's leaf-update counters of the last batch (pmabv LastBatch), prefixed (dfuds_ / front_ / rear_). `seq` counts
// that side's pmabv batch calls: a reader comparing two records tells a new batch from a stale one (a call that never
// reached the side's PMA leaves the previous record in place).
template <class LB> std::string leaf_counts(const char* p, const LB& lb) {
    char buf[512];
    std::snprintf(buf, sizeof buf,
                  "%sleaf_shifts=%zu %sleaf_merges=%zu %sshift_rej_policy=%zu %sshift_rej_runs=%zu %sshift_rej_topology=%zu "
                  "%sshift_rej_labels=%zu %sshift_rej_markers=%zu %sshift_rej_ids=%zu %sshift_rej_movement=%zu %sshift_bits=%zu "
                  "%sshift_entries=%zu %sshift_ids=%zu %slocal_path=%d %srebuild_tasks=%zu %srebuild_blocks=%zu %sseq=%llu",
                  p, lb.leaf_shifts, p, lb.leaf_merges, p, lb.shift_rej_policy, p, lb.shift_rej_runs, p, lb.shift_rej_topology,
                  p, lb.shift_rej_labels, p, lb.shift_rej_flags, p, lb.shift_rej_ids, p, lb.shift_rej_movement, p, lb.shift_bits,
                  p, lb.shift_entries, p, lb.shift_ids, p, int(lb.path == 1), p, lb.rebuild_tasks, p, lb.rebuild_blocks,
                  p, static_cast<unsigned long long>(lb.seq));
    return buf;
}
template <class Pol> std::string policy_text(const Pol& p) {
    char buf[128];
    std::snprintf(buf, sizeof buf, "leaf_policy=%s leaf_max_runs=%zu leaf_move_factor=%g",
                  p.mode == decltype(p.mode)::Adaptive ? "adaptive" : "merge_only", p.max_runs, p.movement_factor);
    return buf;
}

// The direct ordered scan of either structure: its scan_from with this worker's scratch.
template <class D>
class DfudsScanner final : public Scanner {
public:
    explicit DfudsScanner(const D* d) : d_(d) {}
    std::size_t scan_from(std::string_view lo, std::size_t k, std::string& out) override { return d_->scan_from(lo, k, out, sc_); }
private:
    const D* d_;
    typename D::ScanScratch sc_;
};
// The dictionary's structure diagnostic (no flag channel): DFUDS_CAPACITY (flag terms 0: the block type stores none)
// and DFUDS_SHAPE from logical(): nodes, degrees, leaves, END leaves (label 0 on an entry other than the dummy), the
// label histogram; with `dump` the whole encoding (DFUDS_LOGICAL).
std::vector<std::string> dfuds_structure_diag(const trie::dfuds::Dictionary& d, bool dump) {
    const auto& b = d.bits();
    const auto L = d.logical();
    std::vector<std::string> out;
    char buf[1536];
    const std::size_t live_word_bytes = (L.live + 63) / 64 * 8;
    std::snprintf(buf, sizeof buf,
                  "DFUDS_CAPACITY byte_size=%zu pma_byte_size=%zu blocks=%zu block_capacity_bits=%zu logical_bits=%zu "
                  "live_word_bytes=%zu word_capacity_bytes=%zu topology_slack_bytes=%zu label_entries=%zu "
                  "label_capacity_bytes=%zu label_descriptor_bytes=%zu label_bytes=%zu label_slack_bytes=%zu "
                  "label_spare_beyond_reserve=%zu label_growth_slack=%zu label_compact_blocks=%zu label_reserved_blocks=%zu "
                  "label_utilization=%.6f flag_live_bits=0 flag_capacity_bytes=0 flag_descriptor_bytes=0 flag_bytes=0 "
                  "flag_ones=0 rank_index_record_bytes=%zu rank_index_bytes=%zu excess_index_record_bytes=%zu "
                  "excess_index_bytes=%zu tree_bytes=%zu summary_count_bytes=%zu minmax_aug_bytes=%zu live_count_bytes=%zu "
                  "scratch_bytes=%zu backend_object_bytes=%zu key_end=end_leaf",
                  d.byte_size(), b.byte_size(), b.block_count(), b.block_capacity(), b.size(), live_word_bytes,
                  b.word_capacity_bytes(), b.word_capacity_bytes() - std::min(b.word_capacity_bytes(), live_word_bytes),
                  b.label_count(), b.label_capacity_bytes(), b.label_descriptor_bytes(), b.label_bytes(), b.label_slack_bytes(),
                  b.label_spare_beyond_reserve(), b.label_growth_slack(), b.label_compact_blocks(), b.label_reserved_blocks(),
                  b.label_utilization(), b.rank_index_record_bytes(), b.rank_index_bytes(), b.excess_index_record_bytes(),
                  b.excess_index_bytes(), b.tree_bytes(), b.tree_bytes() - b.aug_bytes(), b.aug_bytes(), b.live_count_bytes(),
                  b.scratch_bytes(), sizeof(std::remove_cvref_t<decltype(b)>));
    out.emplace_back(buf);
    std::size_t ones = 0, zeros = 0, deg_hist[4] = {0, 0, 0, 0}, max_deg = 0, root_deg = 0, deg = 0, end_leaves = 0;
    std::size_t leaves = 0, order_violations = 0;
    std::size_t label_hist[256] = {0};
    int prev_label = -1;
    for (std::size_t i = 0; i < L.live; ++i) {
        if (L.bit(i)) {
            const std::size_t e = ones++;
            const unsigned lab = L.labels[e];
            ++label_hist[lab];
            if (e > 0) {
                if (lab == 0) ++end_leaves;
                if (prev_label >= 0 && static_cast<int>(lab) >= prev_label) ++order_violations;
                prev_label = static_cast<int>(lab);
                ++deg;
            }
        } else {
            const std::size_t c = zeros++;
            if (c == 0) root_deg = deg;
            else if (deg == 0) ++leaves;
            ++deg_hist[std::min<std::size_t>(deg, 3)];
            max_deg = std::max(max_deg, deg);
            deg = 0;
            prev_label = -1;
        }
    }
    std::string s;
    std::snprintf(buf, sizeof buf,
                  "DFUDS_SHAPE key_count=%zu node_count=%zu logical_bits=%zu ones=%zu zeros=%zu nodes=%zu edges=%zu "
                  "root_degree=%zu deg0=%zu deg1=%zu deg2=%zu deg_ge3=%zu max_degree=%zu nonroot_leaves=%zu end_leaves=%zu "
                  "run_label_order_violations=%zu key_end=end_leaf",
                  d.key_count(), d.node_count(), d.logical_bits(), ones, zeros, zeros, ones ? ones - 1 : 0, root_deg,
                  deg_hist[0], deg_hist[1], deg_hist[2], deg_hist[3], max_deg, leaves, end_leaves, order_violations);
    s = buf;
    std::size_t distinct = 0;
    for (unsigned v = 0; v < 256; ++v)
        if (label_hist[v]) {
            ++distinct;
            std::snprintf(buf, sizeof buf, " label_%02x=%zu", v, label_hist[v]);
            s += buf;
        }
    s += " labels_distinct=" + std::to_string(distinct);
    out.push_back(std::move(s));
    if (dump) {
        std::string bits, labels;
        for (std::size_t i = 0; i < L.live; ++i) bits += L.bit(i) ? '(' : ')';
        for (std::size_t e = 0; e < L.labels.size(); ++e) {
            std::snprintf(buf, sizeof buf, "%s%02x", e ? "," : "", unsigned(L.labels[e]));
            labels += buf;
        }
        out.push_back("DFUDS_LOGICAL bits=" + bits + " labels=" + labels + " key_end=end_leaf");
    }
    return out;
}

class DfudsIndex final : public Index {
public:
    explicit DfudsIndex(std::size_t bc) : bc_(bc) {}
    DfudsIndex(std::size_t bc, std::unique_ptr<DfudsDict> d) : bc_(bc), d_(std::move(d)) { apply_leaf_policy(); }
    const char* label() const override { return "DFUDS-Dictionary"; }
    void set_batch_grain(std::size_t g) override { grain_ = g; }
    void build(const BuildInput& in) override {
        d_ = DfudsDict::from_sorted_parallel(*in.sorted, bc_);
        apply_leaf_policy();
        if (d_->key_count() != in.sorted->size()) throw std::runtime_error("DFUDS: key_count != N after build");
    }
    bool contains(std::string_view key) const override { return d_->contains(key); }
    class Q final : public Querier {
    public:
        Q(const DfudsDict* d, std::size_t grain) : d_(d), grain_(grain) {}
        bool contains(std::string_view key) const override { return d_->contains(key); }
        bool contains_sorted_parallel(const std::string_view* keys, std::size_t n, std::uint8_t* out) const override {
            d_->contains_sorted_parallel(keys, n, out, grain_);
            return true;
        }
    private:
        const DfudsDict* d_;
        std::size_t grain_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(d_.get(), grain_); }
    std::unique_ptr<Scanner> make_scanner() const override { return std::make_unique<DfudsScanner<DfudsDict>>(d_.get()); }
    // Labels are unsigned bytes compared as unsigned (0x80..0xff ordinary), the empty key is the root leaf of {""} or
    // the root's END child, a key is a path of the trie ending at a leaf (no length field); byte 0 is the END label,
    // so stored keys must be NUL-free (queries may hold it: they are absent).
    const char* key_domain_violation(std::string_view k) const override {
        return std::memchr(k.data(), 0, k.size()) ? "byte 0 (NUL) is reserved for END leaves" : nullptr;
    }
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        const DfudsDict* d = d_.get();
        return timed_batch(q, n, [d](const std::string& k) { return d->contains(k); });
    }
    bool supports_batch_insert() const override { return true; }
    // No native single-key insert: insert_sorted(&key, 1), a batch of one (its planning included).
    bool insert_one(std::string_view key) override { return insert_sorted_batch(&key, 1) == 1; }
#ifdef BENCH_INSERT_PHASES
    // The diagnostic twin (bench_dictcmp_census): the library's InsertStats of the last batch -- its phase clocks
    // (order check, walk, classify, place, order, encode, flips, pmabv merge) and counts -- for Index::last_batch_phases.
    // A non-null stats pointer adds the library's clock laps and one parallel count of the batch's duplicates to the
    // call; the timing binary passes none.
    std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) override { return d_->insert_sorted(keys, n, &stats_); }
    std::string last_batch_phases() const override {
        const trie::dfuds::InsertStats& s = stats_;
        char buf[760];
        std::snprintf(buf, sizeof buf,
                      "dfuds_check_ms=%.3f dfuds_walk_ms=%.3f dfuds_classify_ms=%.3f dfuds_place_ms=%.3f dfuds_order_ms=%.3f "
                      "dfuds_csr_index_ms=%.3f dfuds_csr_init_ms=%.3f "
                      "dfuds_encode_ms=%.3f dfuds_merge_ms=%.3f dfuds_txn_ms=%.3f dfuds_keys=%zu dfuds_added=%zu "
                      "dfuds_dups=%zu dfuds_present=%zu dfuds_end_adds=%zu dfuds_end_prepends=%zu dfuds_root_count=%zu "
                      "dfuds_groups=%zu dfuds_children=%zu dfuds_csr_entries=%zu "
                      "dfuds_new_nodes=%zu dfuds_new_bits=%zu dfuds_blocks_before=%zu dfuds_blocks_after=%zu",
                      s.ns_check / 1e6, s.ns_walk / 1e6, s.ns_classify / 1e6, s.ns_place / 1e6, s.ns_order / 1e6,
                      s.ns_csr_index / 1e6, s.ns_csr_init / 1e6, s.ns_encode / 1e6,
                      s.ns_merge / 1e6, s.ns_txn / 1e6, s.keys, s.added, s.dups, s.present, s.end_adds, s.end_prepends,
                      s.root_count, s.groups, s.children, s.csr_entries, s.new_nodes, s.new_bits, s.blocks_before, s.blocks_after);
        return std::string(buf) + " " + leaf_counts("dfuds_", d_->bits().last_batch());
    }
#else
    std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) override { return d_->insert_sorted(keys, n); }
#endif
    const char* build_kind() const override { return "bulk-parallel"; }
    const char* update_kind() const override { return DfudsDict::transactional ? "transactional" : "in_place"; }
    std::vector<SpaceComponent> space() const override { return components_of(d_->space()); }
    const char* memory_basis() const override {
        return "allocated capacity: trie::dfuds::Dictionary::byte_size() = BitPMAImpl::byte_size() (topology words, "
               "labels, no flag channel, rank/excess slabs, summary counts, BpAug, live counts, scratch) + the dictionary objects";
    }
    std::string info() const override {
        return "bc=" + std::to_string(bc_) + " build=from_sorted_parallel bits=" + std::to_string(d_->logical_bits()) +
               " nodes=" + std::to_string(d_->node_count()) + " key_count=" + std::to_string(d_->key_count()) +
               " chain_key=ref chain_walk=run" +   // fixed: the reference descent / the run walk (trie/dfuds/detail/core.hpp)
               " transactional=" + std::to_string(int(DfudsDict::transactional)) + " " + policy_text(d_->leaf_update_policy()) +
               " " + leaf_counts("dfuds_", d_->bits().last_batch()) + " key_end=" + DfudsDict::key_end_policy;
    }
    std::size_t current_key_count() const override { return d_->key_count(); }
    std::vector<std::string> structure_diag(bool dump) const override { return dfuds_structure_diag(*d_, dump); }
    std::string check_structure() const override {
        try {
            d_->check_invariants();
            return "ok";
        } catch (const std::exception& e) {
            return std::string("fail: ") + e.what();
        }
    }
private:
    void apply_leaf_policy() {
        const auto& r = leaf_policy_request();
        DfudsDict::LeafUpdatePolicy p;
        p.mode = r.adaptive ? DfudsDict::LeafUpdateMode::Adaptive : DfudsDict::LeafUpdateMode::MergeOnly;
        p.max_runs = r.max_runs; p.movement_factor = r.movement_factor;
        d_->set_leaf_update_policy(p);
    }
    std::size_t bc_;
    std::size_t grain_ = 512;
    std::unique_ptr<DfudsDict> d_;
#ifdef BENCH_INSERT_PHASES
    trie::dfuds::InsertStats stats_;
#endif
};

using DfudsDt = trie::dfuds::DoubleTrie;
// The DTs' requested global-table growth (common/dict_index.hpp), passed into every build.
DfudsDt::TableGrowth requested_growth() { return DfudsDt::TableGrowth::parse(dt_table_growth_request()); }

class DfudsDtIndex final : public Index {
public:
    explicit DfudsDtIndex(std::size_t bc) : bc_(bc) {}
    DfudsDtIndex(std::size_t bc, std::unique_ptr<DfudsDt> d) : bc_(bc), d_(std::move(d)) { apply_leaf_policy(); }
    const char* label() const override { return "DFUDS-DoubleTrie(BP rear)"; }
    void set_batch_grain(std::size_t g) override { grain_ = g; }
    void build(const BuildInput& in) override {
        d_ = DfudsDt::from_sorted_parallel(*in.sorted, bc_, 0, false, requested_growth());
        apply_leaf_policy();
        if (d_->key_count() != in.sorted->size()) throw std::runtime_error("DFUDS-DT: key_count != N after build");
    }
    bool contains(std::string_view key) const override { return d_->contains(key); }
    class Q final : public Querier {
    public:
        Q(const DfudsDt* d, std::size_t grain) : d_(d), grain_(grain) {}
        bool contains(std::string_view key) const override { return d_->contains(key); }
        bool contains_sorted_parallel(const std::string_view* keys, std::size_t n, std::uint8_t* out) const override {
            d_->contains_sorted_parallel(keys, n, out, grain_);
            return true;
        }
    private:
        const DfudsDt* d_;
        std::size_t grain_;
    };
    std::unique_ptr<Querier> make_querier() const override { return std::make_unique<Q>(d_.get(), grain_); }
    std::unique_ptr<Scanner> make_scanner() const override { return std::make_unique<DfudsScanner<DfudsDt>>(d_.get()); }
    // The front's cut prefixes and the rear's reversed remainders are byte strings over unsigned labels (0x80..0xff
    // ordinary); byte 0 is the END label of both tries, so stored keys must be NUL-free (queries may hold it: they are
    // absent). The parallel build's remainder ranking compares equal 4-byte sort heads on the bytes themselves
    // (dt::detail::RemLess); no length field anywhere.
    const char* key_domain_violation(std::string_view k) const override {
        return std::memchr(k.data(), 0, k.size()) ? "byte 0 (NUL) is reserved for END leaves" : nullptr;
    }
    BatchResult run_batch(const std::vector<std::string>& q, std::size_t n) const override {
        const DfudsDt* d = d_.get();
        return timed_batch(q, n, [d](const std::string& k) { return d->contains(k); });
    }
    bool supports_batch_insert() const override { return true; }
    std::size_t insert_sorted_batch(const std::string_view* keys, std::size_t n) override { return d_->insert_sorted(keys, n); }
    // No native single-key insert: insert_sorted(&key, 1), a batch of one (its planning included).
    bool insert_one(std::string_view key) override { return d_->insert_sorted(&key, 1) == 1; }
    const char* build_kind() const override { return "bulk-parallel"; }
    const char* update_kind() const override { return DfudsDt::transactional ? "transactional" : "in_place"; }
    // Each side's ID storage (O(blocks); diagnostics only): allocated bytes, live and spare fields.
    std::vector<std::pair<std::string, std::size_t>> counters() const override {
        return {{"front_id_capacity_bytes", d_->front().id_capacity_bytes()}, {"front_id_live_fields", d_->front().id_live_fields()},
                {"front_id_spare_fields", d_->front().id_spare_fields()}, {"rear_id_capacity_bytes", d_->rear().id_capacity_bytes()},
                {"rear_id_live_fields", d_->rear().id_live_fields()}, {"rear_id_spare_fields", d_->rear().id_spare_fields()}};
    }
    // per batch (the diagnostic twin's PHASE_RECORD): each side's leaf-update counters of the last batch
    std::string last_batch_phases() const override {
        return leaf_counts("front_", d_->front().last_batch()) + " " + leaf_counts("rear_", d_->rear().last_batch());
    }
    std::vector<SpaceComponent> space() const override { return components_of(d_->space()); }
    const char* memory_basis() const override {
        return "allocated capacity: trie::dfuds::DoubleTrie::byte_size() = the DFUDS front's BitPMAImpl::byte_size() + "
               "the label-only BP rear's byte_size() (leaf-rank ids with their counts and samples) + handles + owner counts + "
               "free ids + overflow estimate + objects";
    }
    std::string info() const override {
        return "bc=" + std::to_string(bc_) + " build=from_sorted_parallel key_count=" + std::to_string(d_->key_count()) +
               " remainders=" + std::to_string(d_->distinct_remainders()) + " id_width=" + std::to_string(d_->id_width()) +
               " front_bits=" + std::to_string(d_->front().size()) + " rear_bits=" + std::to_string(d_->rear().size()) +
               " transactional=" + std::to_string(int(DfudsDt::transactional)) + " " + policy_text(d_->leaf_update_policy()) +
               " table_growth=" + d_->table_growth().to_string() + " key_end=" + DfudsDt::key_end_policy +
               " " + leaf_counts("front_", d_->front().last_batch()) + " " + leaf_counts("rear_", d_->rear().last_batch());
    }
    std::size_t current_key_count() const override { return d_->key_count(); }
    DtTables dt_tables() const override {
        const auto c = d_->table_capacity();
        return {true, c.slots, c.free_ids, c.live_ids, c.owner_capacity, c.handle_capacity, c.handle_width, c.owner_bytes,
                c.handle_bytes};
    }
private:
    void apply_leaf_policy() {
        const auto& r = leaf_policy_request();
        DfudsDt::LeafUpdatePolicy p;
        p.mode = r.adaptive ? DfudsDt::LeafUpdateMode::Adaptive : DfudsDt::LeafUpdateMode::MergeOnly;
        p.max_runs = r.max_runs; p.movement_factor = r.movement_factor;
        d_->set_leaf_update_policy(p);
    }
    std::size_t bc_;
    std::size_t grain_ = 512;
    std::unique_ptr<DfudsDt> d_;
};

}  // namespace

std::unique_ptr<Index> make_dfuds_index(std::size_t bc) { return std::make_unique<DfudsIndex>(bc); }
std::unique_ptr<Index> make_dfuds_index_from_sorted(const std::string_view* sorted, std::size_t n, std::size_t bc) {
    return std::make_unique<DfudsIndex>(bc, DfudsDict::from_sorted_parallel(std::span<const std::string_view>(sorted, n), bc));
}
std::unique_ptr<Index> make_dfuds_index_from_sorted(const std::vector<std::string>& sorted, std::size_t bc) {
    return std::make_unique<DfudsIndex>(bc, DfudsDict::from_sorted_parallel(sorted, bc));
}
// The ordered scans' implementation label (trie/dfuds/detail/scan.hpp: a descent with the reference step and the
// unary-stretch skip, the linear preorder enumeration), appended after the older fields so that the existing gates'
// prefix matches still hold.
constexpr const char* kScanLabel = " range=dfuds_preorder_scan_v1";
std::string dfuds_config_line() {
    // chain_key / chain_walk are fixed descriptions (contains() = the reference descent, contains_sorted_parallel() =
    // the run walk; trie/dfuds/detail/core.hpp), kept so that the result validation's CONFIG matches still hold.
    return std::string("DFUDS_CONFIG impl=library chain_key=ref chain_walk=run transactional=") +
           std::to_string(int(DfudsDict::transactional)) + kScanLabel + " key_end=" + DfudsDict::key_end_policy;
}

std::unique_ptr<Index> make_dfuds_dt_index(std::size_t bc) { return std::make_unique<DfudsDtIndex>(bc); }
std::unique_ptr<Index> make_dfuds_dt_index_from_sorted(const std::string_view* sorted, std::size_t n, std::size_t bc) {
    return std::make_unique<DfudsDtIndex>(bc, DfudsDt::from_sorted_parallel(std::span<const std::string_view>(sorted, n), bc, 0,
                                                                            false, requested_growth()));
}
std::unique_ptr<Index> make_dfuds_dt_index_from_sorted(const std::vector<std::string>& sorted, std::size_t bc) {
    return std::make_unique<DfudsDtIndex>(bc, DfudsDt::from_sorted_parallel(sorted, bc, 0, false, requested_growth()));
}
// The double trie's front walks (trie/dfuds_double_trie.hpp): its queries take the reference DFUDS steps
// (front_steps=ref: contains' detail::step, contains_sorted_parallel's detail::shared_walk), the batch insert's
// insertion walk is the run walk (front_insert_walk=run, fixed), the scans their own descent (kScanLabel) with
// a per-scan remainder memo by id in front of the record walk up the rear (rear_read). The fields after
// transactional= were appended later; the older gates match the prefix.
std::string dfuds_dt_config_line() {
    return std::string("DFUDS_DT_CONFIG impl=header front=dfuds rear=bp front_steps=ref transactional=") +
           std::to_string(int(DfudsDt::transactional)) + " front_insert_walk=run" + kScanLabel + " rear_read=memo_record_walk" +
           " key_end=" + DfudsDt::key_end_policy;
}

}  // namespace dictcmp
