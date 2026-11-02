// Zero-cost NullAug tests (build-order Step 1).
//
// BitPMAImpl gained a third, defaulted template parameter `Aug`. The default
// NullAug must be exactly what existing users already had: no augmentation
// array, no allocation, no change to object size or memory accounting, and
// no change to observable behavior.
//
// Two tiers of check live here.
//
//   Universal -- run everywhere. The NullAug store is empty; the default
//   template spelling selects NullAug; no augmentation is allocated;
//   raw + aux == byte_size; behavior matches a vector<bool> oracle; non-null
//   augmentation accounting grows by exactly the expected per-node storage.
//
//   Captured-ABI -- the exact sizeof / byte_size / tree_bytes values of the
//   recorded NullAug baseline (AppleClang 21 / arm64). These depend on the
//   C++ ABI and the standard library's container layouts, so they are valid
//   ONLY on the platform they were captured on (AppleClang 21 / arm64 / libc++). They are
//   NOT portable to GCC, libstdc++, Linux, x86-64 or another AppleClang major.
//   CMake defines PMABV_NULLAUG_CAPTURED_APPLE_ARM64_ABI only there.
//
// Scope: the augmentation-policy CONTRACT (aug_store.hpp's concept), storage,
// and NullAug zero-cost. Maintenance behaviour -- that every mutation path
// actually refreshes the augmentation -- lives in tests/test_aug.cpp. No BP
// concepts belong in either file; BpAug is tested in tests/test_bp_*.cpp.

#include "aug_store.hpp"
#include "bit_pma.hpp"
#include "test_helpers.hpp"

#include <cstddef>    // std::size_t
#include <cstdint>
#include <iostream>
#include <sstream>
#include <span>
#include <string>
#include <type_traits>
#include <utility>    // std::declval
#include <vector>

using namespace pmabv;

// The block layout the DEFAULT alias (BitPMA) resolves to: pmabv::DefaultBlock
// (block::Soa). Every "NullAug adds nothing" comparison below holds its block
// layout fixed, so an augmented type is instantiated over this same layout.
using TestDefaultBlock = pmabv::DefaultBlock;
static_assert(std::is_same_v<BitPMA, BitPMAImpl<EytzingerTreePolicy, TestDefaultBlock>>,
              "TestDefaultBlock must be the layout the default alias uses");

namespace {

// ---------------------------------------------------------------------
// A minimal non-null augmentation, test-only and free of BP concepts. It
// deliberately MIRRORS the built-in (live, ones) summary, which makes it a
// self-checking oracle: after any mutation the root value must equal
// (size(), rank(true, size())) obtained through the public API.
//
// `a` = live bits, `b` = one bits.
// ---------------------------------------------------------------------
struct FakeAug {
    struct Value {
        std::int64_t a = 0;
        std::int64_t b = 0;
        bool operator==(const Value&) const = default;
    };
    static Value empty() noexcept { return Value{}; }

    // Masks to exactly `live` bits. Words past `live` are zero-filled padding.
    static Value from_block(std::span<const std::uint64_t> words,
                            std::size_t live) noexcept {
        std::int64_t ones = 0;
        const std::size_t full = live / 64, rem = live % 64;
        for (std::size_t i = 0; i < full; ++i) {
            ones += __builtin_popcountll(words[i]);
        }
        if (rem) {
            const std::uint64_t mask = (~std::uint64_t{0}) >> (64 - rem);
            ones += __builtin_popcountll(words[full] & mask);
        }
        return Value{static_cast<std::int64_t>(live), ones};
    }

    static Value combine(Value l, Value r) noexcept {
        return Value{l.a + r.a, l.b + r.b};
    }
};

// A second VALID policy, distinct from FakeAug. Used to prove that a caller
// cannot pass someone else's policy to root_aug<A>() and reinterpret storage.
struct MirrorLikeAug {
    struct Value { int x = 0; bool operator==(const Value&) const = default; };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>,
                            std::size_t) noexcept { return {}; }
    static Value combine(Value, Value) noexcept { return {}; }
};

// A VALID policy whose Value has NO default constructor. The engine never
// default-constructs a Value -- AugStore fills itself with
// `v_.assign(n, Aug::empty())`, which copy-constructs from the identity -- so
// requiring default-constructibility would over-constrain the concept for no
// reason. This policy exists to keep that from regressing.
struct NoDefaultCtorAug {
    struct Value {
        std::int64_t n;
        Value() = delete;                                  // deliberately absent
        explicit constexpr Value(std::int64_t v) noexcept : n(v) {}
        Value(const Value&) noexcept = default;
        Value(Value&&) noexcept = default;
        Value& operator=(const Value&) noexcept = default;
        Value& operator=(Value&&) noexcept = default;
        bool operator==(const Value&) const noexcept = default;
    };
    static Value empty() noexcept { return Value{0}; }
    static Value from_block(std::span<const std::uint64_t>,
                            std::size_t live) noexcept {
        return Value{static_cast<std::int64_t>(live)};
    }
    static Value combine(Value l, Value r) noexcept { return Value{l.n + r.n}; }
};

// ---------------------------------------------------------------------
// Policies that must be REJECTED, each violating exactly one requirement.
// ---------------------------------------------------------------------
struct NoValueAug {                       // no ::Value at all
    static int empty() noexcept { return 0; }
};
struct MissingCombineAug {                // from_block but no combine
    struct Value { int v = 0; bool operator==(const Value&) const = default; };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>,
                            std::size_t) noexcept { return {}; }
};
struct ThrowingFoldAug {                  // from_block may throw
    struct Value { int v = 0; bool operator==(const Value&) const = default; };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>, std::size_t) {
        return {};                        // deliberately NOT noexcept
    }
    static Value combine(Value, Value) noexcept { return {}; }
};
struct ThrowingCombineAug {               // combine may throw
    struct Value { int v = 0; bool operator==(const Value&) const = default; };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>,
                            std::size_t) noexcept { return {}; }
    static Value combine(Value, Value) { return {}; }
};
struct ThrowingCopyAug {                  // Value copy may throw
    struct Value {
        Value() noexcept = default;
        Value(const Value&) {}            // NOT noexcept
        Value& operator=(const Value&) noexcept { return *this; }
        bool operator==(const Value&) const noexcept { return true; }
    };
    static Value empty() noexcept { return {}; }
    static Value from_block(std::span<const std::uint64_t>,
                            std::size_t) noexcept { return {}; }
    static Value combine(Value, Value) noexcept { return {}; }
};

// Detects a nested ::Value. A requires-expression would not do here: for a
// non-dependent type the missing member is a hard error, not substitution
// failure, so the check has to go through a dependent partial specialization.
template <class A, class = void>
struct has_value_type : std::false_type {};
template <class A>
struct has_value_type<A, std::void_t<typename A::Value>> : std::true_type {};

// Detects `pma.root_aug()` and `pma.template root_aug<A>()`. Dependent for the
// same reason.
template <class T, class = void>
struct has_root_aug : std::false_type {};
template <class T>
struct has_root_aug<T, std::void_t<decltype(std::declval<const T&>().root_aug())>>
    : std::true_type {};

template <class T, class A, class = void>
struct has_root_aug_as : std::false_type {};
template <class T, class A>
struct has_root_aug_as<
    T, A, std::void_t<decltype(std::declval<const T&>().template root_aug<A>())>>
    : std::true_type {};

// ---------------------------------------------------------------------
// Deterministic fixture, mirroring the recorded NullAug baseline.
// ---------------------------------------------------------------------
std::vector<bool> pattern(std::size_t n) {
    std::vector<bool> v(n);
    std::uint64_t x = 0x9E3779B97F4A7C15ULL;
    for (std::size_t i = 0; i < n; ++i) {
        x ^= x << 13; x ^= x >> 7; x ^= x << 17;
        v[i] = (x & 0x11) != 0;
    }
    return v;
}

// True only where the recorded baseline's exact numbers apply: the
// captured platform/toolchain (via CMake) AND the captured engine config.
// A preprocessor guard, not `if constexpr`: a static_assert with a
// non-dependent condition inside a discarded `if constexpr` branch of a
// NON-TEMPLATE function is still evaluated, so the numeric asserts would fire
// on other platforms regardless of the flag.
#if defined(PMABV_NULLAUG_CAPTURED_APPLE_ARM64_ABI) \
    && PMABV_DEFAULT_BC == 16896
#  define PMABV_NULLAUG_PIN_CAPTURED 1
#endif

#ifdef PMABV_NULLAUG_PIN_CAPTURED
constexpr bool kPinned = true;
#else
constexpr bool kPinned = false;
#endif

// ---------------------------------------------------------------------
// 0. The augmentation-policy contract.
// ---------------------------------------------------------------------
void test_policy_concept() {
    // NullAug is accepted despite having no Value and no operations: it
    // bypasses the non-null branch of the disjunction entirely.
    static_assert(AugmentationPolicy<NullAug>);
    static_assert(!NonNullAugmentationPolicy<NullAug>);

    // A complete, all-noexcept policy is accepted both ways.
    static_assert(NonNullAugmentationPolicy<FakeAug>);
    static_assert(AugmentationPolicy<FakeAug>);

    // Not over-constrained: a Value with a DELETED default constructor is
    // valid, because the engine only ever copies from Aug::empty().
    static_assert(!std::is_default_constructible_v<NoDefaultCtorAug::Value>,
                  "the point of this policy is the deleted default ctor");
    static_assert(NonNullAugmentationPolicy<NoDefaultCtorAug>);
    static_assert(AugmentationPolicy<NoDefaultCtorAug>);

    // Each rejection isolates one violated requirement.
    static_assert(!AugmentationPolicy<NoValueAug>,        "needs ::Value");
    static_assert(!AugmentationPolicy<MissingCombineAug>, "needs combine()");
    static_assert(!AugmentationPolicy<ThrowingFoldAug>,
                  "from_block must be noexcept: it runs after the block and "
                  "the built-in summaries have already changed");
    static_assert(!AugmentationPolicy<ThrowingCombineAug>,
                  "combine must be noexcept");
    static_assert(!AugmentationPolicy<ThrowingCopyAug>,
                  "Value copy/move must be noexcept for AugStore");

    // The static_assert above only proves the CONCEPT accepts it. Instantiate
    // and mutate a real engine with that policy, so a store that secretly
    // default-constructs (e.g. resize() instead of assign()) fails to compile.
    using NoDefault = BitPMAImpl<EytzingerTreePolicy, block::Soa,
                                 NoDefaultCtorAug>;
    NoDefault pma(pattern(5000), 1024);
    pma.insert(100, std::vector<bool>{true, false});
    pma.set(7, true);
    pma.erase(3, 9);
    pma.check_invariants();
    // MirrorAug-style semantics: this policy sums live bits, so the root must
    // equal size().
    DBV_CHECK_EQ(pma.root_aug().n, static_cast<std::int64_t>(pma.size()));
}

// ---------------------------------------------------------------------
// 0b. root_aug() is exposed only where it is meaningful.
// ---------------------------------------------------------------------
void test_root_aug_api() {
    using Augmented = BitPMAImpl<EytzingerTreePolicy, block::Soa, FakeAug>;
    // Available on an augmented instantiation...
    static_assert(has_root_aug<Augmented>::value);
    // ...absent on NullAug, rather than a hard error.
    static_assert(!has_root_aug<BitPMA>::value);
    // A caller cannot substitute an unrelated policy to reinterpret storage.
    static_assert(has_root_aug_as<Augmented, FakeAug>::value);
    static_assert(!has_root_aug_as<Augmented, MirrorLikeAug>::value,
                  "root_aug<A> must be pinned to the instantiation's own Aug");
    static_assert(!has_root_aug_as<Augmented, NullAug>::value);
    // Returns a const reference, not a copy.
    static_assert(std::is_same_v<decltype(std::declval<const Augmented&>()
                                              .root_aug()),
                                 const FakeAug::Value&>);
    DBV_CHECK(true);
}

// ---------------------------------------------------------------------
// 1. Existing spellings still compile, and select NullAug.
// ---------------------------------------------------------------------
void test_default_parameter_is_nullaug() {
    // One- and two-argument spellings, exactly as existing code writes them.
    static_assert(std::is_same_v<BitPMAImpl<EytzingerTreePolicy>,
                                 BitPMAImpl<EytzingerTreePolicy, block::Soa,
                                            NullAug>>,
                  "BitPMAImpl<Policy> must select block::Soa and NullAug");
    static_assert(std::is_same_v<BitPMAImpl<EytzingerTreePolicy, block::Soa>,
                                 BitPMAImpl<EytzingerTreePolicy, block::Soa,
                                            NullAug>>,
                  "BitPMAImpl<Policy, Block> must select NullAug");
    // The public alias is unaffected.
    static_assert(std::is_same_v<BitPMA,
                                 BitPMAImpl<EytzingerTreePolicy, block::Soa,
                                            NullAug>>);
    // An augmented instantiation is a *different*, also-valid type.
    static_assert(!std::is_same_v<BitPMA,
                                  BitPMAImpl<EytzingerTreePolicy, block::Soa,
                                             FakeAug>>);
    DBV_CHECK(true);
}

// ---------------------------------------------------------------------
// 2. AugStore<NullAug> is empty and allocation-free.
// ---------------------------------------------------------------------
void test_nullaug_store_is_free() {
    static_assert(std::is_empty_v<AugStore<NullAug>>,
                  "AugStore<NullAug> must be an empty type");
    static_assert(sizeof(AugStore<NullAug>) == 1,
                  "an empty class still has sizeof 1 standalone; "
                  "[[no_unique_address]] is what removes it from the parent");

    AugStore<NullAug> s;
    s.resize_nodes(1u << 20);          // would be 8 MB if it allocated
    DBV_CHECK_EQ(s.bytes(), std::size_t{0});
    DBV_CHECK_EQ(s.size(), std::size_t{0});
    s.clear();
    DBV_CHECK_EQ(s.bytes(), std::size_t{0});

    // NullAug carries no Value: the generic store cannot be instantiated for
    // it, which is why the specialization exists.
    static_assert(!has_value_type<NullAug>::value,
                  "NullAug must not define Value");
    static_assert(has_value_type<FakeAug>::value,
                  "a non-null augmentation must define Value");
}

// ---------------------------------------------------------------------
// 3. The generic store works for a non-null policy (compile + behavior).
//    Storage only -- nothing here maintains values on mutation.
// ---------------------------------------------------------------------
void test_generic_store_for_non_null_aug() {
    AugStore<FakeAug> s;
    DBV_CHECK_EQ(s.size(), std::size_t{0});
    DBV_CHECK_EQ(s.bytes(), std::size_t{0});

    s.resize_nodes(7);
    DBV_CHECK_EQ(s.size(), std::size_t{7});
    DBV_CHECK(s.bytes() >= 7 * sizeof(FakeAug::Value));
    for (std::size_t i = 0; i < 7; ++i) {
        DBV_CHECK(s[i] == FakeAug::empty());
    }
    s[3] = FakeAug::Value{5, -9};
    DBV_CHECK_EQ(s[3].a, std::int64_t{5});
    DBV_CHECK_EQ(s[3].b, std::int64_t{-9});
    // resize_nodes re-establishes the identity everywhere.
    s.resize_nodes(4);
    DBV_CHECK_EQ(s.size(), std::size_t{4});
    DBV_CHECK(s[3] == FakeAug::empty());

    s.clear();
    DBV_CHECK_EQ(s.size(), std::size_t{0});
    DBV_CHECK_EQ(s.bytes(), std::size_t{0});
}

// ---------------------------------------------------------------------
// 4. Object size unchanged against the captured baseline.
// ---------------------------------------------------------------------
void test_sizeof_matches_baseline() {
#ifdef PMABV_NULLAUG_PIN_CAPTURED
    // the recorded baseline's structural sizes. Real preprocessor
    // guard: these are only meaningful on the captured ABI.
    // The captured 432 was the rank-only block::Soa; the retained layout adds the
    // excess slab (a vector and a size_t, 32 bytes on this ABI). Not re-measured
    // on the captured toolchain after the 2026-10-29 layout cleanup.
    static_assert(sizeof(BitPMA) == 464, "sizeof(BitPMA) regressed");
#endif
    // Universal: the NullAug store must contribute nothing to the object.
    // BitPMAImpl<Eytzinger, TestDefaultBlock> is BitPMA spelled out -- so any
    // size these do not share would mean the store leaked in.
    DBV_CHECK_EQ(sizeof(BitPMAImpl<EytzingerTreePolicy, TestDefaultBlock>),
                 sizeof(BitPMA));
    // An augmented instantiation holds a real vector; a NullAug one must not.
    // Same block layout as BitPMA, so the ONLY difference is the store.
    using Augmented = BitPMAImpl<EytzingerTreePolicy, TestDefaultBlock, FakeAug>;
    DBV_CHECK(sizeof(Augmented) > sizeof(BitPMA));
    DBV_CHECK_EQ(sizeof(Augmented) - sizeof(BitPMA),
                 sizeof(std::vector<FakeAug::Value>));
}

// ---------------------------------------------------------------------
// 5. Memory accounting unchanged, against the captured per-fixture values.
// ---------------------------------------------------------------------
struct Expected {
    std::size_t n, bc, byte_size, tree_bytes;
    std::uint64_t func_fnv;
};

// FUNCTIONAL checksum over the logical query surface: every get(), then
// strided rank(true)/rank(false) over [0, n] (both endpoints always probed),
// then strided select(true). Stride is max(1, n/64) so the cost stays bounded.
//
// Unlike byte_size/tree_bytes this is PORTABLE -- it depends only on query
// results, never on STL layout -- so it is pinned unconditionally, outside
// PMABV_NULLAUG_PIN_CAPTURED. That makes it the strongest of these pins: it is
// the one that would catch an augmentation change altering observable results.
//
// The stride convention is stated here because it is part of the value's
// definition. These numbers were regenerated by computing this checksum on
// BOTH the pre-Step-1 revision (2647f9ef, extracted read-only via
// `git archive`) and the current source, and confirming they agree exactly;
// see the recorded baseline's values in this file.
template <class T>
std::uint64_t func_checksum(const T& pma) {
    std::uint64_t h = 14695981039346656037ULL;
    const auto byte = [&h](unsigned char c) { h ^= c; h *= 1099511628211ULL; };
    const auto u64  = [&byte](std::uint64_t v) {
        for (int i = 0; i < 8; ++i) byte(static_cast<unsigned char>(v >> (8 * i)));
    };
    const std::size_t n = pma.size();
    for (std::size_t i = 0; i < n; ++i) byte(static_cast<unsigned char>(pma.get(i)));
    const std::size_t st = n / 64 ? n / 64 : 1;
    for (std::size_t k = 0; k <= n; k += st) {
        u64(pma.rank(true, k));
        u64(pma.rank(false, k));
    }
    u64(pma.rank(true, n));                       // always probe the endpoint
    u64(pma.rank(false, n));
    const std::size_t ones = pma.rank(true, n);
    const std::size_t sst = ones / 64 ? ones / 64 : 1;
    for (std::size_t j = 0; j < ones; j += sst) u64(pma.select(true, j));
    return h;
}

template <class T>
void check_fixture(const char* name, const Expected& e) {
    const T pma(pattern(e.n), e.bc);

    // ---- universal: hold on every platform and configuration ----
    // raw + aux == byte_size must still hold with the new term folded in.
    DBV_CHECK_EQ(pma.raw_storage_bytes() + pma.auxiliary_bytes(),
                 pma.byte_size());

    // Functional baseline: portable, so pinned on every platform.
    DBV_CHECK_EQ(func_checksum(pma), e.func_fnv);

    // ---- captured ABI only ----
    // byte_size/tree_bytes depend on STL container layout, so they are not
    // portable. See the file header.
#ifdef PMABV_NULLAUG_PIN_CAPTURED
    DBV_CHECK_EQ(pma.byte_size(), e.byte_size);
    DBV_CHECK_EQ(pma.tree_bytes(), e.tree_bytes);
#else
    (void)e.byte_size; (void)e.tree_bytes;
#endif
    (void)name;
}

void test_accounting_unchanged() {
    // the recorded baseline's memory accounting and checksums.
    // Regenerated with the corrected FNV-1a offset basis and verified equal
    // between the pre-Step-1 revision and the current source.
    check_fixture<BitPMA>("BitPMA n=0",
        {0, 16896, 508, 16, 0x0c8210784d8af5a5ULL});
    check_fixture<BitPMA>("BitPMA n=1",
        {1, 16896, 508, 16, 0xd82079c57cb144ecULL});
    check_fixture<BitPMA>("BitPMA n=1000",
        {1000, 16896, 648, 16, 0xe7725f797f1a0f6eULL});
    check_fixture<BitPMA>("BitPMA n=100000",
        {100000, 16896, 14478, 240, 0xde2dc5a2ef5d24afULL});
    check_fixture<BitPMA>("BitPMA n=1000 bc=1024",
        {1000, 1024, 690, 48, 0xe7725f797f1a0f6eULL});
    check_fixture<BitPMA>("BitPMA n=100000 bc=1024",
        {100000, 1024, 19408, 4080, 0xde2dc5a2ef5d24afULL});
}

// ---------------------------------------------------------------------
// 6. Observable behavior unchanged: query and mutation paths still agree
//    with a vector<bool> oracle. The delta fast paths are what maintain
//    (live, ones); this is the end-to-end evidence they are untouched.
// ---------------------------------------------------------------------
template <class T>
void test_behavior_unchanged(std::size_t n, std::size_t bc) {
    std::vector<bool> oracle = pattern(n);
    T pma(oracle, bc);

    // set()  -> adjust_ones_to_root
    for (std::size_t i = 0; i < n; i += 311) {
        const bool v = ((i / 311) & 1) != 0;
        pma.set(i, v);
        oracle[i] = v;
    }
    // insert() -> apply_summary_deltas_along
    for (std::size_t i = 0; i < 200; ++i) {
        const std::size_t pos = (i * 7919) % (oracle.size() + 1);
        const std::vector<bool> add{(i & 1) != 0, (i & 2) != 0};
        pma.insert(pos, add);
        oracle.insert(oracle.begin() + static_cast<std::ptrdiff_t>(pos),
                      add.begin(), add.end());
    }
    // erase() -> deduct_erase_to_root
    for (std::size_t i = 0; i < 60; ++i) {
        const std::size_t s = (i * 4441) % oracle.size();
        const std::size_t e = std::min(s + (i % 5) + 1, oracle.size());
        if (s >= e) continue;
        pma.erase(s, e);
        oracle.erase(oracle.begin() + static_cast<std::ptrdiff_t>(s),
                     oracle.begin() + static_cast<std::ptrdiff_t>(e));
    }

    DBV_CHECK_EQ(pma.size(), oracle.size());
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        DBV_CHECK_EQ(pma.get(i), static_cast<std::uint8_t>(oracle[i]));
    }
    std::size_t ones = 0;
    for (std::size_t i = 0; i <= oracle.size(); ++i) {
        DBV_CHECK_EQ(pma.rank(true, i), ones);
        DBV_CHECK_EQ(pma.rank(false, i), i - ones);
        if (i < oracle.size() && oracle[i]) ++ones;
    }
    std::size_t seen = 0;
    for (std::size_t i = 0; i < oracle.size(); ++i) {
        if (oracle[i]) DBV_CHECK_EQ(pma.select(true, seen++), i);
    }
}

// ---------------------------------------------------------------------
// 7. The third parameter really threads through BitPMAImpl, and the
//    accounting terms are wired (not merely zero). Storage only: nothing
//    below maintains an augmentation value.
// ---------------------------------------------------------------------
void test_non_null_instantiation_is_wired() {
    // Over the default layout, so the byte_size() arithmetic below isolates
    // the augmentation store and nothing else.
    using Augmented = BitPMAImpl<EytzingerTreePolicy, TestDefaultBlock, FakeAug>;
    const std::vector<bool> bits = pattern(100000);

    const BitPMA plain(bits, 16896);
    const Augmented aug(bits, 16896);

    // Same content and same block storage: only the extra per-node array differs.
    DBV_CHECK_EQ(aug.size(), plain.size());
    for (std::size_t i = 0; i < aug.size(); i += 1237) {
        DBV_CHECK_EQ(aug.get(i), plain.get(i));
    }
    DBV_CHECK_EQ(aug.raw_storage_bytes(), plain.raw_storage_bytes());

    // The augmentation array is one Value per node, over the same node index
    // space as node_live_/node_one_ -- so tree_bytes() grows by exactly that,
    // which is what proves the accounting is wired rather than stubbed.
    const std::size_t node_bytes = plain.tree_bytes() / 2;   // one uint64 array
    const std::size_t n_nodes = node_bytes / sizeof(std::uint64_t);
    DBV_CHECK_EQ(aug.tree_bytes(),
                 plain.tree_bytes() + n_nodes * sizeof(FakeAug::Value));
    DBV_CHECK_EQ(aug.byte_size(),
                 plain.byte_size() + n_nodes * sizeof(FakeAug::Value)
                     + (sizeof(Augmented) - sizeof(BitPMA)));
    DBV_CHECK(aug.byte_size() > plain.byte_size());

    // raw + aux == byte_size still holds with a non-empty store.
    DBV_CHECK_EQ(aug.raw_storage_bytes() + aug.auxiliary_bytes(),
                 aug.byte_size());
}

}  // namespace

int main() {
    test_policy_concept();
    test_root_aug_api();
    test_default_parameter_is_nullaug();
    test_nullaug_store_is_free();
    test_generic_store_for_non_null_aug();
    test_sizeof_matches_baseline();
    test_accounting_unchanged();
    test_behavior_unchanged<BitPMA>(50000, 16896);
    test_behavior_unchanged<BitPMA>(50000, 1024);
    test_non_null_instantiation_is_wired();
    // Make the tier visible: a green run on a non-captured platform has NOT
    // checked the pinned sizeof/byte_size values.
    std::cout << "nullaug: captured-ABI value pins "
              << (kPinned ? "ENABLED (AppleClang 21 / arm64)"
                          : "SKIPPED (ABI-independent checks only)")
              << '\n';
    return pmabv_test::finalize("nullaug");
}
