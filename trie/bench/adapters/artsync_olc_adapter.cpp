// ART-OLC (optimistic lock coupling) from flode/ARTSynchronized behind the
// comparison drivers' Index interface (--impl art-olc); see artsync_adapter.hpp.
// The ONLY translation unit that includes the OLC headers. The space walk reads
// the tree's private root (access-specifier override; third-party single TU,
// read-only), so every standard and TBB header the
// library pulls in is included before the override.

#include <algorithm>
#include <array>
#include <atomic>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

#include <assert.h>
#include <stdint.h>
#include <string.h>
#include <emmintrin.h>

#include "tbb/combinable.h"
#include "tbb/enumerable_thread_specific.h"

#define private public
#define protected public
#include "OptimisticLockCoupling/Tree.h"
#undef private
#undef protected

#include "adapters/artsync_adapter.hpp"

namespace dictcmp {
namespace {
struct OlcTypes {
    using Tree = ART_OLC::Tree;
    using N = ART_OLC::N;
    using N4 = ART_OLC::N4;
    using N16 = ART_OLC::N16;
    using N48 = ART_OLC::N48;
    using N256 = ART_OLC::N256;
    using NTypes = ART_OLC::NTypes;
    static constexpr const char* name = "olc";
    // The one-sided Tree::lookupRange(start, result[], len, count, ti) is exact only with both scan patches; the
    // seek patch defines ART_OLC_TRIE_BENCH_SCAN_FIXES (artsync_adapter.hpp). Without it no scanner is exposed.
#ifdef ART_OLC_TRIE_BENCH_SCAN_FIXES
    static constexpr bool has_scan = true;
    static constexpr const char* scan_info = "lookupRange+null_child_patch+seek_patch+leading_key_filter";
#else
    static constexpr bool has_scan = false;
    static constexpr const char* scan_info = "unavailable(library_without_the_scan_patches)";
#endif
};
}  // namespace

std::unique_ptr<Index> make_art_olc_index() {
    return std::make_unique<artsync::ArtSyncIndex<OlcTypes>>("ART-OLC (ARTSynchronized)");
}

}  // namespace dictcmp
