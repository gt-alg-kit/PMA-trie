// Self-containment check for include/block/slots.hpp: this translation unit
// includes ONLY that header (no bit_pma.hpp, no block layouts, no test
// harness) and then uses every public concept and function it declares. If
// the header ever came to rely on a transitive include, this file would stop
// compiling. Compile-only in spirit; main() runs the functions once so the
// test also fails if a definition is silently wrong.

#include "block/slots.hpp"

namespace {

// A minimal layout satisfying block::BitLayout, with the two members and two
// functions the coordinate is defined in terms of. wpb = 2 words -> 128 slots
// per interior block; the tail is 1 word -> 64 slots.
struct FakeLayout {
    std::size_t wpb = 2;
    std::size_t n_blocks = 3;
    std::size_t wpb_of(std::size_t b) const { return b + 1 == n_blocks ? 1 : wpb; }
    std::size_t words_total(std::size_t n) const { return n ? (n - 1) * wpb + 1 : 0; }
};
static_assert(block::BitLayout<FakeLayout>);

struct NotALayout { int x = 0; };
static_assert(!block::BitLayout<NotALayout>);

// A layout that names base_type resolves to it; a plain one resolves to itself.
struct Wrapped : FakeLayout { using base_type = FakeLayout; };
static_assert(std::is_same_v<block::bit_layout_t<FakeLayout>, FakeLayout>);
static_assert(std::is_same_v<block::bit_layout_t<Wrapped>, FakeLayout>);

}  // namespace

int main() {
    const FakeLayout l;
    int bad = 0;
    bad += block::bit_slot_base(l, 0) != 0;
    bad += block::bit_slot_base(l, 2) != 256;
    bad += block::bit_slot_of(l, 1, 5) != 133;
    bad += block::bit_slot_capacity(l, 0) != 128;
    bad += block::bit_slot_capacity(l, 2) != 64;      // the short tail
    bad += block::bit_slot_count(l, 3) != 320;        // 2*128 + 64
    bad += block::bit_slot_count(l, 0) != 0;
    const Wrapped w;
    const FakeLayout& base = block::bit_layout_of(w);
    bad += static_cast<const void*>(&base) != static_cast<const void*>(&w);
    bad += block::bit_slot_of(base, 2, 7) != 263;
    return bad == 0 ? 0 : 1;
}
