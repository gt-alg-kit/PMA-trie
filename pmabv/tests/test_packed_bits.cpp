#include "packed_bits.hpp"
#include "test_helpers.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

using namespace pmabv;

namespace {

/*
 * Deterministic per-index bit pattern. Mixed density (~50%) and chosen
 * to spread set bits across both halves of every word.
 */
bool det_bit(std::size_t i) {
    return ((i * 1664525u + 1013904223u) & 1u) != 0u;
}

std::vector<bool> make_pattern(std::size_t n) {
    std::vector<bool> v(n);
    for (std::size_t i = 0; i < n; ++i) v[i] = det_bit(i);
    return v;
}

void check_invariants(const PackedBits& pb) {
    const std::size_t need_words = (pb.live + 63) / 64;
    DBV_CHECK(pb.words.size() >= need_words);
    // Tail-zero invariant: bits in [live, 64*words.size()) are zero.
    if (pb.live < pb.words.size() * 64) {
        const std::size_t tail_word = pb.live >> 6;
        const std::size_t tail_bit  = pb.live & 63;
        if (tail_bit != 0) {
            const std::uint64_t mask = ~((1ULL << tail_bit) - 1ULL);
            DBV_CHECK_EQ(pb.words[tail_word] & mask, 0ULL);
        }
        for (std::size_t w = (tail_bit == 0 ? tail_word : tail_word + 1);
             w < pb.words.size(); ++w) {
            DBV_CHECK_EQ(pb.words[w], 0ULL);
        }
    }
}

void test_roundtrip_sizes() {
    for (std::size_t n : {std::size_t{0}, std::size_t{1}, std::size_t{63},
                          std::size_t{64}, std::size_t{65}, std::size_t{127},
                          std::size_t{128}, std::size_t{129},
                          std::size_t{200}}) {
        auto v = make_pattern(n);
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.live, n);
        check_invariants(pb);
        auto back = pb.to_vector_bool();
        DBV_CHECK_EQ(back.size(), v.size());
        bool eq = (back == v);
        DBV_CHECK(eq);
    }
}

void test_roundtrip_all_zero_all_one() {
    for (std::size_t n : {std::size_t{0}, std::size_t{64}, std::size_t{200}}) {
        std::vector<bool> z(n, false);
        PackedBits pz = PackedBits::from_vector_bool(z);
        check_invariants(pz);
        DBV_CHECK_EQ(pz.popcount(), 0u);
        bool eq_z = (pz.to_vector_bool() == z);
        DBV_CHECK(eq_z);

        std::vector<bool> o(n, true);
        PackedBits po = PackedBits::from_vector_bool(o);
        check_invariants(po);
        DBV_CHECK_EQ(po.popcount(), n);
        bool eq_o = (po.to_vector_bool() == o);
        DBV_CHECK(eq_o);
    }
}

void test_popcount_patterns() {
    {
        std::vector<bool> v(192, false);
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), 0u);
    }
    {
        std::vector<bool> v(192, true);
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), 192u);
    }
    {
        // Alternating 0101...: bit i set iff i is odd.
        std::vector<bool> v(200);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i & 1u) != 0u;
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), 100u);
    }
    {
        // Alternating 1010...: bit i set iff i is even.
        std::vector<bool> v(200);
        for (std::size_t i = 0; i < v.size(); ++i) v[i] = (i & 1u) == 0u;
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), 100u);
    }
    {
        // Sparse: one set bit per 64-bit window.
        std::vector<bool> v(256, false);
        for (std::size_t w = 0; w < 4; ++w) v[w * 64 + 17] = true;
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), 4u);
    }
    {
        // Mid-density deterministic pattern.
        const std::size_t n = 333;
        auto v = make_pattern(n);
        std::size_t expected = 0;
        for (bool b : v) if (b) ++expected;
        PackedBits pb = PackedBits::from_vector_bool(v);
        DBV_CHECK_EQ(pb.popcount(), expected);
    }
}

void check_append_case(std::size_t pre_len, std::size_t src_total,
                       std::size_t src_start, std::size_t take) {
    auto pre = make_pattern(pre_len);
    auto src_v = make_pattern(src_total);
    // Shift the src pattern so it differs from pre, exposing alignment
    // bugs that would otherwise be masked by identical content.
    for (std::size_t i = 0; i < src_v.size(); ++i) {
        src_v[i] = det_bit(i + 7919);
    }

    PackedBits dst = PackedBits::from_vector_bool(pre);
    PackedBits src = PackedBits::from_vector_bool(src_v);
    const std::size_t counted =
        dst.append_from_counting(src, src_start, take);

    DBV_CHECK_EQ(dst.live, pre_len + take);
    check_invariants(dst);
    auto back = dst.to_vector_bool();
    DBV_CHECK_EQ(back.size(), pre_len + take);
    std::size_t expected_ones = 0;
    for (std::size_t i = 0; i < pre_len; ++i) {
        DBV_CHECK_EQ(back[i], pre[i]);
    }
    for (std::size_t i = 0; i < take; ++i) {
        DBV_CHECK_EQ(back[pre_len + i], src_v[src_start + i]);
        if (src_v[src_start + i]) ++expected_ones;
    }
    DBV_CHECK_EQ(counted, expected_ones);
}

void test_append_from_alignments() {
    // Source aligned (start at word boundary) + dst aligned.
    check_append_case(/*pre*/ 0,   /*src*/ 200, /*ss*/ 0,  /*len*/ 64);
    check_append_case(/*pre*/ 64,  /*src*/ 200, /*ss*/ 0,  /*len*/ 64);
    check_append_case(/*pre*/ 128, /*src*/ 200, /*ss*/ 64, /*len*/ 64);

    // Source aligned + dst unaligned.
    check_append_case(/*pre*/ 1,   /*src*/ 200, /*ss*/ 0,   /*len*/ 64);
    check_append_case(/*pre*/ 17,  /*src*/ 200, /*ss*/ 64,  /*len*/ 80);
    check_append_case(/*pre*/ 63,  /*src*/ 256, /*ss*/ 128, /*len*/ 100);

    // Source unaligned + dst aligned.
    check_append_case(/*pre*/ 0,   /*src*/ 200, /*ss*/ 5,   /*len*/ 100);
    check_append_case(/*pre*/ 64,  /*src*/ 200, /*ss*/ 13,  /*len*/ 70);
    check_append_case(/*pre*/ 128, /*src*/ 256, /*ss*/ 99,  /*len*/ 50);

    // Source unaligned + dst unaligned, lens crossing 0/1/2/3 dst word
    // boundaries.
    check_append_case(/*pre*/ 5,   /*src*/ 256, /*ss*/ 7,   /*len*/ 30);   // 0 boundaries
    check_append_case(/*pre*/ 50,  /*src*/ 256, /*ss*/ 11,  /*len*/ 30);   // 1 boundary
    check_append_case(/*pre*/ 60,  /*src*/ 256, /*ss*/ 9,   /*len*/ 80);   // 2 boundaries
    check_append_case(/*pre*/ 60,  /*src*/ 384, /*ss*/ 5,   /*len*/ 200);  // 3 boundaries

    // Edge cases: empty append; full source via shorthand.
    check_append_case(/*pre*/ 64,  /*src*/ 64,  /*ss*/ 0,   /*len*/ 0);
    {
        auto pre = make_pattern(40);
        auto sv  = make_pattern(73);
        PackedBits dst = PackedBits::from_vector_bool(pre);
        PackedBits src = PackedBits::from_vector_bool(sv);
        dst.append(src);
        DBV_CHECK_EQ(dst.live, pre.size() + sv.size());
        check_invariants(dst);
        auto back = dst.to_vector_bool();
        for (std::size_t i = 0; i < pre.size(); ++i) {
            DBV_CHECK_EQ(back[i], pre[i]);
        }
        for (std::size_t i = 0; i < sv.size(); ++i) {
            DBV_CHECK_EQ(back[pre.size() + i], sv[i]);
        }
    }
}

void check_extract_case(std::size_t total, std::size_t start,
                        std::size_t len) {
    auto v = make_pattern(total);
    PackedBits src = PackedBits::from_vector_bool(v);
    PackedBits e = src.extract(start, len);
    auto [counted_e, counted] =
        src.extract_with_capacity_counting(start, len, len + 128);
    DBV_CHECK_EQ(e.live, len);
    DBV_CHECK_EQ(counted_e.live, len);
    DBV_CHECK(counted_e.words.size() >= (len + 128 + 63) / 64);
    check_invariants(e);
    check_invariants(counted_e);
    auto back = e.to_vector_bool();
    auto counted_back = counted_e.to_vector_bool();
    DBV_CHECK_EQ(back.size(), len);
    DBV_CHECK_EQ(counted_back.size(), len);
    std::size_t expected_ones = 0;
    for (std::size_t i = 0; i < len; ++i) {
        DBV_CHECK_EQ(back[i], v[start + i]);
        DBV_CHECK_EQ(counted_back[i], v[start + i]);
        if (v[start + i]) ++expected_ones;
    }
    DBV_CHECK_EQ(counted, expected_ones);
}

void test_extract_alignments() {
    // start == 0, varying spans.
    check_extract_case(/*total*/ 256, /*start*/ 0, /*len*/ 0);
    check_extract_case(/*total*/ 256, /*start*/ 0, /*len*/ 1);
    check_extract_case(/*total*/ 256, /*start*/ 0, /*len*/ 64);
    check_extract_case(/*total*/ 256, /*start*/ 0, /*len*/ 200);

    // start at word boundary.
    check_extract_case(/*total*/ 256, /*start*/ 64,  /*len*/ 64);
    check_extract_case(/*total*/ 256, /*start*/ 128, /*len*/ 100);

    // start mid-word, spans crossing 1/2/3 word boundaries.
    check_extract_case(/*total*/ 320, /*start*/ 5,   /*len*/ 50);   // 0 boundaries
    check_extract_case(/*total*/ 320, /*start*/ 5,   /*len*/ 80);   // 1 boundary
    check_extract_case(/*total*/ 320, /*start*/ 11,  /*len*/ 130);  // 2 boundaries
    check_extract_case(/*total*/ 384, /*start*/ 7,   /*len*/ 200);  // 3 boundaries

    // Tail extract reaches exactly the last live bit.
    check_extract_case(/*total*/ 200, /*start*/ 137, /*len*/ 63);
}

#define RUN(t) do { std::cerr << "running " #t "\n"; t(); } while (0)

} // namespace

int main() {
    RUN(test_roundtrip_sizes);
    RUN(test_roundtrip_all_zero_all_one);
    RUN(test_popcount_patterns);
    RUN(test_append_from_alignments);
    RUN(test_extract_alignments);
    return pmabv_test::finalize("test_packed_bits");
}
