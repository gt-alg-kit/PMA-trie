#pragma once
// INPUT LAYER of the dictionary comparison campaign (bench_dictcmp, tools/dictcmp_prepare;
// docs/dictionary_comparison_experiment_plan_2026-10-23.md sections 3.5, 6, 7 and 14).
//
// Canonical population. One byte stream of distinct keys in unsigned-byte order, each key followed by '\n' (no LF
// inside a key; every other byte, CR and NUL included, is an ordinary key byte). It is stored either as a plain file
// (the protected data64 `*.full.sorted.txt` files of DNA, WebBase and XML, read where they are, never copied) or
// zstd-compressed (new datasets; decoded through `pzstd -d`). Both forms are read into one buffer allocated by this
// process, so its pages follow the launch's memory policy (numactl --interleave=all): no file mapping is used.
// Key IDs are positions in canonical order, 32-bit (every campaign dataset has N < 2^32; checked).
//
// Hashes. stream_hash() is the campaign's fast identity of a decoded stream: FNV-1a-64 over every 64 MiB chunk, then
// FNV-1a-64 over the chunk hashes (8 little-endian bytes each) and the total length; chunks are hashed in parallel, so
// the value does not depend on the worker count. It is recorded at preparation next to the SHA-256 of the stored file
// and of the decoded stream, and checked at every load. ids_hash() identifies a generated ID sequence (FNV-1a-64 over
// the little-endian 32-bit IDs in order).
//
// Split and generators (plan section 14). mix64 is the SplitMix64 finalizer, a bijection on 64-bit words. The 70/30
// permutation orders the key IDs by mix64(seed ^ mix64(id)) (seed 20260928); the first S = floor(7N/10) IDs are the
// initial population in insertion order, the remaining H = N - S the held-out sequence. A batch's identity hash
// starts at 0x6a09e667f3bcc908 (the first 64 bits of the fractional part of sqrt(2)) and folds each field f in order
// as h = mix64(h ^ f); strings enter as their FNV-1a-64. Element i of a batch with seed h draws u = mix64(h +
// mix64(i + 1)) and takes pool[u % |pool|] (hits from S, misses from H, with replacement). A mixed batch has
// floor(bq/2) hit elements (indices i < floor(bq/2)) and bq - floor(bq/2) miss elements, placed in the order of their
// keys mix64(h ^ mix64(i + 2^63)), a bijection of i, so no ties occur. Everything is counter-based: the sequences do
// not depend on the number of workers.
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

namespace dictcmp::input {

// ---- hashing and mixing ------------------------------------------------------------------------------------------

inline constexpr std::uint64_t kFnvOffset = 1469598103934665603ULL;
inline constexpr std::uint64_t kFnvPrime = 1099511628211ULL;
inline constexpr std::uint64_t kIdentityStart = 0x6a09e667f3bcc908ULL;

inline std::uint64_t fnv1a(const void* data, std::size_t n, std::uint64_t h = kFnvOffset) {
    const auto* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) h = (h ^ p[i]) * kFnvPrime;
    return h;
}
inline std::uint64_t str_hash(std::string_view s) { return fnv1a(s.data(), s.size()); }

// The SplitMix64 finalizer (Steele, Lea, Flood; the output function of java.util.SplittableRandom): xor-shift and odd
// multiplications only, hence a bijection on 64-bit words.
inline constexpr std::uint64_t mix64(std::uint64_t z) {
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// The identity hash of a batch (or of any tuple): fields folded in order.
inline std::uint64_t identity_hash(std::initializer_list<std::uint64_t> fields) {
    std::uint64_t h = kIdentityStart;
    for (std::uint64_t f : fields) h = mix64(h ^ f);
    return h;
}

inline constexpr std::size_t kStreamChunk = std::size_t{1} << 26;   // 64 MiB

inline std::uint64_t stream_hash(const char* data, std::size_t n) {
    const std::size_t chunks = (n + kStreamChunk - 1) / kStreamChunk;
    std::vector<std::uint64_t> h(chunks);
    parlay::parallel_for(0, chunks, [&](std::size_t c) {
        const std::size_t lo = c * kStreamChunk, hi = std::min(n, lo + kStreamChunk);
        h[c] = fnv1a(data + lo, hi - lo);
    }, 1);
    std::uint64_t r = kFnvOffset;
    for (std::uint64_t x : h) {
        unsigned char b[8];
        for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(x >> (8 * i));
        r = fnv1a(b, 8, r);
    }
    unsigned char b[8];
    for (int i = 0; i < 8; ++i) b[i] = static_cast<unsigned char>(static_cast<std::uint64_t>(n) >> (8 * i));
    return fnv1a(b, 8, r);
}

inline std::uint64_t ids_hash(const std::uint32_t* ids, std::size_t n) {
    // Sequential over the IDs in order (little-endian bytes); batches are at most a few million IDs.
    std::uint64_t h = kFnvOffset;
    for (std::size_t i = 0; i < n; ++i) {
        const std::uint32_t x = ids[i];
        const unsigned char b[4] = {static_cast<unsigned char>(x), static_cast<unsigned char>(x >> 8),
                                    static_cast<unsigned char>(x >> 16), static_cast<unsigned char>(x >> 24)};
        h = fnv1a(b, 4, h);
    }
    return h;
}

// Unsigned-byte lexicographic order, the campaign's key order. (std::string_view's comparison goes through
// char_traits<char>::compare, which libstdc++ implements with memcmp, i.e. on unsigned bytes too; the explicit form
// states the contract instead of relying on that.)
inline bool byte_less(std::string_view a, std::string_view b) {
    const std::size_t m = std::min(a.size(), b.size());
    const int c = m ? std::memcmp(a.data(), b.data(), m) : 0;
    return c < 0 || (c == 0 && a.size() < b.size());
}

// ---- the canonical population -------------------------------------------------------------------------------------

struct Population {
    std::unique_ptr<char[]> bytes;     // the canonical stream: keys, each followed by '\n'
    std::size_t nbytes = 0;
    std::vector<std::uint64_t> off;    // N + 1 key start offsets; off[N] = nbytes
    std::size_t size() const { return off.empty() ? 0 : off.size() - 1; }
    std::string_view key(std::size_t id) const {
        return std::string_view(bytes.get() + off[id], static_cast<std::size_t>(off[id + 1] - off[id] - 1));
    }
};

inline bool ends_with(const std::string& s, const char* suffix) {
    const std::size_t m = std::strlen(suffix);
    return s.size() >= m && s.compare(s.size() - m, m, suffix) == 0;
}

// Reads a plain file into a fresh buffer with parallel pread of 64 MiB pieces.
inline void read_plain(const std::string& path, Population& p) {
    const int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
    struct stat st{};
    if (::fstat(fd, &st) != 0) { ::close(fd); throw std::runtime_error("cannot stat " + path); }
    p.nbytes = static_cast<std::size_t>(st.st_size);
    p.bytes.reset(new char[p.nbytes ? p.nbytes : 1]);
    const std::size_t pieces = (p.nbytes + kStreamChunk - 1) / kStreamChunk;
    std::atomic<bool> failed{false};
    parlay::parallel_for(0, pieces, [&](std::size_t c) {
        std::size_t lo = c * kStreamChunk;
        const std::size_t hi = std::min(p.nbytes, lo + kStreamChunk);
        while (lo < hi) {
            const ssize_t r = ::pread(fd, p.bytes.get() + lo, hi - lo, static_cast<off_t>(lo));
            if (r <= 0) { failed = true; return; }
            lo += static_cast<std::size_t>(r);
        }
    }, 1);
    ::close(fd);
    if (failed) throw std::runtime_error("read error in " + path);
}

// Decodes a zstd file through `pzstd -d -c` into a buffer of exactly `expected_bytes` (from the manifest).
inline void read_zstd(const std::string& path, std::size_t expected_bytes, int decode_threads, Population& p) {
    if (expected_bytes == 0) throw std::runtime_error("a compressed canonical input needs its decoded size (manifest)");
    for (char c : path)
        if (c == '\'' || c == '\\') throw std::runtime_error("unsupported character in path " + path);
    const std::string cmd = "pzstd -q -d -c -p " + std::to_string(decode_threads) + " '" + path + "'";
    FILE* f = ::popen(cmd.c_str(), "r");
    if (!f) throw std::runtime_error("cannot start: " + cmd);
    p.nbytes = expected_bytes;
    p.bytes.reset(new char[p.nbytes]);
    std::size_t got = 0;
    while (got < p.nbytes) {
        const std::size_t r = std::fread(p.bytes.get() + got, 1, std::min<std::size_t>(p.nbytes - got, 1 << 24), f);
        if (r == 0) break;
        got += r;
    }
    char extra;
    const bool longer = std::fread(&extra, 1, 1, f) == 1;
    const int rc = ::pclose(f);
    if (rc != 0 || got != p.nbytes || longer)
        throw std::runtime_error("decode of " + path + " failed or size mismatch: got " + std::to_string(got) +
                                 (longer ? "+" : "") + " bytes, expected " + std::to_string(p.nbytes) +
                                 ", pzstd status " + std::to_string(rc));
}

// Builds the N + 1 offsets with a parallel newline count; throws if the stream does not end with '\n'.
inline void index_lines(Population& p) {
    if (p.nbytes == 0) { p.off.assign(1, 0); return; }
    if (p.bytes[p.nbytes - 1] != '\n') throw std::runtime_error("canonical stream does not end with a newline");
    const std::size_t chunks = (p.nbytes + kStreamChunk - 1) / kStreamChunk;
    std::vector<std::size_t> cnt(chunks + 1, 0);
    const char* b = p.bytes.get();
    parlay::parallel_for(0, chunks, [&](std::size_t c) {
        const std::size_t lo = c * kStreamChunk, hi = std::min(p.nbytes, lo + kStreamChunk);
        std::size_t k = 0;
        for (const char* q = b + lo; (q = static_cast<const char*>(std::memchr(q, '\n', static_cast<std::size_t>(b + hi - q))));
             ++q) ++k;
        cnt[c] = k;
    }, 1);
    std::size_t total = 0;
    for (std::size_t c = 0; c < chunks; ++c) { const std::size_t k = cnt[c]; cnt[c] = total; total += k; }
    p.off.resize(total + 1);
    p.off[0] = 0;
    parlay::parallel_for(0, chunks, [&](std::size_t c) {
        const std::size_t lo = c * kStreamChunk, hi = std::min(p.nbytes, lo + kStreamChunk);
        std::size_t k = cnt[c];
        for (const char* q = b + lo; (q = static_cast<const char*>(std::memchr(q, '\n', static_cast<std::size_t>(b + hi - q))));
             ++q) p.off[++k] = static_cast<std::uint64_t>(q - b + 1);
    }, 1);
}

// Number of adjacent pairs that are not strictly increasing (0 for a valid canonical population).
inline std::size_t order_violations(const Population& p) {
    const std::size_t n = p.size();
    if (n < 2) return 0;
    return parlay::reduce(parlay::delayed_seq<std::size_t>(n - 1, [&](std::size_t i) -> std::size_t {
        return byte_less(p.key(i), p.key(i + 1)) ? 0 : 1;
    }));
}

struct LoadReport {
    std::size_t n = 0, nbytes = 0;
    std::uint64_t stream_hash = 0;
    std::size_t order_violations = 0;
    double ns_read = 0, ns_index = 0, ns_hash = 0, ns_check = 0;
};

// Loads a canonical population (plain or *.zst) and verifies it: the newline structure, strict unsigned-byte order
// and, when given (non-zero), N, the decoded byte count and the stream hash from the manifest. Throws on a mismatch.
template <class Clock>
inline Population load_canonical(const std::string& path, std::size_t expect_n, std::size_t expect_bytes,
                                 std::uint64_t expect_hash, int decode_threads, LoadReport& rep, Clock now) {
    Population p;
    double t = now();
    if (ends_with(path, ".zst")) read_zstd(path, expect_bytes, decode_threads, p);
    else read_plain(path, p);
    rep.ns_read = now() - t; t = now();
    index_lines(p);
    rep.ns_index = now() - t; t = now();
    rep.stream_hash = stream_hash(p.bytes.get(), p.nbytes);
    rep.ns_hash = now() - t; t = now();
    rep.order_violations = order_violations(p);
    rep.ns_check = now() - t;
    rep.n = p.size(); rep.nbytes = p.nbytes;
    if (p.size() >= (std::size_t{1} << 32)) throw std::runtime_error("N >= 2^32: 32-bit key IDs do not suffice");
    if (rep.order_violations) throw std::runtime_error("canonical stream is not strictly sorted: " +
                                                       std::to_string(rep.order_violations) + " violations");
    if (expect_n && rep.n != expect_n)
        throw std::runtime_error("N mismatch: " + std::to_string(rep.n) + " vs manifest " + std::to_string(expect_n));
    if (expect_bytes && rep.nbytes != expect_bytes)
        throw std::runtime_error("byte count mismatch: " + std::to_string(rep.nbytes) + " vs manifest " +
                                 std::to_string(expect_bytes));
    if (expect_hash && rep.stream_hash != expect_hash) throw std::runtime_error("stream hash mismatch");
    return p;
}

// ---- split ------------------------------------------------------------------------------------------------------

struct Split {
    std::size_t N = 0, S = 0, H = 0;
    std::vector<std::uint32_t> perm;            // first S = initial population in insertion order; rest = held-out
    std::vector<std::uint32_t> stored_sorted;   // the S initial IDs ascending (= canonical key order)
    std::uint64_t perm_hash = 0, stored_sorted_hash = 0;
    const std::uint32_t* stored() const { return perm.data(); }
    const std::uint32_t* held() const { return perm.data() + S; }
};

inline std::size_t initial_size(std::size_t n) { return (7 * n) / 10; }   // floor(7N/10), exact for N < 2^60

// The stored-population policy. Split70 is the campaign's historical 70/30 split (S = floor(7N/10) initial keys, H
// held out for insertions and misses), the default of every mode. Full (the full-population, hit-only round of
// docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md) stores every canonical key: S = N, H = 0, with
// the SAME seed-derived permutation as the insertion order, so stored_sorted is the identity and every query is a hit.
enum class SplitPolicy { Split70, Full };
inline const char* split_policy_name(SplitPolicy p) { return p == SplitPolicy::Full ? "full" : "split70"; }
inline SplitPolicy parse_split_policy(const std::string& s) {
    if (s == "split70") return SplitPolicy::Split70;
    if (s == "full") return SplitPolicy::Full;
    throw std::runtime_error("unknown population policy " + s + " (split70|full)");
}

inline Split make_split(std::size_t n, std::uint64_t seed, SplitPolicy policy = SplitPolicy::Split70) {
    Split s;
    s.N = n; s.S = policy == SplitPolicy::Full ? n : initial_size(n); s.H = n - s.S;
    auto keyed = parlay::tabulate(n, [&](std::size_t id) {
        return std::pair<std::uint64_t, std::uint32_t>(mix64(seed ^ mix64(static_cast<std::uint64_t>(id))),
                                                       static_cast<std::uint32_t>(id));
    });
    parlay::sort_inplace(keyed, [](const auto& a, const auto& b) { return a.first < b.first; });
    s.perm.resize(n);
    parlay::parallel_for(0, n, [&](std::size_t i) { s.perm[i] = keyed[i].second; });
    keyed = decltype(keyed)();   // release the 16-byte pairs before the next allocation
    std::vector<std::uint8_t> in_s(n, 0);
    parlay::parallel_for(0, s.S, [&](std::size_t i) { in_s[s.perm[i]] = 1; });
    auto ids = parlay::pack_index<std::uint32_t>(parlay::make_slice(in_s));
    s.stored_sorted.assign(ids.begin(), ids.end());
    s.perm_hash = ids_hash(s.perm.data(), s.perm.size());
    s.stored_sorted_hash = ids_hash(s.stored_sorted.data(), s.stored_sorted.size());
    return s;
}

// ---- batch generators (plan section 14) -----------------------------------------------------------------------------

enum class Mix { Hit, Miss, Mixed };
inline const char* mix_name(Mix m) { return m == Mix::Hit ? "hit" : m == Mix::Miss ? "miss" : "mixed"; }
inline Mix parse_mix(const std::string& s) {
    if (s == "hit") return Mix::Hit;
    if (s == "miss") return Mix::Miss;
    if (s == "mixed") return Mix::Mixed;
    throw std::runtime_error("unknown mix " + s + " (hit|miss|mixed)");
}

// The identity of one generated batch: campaign seed, dataset, panel ("point" / "range"), mix or k, round, batch.
inline std::uint64_t batch_seed(std::uint64_t campaign_seed, std::string_view dataset, std::string_view panel,
                                std::uint64_t mix_or_k, std::uint64_t round, std::uint64_t batch) {
    return identity_hash({campaign_seed, str_hash(dataset), str_hash(panel), mix_or_k, round, batch});
}

inline std::uint64_t draw(std::uint64_t h, std::uint64_t i) { return mix64(h + mix64(i + 1)); }

// Fills out[0..bq) with the batch's key IDs: hits from S (split.stored()), misses from H (split.held()).
inline void generate_batch(std::uint64_t h, const Split& split, Mix mix, std::size_t bq, std::uint32_t* out) {
    const std::uint32_t* S = split.stored();
    const std::uint32_t* H = split.held();
    const std::uint64_t ns = split.S, nh = split.H;
    if (mix == Mix::Hit) {
        if (!ns) throw std::runtime_error("empty initial population");
        parlay::parallel_for(0, bq, [&](std::size_t i) { out[i] = S[draw(h, i) % ns]; });
        return;
    }
    if (mix == Mix::Miss) {
        if (!nh) throw std::runtime_error("empty held-out set");
        parlay::parallel_for(0, bq, [&](std::size_t i) { out[i] = H[draw(h, i) % nh]; });
        return;
    }
    if (!ns || !nh) throw std::runtime_error("a mixed batch needs both pools");
    const std::size_t hits = bq / 2;
    auto order = parlay::tabulate(bq, [&](std::size_t i) {
        return std::pair<std::uint64_t, std::uint32_t>(mix64(h ^ mix64(static_cast<std::uint64_t>(i) + (std::uint64_t{1} << 63))),
                                                       static_cast<std::uint32_t>(i));
    });
    parlay::sort_inplace(order, [](const auto& a, const auto& b) { return a.first < b.first; });
    parlay::parallel_for(0, bq, [&](std::size_t pos) {
        const std::uint32_t e = order[pos].second;
        out[pos] = e < hits ? S[draw(h, e) % ns] : H[draw(h, e) % nh];
    });
}

// Expected number of members in a batch of the given mix (the timing processes' cheap result check).
inline std::size_t expected_hits(Mix mix, std::size_t bq) { return mix == Mix::Hit ? bq : mix == Mix::Miss ? 0 : bq / 2; }

// ---- materialization ------------------------------------------------------------------------------------------------

inline void views_of(const Population& p, const std::uint32_t* ids, std::size_t m, std::vector<std::string_view>& out) {
    out.resize(m);
    parlay::parallel_for(0, m, [&](std::size_t i) { out[i] = p.key(ids[i]); });
}
inline void strings_of(const Population& p, const std::uint32_t* ids, std::size_t m, std::vector<std::string>& out) {
    out.clear();
    out.resize(m);
    parlay::parallel_for(0, m, [&](std::size_t i) { out[i].assign(p.key(ids[i])); });
}

}  // namespace dictcmp::input
