// Gate of the dictionary comparison campaign's input layer (trie/bench/common/dictcmp_input.hpp;
// docs/dictionary_comparison_experiment_plan_2026-10-23.md sections 3.5 and 14).
//
//   1. plain versus zstd: the same canonical population decodes to the same bytes, offsets, keys and stream hash;
//   2. byte semantics: CR, NUL, high bytes and the empty key are ordinary key bytes (no normalization);
//   3. malformed inputs are refused: unsorted or duplicate keys, a missing final LF, a truncated or corrupt
//      compressed file, a decoded-size, N or stream-hash mismatch against the manifest values;
//   4. offsets are 64-bit: a stream over 4 GiB indexes correctly (DICTCMP_INPUT_BIG=1, about 4.3 GB of RAM);
//   5. the 70/30 permutation is a bijection, S = floor(7N/10), the stored set sorted equals the first S IDs, and the
//      split, every generated batch and its hash are identical with PARLAY_NUM_THREADS=1 and 64 (the test prints its
//      hashes; ctest runs it with 1 and with 64 workers and the two outputs are compared by the second invocation);
//   6. batch generation: hits only from S, misses only from H, a mixed batch has exactly floor(bq/2) hits, and two
//      different (round, batch) identities give different sequences.
// Writes its fixtures to a private temporary directory; needs `pzstd` on PATH (as the campaign loader does).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

#include "common/dictcmp_input.hpp"

namespace in = dictcmp::input;

namespace {

int failures = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "CHECK failed %s:%d: %s\n", __FILE__, __LINE__, #c); ++failures; } } while (0)

double now() { return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

std::string tmpdir() {
    char t[] = "/tmp/dictcmp_input_XXXXXX";
    const char* d = mkdtemp(t);
    if (!d) throw std::runtime_error("mkdtemp failed");
    return d;
}

void write_file(const std::string& path, const std::string& bytes) {
    std::ofstream f(path, std::ios::binary);
    f.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::string canonical_stream(const std::vector<std::string>& keys) {
    std::string s;
    for (const auto& k : keys) { s += k; s += '\n'; }
    return s;
}

bool loads(const std::string& path, std::size_t n, std::size_t bytes, std::uint64_t hash) {
    try {
        in::LoadReport rep;
        (void)in::load_canonical(path, n, bytes, hash, 4, rep, now);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

}  // namespace

int main() {
    const std::string dir = tmpdir();
    // A population with every awkward byte class, sorted in unsigned-byte order.
    std::vector<std::string> keys = {"", std::string("\0a", 2), "a", "a\r", std::string("a\0b", 3), "ab", "abc",
                                     "b\x7f", "\x80", "\xc3\xa9t\xc3\xa9", "\xff", "\xff\xff"};
    std::sort(keys.begin(), keys.end(), [](const std::string& x, const std::string& y) { return in::byte_less(x, y); });
    const std::string stream = canonical_stream(keys);
    const std::uint64_t h = in::stream_hash(stream.data(), stream.size());
    const std::string plain = dir + "/canon.txt", zst = dir + "/canon.zst";
    write_file(plain, stream);
    CHECK(std::system(("pzstd -q -19 -p 2 -c " + plain + " > " + zst).c_str()) == 0);

    // 1 + 2: plain and compressed decode identically, bytes preserved
    {
        in::LoadReport ra, rb;
        const auto a = in::load_canonical(plain, keys.size(), stream.size(), h, 4, ra, now);
        const auto b = in::load_canonical(zst, keys.size(), stream.size(), h, 4, rb, now);
        CHECK(a.size() == keys.size() && b.size() == keys.size());
        CHECK(ra.stream_hash == h && rb.stream_hash == h);
        CHECK(a.off == b.off);
        for (std::size_t i = 0; i < keys.size(); ++i) {
            CHECK(a.key(i) == keys[i]);
            CHECK(b.key(i) == keys[i]);
        }
        CHECK(a.key(0).empty());
    }
    // 3: malformed inputs
    {
        std::vector<std::string> unsorted = keys;
        std::swap(unsorted[3], unsorted[4]);
        const std::string s1 = canonical_stream(unsorted);
        write_file(dir + "/unsorted.txt", s1);
        CHECK(!loads(dir + "/unsorted.txt", 0, 0, 0));
        std::string dup;   // the canonical stream with its fifth key repeated (built as text: no vector reallocation)
        for (std::size_t i = 0; i < keys.size(); ++i) {
            dup += keys[i]; dup += '\n';
            if (i == 4) { dup += keys[i]; dup += '\n'; }
        }
        write_file(dir + "/dup.txt", dup);
        CHECK(!loads(dir + "/dup.txt", 0, 0, 0));
        write_file(dir + "/nolf.txt", stream.substr(0, stream.size() - 1));
        CHECK(!loads(dir + "/nolf.txt", 0, 0, 0));
        CHECK(!loads(plain, keys.size() + 1, 0, 0));                 // N mismatch
        CHECK(!loads(plain, 0, stream.size() + 1, 0));               // byte-count mismatch
        CHECK(!loads(plain, 0, 0, h ^ 1));                           // stream-hash mismatch
        CHECK(!loads(zst, keys.size(), 0, h));                       // a compressed input needs its decoded size
        CHECK(!loads(zst, keys.size(), stream.size() + 7, h));       // decoded size too large
        CHECK(!loads(zst, keys.size(), stream.size() - 7, h));       // decoded size too small (extra bytes)
        std::ifstream zf(zst, std::ios::binary);
        std::string z((std::istreambuf_iterator<char>(zf)), std::istreambuf_iterator<char>());
        write_file(dir + "/trunc.zst", z.substr(0, z.size() / 2));
        CHECK(!loads(dir + "/trunc.zst", keys.size(), stream.size(), h));
        std::string bad = z;
        bad[bad.size() / 2] ^= 0x5a;
        write_file(dir + "/corrupt.zst", bad);
        CHECK(!loads(dir + "/corrupt.zst", keys.size(), stream.size(), h));
        CHECK(loads(plain, keys.size(), stream.size(), h));          // the valid file still loads
    }
    // 4: 64-bit offsets (optional, ~4.3 GB)
    if (const char* e = std::getenv("DICTCMP_INPUT_BIG"); e && *e == '1') {
        in::Population p;
        const std::size_t klen = 1023, n = (std::size_t{1} << 32) / (klen + 1) + 64;
        p.nbytes = n * (klen + 1);
        p.bytes.reset(new char[p.nbytes]);
        parlay::parallel_for(0, n, [&](std::size_t i) {
            char* q = p.bytes.get() + i * (klen + 1);
            std::memset(q, 'x', klen);
            q[klen] = '\n';
        });
        in::index_lines(p);
        CHECK(p.size() == n);
        CHECK(p.off[n] == p.nbytes);
        CHECK(p.off[n - 1] == static_cast<std::uint64_t>(n - 1) * (klen + 1));
        CHECK(p.off[n - 1] > (std::uint64_t{1} << 32));
        CHECK(p.key(n - 1).size() == klen);
        std::printf("BIG_OFFSETS n=%zu bytes=%zu ok\n", n, p.nbytes);
    }
    // 5 + 6: split and generators
    {
        const std::size_t N = 1000003;
        const in::Split s = in::make_split(N, 20260928);
        CHECK(s.S == (7 * N) / 10 && s.H == N - s.S);
        std::vector<char> seen(N, 0);
        for (std::uint32_t id : s.perm) { CHECK(id < N && !seen[id]); seen[id] = 1; }
        std::vector<std::uint32_t> first(s.perm.begin(), s.perm.begin() + s.S);
        std::sort(first.begin(), first.end());
        CHECK(first == s.stored_sorted);
        std::set<std::uint32_t> inS(s.perm.begin(), s.perm.begin() + s.S);
        const std::size_t bq = 10007;
        std::vector<std::uint32_t> b(bq), c(bq);
        for (const in::Mix mix : {in::Mix::Hit, in::Mix::Miss, in::Mix::Mixed}) {
            const std::uint64_t seed = in::batch_seed(20260928, "fixture", "point", static_cast<int>(mix) + 1, 2, 0);
            in::generate_batch(seed, s, mix, bq, b.data());
            std::size_t hits = 0;
            for (std::uint32_t id : b) hits += inS.count(id);
            CHECK(hits == in::expected_hits(mix, bq));
            in::generate_batch(in::batch_seed(20260928, "fixture", "point", static_cast<int>(mix) + 1, 3, 0), s, mix, bq, c.data());
            CHECK(b != c);   // another round gives another sequence
            in::generate_batch(in::batch_seed(20260928, "fixture", "point", static_cast<int>(mix) + 1, 2, 1), s, mix, bq, c.data());
            CHECK(b != c);   // another batch index gives another sequence
            std::printf("BATCH mix=%s hash=%016llx\n", in::mix_name(mix), static_cast<unsigned long long>(in::ids_hash(b.data(), bq)));
        }
        // The values printed here must not depend on PARLAY_NUM_THREADS (ctest compares the 1- and 64-worker runs).
        std::printf("SPLIT perm_hash=%016llx stored_sorted_hash=%016llx\n", static_cast<unsigned long long>(s.perm_hash),
                    static_cast<unsigned long long>(s.stored_sorted_hash));
        constexpr std::uint64_t kExpectedPermHash = 0x0af68e7897167e05ULL;   // pinned October 24 (1 and 64 workers)
        if (kExpectedPermHash) CHECK(s.perm_hash == kExpectedPermHash);
        // The full-population policy (docs/dictionary_comparison_full_dataset_hit_only_plan_2026-10-29.md): S = N,
        // H = 0, the SAME permutation as the 70/30 split (insertion order unchanged), stored_sorted = identity, every
        // hit batch drawn from the whole population, miss / mixed batches refused.
        const in::Split f = in::make_split(N, 20260928, in::SplitPolicy::Full);
        CHECK(f.N == N && f.S == N && f.H == 0);
        CHECK(f.perm == s.perm && f.perm_hash == s.perm_hash);
        CHECK(f.stored_sorted.size() == N);
        bool identity = true;
        for (std::uint32_t i = 0; i < N; ++i) identity = identity && f.stored_sorted[i] == i;
        CHECK(identity);
        CHECK(f.stored_sorted_hash != s.stored_sorted_hash);
        CHECK(std::string(in::split_policy_name(in::SplitPolicy::Full)) == "full" && in::parse_split_policy("split70") == in::SplitPolicy::Split70);
        {
            const std::uint64_t seed = in::batch_seed(20260928, "fixture", "point", 1, 0, 0);
            in::generate_batch(seed, f, in::Mix::Hit, bq, b.data());
            std::size_t ok = 0, above_s = 0;
            for (std::uint32_t id : b) { ok += id < N; above_s += !inS.count(id); }
            CHECK(ok == bq && above_s > 0);   // ids outside the 70 % set appear: the whole population is the pool
            in::generate_batch(seed, f, in::Mix::Hit, bq, c.data());
            CHECK(b == c);   // deterministic
            std::printf("BATCH mix=hit policy=full hash=%016llx\n", static_cast<unsigned long long>(in::ids_hash(b.data(), bq)));
            bool threw_miss = false, threw_mixed = false;
            try { in::generate_batch(seed, f, in::Mix::Miss, bq, c.data()); } catch (const std::exception&) { threw_miss = true; }
            try { in::generate_batch(seed, f, in::Mix::Mixed, bq, c.data()); } catch (const std::exception&) { threw_mixed = true; }
            CHECK(threw_miss && threw_mixed);
        }
        std::printf("SPLIT policy=full perm_hash=%016llx stored_sorted_hash=%016llx\n", static_cast<unsigned long long>(f.perm_hash),
                    static_cast<unsigned long long>(f.stored_sorted_hash));
    }
    if (std::system(("rm -rf " + dir).c_str()) != 0) std::fprintf(stderr, "warning: could not remove %s\n", dir.c_str());
    std::printf("%s failures=%d workers=%zu\n", failures ? "FAIL" : "PASS", failures, parlay::num_workers());
    return failures ? 1 : 0;
}
