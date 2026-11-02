// Dataset preparation and census of the dictionary comparison campaign
// (docs/dictionary_comparison_experiment_plan_2026-10-23.md, sections 3.1, 3.2 and 3.5).
//
// Reads raw lines (one key per LF-terminated line; a missing final LF is added and counted), optionally drops and
// validates one header line (Wikipedia's `page_title`), keeps every other byte exactly (no CR stripping, decoding,
// case folding or truncation), sorts the keys in unsigned-byte order and removes duplicates, and writes the canonical
// stream (each key followed by '\n') to stdout, where the caller compresses it (`| pzstd -19`). It also writes a census
// (JSON): N, key bytes, the exact length and adjacent-LCP histograms with quantiles, the byte alphabet, empty keys, keys
// with NUL / CR / LF-free high bytes / 0xFF, keys over 254 and over 65,535 bytes, the plain-trie node count
// V = 1 + sum(|k_i| - LCP(k_i, k_{i-1})), raw line / header / duplicate counts, and the stream hash of
// common/dictcmp_input.hpp. With --census-only the input must already be canonical (strictly sorted, distinct): the
// protected data64 files are verified in place and nothing is written to stdout.
//
// usage: dictcmp_prepare --dataset <id> --in <path|-> --census <out.json> [--header <line>] [--census-only]
//                        [--reserve-bytes <n>]
// Everything runs in RAM (the input, its line offsets, the key views and the output stream: about 2.3x the input
// plus 24 bytes per key); budget it against free memory before a large run.
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <parlay/parallel.h>
#include <parlay/primitives.h>

#include "common/dictcmp_input.hpp"

namespace {

using dictcmp::input::byte_less;
using dictcmp::input::Population;

double now_ns() {
    return std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Args {
    std::string dataset, in, census, header;
    bool census_only = false;
    std::size_t reserve = 0;
};

Args parse(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; ++i) {
        const std::string s = argv[i];
        auto need = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error("missing value after " + s);
            return argv[++i];
        };
        if (s == "--dataset") a.dataset = need();
        else if (s == "--in") a.in = need();
        else if (s == "--census") a.census = need();
        else if (s == "--header") a.header = need();
        else if (s == "--census-only") a.census_only = true;
        else if (s == "--reserve-bytes") a.reserve = std::stoull(need());
        else throw std::runtime_error("unknown argument " + s);
    }
    if (a.dataset.empty() || a.in.empty() || a.census.empty())
        throw std::runtime_error("usage: dictcmp_prepare --dataset <id> --in <path|-> --census <out.json> [--header <line>] "
                                 "[--census-only] [--reserve-bytes <n>]");
    return a;
}

// stdin into one buffer (geometric growth from the reserve hint).
void read_stdin(Population& p, std::size_t reserve) {
    std::size_t cap = std::max<std::size_t>(reserve, std::size_t{1} << 26), n = 0;
    std::unique_ptr<char[]> buf(new char[cap]);
    for (;;) {
        if (n == cap) {
            const std::size_t ncap = cap * 2;
            std::unique_ptr<char[]> nb(new char[ncap]);
            std::memcpy(nb.get(), buf.get(), n);
            buf.swap(nb);
            cap = ncap;
        }
        const std::size_t r = std::fread(buf.get() + n, 1, cap - n, stdin);
        if (r == 0) break;
        n += r;
    }
    if (std::ferror(stdin)) throw std::runtime_error("read error on stdin");
    p.bytes = std::move(buf);
    p.nbytes = n;
}

std::string json_escape(std::string_view s) {
    std::string o;
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += static_cast<char>(c); }
        else if (c < 0x20 || c >= 0x7f) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o += static_cast<char>(c);
    }
    return o;
}

// Sparse exact histogram of a sorted value array: [[value, count], ...] plus quantiles.
std::string histogram_json(std::vector<std::uint32_t>& v) {
    parlay::sort_inplace(parlay::make_slice(v));
    std::string o = "{\"pairs\": [";
    for (std::size_t i = 0; i < v.size();) {
        std::size_t j = i;
        while (j < v.size() && v[j] == v[i]) ++j;
        if (i) o += ", ";
        o += "[" + std::to_string(v[i]) + ", " + std::to_string(j - i) + "]";
        i = j;
    }
    o += "]";
    auto q = [&](double f) -> std::uint64_t {
        if (v.empty()) return 0;
        std::size_t k = static_cast<std::size_t>(f * static_cast<double>(v.size() - 1));
        return v[k];
    };
    long double sum = 0;
    for (std::uint32_t x : v) sum += x;
    char b[256];
    std::snprintf(b, sizeof b, ", \"min\": %llu, \"p50\": %llu, \"p90\": %llu, \"p99\": %llu, \"p999\": %llu, \"max\": %llu, \"mean\": %.6Lf}",
                  (unsigned long long)q(0), (unsigned long long)q(0.5), (unsigned long long)q(0.9), (unsigned long long)q(0.99),
                  (unsigned long long)q(0.999), (unsigned long long)q(1.0), v.empty() ? 0.0L : sum / static_cast<long double>(v.size()));
    return o + b;
}

std::size_t lcp(std::string_view a, std::string_view b) {
    const std::size_t m = std::min(a.size(), b.size());
    std::size_t i = 0;
    while (i < m && a[i] == b[i]) ++i;
    return i;
}

}  // namespace

int main(int argc, char** argv) try {
    const Args a = parse(argc, argv);
    const double t0 = now_ns();
    Population raw;
    if (a.in == "-") read_stdin(raw, a.reserve);
    else dictcmp::input::read_plain(a.in, raw);
    const double t_read = now_ns();
    bool added_final_lf = false;
    if (raw.nbytes && raw.bytes[raw.nbytes - 1] != '\n') {
        std::unique_ptr<char[]> nb(new char[raw.nbytes + 1]);
        std::memcpy(nb.get(), raw.bytes.get(), raw.nbytes);
        nb[raw.nbytes] = '\n';
        raw.bytes.swap(nb);
        ++raw.nbytes;
        added_final_lf = true;
    }
    dictcmp::input::index_lines(raw);
    const std::size_t raw_lines = raw.size();
    std::size_t first = 0;
    if (!a.header.empty()) {
        if (raw_lines == 0 || raw.key(0) != a.header)
            throw std::runtime_error("expected header line '" + a.header + "' not found as the first line");
        first = 1;
    }
    const std::size_t m = raw_lines - first;
    std::vector<std::string_view> keys(m);
    parlay::parallel_for(0, m, [&](std::size_t i) { keys[i] = raw.key(first + i); });
    const double t_index = now_ns();
    std::size_t order_violations = 0;
    if (a.census_only) {
        order_violations = parlay::reduce(parlay::delayed_seq<std::size_t>(m ? m - 1 : 0, [&](std::size_t i) -> std::size_t {
            return byte_less(keys[i], keys[i + 1]) ? 0 : 1;
        }));
        if (order_violations)
            throw std::runtime_error("--census-only input is not strictly sorted and distinct: " +
                                     std::to_string(order_violations) + " violations");
    } else {
        parlay::sort_inplace(parlay::make_slice(keys), byte_less);
        auto flags = parlay::tabulate(m, [&](std::size_t i) -> bool { return i == 0 || keys[i] != keys[i - 1]; });
        auto uniq = parlay::pack(parlay::make_slice(keys), flags);
        keys.assign(uniq.begin(), uniq.end());
    }
    const std::size_t n = keys.size();
    const double t_sort = now_ns();

    // The canonical output stream (for --census-only, the same bytes as the input after any header line).
    Population out;
    {
        auto lens = parlay::tabulate(n, [&](std::size_t i) -> std::uint64_t { return keys[i].size() + 1; });
        auto [offs, total] = parlay::scan(lens);
        out.nbytes = total;
        out.bytes.reset(new char[total ? total : 1]);
        out.off.resize(n + 1);
        parlay::parallel_for(0, n, [&](std::size_t i) {
            out.off[i] = offs[i];
            std::memcpy(out.bytes.get() + offs[i], keys[i].data(), keys[i].size());
            out.bytes[offs[i] + keys[i].size()] = '\n';
        });
        out.off[n] = total;
    }
    const double t_stream = now_ns();
    const std::uint64_t shash = dictcmp::input::stream_hash(out.bytes.get(), out.nbytes);

    // Census over the canonical keys.
    std::vector<std::uint32_t> lens(n), lcps(n);
    parlay::parallel_for(0, n, [&](std::size_t i) {
        const std::string_view k = out.key(i);
        lens[i] = static_cast<std::uint32_t>(k.size());
        lcps[i] = i ? static_cast<std::uint32_t>(lcp(out.key(i - 1), k)) : 0;
    });
    const std::uint64_t key_bytes = out.nbytes - n;
    const std::uint64_t v_nodes = 1 + parlay::reduce(parlay::delayed_seq<std::uint64_t>(n, [&](std::size_t i) -> std::uint64_t {
        return lens[i] - lcps[i];
    }));
    auto count_if_key = [&](auto pred) {
        return parlay::reduce(parlay::delayed_seq<std::size_t>(n, [&](std::size_t i) -> std::size_t { return pred(out.key(i)) ? 1 : 0; }));
    };
    const std::size_t empty = count_if_key([](std::string_view k) { return k.empty(); });
    const std::size_t with_nul = count_if_key([](std::string_view k) { return std::memchr(k.data(), 0, k.size()) != nullptr; });
    const std::size_t with_cr = count_if_key([](std::string_view k) { return std::memchr(k.data(), '\r', k.size()) != nullptr; });
    const std::size_t with_ff = count_if_key([](std::string_view k) { return std::memchr(k.data(), 0xff, k.size()) != nullptr; });
    const std::size_t with_high = count_if_key([](std::string_view k) {
        for (unsigned char c : k) if (c >= 0x80) return true;
        return false;
    });
    const std::size_t over254 = count_if_key([](std::string_view k) { return k.size() > 254; });
    const std::size_t over65535 = count_if_key([](std::string_view k) { return k.size() > 65535; });
    // Alphabet: per 64 MiB chunk of the stream, then summed; the LF delimiters are removed from the count.
    const std::size_t chunks = (out.nbytes + dictcmp::input::kStreamChunk - 1) / dictcmp::input::kStreamChunk;
    std::vector<std::array<std::uint64_t, 256>> part(chunks);
    parlay::parallel_for(0, chunks, [&](std::size_t c) {
        auto& h = part[c];
        h.fill(0);
        const std::size_t lo = c * dictcmp::input::kStreamChunk, hi = std::min(out.nbytes, lo + dictcmp::input::kStreamChunk);
        for (std::size_t i = lo; i < hi; ++i) ++h[static_cast<unsigned char>(out.bytes[i])];
    }, 1);
    std::array<std::uint64_t, 256> alpha{};
    for (const auto& h : part) for (int c = 0; c < 256; ++c) alpha[c] += h[c];
    alpha['\n'] -= n;
    std::string alphabet = "{";
    int distinct_bytes = 0;
    for (int c = 0; c < 256; ++c) {
        if (!alpha[c]) continue;
        if (distinct_bytes++) alphabet += ", ";
        alphabet += "\"" + std::to_string(c) + "\": " + std::to_string(alpha[c]);
    }
    alphabet += "}";
    const std::string len_hist = histogram_json(lens);
    const std::string lcp_hist = histogram_json(lcps);
    const double t_census = now_ns();

    const std::string header_json = a.header.empty() ? std::string("null") : "\"" + json_escape(a.header) + "\"";
    FILE* cf = std::fopen(a.census.c_str(), "w");
    if (!cf) throw std::runtime_error("cannot write " + a.census);
    std::fprintf(cf,
                 "{\n \"dataset\": \"%s\",\n \"tool\": \"dictcmp_prepare\",\n \"input\": \"%s\",\n \"census_only\": %s,\n"
                 " \"header_line\": %s,\n \"raw_lines\": %zu,\n \"added_final_lf\": %s,\n \"raw_keys\": %zu,\n"
                 " \"duplicates_removed\": %zu,\n \"N\": %zu,\n \"key_bytes\": %llu,\n \"stream_bytes\": %zu,\n"
                 " \"stream_hash\": \"%016llx\",\n \"trie_nodes_V\": %llu,\n \"empty_keys\": %zu,\n \"keys_with_nul\": %zu,\n"
                 " \"keys_with_cr\": %zu,\n \"keys_with_0xff\": %zu,\n \"keys_with_high_byte\": %zu,\n \"keys_over_254\": %zu,\n"
                 " \"keys_over_65535\": %zu,\n \"distinct_bytes\": %d,\n \"alphabet\": %s,\n \"length\": %s,\n \"adjacent_lcp\": %s,\n"
                 " \"seconds\": {\"read\": %.3f, \"index\": %.3f, \"sort_unique\": %.3f, \"stream\": %.3f, \"census\": %.3f}\n}\n",
                 json_escape(a.dataset).c_str(), json_escape(a.in).c_str(), a.census_only ? "true" : "false",
                 header_json.c_str(), raw_lines,
                 added_final_lf ? "true" : "false", m, m - n, n, (unsigned long long)key_bytes, out.nbytes,
                 (unsigned long long)shash, (unsigned long long)v_nodes, empty, with_nul, with_cr, with_ff, with_high, over254,
                 over65535, distinct_bytes, alphabet.c_str(), len_hist.c_str(), lcp_hist.c_str(), (t_read - t0) / 1e9,
                 (t_index - t_read) / 1e9, (t_sort - t_index) / 1e9, (t_stream - t_sort) / 1e9, (t_census - t_stream) / 1e9);
    std::fclose(cf);
    std::fprintf(stderr, "PREPARE_CENSUS dataset=%s N=%zu key_bytes=%llu stream_bytes=%zu stream_hash=%016llx raw_lines=%zu "
                         "duplicates=%zu nul=%zu cr=%zu over254=%zu over65535=%zu empty=%zu\n",
                 a.dataset.c_str(), n, (unsigned long long)key_bytes, out.nbytes, (unsigned long long)shash, raw_lines, m - n,
                 with_nul, with_cr, over254, over65535, empty);
    if (!a.census_only) {
        constexpr std::size_t kPiece = std::size_t{1} << 24;
        for (std::size_t o = 0; o < out.nbytes; o += kPiece) {
            const std::size_t w = std::min(kPiece, out.nbytes - o);
            if (std::fwrite(out.bytes.get() + o, 1, w, stdout) != w) throw std::runtime_error("write error on stdout");
        }
        if (std::fflush(stdout) != 0) throw std::runtime_error("flush error on stdout");
    }
    std::fprintf(stderr, "PREPARE_OK dataset=%s seconds=%.1f\n", a.dataset.c_str(), (now_ns() - t0) / 1e9);
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "PREPARE_FAILED %s\n", e.what());
    return 1;
}
