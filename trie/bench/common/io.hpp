#pragma once
// Helpers every benchmark driver needs: key files, number lists, the clock
// and the round statistics. Nothing here knows any structure.

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dictcmp {

// One key per line, only the line terminator removed (the CR of CRLF too);
// at most `limit` lines. std::runtime_error when the file cannot be opened.
inline std::vector<std::string> read_lines(const std::string& path,
                                           std::size_t limit = static_cast<std::size_t>(-1)) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path);
    std::vector<std::string> v;
    std::string s;
    while (v.size() < limit && std::getline(in, s)) {
        if (!s.empty() && s.back() == '\r') s.pop_back();
        v.push_back(std::move(s));
    }
    return v;
}

// "1,2,4" -> {1, 2, 4}; an item "a-b" is the inclusive range a .. b (CPU
// lists).
inline std::vector<std::size_t> parse_list(const std::string& s) {
    std::vector<std::size_t> v;
    std::size_t i = 0;
    while (i < s.size()) {
        std::size_t j = s.find(',', i);
        if (j == std::string::npos) j = s.size();
        const std::string tok = s.substr(i, j - i);
        const std::size_t dash = tok.find('-');
        if (dash == std::string::npos) v.push_back(std::stoul(tok));
        else {
            const std::size_t a = std::stoul(tok.substr(0, dash)), b = std::stoul(tok.substr(dash + 1));
            for (std::size_t k = a; k <= b; ++k) v.push_back(k);
        }
        i = j + 1;
    }
    return v;
}

inline double ns_now() {
    using namespace std::chrono;
    return duration<double, std::nano>(steady_clock::now().time_since_epoch()).count();
}
// Arithmetic mean of a sample (0 for an empty one): the per-round averages
// the drivers report.
inline double mean_of(const std::vector<double>& v) {
    double t = 0;
    for (double x : v) t += x;
    return v.empty() ? 0.0 : t / double(v.size());
}
// Median of a sample, the upper middle element of an even count (0 for an
// empty one).
inline double median_of(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

}  // namespace dictcmp
