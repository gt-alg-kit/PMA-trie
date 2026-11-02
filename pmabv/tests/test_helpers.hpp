#pragma once

#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Tiny self-contained test harness so the project doesn't need GoogleTest.

// Test-only string-to-bools converter. The production API is
// vector<bool>-shaped; tests read more naturally with "10110" literals
// than with vector<bool>{1,0,1,1,0} initializers, so this helper
// bridges the two without putting a string parser in the public API.
inline std::vector<bool> make_bools(std::string_view bits) {
    std::vector<bool> out;
    out.reserve(bits.size());
    for (char c : bits) {
        if (c == '0') {
            out.push_back(false);
        } else if (c == '1') {
            out.push_back(true);
        } else {
            throw std::invalid_argument(
                "make_bools: invalid bit character (only '0'/'1' allowed)");
        }
    }
    return out;
}

// Inverse: bools to a '0'/'1' string. Useful in failure messages.
inline std::string bools_to_string(const std::vector<bool>& v) {
    std::string s;
    s.reserve(v.size());
    for (bool b : v) s.push_back(b ? '1' : '0');
    return s;
}

// std::vector<bool> has no built-in stream insertion. Provide one so
// DBV_CHECK_EQ can format a useful failure message when the operands
// are vector<bool>. Renders as a '0'/'1' string.
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<bool>& v) {
    return os << bools_to_string(v);
}

namespace pmabv_test {

struct Stats {
    int tests = 0;
    int failures = 0;
};

inline Stats& stats() {
    static Stats s;
    return s;
}

inline void report_failure(const char* file, int line,
                           const std::string& msg) {
    ++stats().failures;
    std::cerr << "FAIL " << file << ":" << line << " " << msg << "\n";
}

inline int finalize(const char* suite_name) {
    std::cerr << suite_name << ": " << stats().tests << " checks, "
              << stats().failures << " failures\n";
    return stats().failures == 0 ? 0 : 1;
}

} // namespace pmabv_test

#define DBV_CHECK(cond)                                                         \
    do {                                                                        \
        ++::pmabv_test::stats().tests;                                            \
        if (!(cond)) {                                                          \
            ::pmabv_test::report_failure(__FILE__, __LINE__, #cond);              \
        }                                                                       \
    } while (0)

#define DBV_CHECK_EQ(a, b)                                                      \
    do {                                                                        \
        ++::pmabv_test::stats().tests;                                            \
        auto _va = (a);                                                         \
        auto _vb = (b);                                                         \
        if (!(_va == _vb)) {                                                    \
            std::ostringstream _os;                                             \
            _os << #a " == " #b " (lhs=" << _va << ", rhs=" << _vb << ")";      \
            ::pmabv_test::report_failure(__FILE__, __LINE__, _os.str());          \
        }                                                                       \
    } while (0)

#define DBV_CHECK_THROWS(expr, exc_type)                                        \
    do {                                                                        \
        ++::pmabv_test::stats().tests;                                            \
        bool _caught = false;                                                   \
        try {                                                                   \
            (void)(expr);                                                       \
        } catch (const exc_type&) {                                             \
            _caught = true;                                                     \
        } catch (...) {                                                         \
            ::pmabv_test::report_failure(__FILE__, __LINE__,                      \
                                       "wrong exception from " #expr);         \
            break;                                                              \
        }                                                                       \
        if (!_caught) {                                                         \
            ::pmabv_test::report_failure(__FILE__, __LINE__,                      \
                                       "no exception thrown from " #expr);     \
        }                                                                       \
    } while (0)
