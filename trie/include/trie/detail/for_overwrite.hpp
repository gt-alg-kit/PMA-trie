#pragma once
// Per-batch scratch arrays that are written in full before they are read
// (the double trie's batch paths).
//
// std::vector<T>(n) value-initializes every element on the calling thread:
// a sequential zero-fill, which on a large block is also where every page
// of freshly mapped memory is first touched and faulted in, while the other
// workers wait. A batch of 1.4 million keys builds several hundred MB of
// such arrays and frees them again at its end, so the next batch pays the
// same faults. ForOverwrite<T> allocates the storage and leaves it as
// allocated: the first write -- and with it the page fault -- happens in the
// parallel loop that fills the array. (The same idea as C++20's
// std::make_unique_for_overwrite, which however default-initializes, i.e.
// runs the default member initializers of class types element by element.)
//
// Contract: every element is written before it is read; the code that uses
// an instance states why. T must be trivially copyable and trivially
// destructible, i.e. an implicit-lifetime type: the allocation function
// implicitly creates the element objects (C++20 [intro.object]/10,
// [basic.stc.dynamic.allocation]), so plain assignment writes them and no
// destructor has to run. Debug builds fill the storage with 0xA5 bytes, so
// a read before the write produces garbage that the tests catch instead of
// a zero that happens to be right.
//
// Exceptions: only the allocation can throw (std::bad_alloc, or
// std::bad_array_new_length when n * sizeof(T) overflows), before anything
// is written; move-only, the destructor and the moves are noexcept.
#include <algorithm>
#include <cstddef>
#include <cstring>
#include <limits>
#include <new>
#include <type_traits>
#include <utility>
#include <parlay/parallel.h>

namespace trie::detail {

template <class T>
class ForOverwrite {
    static_assert(std::is_trivially_copyable_v<T> && std::is_trivially_destructible_v<T>,
                  "ForOverwrite<T>: T must be an implicit-lifetime type (trivially copyable and destructible)");
    static_assert(alignof(T) <= __STDCPP_DEFAULT_NEW_ALIGNMENT__, "ForOverwrite<T>: over-aligned T");
public:
    using value_type = T;
    using size_type = std::size_t;
    using iterator = T*;
    using const_iterator = const T*;

    ForOverwrite() noexcept = default;
    explicit ForOverwrite(std::size_t n) : p_(allocate(n)), n_(n) {
#ifndef NDEBUG
        poison();
#endif
    }
    ForOverwrite(ForOverwrite&& o) noexcept : p_(std::exchange(o.p_, nullptr)), n_(std::exchange(o.n_, 0)) {}
    ForOverwrite& operator=(ForOverwrite&& o) noexcept {
        if (this != &o) {
            release();
            p_ = std::exchange(o.p_, nullptr);
            n_ = std::exchange(o.n_, 0);
        }
        return *this;
    }
    ForOverwrite(const ForOverwrite&) = delete;
    ForOverwrite& operator=(const ForOverwrite&) = delete;
    ~ForOverwrite() { release(); }

    std::size_t size() const noexcept { return n_; }
    bool empty() const noexcept { return n_ == 0; }
    T* data() noexcept { return p_; }
    const T* data() const noexcept { return p_; }
    T& operator[](std::size_t i) noexcept { return p_[i]; }
    const T& operator[](std::size_t i) const noexcept { return p_[i]; }
    T& back() noexcept { return p_[n_ - 1]; }
    const T& back() const noexcept { return p_[n_ - 1]; }
    T* begin() noexcept { return p_; }
    T* end() noexcept { return p_ + n_; }
    const T* begin() const noexcept { return p_; }
    const T* end() const noexcept { return p_ + n_; }

private:
    static T* allocate(std::size_t n) {
        if (n == 0) return nullptr;
        if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_array_new_length();
        return static_cast<T*>(::operator new(n * sizeof(T)));
    }
    void release() noexcept {
        if (p_) ::operator delete(p_, n_ * sizeof(T));
        p_ = nullptr;
        n_ = 0;
    }
#ifndef NDEBUG
    void poison() noexcept {
        unsigned char* b = reinterpret_cast<unsigned char*>(p_);
        const std::size_t bytes = n_ * sizeof(T);
        constexpr std::size_t kChunk = std::size_t{1} << 16;
        parlay::parallel_for(0, (bytes + kChunk - 1) / kChunk, [&](std::size_t c) {
            const std::size_t lo = c * kChunk;
            std::memset(b + lo, 0xA5, std::min(kChunk, bytes - lo));
        }, 1);
    }
#endif
    T* p_ = nullptr;
    std::size_t n_ = 0;
};

// n copies of v, written in parallel: the arrays that need an initial value
// keep it, and their first touch is parallel too.
template <class T>
ForOverwrite<T> filled_par(std::size_t n, const T& v) {
    ForOverwrite<T> a(n);
    parlay::parallel_for(0, n, [&](std::size_t i) { a[i] = v; }, 4096);
    return a;
}

}  // namespace trie::detail
