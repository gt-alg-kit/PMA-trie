#pragma once
// A std::vector whose elements are DEFAULT-initialized on resize / sized construction (indeterminate for trivial
// types: no fill), through an allocator whose construct() default-initializes. Copying and filling behave as for
// std::vector; only sizing without a value skips the initialization. Used by
// global batch rebuilds for storage that the following writers overwrite completely.
#include <cstddef>
#include <memory>
#include <new>
#include <type_traits>
#include <utility>
#include <vector>

namespace pmabv {

template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;
    template <class U> struct rebind { using other = DefaultInitAllocator<U>; };
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) { ::new (static_cast<void*>(p)) U; }
    template <class U, class... Args>
    void construct(U* p, Args&&... args) { ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...); }
};

template <class T>
using overwrite_vec = std::vector<T, DefaultInitAllocator<T>>;

}  // namespace pmabv
