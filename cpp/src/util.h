#include <iostream>
#include <vector>
#include <memory>
#include <cstdlib>

template <typename T, std::size_t Alignment>
struct AlignedAllocator
{
    using value_type = T;

    AlignedAllocator() noexcept = default;

    template <typename U>
    constexpr AlignedAllocator(const AlignedAllocator<U, Alignment> &) noexcept {}

    template <typename U>
    struct rebind
    {
        using other = AlignedAllocator<U, Alignment>;
    };

    T *allocate(std::size_t n)
    {
        void *ptr = std::aligned_alloc(Alignment, n * sizeof(T));
        std::cout << "Allocating " << ptr << std::endl;
        if (!ptr)
        {
            throw std::bad_alloc();
        }
        return reinterpret_cast<T *>(ptr);
    }

    void deallocate(T *p, std::size_t) noexcept
    {
        std::free(p);
    }
};

template <typename T, std::size_t Alignment>
using aligned_vector = std::vector<T, AlignedAllocator<T, Alignment>>;