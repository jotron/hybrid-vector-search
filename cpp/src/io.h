#pragma once
#include <vector>
#include <memory>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include "hybrid_vector_search.h"

using namespace std;

// For aligned Vectors
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

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
             const std::string &path = "output.bin");

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<float>> &data);

/// @brief Reading output bin
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadOutputBin(const std::string &file_path,
                   const int num_queries,
                   std::vector<std::vector<uint32_t>> &data,
                   const int num_dimensions = 100);