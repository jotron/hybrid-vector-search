#include <vector>
#include <memory>
#include <cstdlib>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include "hybrid_vector_search.h"

using namespace std;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
             const std::string &path = "output.bin")
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const uint32_t N = knns.size();
    assert(knns.front().size() == K);
    for (unsigned i = 0; i < N; ++i)
    {
        auto const &knn = knns[i];
        ofs.write(reinterpret_cast<char const *>(&knn[0]), K * sizeof(uint32_t));
    }
    ofs.close();
}

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<float>> &data)
{
    // std::cout << "Reading data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N; // num of points
    ifs.read((char *)&N, sizeof(uint32_t));
    data.resize(N);
    // std::cout << "# of points: " << N << std::endl;
    std::vector<float> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(float)))
    {
        std::vector<float> row(num_dimensions);
        for (int d = 0; d < num_dimensions; d++)
        {
            row[d] = static_cast<float>(buff[d]);
        }
        data[counter++] = std::move(row);
    }
    ifs.close();
}

/// @brief Reading output bin
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadOutputBin(const std::string &file_path,
                   const int num_queries,
                   std::vector<std::vector<uint32_t>> &data,
                   const int num_dimensions = 100)
{
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    data.resize(num_queries);
    std::vector<uint32_t> buff(num_dimensions);
    int counter = 0;
    while (ifs.read((char *)buff.data(), num_dimensions * sizeof(uint32_t)))
    {
        std::vector<uint32_t> row = buff;
        data[counter++] = std::move(row);
    }
    ifs.close();
}