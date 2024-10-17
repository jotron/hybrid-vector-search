/**
 *  Brute force approach. Can be used to generate ground truth.
 *  Usage:
 *   ./brute_force data.bin queries.bin output.bin
 */

#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <assert.h>
#include "hybrid_vector_search.h"

using std::cout;
using std::endl;
using std::pair;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

/// @brief Save knng in binary format (uint32_t) with name "output.bin"
/// @param knn a (N * 100) shape 2-D vector
/// @param path target save path, the output knng should be named as
/// "output.bin" for evaluation
void SaveKNN(const std::vector<std::vector<uint32_t>> &knns,
             const std::string &path = "output.bin")
{
    std::ofstream ofs(path, std::ios::out | std::ios::binary);
    const int K = 100;
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
    std::cout << "Reading Data: " << file_path << std::endl;
    std::ifstream ifs;
    ifs.open(file_path, std::ios::binary);
    assert(ifs.is_open());
    uint32_t N; // num of points
    ifs.read((char *)&N, sizeof(uint32_t));
    data.resize(N);
    std::cout << "# of points: " << N << std::endl;
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
    std::cout << "Finish Reading Data" << endl;
}

/// @brief Reading output bin
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadOutputBin(const std::string &file_path,
                   const int num_queries,
                   std::vector<std::vector<uint32_t>> &data,
                   const int num_dimensions = 100)
{
    std::cout << "Reading output data: " << file_path << std::endl;
    std::cout << "# of points: " << num_queries << std::endl;

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
    std::cout << "Finish reading output data" << endl;
}

/// Copied from winning-solution/utils.h
/// @brief Calculate recall based on query results and ground truth information
float GetKNNRecall(const vector<vector<uint32_t>> &knns, const vector<vector<uint32_t>> &gt)
{
    std::vector<int> recalls(gt.size());
    assert(knns.size() == gt.size());

    uint64_t total_correct = 0;
    size_t nq = knns.size();
    size_t topk = knns[0].size();

    for (size_t i = 0; i < nq; i++)
    {
        size_t correct = 0;
        for (size_t j = 0; j < topk; j++)
        {
            for (size_t k = 0; k < topk; k++)
            {
                if (knns[i][k] == gt[i][j])
                {
                    correct++;
                    break;
                }
            }
        }
        recalls[i] = correct;
        total_correct += correct;
    }
    return (float)total_correct / nq / topk;
}

int main(int argc, char **argv)
{
    string source_path = std::string(argv[1]);
    string query_path = std::string(argv[2]);
    string output_path = std::string(argv[3]);

    auto start = std::chrono::high_resolution_clock::now();

    // Read nodes
    const uint32_t num_data_dimensions = 102;
    vector<vector<float>> nodes;
    ReadBin(source_path, num_data_dimensions, nodes);

    // Read queries
    uint32_t num_query_dimensions = num_data_dimensions + 2;
    vector<vector<float>> queries;
    ReadBin(query_path, num_query_dimensions, queries);

    // Generate ground truth and save to disk
    vector<vector<uint32_t>> knns;
    solve(nodes, queries, knns);
    SaveKNN(knns, output_path);

    // Read groud truth
    vector<vector<uint32_t>> gt_nodes;
    ReadOutputBin(output_path, queries.size(), gt_nodes);

    // Calculate recall
    float recall = GetKNNRecall(knns, gt_nodes);
    std::cout << "Recall: " << recall << "\n";

    auto end = std::chrono::high_resolution_clock::now();

    cout << "Total time: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s" << endl;
    return 0;
}