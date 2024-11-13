#pragma once
#include <vector>

/**
 *  Global Parameters
 */
#define VECTOR_DIM 100
#define VECTOR_DIM_PADDED 104
#define K 100

/**
 * Interface for Implementations
 * @out knn
 */
void solve(std::string &data_path, std::string &queries_path, std::vector<std::vector<uint32_t>> &knn);
