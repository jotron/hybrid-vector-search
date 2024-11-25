/**
 * Naive Single-Threaded C++ Implementation
 */

#include <algorithm>
#include <fstream>
#include <iostream>
#include <numeric>
#include <queue>
#include <chrono>
#include <memory>
#include <vector>
#include <ranges>
#include <cstring>
#include "hybrid_vector_search.h"
#include "io.h"
#include "distance_simd.h"

using namespace std;

class Index
{
private:
  const int num_nodes;
  const aligned_vector<float, 32> &node_vectors;
  const vector<float> &node_timestamps;
  const vector<int> &node_values;
  vector<int> node_idx_sorted_by_timestamp;
  vector<int> node_idx_sorted_by_value;

  std::function<bool(int, int)> cmp_value_l = [&](int i, int value)
  {
    return node_values[i] < value;
  };
  std::function<bool(int, int)> cmp_value_r = [&](int value, int i)
  {
    return node_values[i] > value;
  };
  std::function<bool(int, float)> cmp_timestamp_l = [&](int i, float timestamp)
  {
    return node_timestamps[i] < timestamp;
  };
  std::function<bool(float, int)> cmp_timestamp_r = [&](float timestamp, int i)
  {
    return node_timestamps[i] > timestamp;
  };
  std::function<bool(int, const pair<int, float> &)> cmp_value_timestamp_l = [&](int i, const pair<int, float> &value_timestamp)
  {
    return node_values[i] < value_timestamp.first || (node_values[i] == value_timestamp.first && node_timestamps[i] < value_timestamp.second);
  };
  std::function<bool(const pair<int, float> &, int)> cmp_value_timestamp_r = [&](const pair<int, float> &value_timestamp, int i)
  {
    return node_values[i] > value_timestamp.first || (node_values[i] == value_timestamp.first && node_timestamps[i] > value_timestamp.second);
  };

public:
  Index(const int num_nodes,
        const aligned_vector<float, 32> &node_vectors,
        const vector<float> &node_timestamps,
        const vector<int> &node_values)
      : num_nodes(num_nodes),
        node_vectors(node_vectors),
        node_timestamps(node_timestamps),
        node_values(node_values)
  {
    // Sort by value and timestamp
    node_idx_sorted_by_timestamp.resize(num_nodes);
    node_idx_sorted_by_value.resize(num_nodes);
    iota(node_idx_sorted_by_timestamp.begin(), node_idx_sorted_by_timestamp.end(), 0);
    iota(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), 0);
    std::sort(node_idx_sorted_by_timestamp.begin(), node_idx_sorted_by_timestamp.end(), [&](int i, int j)
              { return node_timestamps[i] < node_timestamps[j]; });
    std::sort(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), [&](int i, int j)
              { return node_values[i] < node_values[j] || (node_values[i] == node_values[j] && node_timestamps[i] < node_timestamps[j]); });
  }

  void search_type0(const float *query_vec, int query_index, vector<vector<uint32_t>> &result)
  {
    std::vector<std::pair<float, int32_t>> dummy_distances(K, make_pair(std::numeric_limits<float>::max(), -1));
    std::priority_queue<std::pair<float, uint32_t>> nearest_nodes(dummy_distances.begin(), dummy_distances.end());
    for (int j = 0; j < num_nodes; ++j)
    {
      const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
      float dist = avx2_l2_distance(node_vec, query_vec);
      if (nearest_nodes.top().first > dist)
      {
        nearest_nodes.pop();
        nearest_nodes.push(std::make_pair(dist, j));
      }
    }
    result[query_index].resize(K);
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = nearest_nodes.top();
      result[query_index][j] = res.second;
      nearest_nodes.pop();
    }
  };

  void search_type1(const float *query_vec, int query_value, int query_index, vector<vector<uint32_t>> &result)
  {
    std::vector<std::pair<float, int32_t>> dummy_distances(K, make_pair(std::numeric_limits<float>::max(), -1));
    std::priority_queue<std::pair<float, uint32_t>> nearest_nodes(dummy_distances.begin(), dummy_distances.end());
    int start_node = std::distance(node_idx_sorted_by_value.begin(), std::lower_bound(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), query_value, cmp_value_l));
    int end_node = std::distance(node_idx_sorted_by_value.begin(), std::upper_bound(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), query_value, cmp_value_r));
    for (int l = start_node; l < end_node; ++l)
    {
      int j = node_idx_sorted_by_value[l];
      const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
      float dist = avx2_l2_distance(node_vec, query_vec);
      if (nearest_nodes.top().first > dist)
      {
        nearest_nodes.pop();
        nearest_nodes.push(std::make_pair(dist, j));
      }
    }
    result[query_index].resize(K);
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = nearest_nodes.top();
      result[query_index][j] = res.second;
      nearest_nodes.pop();
    }
  }

  void search_type2(const float *query_vec, pair<float, float> timestamp, int query_index, vector<vector<uint32_t>> &result)
  {
    std::vector<std::pair<float, int32_t>> dummy_distances(K, make_pair(std::numeric_limits<float>::max(), -1));
    std::priority_queue<std::pair<float, uint32_t>> nearest_nodes(dummy_distances.begin(), dummy_distances.end());
    int start_node = std::distance(node_idx_sorted_by_timestamp.begin(),
                                   std::lower_bound(node_idx_sorted_by_timestamp.begin(), node_idx_sorted_by_timestamp.end(), timestamp.first, cmp_timestamp_l));
    int end_node = std::distance(node_idx_sorted_by_timestamp.begin(),
                                 std::upper_bound(node_idx_sorted_by_timestamp.begin(), node_idx_sorted_by_timestamp.end(), timestamp.second, cmp_timestamp_r));
    for (int l = start_node; l < end_node; ++l)
    {
      int j = node_idx_sorted_by_timestamp[l];
      const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
      float dist = avx2_l2_distance(node_vec, query_vec);
      if (nearest_nodes.top().first > dist)
      {
        nearest_nodes.pop();
        nearest_nodes.push(std::make_pair(dist, j));
      }
    }
    result[query_index].resize(K);
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = nearest_nodes.top();
      result[query_index][j] = res.second;
      nearest_nodes.pop();
    }
  }

  void search_type3(const float *query_vec, int query_value, pair<float, float> timestamp, int query_index, vector<vector<uint32_t>> &result)
  {
    std::vector<std::pair<float, int32_t>> dummy_distances(K, make_pair(std::numeric_limits<float>::max(), -1));
    std::priority_queue<std::pair<float, uint32_t>> nearest_nodes(dummy_distances.begin(), dummy_distances.end());
    int start_node = std::distance(node_idx_sorted_by_value.begin(),
                                   std::lower_bound(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), make_pair(query_value, timestamp.first),
                                                    cmp_value_timestamp_l));
    int end_node = std::distance(node_idx_sorted_by_value.begin(),
                                 std::upper_bound(node_idx_sorted_by_value.begin(), node_idx_sorted_by_value.end(), make_pair(query_value, timestamp.second),
                                                  cmp_value_timestamp_r));
    for (int l = start_node; l < end_node; ++l)
    {
      int j = node_idx_sorted_by_value[l];
      const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
      float dist = avx2_l2_distance(node_vec, query_vec);
      if (nearest_nodes.top().first > dist)
      {
        nearest_nodes.pop();
        nearest_nodes.push(std::make_pair(dist, j));
      }
    }
    result[query_index].resize(K);
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = nearest_nodes.top();
      result[query_index][j] = res.second;
      nearest_nodes.pop();
    }
  }
};

class Queries
{
  const int num_queries;
  aligned_vector<float, 32> query_vectors_sorted;
  vector<int> query_values_sorted;
  vector<pair<float, float>> query_timestamps_sorted;
  vector<int> query_indices;
  int start_for_type[4];

public:
  Queries(int num_queries,
          const aligned_vector<float, 32> &query_vectors,
          const vector<int> &query_types,
          const vector<int> &query_values,
          const vector<pair<float, float>> query_timestamps)
      : num_queries(num_queries)
  {
    // Sort queries by type & value to optimze locality and make measurement by type easier
    // Potential Improvement: Compress Queries
    query_indices.resize(num_queries);
    iota(query_indices.begin(), query_indices.end(), 0);
    std::sort(query_indices.begin(), query_indices.end(), [&](int i, int j)
              { return query_types[i] < query_types[j] || (query_types[i] == query_types[j] && query_values[i] < query_values[j]); });
    query_timestamps_sorted.resize(num_queries);
    query_values_sorted.resize(num_queries);
    query_vectors_sorted.resize(query_vectors.size());
    for (int i = 0; i < num_queries; i++)
    {
      query_timestamps_sorted[i] = query_timestamps[query_indices[i]];
      query_values_sorted[i] = query_values[query_indices[i]];
      memcpy(&query_vectors_sorted[i * VECTOR_DIM_PADDED], &query_vectors[query_indices[i] * VECTOR_DIM_PADDED], VECTOR_DIM_PADDED * sizeof(float));
      if (!start_for_type[query_types[query_indices[i]]])
      {
        start_for_type[query_types[query_indices[i]]] = i;
      }
    }
  }

  void solve_with(Index &index,
                  vector<vector<uint32_t>> &result)
  {
    // Potential improvement: Optimize write locality of result

    auto start0 = std::chrono::high_resolution_clock::now();

#pragma omp parallel for schedule(dynamic, 1000)
    for (int i = 0; i < start_for_type[1]; i++)
    {
      index.search_type0(&query_vectors_sorted[i * VECTOR_DIM_PADDED], query_indices[i], result);
    }
    auto start1 = std::chrono::high_resolution_clock::now();
    cout << "Type 0 queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(start1 - start0).count() << "ms" << endl;

#pragma omp parallel for schedule(dynamic, 1000)
    for (int i = start_for_type[1]; i < start_for_type[2]; i++)
    {
      index.search_type1(&query_vectors_sorted[i * VECTOR_DIM_PADDED], query_values_sorted[i], query_indices[i], result);
    }
    auto start2 = std::chrono::high_resolution_clock::now();
    cout << "Type 1 queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1).count() << "ms" << endl;

#pragma omp parallel for schedule(dynamic, 1000)
    for (int i = start_for_type[2]; i < start_for_type[3]; i++)
    {
      index.search_type2(&query_vectors_sorted[i * VECTOR_DIM_PADDED], query_timestamps_sorted[i], query_indices[i], result);
    }
    auto start3 = std::chrono::high_resolution_clock::now();
    cout << "Type 2 queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2).count() << "ms" << endl;

#pragma omp parallel for schedule(dynamic, 1000)
    for (int i = start_for_type[3]; i < num_queries; i++)
    {
      index.search_type3(&query_vectors_sorted[i * VECTOR_DIM_PADDED], query_values_sorted[i], query_timestamps_sorted[i], query_indices[i], result);
    }
    auto start4 = std::chrono::high_resolution_clock::now();
    cout << "Type 3 queries: " << std::chrono::duration_cast<std::chrono::milliseconds>(start4 - start3).count() << "ms" << endl;
  }
};

void solve_with_algined_input(const int num_nodes,
                              const aligned_vector<float, 32> &node_vectors,
                              const vector<float> &node_timestamps,
                              const vector<int> &node_values,
                              const int num_queries,
                              const aligned_vector<float, 32> &query_vectors,
                              const vector<int> &query_types,
                              const vector<int> &query_values,
                              const vector<pair<float, float>> query_timestamps,
                              vector<vector<uint32_t>> &result)
{
  // Make Index
  auto start0 = std::chrono::high_resolution_clock::now();
  Index index(num_nodes, node_vectors, node_timestamps, node_values);
  auto start1 = std::chrono::high_resolution_clock::now();
  cout << "Index construction overhead: " << std::chrono::duration_cast<std::chrono::milliseconds>(start1 - start0).count() << "ms" << endl;

  // Preprocess Queries
  Queries queries(num_queries, query_vectors, query_types, query_values, query_timestamps);
  auto start2 = std::chrono::high_resolution_clock::now();
  cout << "Query preprocessing overhead: " << std::chrono::duration_cast<std::chrono::milliseconds>(start2 - start1).count() << "ms" << endl;

  // Compute
  queries.solve_with(index, result);
  auto start3 = std::chrono::high_resolution_clock::now();
  cout << "Total calculation time: " << std::chrono::duration_cast<std::chrono::milliseconds>(start3 - start2).count() << "ms" << endl;
}

void solve(std::string &data_path, std::string &queries_path, vector<vector<uint32_t>> &knn)
{
  auto start = std::chrono::high_resolution_clock::now();
  // Read nodes
  const uint32_t num_data_dimensions = 102;
  vector<vector<float>> nodes;
  ReadBin(data_path, num_data_dimensions, nodes);

  // Read queries
  uint32_t num_query_dimensions = num_data_dimensions + 2;
  vector<vector<float>> queries;
  ReadBin(queries_path, num_query_dimensions, queries);

  // Must align vectors to 32 byte for efficient 8x SIMD
  // Extend size of vectors from 100 to 104 so that consecutive vectors are aligned.
  int num_nodes = nodes.size();
  int num_queries = queries.size();
  aligned_vector<float, 32> node_vectors(num_nodes * VECTOR_DIM_PADDED, 0);
  aligned_vector<float, 32> query_vectors(num_queries * VECTOR_DIM_PADDED, 0);

  vector<float> node_timestamps(num_nodes);
  vector<int> node_values(num_nodes);
  vector<int> query_types(num_queries);
  vector<int> query_values(num_queries);
  vector<pair<float, float>> query_timestamps(num_queries);

  for (int i = 0; i < num_nodes; i++)
  {
    node_timestamps[i] = nodes[i][1];
    node_values[i] = nodes[i][0];
    for (int j = 0; j < VECTOR_DIM; j++)
    {
      node_vectors[i * VECTOR_DIM_PADDED + j] = nodes[i][2 + j];
    }
  }
  for (int i = 0; i < num_queries; i++)
  {
    query_types[i] = queries[i][0];
    query_values[i] = queries[i][1];
    query_timestamps[i] = make_pair(queries[i][2], queries[i][3]);
    for (int j = 0; j < VECTOR_DIM; j++)
    {
      query_vectors[i * VECTOR_DIM_PADDED + j] = queries[i][4 + j];
    }
  }
  knn.resize(num_queries);

  auto end = std::chrono::high_resolution_clock::now();
  cout << "Reading input overhead: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;

  solve_with_algined_input(num_nodes, node_vectors, node_timestamps, node_values, num_queries, query_vectors, query_types, query_values, query_timestamps, knn);
}