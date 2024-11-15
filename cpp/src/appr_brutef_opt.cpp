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
#include "hybrid_vector_search.h"
#include "io.h"
#include "distance_simd.h"

using namespace std;

/**
 * @param vectors: array of size VECTOR_DIM*num_nodes
 */
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
  result.resize(num_queries);

  for (int i = 0; i < num_queries; i++)
  {
    // Keep track of K nearest nodes so far
    std::vector<std::pair<float, int32_t>> dummy_distances(K, make_pair(std::numeric_limits<float>::max(), -1));
    std::priority_queue<std::pair<float, uint32_t>> nearest_nodes(dummy_distances.begin(), dummy_distances.end());

    int query_type = query_types[i];
    const float *query_vec = &query_vectors[i * VECTOR_DIM_PADDED];
    int query_value = query_values[i];
    float query_timestamp_l = query_timestamps[i].first;
    float query_timestamp_r = query_timestamps[i].second;

    switch (query_type)
    {
    // Vector Search
    case 0:
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
      break;
    // Vector Search with value constraint
    case 1:
      for (int j = 0; j < num_nodes; ++j)
      {
        const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
        int node_value = node_values[j];
        if (query_value == node_value)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          if (nearest_nodes.top().first > dist)
          {
            nearest_nodes.pop();
            nearest_nodes.push(std::make_pair(dist, j));
          }
        }
      }
      break;
    // Vector Search with timestamp constraint
    case 2:
      for (int j = 0; j < num_nodes; ++j)
      {
        const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
        float node_timestamp = node_timestamps[j];
        if (node_timestamp >= query_timestamp_l && node_timestamp <= query_timestamp_r)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          if (nearest_nodes.top().first > dist)
          {
            nearest_nodes.pop();
            nearest_nodes.push(std::make_pair(dist, j));
          }
        }
      }
      break;
    // Vector Search with value and timestamp constraint
    case 3:
      for (int j = 0; j < num_nodes; ++j)
      {
        const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
        int node_value = node_values[j];
        float node_timestamp = node_timestamps[j];
        if (query_value == node_value && node_timestamp >= query_timestamp_l && node_timestamp <= query_timestamp_r)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          if (nearest_nodes.top().first > dist)
          {
            nearest_nodes.pop();
            nearest_nodes.push(std::make_pair(dist, j));
          }
        }
      }
      break;
    }

    result[i].resize(K);
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = nearest_nodes.top();
      result[i][j] = res.second;
      nearest_nodes.pop();
    }
    if (i % 1000 == 0)
      cout << "Processed " << i << "/" << num_queries << " queries\n";
  }
}

void solve(std::string &data_path, std::string &queries_path, vector<vector<uint32_t>> &knn)
{
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
  auto start = std::chrono::high_resolution_clock::now();
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
  auto end = std::chrono::high_resolution_clock::now();
  cout << "Preprocessing overhead: " << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << endl;

  solve_with_algined_input(num_nodes, node_vectors, node_timestamps, node_values, num_queries, query_vectors, query_types, query_values, query_timestamps, knn);
}