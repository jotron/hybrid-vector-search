/**
 * Tuned Single-Threaded C++ Implementation
 */

#include <algorithm>
#include <iostream>
#include <numeric>
#include <queue>
#include <memory>
#include <chrono>
#include <emmintrin.h>
#include <immintrin.h>
#include <xmmintrin.h>
#include <cstring>
#include "util.h"
#include "hybrid_vector_search.h"

#define VECTOR_DIM_PADDED 104

using namespace std;

inline float avx2_l2_distance(const float *a, const float *b)
{
  unsigned dim = 104;
  __m256 sum = _mm256_setzero_ps(); // Initialize sum to 0
  unsigned i;
  for (i = 0; i + 7 < dim; i += 8)
  {
    __m256 a_vec = _mm256_load_ps(&a[i]);      // Load 8 floats from a
    __m256 b_vec = _mm256_load_ps(&b[i]);      // Load 8 floats from b
    __m256 diff = _mm256_sub_ps(a_vec, b_vec); // Calculate difference
    sum = _mm256_fmadd_ps(diff, diff, sum);    // Calculate sum of squares
  }
  float result = 0;
  for (unsigned j = 0; j < 8; ++j)
  {
    result += ((float *)&sum)[j];
  }
  return result; // Return square root of sum
}

inline float normal_l2(float const *a, float const *b)
{
  unsigned dim = 104;
  float r = 0;
  for (unsigned i = 0; i < dim; ++i)
  {
    float v = float(a[i]) - float(b[i]);
    v *= v;
    r += v;
  }
  return r;
}

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
    std::priority_queue<std::pair<float, uint32_t>> pq;
    int query_type = query_types[i];
    const float *query_vec = &query_vectors[i * VECTOR_DIM_PADDED];
    int query_value = query_values[i];
    float query_timestamp_l = query_timestamps[i].first;
    float query_timestamp_r = query_timestamps[i].second;

    for (int j = 0; j < num_nodes; ++j)
    {
      const float *node_vec = &node_vectors[j * VECTOR_DIM_PADDED];
      int node_value = node_values[j];
      float node_timestamp = node_timestamps[j];
      if (query_type == 0)
      {
        float dist = avx2_l2_distance(node_vec, query_vec);
        pq.push(std::make_pair(-dist, j));
      }
      else if (query_type == 1)
      {
        if (query_value == node_value)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          pq.push(std::make_pair(-dist, j));
        }
      }
      else if (query_type == 2)
      {
        if (node_timestamp >= query_timestamp_l && node_timestamp <= query_timestamp_r)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          pq.push(std::make_pair(-dist, j));
        }
      }
      else if (query_type == 3)
      {
        if (query_value == node_value && node_timestamp >= query_timestamp_l && node_timestamp <= query_timestamp_r)
        {
          float dist = avx2_l2_distance(node_vec, query_vec);
          pq.push(std::make_pair(-dist, j));
        }
      }
    }

    result[i].resize(K);
    if (pq.size() < K)
    {
      cout << "id: " << i << endl;
      cout << "query type: " << query_type << " v: " << query_value << " l: " << query_timestamp_l << " r: " << query_timestamp_r << endl;
      cout << "K: " << pq.size() << endl;
    }
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = pq.top();
      result[i][j] = res.second;
      pq.pop();
    }

    if (i % 1000 == 0)
      cout << "Processed " << i << "/" << num_queries << " queries\n";
  }
}

void solve(const vector<vector<float>> &nodes, const vector<vector<float>> &queries, vector<vector<uint32_t>> &result)
{
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

  solve_with_algined_input(num_nodes, node_vectors, node_timestamps, node_values, num_queries, query_vectors, query_types, query_values, query_timestamps, result);
}