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
#include "distance.h"

using namespace std;

// Single-Threaded brute force
void solve(const vector<vector<float>> &nodes, const vector<vector<float>> &queries, vector<vector<uint32_t>> &gt)
{
  // brute force to get ground truth
  uint32_t n = nodes.size();
  uint32_t d = nodes[0].size() - 2; // skip first 2 dimensions
  uint32_t nq = queries.size();
  gt.resize(nq);

  for (size_t i = 0; i < nq; i++)
  {
    uint32_t query_type = queries[i][0];
    int32_t v = queries[i][1];
    float l = queries[i][2];
    float r = queries[i][3];
    const float *query_vec = queries[i].data() + 4;

    std::priority_queue<std::pair<float, uint32_t>> pq;

    for (uint32_t j = 0; j < n; ++j)
    {
      const float *base_vec = nodes[j].data() + 2;
      int32_t bv = nodes[j][0];
      float bt = nodes[j][1];

      if (query_type == 0)
      {
        float dist = normal_l2(base_vec, query_vec, d);
        pq.push(std::make_pair(-dist, j));
      }
      else if (query_type == 1)
      {
        if (v == bv)
        {
          float dist = normal_l2(base_vec, query_vec, d);
          pq.push(std::make_pair(-dist, j));
        }
      }
      else if (query_type == 2)
      {
        if (bt >= l && bt <= r)
        {
          float dist = normal_l2(base_vec, query_vec, d);
          pq.push(std::make_pair(-dist, j));
        }
      }
      else if (query_type == 3)
      {
        if (v == bv && bt >= l && bt <= r)
        {
          float dist = normal_l2(base_vec, query_vec, d);
          pq.push(std::make_pair(-dist, j));
        }
      }
    }

    gt[i].resize(K);
    if (pq.size() < K)
    {
      cout << "id: " << i << endl;
      cout << "query type: " << query_type << " v: " << v << " l: " << l << " r: " << r << endl;
      cout << "K: " << pq.size() << endl;
    }
    for (int j = K - 1; j >= 0; j--)
    {
      std::pair<float, uint32_t> res = pq.top();
      gt[i][j] = res.second;
      pq.pop();
    }

    if (i % 1000 == 0)
      cout << "Processed " << i << "/" << nq << " queries\n";
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

  // Calculate
  solve(nodes, queries, knn);
}