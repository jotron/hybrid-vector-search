#pragma once
#include <fstream>
#include <iostream>
#include <vector>
#include <queue>
#include <memory>
#include <assert.h>
#include <cstring>
#include <algorithm>
#include <numeric>
#include <unordered_map>
#include "distance.h"

using std::cout;
using std::endl;
using std::priority_queue;
using std::string;
using std::unique_ptr;
using std::unordered_map;
using std::vector;

using PII = std::pair<uint32_t, uint32_t>; // <start_id, num_points>

#include <queue>

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