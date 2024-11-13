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