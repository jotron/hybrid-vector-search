#include <fstream>
#include <iostream>
#include <numeric>
#include <chrono>
#include <memory>
#include <string>
#include <vector>
#include <assert.h>
#include "hybrid_vector_search.h"
#include "io.h"
#include "util.h"

using namespace std;

int main(int argc, char **argv)
{
    if (argc < 3)
    {
        cout << "Invalid Args" << endl;
        return 1;
    }
    string source_path = std::string(argv[1]);
    string query_path = std::string(argv[2]);
    string output_path = "";
    string option = "";
    if (argc > 3)
        output_path = std::string(argv[3]);
    if (argc > 4)
        option = std::string(argv[4]);

    // Call Implementation
    cout << "Hybrid Vector Search on data=" << source_path << ", queries=" << query_path << endl;
    auto start = std::chrono::high_resolution_clock::now();
    vector<vector<uint32_t>> knns;
    solve(source_path, query_path, knns);
    auto end = std::chrono::high_resolution_clock::now();
    cout << "Processed " << knns.size() << " queries" << endl;
    cout << "Total time: " << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << "ms" << endl;

    if (output_path != "")
    {
        if (option == "--overwriteOutput")
        {
            SaveKNN(knns, output_path);
            cout << "Wrote ground truth to " << output_path << endl;
        }

        // Read groud truth
        vector<vector<uint32_t>> gt_nodes;
        ReadOutputBin(output_path, knns.size(), gt_nodes);

        // Calculate recall
        float recall = GetKNNRecall(knns, gt_nodes);
        std::cout << "Recall: " << recall << "\n";
    }

    return 0;
}