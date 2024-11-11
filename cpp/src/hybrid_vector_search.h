/**
 *  Interface for different versions to implement
 */
#include <vector>
using std::vector;

#define VECTOR_DIM 100
#define K 100

/**
 * @param nodes: nodes encoded as 102 size arrays. node[0] = value, node[1] = timestamp, node[2..] = vector
 * @param queries: queries encoded as 104 size arrays.
 * @out knn
 */
void solve(const vector<vector<float>> &nodes, const vector<vector<float>> &queries, vector<vector<uint32_t>> &knn);
