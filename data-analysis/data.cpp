#include <fstream>
#include <iostream>
#include <numeric>
#include <memory>
#include <string>
#include <vector>

using namespace std;

/// @brief Reading binary data vectors. Raw data store as a (N x dim)
/// @param file_path file path of binary data
/// @param data returned 2D data vectors
void ReadBin(const std::string &file_path,
             const int num_dimensions,
             std::vector<std::vector<float>> &data)
{
  std::cout << "Reading data: " << file_path << std::endl;
  std::ifstream ifs;
  ifs.open(file_path, std::ios::binary);
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

template <typename T>
void exportToCSV(const std::vector<T> &data, const std::string &filename)
{
  std::ofstream file(filename);

  if (file.is_open())
  {
    for (size_t i = 0; i < data.size(); ++i)
    {
      file << data[i];
      if (i < data.size() - 1)
      {
        file << ","; // Add comma between values
      }
    }
    file << std::endl; // End the line after all data

    file.close();
    std::cout << "Data exported successfully to " << filename << std::endl;
  }
  else
  {
    std::cerr << "Could not open file " << filename << " for writing." << std::endl;
  }
}

int main(int argc, char **argv)
{
  string source_path = std::string(argv[1]);
  string query_path = std::string(argv[2]);

  // Read nodes
  const uint32_t num_data_dimensions = 102;
  vector<vector<float>> nodes;
  ReadBin(source_path, num_data_dimensions, nodes);

  // Read queries
  uint32_t num_query_dimensions = num_data_dimensions + 2;
  vector<vector<float>> queries;
  ReadBin(query_path, num_query_dimensions, queries);

  int num_nodes = nodes.size();
  int num_queries = queries.size();

  vector<float> node_timestamps(num_nodes);
  vector<int> node_values(num_nodes);
  vector<int> query_types(num_queries);
  vector<int> query_values(num_queries);
  vector<float> query_timestamps_l(num_queries);
  vector<float> query_timestamps_r(num_queries);

  for (int i = 0; i < num_nodes; i++)
  {
    node_timestamps[i] = nodes[i][1];
    node_values[i] = nodes[i][0];
  }
  for (int i = 0; i < num_queries; i++)
  {
    query_types[i] = queries[i][0];
    query_values[i] = queries[i][1];
    query_timestamps_l[i] = queries[i][2];
    query_timestamps_r[i] = queries[i][3];
  }

  exportToCSV(node_timestamps, "node_timestamps.csv");
  exportToCSV(node_values, "node_values.csv");
  exportToCSV(query_values, "query_values.csv");
  exportToCSV(query_types, "query_types.csv");
  exportToCSV(query_timestamps_l, "query_timestamps_l.csv");
  exportToCSV(query_timestamps_r, "query_timestamps_r.csv");

  return 0;
}