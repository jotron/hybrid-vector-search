cmake_minimum_required (VERSION 3.12)
project (HybridVectorSearch)

set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -DNDEBUG -O3 -march=native -mtune=native -fopenmp -fopenmp-simd")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Code intelligence use

add_executable(data data.cpp )