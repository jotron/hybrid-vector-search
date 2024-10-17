cmake_minimum_required (VERSION 3.12)

# To enable OpenMP on MacOSX
set(OpenMP_CXX_FLAGS "-fopenmp -fexperimental-library -stdlib=libc++")
set(OpenMP_CXX_LIB_NAMES "libomp")
set(OpenMP_libomp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/include")
link_directories(${OPENMP_LIBRARIES})

project (HybridVectorSearch)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Code intelligence use

# To enable OpenMP
find_package(OpenMP REQUIRED)
if (OPENMP_FOUND)
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

add_executable(brute_force 
    src/brute_force.cpp 
     src/main.cpp)

add_executable(brute_force_opt 
    src/brute_force_opt.cpp 
    src/main.cpp)

add_executable(brute_force_omp 
    src/brute_force_omp.cpp 
    src/main.cpp)