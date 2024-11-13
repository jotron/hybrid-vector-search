cmake_minimum_required (VERSION 3.12)
project (HybridVectorSearch)

# MAC OSX
if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
    set(OpenMP_CXX_FLAGS "-fopenmp -fexperimental-library -stdlib=libc++")
    set(OpenMP_CXX_LIB_NAMES "libomp")
    set(OpenMP_libomp_LIBRARY "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(OPENMP_LIBRARIES "/opt/homebrew/opt/llvm/lib")
    set(OPENMP_INCLUDES "/usr/local/Cellar/llvm/include")
    link_directories(${OPENMP_LIBRARIES})

    # To enable OpenMP
    find_package(OpenMP REQUIRED)
    if (OPENMP_FOUND)
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    endif()

    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -DNDEBUG -O3 -march=native -mtune=native")

# LINUX
else()
    set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -DNDEBUG -O3 -march=native -mtune=native -fopenmp -fopenmp-simd -mfma -mavx2 -ftree-vectorize")
endif()

set(OMP_NUM_THREADS "40")
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON) # Code intelligence use

add_executable(brutef 
    src/appr_brutef.cpp 
    src/main.cpp
    src/io.cpp)

add_executable(brutef_opt_nosimd 
    src/appr_brutef_opt_nosimd.cpp 
    src/main.cpp
    src/io.cpp)

# Require AVX
if (NOT (${CMAKE_SYSTEM_NAME} MATCHES "Darwin"))
    add_executable(brutef_opt 
    src/appr_brutef_opt.cpp 
    src/main.cpp
    src/io.cpp)

    add_executable(brutef_opt_omp
    src/appr_brutef_opt_omp.cpp 
    src/main.cpp
    src/io.cpp)

    add_executable(sorted
    src/appr_sorted.cpp 
    src/main.cpp
    src/io.cpp)
endif()