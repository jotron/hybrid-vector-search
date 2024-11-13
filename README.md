# hybrid-vector-search

The winners of the ACM SIGMOD 2024 Programming Contest uploaded their approach [here](https://github.com/KevinZeng08/sigmod-2024-contest).
In this project we try to understand how performance is achieved by benchmarking different approaches. 
While special focus is made on the approach of Xiang et al. we do not limit ourselves to only trying approaches they implemented.
We further explore how good of a performance we can with Rust without using external libraries at reasonable complexity code.

## Use

To execute the cpp baseline on the dummy dataset execute.
```
mkdir cpp/build
cd cpp/build
cmake ..
make
cd ../
./build/brute_force ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin
./build/brute_force ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
./build/brutef_opt ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
./build/brutef_opt_omp ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
./build/brutef_opt_nosimd ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
```

To execute rust
```
cd rust
export PATH=$PATH:~/hybrid-vector-search/llvm-project/build/bin/
cargo run ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin
cargo run --release  ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
```

## Performance Measurements for 1M

Measure execution time excluding IO.
Using the 1M dataset with 1_000_000 nodes (~400MB) and 10_000 queries on the IDI machine.

| Version                   | Runtime |
| ------------------------- | ------- |
| C++ Naive Brute Force     | 420s    |
| C++ Opt. Brute no SIMD    | 229s    |
| C++ Opt. Brute Force      | 146s    |
| C++ With OpenMP           | 10s     |
| C++ With Sorting          | 20s     |


- **C++ Naive Brute Force**

  Compiler-Optimized, Single-Threaded, No SIMD, No Sorting

- **C++ Optimized Brute Force**
  
  Single-Threaded, but otherwise all optimizations one can think of, i.e. Sorting, SIMD, ILP

- **C++ OpenMP Parellelized Brute Force**
  
  Like above but with 40 Threads.
  
- Rust Naive Brute Force

- Rust Parallelized
  ***28s***

- Rust with NGT
  ***42s***

- Contest Winners Implementation
  ***1s***

## Performance Measurements for 10M

Using the 10M dataset with 10_000_000 nodes and 100_000 queries on the IDI machine.



