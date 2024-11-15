# hybrid-vector-search

The winners of the ACM SIGMOD 2024 Programming Contest uploaded their approach [here](https://github.com/KevinZeng08/sigmod-2024-contest).
In this project we try to understand how performance is achieved by benchmarking different approaches. 

## Running the Rust Code

To execute code under /rust
```
// 1. Clone repository hybrid-vector-search
// 2. Place data under hybrid-vector-search/data, i.e. hybrid-vector-search/data/dummy-data.bin, hybrid-vector-search/data/dummy-queries.bin
// 3. Run the below
cd hybrid-vector-search/rust
cargo run --bin <BINARY> --release  <DATA_PATH> <QUERIES_PATH> [<GROUND_TRUTH_PATH>]
// e.g. cargo run --bin brute_force --release ../data/dummy-data.bin ../data/dummy-queries.bin
// e.g. cargo run --bin brute_force --release ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin
// e.g. cargo run --bin brute_force --release ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin
```

I also implemented a version leveraging the NGT library.
[NGT](https://github.com/yahoojapan/NGT) is a high-performance vector search library. 
Since NGT leverages OpenMP, a working OpenMP installation is a prerequisite.
[ngt-rs](https://github.com/lerouxrgd/ngt-rs?tab=readme-ov-file) provides rust bindings to the library. 
Documentation can be found [here](https://docs.rs/ngt/latest/ngt/).
ngt-rs uses [bindgen](https://github.com/rust-lang/rust-bindgen) to automatically generate rust bindings for C++. 
Since bindgen uses libclang to parse C++ header files a working a clang environment is a prerequisite too.


To get libclang without installing it system-wide run:
1. Follow https://clang.llvm.org/get_started.html to build llvm-project froum source
2. Add the targets to path: `export PATH=$PATH:<working_dir>/llvm-project/build/bin/`

This should suffice to run:

```
cd rust-with-ngt
cargo run --release ../data/dummy-data.bin ../data/dummy-queries.bin 
cargo run --release  ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin
```

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
./build/sorted ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin

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
| C++ With Sorting          | 10s     |


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



