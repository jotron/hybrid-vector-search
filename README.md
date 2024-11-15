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

## Running the CPP Code

The code uses OpenMP for data-parellism. A working OpenMP installation is a prerequisite.
Some of the versions further leverage SIMD AVX instructions.

### Mac OSX
Getting OpenMP to work on macOS is not trivial. 
Apple disables OpenMP in the clang version shipped by default. 
A fix is using clang via *brew install llvm* with ompiler arguments *-fexperimental-library -stdlib=libc++*.

The build script detects if the platform is macOS and performs most of the above automatically. Manually one only needs to run *brew install llvm* first.
If the detected platform is macOS the build script does not generate versions requiring AVX.

### Workflow

```
mkdir cpp/build
cd cpp/build
cmake ..
make
cd ../
./build/brutef ../data/dummy-data.bin ../data/dummy-queries.bin 
./build/brutef_opt ../data/dummy-data.bin ../data/dummy-queries.bin 
./build/brutef_opt_omp ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin
./build/brutef_force_omp ../data/contest-data-release-10m.bin ../data/contest-queries-release-10m.bin

// To generate ground truth
./build/brutef ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin --overwriteOutput
// To calculate recall in comparison to ground truth
./build/brutef ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin
```

## Performance Measurements

Details can be found in the report. For intuition, on a powerful machine I measure the following for the 1M dataset: 

| Version                   | Runtime |
| ------------------------- | ------- |
| C++ Naive Brute Force     | 420s    |
| C++ Opt. Brute no SIMD    | 229s    |
| C++ Opt. Brute            | 146s    |
| C++ With OpenMP           | 10s     |
| C++ With Sorting          | 10s     |



