# hybrid-vector-search

To execute the cpp baseline on the dummy dataset execute.
```
mkdir cpp/build
cd cpp/build
cmake ..
make
cd ../
./build/hybrid_vector_search ../data/dummy-data.bin ../data/dummy-queries.bin
```


To execute rust
```
cd rust
cargo run ../data/dummy-data.bin ../data/dummy-queries.bin ../data/dummy-gt.bin
cargo run --release  ../data/contest-data-release-1m.bin ../data/contest-queries-release-1m.bin ../data/1m-gt.bin
```

## Performance Measurements

