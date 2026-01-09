# Strategy 3 - Grid over Big Group (Database)

Grid maps to database vectors, threads loop over queries.

## Files

### Source Files
- **`estrategia3_cosine.cu`** - Cosine similarity implementation
- **`estrategia3_euclidean.cu`** - Euclidean distance implementation
- **`estrategia3_pearson.cu`** - Pearson correlation implementation
- **`estrategia3_pearson_opt.cu`** - Optimized Pearson implementation

### Executables
- **`estrategia3_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia3_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia3_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia3_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

### Documentation
- **`STRATEGY_DESCRIPTION.md`** - Detailed strategy explanation

## Strategy

- **Grid:** 1D (num_database blocks)
- **Threads per block:** Configurable (32-1024, typically 128-256)
- **Parallelization:** 1 block = 1 database vector, threads loop over queries
- **Best for:** Large number of queries, smaller database

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia3_cosine.cu -o estrategia3_cosine
```

## Run

```bash
./estrategia3_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 128
```

**Note:** Third parameter is block size (try 32, 64, 128, 256, 512, 1024).
