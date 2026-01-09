# Strategy 6 - Grid over Database + Dimension Reduction

Grid maps to database vectors, threads perform parallel reduction over dimensions for each query.

## Files

### Source Files
- **`estrategia6_cosine.cu`** - Cosine similarity with reduction
- **`estrategia6_euclidean.cu`** - Euclidean distance with reduction
- **`estrategia6_pearson.cu`** - Pearson correlation with reduction
- **`estrategia6_pearson_opt.cu`** - Optimized Pearson with reduction

### Executables
- **`estrategia6_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia6_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia6_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia6_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

## Strategy

- **Grid:** 1D (num_database blocks)
- **Threads per block:** Configurable (32-1024, typically 256-512)
- **Parallelization:** 1 block = 1 database vector, threads reduce over dimensions for each query
- **Best for:** Large queries, moderate database, high dimensions
- **Performance:** Usually better than Strategy 5 for typical workloads

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia6_cosine.cu -o estrategia6_cosine
```

## Run

```bash
./estrategia6_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 256
```

**Note:** Third parameter is block size (try 128, 256, 512, 1024).
