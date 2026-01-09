# Strategy 5 - Grid over Queries + Dimension Reduction

Grid maps to queries, threads perform parallel reduction over dimensions for each database vector.

## Files

### Source Files
- **`estrategia5_cosine.cu`** - Cosine similarity with reduction
- **`estrategia5_euclidean.cu`** - Euclidean distance with reduction
- **`estrategia5_pearson.cu`** - Pearson correlation with reduction
- **`estrategia5_pearson_opt.cu`** - Optimized Pearson with reduction

### Executables
- **`estrategia5_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia5_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia5_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia5_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

## Strategy

- **Grid:** 1D (num_queries blocks)
- **Threads per block:** Configurable (32-1024, typically 256-512)
- **Parallelization:** 1 block = 1 query, threads reduce over dimensions for each database vector
- **Best for:** Moderate queries, large database, high dimensions

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia5_cosine.cu -o estrategia5_cosine
```

## Run

```bash
./estrategia5_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 256
```

**Note:** Third parameter is block size (try 128, 256, 512, 1024).
