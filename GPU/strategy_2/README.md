# Strategy 2 - Grid over Small Group (Queries)

Grid maps to queries, threads loop over database vectors.

## Files

### Source Files
- **`estrategia2_cosine.cu`** - Cosine similarity implementation
- **`estrategia2_euclidean.cu`** - Euclidean distance implementation
- **`estrategia2_pearson.cu`** - Pearson correlation implementation
- **`estrategia2_pearson_opt.cu`** - Optimized Pearson implementation

### Executables
- **`estrategia2_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia2_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia2_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia2_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

## Strategy

- **Grid:** 1D (num_queries blocks)
- **Threads per block:** Configurable (32-1024, typically 128-256)
- **Parallelization:** 1 block = 1 query vector, threads loop over database
- **Best for:** Small number of queries, large database

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia2_cosine.cu -o estrategia2_cosine
```

## Run

```bash
./estrategia2_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 128
```

**Note:** Third parameter is block size (try 32, 64, 128, 256, 512, 1024).
