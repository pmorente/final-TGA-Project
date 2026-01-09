# Strategy 1 - Grid over Pairs (Baseline)

One thread per vector comparison. Simple but slow baseline implementation.

## Files

### Source Files
- **`estrategia1_cosine.cu`** - Cosine similarity implementation
- **`estrategia1_euclidean.cu`** - Euclidean distance implementation
- **`estrategia1_pearson.cu`** - Pearson correlation implementation
- **`estrategia1_pearson_opt.cu`** - Optimized Pearson implementation

### Executables
- **`estrategia1_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia1_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia1_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia1_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

## Strategy

- **Grid:** 2D (num_queries × num_database)
- **Threads per block:** 1
- **Parallelization:** 1 thread = 1 comparison (sequential dimension loop)
- **Best for:** Reference/baseline only

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia1_cosine.cu -o estrategia1_cosine
```

## Run

```bash
./estrategia1_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin
```
