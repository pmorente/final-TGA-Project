# Strategy 7 - 2D Tiled with Dimension Reduction (Best Practice)

2D tiling strategy with shared memory optimization and parallel dimension reduction.

## Files

### Source Files
- **`estrategia7_cosine.cu`** - Cosine similarity with 2D tiling
- **`estrategia7_euclidean.cu`** - Euclidean distance with 2D tiling
- **`estrategia7_pearson.cu`** - Pearson correlation with 2D tiling
- **`estrategia7_*_temp.cu`** - Template/experimental versions

### Executables
- **`estrategia7_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia7_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia7_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia7_*_temp`** / **`.exe`** - Template/experimental executables

### Result Files
- **`gpu_results_*_384.bin`** - Results for 384-dimensional vectors
- **`gpu_results_*_768.bin`** - Results for 768-dimensional vectors
- **`gpu_results_*_1024.bin`** - Results for 1024-dimensional vectors

## Strategy

- **Grid:** 2D (ceil(num_queries/TILE_A) × ceil(num_database/TILE_B))
- **Threads per block:** Configurable (256-1024 recommended)
- **Parallelization:** 
  - Each block processes TILE_A × TILE_B comparisons
  - Threads reduce over dimensions within each comparison
  - Shared memory for data reuse
- **Best for:** Most scenarios, especially large datasets
- **Performance:** Typically the fastest strategy

## Compile

```bash
nvcc -O3 -arch=sm_86 estrategia7_cosine.cu -o estrategia7_cosine
```

## Run

```bash
./estrategia7_cosine ../../inputs/384/file_a.bin ../../inputs/384/file_b.bin 30
```

**Note:** Third parameter is tile size (try 4, 8, 16, 30, 31, 32). Optimal is usually 30-32.

## Tile Size Selection

- **Small tiles (4-8):** Less shared memory, more blocks
- **Medium tiles (16):** Balanced approach
- **Large tiles (30-32):** Maximum reuse, fewer blocks
- **Optimal:** Usually 30 or 31 (avoid exact powers of 2 for better warp utilization)
