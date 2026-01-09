# CPU Implementation Strategies

This directory contains three different CPU strategies for computing one-to-many vector comparisons.

## Files

### Executables
- **`compare_one_to_many`** - Sequential strategy (Strategy 1)
- **`parallel_compare_one_to_many`** - Vector-level parallelization (Strategy 2)
- **`parallel_metric_compare_one_to_many`** - Metric-level parallelization (Strategy 3)

### Source Files
- **`compare_one_to_many.cpp`** - Sequential implementation
- **`parallel_compare_one_to_many.cpp`** - Parallelizes outer loop over vectors
- **`parallel_metric_compare_one_to_many.cpp`** - Parallelizes inner dimension loops

### Metric Implementations

#### `cosine/` - Cosine Similarity
- **`cosine.cpp`** / **`cosine.hpp`** - Sequential cosine similarity
- **`cosine_parallel.cpp`** / **`cosine_parallel.hpp`** - Parallel cosine with OpenMP reduction

#### `euclidean/` - Euclidean Distance
- **`euclidean.cpp`** / **`euclidean.hpp`** - Sequential Euclidean distance
- **`euclidean_parallel.cpp`** / **`euclidean_parallel.hpp`** - Parallel Euclidean with OpenMP reduction

#### `pearson/` - Pearson Correlation
- **`pearson.cpp`** / **`pearson.hpp`** - Sequential Pearson correlation
- **`pearson_parallel.cpp`** / **`pearson_parallel.hpp`** - Parallel Pearson with OpenMP reduction

---

## Strategies

## Strategy 1: Sequential (No Parallelism)

**File:** `compare_one_to_many.cpp`

**Description:** 
- Single-threaded, sequential execution
- Processes all comparisons one by one in nested loops
- No parallelization overhead
- Baseline for performance comparison

**Compile:**
```bash
cd CPU && g++ -O2 compare_one_to_many.cpp cosine/cosine.cpp pearson/pearson.cpp euclidean/euclidean.cpp -o compare_one_to_many
```

**Run:**
```bash
cd CPU && ./compare_one_to_many input_data/A/embeddings.bin input_data/B/embeddings.bin cosine
```

**Note:** The third parameter specifies the metric: `cosine`, `euclidean`, or `pearson`. Only the selected metric will be computed.

**Note:** This version only prints execution time metrics, it does NOT save results to a file.

---

## Strategy 2: Parallel at Vector Level

**File:** `parallel_compare_one_to_many.cpp`

**Description:**
- Multi-threaded using OpenMP
- Parallelizes the **outer loop** over vectors from file A
- Each thread processes a subset of vectors from A
- Uses dynamic scheduling for better load balancing
- Similar to task-level parallelism

**How it works:**
- The outer loop `for (i = 0; i < records_a.size(); ++i)` is parallelized
- Each thread gets assigned different vectors from A to process
- All comparisons for a given vector A[i] are done sequentially by one thread
- Thread-safe progress tracking with atomic counters

**Compile:**
```bash
cd CPU && g++ -fopenmp -O2 parallel_compare_one_to_many.cpp cosine/cosine.cpp pearson/pearson.cpp euclidean/euclidean.cpp -o parallel_compare_one_to_many
```

**Run:**
```bash
cd CPU && ./parallel_compare_one_to_many input_data/A/embeddings.bin input_data/B/embeddings.bin cosine
```

**Note:** The third parameter specifies the metric: `cosine`, `euclidean`, or `pearson`. Only the selected metric will be computed.

**Note:** This version only prints execution time metrics, it does NOT save results to a file.

---

## Strategy 3: Parallel at Metric Computation Level

**File:** `parallel_metric_compare_one_to_many.cpp`

**Description:**
- Multi-threaded using OpenMP
- Parallelizes the **inner loops** within each similarity metric computation
- Each metric function (cosine, euclidean, pearson) parallelizes its loop over vector dimensions
- Similar to CUDA reduction pattern where threads work on different elements
- Best for high-dimensional vectors (768, 1024, etc.)

**How it works:**
- The outer loops over vectors remain sequential
- Inside each similarity computation, the loop over dimensions is parallelized
- Uses OpenMP reduction for accumulations (dot products, sums, etc.)
- Multiple threads work on different dimensions of the same vector pair simultaneously

**Example for Cosine Similarity:**
```cpp
// Parallelized dimension loop
#pragma omp parallel for reduction(+:dot,normA,normB)
for (int i = 0; i < D; ++i) {
    dot += A[i] * B[i];
    normA += A[i] * A[i];
    normB += B[i] * B[i];
}
```

**Compile:**
```bash
cd CPU && g++ -fopenmp -O2 parallel_metric_compare_one_to_many.cpp cosine/cosine_parallel.cpp pearson/pearson_parallel.cpp euclidean/euclidean_parallel.cpp -o parallel_metric_compare_one_to_many
```

**Run:**
```bash
cd CPU && ./parallel_metric_compare_one_to_many input_data/A/embeddings.bin input_data/B/embeddings.bin cosine
```

**Note:** The third parameter specifies the metric: `cosine`, `euclidean`, or `pearson`. Only the selected metric will be computed.

**Note:** This version only prints execution time metrics, it does NOT save results to a file.

---

## Comparison of Strategies

| Strategy | Parallelization Level | Best For | Overhead |
|----------|----------------------|----------|----------|
| Strategy 1 (Sequential) | None | Baseline, small datasets | None |
| Strategy 2 (Vector-level) | Outer loop (vectors) | Many vectors, fewer dimensions | Thread management |
| Strategy 3 (Metric-level) | Inner loop (dimensions) | High-dimensional vectors (768, 1024+) | Reduction operations |

### When to Use Each Strategy:

- **Strategy 1**: Use as baseline or for very small datasets where parallelization overhead isn't worth it
- **Strategy 2**: Use when you have many vectors but moderate dimensions (e.g., 128-512 dimensions)
- **Strategy 3**: Use when you have high-dimensional vectors (768, 1024, 1536+) where parallelizing the dimension loop provides better speedup

### Performance Notes:

- Strategy 2 scales with the number of vectors in file A
- Strategy 3 scales with the embedding dimension size
- For very high dimensions (1024+), Strategy 3 may outperform Strategy 2
- The optimal strategy depends on: number of vectors, embedding dimension, and available CPU cores
