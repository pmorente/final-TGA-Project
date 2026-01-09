# Trust Files - 1024 Dimensions

Ground truth CPU results for 1024-dimensional vectors.

## Files

- **`cpu_results_cosine_1024.bin`** - Cosine similarity results
- **`cpu_results_euclidean_1024.bin`** - Euclidean distance results
- **`cpu_results_pearson_1024.bin`** - Pearson correlation results

## Size

Each file contains `num_queries × num_database` float32 values:
- Typically: 1,000 × 10,000 = 10,000,000 values
- File size: ~40 MB per file

## Generation

Generated using sequential CPU implementation (Strategy 1) to ensure deterministic results.
