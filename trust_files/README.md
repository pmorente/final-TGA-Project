# Trust Files (Ground Truth Results)

CPU-generated reference results used to verify correctness of GPU implementations.

## Structure

```
trust_files/
├── 384/     # Results for 384-dimensional vectors
├── 768/     # Results for 768-dimensional vectors
└── 1024/    # Results for 1024-dimensional vectors
```

## Files

Each dimension folder contains results for three metrics:

- **`cpu_results_cosine_<dim>.bin`** - Cosine similarity results
- **`cpu_results_euclidean_<dim>.bin`** - Euclidean distance results
- **`cpu_results_pearson_<dim>.bin`** - Pearson correlation results

## Format

Binary format containing comparison results:
```
[float32 array: num_queries × num_database values]
```

- **Data type:** `float32` (4 bytes per value)
- **Layout:** Row-major order
- **Size:** `num_queries × num_database × 4` bytes
- **Index formula:** `result[query_idx * num_database + db_idx]`

## Generation

Trust files are generated using the sequential CPU implementation (Strategy 1):

```bash
cd scripts_get_results_cpu
./run_cpu_benchmark.sh
```

This ensures deterministic, single-threaded computation as ground truth.

## Usage

GPU results are compared against these files using:

```bash
cd scripts_get_results_gpu
./compare_with_trust.sh ../GPU/strategy_1
```

The comparison script:
- Verifies file format and size match
- Compares values with tolerance (default: 1e-7)
- Reports any discrepancies

## Notes

- These are considered "correct" reference results
- Generated using sequential, well-tested CPU code
- Used to validate all GPU implementations
- Small floating-point differences are expected due to different computation order
