# CPU Benchmark Scripts

Scripts for running CPU implementations and generating ground truth results.

## Files

- **`run_cpu_benchmark.sh`** - Comprehensive CPU benchmark script
  - Compiles all three CPU strategies (sequential, vector-parallel, metric-parallel)
  - Runs benchmarks on all dimensions (384, 768, 1024)
  - Tests all metrics (cosine, euclidean, pearson)
  - Generates performance comparison tables
  - Saves results to `trust_files/` directory

## Usage

```bash
cd scripts_get_results_cpu
chmod +x run_cpu_benchmark.sh
./run_cpu_benchmark.sh
```

## Output

### Performance Tables
Generated as Markdown tables showing:
- Execution time for each strategy
- Per-metric breakdowns
- Speedup comparisons between strategies

### Result Files
Binary files saved to `trust_files/` directory:
```
trust_files/
├── 384/
│   ├── cpu_results_cosine_384.bin
│   ├── cpu_results_euclidean_384.bin
│   └── cpu_results_pearson_384.bin
├── 768/
│   └── ...
└── 1024/
    └── ...
```

## Requirements

- `g++` compiler with OpenMP support (`-fopenmp`)
- CPU with multiple cores (for parallel strategies)
- Input files in `inputs/` directory

## Notes

- Sequential results are used as ground truth for GPU verification
- Parallel strategies demonstrate CPU-side optimizations
- Script automatically creates `trust_files/` subdirectories
