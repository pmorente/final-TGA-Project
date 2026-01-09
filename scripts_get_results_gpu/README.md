# GPU Benchmark Scripts

This directory contains scripts to run GPU benchmarks, compare results with CPU trust files, and verify correctness.

## Files

### Benchmark Scripts (Individual Strategies)
- **`run_strategy1_benchmark.sh`** - Strategy 1 benchmarks (no parameters)
- **`run_strategy2_benchmark.sh`** - Strategy 2 benchmarks (requires block_size)
- **`run_strategy3_benchmark.sh`** - Strategy 3 benchmarks (requires block_size)
- **`run_strategy5_benchmark.sh`** - Strategy 5 benchmarks (requires block_size)
- **`run_strategy6_benchmark.sh`** - Strategy 6 benchmarks (requires block_size)
- **`run_strategy7_benchmark.sh`** - Strategy 7 benchmarks (requires tile_size)

### Comparison Scripts (Multiple Configurations)
- **`run_strategy2_compare_blocksizes.sh`** - Tests Strategy 2 with multiple block sizes (32, 64, 128, 256, 512, 1024)
- **`run_strategy3_compare_blocksizes.sh`** - Tests Strategy 3 with multiple block sizes
- **`run_strategy5_compare_blocksizes.sh`** - Tests Strategy 5 with multiple block sizes
- **`run_strategy6_compare_blocksizes.sh`** - Tests Strategy 6 with multiple block sizes
- **`run_strategy7_compare_tiles.sh`** - Tests Strategy 7 with multiple tile sizes (4, 8, 16, 30, 31, 32)

### Verification Scripts
- **`compare_with_trust.sh`** - Compares GPU results with CPU trust files
- **`verify_format.py`** - Python script to verify binary file format consistency

## Format Verification

Both CPU and GPU result files use the **same binary format**:

- **Data type:** `float32` (4 bytes per value)
- **Format:** Binary (no text encoding)
- **Layout:** Flat array of float32 values
- **Order:** Row-major order
  - Index calculation: `result[query_idx * num_database + db_idx]`
  - Same for both CPU and GPU

### CPU Format
```cpp
std::ofstream fout("cpu_results_<metric>_<dimension>.bin", std::ios::binary);
fout.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(float));
```
- Uses `sizeof(float)` = 4 bytes
- Saves `results.size() * 4` bytes

### GPU Format
```cpp
std::ofstream fout("gpu_results_<metric>.bin", std::ios::binary);
fout.write(reinterpret_cast<const char*>(h_results), num_queries * num_database * sizeof(float));
```
- Uses `sizeof(float)` = 4 bytes
- Saves `num_queries * num_database * 4` bytes

**Both are identical in format!**

## Usage

### 1. Run GPU Benchmarks

```bash
cd scripts_get_results_gpu
./run_strategy1_benchmark.sh
```

This will:
- Run GPU Strategy 1 for all metrics and dimensions
- Save results as `gpu_results_<metric>_<dimension>.bin` in the GPU strategy directory
- Generate performance tables

### 2. Compare with Trust Files

```bash
cd scripts_get_results_gpu
chmod +x compare_with_trust.sh
./compare_with_trust.sh ../GPU/strategy_1
```

Or with custom tolerance:
```bash
./compare_with_trust.sh ../GPU/strategy_1 --tolerance 1e-6
```

The script will:
- Verify file formats (both use float32)
- Check file sizes match
- Compare values with specified tolerance
- Report differences if any

### 3. Verify Format Only

```bash
python3 verify_format.py <cpu_file> <gpu_file>
```

## Trust Files Structure

Trust files are organized by dimension:
```
trust_files/
├── 384/
│   ├── cpu_results_compare_one_to_many_cosine_384.bin
│   ├── cpu_results_compare_one_to_many_euclidean_384.bin
│   └── cpu_results_compare_one_to_many_pearson_384.bin
├── 768/
│   └── ...
└── 1024/
    └── ...
```

These are the "ground truth" CPU results from Strategy 1 (sequential).

## Expected Results

If GPU implementation is correct:
- All comparisons should pass within tolerance (default: 1e-7)
- File formats should match exactly
- File sizes should be identical
- Values should match within numerical precision

If differences are found:
- Check that block sizes are correct
- Verify kernel implementations
- Check for race conditions or synchronization issues
- Verify input data is the same

## Notes

- The comparison uses `numpy.allclose()` with absolute tolerance
- Default tolerance is 1e-7 (0.0000001)
- Differences may occur due to:
  - Floating-point precision differences between CPU and GPU
  - Different reduction order in parallel computations
  - Numerical stability issues
- Format differences (if any) will be caught immediately by the format check

