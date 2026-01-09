# GPU CUDA Implementation Strategies

This directory contains 7 different CUDA strategies for computing one-to-many vector comparisons on GPU.

## Files

### Build System
- **`Makefile`** - Build script for all GPU strategies
  - Compiles all strategies with optimized flags
  - Targets: `all`, `s1`, `s2`, `s3`, `s5`, `s6`, `s7`, `clean`

### Strategy Folders

Each `strategy_*/` folder contains:

#### Source Files
- **`estrategia*_cosine.cu`** - CUDA implementation for cosine similarity
- **`estrategia*_euclidean.cu`** - CUDA implementation for Euclidean distance
- **`estrategia*_pearson.cu`** - CUDA implementation for Pearson correlation
- **`estrategia*_pearson_opt.cu`** - Optimized Pearson implementation (if applicable)

#### Executables
- **`estrategia*_cosine`** / **`.exe`** - Compiled cosine executable
- **`estrategia*_euclidean`** / **`.exe`** - Compiled Euclidean executable
- **`estrategia*_pearson`** / **`.exe`** - Compiled Pearson executable
- **`estrategia*_pearson_opt`** / **`.exe`** - Compiled optimized Pearson executable

#### Result Files
- **`gpu_results_cosine_<dim>.bin`** - GPU cosine results (384, 768, 1024)
- **`gpu_results_euclidean_<dim>.bin`** - GPU Euclidean results
- **`gpu_results_pearson_<dim>.bin`** - GPU Pearson results
- **`gpu_results_pearson_opt_<dim>.bin`** - GPU optimized Pearson results

#### Documentation
- **`STRATEGY_DESCRIPTION.md`** - Detailed strategy description (select folders)

---

## Strategies

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit installed (tested with CUDA 11.0+)
- `nvcc` compiler in your PATH
- Binary embedding files in the format: `CPU/input_data/A/embeddings.bin` and `CPU/input_data/B/embeddings.bin`

### Installing CUDA Toolkit (WSL/Ubuntu)

If `nvcc` is not found, install the CUDA Toolkit:

**Option 1: Install via apt (Recommended for WSL)**
```bash
sudo apt update
sudo apt install -y nvidia-cuda-toolkit
```

After installation, verify:
```bash
nvcc --version
```

**Option 2: Install CUDA Toolkit from NVIDIA (For full CUDA SDK)**
```bash
# Add NVIDIA package repositories
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update

# Install CUDA Toolkit (replace 12.x with your desired version)
sudo apt-get -y install cuda-toolkit-12-4

# Add to PATH (add to ~/.bashrc for persistence)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Verify Installation:**
```bash
nvcc --version
nvidia-smi
```

**Note:** For WSL, ensure you have:
- WSL 2 installed
- NVIDIA drivers installed on Windows host
- NVIDIA Container Toolkit (if using containers)

## General Compilation Pattern

All strategies follow the same compilation pattern:
```bash
nvcc -O3 -arch=sm_XX <strategy_file.cu> -o <executable_name>
```

Replace `sm_XX` with your GPU's compute capability (e.g., `sm_75` for Turing, `sm_80` for Ampere, `sm_86` for Ampere mobile, `sm_89` for Ada).

To find your GPU's compute capability:
```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

**Compute Capability Mapping:**
- 7.5 → `sm_75` (Turing: RTX 20-series, GTX 16-series)
- 8.0 → `sm_80` (Ampere: A100)
- 8.6 → `sm_86` (Ampere: RTX 30-series, RTX A-series)
- 8.9 → `sm_89` (Ada: RTX 40-series)

Or compile for multiple architectures:
```bash
nvcc -O3 -arch=sm_75 -arch=sm_80 -arch=sm_86 -arch=sm_89 <strategy_file.cu> -o <executable_name>
```

## General Execution Pattern

All executables follow the same execution pattern:
```bash
./<executable_name> <file_a> <file_b>
```

Example:
```bash
./estrategia2_cosine ../CPU/input_data/A/embeddings.bin ../CPU/input_data/B/embeddings.bin
```

---

## Strategy 1 — Grid over pairs, sequential per thread
**(Baseline, correct but slow)**

**Mapping:**
- 1 thread = 1 vector comparison
- Grid is 2D: `blockIdx.x → index in A`, `blockIdx.y → index in B`
- Configuration: `dim3 grid(1000, 10000); dim3 block(1);`

**Files:** `strategy_1/estrategia1_cosine.cu`, `estrategia1_euclidean.cu`, `estrategia1_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_1
nvcc -O3 -arch=sm_89 estrategia1_cosine.cu -o estrategia1_cosine
nvcc -O3 -arch=sm_89 estrategia1_euclidean.cu -o estrategia1_euclidean
nvcc -O3 -arch=sm_89 estrategia1_pearson.cu -o estrategia1_pearson
```

**Run:**
```bash
./estrategia1_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia1_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia1_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 2 — Grid over SMALL group, loop BIG group, sequential
**(Teacher's first grid explanation)**

**Mapping:**
- 1 block = 1 vector from A
- Threads in block iterate over B
- Configuration: `dim3 grid(1000); dim3 block(256);`

**Files:** `strategy_2/estrategia2_cosine.cu`, `estrategia2_euclidean.cu`, `estrategia2_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_2
nvcc -O3 -arch=sm_89 estrategia2_cosine.cu -o estrategia2_cosine
nvcc -O3 -arch=sm_89 estrategia2_euclidean.cu -o estrategia2_euclidean
nvcc -O3 -arch=sm_89 estrategia2_pearson.cu -o estrategia2_pearson
```

**Run:**
```bash
./estrategia2_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia2_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia2_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 3 — Grid over BIG group, loop SMALL group, sequential

**Mapping:**
- 1 block = 1 vector from B
- Threads iterate over A
- Configuration: `dim3 grid(10000); dim3 block(256);`

**Files:** `strategy_3/estrategia3_cosine.cu`, `estrategia3_euclidean.cu`, `estrategia3_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_3
nvcc -O3 -arch=sm_89 estrategia3_cosine.cu -o estrategia3_cosine
nvcc -O3 -arch=sm_89 estrategia3_euclidean.cu -o estrategia3_euclidean
nvcc -O3 -arch=sm_89 estrategia3_pearson.cu -o estrategia3_pearson
```

**Run:**
```bash
./estrategia3_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia3_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia3_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 4 — Grid over PAIRS, reduction over dimensions
**(Teacher's reduction idea)**

**Mapping:**
- 1 block = 1 vector comparison
- 1024 threads = 1024 dimensions
- Configuration: `dim3 grid(1000, 10000); dim3 block(1024);`

**Files:** `strategy_4/estrategia4_cosine.cu`, `estrategia4_euclidean.cu`, `estrategia4_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_4
nvcc -O3 -arch=sm_89 estrategia4_cosine.cu -o estrategia4_cosine
nvcc -O3 -arch=sm_89 estrategia4_euclidean.cu -o estrategia4_euclidean
nvcc -O3 -arch=sm_89 estrategia4_pearson.cu -o estrategia4_pearson
```

**Run:**
```bash
./estrategia4_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia4_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia4_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 5 — Grid over SMALL group, reduction over dimensions
**(First "correct" GPU approach)**

**Mapping:**
- 1 block = A[i]
- Loop over B[j]
- For each B[j], do reduction
- Configuration: `dim3 grid(1000); dim3 block(256 or 512);`

**Files:** `strategy_5/estrategia5_cosine.cu`, `estrategia5_euclidean.cu`, `estrategia5_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_5
nvcc -O3 -arch=sm_89 estrategia5_cosine.cu -o estrategia5_cosine
nvcc -O3 -arch=sm_89 estrategia5_euclidean.cu -o estrategia5_euclidean
nvcc -O3 -arch=sm_89 estrategia5_pearson.cu -o estrategia5_pearson
```

**Run:**
```bash
./estrategia5_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia5_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia5_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 6 — Grid over BIG group, reduction over dimensions
**(Usually better)**

**Mapping:**
- 1 block = B[j]
- Threads reduce over dimensions
- Loop over A
- Configuration: `dim3 grid(10000); dim3 block(256 or 512);`

**Files:** `strategy_6/estrategia6_cosine.cu`, `estrategia6_euclidean.cu`, `estrategia6_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_6
nvcc -O3 -arch=sm_89 estrategia6_cosine.cu -o estrategia6_cosine
nvcc -O3 -arch=sm_89 estrategia6_euclidean.cu -o estrategia6_euclidean
nvcc -O3 -arch=sm_89 estrategia6_pearson.cu -o estrategia6_pearson
```

**Run:**
```bash
./estrategia6_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia6_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia6_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Strategy 7 — 2D TILED + REDUCTION
**(BEST PRACTICE)**

**Mapping:**
- Grid is 2D
- Block handles: TA vectors from A, TB vectors from B
- Threads handle dimensions
- Configuration: `dim3 grid(ceil(1000/TA), ceil(10000/TB)); dim3 block(256);`
- Tile sizes: TA=4, TB=4

**Files:** `strategy_7/estrategia7_cosine.cu`, `estrategia7_euclidean.cu`, `estrategia7_pearson.cu`

**Compile:**
```bash
cd GPU/strategy_7
nvcc -O3 -arch=sm_89 estrategia7_cosine.cu -o estrategia7_cosine
nvcc -O3 -arch=sm_89 estrategia7_euclidean.cu -o estrategia7_euclidean
nvcc -O3 -arch=sm_89 estrategia7_pearson.cu -o estrategia7_pearson
```

**Run:**
```bash
./estrategia7_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia7_euclidean ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
./estrategia7_pearson ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

---

## Batch Compilation Script

Create a script `compile_all.sh` to compile all strategies at once:

```bash
#!/bin/bash

# Set your GPU compute capability
# Default: sm_89 (Ada - RTX 40-series)
ARCH="sm_89"  # Change this to match your GPU

# Strategy 2
cd strategy_2
nvcc -O3 -arch=$ARCH estrategia2_cosine.cu -o estrategia2_cosine
nvcc -O3 -arch=$ARCH estrategia2_euclidean.cu -o estrategia2_euclidean
nvcc -O3 -arch=$ARCH estrategia2_pearson.cu -o estrategia2_pearson
cd ..

# Strategy 3
cd strategy_3
nvcc -O3 -arch=$ARCH estrategia3_cosine.cu -o estrategia3_cosine
nvcc -O3 -arch=$ARCH estrategia3_euclidean.cu -o estrategia3_euclidean
nvcc -O3 -arch=$ARCH estrategia3_pearson.cu -o estrategia3_pearson
cd ..

# Strategy 5
cd strategy_5
nvcc -O3 -arch=$ARCH estrategia5_cosine.cu -o estrategia5_cosine
nvcc -O3 -arch=$ARCH estrategia5_euclidean.cu -o estrategia5_euclidean
nvcc -O3 -arch=$ARCH estrategia5_pearson.cu -o estrategia5_pearson
cd ..

# Strategy 6
cd strategy_6
nvcc -O3 -arch=$ARCH estrategia6_cosine.cu -o estrategia6_cosine
nvcc -O3 -arch=$ARCH estrategia6_euclidean.cu -o estrategia6_euclidean
nvcc -O3 -arch=$ARCH estrategia6_pearson.cu -o estrategia6_pearson
cd ..

# Strategy 7
cd strategy_7
nvcc -O3 -arch=$ARCH estrategia7_cosine.cu -o estrategia7_cosine
nvcc -O3 -arch=$ARCH estrategia7_euclidean.cu -o estrategia7_euclidean
nvcc -O3 -arch=$ARCH estrategia7_pearson.cu -o estrategia7_pearson
cd ..

echo "Compilation complete!"
```

Make it executable and run:
```bash
chmod +x compile_all.sh
./compile_all.sh
```

---

## Batch Execution Script

Create a script `run_all.sh` to run all strategies and compare performance:

```bash
#!/bin/bash

FILE_A="../../CPU/input_data/A/embeddings.bin"
FILE_B="../../CPU/input_data/B/embeddings.bin"

echo "=== Running Strategy 2 ==="
cd strategy_2
./estrategia2_cosine $FILE_A $FILE_B
cd ..

echo "=== Running Strategy 3 ==="
cd strategy_3
./estrategia3_cosine $FILE_A $FILE_B
cd ..

echo "=== Running Strategy 5 ==="
cd strategy_5
./estrategia5_cosine $FILE_A $FILE_B
cd ..

echo "=== Running Strategy 6 ==="
cd strategy_6
./estrategia6_cosine $FILE_A $FILE_B
cd ..

echo "=== Running Strategy 7 ==="
cd strategy_7
./estrategia7_cosine $FILE_A $FILE_B
cd ..

echo "All strategies completed!"
```

---

## Performance Comparison

| Strategy | Grid Configuration | Block Size | Parallelization | Best For |
|----------|-------------------|------------|------------------|----------|
| Strategy 1 | 2D (A×B) | 1 | Sequential per thread | Baseline, reference |
| Strategy 2 | 1D (A) | 256 | Loop over B, sequential dims | Small A, large B |
| Strategy 3 | 1D (B) | 256 | Loop over A, sequential dims | Large A, small B |
| Strategy 4 | 2D (A×B) | 1024 | Reduction over dimensions | Pedagogical |
| Strategy 5 | 1D (A) | 256 | Loop B, reduction dims | Small A, large B |
| Strategy 6 | 1D (B) | 256 | Loop A, reduction dims | Large A, small B |
| Strategy 7 | 2D Tiled | 256 | Tiled + reduction | **Best practice** |

---

## Troubleshooting

### Error: "No CUDA-capable device is detected"
- Check that your GPU is NVIDIA and supports CUDA
- Verify CUDA installation: `nvcc --version`
- Check GPU: `nvidia-smi`

### Error: "compute capability mismatch"
- Find your GPU's compute capability
- Update the `-arch=sm_XX` flag in compilation commands
- Or compile for multiple architectures as shown above

### Error: "out of memory"
- Reduce batch sizes or tile sizes in Strategy 7
- Use smaller input files for testing
- Check available GPU memory: `nvidia-smi`

### Error: "file not found"
- Ensure you're running from the correct directory
- Check that embedding files exist at the specified paths
- Use absolute paths if relative paths don't work

---

## Notes

- All strategies read the same binary embedding format as the CPU implementation
- Each executable is standalone and includes its own binary file reader
- Performance metrics (computation time, throughput) are printed to stdout
- Results are stored in device memory during computation but not saved to disk
- For production use, modify the code to save results to a file if needed

---

## Quick Start Example

```bash
# Navigate to GPU directory
cd GPU/strategy_7

# Compile
nvcc -O3 -arch=sm_89 estrategia7_cosine.cu -o estrategia7_cosine

# Run
./estrategia7_cosine ../../CPU/input_data/A/embeddings.bin ../../CPU/input_data/B/embeddings.bin
```

