# Setup Guide: Dataset and Trust Files Generation

This guide explains how to set up your environment and run the `setup_datasets_and_trust_files.sh` script to generate all necessary datasets, embeddings, input files, and trust files for the project.

## Overview

The setup script automates the following tasks:
1. **Creates datasets**: Generates CSV files with 1,000 and 10,000 text samples from the BookCorpus dataset
2. **Generates embeddings**: Creates vector embeddings in 384, 768, and 1024 dimensions using different sentence transformer models
3. **Organizes input files**: Copies embeddings to the `inputs/{dim}/` folders as `file_a.bin` and `file_b.bin`
4. **Compiles CPU code**: Builds the `compare_one_to_many` executable
5. **Generates trust files**: Computes ground truth results for all metrics (cosine, euclidean, pearson) and dimensions

## Prerequisites

### System Requirements

- **Operating System**: Linux (tested on Ubuntu/Debian-based systems)
- **Python**: Python 3.7 or higher
- **C++ Compiler**: GCC with C++11 support (g++ version 7.0 or higher)
- **Disk Space**: At least 5-10 GB free space (for models, datasets, and generated files)
- **RAM**: Recommended 8 GB or more (embedding generation can be memory-intensive)

### Required System Tools

Ensure the following tools are installed:

```bash
# Check Python version
python3 --version  # Should be 3.7+

# Check GCC compiler
g++ --version  # Should be 7.0+

# Check if bash is available
bash --version
```

If any tool is missing, install it using your package manager:

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install python3 python3-pip g++ build-essential

# Fedora/RHEL
sudo dnf install python3 python3-pip gcc-c++ make
```

## Installation Steps

### Step 1: Python Environment Setup

1. **Create a virtual environment** (recommended):

```bash
cd /path/to/project
python3 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows (if using Git Bash)
```

2. **Install Python dependencies**:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

The `requirements.txt` includes:
- `sentence-transformers>=2.2.0` - For generating embeddings
- `numpy>=1.21.0` - For numerical operations
- `torch>=1.9.0` - PyTorch backend for sentence-transformers
- `transformers>=4.20.0` - Hugging Face transformers library
- `datasets>=2.0.0` - For loading the BookCorpus dataset

**Note**: The first time you run the script, sentence-transformers will download the models automatically. This may take several minutes and requires internet connection.

### Step 2: Verify Directory Structure

Ensure your project has the following structure:

```
project_root/
├── tooling_dataset_creation/
│   ├── create_dataset.py
│   ├── generateEmbeddings.py
│   └── tools/
│       └── embeddingGenerator.py
├── CPU/
│   ├── compare_one_to_many.cpp
│   ├── cosine/
│   │   ├── cosine.cpp
│   │   └── cosine.hpp
│   ├── euclidean/
│   │   ├── euclidean.cpp
│   │   └── euclidean.hpp
│   └── pearson/
│       ├── pearson.cpp
│       └── pearson.hpp
├── inputs/
│   ├── 384/
│   ├── 768/
│   └── 1024/
├── trust_files/
│   ├── 384/
│   ├── 768/
│   └── 1024/
├── requirements.txt
└── setup_datasets_and_trust_files.sh
```

### Step 3: Make Script Executable

```bash
chmod +x setup_datasets_and_trust_files.sh
```

## What Gets Created

After running the setup script, the following files and directories will be created:

### 1. Dataset Files (in project root)
- `dataset_1000.csv` - CSV file with 1,000 text samples (id, text columns)
- `dataset_10000.csv` - CSV file with 10,000 text samples (id, text columns)

### 2. Embedding Files (in `tooling_dataset_creation/outputs_embeddings/`)
- `dimension-384/embeddings_1000_384.bin` - 384-dim embeddings for 1,000 samples
- `dimension-384/embeddings_10000_384.bin` - 384-dim embeddings for 10,000 samples
- `dimension-768/embeddings_1000_768.bin` - 768-dim embeddings for 1,000 samples
- `dimension-768/embeddings_10000_768.bin` - 768-dim embeddings for 10,000 samples
- `dimension-1024/embeddings_1000_1024.bin` - 1024-dim embeddings for 1,000 samples
- `dimension-1024/embeddings_10000_1024.bin` - 1024-dim embeddings for 10,000 samples

### 3. Input Files (in `inputs/{dim}/`)
- `inputs/384/file_a.bin` - Query vectors (1,000 vectors × 384 dimensions)
- `inputs/384/file_b.bin` - Database vectors (10,000 vectors × 384 dimensions)
- `inputs/768/file_a.bin` - Query vectors (1,000 vectors × 768 dimensions)
- `inputs/768/file_b.bin` - Database vectors (10,000 vectors × 768 dimensions)
- `inputs/1024/file_a.bin` - Query vectors (1,000 vectors × 1024 dimensions)
- `inputs/1024/file_b.bin` - Database vectors (10,000 vectors × 1024 dimensions)

### 4. Compiled Executable (in `CPU/`)
- `CPU/compare_one_to_many` - Compiled C++ executable for computing comparisons

### 5. Trust Files (in `trust_files/{dim}/`)
- `trust_files/384/cpu_results_cosine_384.bin` - Cosine similarity results
- `trust_files/384/cpu_results_euclidean_384.bin` - Euclidean distance results
- `trust_files/384/cpu_results_pearson_384.bin` - Pearson correlation results
- `trust_files/768/cpu_results_cosine_768.bin` - Cosine similarity results
- `trust_files/768/cpu_results_euclidean_768.bin` - Euclidean distance results
- `trust_files/768/cpu_results_pearson_768.bin` - Pearson correlation results
- `trust_files/1024/cpu_results_cosine_1024.bin` - Cosine similarity results
- `trust_files/1024/cpu_results_euclidean_1024.bin` - Euclidean distance results
- `trust_files/1024/cpu_results_pearson_1024.bin` - Pearson correlation results

## Running the Setup Script

### Basic Usage

```bash
# Make sure you're in the project root directory
cd /path/to/project

# Activate virtual environment (if using one)
source venv/bin/activate

# Run the setup script
./setup_datasets_and_trust_files.sh
```

### Expected Execution Time

The script may take **30 minutes to 2 hours** to complete, depending on:
- Your CPU speed
- Internet connection (for downloading models on first run)
- System resources

**Breakdown of time:**
- Dataset creation: ~5-10 minutes (downloading BookCorpus samples)
- Embedding generation: ~20-60 minutes (depends on model size and CPU)
  - 384 dimensions: ~5-10 minutes
  - 768 dimensions: ~10-20 minutes
  - 1024 dimensions: ~15-30 minutes
- Trust file generation: ~10-30 minutes (CPU computation)

### Progress Monitoring

The script provides progress output for each step. You'll see:
- Dataset creation progress
- Embedding generation progress (with progress bars)
- File copying operations
- Compilation status
- Trust file computation progress

## Models Used

The script uses the following sentence transformer models:

| Dimension | Model | Description |
|-----------|-------|-------------|
| 384 | `all-MiniLM-L6-v2` | Fastest, smallest model (~80MB) |
| 768 | `all-mpnet-base-v2` | Better quality, medium size (~420MB) |
| 1024 | `all-roberta-large-v1` | Best quality, largest model (~1.3GB) |

**Note**: Models are downloaded automatically on first use and cached in `~/.cache/huggingface/`.

