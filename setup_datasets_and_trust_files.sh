#!/bin/bash

# Script to create datasets, generate embeddings, and compute trust files
# This script:
# 1. Creates datasets of 1000 and 10000 items
# 2. Generates embeddings for 384, 768, and 1024 dimensions
# 3. Copies embeddings to inputs/{dim}/file_a.bin and file_b.bin
# 4. Compiles compare_one_to_many.cpp
# 5. Generates trust files for all metrics and dimensions

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Dataset and Trust Files Setup Script"
echo "=========================================="
echo ""

# Step 1: Create datasets
echo "Step 1: Creating datasets..."
echo "Creating dataset with 1000 items..."
python3 tooling_dataset_creation/create_dataset.py 1000 dataset_1000.csv

echo "Creating dataset with 10000 items..."
python3 tooling_dataset_creation/create_dataset.py 10000 dataset_10000.csv

echo "Datasets created successfully!"
echo ""

# Step 2: Generate embeddings for each dimension
echo "Step 2: Generating embeddings..."
echo ""

# 384 dimensions (all-MiniLM-L6-v2)
echo "Generating 384-dimensional embeddings for dataset_1000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_1000.csv -m all-MiniLM-L6-v2 -o embeddings_1000_384.bin --mode batch --batch-size 64

echo "Generating 384-dimensional embeddings for dataset_10000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_10000.csv -m all-MiniLM-L6-v2 -o embeddings_10000_384.bin --mode batch --batch-size 64

# 768 dimensions (all-mpnet-base-v2)
echo "Generating 768-dimensional embeddings for dataset_1000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_1000.csv -m all-mpnet-base-v2 -o embeddings_1000_768.bin --mode batch --batch-size 32

echo "Generating 768-dimensional embeddings for dataset_10000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_10000.csv -m all-mpnet-base-v2 -o embeddings_10000_768.bin --mode batch --batch-size 32

# 1024 dimensions (all-roberta-large-v1)
echo "Generating 1024-dimensional embeddings for dataset_1000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_1000.csv -m all-roberta-large-v1 -o embeddings_1000_1024.bin --mode batch --batch-size 16

echo "Generating 1024-dimensional embeddings for dataset_10000..."
python3 tooling_dataset_creation/generateEmbeddings.py dataset_10000.csv -m all-roberta-large-v1 -o embeddings_10000_1024.bin --mode batch --batch-size 16

echo "Embeddings generated successfully!"
echo ""

# Step 3: Copy embeddings to inputs folders
echo "Step 3: Copying embeddings to inputs folders..."
echo ""

# Create inputs directories if they don't exist
mkdir -p inputs/384
mkdir -p inputs/768
mkdir -p inputs/1024

# Copy 384-dimensional embeddings
echo "Copying 384-dimensional embeddings..."
cp tooling_dataset_creation/outputs_embeddings/dimension-384/embeddings_1000_384.bin inputs/384/file_a.bin
cp tooling_dataset_creation/outputs_embeddings/dimension-384/embeddings_10000_384.bin inputs/384/file_b.bin

# Copy 768-dimensional embeddings
echo "Copying 768-dimensional embeddings..."
cp tooling_dataset_creation/outputs_embeddings/dimension-768/embeddings_1000_768.bin inputs/768/file_a.bin
cp tooling_dataset_creation/outputs_embeddings/dimension-768/embeddings_10000_768.bin inputs/768/file_b.bin

# Copy 1024-dimensional embeddings
echo "Copying 1024-dimensional embeddings..."
cp tooling_dataset_creation/outputs_embeddings/dimension-1024/embeddings_1000_1024.bin inputs/1024/file_a.bin
cp tooling_dataset_creation/outputs_embeddings/dimension-1024/embeddings_10000_1024.bin inputs/1024/file_b.bin

echo "Embeddings copied to inputs folders!"
echo ""

# Step 4: Compile compare_one_to_many
echo "Step 4: Compiling compare_one_to_many..."
cd CPU
g++ -O2 compare_one_to_many.cpp cosine/cosine.cpp pearson/pearson.cpp euclidean/euclidean.cpp -o compare_one_to_many
cd ..
echo "Compilation successful!"
echo ""

# Step 5: Generate trust files
echo "Step 5: Generating trust files..."
echo ""

# Create trust_files directories if they don't exist
mkdir -p trust_files/384
mkdir -p trust_files/768
mkdir -p trust_files/1024

# Metrics to compute
METRICS=("cosine" "euclidean" "pearson")

# Dimensions to process
DIMENSIONS=(384 768 1024)

# Generate trust files for each dimension and metric
for dim in "${DIMENSIONS[@]}"; do
    echo "Processing dimension $dim..."
    
    for metric in "${METRICS[@]}"; do
        echo "  Computing $metric similarity for dimension $dim..."
        
        # Run compare_one_to_many with --save-results
        cd CPU
        ./compare_one_to_many "../inputs/$dim/file_a.bin" "../inputs/$dim/file_b.bin" "$metric" --save-results
        cd ..
        
        # Move the result file to trust_files/{dim}/
        result_file="CPU/cpu_results_${metric}_${dim}.bin"
        if [ -f "$result_file" ]; then
            mv "$result_file" "trust_files/$dim/"
            echo "  Saved: trust_files/$dim/cpu_results_${metric}_${dim}.bin"
        else
            echo "  WARNING: Result file not found: $result_file"
        fi
    done
    
    echo ""
done

echo "=========================================="
echo "Setup completed successfully!"
echo "=========================================="
echo ""
echo "Summary:"
echo "  - Datasets: dataset_1000.csv, dataset_10000.csv"
echo "  - Input files: inputs/{384,768,1024}/file_a.bin, file_b.bin"
echo "  - Trust files: trust_files/{384,768,1024}/cpu_results_{metric}_{dim}.bin"
echo ""
