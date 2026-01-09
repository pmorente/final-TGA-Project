# 1024-Dimensional Vectors

Input files containing 1024-dimensional vector embeddings.

## Files

- **`file_a.bin`** - Query vectors (~1,000 vectors × 1024 dimensions)
- **`file_b.bin`** - Database vectors (~10,000 vectors × 1024 dimensions)

## Size

- Each vector: 1024 × 4 bytes = 4,096 bytes (float32)
- File A: ~4 MB
- File B: ~40 MB

## Usage

Common dimension size for high-dimensional embeddings (e.g., vision transformers, large language models).
