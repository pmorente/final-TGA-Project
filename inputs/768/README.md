# 768-Dimensional Vectors

Input files containing 768-dimensional vector embeddings.

## Files

- **`file_a.bin`** - Query vectors (~1,000 vectors × 768 dimensions)
- **`file_b.bin`** - Database vectors (~10,000 vectors × 768 dimensions)

## Size

- Each vector: 768 × 4 bytes = 3,072 bytes (float32)
- File A: ~3 MB
- File B: ~30 MB

## Usage

Common dimension size for standard embedding models (e.g., BERT-large, many sentence embedding models).
