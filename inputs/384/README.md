# 384-Dimensional Vectors

Input files containing 384-dimensional vector embeddings.

## Files

- **`file_a.bin`** - Query vectors (~1,000 vectors × 384 dimensions)
- **`file_b.bin`** - Database vectors (~10,000 vectors × 384 dimensions)

## Size

- Each vector: 384 × 4 bytes = 1,536 bytes (float32)
- File A: ~1.5 MB
- File B: ~15 MB

## Usage

Common dimension size for smaller embedding models (e.g., BERT-base variants, sentence transformers).
