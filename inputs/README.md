# Input Files

Binary files containing vector embeddings for testing CPU and GPU implementations.

## Structure

```
inputs/
├── 384/     # 384-dimensional vectors
├── 768/     # 768-dimensional vectors
└── 1024/    # 1024-dimensional vectors
```

## Files

Each dimension folder contains two files:

- **`file_a.bin`** - Query vectors (smaller set, typically ~1,000 vectors)
- **`file_b.bin`** - Database vectors (larger set, typically ~10,000 vectors)

## Format

Binary format with the following structure:
```
[num_records: int32]  [dimension: int32]  [vector_data: float32 array]
```

- **Header:** 8 bytes (2 × int32)
  - Number of vectors in the file
  - Dimension of each vector
- **Data:** `num_records × dimension × 4` bytes (float32 array)

## Usage

These files are used by all CPU and GPU implementations:

```bash
# CPU example
./compare_one_to_many inputs/384/file_a.bin inputs/384/file_b.bin cosine

# GPU example
./estrategia1_cosine inputs/384/file_a.bin inputs/384/file_b.bin
```

## Notes

- All implementations expect the same binary format
- Files contain randomly generated or real embedding vectors
- Dimension sizes (384, 768, 1024) are common in NLP/CV models
