# Cosine Similarity Implementation

Measures the cosine of the angle between two vectors (similarity ranges from -1 to 1).

## Files

### Sequential Implementation
- **`cosine.cpp`** - Sequential cosine similarity calculation
- **`cosine.hpp`** - Header with function declarations

### Parallel Implementation
- **`cosine_parallel.cpp`** - OpenMP parallelized cosine similarity
- **`cosine_parallel.hpp`** - Header with parallel function declarations

## Formula

```
similarity = (A · B) / (||A|| × ||B||)
```

Where:
- `A · B` = dot product
- `||A||` = L2 norm of vector A
- `||B||` = L2 norm of vector B

## Parallelization

The parallel version uses OpenMP reduction on the dimension loop:

```cpp
#pragma omp parallel for reduction(+:dot,normA,normB)
for (int i = 0; i < D; ++i) {
    dot += A[i] * B[i];
    normA += A[i] * A[i];
    normB += B[i] * B[i];
}
```

## Usage

Compiled as part of the main CPU executables.
