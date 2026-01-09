# Euclidean Distance Implementation

Measures the straight-line distance between two vectors in n-dimensional space.

## Files

### Sequential Implementation
- **`euclidean.cpp`** - Sequential Euclidean distance calculation
- **`euclidean.hpp`** - Header with function declarations

### Parallel Implementation
- **`euclidean_parallel.cpp`** - OpenMP parallelized Euclidean distance
- **`euclidean_parallel.hpp`** - Header with parallel function declarations

## Formula

```
distance = sqrt(Σ(A[i] - B[i])²)
```

Where:
- Sum over all dimensions
- Squared differences are accumulated

## Parallelization

The parallel version uses OpenMP reduction on the dimension loop:

```cpp
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < D; ++i) {
    float diff = A[i] - B[i];
    sum += diff * diff;
}
```

## Usage

Compiled as part of the main CPU executables.
