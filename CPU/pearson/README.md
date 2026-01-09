# Pearson Correlation Implementation

Measures the linear correlation between two vectors (ranges from -1 to 1).

## Files

### Sequential Implementation
- **`pearson.cpp`** - Sequential Pearson correlation calculation
- **`pearson.hpp`** - Header with function declarations

### Parallel Implementation
- **`pearson_parallel.cpp`** - OpenMP parallelized Pearson correlation
- **`pearson_parallel.hpp`** - Header with parallel function declarations

## Formula

```
correlation = Σ((A[i] - mean_A) × (B[i] - mean_B)) / (std_A × std_B)
```

Where:
- `mean_A`, `mean_B` = means of vectors A and B
- `std_A`, `std_B` = standard deviations
- Covariance divided by product of standard deviations

## Parallelization

The parallel version uses OpenMP reduction on multiple loops:

```cpp
// Compute means
#pragma omp parallel for reduction(+:sumA,sumB)
for (int i = 0; i < D; ++i) {
    sumA += A[i];
    sumB += B[i];
}

// Compute covariance and variances
#pragma omp parallel for reduction(+:cov,varA,varB)
for (int i = 0; i < D; ++i) {
    float diffA = A[i] - meanA;
    float diffB = B[i] - meanB;
    cov += diffA * diffB;
    varA += diffA * diffA;
    varB += diffB * diffB;
}
```

## Usage

Compiled as part of the main CPU executables.
