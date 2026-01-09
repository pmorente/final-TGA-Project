#include "euclidean_parallel.hpp"
#include <cmath>
#include <omp.h>

// ------------------ Euclidean Distance with parallelized inner loops ------------------
float euclidean_distance_cpu_parallel(const float* __restrict A, const float* __restrict B, int D) {
    double acc = 0.0;
    
    // Parallelize the loop over dimensions using OpenMP reduction
    #pragma omp parallel for reduction(+:acc)
    for (int i = 0; i < D; ++i) {
        double d = A[i] - B[i];
        acc += d * d;
    }
    
    return static_cast<float>(sqrt(acc));
}
