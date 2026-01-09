#include "pearson_parallel.hpp"
#include <cmath>
#include <omp.h>

// ------------------ Pearson Correlation with parallelized inner loops ------------------
float pearson_corr_cpu_parallel(const float* __restrict A, const float* __restrict B, int D) {
    double meanA = 0.0, meanB = 0.0;
    
    // First pass: compute means (parallelized)
    #pragma omp parallel for reduction(+:meanA,meanB)
    for (int i = 0; i < D; ++i) {
        meanA += A[i];
        meanB += B[i];
    }
    meanA /= D;
    meanB /= D;
    
    // Second pass: compute correlation (parallelized)
    double num = 0.0, denA = 0.0, denB = 0.0;
    #pragma omp parallel for reduction(+:num,denA,denB)
    for (int i = 0; i < D; ++i) {
        double da = A[i] - meanA;
        double db = B[i] - meanB;
        num  += da * db;
        denA += da * da;
        denB += db * db;
    }
    
    double denom = sqrt(denA * denB) + 1e-12;
    return static_cast<float>(num / denom);
}
