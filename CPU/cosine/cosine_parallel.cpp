#include "cosine_parallel.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>
#include <omp.h>

// ------------------ CPU cosine similarity with parallelized inner loops -------------------
float cosine_cpu_parallel(const std::vector<float> &A, const std::vector<float> &B) {
    if (A.size() != B.size()) throw std::runtime_error("Vector sizes differ");
    
    int D = A.size();
    double dot = 0.0, normA = 0.0, normB = 0.0;
    
    // Parallelize the loop over dimensions using OpenMP reduction
    #pragma omp parallel for reduction(+:dot,normA,normB)
    for (int i = 0; i < D; ++i) {
        double ai = A[i], bi = B[i];
        dot += ai * bi;
        normA += ai * ai;
        normB += bi * bi;
    }
    
    double denom = sqrt(normA) * sqrt(normB) + 1e-12;
    return static_cast<float>(dot / denom);
}
