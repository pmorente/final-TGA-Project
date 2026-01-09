#include "euclidean.hpp"
#include <cmath>

// ------------------ Distancia Euclidiana ------------------
float euclidean_distance_cpu(const float* __restrict A, const float* __restrict B, int D) {
    double acc = 0.0;
    int i = 0;

    // Loop unrolling para optimización
    for (; i + 4 <= D; i += 4) {
        double d0 = A[i]   - B[i];
        double d1 = A[i+1] - B[i+1];
        double d2 = A[i+2] - B[i+2];
        double d3 = A[i+3] - B[i+3];
        acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    for (; i < D; ++i) {
        double d = A[i] - B[i];
        acc += d * d;
    }

    return static_cast<float>(sqrt(acc));
}
