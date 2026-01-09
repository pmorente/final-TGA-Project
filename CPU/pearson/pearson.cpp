#include "pearson.hpp"
#include <cmath>

// ------------------ Pearson Correlation ------------------
float pearson_corr_cpu(const float* __restrict A, const float* __restrict B, int D) {
    double meanA = 0.0, meanB = 0.0;
    for (int i = 0; i < D; ++i) {
        meanA += A[i];
        meanB += B[i];
    }
    meanA /= D;
    meanB /= D;

    double num = 0.0, denA = 0.0, denB = 0.0;
    int i = 0;
    for (; i + 4 <= D; i += 4) {
        double da0 = A[i]   - meanA;  double db0 = B[i]   - meanB;
        double da1 = A[i+1] - meanA;  double db1 = B[i+1] - meanB;
        double da2 = A[i+2] - meanA;  double db2 = B[i+2] - meanB;
        double da3 = A[i+3] - meanA;  double db3 = B[i+3] - meanB;
        num  += da0*db0 + da1*db1 + da2*db2 + da3*db3;
        denA += da0*da0 + da1*da1 + da2*da2 + da3*da3;
        denB += db0*db0 + db1*db1 + db2*db2 + db3*db3;
    }
    for (; i < D; ++i) {
        double da = A[i] - meanA;
        double db = B[i] - meanB;
        num  += da * db;
        denA += da * da;
        denB += db * db;
    }

    double denom = sqrt(denA * denB) + 1e-12;
    return static_cast<float>(num / denom);
}
