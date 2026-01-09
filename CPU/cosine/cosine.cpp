#include "cosine.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

// ------------------ CPU cosine similarity -------------------
float cosine_cpu(const std::vector<float> &A, const std::vector<float> &B) {
    if (A.size() != B.size()) throw std::runtime_error("Vector sizes differ");
    double dot = 0.0, normA = 0.0, normB = 0.0;
    int D = A.size();
    for (int i = 0; i < D; ++i) {
        double ai = A[i], bi = B[i];
        dot += ai * bi;
        normA += ai * ai;
        normB += bi * bi;
    }
    double denom = sqrt(normA) * sqrt(normB) + 1e-12;
    return static_cast<float>(dot / denom);
}
