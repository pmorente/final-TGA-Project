#ifndef COSINE_PARALLEL_HPP
#define COSINE_PARALLEL_HPP

#include <vector>

// CPU cosine similarity with parallelized inner loops (OpenMP)
float cosine_cpu_parallel(const std::vector<float> &A, const std::vector<float> &B);

#endif // COSINE_PARALLEL_HPP
