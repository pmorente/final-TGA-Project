#ifndef PEARSON_PARALLEL_HPP
#define PEARSON_PARALLEL_HPP

// Pearson Correlation with parallelized inner loops (OpenMP)
float pearson_corr_cpu_parallel(const float* __restrict A, const float* __restrict B, int D);

#endif // PEARSON_PARALLEL_HPP
