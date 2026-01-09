#ifndef EUCLIDEAN_PARALLEL_HPP
#define EUCLIDEAN_PARALLEL_HPP

// Euclidean Distance with parallelized inner loops (OpenMP)
float euclidean_distance_cpu_parallel(const float* __restrict A, const float* __restrict B, int D);

#endif // EUCLIDEAN_PARALLEL_HPP
