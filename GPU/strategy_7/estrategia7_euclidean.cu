// ============================================================================
// STRATEGY 7 — 2D TILED + REDUCTION (Dynamic Shared Memory)
// Algorithm: Euclidean Distance
// ============================================================================

#include <cuda_runtime.h>
#include <cmath>
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cstring>
#include <cstdint>
#include <chrono>
#include <iomanip>

struct EmbeddingRecord {
    std::string id;
    std::vector<float> embedding;
};

__global__ void euclidean_strategy7(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int num_A, int num_B, int dim) {
    int TILE = blockDim.x; 
    int row = blockIdx.y * TILE + threadIdx.y; 
    int col = blockIdx.x * TILE + threadIdx.x; 
    float acc = 0.0f;

    extern __shared__ float s[];
    float* As = s;
    float* Bs = s + TILE * TILE;

    for (int m = 0; m < (dim + TILE - 1) / TILE; ++m) {
        int t_dim_A = m * TILE + threadIdx.x;
        if (row < num_A && t_dim_A < dim) As[threadIdx.y * TILE + threadIdx.x] = A[row * dim + t_dim_A];
        else As[threadIdx.y * TILE + threadIdx.x] = 0.0f;

        int t_col = blockIdx.x * TILE + threadIdx.x;
        int t_dim_B = m * TILE + threadIdx.y;
        if (t_col < num_B && t_dim_B < dim) Bs[threadIdx.x * TILE + threadIdx.y] = B[t_col * dim + t_dim_B];
        else Bs[threadIdx.x * TILE + threadIdx.y] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE; ++k) {
            float diff = As[threadIdx.y * TILE + k] - Bs[threadIdx.x * TILE + k];
            acc += diff * diff;
        }
        __syncthreads();
    }

    if (row < num_A && col < num_B) C[row * num_B + col] = sqrtf(acc);
}

void launch_euclidean_strategy7(int tile_size, const float* d_A, const float* d_B, float* d_C, int num_A, int num_B, int dim) {
    dim3 grid((num_B + tile_size - 1) / tile_size, (num_A + tile_size - 1) / tile_size);
    dim3 block(tile_size, tile_size);
    size_t shmem = 2 * tile_size * tile_size * sizeof(float);
    euclidean_strategy7<<<grid, block, shmem>>>(d_A, d_B, d_C, num_A, num_B, dim);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) std::cerr << "Kernel launch error: " << cudaGetErrorString(err) << std::endl;
}

std::vector<EmbeddingRecord> read_embeddings_binary(const std::string& file_path) {
    std::vector<EmbeddingRecord> records;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + file_path);
    char magic[4]; file.read(magic, 4);
    uint8_t version; file.read((char*)&version, 1);
    int32_t num, dim; file.read((char*)&num, 4); file.read((char*)&dim, 4);
    for (int i = 0; i < num; ++i) {
        EmbeddingRecord r; int32_t len; file.read((char*)&len, 4);
        std::vector<char> id(len); file.read(id.data(), len); r.id.assign(id.data(), len);
        r.embedding.resize(dim); file.read((char*)r.embedding.data(), dim * 4);
        records.push_back(r);
    }
    return records;
}

int main(int argc, char* argv[]) {
    int tile_size = 16;
    if (argc < 3) { std::cerr << "Usage: " << argv[0] << " <file_a> <file_b> [tile_size]" << std::endl; return 1; }
    if (argc >= 4) tile_size = std::stoi(argv[3]);

    std::string file_a = argv[1]; std::string file_b = argv[2];
    try {
        auto rec_a = read_embeddings_binary(file_a); auto rec_b = read_embeddings_binary(file_b);
        int num_queries = rec_a.size(); int num_database = rec_b.size(); int dim = rec_a[0].embedding.size();
        
        std::cout << "\n=== Strategy 7: 2D Tiled (DYNAMIC) ===" << std::endl;
        std::cout << "Algorithm: Euclidean" << std::endl;
        std::cout << "Configuration: Tile Size = " << tile_size << "x" << tile_size << " (" << tile_size*tile_size << " threads/block)" << std::endl;

        float *h_A = new float[num_queries*dim]; float *h_B = new float[num_database*dim]; float *h_C = new float[num_queries*num_database];
        for(int i=0; i<num_queries; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<num_database; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, num_queries*dim*4); cudaMalloc(&d_B, num_database*dim*4); cudaMalloc(&d_C, num_queries*num_database*4);
        cudaMemcpy(d_A, h_A, num_queries*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, num_database*dim*4, cudaMemcpyHostToDevice);
        
        std::cout << "\nWarming up GPU (1 run)..." << std::endl;
        launch_euclidean_strategy7(tile_size, d_A, d_B, d_C, num_queries, num_database, dim);
        cudaDeviceSynchronize();

        std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total_time = 0.0;
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();
            launch_euclidean_strategy7(tile_size, d_A, d_B, d_C, num_queries, num_database, dim);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            total_time += diff.count();
            double tput = ((double)num_queries * num_database) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s  (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_C, d_C, num_queries*num_database*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_euclidean_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, num_queries*num_database*4); f.close();
        
        double avg_time = total_time / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)num_queries*num_database/avg_time/1e6) << " million comparisons/second" << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
    return 0;
}