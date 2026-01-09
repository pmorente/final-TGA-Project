// ============================================================================
// STRATEGY 7 — 2D TILED + REDUCTION (Template)
// Algorithm: Cosine Similarity
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

template <int TILE>
__global__ void cosine_strategy7_template(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int num_A, int num_B, int dim) {
    int row = blockIdx.y * TILE + threadIdx.y; 
    int col = blockIdx.x * TILE + threadIdx.x; 
    float dot = 0.0f, nA = 0.0f, nB = 0.0f;

    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    for (int m = 0; m < (dim + TILE - 1) / TILE; ++m) {
        int t_dim_A = m * TILE + threadIdx.x;
        if (row < num_A && t_dim_A < dim) As[threadIdx.y][threadIdx.x] = A[row * dim + t_dim_A];
        else As[threadIdx.y][threadIdx.x] = 0.0f;

        int t_col = blockIdx.x * TILE + threadIdx.x;
        int t_dim_B = m * TILE + threadIdx.y;
        if (t_col < num_B && t_dim_B < dim) Bs[threadIdx.x][threadIdx.y] = B[t_col * dim + t_dim_B];
        else Bs[threadIdx.x][threadIdx.y] = 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            float valA = As[threadIdx.y][k];
            float valB = Bs[threadIdx.x][k];
            dot += valA * valB;
            nA  += valA * valA;
            nB  += valB * valB;
        }
        __syncthreads();
    }

    if (row < num_A && col < num_B) {
        float denom = sqrtf(nA) * sqrtf(nB) + 1e-12f;
        C[row * num_B + col] = dot / denom;
    }
}

void launch_cosine_template(int tile_size, const float* d_A, const float* d_B, float* d_C, int num_A, int num_B, int dim) {
    if (tile_size == 4) {
        dim3 grid((num_B+3)/4, (num_A+3)/4); dim3 block(4,4);
        cosine_strategy7_template<4><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else if (tile_size == 8) {
        dim3 grid((num_B+7)/8, (num_A+7)/8); dim3 block(8,8);
        cosine_strategy7_template<8><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else if (tile_size == 16) {
        dim3 grid((num_B+15)/16, (num_A+15)/16); dim3 block(16,16);
        cosine_strategy7_template<16><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else if (tile_size == 30) {
        dim3 grid((num_B+29)/30, (num_A+29)/30); dim3 block(30,30);
        cosine_strategy7_template<30><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else if (tile_size == 31) {
        dim3 grid((num_B+30)/31, (num_A+30)/31); dim3 block(31,31);
        cosine_strategy7_template<31><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else if (tile_size == 32) {
        dim3 grid((num_B+31)/32, (num_A+31)/32); dim3 block(32,32);
        cosine_strategy7_template<32><<<grid, block>>>(d_A, d_B, d_C, num_A, num_B, dim);
    } else { std::cerr << "Error: Template supports 4, 8, 16, 30, 31, 32." << std::endl; exit(1); }
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
        
        std::cout << "\n=== Strategy 7: 2D Tiled (TEMPLATE) ===" << std::endl;
        std::cout << "Algorithm: Cosine Similarity" << std::endl;
        std::cout << "Configuration: Tile " << tile_size << "x" << tile_size << " (" << tile_size*tile_size << " threads/block)" << std::endl;

        float *h_A = new float[num_queries*dim]; float *h_B = new float[num_database*dim]; float *h_C = new float[num_queries*num_database];
        for(int i=0; i<num_queries; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<num_database; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, num_queries*dim*4); cudaMalloc(&d_B, num_database*dim*4); cudaMalloc(&d_C, num_queries*num_database*4);
        cudaMemcpy(d_A, h_A, num_queries*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, num_database*dim*4, cudaMemcpyHostToDevice);
        
        std::cout << "Warming up GPU (1 run)..." << std::endl;
        launch_cosine_template(tile_size, d_A, d_B, d_C, num_queries, num_database, dim);
        cudaDeviceSynchronize();

        std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total_time = 0.0;
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();
            launch_cosine_template(tile_size, d_A, d_B, d_C, num_queries, num_database, dim);
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            total_time += diff.count();
            double tput = ((double)num_queries * num_database) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s  (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_C, d_C, num_queries*num_database*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_cosine_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, num_queries*num_database*4); f.close();
        
        double avg_time = total_time / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)num_queries*num_database/avg_time/1e6) << " million comparisons/second" << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
    return 0;
}