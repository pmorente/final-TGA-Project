// ============================================================================
// STRATEGY 5 TEMPLATE: Grid over Small (A), Reduction over Dim
// Algorithm: Euclidean Distance
// ============================================================================

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <cstring>
#include <cmath>
#include <iomanip>
#include <chrono>

struct EmbeddingRecord { std::string id; std::vector<float> embedding; };

template <int BLOCK_SIZE>
__global__ void euclidean_strategy5_template(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim)
{
    extern __shared__ float s[];
    float* s_query = s;
    float* s_acc = s + dim;
    
    int q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;
    
    const float* query = queries + q_idx * dim;
    int tid = threadIdx.x;
    
    // Load A[i] into shared memory
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        s_query[d] = query[d];
    }
    __syncthreads();
    
    // For each B[j]
    for (int db_idx = 0; db_idx < num_database; db_idx++) {
        const float* db_vec = database + db_idx * dim;
        
        float thread_acc = 0.0f;
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            float diff = s_query[d] - db_vec[d];
            thread_acc += diff * diff;
        }
        
        s_acc[tid] = thread_acc;
        __syncthreads();
        
        // Block reduction
        for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
            if (tid < s) {
                s_acc[tid] += s_acc[tid + s];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            volatile float *smem = s_acc;
            if (BLOCK_SIZE > 32) smem[tid] += smem[tid + 32];
            smem[tid] += smem[tid + 16];
            smem[tid] += smem[tid + 8];
            smem[tid] += smem[tid + 4];
            smem[tid] += smem[tid + 2];
            smem[tid] += smem[tid + 1];
        }
        
        if (tid == 0) {
            results[q_idx * num_database + db_idx] = sqrtf(s_acc[0]);
        }
        
        __syncthreads();
    }
}

void launch_euclidean_template(int bs, const float* A, const float* B, float* C, int nA, int nB, int dim) {
    dim3 grid(nA); dim3 block(bs); size_t sh = (dim + bs) * sizeof(float);
    if (bs==32) euclidean_strategy5_template<32><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==64) euclidean_strategy5_template<64><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==128) euclidean_strategy5_template<128><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==256) euclidean_strategy5_template<256><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==512) euclidean_strategy5_template<512><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==1024) euclidean_strategy5_template<1024><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else std::cerr << "Error: Block size must be power of 2 (32-1024)" << std::endl;
}

std::vector<EmbeddingRecord> read_embeddings_binary(const std::string& file_path) {
    std::vector<EmbeddingRecord> records;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + file_path);
    char magic[4]; file.read(magic, 4);
    uint8_t version; file.read((char*)&version, 1);
    int32_t num, dim_sz; file.read((char*)&num, 4); file.read((char*)&dim_sz, 4);
    for (int i = 0; i < num; ++i) {
        EmbeddingRecord r; int32_t len; file.read((char*)&len, 4);
        std::vector<char> id(len); file.read(id.data(), len); r.id.assign(id.data(), len);
        r.embedding.resize(dim_sz); file.read((char*)r.embedding.data(), dim_sz * 4);
        records.push_back(r);
    }
    return records;
}

int main(int argc, char* argv[]) {
    int bs = 256;
    if (argc >= 4) bs = std::stoi(argv[3]);
    try {
        auto rec_a = read_embeddings_binary(argv[1]); auto rec_b = read_embeddings_binary(argv[2]);
        int nA = rec_a.size(); int nB = rec_b.size(); int dim = rec_a[0].embedding.size();
        
        std::cout << "\n=== Strategy 5: 2D Tiled (TEMPLATE) ===" << std::endl;
        std::cout << "Algorithm: Euclidean" << std::endl;
        std::cout << "Configuration: Block Size = " << bs << " (Static)" << std::endl;

        float *h_A = new float[nA*dim]; float *h_B = new float[nB*dim]; float *h_C = new float[nA*nB];
        for(int i=0; i<nA; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<nB; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nA*dim*4); cudaMalloc(&d_B, nB*dim*4); cudaMalloc(&d_C, nA*nB*4);
        cudaMemcpy(d_A, h_A, nA*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, nB*dim*4, cudaMemcpyHostToDevice);
        
        std::cout << "\nWarming up GPU (1 run)..." << std::endl;
        launch_euclidean_template(bs, d_A, d_B, d_C, nA, nB, dim); cudaDeviceSynchronize();
        
        std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total_time = 0.0;
        
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();
            
            launch_euclidean_template(bs, d_A, d_B, d_C, nA, nB, dim);
            
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            total_time += diff.count();
            
            double tput = ((double)nA * nB) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s  (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_C, d_C, nA*nB*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_euclidean_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, nA*nB*4); f.close();
        
        double avg_time = total_time / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)nA*nB/avg_time/1e6) << " million comparisons/second" << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
    return 0;
}