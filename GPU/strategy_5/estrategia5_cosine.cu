// ============================================================================
// STRATEGY 5 TEMPLATE: Grid over Small (A), Reduction over Dim
// Algorithm: Cosine Similarity
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
__global__ void cosine_strategy5_template(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim)
{
    extern __shared__ float s[];
    float* s_query = s;
    float* s_dot   = s_query + dim;
    float* s_normA = s_dot   + BLOCK_SIZE;
    float* s_normB = s_normA + BLOCK_SIZE;
    
    int q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;
    
    const float* query = queries + q_idx * dim;
    int tid = threadIdx.x;
    
    // Cooperatively load A[i] into shared memory
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        s_query[d] = query[d];
    }
    __syncthreads();
    
    // For each B[j]
    for (int db_idx = 0; db_idx < num_database; db_idx++) {
        const float* db_vec = database + db_idx * dim;
        
        float thread_dot = 0.0f;
        float thread_normA = 0.0f;
        float thread_normB = 0.0f;
        
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            float a = s_query[d];
            float b = db_vec[d];
            thread_dot += a * b;
            thread_normA += a * a;
            thread_normB += b * b;
        }
        
        s_dot[tid] = thread_dot;
        s_normA[tid] = thread_normA;
        s_normB[tid] = thread_normB;
        __syncthreads();
        
        // Block reduction
        for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
            if (tid < s) {
                s_dot[tid] += s_dot[tid + s];
                s_normA[tid] += s_normA[tid + s];
                s_normB[tid] += s_normB[tid + s];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            volatile float *smem_dot = s_dot;
            volatile float *smem_normA = s_normA;
            volatile float *smem_normB = s_normB;
            
            if (BLOCK_SIZE > 32) {
                smem_dot[tid] += smem_dot[tid + 32];
                smem_normA[tid] += smem_normA[tid + 32];
                smem_normB[tid] += smem_normB[tid + 32];
            }

            smem_dot[tid] += smem_dot[tid + 16]; smem_dot[tid] += smem_dot[tid + 8];
            smem_dot[tid] += smem_dot[tid + 4];  smem_dot[tid] += smem_dot[tid + 2]; smem_dot[tid] += smem_dot[tid + 1];
            
            smem_normA[tid] += smem_normA[tid + 16]; smem_normA[tid] += smem_normA[tid + 8];
            smem_normA[tid] += smem_normA[tid + 4];  smem_normA[tid] += smem_normA[tid + 2]; smem_normA[tid] += smem_normA[tid + 1];
            
            smem_normB[tid] += smem_normB[tid + 16]; smem_normB[tid] += smem_normB[tid + 8];
            smem_normB[tid] += smem_normB[tid + 4];  smem_normB[tid] += smem_normB[tid + 2]; smem_normB[tid] += smem_normB[tid + 1];
        }
        
        if (tid == 0) {
            float denom = sqrtf(s_normA[0]) * sqrtf(s_normB[0]) + 1e-12f;
            results[q_idx * num_database + db_idx] = s_dot[0] / denom;
        }
        
        __syncthreads();
    }
}

void launch_cosine_template(int bs, const float* A, const float* B, float* C, int nA, int nB, int dim) {
    dim3 grid(nA); dim3 block(bs); size_t sh = (dim + 3 * bs) * sizeof(float);
    if (bs==32) cosine_strategy5_template<32><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==64) cosine_strategy5_template<64><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==128) cosine_strategy5_template<128><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==256) cosine_strategy5_template<256><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==512) cosine_strategy5_template<512><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==1024) cosine_strategy5_template<1024><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else std::cerr << "Error: Block size must be power of 2" << std::endl;
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
        std::cout << "Algorithm: Cosine Similarity" << std::endl;
        std::cout << "Configuration: Block Size = " << bs << " (Static)" << std::endl;

        float *h_A = new float[nA*dim]; float *h_B = new float[nB*dim]; float *h_C = new float[nA*nB];
        for(int i=0; i<nA; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<nB; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nA*dim*4); cudaMalloc(&d_B, nB*dim*4); cudaMalloc(&d_C, nA*nB*4);
        cudaMemcpy(d_A, h_A, nA*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, nB*dim*4, cudaMemcpyHostToDevice);
        
        std::cout << "\nWarming up GPU (1 run)..." << std::endl;
        launch_cosine_template(bs, d_A, d_B, d_C, nA, nB, dim); cudaDeviceSynchronize();
        
        std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total_time = 0.0;
        
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize();
            auto start = std::chrono::high_resolution_clock::now();
            
            launch_cosine_template(bs, d_A, d_B, d_C, nA, nB, dim);
            
            cudaDeviceSynchronize();
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            total_time += diff.count();
            
            double tput = ((double)nA * nB) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s  (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_C, d_C, nA*nB*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_cosine_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, nA*nB*4); f.close();
        
        double avg_time = total_time / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)nA*nB/avg_time/1e6) << " million comparisons/second" << std::endl;

        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
    return 0;
}