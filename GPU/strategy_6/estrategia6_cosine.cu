// ============================================================================
// STRATEGY 6 TEMPLATE: Grid over Big (B), Reduction over Dim
// Algorithm: Cosine Similarity (Hybrid Reduction)
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
__global__ void cosine_strategy6_template(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim)
{
    extern __shared__ float s[];
    float* s_db_vec = s;
    float* s_dot    = s_db_vec + dim;
    float* s_normA  = s_dot    + BLOCK_SIZE;
    
    int db_idx = blockIdx.x;
    if (db_idx >= num_database) return;
    
    const float* db_vec = database + db_idx * dim;
    int tid = threadIdx.x;
    
    // 1. Cargar B[j] en Shared Memory
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        s_db_vec[d] = db_vec[d];
    }
    __syncthreads();
    
    // 2. PRE-CALCULAR NORMA DE B (Optimización: Solo 1 vez por bloque)
    float thread_normB = 0.0f;
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        float b = s_db_vec[d];
        thread_normB += b * b;
    }
    // Usamos s_dot temporalmente para reducir
    s_dot[tid] = thread_normB;
    __syncthreads();
    
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) s_dot[tid] += s_dot[tid + s];
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *sm = s_dot;
        if (BLOCK_SIZE > 32) sm[tid] += sm[tid + 32];
        sm[tid] += sm[tid + 16]; sm[tid] += sm[tid + 8];
        sm[tid] += sm[tid + 4];  sm[tid] += sm[tid + 2]; sm[tid] += sm[tid + 1];
    }
    
    __shared__ float final_normB;
    if (tid == 0) final_normB = s_dot[0];
    __syncthreads();
    
    // 3. Loop sobre Queries (A)
    for (int q_idx = 0; q_idx < num_queries; q_idx++) {
        const float* query = queries + q_idx * dim;
        
        float thread_dot = 0.0f;
        float thread_normA = 0.0f;
        
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            float a = query[d];
            float b = s_db_vec[d];
            thread_dot += a * b;
            thread_normA += a * a;
        }
        
        s_dot[tid] = thread_dot;
        s_normA[tid] = thread_normA;
        __syncthreads();
        
        // Reducción Paralela
        for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
            if (tid < s) {
                s_dot[tid] += s_dot[tid + s];
                s_normA[tid] += s_normA[tid + s];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            volatile float *sm_dot = s_dot;
            volatile float *sm_nA = s_normA;
            if (BLOCK_SIZE > 32) {
                sm_dot[tid] += sm_dot[tid + 32];
                sm_nA[tid] += sm_nA[tid + 32];
            }
            sm_dot[tid] += sm_dot[tid + 16]; sm_dot[tid] += sm_dot[tid + 8];
            sm_dot[tid] += sm_dot[tid + 4];  sm_dot[tid] += sm_dot[tid + 2]; sm_dot[tid] += sm_dot[tid + 1];
            
            sm_nA[tid] += sm_nA[tid + 16]; sm_nA[tid] += sm_nA[tid + 8];
            sm_nA[tid] += sm_nA[tid + 4];  sm_nA[tid] += sm_nA[tid + 2]; sm_nA[tid] += sm_nA[tid + 1];
        }
        
        if (tid == 0) {
            float denom = sqrtf(s_normA[0]) * sqrtf(final_normB) + 1e-12f;
            results[q_idx * num_database + db_idx] = s_dot[0] / denom;
        }
        __syncthreads();
    }
}

void launch_cosine_template(int bs, const float* A, const float* B, float* C, int nA, int nB, int dim) {
    dim3 grid(nB); dim3 block(bs); size_t sh = (dim + 3 * bs) * sizeof(float);
    if (bs==32) cosine_strategy6_template<32><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==64) cosine_strategy6_template<64><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==128) cosine_strategy6_template<128><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==256) cosine_strategy6_template<256><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==512) cosine_strategy6_template<512><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==1024) cosine_strategy6_template<1024><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
}

std::vector<EmbeddingRecord> read_embeddings_binary(const std::string& file_path) {
    std::vector<EmbeddingRecord> records;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file");
    char magic[4]; file.read(magic, 4); uint8_t v; file.read((char*)&v, 1);
    int32_t num, d; file.read((char*)&num, 4); file.read((char*)&d, 4);
    for(int i=0; i<num; ++i) {
        EmbeddingRecord r; int32_t len; file.read((char*)&len, 4);
        std::vector<char> id(len); file.read(id.data(), len); r.id.assign(id.data(), len);
        r.embedding.resize(d); file.read((char*)r.embedding.data(), d*4);
        records.push_back(r);
    }
    return records;
}

int main(int argc, char* argv[]) {
    int bs = 256; if (argc >= 4) bs = std::stoi(argv[3]);
    try {
        auto rec_a = read_embeddings_binary(argv[1]); auto rec_b = read_embeddings_binary(argv[2]);
        int nA = rec_a.size(); int nB = rec_b.size(); int dim = rec_a[0].embedding.size();
        std::cout << "\n=== Strategy 6: 2D Tiled (TEMPLATE) ===" << std::endl;
        std::cout << "Algorithm: Cosine\nConfig: Block " << bs << std::endl;
        float *h_A = new float[nA*dim]; float *h_B = new float[nB*dim]; float *h_C = new float[nA*nB];
        for(int i=0; i<nA; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<nB; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nA*dim*4); cudaMalloc(&d_B, nB*dim*4); cudaMalloc(&d_C, nA*nB*4);
        cudaMemcpy(d_A, h_A, nA*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, nB*dim*4, cudaMemcpyHostToDevice);
        
        launch_cosine_template(bs, d_A, d_B, d_C, nA, nB, dim); cudaDeviceSynchronize();
        std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total = 0;
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize(); auto start = std::chrono::high_resolution_clock::now();
            launch_cosine_template(bs, d_A, d_B, d_C, nA, nB, dim);
            cudaDeviceSynchronize();
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
            total += diff.count();
            double tput = ((double)nA * nB) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        cudaMemcpy(h_C, d_C, nA*nB*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_cosine_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, nA*nB*4); f.close();
        double avg_time = total / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)nA*nB/avg_time/1e6) << " million comparisons/second" << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (...) { return 1; } return 0;
}