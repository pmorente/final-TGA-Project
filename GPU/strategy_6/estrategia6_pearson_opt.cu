// ============================================================================
// STRATEGY 6 TEMPLATE: Grid over Big (B), Reduction over Dim
// Algorithm: Pearson Correlation (Optimized One-Pass, Hybrid Reduction)
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
__global__ void pearson_strategy6_opt_template(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim)
{
    extern __shared__ float s[];
    float* s_db_vec = s;
    float* s_buff1  = s_db_vec + dim; // Reutilizable
    float* s_buff2  = s_buff1 + BLOCK_SIZE;
    float* s_buff3  = s_buff2 + BLOCK_SIZE;
    
    int db_idx = blockIdx.x;
    if (db_idx >= num_database) return;
    
    const float* db_vec = database + db_idx * dim;
    int tid = threadIdx.x;
    
    // 1. Cargar B[j] en Shared Memory
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        s_db_vec[d] = db_vec[d];
    }
    __syncthreads();
    
    // 2. PRE-CALCULAR Stats de B (One-Pass components)
    // Necesitamos SumB y SumSqB (solo una vez)
    float thread_sumB = 0.0f;
    float thread_sqB = 0.0f;
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        float val = s_db_vec[d];
        thread_sumB += val;
        thread_sqB  += val * val;
    }
    s_buff1[tid] = thread_sumB;
    s_buff2[tid] = thread_sqB;
    __syncthreads();
    
    // Reducción de Stats B
    for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
        if (tid < s) {
            s_buff1[tid] += s_buff1[tid + s];
            s_buff2[tid] += s_buff2[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        volatile float *sm1 = s_buff1; volatile float *sm2 = s_buff2;
        if (BLOCK_SIZE > 32) { sm1[tid]+=sm1[tid+32]; sm2[tid]+=sm2[tid+32]; }
        sm1[tid]+=sm1[tid+16]; sm1[tid]+=sm1[tid+8]; sm1[tid]+=sm1[tid+4]; sm1[tid]+=sm1[tid+2]; sm1[tid]+=sm1[tid+1];
        sm2[tid]+=sm2[tid+16]; sm2[tid]+=sm2[tid+8]; sm2[tid]+=sm2[tid+4]; sm2[tid]+=sm2[tid+2]; sm2[tid]+=sm2[tid+1];
    }
    
    __shared__ float total_sumB, total_sqB;
    if (tid == 0) {
        total_sumB = s_buff1[0];
        total_sqB  = s_buff2[0];
    }
    __syncthreads();
    
    // 3. Loop sobre Queries (A) - One Pass
    for (int q_idx = 0; q_idx < num_queries; q_idx++) {
        const float* query = queries + q_idx * dim;
        
        float thread_sumA = 0.0f;
        float thread_sqA  = 0.0f;
        float thread_prod = 0.0f;
        
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            float valA = query[d];
            float valB = s_db_vec[d];
            
            thread_sumA += valA;
            thread_sqA  += valA * valA;
            thread_prod += valA * valB;
        }
        
        s_buff1[tid] = thread_sumA;
        s_buff2[tid] = thread_sqA;
        s_buff3[tid] = thread_prod;
        __syncthreads();
        
        // Reducción Triple
        for (unsigned int s = BLOCK_SIZE / 2; s > 32; s >>= 1) {
            if (tid < s) {
                s_buff1[tid] += s_buff1[tid + s];
                s_buff2[tid] += s_buff2[tid + s];
                s_buff3[tid] += s_buff3[tid + s];
            }
            __syncthreads();
        }
        if (tid < 32) {
            volatile float *sm1 = s_buff1; volatile float *sm2 = s_buff2; volatile float *sm3 = s_buff3;
            if (BLOCK_SIZE > 32) {
                sm1[tid]+=sm1[tid+32]; sm2[tid]+=sm2[tid+32]; sm3[tid]+=sm3[tid+32];
            }
            sm1[tid]+=sm1[tid+16]; sm1[tid]+=sm1[tid+8]; sm1[tid]+=sm1[tid+4]; sm1[tid]+=sm1[tid+2]; sm1[tid]+=sm1[tid+1];
            sm2[tid]+=sm2[tid+16]; sm2[tid]+=sm2[tid+8]; sm2[tid]+=sm2[tid+4]; sm2[tid]+=sm2[tid+2]; sm2[tid]+=sm2[tid+1];
            sm3[tid]+=sm3[tid+16]; sm3[tid]+=sm3[tid+8]; sm3[tid]+=sm3[tid+4]; sm3[tid]+=sm3[tid+2]; sm3[tid]+=sm3[tid+1];
        }
        
        if (tid == 0) {
            float sumA  = s_buff1[0];
            float sqA   = s_buff2[0];
            float prod  = s_buff3[0];
            
            float num = dim * prod - sumA * total_sumB;
            float denA = dim * sqA - sumA * sumA;
            float denB = dim * total_sqB - total_sumB * total_sumB;
            
            float denom = sqrtf(denA) * sqrtf(denB);
            if (denom > 1e-12f)
                results[q_idx * num_database + db_idx] = num / denom;
            else
                results[q_idx * num_database + db_idx] = 0.0f;
        }
        __syncthreads();
    }
}

void launch_pearson_template(int bs, const float* A, const float* B, float* C, int nA, int nB, int dim) {
    dim3 grid(nB); dim3 block(bs);
    // Shared: dim + 3*bs
    size_t sh = (dim + 3 * bs) * sizeof(float);
    
    if (bs==32) pearson_strategy6_opt_template<32><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==64) pearson_strategy6_opt_template<64><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==128) pearson_strategy6_opt_template<128><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==256) pearson_strategy6_opt_template<256><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==512) pearson_strategy6_opt_template<512><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==1024) pearson_strategy6_opt_template<1024><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
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
        std::cout << "\n=== Strategy 6: 2D Tiled (TEMPLATE HYBRID) ===" << std::endl;
        std::cout << "Algorithm: Pearson (Opt One-Pass)\nConfig: Block " << bs << std::endl;
        float *h_A = new float[nA*dim]; float *h_B = new float[nB*dim]; float *h_C = new float[nA*nB];
        for(int i=0; i<nA; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<nB; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nA*dim*4); cudaMalloc(&d_B, nB*dim*4); cudaMalloc(&d_C, nA*nB*4);
        cudaMemcpy(d_A, h_A, nA*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, nB*dim*4, cudaMemcpyHostToDevice);
        
        launch_pearson_template(bs, d_A, d_B, d_C, nA, nB, dim); cudaDeviceSynchronize();
                std::cout << "Running benchmark (10 iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total = 0;
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize(); auto start = std::chrono::high_resolution_clock::now();
            launch_pearson_template(bs, d_A, d_B, d_C, nA, nB, dim);
            cudaDeviceSynchronize();
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
            total += diff.count();
            double tput = ((double)nA * nB) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        cudaMemcpy(h_C, d_C, nA*nB*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_pearson_opt_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, nA*nB*4); f.close();
        double avg_time = total / 10.0;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << ((double)nA*nB/avg_time/1e6) << " million comparisons/second" << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (...) { return 1; } return 0;
}