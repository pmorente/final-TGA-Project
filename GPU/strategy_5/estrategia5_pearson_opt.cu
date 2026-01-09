// ============================================================================
// STRATEGY 5 TEMPLATE: Grid over Small (A) - Pearson Opt (Conservative)
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
__global__ void pearson_strategy5_opt_template(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim)
{
    // Shared Memory Layout:
    // [0 ... dim-1]                -> s_query (Vector A)
    // [dim ... dim+BS-1]           -> s_sumB  (Reutilizado para suma A)
    // [dim+BS ... dim+2*BS-1]      -> s_sumSqB (Reutilizado para sumaSq A)
    // [dim+2*BS ... dim+3*BS-1]    -> s_sumAB
    extern __shared__ float s[];
    float* s_query  = s;
    float* s_sumB   = s_query + dim;
    float* s_sumSqB = s_sumB + BLOCK_SIZE;
    float* s_sumAB  = s_sumSqB + BLOCK_SIZE;
    
    int q_idx = blockIdx.x;
    if (q_idx >= num_queries) return;
    
    const float* query = queries + q_idx * dim;
    int tid = threadIdx.x;
    
    // =========================================================
    // FASE 1: Cargar A y Pre-calcular sus estadísticas
    // =========================================================
    float thread_sumA = 0.0f;
    float thread_sumSqA = 0.0f;
    
    for (int d = tid; d < dim; d += BLOCK_SIZE) {
        float val = query[d];
        s_query[d] = val; 
        thread_sumA += val;
        thread_sumSqA += val * val;
    }
    
    s_sumB[tid] = thread_sumA;
    s_sumSqB[tid] = thread_sumSqA;
    __syncthreads();
    
    // Reducción de A
    for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            s_sumB[tid]   += s_sumB[tid + stride];
            s_sumSqB[tid] += s_sumSqB[tid + stride];
        }
        __syncthreads();
    }
    
    if (tid < 32) {
        volatile float* sm1 = s_sumB;
        volatile float* sm2 = s_sumSqB;
        if (BLOCK_SIZE > 32) {
            sm1[tid] += sm1[tid + 32];
            sm2[tid] += sm2[tid + 32];
        }
        sm1[tid] += sm1[tid + 16]; sm1[tid] += sm1[tid + 8];  
        sm1[tid] += sm1[tid + 4];  sm1[tid] += sm1[tid + 2];  sm1[tid] += sm1[tid + 1];
        sm2[tid] += sm2[tid + 16]; sm2[tid] += sm2[tid + 8];  
        sm2[tid] += sm2[tid + 4];  sm2[tid] += sm2[tid + 2];  sm2[tid] += sm2[tid + 1];
    }
    
    float total_sumA, total_sumSqA;
    if (tid == 0) {
        total_sumA = s_sumB[0];
        total_sumSqA = s_sumSqB[0];
    }
    __syncthreads(); 
    
    // =========================================================
    // FASE 2: Bucle principal sobre Base de Datos (B)
    // =========================================================
    
    for (int db_idx = 0; db_idx < num_database; db_idx++) {
        const float* db_vec = database + db_idx * dim;
        
        float thread_sumB = 0.0f;
        float thread_sumSqB = 0.0f;
        float thread_sumAB = 0.0f;
        
        for (int d = tid; d < dim; d += BLOCK_SIZE) {
            float val_a = s_query[d]; 
            float val_b = db_vec[d];
            
            thread_sumB   += val_b;
            thread_sumSqB += val_b * val_b;
            thread_sumAB  += val_a * val_b;
        }
        
        s_sumB[tid]   = thread_sumB;
        s_sumSqB[tid] = thread_sumSqB;
        s_sumAB[tid]  = thread_sumAB;
        __syncthreads();
        
        // Reducción Triple
        for (unsigned int stride = BLOCK_SIZE / 2; stride > 32; stride >>= 1) {
            if (tid < stride) {
                s_sumB[tid]   += s_sumB[tid + stride];
                s_sumSqB[tid] += s_sumSqB[tid + stride];
                s_sumAB[tid]  += s_sumAB[tid + stride];
            }
            __syncthreads();
        }
        
        if (tid < 32) {
            volatile float* sm1 = s_sumB;
            volatile float* sm2 = s_sumSqB;
            volatile float* sm3 = s_sumAB;
            
            if (BLOCK_SIZE > 32) {
                sm1[tid] += sm1[tid + 32];
                sm2[tid] += sm2[tid + 32];
                sm3[tid] += sm3[tid + 32];
            }
            sm1[tid] += sm1[tid + 16]; sm1[tid] += sm1[tid + 8]; 
            sm1[tid] += sm1[tid + 4];  sm1[tid] += sm1[tid + 2];  sm1[tid] += sm1[tid + 1];
            sm2[tid] += sm2[tid + 16]; sm2[tid] += sm2[tid + 8];
            sm2[tid] += sm2[tid + 4];  sm2[tid] += sm2[tid + 2];  sm2[tid] += sm2[tid + 1];
            sm3[tid] += sm3[tid + 16]; sm3[tid] += sm3[tid + 8];
            sm3[tid] += sm3[tid + 4];  sm3[tid] += sm3[tid + 2];  sm3[tid] += sm3[tid + 1];
        }
        
        if (tid == 0) {
            float sumB   = s_sumB[0];
            float sumSqB = s_sumSqB[0];
            float sumAB  = s_sumAB[0];
            float N      = (float)dim;
            
            float num = N * sumAB - total_sumA * sumB;
            float varA = N * total_sumSqA - total_sumA * total_sumA;
            float varB = N * sumSqB - sumB * sumB;
            float den = sqrtf(varA * varB);
            
            if (den > 1e-12f) results[q_idx * num_database + db_idx] = num / den;
            else results[q_idx * num_database + db_idx] = 0.0f;
        }
        __syncthreads();
    }
}

void launch_pearson_template(int bs, const float* A, const float* B, float* C, int nA, int nB, int dim) {
    dim3 grid(nA); dim3 block(bs); size_t sh = (dim + 5 * bs) * sizeof(float);
    
    if (bs==32) pearson_strategy5_opt_template<32><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==64) pearson_strategy5_opt_template<64><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==128) pearson_strategy5_opt_template<128><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==256) pearson_strategy5_opt_template<256><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==512) pearson_strategy5_opt_template<512><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else if (bs==1024) pearson_strategy5_opt_template<1024><<<grid, block, sh>>>(A,B,C,nA,nB,dim);
    else std::cerr << "Error: Block size must be power of 2" << std::endl;
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
    int bs = 256;
    if (argc >= 4) bs = std::stoi(argv[3]);
    try {
        auto rec_a = read_embeddings_binary(argv[1]); auto rec_b = read_embeddings_binary(argv[2]);
        int nA = rec_a.size(); int nB = rec_b.size(); int dim = rec_a[0].embedding.size();
        
        std::cout << "\n=== Strategy 5: 2D Tiled (TEMPLATE) ===" << std::endl;
        std::cout << "Algorithm: Pearson (Opt One-Pass)" << std::endl;
        std::cout << "Configuration: Block Size = " << bs << " (Static)" << std::endl;

        float *h_A = new float[nA*dim]; float *h_B = new float[nB*dim]; float *h_C = new float[nA*nB];
        for(int i=0; i<nA; i++) memcpy(h_A + i*dim, rec_a[i].embedding.data(), dim*4);
        for(int i=0; i<nB; i++) memcpy(h_B + i*dim, rec_b[i].embedding.data(), dim*4);
        float *d_A, *d_B, *d_C;
        cudaMalloc(&d_A, nA*dim*4); cudaMalloc(&d_B, nB*dim*4); cudaMalloc(&d_C, nA*nB*4);
        cudaMemcpy(d_A, h_A, nA*dim*4, cudaMemcpyHostToDevice); cudaMemcpy(d_B, h_B, nB*dim*4, cudaMemcpyHostToDevice);
        
        std::cout << "Warming up..." << std::endl;
        launch_pearson_template(bs, d_A, d_B, d_C, nA, nB, dim); cudaDeviceSynchronize();
        
        std::cout << "Benchmarking..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;
        double total_time = 0.0;
        for(int i=0; i<10; i++) {
            cudaDeviceSynchronize(); auto start = std::chrono::high_resolution_clock::now();
            launch_pearson_template(bs, d_A, d_B, d_C, nA, nB, dim);
            cudaDeviceSynchronize(); 
            std::chrono::duration<double> diff = std::chrono::high_resolution_clock::now() - start;
            total_time += diff.count();
            double tput = ((double)nA * nB) / diff.count() / 1e6;
            std::cout << "Run " << std::setw(2) << i+1 << ": " << std::fixed << std::setprecision(6) << diff.count() << " s  (" << std::setprecision(2) << tput << " M ops/s)" << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_C, d_C, nA*nB*4, cudaMemcpyDeviceToHost);
        std::string output_filename = "gpu_results_pearson_opt_" + std::to_string(dim) + ".bin";
        std::ofstream f(output_filename, std::ios::binary); f.write((char*)h_C, nA*nB*4); f.close();
        std::cout << "Avg Throughput: " << std::fixed << std::setprecision(2) << ((double)nA*nB/(total_time/10.0)/1e6) << " M ops/s" << std::endl;
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); delete[] h_A; delete[] h_B; delete[] h_C;
    } catch (const std::exception& e) { std::cerr << e.what() << std::endl; return 1; }
    return 0;
}