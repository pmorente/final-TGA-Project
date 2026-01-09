// ============================================================================
// STRATEGY 3 — Grid over BIG group, loop SMALL group, sequential computation
// Algorithm: Euclidean Distance (Sequential per thread, no reduction)
// Features: Benchmark Mode (Warmup + 10 runs)
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

// ============================================================================
// KERNEL EUCLIDEAN
// ============================================================================
__global__ void euclidean_strategy3(
    const float* __restrict__ queries,
    const float* __restrict__ database,
    float* __restrict__ results,
    int num_queries,
    int num_database,
    int dim,
    int BLOCK_SIZE)
{
    // blockIdx.x = j → vector B[j] (database)
    int db_idx = blockIdx.x;
    if (db_idx >= num_database) return;
    
    const float* db_vec = database + db_idx * dim;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;
    
    for (int q_idx = tid; q_idx < num_queries; q_idx += num_threads) {
        const float* query = queries + q_idx * dim;
        
        // Sequential loop over dimensions (no reduction)
        float acc = 0.0f;
        for (int k = 0; k < dim; k++) {
            float d = query[k] - db_vec[k];
            acc += d * d;
        }
        
        results[q_idx * num_database + db_idx] = sqrtf(acc);
    }
}

void launch_euclidean_strategy3(
    const float* d_queries,
    const float* d_database,
    float* d_results,
    int num_queries,
    int num_database,
    int dim,
    int BLOCK_SIZE)
{
    dim3 grid(num_database);  // One block per database vector
    dim3 block(BLOCK_SIZE);   // BLOCK_SIZE threads per block
    
    euclidean_strategy3<<<grid, block>>>(
        d_queries, d_database, d_results,
        num_queries, num_database, dim, BLOCK_SIZE
    );
}

std::vector<EmbeddingRecord> read_embeddings_binary(const std::string& file_path) {
    std::vector<EmbeddingRecord> records;
    std::ifstream file(file_path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open file: " + file_path);
    
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4 || strncmp(magic, "EMBD", 4) != 0) throw std::runtime_error("Invalid file format");
    
    uint8_t version;
    file.read(reinterpret_cast<char*>(&version), 1);
    
    int32_t num_records, embedding_dim;
    file.read(reinterpret_cast<char*>(&num_records), sizeof(int32_t));
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int32_t));
    
    std::cout << "Reading file: " << file_path << std::endl;
    std::cout << "  Records: " << num_records << ", Dimension: " << embedding_dim << std::endl;
    
    for (int i = 0; i < num_records; ++i) {
        EmbeddingRecord record;
        int32_t id_length;
        file.read(reinterpret_cast<char*>(&id_length), sizeof(int32_t));
        std::vector<char> id_bytes(id_length);
        file.read(id_bytes.data(), id_length);
        record.id = std::string(id_bytes.data(), id_length);
        record.embedding.resize(embedding_dim);
        file.read(reinterpret_cast<char*>(record.embedding.data()), embedding_dim * sizeof(float));
        records.push_back(record);
    }
    file.close();
    return records;
}

int main(int argc, char* argv[]) {
    int block_size = 256;
    if (argc != 3) {
        if(argc == 4) block_size = std::stoi(argv[3]);
        else {
            std::cerr << "Usage: " << argv[0] << " <file_a> <file_b> [block_size]" << std::endl;
            return 1;
        }
    }
    
    std::string file_a = argv[1];
    std::string file_b = argv[2];
    
    try {
        std::cout << "\n=== Reading Embedding Files ===" << std::endl;
        std::vector<EmbeddingRecord> records_a = read_embeddings_binary(file_a);
        std::vector<EmbeddingRecord> records_b = read_embeddings_binary(file_b);
        
        int num_queries = records_a.size();
        int num_database = records_b.size();
        int dim = records_a.empty() ? 0 : records_a[0].embedding.size();
        
        std::cout << "\n=== Strategy 3: Grid over BIG group, Euclidean Benchmark ===" << std::endl;
        std::cout << "File A records: " << num_queries << std::endl;
        std::cout << "File B records: " << num_database << std::endl;
        std::cout << "Dimension: " << dim << std::endl;
        
        float *h_queries = new float[num_queries * dim];
        float *h_database = new float[num_database * dim];
        float *h_results = new float[num_queries * num_database];
        
        for (int i = 0; i < num_queries; i++) 
            std::memcpy(h_queries + i * dim, records_a[i].embedding.data(), dim * sizeof(float));
        for (int i = 0; i < num_database; i++) 
            std::memcpy(h_database + i * dim, records_b[i].embedding.data(), dim * sizeof(float));
        
        float *d_queries, *d_database, *d_results;
        cudaMalloc(&d_queries, num_queries * dim * sizeof(float));
        cudaMalloc(&d_database, num_database * dim * sizeof(float));
        cudaMalloc(&d_results, num_queries * num_database * sizeof(float));
        
        cudaMemcpy(d_queries, h_queries, num_queries * dim * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_database, h_database, num_database * dim * sizeof(float), cudaMemcpyHostToDevice);
        
        // =========================================================
        // BENCHMARKING (WARM-UP + LOOP)
        // =========================================================
        
        // 1. Warm-up
        std::cout << "\nWarming up GPU (1 run)..." << std::endl;
        launch_euclidean_strategy3(d_queries, d_database, d_results, num_queries, num_database, dim, block_size);
        cudaDeviceSynchronize();

        // 2. Ejecución Medida (10 iteraciones)
        int iterations = 10;
        std::cout << "Running benchmark (" << iterations << " iterations)..." << std::endl;
        std::cout << "------------------------------------------------" << std::endl;

        double total_compute_time = 0.0;
        
        for(int i = 0; i < iterations; i++) {
            cudaDeviceSynchronize(); 
            auto iter_start = std::chrono::high_resolution_clock::now();
            
            launch_euclidean_strategy3(d_queries, d_database, d_results, num_queries, num_database, dim, block_size);
            
            cudaDeviceSynchronize();
            auto iter_end = std::chrono::high_resolution_clock::now();
            
            std::chrono::duration<double> iter_duration = iter_end - iter_start;
            double iter_seconds = iter_duration.count();
            total_compute_time += iter_seconds;

            double throughput = ((double)num_queries * num_database) / iter_seconds / 1e6;
            std::cout << "Run " << std::setw(2) << i + 1 << ": " 
                      << std::fixed << std::setprecision(6) << iter_seconds << " s  "
                      << "(" << std::fixed << std::setprecision(2) << throughput << " M ops/s)" 
                      << std::endl;
        }
        std::cout << "------------------------------------------------" << std::endl;
        
        cudaMemcpy(h_results, d_results, num_queries * num_database * sizeof(float), cudaMemcpyDeviceToHost);
        
        double avg_compute_time = total_compute_time / iterations;
        double avg_throughput = ((double)num_queries * num_database) / avg_compute_time / 1e6;

        std::string output_filename = "gpu_results_euclidean_" + std::to_string(dim) + ".bin";
        std::ofstream fout(output_filename, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(h_results), num_queries * num_database * sizeof(float));
        fout.close();

        std::cout << "\n=== Average Performance Metrics ===" << std::endl;
        std::cout << "Avg Computation time: " << std::fixed << std::setprecision(6) << avg_compute_time << " seconds" << std::endl;
        std::cout << "Avg Throughput:       " << std::fixed << std::setprecision(2) << avg_throughput << " million comparisons/second" << std::endl;
        
        cudaFree(d_queries);
        cudaFree(d_database);
        cudaFree(d_results);
        delete[] h_queries;
        delete[] h_database;
        delete[] h_results;
        
        std::cout << "\nDone!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
