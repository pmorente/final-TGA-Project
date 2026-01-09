#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include "cosine/cosine.hpp"
#include "pearson/pearson.hpp"
#include "euclidean/euclidean.hpp"

using namespace std;

enum MetricType { COSINE, EUCLIDEAN, PEARSON };

MetricType parse_metric(const string& metric_str) {
    if (metric_str == "cosine") return COSINE;
    if (metric_str == "euclidean") return EUCLIDEAN;
    if (metric_str == "pearson") return PEARSON;
    throw runtime_error("Invalid metric. Must be: cosine, euclidean, or pearson");
}

// Forward declaration
float compute_metric(MetricType metric, const vector<float>& A, const vector<float>& B);

// Structure to hold an embedding record
struct EmbeddingRecord {
    string id;
    vector<float> embedding;
};

// ------------------ Binary File Reading -------------------
vector<EmbeddingRecord> read_embeddings_binary(const string& file_path) {
    vector<EmbeddingRecord> records;
    
    ifstream file(file_path, ios::binary);
    if (!file.is_open()) {
        throw runtime_error("Cannot open file: " + file_path);
    }
    
    // Read header
    char magic[4];
    file.read(magic, 4);
    if (file.gcount() != 4 || strncmp(magic, "EMBD", 4) != 0) {
        throw runtime_error("Invalid file format: magic number mismatch or file too short");
    }
    
    uint8_t version;
    file.read(reinterpret_cast<char*>(&version), 1);
    if (file.gcount() != 1) {
        throw runtime_error("Failed to read version byte");
    }
    
    int32_t num_records, embedding_dim;
    file.read(reinterpret_cast<char*>(&num_records), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t)) {
        throw runtime_error("Failed to read num_records");
    }
    file.read(reinterpret_cast<char*>(&embedding_dim), sizeof(int32_t));
    if (file.gcount() != sizeof(int32_t)) {
        throw runtime_error("Failed to read embedding_dim");
    }
    
    cout << "Reading file: " << file_path << endl;
    cout << "  Records: " << num_records << ", Dimension: " << embedding_dim << endl;
    
    // Read each record
    for (int i = 0; i < num_records; ++i) {
        EmbeddingRecord record;
        
        // Read ID length
        int32_t id_length;
        file.read(reinterpret_cast<char*>(&id_length), sizeof(int32_t));
        if (file.gcount() != sizeof(int32_t)) {
            throw runtime_error("Failed to read id_length for record " + to_string(i));
        }
        if (id_length < 0 || id_length > 10000) {
            throw runtime_error("Invalid id_length: " + to_string(id_length));
        }
        
        // Read ID
        vector<char> id_bytes(id_length);
        file.read(id_bytes.data(), id_length);
        if (file.gcount() != id_length) {
            throw runtime_error("Failed to read ID for record " + to_string(i));
        }
        record.id = string(id_bytes.data(), id_length);
        
        // Read embedding
        record.embedding.resize(embedding_dim);
        size_t embedding_bytes = embedding_dim * sizeof(float);
        file.read(reinterpret_cast<char*>(record.embedding.data()), embedding_bytes);
        if (file.gcount() != static_cast<streamsize>(embedding_bytes)) {
            throw runtime_error("Failed to read embedding for record " + to_string(i));
        }
        
        records.push_back(record);
    }
    
    file.close();
    return records;
}

// ------------------ One-to-Many Comparison Function -------------------
void compare_one_to_many(const string& file_a, const string& file_b, MetricType metric, bool save_results = false) {
    
    auto start_time = chrono::high_resolution_clock::now();
    
    // Read both embedding files
    cout << "\n=== Reading Embedding Files ===" << endl;
    vector<EmbeddingRecord> records_a = read_embeddings_binary(file_a);
    vector<EmbeddingRecord> records_b = read_embeddings_binary(file_b);
    
    // Validate embedding dimensions match
    if (!records_a.empty() && !records_b.empty()) {
        if (records_a[0].embedding.size() != records_b[0].embedding.size()) {
            throw runtime_error("Embedding dimensions do not match between files");
        }
    }
    
    size_t total_comparisons = records_a.size() * records_b.size();
    
    string metric_name = (metric == COSINE) ? "cosine" : (metric == EUCLIDEAN) ? "euclidean" : "pearson";
    cout << "\n=== Computing One-to-Many Comparisons ===" << endl;
    cout << "Metric: " << metric_name << endl;
    cout << "File A records: " << records_a.size() << endl;
    cout << "File B records: " << records_b.size() << endl;
    cout << "Total comparisons: " << total_comparisons << endl;
    
    std::vector<float> results(records_a.size() * records_b.size());
    auto compute_start = chrono::high_resolution_clock::now();
    
    int count = 0;
    for (size_t i = 0; i < records_a.size(); ++i) {
        for (size_t j = 0; j < records_b.size(); ++j) {
            float result = compute_metric(metric, records_a[i].embedding, records_b[j].embedding);
            results[i * records_b.size() + j] = result;
            
            count++;
            if (count % 1000 == 0) {
                cout << "  Progress: " << count << "/" << total_comparisons 
                     << " (" << fixed << setprecision(1) << (100.0 * count / total_comparisons) << "%)" << endl;
            }
        }
    }
    
    auto compute_end = chrono::high_resolution_clock::now();
    auto total_end = chrono::high_resolution_clock::now();
    
    // Save results to binary file if requested
    if (save_results) {
        int dimension = records_a.empty() ? 0 : static_cast<int>(records_a[0].embedding.size());
        string filename = "cpu_results_" + metric_name + "_" + to_string(dimension) + ".bin";
        std::ofstream fout(filename, std::ios::binary);
        fout.write(reinterpret_cast<const char*>(results.data()), results.size() * sizeof(float));
        fout.close();
        cout << "Results saved to: " << filename << endl;
    }
    
    chrono::duration<double> compute_time = compute_end - compute_start;
    chrono::duration<double> total_time = total_end - start_time;
    
    cout << "\n=== Execution Metrics ===" << endl;
    cout << "Metric: " << metric_name << endl;
    cout << "Total comparisons: " << total_comparisons << endl;
    cout << "Computation time: " << fixed << setprecision(6) 
        << compute_time.count() << " seconds" << endl;
    cout << "Total time (including I/O): " << fixed << setprecision(6) 
        << total_time.count() << " seconds" << endl;
    cout << "Comparisons per second: " << fixed << setprecision(2) 
        << (total_comparisons / compute_time.count()) << endl;
}

// ------------------ Main Function -------------------
int main(int argc, char* argv[]) {
    bool save_results = false;
    string file_a, file_b, metric_str;
    
    // Parse arguments
    if (argc == 4) {
        file_a = argv[1];
        file_b = argv[2];
        metric_str = argv[3];
    } else if (argc == 5 && string(argv[4]) == "--save-results") {
        file_a = argv[1];
        file_b = argv[2];
        metric_str = argv[3];
        save_results = true;
    } else {
        cerr << "Usage: " << argv[0] 
            << " <file_a> <file_b> <metric> [--save-results]" << endl;
        cerr << "  file_a: Path to first embedding file" << endl;
        cerr << "  file_b: Path to second embedding file" << endl;
        cerr << "  metric: One of: cosine, euclidean, pearson" << endl;
        cerr << "  --save-results: (optional) Save comparison results to binary file" << endl;
        cerr << "\nExample:" << endl;
        cerr << "  " << argv[0] 
            << " input_data/A/embeddings.bin input_data/B/embeddings.bin cosine" 
            << endl;
        cerr << "  " << argv[0] 
            << " input_data/A/embeddings.bin input_data/B/embeddings.bin cosine --save-results" 
            << endl;
        return 1;
    }
    
    try {
        MetricType metric = parse_metric(metric_str);
        compare_one_to_many(file_a, file_b, metric, save_results);
        cout << "\nDone!" << endl;
        return 0;
    } catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }
}

float compute_metric(MetricType metric, const vector<float>& A, const vector<float>& B) {
    if (A.size() != B.size()) {
        throw runtime_error("Vector sizes differ");
    }
    
    switch (metric) {
        case COSINE:
            return cosine_cpu(A, B);
        case EUCLIDEAN:
            return euclidean_distance_cpu(A.data(), B.data(), static_cast<int>(A.size()));
        case PEARSON:
            return pearson_corr_cpu(A.data(), B.data(), static_cast<int>(A.size()));
        default:
            throw runtime_error("Unknown metric type");
    }
}
