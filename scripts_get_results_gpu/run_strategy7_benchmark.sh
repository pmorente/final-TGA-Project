#!/bin/bash
# ============================================================================
# GPU Strategy 7 Benchmark Script (Bash version for WSL)
# Runs all metrics for dimensions 384, 768, and 1024 with specified tile size
# Generates performance tables for RTX 3050
# ============================================================================

set -e  # Exit on error

# Set numeric locale to C to ensure dot as decimal separator
export LC_NUMERIC=C

# Check if tile size is provided
if [ $# -lt 1 ]; then
    echo "Usage: $0 <tile_size>"
    echo "Example: $0 16"
    echo ""
    echo "Tile size must be between 4 and 32 (e.g., 4, 8, 16, 30, 31, 32)"
    echo ""
    echo "Note: Strategy 7 uses 2D tiles. Recommended sizes: 16, 30, 32"
    echo "      Tile size 30 is often optimal (not a power of 2, but works well)"
    exit 1
fi

TILE_SIZE=$1

# Validate tile size
if ! [[ "$TILE_SIZE" =~ ^[0-9]+$ ]] || [ "$TILE_SIZE" -lt 4 ] || [ "$TILE_SIZE" -gt 32 ]; then
    echo "Error: Tile size must be between 4 and 32"
    exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
GPU_DIR="$(cd ../GPU/strategy_7 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$PROJECT_ROOT/inputs"
DIMENSIONS=(384 768 1024)
METRICS=("cosine" "euclidean" "pearson")
RESULTS_FILE="$SCRIPT_DIR/strategy7_results_tile${TILE_SIZE}.md"

# Check if GPU executables exist
echo ""
echo "=== Checking GPU Executables ==="
for metric in "${METRICS[@]}"; do
    exe="$GPU_DIR/estrategia7_${metric}"
    if [ ! -f "$exe" ]; then
        echo "Error: Missing executable: $exe"
        echo "Please compile first with:"
        echo "  cd $GPU_DIR && nvcc -O3 -arch=sm_89 estrategia7_${metric}.cu -o estrategia7_${metric}"
        exit 1
    fi
done
echo "All executables found."
echo ""
echo "Using tile size: $TILE_SIZE"
echo ""
echo "=== Tile Information ==="
echo "Strategy 7 uses 2D tiles for data reuse and shared memory optimization."
echo "Tile configuration: ${TILE_SIZE}×${TILE_SIZE} = $((TILE_SIZE * TILE_SIZE)) threads per block"
echo "Shared memory: $((2 * TILE_SIZE * TILE_SIZE * 4)) bytes per block"
echo ""

# Storage for results (using associative array)
declare -A results_avg_time
declare -A results_avg_throughput
declare -A results_runs

# Function to parse output and extract metrics
parse_output() {
    local output="$1"
    local runs=()
    local avg_time=""
    local avg_throughput=""
    local total_time=0.0
    local run_count=0
    
    # Parse individual runs: "Run  1: 0.123456 s  (12.34 M ops/s)"
    while IFS= read -r line; do
        if [[ $line =~ Run[[:space:]]+([0-9]+):[[:space:]]+([0-9.]+)[[:space:]]+s[[:space:]]+\(([0-9.]+)[[:space:]]+M[[:space:]]+ops/s\) ]]; then
            runs+=("${BASH_REMATCH[1]}:${BASH_REMATCH[2]}:${BASH_REMATCH[3]}")
            local run_time="${BASH_REMATCH[2]}"
            total_time=$(awk "BEGIN {print $total_time + $run_time}")
            run_count=$((run_count + 1))
        # Parse average time: "Avg Computation time: 0.123456 seconds"
        elif [[ $line =~ Avg[[:space:]]+Computation[[:space:]]+time:[[:space:]]+([0-9.]+)[[:space:]]+seconds ]]; then
            avg_time="${BASH_REMATCH[1]}"
        # Parse average throughput: "Avg Throughput: 12.34 million comparisons/second"
        elif [[ $line =~ Avg[[:space:]]+Throughput:[[:space:]]+([0-9.]+)[[:space:]]+million[[:space:]]+comparisons/second ]]; then
            avg_throughput="${BASH_REMATCH[1]}"
        # Parse alternative throughput format: "Avg Throughput: 12.34 M ops/s"
        elif [[ $line =~ Avg[[:space:]]+Throughput:[[:space:]]+([0-9.]+)[[:space:]]+M[[:space:]]+ops/s ]]; then
            avg_throughput="${BASH_REMATCH[1]}"
        fi
    done <<< "$output"
    
    # If avg_time not found but we have runs, calculate it
    if [ -z "$avg_time" ] && [ "$run_count" -gt 0 ]; then
        avg_time=$(awk "BEGIN {print $total_time / $run_count}")
    fi
    
    # Return results (we'll use global arrays to store)
    echo "$avg_time|$avg_throughput|${runs[*]}"
}

# Function to run a benchmark
run_benchmark() {
    local metric="$1"
    local dimension="$2"
    
    local exe="$GPU_DIR/estrategia7_${metric}"
    local fileA="$INPUT_DIR/$dimension/file_a.bin"
    local fileB="$INPUT_DIR/$dimension/file_b.bin"
    
    if [ ! -f "$fileA" ] || [ ! -f "$fileB" ]; then
        echo "Error: Input files not found for dimension $dimension"
        return 1
    fi
    
    echo -n "Running $metric (dimension $dimension, tile_size $TILE_SIZE)... "
    
    # Change to GPU directory to run executable
    cd "$GPU_DIR"
    output=$(./estrategia7_${metric} "$fileA" "$fileB" "$TILE_SIZE" 2>&1)
    
    # Save GPU result file with dimension in filename to avoid overwrites
    local source_file="gpu_results_${metric}.bin"
    local dest_file="gpu_results_${metric}_${dimension}.bin"
    if [ -f "$source_file" ]; then
        cp "$source_file" "$dest_file"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Parse output
    parsed=$(parse_output "$output")
    IFS='|' read -r avg_time avg_throughput runs_str <<< "$parsed"
    
    if [ -z "$avg_time" ] || [ -z "$avg_throughput" ]; then
        echo "FAILED - Could not parse output"
        echo "Output: $output"
        return 1
    fi
    
    # Store results
    local key="${metric}_${dimension}"
    results_avg_time["$key"]="$avg_time"
    results_avg_throughput["$key"]="$avg_throughput"
    results_runs["$key"]="$runs_str"
    
    printf "OK (Avg: %.6fs, %.2f M ops/s)\n" "$avg_time" "$avg_throughput"
}

# Run all benchmarks
echo ""
echo "=== Running Benchmarks ==="
set +e  # Temporarily disable exit on error to continue even if one benchmark fails
for dim in "${DIMENSIONS[@]}"; do
    echo ""
    echo "--- Dimension $dim ---"
    for metric in "${METRICS[@]}"; do
        run_benchmark "$metric" "$dim" || echo "Warning: Benchmark failed, continuing..."
    done
done
set -e  # Re-enable exit on error

# Generate results file
echo ""
echo "=== Generating Results File ==="

# Build summary table
summary_content="| RTX 3050 | 384 dimensions |            | 768 dimensions |            | 1024 dimensions |            |\n"
summary_content+="|------------------|----------------|------------|----------------|------------|------------------|------------|\n"
summary_content+="|                  | Avg Time       | Avg Throughput | Avg Time       | Avg Throughput | Avg Time         | Avg Throughput |\n"

summary_rows=("Cosine" "Euclidean" "Pearson")
for metric_name in "${summary_rows[@]}"; do
    metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    row=$(printf "| %-18s |" "$metric_name")
    
    for dim in "${DIMENSIONS[@]}"; do
        key="${metric}_${dim}"
        if [ -n "${results_avg_time[$key]}" ]; then
            avg_time="${results_avg_time[$key]}"
            avg_throughput="${results_avg_throughput[$key]}"
            row+=" $(printf "%14.6f" "$avg_time") | $(printf "%17.2f" "$avg_throughput") |"
        else
            row+=" $(printf "%14s" "ERROR") | $(printf "%17s" "ERROR") |"
        fi
    done
    summary_content+="$row\n"
done

# Function to build runs table for a specific dimension
build_runs_table() {
    local dimension="$1"
    local table_content="| RTX 3050 | run 1       | run 2       | run 3       | run 4       | run 5       | run 6       | run 7       | run 8       | run 9       | run 10      |\n"
    table_content+="|------------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|-------------|\n"

    for metric_name in "${summary_rows[@]}"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-18s |" "$metric_name")
        
        key="${metric}_${dimension}"
        if [ -n "${results_runs[$key]}" ]; then
            runs_str="${results_runs[$key]}"
            # Parse runs string (format: "1:time:throughput 2:time:throughput ...")
            declare -a runs_array
            IFS=' ' read -ra runs_array <<< "$runs_str"
            
            # Extract times for runs 1-10
            for i in {1..10}; do
                found=0
                for run_entry in "${runs_array[@]}"; do
                    IFS=':' read -r run_num time_val throughput_val <<< "$run_entry"
                    if [ "$run_num" -eq "$i" ]; then
                        row+=" $(printf "%11.6f" "$time_val") |"
                        found=1
                        break
                    fi
                done
                if [ $found -eq 0 ]; then
                    row+=" $(printf "%11s" "N/A") |"
                fi
            done
        else
            row+=" $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") | $(printf "%11s" "ERROR") |"
        fi
        table_content+="$row\n"
    done
    
    echo -e "$table_content"
}

# Build runs tables for all dimensions
runs_content_384=$(build_runs_table 384)
runs_content_768=$(build_runs_table 768)
runs_content_1024=$(build_runs_table 1024)

# Build complete content
timestamp=$(date "+%Y-%m-%d %H:%M:%S")
content="# GPU Strategy 7 Benchmark Results - RTX 3050

Generated on: $timestamp

**Tile Size:** $TILE_SIZE (${TILE_SIZE}×${TILE_SIZE} = $((TILE_SIZE * TILE_SIZE)) threads per block)

**Tile Information:** Strategy 7 uses 2D tiles for data reuse and shared memory optimization.
- **Threads per block:** $((TILE_SIZE * TILE_SIZE))
- **Shared memory:** $((2 * TILE_SIZE * TILE_SIZE * 4)) bytes per block
- **Grid structure:** 2D grid over result matrix (queries × database)

## RTX 3050 — Summary by Dimensions

$summary_content

## RTX 3050 — Runs

### Dimension 384

$runs_content_384

### Dimension 768

$runs_content_768

### Dimension 1024

$runs_content_1024

---

## Detailed Results

"

# Add detailed results section
for dim in "${DIMENSIONS[@]}"; do
    content+="### Dimension $dim\n\n"
    for metric in "${METRICS[@]}"; do
        key="${metric}_${dim}"
        if [ -n "${results_avg_time[$key]}" ]; then
            avg_time="${results_avg_time[$key]}"
            avg_throughput="${results_avg_throughput[$key]}"
            content+="**$metric**:\n"
            content+="- Average Time: $(printf "%.6f" "$avg_time") seconds\n"
            content+="- Average Throughput: $(printf "%.2f" "$avg_throughput") million comparisons/second\n"
            content+="- Individual Runs:\n"
            
            runs_str="${results_runs[$key]}"
            declare -a runs_array
            IFS=' ' read -ra runs_array <<< "$runs_str"
            for run_entry in "${runs_array[@]}"; do
                IFS=':' read -r run_num time_val throughput_val <<< "$run_entry"
                content+="  - Run $run_num: $(printf "%.6f" "$time_val")s ($(printf "%.2f" "$throughput_val") M ops/s)\n"
            done
            content+="\n"
        fi
    done
done

# Write results file
echo -e "$content" > "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"

# Display summary
echo ""
echo "=== Summary ==="
echo -e "$summary_content"
echo ""
echo "=== Runs (Dimension 384) ==="
echo -e "$runs_content_384"
echo ""
echo "=== Runs (Dimension 768) ==="
echo -e "$runs_content_768"
echo ""
echo "=== Runs (Dimension 1024) ==="
echo -e "$runs_content_1024"

echo ""
echo "Done! Check $RESULTS_FILE for complete results."

