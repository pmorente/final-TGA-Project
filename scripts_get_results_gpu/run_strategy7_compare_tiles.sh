#!/bin/bash
# ============================================================================
# GPU Strategy 7 Tile Size Comparison Script
# Tests different tile sizes (4, 8, 16, 30, 31, 32) and compares results with trust files
# Generates comparison table for all metrics and dimensions
# ============================================================================

set -e  # Exit on error

# Set numeric locale to C to ensure dot as decimal separator
export LC_NUMERIC=C

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
GPU_DIR="$(cd ../GPU/strategy_7 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$PROJECT_ROOT/inputs"
TRUST_DIR="$PROJECT_ROOT/trust_files"
DIMENSIONS=(384 768 1024)
METRICS=("cosine" "euclidean" "pearson")
TILE_SIZES=(4 8 16 30 31 32)
TILE_SIZE_NAMES=("4" "8" "16" "30" "31" "32")
RESULTS_FILE="$SCRIPT_DIR/strategy7_tilesize_comparison.md"
TOLERANCE="1e-7"

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

# Check if trust files exist
if [ ! -d "$TRUST_DIR" ]; then
    echo "Error: Trust files directory not found: $TRUST_DIR"
    exit 1
fi

echo "=== Tile Information ==="
echo "Strategy 7 uses 2D tiles for data reuse and shared memory optimization."
echo "Testing tile sizes: 4, 8, 16, 30, 31, 32"
echo "Note: Testing various tile sizes including powers of 2 and non-power-of-2 sizes (30, 31) which can be optimal"
echo ""

# Storage for results (using associative array)
# Key format: tile_size_metric_dimension
declare -A results_avg_time
declare -A results_avg_throughput
declare -A results_correctness

# Function to parse output and extract metrics
parse_output() {
    local output="$1"
    local avg_time=""
    local avg_throughput=""
    local runs=()
    local total_time=0.0
    local run_count=0
    
    # Parse individual runs and calculate average time
    while IFS= read -r line; do
        # Parse individual runs: "Run  1: 0.123456 s  (12.34 M ops/s)"
        if [[ $line =~ Run[[:space:]]+([0-9]+):[[:space:]]+([0-9.]+)[[:space:]]+s[[:space:]]+\(([0-9.]+)[[:space:]]+M[[:space:]]+ops/s\) ]]; then
            local run_time="${BASH_REMATCH[2]}"
            if [ -n "$run_time" ]; then
                total_time=$(LC_ALL=C awk "BEGIN {print ${total_time:-0} + ${run_time:-0}}")
            fi
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
    if [ -z "$avg_time" ] && [ "$run_count" -gt 0 ] && [ -n "$total_time" ]; then
        avg_time=$(LC_ALL=C awk "BEGIN {print ${total_time:-0} / ${run_count:-1}}")
    fi
    
    echo "$avg_time|$avg_throughput"
}

# Function to compare with trust file
compare_with_trust() {
    local gpu_file="$1"
    local cpu_file="$2"
    
    if [ ! -f "$gpu_file" ] || [ ! -f "$cpu_file" ]; then
        echo "ERROR"
        return
    fi
    
    # Use Python to compare (same logic as compare_with_trust.sh)
    python3 << EOF
import numpy as np
import sys

try:
    gpu_data = np.fromfile("$gpu_file", dtype=np.float32)
    cpu_data = np.fromfile("$cpu_file", dtype=np.float32)
    
    if gpu_data.shape != cpu_data.shape:
        print("ERROR")
        sys.exit(1)
    
    if np.allclose(gpu_data, cpu_data, atol=$TOLERANCE):
        max_diff = np.abs(gpu_data - cpu_data).max()
        print(f"PASS ({max_diff:.2e})")
    else:
        diff = np.abs(gpu_data - cpu_data)
        max_diff = diff.max()
        num_different = np.sum(diff > $TOLERANCE)
        print(f"FAIL ({max_diff:.2e}, {num_different} diff)")
    sys.exit(0)
except Exception as e:
    print("ERROR")
    sys.exit(1)
EOF
}

# Function to run a benchmark
run_benchmark() {
    local metric="$1"
    local dimension="$2"
    local tile_size="$3"
    
    local exe="$GPU_DIR/estrategia7_${metric}"
    local fileA="$INPUT_DIR/$dimension/file_a.bin"
    local fileB="$INPUT_DIR/$dimension/file_b.bin"
    
    if [ ! -f "$fileA" ] || [ ! -f "$fileB" ]; then
        echo "Error: Input files not found for dimension $dimension"
        return 1
    fi
    
    echo -n "Running $metric (dimension $dimension, tile_size $tile_size)... "
    
    # Change to GPU directory to run executable
    cd "$GPU_DIR"
    output=$(./estrategia7_${metric} "$fileA" "$fileB" "$tile_size" 2>&1)
    
    # Save GPU result file with dimension in filename
    local source_file="gpu_results_${metric}.bin"
    local dest_file="gpu_results_${metric}_${dimension}.bin"
    if [ -f "$source_file" ]; then
        cp "$source_file" "$dest_file"
    fi
    
    cd "$SCRIPT_DIR"
    
    # Parse output
    parsed=$(parse_output "$output")
    IFS='|' read -r avg_time avg_throughput <<< "$parsed"
    
    if [ -z "$avg_time" ] || [ -z "$avg_throughput" ]; then
        echo "FAILED - Could not parse output"
        echo "Output: $output"
        return 1
    fi
    
    # Compare with trust file
    local gpu_file="$GPU_DIR/gpu_results_${metric}_${dimension}.bin"
    local cpu_file="$TRUST_DIR/${dimension}/cpu_results_${metric}_${dimension}.bin"
    local correctness=$(compare_with_trust "$gpu_file" "$cpu_file")
    
    # Store results
    local key="${tile_size}_${metric}_${dimension}"
    results_avg_time["$key"]="$avg_time"
    results_avg_throughput["$key"]="$avg_throughput"
    results_correctness["$key"]="$correctness"
    
    printf "OK (Avg: %.6fs, %.2f M ops/s, %s)\n" "$avg_time" "$avg_throughput" "$correctness"
}

# Run all benchmarks
echo ""
echo "=== Running Benchmarks and Comparing with Trust Files ==="
set +e  # Temporarily disable exit on error to continue even if one benchmark fails
for dim in "${DIMENSIONS[@]}"; do
    echo ""
    echo "--- Dimension $dim ---"
    for metric in "${METRICS[@]}"; do
        for tile_size in "${TILE_SIZES[@]}"; do
            run_benchmark "$metric" "$dim" "$tile_size" || echo "Warning: Benchmark failed, continuing..."
        done
    done
done
set -e  # Re-enable exit on error

# Generate results file
echo ""
echo "=== Generating Comparison Tables ==="

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
content="# GPU Strategy 7 Tile Size Comparison - RTX 3050

Generated on: $timestamp

This document compares performance across different tile sizes (4, 8, 16, 30, 31, 32) for Strategy 7.
**All results are compared with CPU trust files for correctness verification.**

**Tile Information:** Strategy 7 uses 2D tiles for data reuse and shared memory optimization.
- **Tile Size 4:** 4×4 = 16 threads per block
- **Tile Size 8:** 8×8 = 64 threads per block
- **Tile Size 16:** 16×16 = 256 threads per block
- **Tile Size 30:** 30×30 = 900 threads per block (often optimal, not power of 2)
- **Tile Size 31:** 31×31 = 961 threads per block (near maximum, not power of 2)
- **Tile Size 32:** 32×32 = 1024 threads per block (maximum)

**Correctness:** Results are compared with CPU trust files using tolerance $TOLERANCE.
- **PASS:** Results match within tolerance
- **FAIL:** Results differ (shows max difference and number of differing values)
- **ERROR:** Comparison failed (file not found or format error)

"

# Build comparison tables for each dimension
for dim in "${DIMENSIONS[@]}"; do
    content+="## Dimension $dim\n\n"
    
    # Table for Average Time
    content+="### Average Computation Time (seconds)\n\n"
    content+="| Metric | 4 | 8 | 16 | 30 | 31 | 32 | Best |\n"
    content+="|--------|---|---|----|----|----|----|------|\n"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        best_time=""
        best_ts=""
        best_val=999999.0
        
        for i in "${!TILE_SIZES[@]}"; do
            tile_size="${TILE_SIZES[$i]}"
            key="${tile_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                row+=" $(printf "%8.6f" "$time_val") |"
                
                # Track best (lowest time)
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_ts="${TILE_SIZE_NAMES[$i]}"
                fi
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        
        if [ -n "$best_ts" ]; then
            row+=" $best_ts |"
        else
            row+=" N/A |"
        fi
        
        content+="$row\n"
    done
    
    content+="\n"
    
    # Table for Average Throughput
    content+="### Average Throughput (million comparisons/second)\n\n"
    content+="| Metric | 4 | 8 | 16 | 30 | 31 | 32 | Best |\n"
    content+="|--------|---|---|----|----|----|----|------|\n"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        best_throughput=""
        best_ts=""
        best_val=0.0
        
        for i in "${!TILE_SIZES[@]}"; do
            tile_size="${TILE_SIZES[$i]}"
            key="${tile_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_throughput[$key]}" ]; then
                throughput_val="${results_avg_throughput[$key]}"
                row+=" $(printf "%8.2f" "$throughput_val") |"
                
                # Track best (highest throughput)
                if LC_ALL=C awk "BEGIN {exit !(${throughput_val:-0} > ${best_val:-0})}"; then
                    best_val=$throughput_val
                    best_ts="${TILE_SIZE_NAMES[$i]}"
                fi
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        
        if [ -n "$best_ts" ]; then
            row+=" $best_ts |"
        else
            row+=" N/A |"
        fi
        
        content+="$row\n"
    done
    
    content+="\n"
    
    # Table for Correctness
    content+="### Correctness (Comparison with CPU Trust Files)\n\n"
    content+="| Metric | 4 | 8 | 16 | 30 | 31 | 32 |\n"
    content+="|--------|---|---|----|----|----|----|\n"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for tile_size in "${TILE_SIZES[@]}"; do
            key="${tile_size}_${metric}_${dim}"
            if [ -n "${results_correctness[$key]}" ]; then
                correctness="${results_correctness[$key]}"
                row+=" $(printf "%10s" "$correctness") |"
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        
        content+="$row\n"
    done
    
    content+="\n---\n\n"
done

# Summary table across all dimensions
content+="## Summary Across All Dimensions\n\n"
content+="### Best Tile Size by Metric and Dimension\n\n"
content+="| Metric | Dimension 384 | Dimension 768 | Dimension 1024 |\n"
content+="|--------|---------------|---------------|----------------|\n"

for metric_name in "Cosine" "Euclidean" "Pearson"; do
    metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    row=$(printf "| %-7s |" "$metric_name")
    
    for dim in "${DIMENSIONS[@]}"; do
        best_ts=""
        best_val=999999.0
        
        for i in "${!TILE_SIZES[@]}"; do
            tile_size="${TILE_SIZES[$i]}"
            key="${tile_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_ts="${TILE_SIZES[$i]}"
                fi
            fi
        done
        
        if [ -n "$best_ts" ]; then
            row+=" $(printf "%13s" "$best_ts") |"
        else
            row+=" $(printf "%13s" "N/A") |"
        fi
    done
    
    content+="$row\n"
done

# Write results file
echo -e "$content" > "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"

# Display summary tables
echo ""
echo "=== Tile Size Comparison Summary ==="
echo ""

for dim in "${DIMENSIONS[@]}"; do
    echo "--- Dimension $dim ---"
    echo ""
    echo "Average Time (seconds):"
    echo "| Metric | 4 | 8 | 16 | 30 | 31 | 32 |"
    echo "|--------|---|---|----|----|----|----|"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for tile_size in "${TILE_SIZES[@]}"; do
            key="${tile_size}_${metric}_${dim}"
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                row+=" $(printf "%8.6f" "$time_val") |"
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        echo "$row"
    done
    
    echo ""
    echo "Average Throughput (M ops/s):"
    echo "| Metric | 4 | 8 | 16 | 30 | 31 | 32 |"
    echo "|--------|---|---|----|----|----|----|"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for tile_size in "${TILE_SIZES[@]}"; do
            key="${tile_size}_${metric}_${dim}"
            if [ -n "${results_avg_throughput[$key]}" ]; then
                throughput_val="${results_avg_throughput[$key]}"
                row+=" $(printf "%8.2f" "$throughput_val") |"
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        echo "$row"
    done
    
    echo ""
    echo "Correctness (vs CPU Trust Files):"
    echo "| Metric | 4 | 8 | 16 | 30 | 31 | 32 |"
    echo "|--------|---|---|----|----|----|----|"
    
    for metric_name in "Cosine" "Euclidean" "Pearson"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for tile_size in "${TILE_SIZES[@]}"; do
            key="${tile_size}_${metric}_${dim}"
            if [ -n "${results_correctness[$key]}" ]; then
                correctness="${results_correctness[$key]}"
                row+=" $(printf "%10s" "$correctness") |"
            else
                row+=" $(printf "%8s" "ERROR") |"
            fi
        done
        echo "$row"
    done
    echo ""
done

# Display best tile size summary
echo ""
echo "=== BEST TILE SIZE SUMMARY ==="
echo ""
echo "Best tile size (lowest average time) for each metric and dimension:"
echo ""

for metric_name in "Cosine" "Euclidean" "Pearson"; do
    metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    echo "  $metric_name:"
    
    for dim in "${DIMENSIONS[@]}"; do
        best_ts=""
        best_val=999999.0
        best_time=""
        
        for i in "${!TILE_SIZES[@]}"; do
            tile_size="${TILE_SIZES[$i]}"
            key="${tile_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_ts="${TILE_SIZES[$i]}"
                    best_time="$time_val"
                fi
            fi
        done
        
        if [ -n "$best_ts" ]; then
            printf "    Dimension %d: Tile Size %s (Time: %.6fs)\n" "$dim" "$best_ts" "$best_time"
        else
            printf "    Dimension %d: N/A\n" "$dim"
        fi
    done
    echo ""
done

echo ""
echo "Done! Check $RESULTS_FILE for complete comparison results."

