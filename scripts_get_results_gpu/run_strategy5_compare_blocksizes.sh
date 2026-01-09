#!/bin/bash
# ============================================================================
# GPU Strategy 5 Block Size Comparison Script
# Tests small (128), medium (256), and large (512) block sizes
# Generates comparison table for all metrics and dimensions
# ============================================================================

set -e  # Exit on error

# Set numeric locale to C to ensure dot as decimal separator
export LC_NUMERIC=C

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Configuration
GPU_DIR="$(cd ../GPU/strategy_5 && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
INPUT_DIR="$PROJECT_ROOT/inputs"
DIMENSIONS=(384 768 1024)
METRICS=("cosine" "euclidean" "pearson" "pearson_opt")
BLOCK_SIZES=(32 64 128 256 512 1024)
BLOCK_SIZE_NAMES=("32" "64" "128" "256" "512" "1024")
RESULTS_FILE="$SCRIPT_DIR/strategy5_blocksize_comparison.md"

# Check if GPU executables exist
echo ""
echo "=== Checking GPU Executables ==="
for metric in "${METRICS[@]}"; do
    exe="$GPU_DIR/estrategia5_${metric}"
    if [ ! -f "$exe" ]; then
        echo "Error: Missing executable: $exe"
        echo "Please compile first with:"
        echo "  cd $GPU_DIR && nvcc -O3 -arch=sm_89 estrategia5_${metric}.cu -o estrategia5_${metric}"
        exit 1
    fi
done
echo "All executables found."
echo ""
echo "=== Template Information ==="
echo "Strategy 5 uses C++ templates for compile-time optimization."
echo "Each block size gets a specialized kernel version for optimal performance."
echo "Testing block sizes: 32, 64, 128, 256, 512, 1024"
echo ""

# Storage for results (using associative array)
# Key format: block_size_metric_dimension
declare -A results_avg_time
declare -A results_avg_throughput

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

# Function to run a benchmark
run_benchmark() {
    local metric="$1"
    local dimension="$2"
    local block_size="$3"
    
    local exe="$GPU_DIR/estrategia5_${metric}"
    local fileA="$INPUT_DIR/$dimension/file_a.bin"
    local fileB="$INPUT_DIR/$dimension/file_b.bin"
    
    if [ ! -f "$fileA" ] || [ ! -f "$fileB" ]; then
        echo "Error: Input files not found for dimension $dimension"
        return 1
    fi
    
    echo -n "Running $metric (dimension $dimension, block_size $block_size)... "
    
    # Change to GPU directory to run executable
    cd "$GPU_DIR"
    output=$(./estrategia5_${metric} "$fileA" "$fileB" "$block_size" 2>&1)
    
    cd "$SCRIPT_DIR"
    
    # Parse output
    parsed=$(parse_output "$output")
    IFS='|' read -r avg_time avg_throughput <<< "$parsed"
    
    if [ -z "$avg_time" ] || [ -z "$avg_throughput" ]; then
        echo "FAILED - Could not parse output"
        echo "Output: $output"
        return 1
    fi
    
    # Store results
    local key="${block_size}_${metric}_${dimension}"
    results_avg_time["$key"]="$avg_time"
    results_avg_throughput["$key"]="$avg_throughput"
    
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
        for block_size in "${BLOCK_SIZES[@]}"; do
            run_benchmark "$metric" "$dim" "$block_size" || echo "Warning: Benchmark failed, continuing..."
        done
    done
done
set -e  # Re-enable exit on error

# Generate results file
echo ""
echo "=== Generating Comparison Tables ==="

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
content="# GPU Strategy 5 Block Size Comparison - RTX 3050

Generated on: $timestamp

This document compares performance across different block sizes (32, 64, 128, 256, 512, 1024) for Strategy 5.

**Template Option:** Strategy 5 uses C++ templates for compile-time optimization. Each block size gets a specialized kernel version at compile-time, providing:
- Loop unrolling optimizations
- Eliminated runtime checks (e.g., \`if (BLOCK_SIZE > 32)\`)
- Better register allocation
- Optimized memory access patterns

**Supported Template Specializations:** 32, 64, 128, 256, 512, 1024 (powers of 2 only)

"

# Build comparison tables for each dimension
for dim in "${DIMENSIONS[@]}"; do
    content+="## Dimension $dim\n\n"
    
    # Table for Average Time
    content+="### Average Computation Time (seconds)\n\n"
    content+="| Metric | 32 | 64 | 128 | 256 | 512 | 1024 | Best |\n"
    content+="|--------|----|----|-----|-----|-----|------|------|\n"
    
    for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        best_time=""
        best_bs=""
        best_val=999999.0
        
        for i in "${!BLOCK_SIZES[@]}"; do
            block_size="${BLOCK_SIZES[$i]}"
            key="${block_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                row+=" $(printf "%11.6f" "$time_val") |"
                
                # Track best (lowest time)
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_bs="${BLOCK_SIZE_NAMES[$i]}"
                fi
            else
                row+=" $(printf "%11s" "ERROR") |"
            fi
        done
        
        if [ -n "$best_bs" ]; then
            row+=" $best_bs |"
        else
            row+=" N/A |"
        fi
        
        content+="$row\n"
    done
    
    content+="\n"
    
    # Table for Average Throughput
    content+="### Average Throughput (million comparisons/second)\n\n"
    content+="| Metric | 32 | 64 | 128 | 256 | 512 | 1024 | Best |\n"
    content+="|--------|----|----|-----|-----|-----|------|------|\n"
    
    for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        best_throughput=""
        best_bs=""
        best_val=0.0
        
        for i in "${!BLOCK_SIZES[@]}"; do
            block_size="${BLOCK_SIZES[$i]}"
            key="${block_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_throughput[$key]}" ]; then
                throughput_val="${results_avg_throughput[$key]}"
                row+=" $(printf "%11.2f" "$throughput_val") |"
                
                # Track best (highest throughput)
                if LC_ALL=C awk "BEGIN {exit !(${throughput_val:-0} > ${best_val:-0})}"; then
                    best_val=$throughput_val
                    best_bs="${BLOCK_SIZE_NAMES[$i]}"
                fi
            else
                row+=" $(printf "%11s" "ERROR") |"
            fi
        done
        
        if [ -n "$best_bs" ]; then
            row+=" $best_bs |"
        else
            row+=" N/A |"
        fi
        
        content+="$row\n"
    done
    
    content+="\n---\n\n"
done

# Summary table across all dimensions
content+="## Summary Across All Dimensions\n\n"
content+="### Best Block Size by Metric and Dimension\n\n"
content+="| Metric | Dimension 384 | Dimension 768 | Dimension 1024 |\n"
content+="|--------|---------------|---------------|----------------|\n"

for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
    metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    row=$(printf "| %-7s |" "$metric_name")
    
    for dim in "${DIMENSIONS[@]}"; do
        best_bs=""
        best_val=999999.0
        
        for i in "${!BLOCK_SIZES[@]}"; do
            block_size="${BLOCK_SIZES[$i]}"
            key="${block_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_bs="${BLOCK_SIZES[$i]}"
                fi
            fi
        done
        
        if [ -n "$best_bs" ]; then
            row+=" $(printf "%13s" "$best_bs") |"
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
echo "=== Block Size Comparison Summary ==="
echo ""

for dim in "${DIMENSIONS[@]}"; do
    echo "--- Dimension $dim ---"
    echo ""
    echo "Average Time (seconds):"
    echo "| Metric | 32 | 64 | 128 | 256 | 512 | 1024 |"
    echo "|--------|----|----|-----|-----|-----|------|"
    
    for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for block_size in "${BLOCK_SIZES[@]}"; do
            key="${block_size}_${metric}_${dim}"
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                row+=" $(printf "%11.6f" "$time_val") |"
            else
                row+=" $(printf "%11s" "ERROR") |"
            fi
        done
        echo "$row"
    done
    
    echo ""
    echo "Average Throughput (M ops/s):"
    echo "| Metric | 32 | 64 | 128 | 256 | 512 | 1024 |"
    echo "|--------|----|----|-----|-----|-----|------|"
    
    for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
        metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
        row=$(printf "| %-7s |" "$metric_name")
        
        for block_size in "${BLOCK_SIZES[@]}"; do
            key="${block_size}_${metric}_${dim}"
            if [ -n "${results_avg_throughput[$key]}" ]; then
                throughput_val="${results_avg_throughput[$key]}"
                row+=" $(printf "%11.2f" "$throughput_val") |"
            else
                row+=" $(printf "%11s" "ERROR") |"
            fi
        done
        echo "$row"
    done
    echo ""
done

# Display best block size summary
echo ""
echo "=== BEST BLOCK SIZE SUMMARY ==="
echo ""
echo "Best block size (lowest average time) for each metric and dimension:"
echo ""

for metric_name in "Cosine" "Euclidean" "Pearson" "Pearson Opt"; do
    metric=$(echo "$metric_name" | tr '[:upper:]' '[:lower:]' | tr ' ' '_')
    echo "  $metric_name:"
    
    for dim in "${DIMENSIONS[@]}"; do
        best_bs=""
        best_val=999999.0
        best_time=""
        
        for i in "${!BLOCK_SIZES[@]}"; do
            block_size="${BLOCK_SIZES[$i]}"
            key="${block_size}_${metric}_${dim}"
            
            if [ -n "${results_avg_time[$key]}" ]; then
                time_val="${results_avg_time[$key]}"
                if LC_ALL=C awk "BEGIN {exit !(${time_val:-999999} < ${best_val:-999999})}"; then
                    best_val=$time_val
                    best_bs="${BLOCK_SIZES[$i]}"
                    best_time="$time_val"
                fi
            fi
        done
        
        if [ -n "$best_bs" ]; then
            printf "    Dimension %d: Block Size %s (Time: %.6fs)\n" "$dim" "$best_bs" "$best_time"
        else
            printf "    Dimension %d: N/A\n" "$dim"
        fi
    done
    echo ""
done

echo ""
echo "Done! Check $RESULTS_FILE for complete comparison results."

