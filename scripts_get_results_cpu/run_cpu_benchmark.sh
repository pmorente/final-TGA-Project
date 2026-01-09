#!/bin/bash
# ============================================================================
# CPU Benchmark Script
# Runs all 3 CPU strategies for dimensions 384, 768, and 1024
# Generates performance results
# ============================================================================
#
# Usage: ./run_cpu_benchmark.sh [--save-bin] [--bin-dir DIR]
#   --save-bin    Save binary result files (default: false)
#   --bin-dir DIR Directory to save binary files (default: ../CPU)
#

set -e  # Exit on error

# Parse command line arguments
SAVE_BIN=false
BIN_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --save-bin)
            SAVE_BIN=true
            shift
            ;;
        --bin-dir)
            BIN_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--save-bin] [--bin-dir DIR]"
            exit 1
            ;;
    esac
done

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
CPU_DIR="$(cd ../CPU && pwd)"
INPUT_DIR="$PROJECT_ROOT/inputs"
DIMENSIONS=(384 768 1024)
METRICS=("cosine" "euclidean" "pearson")
STRATEGIES=("compare_one_to_many" "parallel_compare_one_to_many" "parallel_metric_compare_one_to_many")
STRATEGY_NAMES=("Strategy 1 (Sequential)" "Strategy 2 (Vector-level)" "Strategy 3 (Metric-level)")
RESULTS_FILE="$SCRIPT_DIR/cpu_results.md"

# Set default binary directory if not specified
if [ -z "$BIN_DIR" ]; then
    BIN_DIR="$CPU_DIR"
fi

# Create binary directory if it doesn't exist and saving is enabled
if [ "$SAVE_BIN" = true ]; then
    mkdir -p "$BIN_DIR"
    echo "Binary results will be saved to: $BIN_DIR"
fi

# Check if CPU executables exist
echo ""
echo "=== Checking CPU Executables ==="
for strategy in "${STRATEGIES[@]}"; do
    exe="$CPU_DIR/$strategy"
    if [ ! -f "$exe" ]; then
        echo "Error: Missing executable: $exe"
        echo "Please compile first. See CPU/README.md for compilation instructions."
        exit 1
    fi
done
echo "All executables found."
echo ""

# Storage for results
declare -A results_time
declare -A results_throughput

# Function to parse output and extract metrics
parse_output() {
    local output="$1"
    local compute_time=""
    local throughput=""
    
    # Parse computation time: "Computation time: 0.123456 seconds"
    while IFS= read -r line; do
        if [[ $line =~ Computation[[:space:]]+time:[[:space:]]+([0-9.]+)[[:space:]]+seconds ]]; then
            compute_time="${BASH_REMATCH[1]}"
        # Parse throughput: "Comparisons per second: 1234.56"
        elif [[ $line =~ Comparisons[[:space:]]+per[[:space:]]+second:[[:space:]]+([0-9.]+) ]]; then
            throughput="${BASH_REMATCH[1]}"
        fi
    done <<< "$output"
    
    echo "$compute_time|$throughput"
}

# Function to run a benchmark
run_benchmark() {
    local strategy_exe="$1"
    local strategy_name="$2"
    local metric="$3"
    local dimension="$4"
    
    local fileA="$INPUT_DIR/$dimension/file_a.bin"
    local fileB="$INPUT_DIR/$dimension/file_b.bin"
    
    if [ ! -f "$fileA" ] || [ ! -f "$fileB" ]; then
        echo "Error: Input files not found for dimension $dimension"
        return 1
    fi
    
    echo -n "Running $strategy_name - $metric (dimension $dimension)... "
    
    # Change to CPU directory to run executable
    cd "$CPU_DIR"
    if [ "$SAVE_BIN" = true ]; then
        output=$(./"$strategy_exe" "$fileA" "$fileB" "$metric" --save-results 2>&1)
    else
        output=$(./"$strategy_exe" "$fileA" "$fileB" "$metric" 2>&1)
    fi
    cd "$SCRIPT_DIR"
    
    # Parse output
    parsed=$(parse_output "$output")
    IFS='|' read -r compute_time throughput <<< "$parsed"
    
    if [ -z "$compute_time" ] || [ -z "$throughput" ]; then
        echo "FAILED - Could not parse output"
        echo "Output: $output"
        return 1
    fi
    
    # Handle binary file saving if enabled
    if [ "$SAVE_BIN" = true ]; then
        local source_bin="$CPU_DIR/cpu_results_${metric}_${dimension}.bin"
        if [ -f "$source_bin" ]; then
            # Copy to organized location with strategy name
            local dest_bin="$BIN_DIR/cpu_results_${strategy_exe}_${metric}_${dimension}.bin"
            cp "$source_bin" "$dest_bin"
        fi
    fi
    
    # Store results
    local key="${strategy_exe}_${metric}_${dimension}"
    results_time["$key"]="$compute_time"
    results_throughput["$key"]="$throughput"
    
    printf "OK (Time: %.6fs, Throughput: %.2f ops/s)\n" "$compute_time" "$throughput"
}

# Run all benchmarks
echo ""
echo "=== Running CPU Benchmarks ==="
for dim in "${DIMENSIONS[@]}"; do
    echo ""
    echo "--- Dimension $dim ---"
    for i in "${!STRATEGIES[@]}"; do
        strategy="${STRATEGIES[$i]}"
        strategy_name="${STRATEGY_NAMES[$i]}"
        for metric in "${METRICS[@]}"; do
            run_benchmark "$strategy" "$strategy_name" "$metric" "$dim"
        done
    done
done

# Generate results file
echo ""
echo "=== Generating Results File ==="

timestamp=$(date "+%Y-%m-%d %H:%M:%S")
content="# CPU Benchmark Results

Generated on: $timestamp

## Summary by Strategy and Dimension

"

# Build summary tables for each strategy
for i in "${!STRATEGIES[@]}"; do
    strategy="${STRATEGIES[$i]}"
    strategy_name="${STRATEGY_NAMES[$i]}"
    
    content+="### $strategy_name\n\n"
    content+="| Metric | 384 dimensions |            | 768 dimensions |            | 1024 dimensions |            |\n"
    content+="|--------|----------------|------------|----------------|------------|------------------|------------|\n"
    content+="|        | Time (s)        | Throughput | Time (s)       | Throughput | Time (s)         | Throughput |\n"
    
    for metric in "${METRICS[@]}"; do
        row=$(printf "| %-7s |" "$metric")
        
        for dim in "${DIMENSIONS[@]}"; do
            key="${strategy}_${metric}_${dim}"
            if [ -n "${results_time[$key]}" ]; then
                time_val="${results_time[$key]}"
                throughput_val="${results_throughput[$key]}"
                row+=" $(printf "%14.6f" "$time_val") | $(printf "%11.2f" "$throughput_val") |"
            else
                row+=" $(printf "%14s" "ERROR") | $(printf "%11s" "ERROR") |"
            fi
        done
        content+="$row\n"
    done
    content+="\n"
done

# Add detailed results section
content+="---\n\n## Detailed Results\n\n"

for dim in "${DIMENSIONS[@]}"; do
    content+="### Dimension $dim\n\n"
    for i in "${!STRATEGIES[@]}"; do
        strategy="${STRATEGIES[$i]}"
        strategy_name="${STRATEGY_NAMES[$i]}"
        content+="#### $strategy_name\n\n"
        
        for metric in "${METRICS[@]}"; do
            key="${strategy}_${metric}_${dim}"
            if [ -n "${results_time[$key]}" ]; then
                time_val="${results_time[$key]}"
                throughput_val="${results_throughput[$key]}"
                content+="**$metric**:\n"
                content+="- Computation Time: $(printf "%.6f" "$time_val") seconds\n"
                content+="- Throughput: $(printf "%.2f" "$throughput_val") comparisons/second\n"
                content+="\n"
            fi
        done
    done
done

# Write results file
echo -e "$content" > "$RESULTS_FILE"
echo "Results saved to: $RESULTS_FILE"

# Display summary
echo ""
echo "=== Summary ==="
for i in "${!STRATEGIES[@]}"; do
    strategy="${STRATEGIES[$i]}"
    strategy_name="${STRATEGY_NAMES[$i]}"
    echo ""
    echo "$strategy_name:"
    echo "| Metric | 384 dimensions |            | 768 dimensions |            | 1024 dimensions |            |"
    echo "|--------|----------------|------------|----------------|------------|------------------|------------|"
    echo "|        | Time (s)        | Throughput | Time (s)       | Throughput | Time (s)         | Throughput |"
    
    for metric in "${METRICS[@]}"; do
        row=$(printf "| %-7s |" "$metric")
        
        for dim in "${DIMENSIONS[@]}"; do
            key="${strategy}_${metric}_${dim}"
            if [ -n "${results_time[$key]}" ]; then
                time_val="${results_time[$key]}"
                throughput_val="${results_throughput[$key]}"
                row+=" $(printf "%14.6f" "$time_val") | $(printf "%11.2f" "$throughput_val") |"
            else
                row+=" $(printf "%14s" "ERROR") | $(printf "%11s" "ERROR") |"
            fi
        done
        echo "$row"
    done
done

echo ""
echo "Done! Check $RESULTS_FILE for complete results."

if [ "$SAVE_BIN" = true ]; then
    echo ""
    echo "Binary result files saved to: $BIN_DIR"
    echo "Files saved:"
    for strategy in "${STRATEGIES[@]}"; do
        for metric in "${METRICS[@]}"; do
            for dim in "${DIMENSIONS[@]}"; do
                bin_file="$BIN_DIR/cpu_results_${strategy}_${metric}_${dim}.bin"
                if [ -f "$bin_file" ]; then
                    file_size=$(stat -f%z "$bin_file" 2>/dev/null || stat -c%s "$bin_file" 2>/dev/null || echo "unknown")
                    echo "  - cpu_results_${strategy}_${metric}_${dim}.bin ($file_size bytes)"
                fi
            done
        done
    done
    echo ""
    echo "Original files (with dimension in name) are in: $CPU_DIR"
    echo "  Format: cpu_results_<metric>_<dimension>.bin"
fi

