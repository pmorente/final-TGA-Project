#!/bin/bash
# ============================================================================
# Compare GPU Results with CPU Trust Files
# Compares GPU binary results with CPU trust files and verifies format consistency
# ============================================================================

set -e

# Set numeric locale to C to ensure dot as decimal separator
export LC_NUMERIC=C

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$SCRIPT_DIR"

# Configuration
TRUST_DIR="$PROJECT_ROOT/trust_files"
GPU_STRATEGY_DIR=""
DIMENSIONS=(384 768 1024)
METRICS=("cosine" "euclidean" "pearson")

# Parse arguments
if [ $# -lt 1 ]; then
    echo "Usage: $0 <gpu_strategy_dir> [--tolerance TOL]"
    echo "  gpu_strategy_dir: Path to GPU strategy directory (e.g., ../GPU/strategy_1)"
    echo "  --tolerance TOL:  Tolerance for comparison (default: 1e-7)"
    exit 1
fi

GPU_STRATEGY_DIR="$1"
TOLERANCE="1e-7"

if [ "$2" = "--tolerance" ] && [ -n "$3" ]; then
    TOLERANCE="$3"
fi

# Resolve absolute paths
GPU_STRATEGY_DIR="$(cd "$GPU_STRATEGY_DIR" && pwd)"
# TRUST_DIR is already set to project root/trust_files
if [ ! -d "$TRUST_DIR" ]; then
    echo "Error: Trust files directory not found: $TRUST_DIR"
    exit 1
fi

# Check if GPU strategy directory exists
if [ ! -d "$GPU_STRATEGY_DIR" ]; then
    echo "Error: GPU strategy directory not found: $GPU_STRATEGY_DIR"
    exit 1
fi

# Check if trust files directory exists
if [ ! -d "$TRUST_DIR" ]; then
    echo "Error: Trust files directory not found: $TRUST_DIR"
    exit 1
fi

echo "=== Comparing GPU Results with CPU Trust Files ==="
echo "GPU Strategy Directory: $GPU_STRATEGY_DIR"
echo "Trust Files Directory: $TRUST_DIR"
echo "Tolerance: $TOLERANCE"
echo ""

# Function to check file format
check_format() {
    local file="$1"
    local label="$2"
    
    if [ ! -f "$file" ]; then
        echo "  ERROR: File not found: $file"
        return 1
    fi
    
    # Check file size (must be multiple of 4 bytes for float32)
    local size=$(stat -f%z "$file" 2>/dev/null || stat -c%s "$file" 2>/dev/null)
    if [ $((size % 4)) -ne 0 ]; then
        echo "  ERROR: $label file size ($size bytes) is not a multiple of 4 (float32)"
        return 1
    fi
    
    local num_floats=$((size / 4))
    echo "  $label: $num_floats float32 values ($size bytes)"
    return 0
}

# Function to compare files
compare_files() {
    local gpu_file="$1"
    local cpu_file="$2"
    local metric="$3"
    local dimension="$4"
    
    echo ""
    echo "--- Comparing $metric (dimension $dimension) ---"
    
    # Check if files exist
    if [ ! -f "$gpu_file" ]; then
        echo "  ERROR: GPU file not found: $gpu_file"
        return 1
    fi
    
    if [ ! -f "$cpu_file" ]; then
        echo "  ERROR: CPU trust file not found: $cpu_file"
        return 1
    fi
    
    # Check format consistency
    echo "  Checking file formats..."
    if ! check_format "$gpu_file" "GPU"; then
        echo "  Format check failed for GPU file"
        return 1
    fi
    if ! check_format "$cpu_file" "CPU"; then
        echo "  Format check failed for CPU file"
        return 1
    fi
    
    # Verify both use float32 (4 bytes per value)
    echo "  ✓ Both files use float32 binary format (4 bytes per value)"
    
    # Get file sizes
    local gpu_size=$(stat -f%z "$gpu_file" 2>/dev/null || stat -c%s "$gpu_file" 2>/dev/null)
    local cpu_size=$(stat -f%z "$cpu_file" 2>/dev/null || stat -c%s "$cpu_file" 2>/dev/null)
    
    if [ "$gpu_size" -ne "$cpu_size" ]; then
        echo "  ERROR: File sizes differ! GPU: $gpu_size bytes, CPU: $cpu_size bytes"
        echo "  This indicates different number of comparisons or format mismatch"
        return 1
    fi
    
    # Use Python to compare (more reliable than bash)
    python3 << EOF
import numpy as np
import sys

try:
    gpu_data = np.fromfile("$gpu_file", dtype=np.float32)
    cpu_data = np.fromfile("$cpu_file", dtype=np.float32)
    
    if gpu_data.shape != cpu_data.shape:
        print(f"  ERROR: Array shapes differ! GPU: {gpu_data.shape}, CPU: {cpu_data.shape}")
        sys.exit(1)
    
    if np.allclose(gpu_data, cpu_data, atol=$TOLERANCE):
        max_diff = np.abs(gpu_data - cpu_data).max()
        print(f"  ✓ Results match within tolerance ($TOLERANCE)")
        print(f"    Maximum difference: {max_diff:.2e}")
        print(f"    Total comparisons: {len(gpu_data)}")
    else:
        diff = np.abs(gpu_data - cpu_data)
        max_diff = diff.max()
        max_idx = diff.argmax()
        num_different = np.sum(diff > $TOLERANCE)
        
        print(f"  ✗ Results differ!")
        print(f"    Maximum difference: {max_diff:.2e} at index {max_idx}")
        print(f"    GPU value: {gpu_data[max_idx]:.8f}")
        print(f"    CPU value: {cpu_data[max_idx]:.8f}")
        print(f"    Number of differing values: {num_different} / {len(gpu_data)}")
        print(f"    Percentage different: {100.0 * num_different / len(gpu_data):.2f}%")
        
        # Show first 10 differences
        diff_indices = np.where(diff > $TOLERANCE)[0][:10]
        if len(diff_indices) > 0:
            print(f"    First {len(diff_indices)} differences:")
            for idx in diff_indices:
                print(f"      Index {idx}: GPU={gpu_data[idx]:.8f}, CPU={cpu_data[idx]:.8f}, Diff={diff[idx]:.2e}")
        
        sys.exit(1)
except Exception as e:
    print(f"  ERROR: {e}")
    sys.exit(1)
EOF
    
    return $?
}

# Compare all metrics and dimensions
TOTAL_COMPARISONS=0
PASSED_COMPARISONS=0
FAILED_COMPARISONS=0

for dim in "${DIMENSIONS[@]}"; do
    for metric in "${METRICS[@]}"; do
        # Use dimension-specific filename (required format)
        gpu_file="$GPU_STRATEGY_DIR/gpu_results_${metric}_${dim}.bin"
        cpu_file="$TRUST_DIR/${dim}/cpu_results_${metric}_${dim}.bin"
        
        TOTAL_COMPARISONS=$((TOTAL_COMPARISONS + 1))
        
        if compare_files "$gpu_file" "$cpu_file" "$metric" "$dim"; then
            PASSED_COMPARISONS=$((PASSED_COMPARISONS + 1))
        else
            FAILED_COMPARISONS=$((FAILED_COMPARISONS + 1))
        fi
    done
done

# Summary
echo ""
echo "=== Summary ==="
echo "Total comparisons: $TOTAL_COMPARISONS"
echo "Passed: $PASSED_COMPARISONS"
echo "Failed: $FAILED_COMPARISONS"

if [ $FAILED_COMPARISONS -eq 0 ]; then
    echo ""
    echo "✓ All comparisons passed! GPU results match CPU trust files."
    echo ""
    echo "Format Verification Summary:"
    echo "  - CPU files: float32 binary format (4 bytes per value)"
    echo "  - GPU files: float32 binary format (4 bytes per value)"
    echo "  - Layout: Both store results as flat array of float32 values"
    echo "  - Order: Both use row-major order (query_idx * num_database + db_idx)"
    exit 0
else
    echo ""
    echo "✗ Some comparisons failed. Check the output above for details."
    exit 1
fi
