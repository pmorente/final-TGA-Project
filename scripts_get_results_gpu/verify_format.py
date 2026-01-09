#!/usr/bin/env python3
"""
Verify that CPU and GPU binary result files use the same format.
Both should use float32 (4 bytes per value) in binary format.
"""

import os
import sys
import numpy as np

def verify_file_format(filepath, label):
    """Verify that a file uses float32 binary format."""
    if not os.path.exists(filepath):
        print(f"  ERROR: {label} file not found: {filepath}")
        return False, None
    
    # Check file size (must be multiple of 4)
    file_size = os.path.getsize(filepath)
    if file_size % 4 != 0:
        print(f"  ERROR: {label} file size ({file_size} bytes) is not a multiple of 4")
        return False, None
    
    # Try to read as float32
    try:
        data = np.fromfile(filepath, dtype=np.float32)
        num_values = len(data)
        print(f"  {label}: {num_values} float32 values ({file_size} bytes)")
        return True, data
    except Exception as e:
        print(f"  ERROR: Could not read {label} file as float32: {e}")
        return False, None

def compare_formats(cpu_file, gpu_file):
    """Compare CPU and GPU file formats."""
    print(f"\nComparing formats:")
    print(f"  CPU: {cpu_file}")
    print(f"  GPU: {gpu_file}")
    
    cpu_ok, cpu_data = verify_file_format(cpu_file, "CPU")
    gpu_ok, gpu_data = verify_file_format(gpu_file, "GPU")
    
    if not cpu_ok or not gpu_ok:
        return False
    
    # Check if shapes match
    if cpu_data.shape != gpu_data.shape:
        print(f"  ERROR: Array shapes differ! CPU: {cpu_data.shape}, GPU: {gpu_data.shape}")
        return False
    
    print(f"  ✓ Both files use float32 binary format")
    print(f"  ✓ Both have {len(cpu_data)} values")
    print(f"  ✓ Formats are consistent")
    
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python3 verify_format.py <cpu_file> <gpu_file>")
        sys.exit(1)
    
    cpu_file = sys.argv[1]
    gpu_file = sys.argv[2]
    
    if compare_formats(cpu_file, gpu_file):
        print("\n✓ Format verification passed!")
        sys.exit(0)
    else:
        print("\n✗ Format verification failed!")
        sys.exit(1)

