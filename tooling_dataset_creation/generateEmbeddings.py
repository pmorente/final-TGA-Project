#!/usr/bin/env python3
"""
Script to generate embeddings from CSV files using sentence transformers.
Reads CSV with id,text columns and saves embeddings to binary file.
"""

import argparse
import sys
import os
import csv
import struct
import numpy as np
from pathlib import Path
from typing import List, Tuple

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from tools.embeddingGenerator import EmbeddingGenerator


def read_csv_file(file_path: str) -> Tuple[List[str], List[str]]:
    """
    Read CSV file with id,text columns and return lists of ids and texts.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        Tuple of (ids, texts) lists
    """
    try:
        ids = []
        texts = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            # Check if required columns exist
            if 'id' not in reader.fieldnames or 'text' not in reader.fieldnames:
                print(f"Error: CSV file must have 'id' and 'text' columns.")
                print(f"Found columns: {reader.fieldnames}")
                sys.exit(1)
            
            for row in reader:
                if row['id'] and row['text']:  # Skip empty rows
                    ids.append(row['id'].strip())
                    texts.append(row['text'].strip())
        
        return ids, texts
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading CSV file '{file_path}': {e}")
        sys.exit(1)


def save_embeddings_binary(ids: List[str], embeddings: np.ndarray, output_path: str):
    """
    Save embeddings to a binary file with format:
    - Header: magic number (4 bytes "EMBD"), version (1 byte), 
              num_records (int32), embedding_dim (int32)
    - For each record: id_length (int32), id (utf-8 bytes), 
                       embedding (float32 array of embedding_dim size)
    
    Args:
        ids: List of IDs corresponding to embeddings
        embeddings: Numpy array of embeddings (shape: [num_records, embedding_dim])
        output_path: Path to save the binary file
    """
    try:
        num_records = len(ids)
        embedding_dim = embeddings.shape[1]
        
        if num_records != embeddings.shape[0]:
            raise ValueError(f"Mismatch: {num_records} IDs but {embeddings.shape[0]} embeddings")
        
        with open(output_path, 'wb') as f:
            # Write header
            # Magic number: "EMBD" (4 bytes)
            f.write(b'EMBD')
            # Version: 1 (1 byte)
            f.write(struct.pack('B', 1))
            # Number of records (int32)
            f.write(struct.pack('i', num_records))
            # Embedding dimension (int32)
            f.write(struct.pack('i', embedding_dim))
            
            # Write each record
            for id_str, embedding in zip(ids, embeddings):
                # Convert ID to bytes
                id_bytes = id_str.encode('utf-8')
                id_length = len(id_bytes)
                
                # Write id_length (int32)
                f.write(struct.pack('i', id_length))
                # Write id (bytes)
                f.write(id_bytes)
                # Write embedding (float32 array)
                f.write(embedding.astype(np.float32).tobytes())
        
        print(f"Embeddings saved to: {output_path}")
        print(f"Total records: {num_records}")
        print(f"Embedding dimension: {embedding_dim}")
        print(f"File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")
    except Exception as e:
        print(f"Error saving embeddings: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Generate embeddings from CSV files using sentence transformers',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python generateEmbeddings.py input.csv -o embeddings.bin
  python generateEmbeddings.py input.csv -m all-mpnet-base-v2 -o embeddings.bin
  python generateEmbeddings.py input.csv --mode batch --batch-size 32 -o embeddings.bin
        """
    )
    
    parser.add_argument(
        'input_file',
        type=str,
        help='Path to input CSV file with id,text columns'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Output file path for embeddings (.bin file)'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='all-MiniLM-L6-v2',
        choices=['all-mpnet-base-v2', 'all-MiniLM-L6-v2', 'all-roberta-large-v1'],
        help='Model to use (default: all-MiniLM-L6-v2)'
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='sequential',
        choices=['sequential', 'batch'],
        help='Encoding mode: sequential (one by one) or batch (default: sequential)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Batch size for batch mode (required if --mode is batch)'
    )
    
    parser.add_argument(
        '--no-progress',
        action='store_true',
        help='Disable progress bar during encoding'
    )
    
    args = parser.parse_args()
    
    # Build full model name
    model_name = f"sentence-transformers/{args.model}"
    
    # Read input CSV file
    print(f"Reading input CSV file: {args.input_file}")
    ids, texts = read_csv_file(args.input_file)
    print(f"Loaded {len(texts)} records from CSV file")
    
    # Initialize embedding generator
    print(f"\n{'='*60}")
    print(f"Loading model: {model_name}")
    print(f"{'='*60}")
    generator = EmbeddingGenerator(model_name)
    
    # Validate arguments for batch mode
    if args.mode == 'batch' and args.batch_size is None:
        print(f"Error: --batch-size is required for batch mode")
        sys.exit(1)
    
    # Time embedding generation
    print(f"\nGenerating embeddings for {len(texts)} sentences using {args.mode} mode...")
    if args.mode == 'batch':
        print(f"Batch size: {args.batch_size}")
    
    embeddings, encode_time_sec = generator.encode(
        texts, 
        mode=args.mode,
        batch_size=args.batch_size,
        show_progress_bar=not args.no_progress
    )
    
    print(f"\nEmbedding generation completed in {encode_time_sec:.5f} seconds")
    
    # Get embedding dimension from the generated embeddings
    embedding_dim = embeddings.shape[1]
    
    # Prepare output path in outputs_embeddings folder based on dimension
    outputs_dir = Path(__file__).parent / 'outputs_embeddings' / f'dimension-{embedding_dim}'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract filename from user-provided output path
    output_filename = Path(args.output).name
    output_path = outputs_dir / output_filename
    
    # Save embeddings to binary file
    print(f"\nSaving embeddings to: {output_path}")
    save_embeddings_binary(ids, embeddings, str(output_path))
    
    print("\nDone!")


if __name__ == "__main__":
    main()
