#!/usr/bin/env python3
"""
Convert SIFT1M dataset from binary fvecs/ivecs format to JSON.

Parses fvecs (float32 vectors) and ivecs (int32 vectors) binary formats
and outputs JSON files for use by the Benchmark Client and data loading scripts.

Binary formats:
  fvecs: [dim: i32][v0: f32][v1: f32]...[v_{dim-1}: f32] repeated
  ivecs: [dim: i32][v0: i32][v1: i32]...[v_{dim-1}: i32] repeated

Usage:
    python scripts/convert_data.py [--data-dir data/] [--shard-size 100000]
"""

import argparse
import json
import os
import struct
import sys


def read_fvecs(filepath):
    """Parse an fvecs file and return a list of float32 vectors.

    Each vector is stored as:
        4 bytes (int32 dimension) + dim * 4 bytes (float32 values)

    Args:
        filepath: Path to the .fvecs file.

    Returns:
        List of lists of floats.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or dimensions are inconsistent.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"fvecs file not found: {filepath}")

    vectors = []
    expected_dim = None
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        while f.tell() < file_size:
            # Read dimension (4 bytes, int32)
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                if len(dim_bytes) == 0:
                    break  # Clean EOF
                raise ValueError(
                    f"Unexpected end of file reading dimension at byte {f.tell() - len(dim_bytes)} "
                    f"in {filepath}"
                )

            (dim,) = struct.unpack("<i", dim_bytes)
            if dim <= 0:
                raise ValueError(
                    f"Invalid dimension {dim} at vector index {len(vectors)} in {filepath}"
                )

            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at vector index {len(vectors)}: "
                    f"expected {expected_dim}, got {dim} in {filepath}"
                )

            # Read vector data (dim * 4 bytes, float32)
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                raise ValueError(
                    f"Unexpected end of file reading vector data at vector index {len(vectors)} "
                    f"in {filepath}. Expected {dim * 4} bytes, got {len(vec_bytes)}"
                )

            values = list(struct.unpack(f"<{dim}f", vec_bytes))
            vectors.append(values)

    if not vectors:
        raise ValueError(f"No vectors found in {filepath}")

    return vectors


def read_ivecs(filepath):
    """Parse an ivecs file and return a list of int32 vectors.

    Each vector is stored as:
        4 bytes (int32 dimension) + dim * 4 bytes (int32 values)

    Args:
        filepath: Path to the .ivecs file.

    Returns:
        List of lists of ints.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file format is invalid or dimensions are inconsistent.
    """
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"ivecs file not found: {filepath}")

    vectors = []
    expected_dim = None
    file_size = os.path.getsize(filepath)

    with open(filepath, "rb") as f:
        while f.tell() < file_size:
            # Read dimension (4 bytes, int32)
            dim_bytes = f.read(4)
            if len(dim_bytes) < 4:
                if len(dim_bytes) == 0:
                    break  # Clean EOF
                raise ValueError(
                    f"Unexpected end of file reading dimension at byte {f.tell() - len(dim_bytes)} "
                    f"in {filepath}"
                )

            (dim,) = struct.unpack("<i", dim_bytes)
            if dim <= 0:
                raise ValueError(
                    f"Invalid dimension {dim} at vector index {len(vectors)} in {filepath}"
                )

            if expected_dim is None:
                expected_dim = dim
            elif dim != expected_dim:
                raise ValueError(
                    f"Dimension mismatch at vector index {len(vectors)}: "
                    f"expected {expected_dim}, got {dim} in {filepath}"
                )

            # Read vector data (dim * 4 bytes, int32)
            vec_bytes = f.read(dim * 4)
            if len(vec_bytes) < dim * 4:
                raise ValueError(
                    f"Unexpected end of file reading vector data at vector index {len(vectors)} "
                    f"in {filepath}. Expected {dim * 4} bytes, got {len(vec_bytes)}"
                )

            values = list(struct.unpack(f"<{dim}i", vec_bytes))
            vectors.append(values)

    if not vectors:
        raise ValueError(f"No vectors found in {filepath}")

    return vectors


def convert_base_vectors(vectors, output_dir, shard_size):
    """Convert base vectors to JSON format, with optional sharding.

    Output format per entry: {"id": <int>, "vector": [<float>, ...]}

    Args:
        vectors: List of float vectors.
        output_dir: Directory to write JSON files.
        shard_size: Max vectors per shard file. If total <= shard_size,
                    writes a single base_vectors.json.

    Returns:
        List of output file paths written.
    """
    os.makedirs(output_dir, exist_ok=True)
    total = len(vectors)
    output_files = []

    if total <= shard_size:
        # Single file output
        records = [{"id": i, "vector": v} for i, v in enumerate(vectors)]
        path = os.path.join(output_dir, "base_vectors.json")
        _write_json(records, path)
        output_files.append(path)
    else:
        # Sharded output
        shard_idx = 0
        for start in range(0, total, shard_size):
            end = min(start + shard_size, total)
            records = [{"id": i, "vector": vectors[i]} for i in range(start, end)]
            path = os.path.join(output_dir, f"base_vectors_{shard_idx}.json")
            _write_json(records, path)
            output_files.append(path)
            shard_idx += 1

    return output_files


def convert_query_vectors(vectors, output_dir):
    """Convert query vectors to JSON format.

    Output format per entry: {"id": <int>, "vector": [<float>, ...]}

    Args:
        vectors: List of float vectors.
        output_dir: Directory to write the JSON file.

    Returns:
        Output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = [{"id": i, "vector": v} for i, v in enumerate(vectors)]
    path = os.path.join(output_dir, "query_vectors.json")
    _write_json(records, path)
    return path


def convert_ground_truth(vectors, output_dir):
    """Convert ground truth neighbor IDs to JSON format.

    Output format per entry: {"query_id": <int>, "neighbors": [<int>, ...]}

    Args:
        vectors: List of int vectors (neighbor ID lists).
        output_dir: Directory to write the JSON file.

    Returns:
        Output file path.
    """
    os.makedirs(output_dir, exist_ok=True)
    records = [{"query_id": i, "neighbors": v} for i, v in enumerate(vectors)]
    path = os.path.join(output_dir, "ground_truth.json")
    _write_json(records, path)
    return path


def _write_json(data, path):
    """Write data to a JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Written {path} ({size_mb:.1f} MB, {len(data)} records)")


def main():
    parser = argparse.ArgumentParser(
        description="Convert SIFT1M fvecs/ivecs binary files to JSON format."
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Directory containing fvecs/ivecs files and for JSON output (default: data/)",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100000,
        help="Maximum vectors per shard file for base vectors (default: 100000)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    shard_size = args.shard_size

    if shard_size <= 0:
        print("Error: --shard-size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    # Input file paths
    base_fvecs = os.path.join(data_dir, "sift_base.fvecs")
    query_fvecs = os.path.join(data_dir, "sift_query.fvecs")
    gt_ivecs = os.path.join(data_dir, "sift_groundtruth.ivecs")

    # --- Convert base vectors ---
    print(f"Parsing base vectors from {base_fvecs}...")
    try:
        base_vectors = read_fvecs(base_fvecs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(base_vectors)} vectors, dim={len(base_vectors[0])}")

    print("Converting base vectors to JSON...")
    base_files = convert_base_vectors(base_vectors, data_dir, shard_size)

    # --- Convert query vectors ---
    print(f"\nParsing query vectors from {query_fvecs}...")
    try:
        query_vectors = read_fvecs(query_fvecs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(query_vectors)} vectors, dim={len(query_vectors[0])}")

    print("Converting query vectors to JSON...")
    query_file = convert_query_vectors(query_vectors, data_dir)

    # --- Convert ground truth ---
    print(f"\nParsing ground truth from {gt_ivecs}...")
    try:
        gt_vectors = read_ivecs(gt_ivecs)
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    print(f"  Loaded {len(gt_vectors)} entries, neighbors per query={len(gt_vectors[0])}")

    print("Converting ground truth to JSON...")
    gt_file = convert_ground_truth(gt_vectors, data_dir)

    # --- Summary ---
    print("\nConversion complete!")
    print(f"  Base vectors: {len(base_files)} file(s)")
    for f in base_files:
        print(f"    {f}")
    print(f"  Query vectors: {query_file}")
    print(f"  Ground truth:  {gt_file}")


if __name__ == "__main__":
    main()
