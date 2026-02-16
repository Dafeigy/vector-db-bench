#!/usr/bin/env python3
"""
Load base vectors from JSON files into the vector database service via HTTP /bulk_insert.

Reads base vectors in JSON format (single file or sharded) and sends them
to the target server in configurable batches.

JSON format: [{"id": int, "vector": [float, ...]}, ...]
/bulk_insert expects: {"vectors": [{"id": int, "vector": [float, ...]}, ...]}
/bulk_insert returns: {"status": "ok", "inserted": N}

This script is READ-ONLY — the model under test cannot modify it.

Usage:
    python scripts/load_data.py [--server-url http://127.0.0.1:8080] [--data-dir data/] [--batch-size 5000]
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.error
import urllib.request


def discover_base_vector_files(data_dir):
    """Find base vector JSON files in the data directory.

    Supports two layouts:
      - Single file: base_vectors.json
      - Sharded files: base_vectors_0.json, base_vectors_1.json, ...

    Args:
        data_dir: Path to the data directory.

    Returns:
        Sorted list of file paths.

    Raises:
        FileNotFoundError: If no base vector files are found.
    """
    single = os.path.join(data_dir, "base_vectors.json")
    if os.path.isfile(single):
        return [single]

    pattern = os.path.join(data_dir, "base_vectors_*.json")
    shards = sorted(glob.glob(pattern))
    if not shards:
        raise FileNotFoundError(
            f"No base vector files found in {data_dir}. "
            f"Expected base_vectors.json or base_vectors_0.json, base_vectors_1.json, ..."
        )
    return shards


def load_vectors_from_file(filepath):
    """Load vectors from a single JSON file.

    Args:
        filepath: Path to the JSON file.

    Returns:
        List of dicts with "id" and "vector" keys.

    Raises:
        ValueError: If the file cannot be parsed or has unexpected format.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON array in {filepath}, got {type(data).__name__}")

    return data


def send_batch(server_url, batch):
    """Send a batch of vectors to the server via POST /bulk_insert.

    Args:
        server_url: Base URL of the server (e.g. http://127.0.0.1:8080).
        batch: List of vector dicts ({"id": int, "vector": [float, ...]}).

    Returns:
        Number of vectors inserted as reported by the server.

    Raises:
        RuntimeError: If the server returns an error or unexpected response.
    """
    url = f"{server_url.rstrip('/')}/bulk_insert"
    payload = json.dumps({"vectors": batch}).encode("utf-8")

    req = urllib.request.Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Server returned HTTP {e.code}: {error_body}"
        ) from e
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Cannot connect to {url}: {e.reason}"
        ) from e

    if body.get("status") != "ok":
        raise RuntimeError(f"Unexpected response from server: {body}")

    return body.get("inserted", len(batch))


def load_data(server_url, data_dir, batch_size):
    """Load all base vectors into the server.

    Args:
        server_url: Base URL of the target server.
        data_dir: Directory containing base vector JSON files.
        batch_size: Number of vectors per bulk_insert request.

    Returns:
        Total number of vectors inserted.
    """
    files = discover_base_vector_files(data_dir)
    print(f"Found {len(files)} base vector file(s) in {data_dir}")

    total_inserted = 0
    total_vectors = 0
    start_time = time.time()

    for filepath in files:
        print(f"  Loading {os.path.basename(filepath)}...")
        vectors = load_vectors_from_file(filepath)
        total_vectors += len(vectors)

        for i in range(0, len(vectors), batch_size):
            batch = vectors[i : i + batch_size]
            inserted = send_batch(server_url, batch)
            total_inserted += inserted

            elapsed = time.time() - start_time
            rate = total_inserted / elapsed if elapsed > 0 else 0
            print(
                f"    Inserted {total_inserted}/{total_vectors} vectors "
                f"({elapsed:.1f}s elapsed, {rate:.0f} vec/s)",
                flush=True,
            )

    elapsed = time.time() - start_time
    print(f"\nLoad complete: {total_inserted} vectors inserted in {elapsed:.1f}s")
    return total_inserted


def main():
    parser = argparse.ArgumentParser(
        description="Load base vectors into the vector database service via /bulk_insert."
    )
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8080",
        help="Base URL of the vector database server (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--data-dir",
        default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"),
        help="Directory containing base vector JSON files (default: data/)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=5000,
        help="Number of vectors per bulk_insert request (default: 5000)",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        print("Error: --batch-size must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    try:
        load_data(args.server_url, args.data_dir, args.batch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
