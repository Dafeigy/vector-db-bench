#!/usr/bin/env python3
"""
Download the SIFT1M dataset from HuggingFace.

Clones the qbo-odp/sift1m repository using git-lfs and moves the relevant
files (sift_base.fvecs, sift_query.fvecs, sift_groundtruth.ivecs) into
the data/ directory.

Requires: git, git-lfs

Usage:
    python scripts/download_dataset.py
"""

import os
import shutil
import subprocess
import sys

HF_REPO = "https://huggingface.co/datasets/qbo-odp/sift1m"

REQUIRED_FILES = {
    "sift_base.fvecs": "sift_base.fvecs",
    "sift_query.fvecs": "sift_query.fvecs",
    "sift_groundtruth.ivecs": "sift_groundtruth.ivecs",
}

DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"
)
TMP_DIR = os.path.join(DATA_DIR, "_tmp_download")


def all_files_exist():
    """Check if all required dataset files already exist in data/."""
    for dest_name in REQUIRED_FILES.values():
        if not os.path.isfile(os.path.join(DATA_DIR, dest_name)):
            return False
    return True


def check_git_lfs():
    """Verify that git and git-lfs are available."""
    try:
        subprocess.run(
            ["git", "--version"],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print("Error: git is not installed or not in PATH.", file=sys.stderr)
        sys.exit(1)

    try:
        subprocess.run(
            ["git", "lfs", "version"],
            check=True, capture_output=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        print(
            "Error: git-lfs is not installed. Install it with:\n"
            "  brew install git-lfs   (macOS)\n"
            "  apt install git-lfs    (Ubuntu/Debian)\n"
            "  git lfs install        (after installing the binary)",
            file=sys.stderr,
        )
        sys.exit(1)


def clone_repo():
    """Clone the HuggingFace dataset repo with LFS files."""
    clone_dir = os.path.join(TMP_DIR, "sift1m")

    if os.path.isdir(clone_dir):
        shutil.rmtree(clone_dir)

    print(f"Cloning {HF_REPO} ...")
    print("(This downloads ~570 MB of LFS files, may take a few minutes)")

    subprocess.run(
        ["git", "clone", HF_REPO, clone_dir],
        check=True,
    )

    return clone_dir


def move_files(clone_dir):
    """Move required files from the cloned repo to data/."""
    for src_name, dest_name in REQUIRED_FILES.items():
        src_path = os.path.join(clone_dir, src_name)
        dest_path = os.path.join(DATA_DIR, dest_name)

        if not os.path.isfile(src_path):
            raise FileNotFoundError(
                f"Expected file not found in cloned repo: {src_path}"
            )

        size_mb = os.path.getsize(src_path) / (1024 * 1024)
        print(f"  {src_name} -> data/{dest_name}  ({size_mb:.1f} MB)")
        shutil.move(src_path, dest_path)


def cleanup():
    """Remove temporary clone directory."""
    if os.path.exists(TMP_DIR):
        print("Cleaning up temporary files...")
        shutil.rmtree(TMP_DIR)


def main():
    if all_files_exist():
        print("All dataset files already exist in data/. Skipping download.")
        print("Files:")
        for dest_name in REQUIRED_FILES.values():
            path = os.path.join(DATA_DIR, dest_name)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")
        return

    check_git_lfs()

    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(TMP_DIR, exist_ok=True)

    try:
        clone_dir = clone_repo()
        move_files(clone_dir)

        print("\nDataset download complete! Files in data/:")
        for dest_name in REQUIRED_FILES.values():
            path = os.path.join(DATA_DIR, dest_name)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {path} ({size_mb:.1f} MB)")

    except subprocess.CalledProcessError as e:
        print(f"\nGit clone failed (exit code {e.returncode})", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    finally:
        cleanup()


if __name__ == "__main__":
    main()
