#!/usr/bin/env python3
"""Unit tests for convert_data.py fvecs/ivecs parsing and JSON conversion."""

import json
import os
import struct
import sys
import tempfile
import unittest

# Add scripts directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_data import (
    read_fvecs,
    read_ivecs,
    convert_base_vectors,
    convert_query_vectors,
    convert_ground_truth,
)


def write_fvecs(filepath, vectors):
    """Write vectors in fvecs binary format."""
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}f", *vec))


def write_ivecs(filepath, vectors):
    """Write vectors in ivecs binary format."""
    with open(filepath, "wb") as f:
        for vec in vectors:
            dim = len(vec)
            f.write(struct.pack("<i", dim))
            f.write(struct.pack(f"<{dim}i", *vec))


class TestReadFvecs(unittest.TestCase):
    def test_single_vector(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            write_fvecs(path, [[1.0, 2.0, 3.0]])
        try:
            result = read_fvecs(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 3)
            self.assertAlmostEqual(result[0][0], 1.0)
            self.assertAlmostEqual(result[0][1], 2.0)
            self.assertAlmostEqual(result[0][2], 3.0)
        finally:
            os.unlink(path)

    def test_multiple_vectors(self):
        vecs = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            write_fvecs(path, vecs)
        try:
            result = read_fvecs(path)
            self.assertEqual(len(result), 3)
            for i, vec in enumerate(vecs):
                for j, val in enumerate(vec):
                    self.assertAlmostEqual(result[i][j], val)
        finally:
            os.unlink(path)

    def test_128_dim_vector(self):
        vec = [float(i) for i in range(128)]
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            write_fvecs(path, [vec])
        try:
            result = read_fvecs(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(len(result[0]), 128)
            for i in range(128):
                self.assertAlmostEqual(result[i // 128][i % 128], float(i))
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_fvecs("/nonexistent/path.fvecs")

    def test_dimension_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            # Write first vector with dim=2
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<2f", 1.0, 2.0))
            # Write second vector with dim=3 (mismatch)
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<3f", 1.0, 2.0, 3.0))
        try:
            with self.assertRaises(ValueError) as ctx:
                read_fvecs(path)
            self.assertIn("Dimension mismatch", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_truncated_file(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<f", 1.0))  # Only 1 of 3 floats
        try:
            with self.assertRaises(ValueError) as ctx:
                read_fvecs(path)
            self.assertIn("Unexpected end of file", str(ctx.exception))
        finally:
            os.unlink(path)

    def test_invalid_dimension(self):
        with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
            path = f.name
            f.write(struct.pack("<i", -1))
        try:
            with self.assertRaises(ValueError) as ctx:
                read_fvecs(path)
            self.assertIn("Invalid dimension", str(ctx.exception))
        finally:
            os.unlink(path)


class TestReadIvecs(unittest.TestCase):
    def test_single_vector(self):
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            path = f.name
            write_ivecs(path, [[10, 20, 30]])
        try:
            result = read_ivecs(path)
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0], [10, 20, 30])
        finally:
            os.unlink(path)

    def test_multiple_vectors(self):
        vecs = [[1, 2, 3], [4, 5, 6]]
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            path = f.name
            write_ivecs(path, vecs)
        try:
            result = read_ivecs(path)
            self.assertEqual(result, vecs)
        finally:
            os.unlink(path)

    def test_file_not_found(self):
        with self.assertRaises(FileNotFoundError):
            read_ivecs("/nonexistent/path.ivecs")

    def test_dimension_mismatch(self):
        with tempfile.NamedTemporaryFile(suffix=".ivecs", delete=False) as f:
            path = f.name
            f.write(struct.pack("<i", 2))
            f.write(struct.pack("<2i", 1, 2))
            f.write(struct.pack("<i", 3))
            f.write(struct.pack("<3i", 1, 2, 3))
        try:
            with self.assertRaises(ValueError) as ctx:
                read_ivecs(path)
            self.assertIn("Dimension mismatch", str(ctx.exception))
        finally:
            os.unlink(path)


class TestJsonConversion(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmpdir)

    def test_base_vectors_single_file(self):
        vecs = [[1.0, 2.0], [3.0, 4.0]]
        files = convert_base_vectors(vecs, self.tmpdir, shard_size=100)
        self.assertEqual(len(files), 1)
        self.assertTrue(files[0].endswith("base_vectors.json"))
        with open(files[0]) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["id"], 0)
        self.assertEqual(data[0]["vector"], [1.0, 2.0])
        self.assertEqual(data[1]["id"], 1)
        self.assertEqual(data[1]["vector"], [3.0, 4.0])

    def test_base_vectors_sharded(self):
        vecs = [[float(i)] for i in range(5)]
        files = convert_base_vectors(vecs, self.tmpdir, shard_size=2)
        self.assertEqual(len(files), 3)  # 2 + 2 + 1
        self.assertTrue(files[0].endswith("base_vectors_0.json"))
        self.assertTrue(files[1].endswith("base_vectors_1.json"))
        self.assertTrue(files[2].endswith("base_vectors_2.json"))

        # Verify shard contents
        with open(files[0]) as f:
            shard0 = json.load(f)
        self.assertEqual(len(shard0), 2)
        self.assertEqual(shard0[0]["id"], 0)
        self.assertEqual(shard0[1]["id"], 1)

        with open(files[2]) as f:
            shard2 = json.load(f)
        self.assertEqual(len(shard2), 1)
        self.assertEqual(shard2[0]["id"], 4)

    def test_query_vectors(self):
        vecs = [[0.5, 0.6], [0.7, 0.8]]
        path = convert_query_vectors(vecs, self.tmpdir)
        self.assertTrue(path.endswith("query_vectors.json"))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["id"], 0)
        self.assertEqual(data[0]["vector"], [0.5, 0.6])

    def test_ground_truth(self):
        gt = [[23, 456, 789], [100, 200, 300]]
        path = convert_ground_truth(gt, self.tmpdir)
        self.assertTrue(path.endswith("ground_truth.json"))
        with open(path) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)
        self.assertEqual(data[0]["query_id"], 0)
        self.assertEqual(data[0]["neighbors"], [23, 456, 789])
        self.assertEqual(data[1]["query_id"], 1)
        self.assertEqual(data[1]["neighbors"], [100, 200, 300])


class TestRoundTrip(unittest.TestCase):
    """Test full round-trip: write binary -> parse -> convert to JSON -> verify."""

    def test_fvecs_to_json_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        try:
            original = [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
            fvecs_path = os.path.join(tmpdir, "test.fvecs")
            write_fvecs(fvecs_path, original)

            parsed = read_fvecs(fvecs_path)
            files = convert_base_vectors(parsed, tmpdir, shard_size=100)

            with open(files[0]) as f:
                data = json.load(f)

            for i, rec in enumerate(data):
                self.assertEqual(rec["id"], i)
                for j, val in enumerate(rec["vector"]):
                    self.assertAlmostEqual(val, original[i][j], places=5)
        finally:
            import shutil
            shutil.rmtree(tmpdir)

    def test_ivecs_to_json_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        try:
            original = [[10, 20, 30], [40, 50, 60]]
            ivecs_path = os.path.join(tmpdir, "test.ivecs")
            write_ivecs(ivecs_path, original)

            parsed = read_ivecs(ivecs_path)
            path = convert_ground_truth(parsed, tmpdir)

            with open(path) as f:
                data = json.load(f)

            for i, rec in enumerate(data):
                self.assertEqual(rec["query_id"], i)
                self.assertEqual(rec["neighbors"], original[i])
        finally:
            import shutil
            shutil.rmtree(tmpdir)


if __name__ == "__main__":
    unittest.main()
