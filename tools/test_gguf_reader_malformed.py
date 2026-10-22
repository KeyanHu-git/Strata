"""A malformed GGUF header is refused with a ValueError that names the file, never a KeyError, OverflowError,
MemoryError or struct.error (#1919). Tensor-size support and setup compatibility policy are unchanged.
Over minimal GGUF v3 files written here (no download, no model, no GPU).

    python -m unittest tools.test_gguf_reader_malformed
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


class Stop(Exception):
    """setup.fail(), caught instead of exiting."""


def gstr(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def header(n_tensors: int, kv: bytes = b"", n_kv: int = 0) -> bytes:
    return struct.pack("<IIQQ", 0x46554747, 3, n_tensors, n_kv) + kv


def tensor(name: bytes, n: int, type_id: int, offset: int) -> bytes:
    return gstr(name) + struct.pack("<IQIQ", 1, n, type_id, offset)


def pad(b: bytes, data: int) -> bytes:
    return b + bytes(-len(b) % 32 + data)


class MalformedHeader(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "m-00001-of-00002.gguf"

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, raw: bytes) -> str:
        self.path.write_bytes(raw)
        with self.assertRaises(ValueError) as cm:
            GGUFFile(self.path)
        self.assertIn(self.path.name, str(cm.exception))
        return str(cm.exception)

    def test_unknown_metadata_value_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<I", 99), 1), 0))

    def test_unknown_array_element_type(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 99, 1), 1), 0))

    def test_string_length_past_the_end(self):
        self.refused(pad(header(0, struct.pack("<Q", 2 ** 63), 1), 0))

    def test_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 4, 2 ** 40), 1), 0))

    def test_string_array_count_past_the_end(self):
        self.refused(pad(header(0, gstr(b"k") + struct.pack("<IIQ", 9, 8, 2 ** 62), 1), 0))

    def test_tensor_count_past_the_end(self):
        self.refused(header(2 ** 40))

    def test_header_cut_mid_tensor_directory(self):
        self.refused(header(1) + tensor(b"blk.0.w", 8, 0, 0)[:-5])

    def test_header_cut_inside_the_magic(self):
        self.refused(b"GG")


class Alignment(unittest.TestCase):
    setUp = MalformedHeader.setUp
    tearDown = MalformedHeader.tearDown
    refused = MalformedHeader.refused
    def test_invalid_alignment(self):
        for alignment in (0, -8, 3):
            with self.subTest(alignment=alignment):
                kv = gstr(b"general.alignment") + struct.pack("<Ii", 5, alignment)
                self.assertIn("alignment", self.refused(pad(header(0, kv, 1), 0)))


class UnknownTensorCompatibility(unittest.TestCase):
    def test_readable_unknown_tensor_keeps_setup_policy(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "future.gguf"
            raw = pad(header(2) + tensor(b"known", 8, 0, 0) + tensor(b"future", 64, 999, 32), 256)
            p.write_bytes(raw)
            g = GGUFFile(p)
            self.assertIsNone(g.tensors[1].expected_bytes())
            self.assertTrue(S.whole_shard(p))
            S.check_shards([p])
            self.assertEqual(p.read_bytes(), raw)


if __name__ == "__main__":
    unittest.main()
