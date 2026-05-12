"""Smoke tests for TTModelRunner.

Heavy ModelRunner construction is exercised end-to-end in Phase E.5/H.1
(real server launch). Here we just check the import surface and structural
overrides — the device-string fix and the dummy KV/Model classes.
"""

from __future__ import annotations

import unittest

import torch


class TestTTModelRunnerImport(unittest.TestCase):
    def test_model_runner_class_importable(self):
        from sglang.srt.hardware_backend.tenstorrent.model_runner import TTModelRunner

        self.assertTrue(hasattr(TTModelRunner, "load_model"))
        self.assertTrue(hasattr(TTModelRunner, "initialize"))
        self.assertTrue(hasattr(TTModelRunner, "__init__"))

    def test_dummy_kv_cache_shape(self):
        from sglang.srt.hardware_backend.tenstorrent.model_runner_stub import (
            _DummyKVCache,
        )

        kv = _DummyKVCache(size=128, dtype=torch.bfloat16, device="cpu")
        self.assertEqual(kv.size, 128)
        self.assertEqual(kv.page_size, 1)
        self.assertEqual(kv.layer_num, 0)
        self.assertEqual(kv.mem_usage, 0)
        self.assertIsNone(kv.custom_mem_pool)
        self.assertEqual(kv.get_kv_size_bytes(), (0, 0))

    def test_dummy_kv_cache_buffers_raise(self):
        from sglang.srt.hardware_backend.tenstorrent.model_runner_stub import (
            _DummyKVCache,
        )

        kv = _DummyKVCache(size=8, dtype=torch.bfloat16, device="cpu")
        with self.assertRaises(RuntimeError):
            kv.get_key_buffer(0)
        with self.assertRaises(RuntimeError):
            kv.get_value_buffer(0)
        with self.assertRaises(RuntimeError):
            kv.get_kv_buffer(0)
        with self.assertRaises(RuntimeError):
            kv.set_kv_buffer(None, None, None, None)

    def test_dummy_model_has_forward(self):
        from sglang.srt.hardware_backend.tenstorrent.model_runner_stub import (
            _DummyModel,
        )

        m = _DummyModel()
        self.assertTrue(callable(m.forward))
        self.assertIsNone(m.forward())


if __name__ == "__main__":
    unittest.main()
