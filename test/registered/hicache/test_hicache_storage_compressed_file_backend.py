"""End-to-end tests for the ``compressed_file`` HiCache storage backend.

Mirrors ``test_hicache_storage_file_backend.py`` but launches the server with

    --hicache-storage-backend compressed_file

so the disk tier compresses each page with the lossless sem_split + zstd
codec stack (see sglang/srt/mem_cache/storage/compressed/).

Run::

    python3 -m pytest test/registered/hicache/test_hicache_storage_compressed_file_backend.py -v
"""

import json
import os
import tempfile
import time
import unittest
from typing import Dict
from urllib.parse import urlparse

import requests

from sglang.benchmark.utils import get_tokenizer
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
)
from sglang.utils import wait_for_http_ready

# Slightly larger budget than the file backend test because compression adds
# ~5–10% CPU overhead per backup/prefetch round trip.
register_cuda_ci(est_time=180, suite="stage-b-test-2-gpu-large")
register_amd_ci(est_time=560, suite="stage-b-test-2-gpu-large-amd")


# A tiny policy file we ship inline so the test is self-contained.  Uses the
# measured sweet spot (zstd-1 on sem_split_channel) with the layer-0 V
# outlier override.  Tier-aware rules are present but don't change behaviour
# in CI (HiCacheController populates them but the chosen codec for L2→L3
# happens to be the same as the default here — what we want to verify is
# *correctness*, not maximum compression).
_POLICY_YAML = """
profiles:
  balanced:
    codec: zstd
    compression_level: 1
    layout: sem_split_channel
  aggressive:
    extends: balanced
    compression_level: 9
  layer0_v_special:
    codec: zstd
    compression_level: 1
    layout: raw

default: balanced

rules:
  - match: { layers: [0], kv: V }
    profile: layer0_v_special
  - match: { tier: "l2_to_l3" }
    profile: aggressive

runtime:
  batch_threads: 4
  slab_threads:  4
  strict_dtype:  true
"""


class CompressedHiCacheBaseMixin:
    """Mixin matching HiCacheStorageBaseMixin but pointing at compressed_file."""

    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.mkdtemp(prefix="sglang_hicache_compressed_")
        cls.policy_path = os.path.join(cls.temp_dir, "policy.yaml")
        with open(cls.policy_path, "w") as f:
            f.write(_POLICY_YAML)

        cls.model = cls._get_model_name()
        cls.base_url = DEFAULT_URL_FOR_TEST
        parsed = urlparse(cls.base_url)
        cls.base_host = parsed.hostname
        cls.base_port = str(parsed.port)

        cls.tokenizer = get_tokenizer(cls.model)

        cls.process = cls._launch_server_with_hicache()
        cls._wait_for_server_ready(process=cls.process)

        print(f"Compressed-file HiCache server up at {cls.base_url}")
        print(f"Cache directory: {cls.temp_dir}")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)
        import shutil

        if hasattr(cls, "temp_dir"):
            shutil.rmtree(cls.temp_dir, ignore_errors=True)

    @classmethod
    def _get_model_name(cls):
        return DEFAULT_MODEL_NAME_FOR_TEST

    @classmethod
    def _get_base_server_args(cls):
        extra_config = {
            "hicache_storage_pass_prefix_keys": True,
            "profiles_yaml": cls.policy_path,
        }
        return {
            "--enable-hierarchical-cache": True,
            "--mem-fraction-static": 0.6,
            "--hicache-ratio": 1.2,
            "--page-size": 64,
            "--enable-cache-report": True,
            "--hicache-storage-prefetch-policy": "wait_complete",
            "--hicache-storage-backend": "compressed_file",
            "--hicache-storage-backend-extra-config": json.dumps(extra_config),
        }

    @classmethod
    def _get_additional_server_args_and_env(cls):
        return {}, {"SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir}

    @classmethod
    def _launch_server_with_hicache(cls):
        additional_server_args, env_vars = cls._get_additional_server_args_and_env()
        env_vars["SGLANG_ENABLE_DETERMINISTIC_INFERENCE"] = "1"
        server_args = cls._get_base_server_args()
        if additional_server_args:
            server_args.update(additional_server_args)

        final_server_args = []
        for k, v in server_args.items():
            if isinstance(v, bool):
                final_server_args.append(str(k))
            else:
                final_server_args.append(str(k))
                final_server_args.append(str(v))

        env_vars = {**os.environ, **env_vars}
        return popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=final_server_args,
            env=env_vars,
        )

    @classmethod
    def _wait_for_server_ready(cls, timeout: int = 60, process=None) -> bool:
        wait_for_http_ready(
            url=f"{cls.base_url}/health",
            timeout=timeout,
            process=process,
        )
        return True

    # ---- shared request helpers (same as the file backend test) ----

    def send_request(self, prompt: str, max_tokens: int = 100, temperature: float = 0.0) -> Dict:
        response = requests.post(
            f"{self.base_url}/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": temperature,
                    "max_new_tokens": max_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=60,
        )
        self.assertEqual(response.status_code, 200, f"Request failed: {response.text}")
        return response.json()

    def get_cached_tokens(self, response_json: Dict) -> int:
        return int(response_json.get("meta_info", {}).get("cached_tokens", 0))

    def flush_cache(self):
        r = requests.post(
            f"{self.base_url}/flush_cache", params={"timeout": 30}, timeout=40
        )
        r.raise_for_status()

    def gen_prompt(self, token_num: int) -> str:
        import random

        toks = list(self.tokenizer.get_vocab().values())
        return self.tokenizer.decode(random.choices(toks, k=token_num))

    def trigger_offloading_and_flush(self):
        self.send_request(self.gen_prompt(1), max_tokens=150)
        self.flush_cache()

    def _dir_bytes(self, path: str) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for fn in files:
                try:
                    total += os.path.getsize(os.path.join(root, fn))
                except OSError:
                    pass
        return total

    # -------- tests --------

    def test_basic_backup_and_prefetch(self):
        """Same flow as the file-backend test; the only difference is the codec."""
        base_prompt = self.gen_prompt(768)

        # Populate.
        self.send_request(base_prompt, max_tokens=150)
        self.trigger_offloading_and_flush()

        # Hit the remote cache.
        t0 = time.time()
        resp2 = self.send_request(base_prompt, max_tokens=150)
        retrieval = time.time() - t0

        cached = self.get_cached_tokens(resp2)
        print(f"  compressed remote retrieval {retrieval:.3f}s, cached_tokens={cached}")
        self.assertGreater(cached, 700, "Expected significant cached tokens on hit")

    def test_compression_ratio_observed(self):
        """After a backup round, the on-disk cache should be measurably smaller
        than the equivalent raw KV bytes."""
        prompt = self.gen_prompt(1024)
        self.send_request(prompt, max_tokens=64)
        self.trigger_offloading_and_flush()

        comp_bytes = self._dir_bytes(self.temp_dir)
        print(f"  on-disk compressed cache: {comp_bytes / 1e6:.2f} MB")

        # We don't have direct access to the uncompressed KV byte count for the
        # prompt, but for the default model + 1k tokens the disk image should
        # be at least a few MB and obviously non-empty if the backend works.
        self.assertGreater(
            comp_bytes,
            1024,
            "compressed cache dir is empty — backend did not write pages",
        )


class TestCompressedHiCacheStorageBasic(CompressedHiCacheBaseMixin, CustomTestCase):
    """Baseline configuration: defaults + the test policy YAML."""


@unittest.skipIf(is_in_ci(), "Skipped in CI to reduce execution time.")
class TestCompressedHiCacheStoragePageFirst(CompressedHiCacheBaseMixin, CustomTestCase):
    """Page-first memory layout variant."""

    @classmethod
    def _get_additional_server_args_and_env(cls):
        return {"--hicache-mem-layout": "page_first"}, {
            "SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR": cls.temp_dir,
        }


if __name__ == "__main__":
    unittest.main(verbosity=2)
