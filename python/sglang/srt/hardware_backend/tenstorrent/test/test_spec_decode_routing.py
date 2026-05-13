# SPDX-License-Identifier: Apache-2.0
"""INV-8 spec_info routing test.

CPU-only — verifies SpecDecodeAdapter correctly dispatches verify-batches
to _verify_forward, and standard batches to existing prefill/decode.
"""
import pytest
import torch
from types import SimpleNamespace


def test_spec_decode_adapter_routes_verify():
    from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
    adapter = SpecDecodeAdapter(model=SimpleNamespace())
    fake_batch = SimpleNamespace(
        spec_info=SimpleNamespace(draft_tokens=torch.tensor([1, 2, 3])),
        forward_mode=SimpleNamespace(is_extend=lambda: False, is_decode=lambda: False),
        input_ids=torch.tensor([10, 20, 30]),
        positions=torch.tensor([0, 1, 2]),
    )
    # Adapter should call _verify_forward; mock it
    adapter._verify_forward = lambda fb: "verify_path"
    adapter._standard_forward = lambda fb: "standard_path"
    assert adapter.forward(fake_batch) == "verify_path"


def test_spec_decode_adapter_routes_standard():
    from sglang.srt.hardware_backend.tenstorrent.models.spec_decode import SpecDecodeAdapter
    adapter = SpecDecodeAdapter(model=SimpleNamespace())
    fake_batch = SimpleNamespace(
        spec_info=None,
        forward_mode=SimpleNamespace(is_extend=lambda: True, is_decode=lambda: False),
        input_ids=torch.tensor([10, 20, 30]),
        positions=torch.tensor([0, 1, 2]),
    )
    adapter._verify_forward = lambda fb: "verify_path"
    adapter._standard_forward = lambda fb: "standard_path"
    assert adapter.forward(fake_batch) == "standard_path"
