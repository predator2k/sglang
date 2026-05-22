"""WS-A.10 per-op layer-3 PCC probe.

Re-runs the WS-A.7 layer-3 divergence harness with load-time KV-head
replicate enabled by default (factor=2 via _set_model_specific_params).
Compares TT per-op dumps to the HF reference at /tmp/qwen35_diag_hf_layer3.pt
using the same slicing logic as the WS-A.7 ``compare`` function (each TT
device-0 dump is sliced against the matching slice of the HF reference).
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch


HUNT_PATH = "/tmp/qwen35_layer3_divergence_hunt.py"
HF_DUMP = "/tmp/qwen35_diag_hf_layer3.pt"
TT_DUMP = "/tmp/qwen35_diag_tt_wsa10.pt"


def pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() != b.numel():
        n = min(a.numel(), b.numel())
        a = a[:n]
        b = b[:n]
    a_c = a - a.mean()
    b_c = b - b.mean()
    den = (a_c.norm() * b_c.norm()).clamp(min=1e-8)
    return float((a_c * b_c).sum() / den)


def _row0_slice(t: torch.Tensor) -> torch.Tensor:
    if t.ndim == 4 and t.shape[2] == 32:
        return t[:, :, :1, :]
    return t


def main() -> int:
    os.environ["SGLANG_TT_DUMP_LAYER3"] = "1"
    os.environ["SGLANG_TT_DUMP_LAYER3_PATH"] = TT_DUMP
    if os.path.exists(TT_DUMP):
        os.remove(TT_DUMP)

    sys.path.insert(
        0,
        "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone",
    )
    spec = importlib.util.spec_from_file_location("hunt", HUNT_PATH)
    hunt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hunt)
    dump = hunt.run_tt(install_loader_shims=True, force_no_permute=False)

    hf = torch.load(HF_DUMP, map_location="cpu", weights_only=False)
    print("HF_KEYS:", sorted(k for k in hf.keys() if not k.startswith("meta")))
    print("DUMP_KEYS:", sorted(dump.keys()))

    # (hf_key, tt_key, label, hf_slice, tt_slice). For load-time KV-replicate,
    # TT device-0 K/V/Knorm/RopeK shape is [1,1,2,256] (both replicas of head
    # 0). HF k/v has 2 heads [h0,h1]. Compare TT head 0 (or any replica) vs
    # HF head 0.
    pairs = [
        ("00_layer3_input", "00_layer3_input", "layer3_input", None, None),
        ("01_post_input_layernorm", "01_post_input_layernorm", "post_input_layernorm", None, None),
        ("03a_query_states_pre_chunk", "03a_q_only", "q_after_split", lambda t: t[:, :, :4, :], None),
        ("03b_gate_chunk", "03b_gate_only", "gate_after_split", lambda t: t[:, :, :4, :], None),
        # HF k_proj: [1,1,512]=[1,1,2*256]. Slice first head: [..., :256].
        # TT post-replicate: [1,1,2,256] (per-device). Slice head 0: t[:, :, :1, :]
        ("04_post_k_proj", "04_post_k_proj", "post_k_proj", lambda t: t[..., :256], lambda t: t[:, :, :1, :]),
        ("05_post_v_proj", "05_post_v_proj", "post_v_proj", lambda t: t[..., :256], lambda t: t[:, :, :1, :]),
        ("06_post_q_norm", "06_post_q_norm", "post_q_norm", lambda t: t[:, :, :4, :], None),
        ("07_post_k_norm", "07_post_k_norm", "post_k_norm", lambda t: t[:, :, :1, :], lambda t: t[:, :, :1, :]),
        ("08_post_rope_q", "08_post_rope_q", "post_rope_q", lambda t: t[0, :4, :, :].unsqueeze(0), None),
        ("09_post_rope_k", "09_post_rope_k", "post_rope_k", lambda t: t[0, :1, :, :].unsqueeze(0), lambda t: t[:, :, :1, :]),
        # post sdpa: HF reshaped [1,1,2048]=[1,1,8*256]. TT per-device
        # [1,1,32,1024]=[1,1,32,n_local_heads*head_dim]=[1,1,32,4*256]. Take
        # row 0, first half (q-heads 0-3 = HF heads 0-3).
        ("10b_post_sdpa_reshaped", "10_post_sdpa", "post_sdpa_concat", lambda t: t[..., :1024], lambda t: t[:, :, :1, :]),
        ("11b_sigmoid_gate", "11_sigmoid_gate", "sigmoid_of_gate", lambda t: t[..., :1024], None),
        ("11_post_sigmoid_gate_mul", "11b_post_gate_mul", "post_gate_mul", lambda t: t[..., :1024], lambda t: t[:, :, :1, :]),
        ("12_post_o_proj", "12_post_o_proj", "post_o_proj", lambda t: t[..., :512], lambda t: t[:, :, :1, :]),
        ("13_layer3_output", "13_layer3_output", "layer3_output_with_residual", lambda t: t[..., :512], lambda t: t[:, :, :1, :]),
    ]

    print()
    print(f"{'label':<32s}  {'tt':<22s}  {'hf':<22s}  {'PCC':>8s}")
    for hk, tk, label, hf_sl, tt_sl in pairs:
        if hk not in hf or tk not in dump:
            print(f"{label:<32s}  MISSING tt={tk in dump} hf={hk in hf}")
            continue
        hf_t = hf[hk]
        tt_t = dump[tk]
        # TT-side: row0 slice first if [1,1,32,H], then user tt_sl if given.
        tt_use = _row0_slice(tt_t) if tt_t.ndim == 4 and tt_t.shape[2] == 32 else tt_t
        if tt_sl is not None:
            try:
                tt_use = tt_sl(tt_t)
            except Exception:
                pass
        hf_use = hf_t if hf_sl is None else hf_sl(hf_t)
        try:
            p = pcc(tt_use, hf_use)
        except Exception as e:
            print(f"{label:<32s}  ERROR {e}")
            continue
        print(f"{label:<32s}  {str(tuple(tt_use.shape)):<22s}  {str(tuple(hf_use.shape)):<22s}  {p:8.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
