"""WS-A.11 Phase 1: verify KV-cache contents at decode steps 0 and 1.

Hypothesis under test (highest-ranked from WS-A.10 hand-off):
    paged_update_cache at pos 0 may write only the first replicated K head
    leaving the second slot zero -> SDPA reads zero for q-heads 2/3 ->
    step-0 logits are garbage (PCC 0.03 vs HF) even though steps 1+ recover
    (PCC ~0.70).

Run inside p3a-ngram with HF_MODEL pointing to the local checkpoint.

What this script does:
  1. Build Qwen3.5-0.8B on a 2x Blackhole mesh (load-time KV-replicate on).
  2. Allocate paged KV cache.
  3. Pre-decode K snapshot for layer-3 ("L3 first full-attention layer").
  4. Run ONE decode step at pos=0.
  5. Post-decode K snapshot, compare to pre-decode.
  6. Print, per-device, K[layer3][pos=0][head=0] vs K[layer3][pos=0][head=1]
     (must be IDENTICAL because load-time replicate duplicated rows).
  7. Print K[layer3][pos=1..63] norms (must be 0 -- only pos=0 written).
  8. Run another decode at pos=1, repeat the comparison for both pos=0 and pos=1.
  9. Compare TT cache slot to HF reference at layer 3 pos=0.

Output: prints a structured report, no exceptions thrown if anything is
"wrong" -- we want to see the corruption.
"""
from __future__ import annotations

import os
import pickle
import sys

import torch

sys.path.insert(
    0,
    "/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone",
)


def _norm(t: torch.Tensor) -> float:
    return float(t.float().norm().item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    da = a.norm().clamp(min=1e-12)
    db = b.norm().clamp(min=1e-12)
    return float((a * b).sum() / (da * db))


def _pcc(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.reshape(-1).float()
    b = b.reshape(-1).float()
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    n = min(a.numel(), b.numel())
    a = a[:n]
    b = b[:n]
    a_c = a - a.mean()
    b_c = b - b.mean()
    den = (a_c.norm() * b_c.norm()).clamp(min=1e-12)
    return float((a_c * b_c).sum() / den)


def _read_cache_per_device(kv_tt_layer):
    """Return list-of-(K_dev, V_dev) one tuple per mesh device, on CPU."""
    import ttnn
    k_ttnn, v_ttnn = kv_tt_layer
    out = []
    for dev_idx, k_dev in enumerate(ttnn.get_device_tensors(k_ttnn)):
        v_dev = ttnn.get_device_tensors(v_ttnn)[dev_idx]
        k_t = ttnn.to_torch(k_dev).float().cpu()
        v_t = ttnn.to_torch(v_dev).float().cpu()
        out.append((k_t, v_t))
    return out


def _summarize_cache(label, per_dev_caches, layer_num, max_pos_print=4):
    print(f"\n=== {label} | layer {layer_num} ===", flush=True)
    for dev_idx, (k_t, v_t) in enumerate(per_dev_caches):
        # Shape: [num_blocks, n_local_kv_heads, block_size, head_dim]
        nb, nh, bs, hd = k_t.shape
        print(
            f"[dev{dev_idx}] K.shape={tuple(k_t.shape)} V.shape={tuple(v_t.shape)}",
            flush=True,
        )
        # Look at block 0 (pos 0..block_size-1)
        for pos in range(min(max_pos_print, bs)):
            for h in range(nh):
                k_vec = k_t[0, h, pos, :]
                v_vec = v_t[0, h, pos, :]
                print(
                    f"  block0 head{h} pos{pos}: |K|={_norm(k_vec):8.4f} "
                    f"|V|={_norm(v_vec):8.4f}",
                    flush=True,
                )
        # Whole-block sum for sanity
        block0_K_norm = _norm(k_t[0])
        block0_V_norm = _norm(v_t[0])
        rest_K_norm = _norm(k_t[1:]) if nb > 1 else 0.0
        rest_V_norm = _norm(v_t[1:]) if nb > 1 else 0.0
        print(
            f"  block0 |K|={block0_K_norm:.4f} |V|={block0_V_norm:.4f} | "
            f"blocks1+: |K|={rest_K_norm:.4f} |V|={rest_V_norm:.4f}",
            flush=True,
        )

        # Cross-head check: head 0 vs head 1 at the same position
        if nh >= 2:
            for pos in range(min(max_pos_print, bs)):
                k0 = k_t[0, 0, pos, :]
                k1 = k_t[0, 1, pos, :]
                v0 = v_t[0, 0, pos, :]
                v1 = v_t[0, 1, pos, :]
                if _norm(k0) < 1e-6 and _norm(k1) < 1e-6:
                    note = "(both zero -- no write here)"
                else:
                    diff_K = _norm(k0 - k1)
                    rel_K = diff_K / (_norm(k0) + 1e-12)
                    diff_V = _norm(v0 - v1)
                    rel_V = diff_V / (_norm(v0) + 1e-12)
                    cos_K = _cosine(k0, k1)
                    cos_V = _cosine(v0, v1)
                    note = (
                        f"|K0-K1|={diff_K:.4f} (rel={rel_K:.4f} cos={cos_K:.4f}) "
                        f"|V0-V1|={diff_V:.4f} (rel={rel_V:.4f} cos={cos_V:.4f})"
                    )
                print(f"  HEAD-PAIR-CHECK pos{pos}: {note}", flush=True)


def main() -> int:
    import ttnn
    from _prefetcher_harness import (
        build_paged_kv_cache,
        decode_one_step,
    )
    from _qwen35_harness import (
        build_qwen35_model,
        close_mesh_device_with_fabric,
        open_2x_blackhole_mesh,
    )

    LAYER_PROBE = 3  # first full_attention layer in Qwen3.5-0.8B
    TOKEN_ID = 42

    mesh = open_2x_blackhole_mesh()
    try:
        model, args = build_qwen35_model(mesh, install_loader_shims=True)
        print(
            f"MODEL_LOAD_OK n_layers={args.n_layers} dim={args.dim} "
            f"n_heads={args.n_heads} n_kv_heads={args.n_kv_heads} "
            f"head_dim={args.head_dim} num_devices={args.num_devices}",
            flush=True,
        )
        n_local_kv = args.n_kv_heads // args.num_devices
        print(
            f"n_local_kv_heads={n_local_kv} (expected 2 if KV-replicate factor=2)",
            flush=True,
        )

        kv, page = build_paged_kv_cache(model, args)
        print(
            f"KV cache allocated: {len(kv)} layers; first non-None type={type(kv[LAYER_PROBE])}",
            flush=True,
        )

        # Snapshot BEFORE any decode (should be all zeros).
        pre = _read_cache_per_device(kv[LAYER_PROBE])
        _summarize_cache("PRE-DECODE", pre, LAYER_PROBE)

        # ---- Decode step 0 ----
        print("\n>>> DECODE step=0 token=42", flush=True)
        logits0 = decode_one_step(
            model, args, step=0, token_id=TOKEN_ID,
            kv_cache=kv, page_table_host=page,
        )
        print(
            f"  step0 logits: shape={tuple(logits0.shape)} "
            f"norm={_norm(logits0):.4f} top1={int(logits0.argmax().item())}",
            flush=True,
        )
        post_step0 = _read_cache_per_device(kv[LAYER_PROBE])
        _summarize_cache("POST-STEP0", post_step0, LAYER_PROBE)

        # ---- Decode step 1 ----
        next_tok = int(logits0.argmax().item())
        print(f"\n>>> DECODE step=1 token={next_tok}", flush=True)
        logits1 = decode_one_step(
            model, args, step=1, token_id=next_tok,
            kv_cache=kv, page_table_host=page,
        )
        print(
            f"  step1 logits: shape={tuple(logits1.shape)} "
            f"norm={_norm(logits1):.4f} top1={int(logits1.argmax().item())}",
            flush=True,
        )
        post_step1 = _read_cache_per_device(kv[LAYER_PROBE])
        _summarize_cache("POST-STEP1", post_step1, LAYER_PROBE)

        # ---- Compare against HF layer-3 dump if available ----
        hf_path = "/tmp/qwen35_diag_hf_layer3.pt"
        if os.path.exists(hf_path):
            hf = torch.load(hf_path, map_location="cpu", weights_only=False)
            print(f"\nHF layer-3 dump keys: {sorted(hf.keys())}", flush=True)
            if "09_post_rope_k" in hf:
                hf_rope_k = hf["09_post_rope_k"]
                print(
                    f"HF post_rope_k shape={tuple(hf_rope_k.shape)} "
                    f"|norm|={_norm(hf_rope_k):.4f}",
                    flush=True,
                )
                # HF k is typically [1, n_kv_heads_full, seq=1, head_dim].
                # E.g. shape [1, 2, 1, 256] (2 KV heads, single token).
                if hf_rope_k.ndim == 4 and hf_rope_k.shape[2] == 1:
                    for h in range(min(2, hf_rope_k.shape[1])):
                        kvec = hf_rope_k[0, h, 0, :]
                        print(
                            f"  HF head{h} pos0: |K|={_norm(kvec):.4f}",
                            flush=True,
                        )
                    # Compare HF head0 to TT dev0 head0 pos0 (after-step0).
                    tt_dev0_k_pos0 = post_step0[0][0][0, 0, 0, :]
                    hf_h0 = hf_rope_k[0, 0, 0, :]
                    cos = _cosine(tt_dev0_k_pos0, hf_h0)
                    pcc = _pcc(tt_dev0_k_pos0, hf_h0)
                    print(
                        f"  TT[dev0,head0,pos0] vs HF[head0,pos0]: "
                        f"cosine={cos:.4f} pcc={pcc:.4f}",
                        flush=True,
                    )

        # ---- HF reference logits comparison for step 0 ----
        hf_ref_path = "/tmp/qwen35_hf_ref_logits.pkl"
        if os.path.exists(hf_ref_path):
            with open(hf_ref_path, "rb") as f:
                ref = pickle.load(f)
            for s, ll in [(0, logits0), (1, logits1)]:
                if s in ref["logits"]:
                    p = _pcc(ref["logits"][s], ll[0, 0])
                    c = _cosine(ref["logits"][s], ll[0, 0])
                    print(
                        f"END-TO-END step={s}: TT vs HF PCC={p:.4f} cos={c:.4f} "
                        f"TT_top1={int(ll[0, 0].argmax().item())} "
                        f"HF_top1={int(ref['logits'][s].argmax().item())}",
                        flush=True,
                    )

        # WS-A.11 ADDITIONAL: read post_v_proj for BOTH devices, compare to HF V heads 0 & 1.
        # Easiest: read the cache contents to confirm each device's V at pos 0.
        hf2 = "/tmp/qwen35_diag_hf_layer3.pt"
        if os.path.exists(hf2):
            hf_dump = torch.load(hf2, map_location="cpu", weights_only=False)
            hf_v_full = hf_dump.get("05_post_v_proj")  # [1, 1, 512] = 2 heads*256
            if hf_v_full is not None:
                hf_v0 = hf_v_full[0, 0, :256].float()
                hf_v1 = hf_v_full[0, 0, 256:].float()
                print(
                    f"\nHF V heads: |v0|={hf_v0.norm():.4f} |v1|={hf_v1.norm():.4f} "
                    f"PCC_v0_v1={_pcc(hf_v0, hf_v1):.4f}",
                    flush=True,
                )
                for dev_idx, (k_t, v_t) in enumerate(post_step0):
                    print(f"\n=== Device {dev_idx} V at pos=0, head=0/1 vs HF V heads 0/1 ===", flush=True)
                    for h in range(min(2, v_t.shape[1])):
                        tt_v = v_t[0, h, 0, :].float()  # [256]
                        print(
                            f"  dev{dev_idx} h{h} pos0: |v|={tt_v.norm():.4f}  "
                            f"PCC_vs_HF_v0={_pcc(tt_v, hf_v0):.4f}  "
                            f"PCC_vs_HF_v1={_pcc(tt_v, hf_v1):.4f}",
                            flush=True,
                        )

        # Save snapshots for later analysis
        save_path = "/tmp/qwen35_ws_a11_cache_snapshot.pt"
        torch.save(
            {
                "pre": pre,
                "post_step0": post_step0,
                "post_step1": post_step1,
                "n_local_kv_heads": n_local_kv,
                "num_devices": args.num_devices,
                "layer_num": LAYER_PROBE,
            },
            save_path,
        )
        print(f"\nCache snapshots saved to {save_path}", flush=True)

    finally:
        try:
            close_mesh_device_with_fabric(mesh)
        except Exception as exc:
            print(f"MESH_CLOSE_WARN: {type(exc).__name__}: {exc}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
