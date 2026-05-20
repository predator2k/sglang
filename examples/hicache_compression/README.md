# `compressed_file` HiCache backend — demo

End-to-end recipes for exercising the lossless KV-cache compression backend in
two SGLang deployment modes:

1. **Single server with hierarchical cache** — GPU → host RAM → disk, the disk
   tier going through `compressed_file`.
2. **PD disaggregation** — separate prefill / decode workers, decode-side KV
   offload going through `compressed_file`.

The backend lives at
`python/sglang/srt/mem_cache/storage/compressed/`; see the module docstrings
for the codec / layout / policy schema.

## Prerequisites

```bash
pip install zstandard lz4 blosc2 isal pyyaml
# Optional, for the GPU-decode codec:
# pip install nvidia-nvcomp-cu12

# 1× GPU (Ampere or newer recommended). Llama-3.1-8B BF16 needs ~17 GB device.
# At least 32 GB host RAM for the L2 host pool.
huggingface-cli login  # if the model is gated
```

## Mode 1: Single-server hierarchical cache

```bash
./launch_single_server.sh
# (in another terminal)
python client_bench.py --host localhost --port 30000 \
    --num-prompts 32 \
    --cache-dir /tmp/sglang_hicache_compressed_demo
```

What you should see:

- Cold pass walks each unique prompt through prefill, evicts shared-prefix
  pages to disk via `compressed_file` (set path).
- Hot pass re-uses the prefix; pages are fetched back from disk
  (get path). On a healthy build the hot mean TTFT should drop noticeably
  vs cold.
- The cache directory size is the **compressed** footprint — expect ≈1.5–1.6×
  smaller than the equivalent `file` backend would produce (this is what
  our prototype's offline benchmark reports too).

Where to inspect:

- `ls -la /tmp/sglang_hicache_compressed_demo/` — one `.bin` per page key.
- Each `.bin` starts with the magic bytes `SGLKVCMP`; decode the 32-byte page
  header to verify the version + num_slabs.
- `--hicache-storage-backend-extra-config '{"profiles_yaml": "policy.yaml"}'`
  swaps in the per-layer policy described in `policy.yaml`.

## Mode 2: PD disaggregation

Three terminals:

```bash
# Term A — prefill worker (uses the GPU for prompt processing)
./launch_pd_prefill.sh    # listens on :30000, bootstrap :30001

# Term B — decode worker (uses our backend for KV offload)
./launch_pd_decode.sh     # listens on :30100

# Term C — client (PD requests are sent to the prefill side)
python client_bench.py --host localhost --port 30000 --num-prompts 32
```

What this exercises:

- `DecodeKVCacheOffloadManager` is constructed because we passed
  `--disaggregation-decode-enable-offload-kvcache`. It demands a storage
  backend (`compressed_file`).
- Once decode progresses past the prefill prefix, the prefix-aligned KV pages
  are flushed to disk via the backend's set path. Subsequent requests hitting
  the same prefix re-load from disk via the get path.

## Knobs to play with

Edit `policy.yaml` for codec/layout choice, then redeploy. The fields you'll
care about:

- `default: <profile_name>` — the global fallback profile.
- `profiles.fast.codec: blosc2` vs `lz4` — pick speed vs ratio knee.
- `rules: - match: { layers: [0], kv: V }` — layer-0 V outlier handling
  (4–5× ratio on raw zstd alone).
- `runtime.batch_threads` / `runtime.slab_threads` — Python and intra-page
  parallelism. Bump to (cpu_count/2) for production.

Pass a calibrated YAML from `scripts/calibrate.py` (in `kvcache_comp/`) to
get a policy tuned to your actual KV distribution.

## Tier-aware policy (advisory, not yet wired)

`policy.yaml` already contains rules keyed on `tier: l2_to_l3` and
`tier: l1_to_l2`. They're *inert* until a follow-up SGLang patch teaches
`HiCacheController` to populate `HiCacheStorageExtraInfo.extra_info["tier"]`
on the per-page set/get calls. Once that exists, the backend will read it via
`_extra_to_dict()` and route through the matching rule. The schema and the
backend side are both implemented — see
`storage/compressed/policy.py::_Rule.matches_extra` and
`storage/compressed/hicache_compressed.py::_extra_to_dict`.
