"""WS-B Qwen3.5 smoke harness.

This is the minimum scaffolding needed to make tt_transformers try to load a
Qwen3.5-0.8B model end-to-end on a 1× or 2× Blackhole mesh. The goal is to
SURFACE the first kernel/feature gap so WS-A knows what to attack first; it
is NOT to make the model work.

Reuses the same fabric/mesh helpers as _prefetcher_harness.py but swaps the
model id over to Qwen3.5-0.8B and registers SGLang's Qwen3.5 config schema
with HF AutoConfig before calling tt_transformers.

Usage (inside the p3a-ngram container):
    PYTHONPATH=/sglang/python/sglang/srt/hardware_backend/tenstorrent/test/standalone:\
              /sglang/python:$PYTHONPATH \
    HF_MODEL=/tt-metal/models/weights/Qwen3.5-0.8B \
    python3 -c "from _qwen35_harness import smoke_load_qwen35; smoke_load_qwen35()"
"""

from __future__ import annotations

import os
import sys
import traceback

import ttnn

# Reuse mesh open/close helpers; safe because they have no Qwen3-8B-specific state.
from _prefetcher_harness import close_mesh_device_with_fabric, open_2x_blackhole_mesh


DEFAULT_QWEN35_PATH = "/tt-metal/models/weights/Qwen3.5-0.8B"
MAX_BATCH = 1
MAX_SEQ_LEN = 1024


def _register_sglang_qwen35_config():
    """Make HF AutoConfig aware of model_type='qwen3_5'/'qwen3_5_moe'.

    Side-effect import of sglang.srt.utils.hf_transformers.common runs the
    AutoConfig.register loop (see common.py:137-143). Without this, AutoConfig
    on a Qwen3.5 checkpoint raises 'qwen3_5 not a recognized model_type'.
    """
    # The plugin's __init__.py does not chain to this module, so import explicitly.
    try:
        import sglang.srt.utils.hf_transformers.common  # noqa: F401
    except Exception as exc:
        # Surface the import failure; it's the most likely first blocker.
        print(f"REGISTER_QWEN35_CONFIG_FAIL: {type(exc).__name__}: {exc}", flush=True)
        raise


def open_1x_blackhole_mesh():
    """Single-Blackhole mesh — Qwen3.5-0.8B fits comfortably on one chip in BF16."""
    ttnn.set_fabric_config(
        True,
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    dispatch_core_config = ttnn.DispatchCoreConfig(None, None, ttnn.FabricTensixConfig.DISABLED)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(1, 1),
        dispatch_core_config=dispatch_core_config,
    )


def _resolve_model_path() -> str:
    """HF_MODEL must point at the local Qwen3.5-0.8B checkpoint."""
    path = os.environ.get("HF_MODEL", DEFAULT_QWEN35_PATH)
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"HF_MODEL={path} does not exist. Run "
            f"`huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir {DEFAULT_QWEN35_PATH}` first."
        )
    return path


def _install_qwen35_loader_shims():
    """Monkey-patch tt_transformers to read Qwen3.5 weights from safetensors.

    WS-B-cycle-2 shim: HF transformers 4.55 does NOT ship a Qwen3_5* model
    class (Qwen3.5's config.json declares ``transformers_version: 4.57.0.dev0``),
    so ``ModelArgs.get_hf_model_cls()`` raises
    ``ValueError: Unknown model for config <Qwen3_5Config>`` before any TT op
    is reached. To surface what fails NEXT (inside ``Transformer.__init__``),
    we patch:

      1. ``ModelArgs.get_hf_model_cls`` → returns a stub that yields a tiny
         object with ``.state_dict() = <safetensors dict>``; bypasses HF
         entirely.
      2. ``ModelArgs.is_multimodal`` is left True so the multimodal vision
         convert path still runs (it will likely fail on a Qwen3.5 layer-name
         mismatch — that's the next gap we want to see).

    NOT a permanent fix. WS-A's real solution is either to upgrade transformers
    or write a proper Qwen3.5 weight loader inside tt_transformers.
    """
    from types import SimpleNamespace

    import safetensors.torch as st

    from models.tt_transformers.tt import model_config as _mc

    def _direct_safetensors_loader(self):
        # Mirror ModelArgs.load_state_dict's contract: returns the final
        # post-standardize state_dict. We do JUST enough to surface the next
        # failure inside Transformer.__init__: read raw tensors, then funnel
        # through the existing standardize/convert helpers.
        import glob
        sd = {}
        for p in sorted(glob.glob(os.path.join(self.CKPT_DIR, "model.safetensors*.safetensors"))):
            sd.update(st.load_file(p))
        # Strip the "model." prefix tt_transformers expects to be already gone
        # (standardize_hf_keys does this for normal HF models).
        from models.tt_transformers.tt.load_checkpoints import (
            convert_hf_to_meta,
            standardize_hf_keys,
        )
        # Qwen3.5 nests text params under "model.language_model.*" (multimodal
        # vision-prefix layout); standardize_hf_keys only knows the flat
        # "model.*" layout, so we re-key up front.
        sd = {k.replace("model.language_model.", "model.", 1): v for k, v in sd.items()}
        # Drop the visual + MTP heads — text-only port doesn't need them and
        # they confuse the downstream HF→Meta rename pass.
        sd = {k: v for k, v in sd.items() if not k.startswith("model.visual.") and not k.startswith("mtp.")}
        # Qwen3.5 ties word embeddings, so the safetensors only ship
        # ``model.embed_tokens.weight`` (no ``lm_head.weight``). Synthesize
        # ``lm_head.weight`` from the embedding before standardize_hf_keys
        # deletes the embed key. Without this, ``embedding.py:22`` raises
        # ``KeyError: 'tok_embeddings.weight'`` because standardize collapses
        # both to a single key.
        if "lm_head.weight" not in sd and "model.embed_tokens.weight" in sd:
            sd["lm_head.weight"] = sd["model.embed_tokens.weight"].clone()
        sd = standardize_hf_keys(sd)
        sd = convert_hf_to_meta(sd, self.head_dim, self.n_heads, self.n_kv_heads)
        return sd

    _mc.ModelArgs.load_state_dict = _direct_safetensors_loader  # type: ignore[assignment]

    # Also force is_multimodal=False so the text-only path runs. This is
    # consistent with WS-B's "no vision in first port" scope.
    _orig_set_hf_params = _mc.ModelArgs._set_hf_params

    def _set_hf_params_text_only(self, ckpt_dir):
        _orig_set_hf_params(self, ckpt_dir)
        if self.is_multimodal:
            print(
                "[WS-B-SHIM] forcing is_multimodal=False for Qwen3.5 text-only path",
                flush=True,
            )
            self.is_multimodal = False

    _mc.ModelArgs._set_hf_params = _set_hf_params_text_only  # type: ignore[assignment]


def build_qwen35_model(mesh_device, n_layers=None, max_batch_size=MAX_BATCH, max_seq_len=MAX_SEQ_LEN,
                       install_loader_shims: bool = False):
    """Try to construct a tt_transformers Transformer for Qwen3.5-0.8B.

    Returns (model, model_args) on success. Surface any exception to caller for
    smoke classification. Mirrors build_prefetcher_off_model in
    _prefetcher_harness.py but with Qwen3.5 config registration baked in.

    install_loader_shims=True applies the WS-B-cycle-2 safetensors-direct
    loader so the smoke can probe failures BEYOND the HF-no-Qwen3.5 wall.
    """
    from transformers import AutoConfig

    from models.tt_transformers.tt.generator_sglang import initialize_sglang_text_transformer

    _register_sglang_qwen35_config()

    if install_loader_shims:
        _install_qwen35_loader_shims()

    model_path = _resolve_model_path()
    # Force tt_transformers to derive its model_name from this path.
    os.environ["HF_MODEL"] = model_path

    hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    tt_models, model_args = initialize_sglang_text_transformer(
        hf_config=hf_config,
        tt_data_parallel=1,
        mesh_device=mesh_device,
        max_batch_size=max_batch_size,
        max_seq_len=max_seq_len,
        n_layers=n_layers,
        use_prefetcher=False,  # prefetcher OFF in smoke (per WS-B rule 8)
    )
    return tt_models[0], model_args[0]


def smoke_load_qwen35(use_2x: bool = False, n_layers=None, install_loader_shims: bool = False):
    """End-to-end smoke entry point. Returns 0 on success, 1 on failure.

    Prints a SINGLE marker line at the end:
      MODEL_LOAD_OK           — Transformer constructed
      MODEL_LOAD_FAIL: ...    — first surfaced error (typed)
    The caller (bash one-liner) just greps for these markers.
    """
    mesh = open_2x_blackhole_mesh() if use_2x else open_1x_blackhole_mesh()
    try:
        try:
            model, args = build_qwen35_model(
                mesh, n_layers=n_layers, install_loader_shims=install_loader_shims
            )
            print(
                f"MODEL_LOAD_OK: n_layers={args.n_layers} dim={args.dim} "
                f"n_heads={args.n_heads} n_kv_heads={args.n_kv_heads} "
                f"head_dim={args.head_dim} vocab_size={args.vocab_size}",
                flush=True,
            )
            return 0
        except BaseException as exc:
            tb = traceback.format_exc()
            print(f"MODEL_LOAD_FAIL: {type(exc).__name__}: {exc}", flush=True)
            print("---traceback---", flush=True)
            print(tb, flush=True)
            return 1
    finally:
        try:
            close_mesh_device_with_fabric(mesh)
        except Exception as exc:
            print(f"MESH_CLOSE_WARN: {type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    use_2x = "--2x" in sys.argv
    install_shims = "--shims" in sys.argv
    nlayers_arg = None
    for arg in sys.argv[1:]:
        if arg.startswith("--n-layers="):
            nlayers_arg = int(arg.split("=", 1)[1])
    sys.exit(smoke_load_qwen35(use_2x=use_2x, n_layers=nlayers_arg,
                                install_loader_shims=install_shims))
