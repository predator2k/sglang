"""Shared harness for prefetcher standalone tests.

Used by test_prefetcher_smoke.py, capture_pcc_baseline.py, and
test_prefetcher_pcc.py. The decode call pattern (decode_one_step
below) is templated from tt-metal-sglang's
models/tt_transformers/tt/generator.py (_decode_forward_no_trace_text)
and models/tt_transformers/tests/test_model.py — keep the two in
sync if the upstream test changes.

Phase A.4 of the Option A prefetcher plan.
"""

import glob
import math
import os

import torch
import ttnn
from transformers import AutoConfig

from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator_sglang import (
    allocate_sglang_kv_cache,
    initialize_sglang_text_transformer,
)


MODEL_ID = "Qwen/Qwen3-8B"
MAX_BATCH = 1
MAX_SEQ_LEN = 2048

# Paged KV-cache parameters.  These must match whatever allocate_sglang_kv_cache
# allocates so that paged_update_cache / paged_sdpa can find valid blocks.
BLOCK_SIZE = 32
# num_blocks: enough to cover MAX_SEQ_LEN for every user in the batch.
NUM_KV_BLOCKS = math.ceil(MAX_SEQ_LEN * MAX_BATCH / BLOCK_SIZE)


def open_2x_blackhole_mesh():
    """Open a 2-device Blackhole mesh with Ethernet fabric enabled.

    test_decoder.py uses @pytest.mark.parametrize("device_params",
    [{"fabric_config": True}], indirect=True), which causes conftest.py's
    mesh_device fixture to call ttnn.set_fabric_config(True, ...) before
    ttnn.open_mesh_device. The prefetcher requires the Ethernet fabric
    (TT_FATAL if fabric_context_ is nullptr). We replicate that setup here.
    """
    ttnn.set_fabric_config(
        True,  # FabricConfig.FABRIC_1D — conftest passes the bool directly
        ttnn.FabricReliabilityMode.STRICT_INIT,
        None,  # num_planes
        ttnn.FabricTensixConfig.DISABLED,
        ttnn.FabricUDMMode.DISABLED,
        ttnn.FabricManagerMode.DEFAULT,
    )
    dispatch_core_config = ttnn.DispatchCoreConfig(None, None, ttnn.FabricTensixConfig.DISABLED)
    return ttnn.open_mesh_device(
        ttnn.MeshShape(1, 2),
        dispatch_core_config=dispatch_core_config,
    )


def close_mesh_device_with_fabric(mesh_device):
    """Close the mesh and tear down the fabric (mirrors conftest reset_fabric)."""
    ttnn.close_mesh_device(mesh_device)
    ttnn.set_fabric_config(ttnn.FabricConfig.DISABLED)


def _require_hf_model_env():
    """Prefetcher.__init__ asserts HF_MODEL is set (prefetcher.py:315)."""
    if not os.environ.get("HF_MODEL"):
        # Standalone tests don't go through SGLang's plugin glue, so the
        # env that the server normally sets isn't present. Set it explicitly.
        os.environ["HF_MODEL"] = MODEL_ID


def build_prefetcher_off_model(mesh_device, n_layers=None):
    """Load Qwen3-8B with use_prefetcher=False. Returns (model, model_args)."""
    _require_hf_model_env()
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    tt_models, model_args = initialize_sglang_text_transformer(
        hf_config=hf_config,
        tt_data_parallel=1,
        mesh_device=mesh_device,
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        n_layers=n_layers,
        use_prefetcher=False,
    )
    return tt_models[0], model_args[0]


def build_prefetcher_on_model(mesh_device, n_layers=None):
    """Load Qwen3-8B with use_prefetcher=True. Returns (model, model_args).

    Note: SGLANG_TT_USE_PREFETCHER is read only by tt_llm.py (the SGLang
    server path). initialize_sglang_text_transformer takes use_prefetcher
    as an explicit kwarg, so no env-var dance is needed here.
    """
    _require_hf_model_env()
    hf_config = AutoConfig.from_pretrained(MODEL_ID)
    tt_models, model_args = initialize_sglang_text_transformer(
        hf_config=hf_config,
        tt_data_parallel=1,
        mesh_device=mesh_device,
        max_batch_size=MAX_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        n_layers=n_layers,
        use_prefetcher=True,
    )
    return tt_models[0], model_args[0]


def build_paged_kv_cache(model, model_args):
    """Allocate a paged KV cache on the TT device for standalone decode tests.

    initialize_sglang_text_transformer always uses use_paged_kv_cache=True, so
    layer.attention.layer_past is never set and ttnn_decode_forward requires an
    explicit kv_cache=[k_ttnn, v_ttnn] per layer passed as kv_cache kwarg.

    Returns (kv_cache, page_table_host) where:
      kv_cache — list of [k_ttnn, v_ttnn] per layer, on mesh_device.
      page_table_host — torch.Tensor [MAX_BATCH, num_blocks_per_user] with block IDs.

    NOTE: allocate_sglang_kv_cache caches the KV buffers to disk
    (``empty_{k,v}cache_paged_attention{shape}*.tensorbin``). If those files
    exist from a prior run the tensors are initialised from them rather than
    from torch.zeros, causing non-deterministic outputs when the same device is
    reused across multiple model constructions. We delete these files before
    each allocation so every call starts from a truly-zero KV state.
    """
    # Shape: [num_kv_blocks, n_local_kv_heads, block_size, head_dim]
    # WS-A.10 fix: use the PER-DEVICE count, not the cluster total.
    # ``allocate_sglang_kv_cache`` calls ``ttnn.as_tensor`` with
    # ``ReplicateTensorToMesh``, so the per-device cache shape must already be
    # the local slice. Previously we passed ``model_args.n_kv_heads`` (total),
    # which gave the device extra empty cache slots beyond its local KV heads.
    # ``paged_update_cache`` only writes into ``[0 : n_local_kv_heads)`` slots,
    # so the trailing slots stayed zero — and SDPA's GQA grouping then read
    # those zero slots for the late q-heads, dropping per-device q-heads 2-3
    # to all-zero (the "WS-A.9 head-2/3 zeros" symptom). By using the
    # per-device count, the cache has exactly the slots SDPA needs, all
    # populated by the matching device. ``init_kv_cache`` in
    # ``models/tt_transformers/tt/attention.py`` already uses
    # ``self.n_local_kv_heads`` for the same reason.
    n_local_kv_heads = model_args.n_kv_heads // model_args.num_devices
    head_dim = model_args.head_dim
    kv_cache_shape = (NUM_KV_BLOCKS, n_local_kv_heads, BLOCK_SIZE, head_dim)

    # Wipe any on-disk KV-cache .tensorbin files so the next allocation starts
    # from torch.zeros instead of whatever the previous run wrote.
    _tt_cache_path = model_args.weight_cache_path(ttnn.bfloat8_b)
    for _pat in ("empty_kcache_paged_attention*", "empty_vcache_paged_attention*"):
        for _f in glob.glob(str(_tt_cache_path / _pat)):
            os.remove(_f)

    kv_cache = allocate_sglang_kv_cache(
        kv_cache_shape=kv_cache_shape,
        dtype=torch.bfloat16,
        num_layers=model_args.n_layers,
        dp_model=[model],
        tt_cache_path=_tt_cache_path,
    )
    # allocate_sglang_kv_cache returns kv_cache[dp_idx][layer] = [k_ttnn, v_ttnn]
    # We have dp=1, so kv_cache[0] is the per-layer list.
    kv_cache_single = kv_cache[0]

    # Page table: trivial linear mapping — block i goes to block i.
    # Shape [batch, blocks_per_user]. Each user gets its own contiguous slice.
    blocks_per_user = math.ceil(MAX_SEQ_LEN / BLOCK_SIZE)
    page_table_host = torch.zeros(MAX_BATCH, blocks_per_user, dtype=torch.int32)
    for user_id in range(MAX_BATCH):
        start_block = user_id * blocks_per_user
        page_table_host[user_id] = torch.arange(start_block, start_block + blocks_per_user, dtype=torch.int32)

    return kv_cache_single, page_table_host


def decode_one_step(
    model,
    model_args,
    step: int,
    token_id: int = 42,
    kv_cache=None,
    page_table_host=None,
) -> torch.Tensor:
    """Run one decode step and return logits as a CPU torch.Tensor.

    Decode call pattern discovered from:
    - models/tt_transformers/tt/model.py: prepare_inputs_decode + ttnn_decode_forward
    - models/tt_transformers/tt/generator.py: _decode_forward_no_trace_text
    - models/tt_transformers/tests/test_model.py: generation loop

    Steps:
      (a) Build host token + current_pos tensors (padded to max_batch_size).
      (b) model.prepare_inputs_decode handles tilize/shard onto the mesh.
      (c) model.ttnn_decode_forward runs embedding + transformer forward + all-gather.
      (d) Process the untilized all-gathered logits to return [batch, 1, vocab_size].

    initialize_sglang_text_transformer always sets use_paged_kv_cache=True, so
    layer.attention.layer_past is never populated. The caller MUST supply a
    kv_cache (allocated by build_paged_kv_cache()) and a matching page_table_host.
    Calling without them passes None through to ttnn_decode_forward, which then
    raises AttributeError when the attention layer reaches for layer_past.

    Reused by capture_pcc_baseline.py (prefetcher=False) and
    test_prefetcher_pcc.py (both paths). Works for both use_prefetcher=True/False
    because all dispatch happens inside ttnn_decode_forward.

    Note: model.py's ttnn_decode_forward was patched to move tt_logits to DRAM
    before all_gather_async, fixing a Blackhole P150a issue where L1 INTERLEAVED
    input to all_gather_async returns a zero buffer.
    """
    # Ensure the model is in decode mode (initializes prefetcher sub-devices
    # when use_prefetcher=True; no-op when prefetcher is None).
    model.switch_mode(Mode.DECODE)

    batch_size = model_args.max_batch_size  # == MAX_BATCH == 1

    # (a) Host-side token and position tensors.
    tokens = torch.tensor([token_id] * batch_size)     # shape [batch]
    current_pos = torch.tensor([step] * batch_size)    # shape [batch], 0-indexed

    # (b) Prepare device tensors: tilize tokens, shard current_pos, compute
    #     rope rotation indices.  Pass the page_table_host so prepare_decode
    #     converts it to a device tensor (tt_page_table).
    tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = (
        model.prepare_inputs_decode(tokens, current_pos, page_table_host)
    )

    # (c) Run one decode step: embedding lookup, all transformer layers,
    #     final norm + lm_head, optional all-gather, and untilize.
    #     kv_cache must be passed because initialize_sglang_text_transformer
    #     uses use_paged_kv_cache=True, so attention.layer_past is never set.
    tt_logits, _tt_log_probs = model.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rot_mat_idxs,
        page_table=tt_page_table,
        kv_cache=kv_cache,
    )

    # (d) Bring logits back to CPU host.
    #     tt_logits is the untilized output of all_gather_async.
    #     After all_gather every device has the full vocab; we take device 0.
    #     Shape per device: [1, 1, 32, vocab_size] (row-major after untilize).
    #     User 0 (the only active user in batch=1) appears at the row with
    #     the highest L2 norm among the 32 tile-padded batch rows.
    dev0 = ttnn.get_device_tensors(tt_logits)[0]
    logits_host = ttnn.to_torch(dev0).float()          # [1, 1, 32, vocab_size]
    logits_4d = logits_host[:, :, :, : model_args.vocab_size]  # clip vocab
    row_norms = logits_4d[0, 0].norm(dim=-1)                    # [32]
    best_row = int(row_norms.argmax())
    logits_out = logits_4d[:, :, best_row : best_row + 1, :].squeeze(2)  # [1, 1, V]
    ttnn.deallocate(tt_logits)
    return logits_out.cpu()
