"""Shared harness for prefetcher standalone tests.

Used by test_prefetcher_smoke.py, capture_pcc_baseline.py, and
test_prefetcher_pcc.py. The decode call pattern (decode_one_step
below) is templated from tt-metal-sglang's
models/tt_transformers/tt/generator.py (_decode_forward_no_trace_text)
and models/tt_transformers/tests/test_model.py — keep the two in
sync if the upstream test changes.

Phase A.4 of the Option A prefetcher plan.
"""

import os

import torch
import ttnn
from transformers import AutoConfig

from models.tt_transformers.tt.common import Mode
from models.tt_transformers.tt.generator_sglang import (
    initialize_sglang_text_transformer,
)


MODEL_ID = "Qwen/Qwen3-8B"
MAX_BATCH = 1
MAX_SEQ_LEN = 2048


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


def decode_one_step(model, model_args, step: int, token_id: int = 42) -> torch.Tensor:
    """Run one decode step and return logits as a CPU torch.Tensor.

    Decode call pattern discovered from:
    - models/tt_transformers/tt/model.py: prepare_inputs_decode + ttnn_decode_forward
    - models/tt_transformers/tt/generator.py: _decode_forward_no_trace_text
    - models/tt_transformers/tests/test_model.py: generation loop

    Steps:
      (a) Build host token + current_pos tensors (padded to max_batch_size).
      (b) model.prepare_inputs_decode handles tilize/shard onto the mesh.
      (c) model.ttnn_decode_forward runs embedding + transformer forward + all-gather.
      (d) ttnn.to_torch + reshape yields [batch, 1, vocab_size] CPU logits.

    Reused by capture_pcc_baseline.py (prefetcher=False) and
    test_prefetcher_pcc.py (both paths). Works for both use_prefetcher=True/False
    because all dispatch happens inside ttnn_decode_forward.
    """
    # Ensure the model is in decode mode (initializes prefetcher sub-devices
    # when use_prefetcher=True; no-op when prefetcher is None).
    model.switch_mode(Mode.DECODE)

    batch_size = model_args.max_batch_size  # == MAX_BATCH == 1

    # (a) Host-side token and position tensors.
    tokens = torch.tensor([token_id] * batch_size)     # shape [batch]
    current_pos = torch.tensor([step] * batch_size)    # shape [batch], 0-indexed

    # (b) Prepare device tensors: tilize tokens, shard current_pos, compute
    #     rope rotation indices. page_table=None uses linear (non-paged) KV.
    #     prepare_inputs_decode takes *inputs positionally (delegates to
    #     prepare_decode_inputs_host(tokens, current_pos, page_table)).
    tt_tokens, tt_current_pos, tt_rot_mat_idxs, tt_page_table = (
        model.prepare_inputs_decode(tokens, current_pos, None)
    )

    # (c) Run one decode step: embedding lookup, all transformer layers,
    #     final norm + lm_head, optional all-gather, and untilize.
    tt_logits, _tt_log_probs = model.ttnn_decode_forward(
        tt_tokens,
        tt_current_pos,
        rot_mat_idxs=tt_rot_mat_idxs,
        page_table=tt_page_table,
    )

    # (d) Bring logits back to CPU host.
    #     ConcatMesh2dToTensor on dims=(1, -1) is the non-galaxy 2-device
    #     reduction used in test_model.py's decode loop.
    mesh_composer = ttnn.ConcatMesh2dToTensor(
        model.mesh_device,
        dims=(1, -1),
        mesh_shape=model_args.cluster_shape,
    )
    logits_torch = (
        ttnn.to_torch(tt_logits, mesh_composer=mesh_composer)
        .permute(2, 1, 0, 3)
        .squeeze(2)[: model_args.max_batch_size, 0:1, : model_args.vocab_size]
    )
    ttnn.deallocate(tt_logits)
    return logits_torch.cpu()
