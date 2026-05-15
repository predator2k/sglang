# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""Pure-torch fallbacks for sgl_kernel's EAGLE C++ ops.

SGLang gates two EAGLE primitives behind ``_is_cuda or _is_hip or _is_musa``:

1. ``sgl_build_tree_kernel_efficient`` — builds tree_mask, positions, and the
   retrieve_index/retrieve_next_token/retrieve_next_sibling arrays from the
   draft model's per-step parent_list/top_scores_index/seq_lens.
2. ``verify_tree_greedy`` — walks the verification tree against the target's
   greedy predictions and writes predicts/accept_index/accept_token_num.

Both are not present on a TT-only image (no CUDA/HIP/MUSA wheels installed).
The eagle_utils.build_tree_kernel_efficient wrapper still references the
free name ``sgl_build_tree_kernel_efficient`` at line 138; without an
injection, calling it raises NameError. Likewise verify_tree_greedy_func
silently no-ops, producing zero-accept results.

The functions below mirror the CUDA kernel semantics exactly (validated
against test/registered/spec/utils/test_build_eagle_tree.py expected
output). They are wired into eagle_utils' module namespace at TT plugin
worker init time via ``install_tt_eagle_kernels()``.
"""
from __future__ import annotations

import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def _tt_build_tree_kernel_efficient(
    parent_list: torch.Tensor,
    selected_index: torch.Tensor,
    verified_seq_len: torch.Tensor,
    tree_mask: torch.Tensor,
    positions: torch.Tensor,
    retrieve_index: torch.Tensor,
    retrieve_next_token: torch.Tensor,
    retrieve_next_sibling: torch.Tensor,
    topk: int,
    depth: int,
    draft_token_num: int,
    tree_mask_mode: int,
) -> None:
    """Pure-torch port of csrc/speculative/eagle_utils.cu::build_tree_efficient.

    All output tensors are written in place. ``tree_mask`` semantics:
      FULL_MASK (0): flat [sum(seq_len)*dtn + bs*dtn*dtn] bool tensor where
        for each bid a slab of (seq_len[bid]+dtn)*dtn cells encodes
        attention from each draft row to (prompt_token | bonus | drafts).
      QLEN_ONLY (1)/QLEN_ONLY_BITPACKING (2): not used by TT plugin yet.
    """
    bs = int(verified_seq_len.numel())
    pl = parent_list.to(torch.int64).cpu()
    si = selected_index.to(torch.int64).cpu()
    vsl = verified_seq_len.to(torch.int64).cpu().tolist()

    flat_mask = tree_mask.view(-1) if tree_mask_mode == 0 else None
    pos = positions.view(-1)
    ri = retrieve_index.view(bs, draft_token_num)
    rnt = retrieve_next_token.view(bs, draft_token_num)
    rns = retrieve_next_sibling.view(bs, draft_token_num)

    # Initialize index buffers to -1 (same as eagle_utils.py:108 retrieve_buf).
    ri.fill_(-1)
    rnt.fill_(-1)
    rns.fill_(-1)

    pl_stride = topk * (depth - 1) + 1

    seq_tree_off = [0] * bs
    if tree_mask_mode == 0:
        running = bs * draft_token_num * draft_token_num
        # eagle_utils.cu:53 uses dtn*dtn*bid + sum_{i<bid}(seq_len[i] * dtn)
        # Pre-compute offsets per bid.
        for bid in range(bs):
            off = bid * draft_token_num * draft_token_num
            for i in range(bid):
                off += vsl[i] * draft_token_num
            seq_tree_off[bid] = off

    for bid in range(bs):
        seq_len = vsl[bid]

        # ---- tree_mask write phase (FULL_MASK only) ----
        if tree_mask_mode == 0:
            base = seq_tree_off[bid]
            for tid in range(draft_token_num):
                token_tree_idx = base + (seq_len + draft_token_num) * tid + seq_len + 1
                flat_mask[token_tree_idx - 1] = True
                for i in range(draft_token_num - 1):
                    flat_mask[token_tree_idx + i] = False
                if tid > 0:
                    cur_position = tid - 1
                    while True:
                        flat_mask[token_tree_idx + cur_position] = True
                        parent_tb_idx = int(si[bid, cur_position].item()) // topk
                        if parent_tb_idx == 0:
                            break
                        token_idx = int(pl[bid, parent_tb_idx].item())
                        nxt = draft_token_num  # sentinel: not found
                        for cp in range(draft_token_num):
                            if int(si[bid, cp].item()) == token_idx:
                                nxt = cp
                                break
                        if nxt == draft_token_num:
                            break  # invalid tree
                        cur_position = nxt

        # ---- positions phase ----
        pos[bid * draft_token_num] = seq_len
        for tid in range(1, draft_token_num):
            cur_position = tid - 1
            position = 0
            while True:
                position += 1
                parent_tb_idx = int(si[bid, cur_position].item()) // topk
                if parent_tb_idx == 0:
                    break
                token_idx = int(pl[bid, parent_tb_idx].item())
                nxt = draft_token_num
                for cp in range(draft_token_num):
                    if int(si[bid, cp].item()) == token_idx:
                        nxt = cp
                        break
                if nxt == draft_token_num:
                    break
                cur_position = nxt
            pos[bid * draft_token_num + tid] = position + seq_len

        # ---- retrieve_index / retrieve_next_token / retrieve_next_sibling ----
        ri[bid, 0] = bid * draft_token_num
        for i in range(draft_token_num - 1, 0, -1):
            current_token_idx = bid * draft_token_num + i
            ri[bid, i] = current_token_idx
            parent_tb_idx = int(si[bid, i - 1].item()) // topk
            parent_position = 0
            if parent_tb_idx > 0:
                parent_token_idx = int(pl[bid, parent_tb_idx].item())
                found = False
                for cp in range(draft_token_num):
                    if int(si[bid, cp].item()) == parent_token_idx:
                        parent_position = cp + 1
                        found = True
                        break
                if not found:
                    parent_position = draft_token_num
            if parent_position == draft_token_num:
                continue  # invalid tree, skip
            if int(rnt[bid, parent_position].item()) == -1:
                rnt[bid, parent_position] = i
            else:
                origin = int(rnt[bid, parent_position].item())
                rnt[bid, parent_position] = i
                rns[bid, i] = origin


def _tt_verify_tree_greedy(
    predicts: torch.Tensor,
    accept_index: torch.Tensor,
    accept_token_num: torch.Tensor,
    candidates: torch.Tensor,
    retrive_index: torch.Tensor,
    retrive_next_token: torch.Tensor,
    retrive_next_sibling: torch.Tensor,
    target_predict: torch.Tensor,
) -> None:
    """Pure-torch port of csrc/speculative/eagle_utils.cu::VerifyTreeGreedy.

    Walks the tree top-down. For each level, follow ``retrive_next_token`` to
    the first child; while the child's draft token equals the parent's
    target prediction, accept it and descend. Otherwise hop to a sibling.
    """
    batch_size = int(candidates.size(0))
    num_spec = int(accept_index.size(1))
    num_draft = int(candidates.size(1))

    cand = candidates.cpu()
    ri = retrive_index.cpu()
    rnt = retrive_next_token.cpu()
    rns = retrive_next_sibling.cpu()
    # target_predict is [bs, num_draft]; CUDA kernel treats it as flat
    # (bs*num_draft) indexed by retrive_index values (which are global indices
    # in [bx*num_draft, (bx+1)*num_draft-1]). Flatten for matching semantics.
    tp = target_predict.reshape(-1).cpu()
    ai = accept_index.cpu().clone()
    an = accept_token_num.cpu().clone()
    # predicts is 1D [tot_num_draft_tokens] per the C++ comment — already flat.
    pr = predicts.reshape(-1).cpu().clone()

    for bx in range(batch_size):
        last_accepted = int(ri[bx, 0].item())
        ai[bx, 0] = last_accepted
        num_accepted = 0
        cur_index = 0

        for _j in range(1, num_spec):
            cur_index = int(rnt[bx, cur_index].item())
            while cur_index != -1:
                draft_index = int(ri[bx, cur_index].item())
                draft_tok = int(cand[bx, cur_index].item())
                target_tok = int(tp[last_accepted].item())
                if draft_tok == target_tok:
                    pr[last_accepted] = target_tok
                    num_accepted += 1
                    ai[bx, num_accepted] = draft_index
                    last_accepted = draft_index
                    break
                else:
                    cur_index = int(rns[bx, cur_index].item())
            if cur_index == -1:
                break

        an[bx] = num_accepted
        pr[last_accepted] = int(tp[last_accepted].item())

    accept_index.copy_(ai.to(accept_index.dtype))
    accept_token_num.copy_(an.to(accept_token_num.dtype))
    predicts.copy_(pr.to(predicts.dtype))


def install_tt_eagle_kernels() -> None:
    """Inject pure-torch fallbacks into eagle_utils so EAGLE can run on TT.

    Two injections:
      1. ``sglang.srt.speculative.eagle_utils.sgl_build_tree_kernel_efficient``
         — referenced as a free name at line ~138 of eagle_utils.py inside the
         else branch (non-NPU). Plain attribute on the module object is
         sufficient; Python resolves the free name from the module's globals
         at call time.
      2. ``sglang.srt.speculative.eagle_utils.verify_tree_greedy_func`` —
         the function itself is monkey-patched because its dispatch is in
         the function body (``if _is_cuda or _is_hip ...``) and there is no
         TT branch.
    """
    try:
        from sglang.srt.speculative import eagle_utils
    except Exception as exc:
        logger.warning(f"eagle_utils import failed; skip fallback install: {exc!r}")
        return

    if getattr(eagle_utils, "_tt_eagle_kernels_installed", False):
        return

    eagle_utils.sgl_build_tree_kernel_efficient = _tt_build_tree_kernel_efficient

    original_verify = eagle_utils.verify_tree_greedy_func

    def patched_verify(
        predicts,
        accept_index,
        accept_token_num,
        candidates,
        retrieve_index,
        retrieve_next_token,
        retrieve_next_sibling,
        target_predict,
        topk: int = -1,
    ):
        if eagle_utils._is_cuda or eagle_utils._is_hip or eagle_utils._is_musa or eagle_utils._is_npu:
            return original_verify(
                predicts,
                accept_index,
                accept_token_num,
                candidates,
                retrieve_index,
                retrieve_next_token,
                retrieve_next_sibling,
                target_predict,
                topk=topk,
            )
        _tt_verify_tree_greedy(
            predicts,
            accept_index,
            accept_token_num,
            candidates,
            retrieve_index,
            retrieve_next_token,
            retrieve_next_sibling,
            target_predict,
        )
        return predicts, accept_index, accept_token_num

    eagle_utils.verify_tree_greedy_func = patched_verify
    eagle_utils._tt_eagle_kernels_installed = True
    logger.info("[TT-Plugin] ✓ Installed torch fallbacks for EAGLE tree-build + verify_tree_greedy")
