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
        last_accept = int(ri[bx, 0].item())
        ai[bx, 0] = last_accept
        num_accept = 0
        cur_index = 0

        for _j in range(1, num_spec):
            cur_index = int(rnt[bx, cur_index].item())
            while cur_index != -1:
                draft_index = int(ri[bx, cur_index].item())
                draft_tok = int(cand[bx, cur_index].item())
                target_tok = int(tp[last_accept].item())
                if draft_tok == target_tok:
                    pr[last_accept] = target_tok
                    num_accept += 1
                    ai[bx, num_accept] = draft_index
                    last_accept = draft_index
                    break
                else:
                    cur_index = int(rns[bx, cur_index].item())
            if cur_index == -1:
                break

        an[bx] = num_accept
        pr[last_accept] = int(tp[last_accept].item())

    accept_index.copy_(ai.to(accept_index.dtype))
    accept_token_num.copy_(an.to(accept_token_num.dtype))
    predicts.copy_(pr.to(predicts.dtype))


def _tt_create_extend_after_decode_spec_info(
    accept_tokens: torch.Tensor,
    seq_lens: torch.Tensor,
    accept_lens: torch.Tensor,
    positions: torch.Tensor,
    bonus_tokens_ptr: torch.Tensor,
    bs_upper: int,
) -> None:
    """Pure-torch port of spec_utils.create_extend_after_decode_spec_info."""
    bs = seq_lens.shape[0]
    for pid in range(bs):
        seq_length = int(seq_lens[pid].item())
        accept_len = int(accept_lens[pid].item())
        accept_len_cumsum = int(accept_lens[:pid].sum().item())

        for i in range(accept_len):
            positions[accept_len_cumsum + i] = seq_length - accept_len + i

        bonus_idx = accept_len_cumsum + accept_len - 1
        bonus_tokens_ptr[pid] = accept_tokens[bonus_idx]


def _tt_assign_req_to_token_pool(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    pool_len: int,
    bs_upper: int,
) -> None:
    """Pure-torch port of spec_utils.assign_req_to_token_pool."""
    bs = req_pool_indices.shape[0]
    for pid in range(bs):
        kv_start = int(start_offset[pid].item())
        kv_end = int(end_offset[pid].item())
        pool_idx = int(req_pool_indices[pid].item())

        out_offset = int((end_offset[:pid] - start_offset[:pid]).sum().item())
        length = kv_end - kv_start
        if length > 0:
            req_to_token[pool_idx, kv_start:kv_end] = (
                out_cache_loc[out_offset : out_offset + length]
            )


def _tt_assign_req_to_token_pool_func(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    start_offset: torch.Tensor,
    end_offset: torch.Tensor,
    out_cache_loc: torch.Tensor,
    batch_size: int,
) -> None:
    """Torch fallback for assign_req_to_token_pool_func wrapper."""
    _tt_assign_req_to_token_pool(
        req_pool_indices,
        req_to_token,
        start_offset,
        end_offset,
        out_cache_loc,
        req_to_token.shape[1],
        batch_size,
    )


def _tt_get_target_cache_loc(
    tgt_cache_loc: torch.Tensor,
    to_free_slots: torch.Tensor,
    num_accepted_drafts: torch.Tensor,
    to_free_num_slots: torch.Tensor,
    out_cache_loc: torch.Tensor,
    num_verify_tokens: int,
    num_verify_tokens_upper: int,
    bs_upper: int,
) -> None:
    """Pure-torch port of spec_utils.get_target_cache_loc."""
    bs = num_accepted_drafts.shape[0]
    for bid in range(bs):
        # Part 1: write accepted cache locs to tgt_cache_loc
        tgt_start = int(num_accepted_drafts[:bid].sum().item()) + bid
        copy_len = int(num_accepted_drafts[bid].item()) + 1
        src_start = bid * num_verify_tokens
        tgt_cache_loc[tgt_start : tgt_start + copy_len] = (
            out_cache_loc[src_start : src_start + copy_len]
        )

        # Part 2: write free slots
        to_free_start = int(to_free_num_slots[:bid].sum().item())
        free_count = int(to_free_num_slots[bid].item())
        if free_count > 0:
            out_start = src_start + num_verify_tokens - free_count
            to_free_slots[to_free_start : to_free_start + free_count] = (
                out_cache_loc[out_start : out_start + free_count]
            )


def _tt_filter_finished_cache_loc_kernel(
    out_cache_loc: torch.Tensor,
    tgt_cache_loc: torch.Tensor,
    num_accepted_drafts: torch.Tensor,
    num_accepted_drafts_filter: torch.Tensor,
    bs_upper: int,
    num_verify_tokens_upper: int,
) -> None:
    """Pure-torch port of spec_utils.filter_finished_cache_loc_kernel."""
    bs = num_accepted_drafts.shape[0]
    for bid in range(bs):
        old_start = int(num_accepted_drafts[:bid].sum().item()) + bid
        new_start = int(num_accepted_drafts_filter[:bid].sum().item())
        copy_len = int(num_accepted_drafts_filter[bid].item())
        if copy_len > 0:
            out_cache_loc[new_start : new_start + copy_len] = (
                tgt_cache_loc[old_start : old_start + copy_len]
            )


def _tt_align_evict_mask_to_page_size(
    seq_lens: torch.Tensor,
    evict_mask: torch.Tensor,
    page_size: int,
    num_draft_tokens: int,
    BLOCK_SIZE: int,
) -> None:
    """Pure-torch port of spec_utils.align_evict_mask_to_page_size."""
    bs = seq_lens.shape[0]
    for bid in range(bs):
        seq_len = int(seq_lens[bid].item())
        mask_row = evict_mask[bid * num_draft_tokens : (bid + 1) * num_draft_tokens]
        num_trues = int(mask_row.sum().item())
        num_false = num_draft_tokens - num_trues
        start = (seq_len + num_false - 1) // page_size * page_size - seq_len
        for i in range(max(start, 0), min(start + page_size, num_draft_tokens)):
            evict_mask[bid * num_draft_tokens + i] = False


def _tt_create_flashinfer_kv_indices(
    req_to_token_ptr: torch.Tensor,
    req_pool_indices_ptr: torch.Tensor,
    page_kernel_lens_ptr: torch.Tensor,
    kv_indptr: torch.Tensor,
    kv_start_idx,
    kv_indices_ptr: torch.Tensor,
    req_to_token_ptr_stride: int,
) -> None:
    """Pure-torch port of create_flashinfer_kv_indices_triton."""
    bs = req_pool_indices_ptr.shape[0]
    for pid in range(bs):
        req_pool_index = int(req_pool_indices_ptr[pid].item())
        kv_indices_offset = int(kv_indptr[pid].item())
        kv_start = 0
        if kv_start_idx is not None:
            kv_start = int(kv_start_idx[pid].item())
        kv_end = kv_start + int(page_kernel_lens_ptr[pid].item())
        length = kv_end - kv_start
        if length > 0:
            row = req_to_token_ptr[req_pool_index]
            kv_indices_ptr[kv_indices_offset : kv_indices_offset + length] = (
                row[kv_start:kv_end]
            )


def _tt_assign_draft_cache_locs(
    req_pool_indices: torch.Tensor,
    req_to_token: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_lens: torch.Tensor,
    num_new_pages_per_topk: torch.Tensor,
    out_cache_loc: torch.Tensor,
    source_cache_loc,
    target_cache_loc,
    last_page_lens_cumsum,
    duplicate_cache_len: int,
    pool_len: int,
    topk: int,
    speculative_num_steps: int,
    page_size: int,
    bs_upper: int,
    iter_upper: int,
) -> None:
    """Pure-torch port of spec_utils.assign_draft_cache_locs Triton kernel.

    Copies newly-allocated cache slot IDs from out_cache_loc into the
    per-request req_to_token pool at positions [seq_len, seq_len + copy_len).
    For topk=1 or page_size=1 (the TT-plugin default), only Part 1 (the
    copy) is needed. Part 2/3 (page duplication for topk>1 with large pages)
    is included for correctness but not performance-critical on TT.
    """
    num_seqs = req_pool_indices.shape[0]

    for pid in range(num_seqs):
        if page_size == 1 or topk == 1:
            copy_len = topk * speculative_num_steps
            out_start = pid * topk * speculative_num_steps
        else:
            copy_len = int(extend_lens[pid].item())
            out_start = int(extend_lens[:pid].sum().item())

        kv_start = int(seq_lens[pid].item())
        pool_idx = int(req_pool_indices[pid].item())

        # Part 1: Copy out_cache_loc entries into req_to_token
        req_to_token[pool_idx, kv_start : kv_start + copy_len] = (
            out_cache_loc[out_start : out_start + copy_len]
        )

        # Part 2 & 3: page duplication (only when page_size > 1 and topk > 1)
        if (page_size != 1 and topk != 1) and duplicate_cache_len > 0:
            prefix_len = kv_start
            last_page_len = prefix_len % page_size
            nnpp = int(num_new_pages_per_topk[pid].item())
            prefix_base_start = prefix_len - last_page_len

            src_indices = req_to_token[
                pool_idx, prefix_base_start : prefix_base_start + last_page_len
            ].clone()

            lplc = int(last_page_lens_cumsum[pid].item())

            # Part 2: fill source/target cache locs
            for topk_id in range(1, topk):
                base_off = (topk - 1) * (lplc - last_page_len) + (topk_id - 1) * last_page_len
                source_cache_loc[base_off : base_off + last_page_len] = src_indices

                tgt_off = prefix_base_start + topk_id * nnpp * page_size
                tgt_indices = req_to_token[
                    pool_idx, tgt_off : tgt_off + last_page_len
                ].clone()
                target_cache_loc[base_off : base_off + last_page_len] = tgt_indices

            # Part 3: shift out_cache_loc for duplication
            ptr_offset = pid * speculative_num_steps * topk
            for topk_id in range(topk):
                src_off = prefix_base_start + topk_id * nnpp * page_size
                for j in range(last_page_len, speculative_num_steps + last_page_len):
                    val = req_to_token[pool_idx, src_off + j]
                    out_cache_loc[
                        ptr_offset + topk_id * speculative_num_steps + j - last_page_len
                    ] = val


def _tt_assign_draft_cache_locs_wrapper(*args, **kwargs):
    """Callable wrapper matching Triton kernel's [(grid,)](...) call convention.

    The Triton JIT kernel is called as ``assign_draft_cache_locs[(num_seqs,)](args...)``.
    This class mimics that: ``wrapper[(grid,)](args...)`` just calls the torch impl.
    """
    return _tt_assign_draft_cache_locs(*args, **kwargs)


class _TritonKernelShim:
    """Shim that accepts Triton's ``kernel[(grid,)](args...)`` call pattern."""

    def __init__(self, fn):
        self._fn = fn

    def __getitem__(self, grid):
        # grid is a tuple like (num_seqs,); ignore it (torch loops over batch)
        return self._fn


def install_tt_eagle_kernels() -> None:
    """Inject pure-torch fallbacks into eagle_utils so EAGLE can run on TT.

    Three injections:
      1. ``sglang.srt.speculative.eagle_utils.sgl_build_tree_kernel_efficient``
         -- referenced as a free name at line ~138 of eagle_utils.py inside the
         else branch (non-NPU). Plain attribute on the module object is
         sufficient; Python resolves the free name from the module's globals
         at call time.
      2. ``sglang.srt.speculative.eagle_utils.verify_tree_greedy_func`` --
         the function itself is monkey-patched because its dispatch is in
         the function body (``if _is_cuda or _is_hip ...``) and there is no
         TT branch.
      3. ``sglang.srt.speculative.spec_utils.assign_draft_cache_locs`` --
         Triton JIT kernel that fails on TT (no CUDA driver). Replaced with
         a torch fallback wrapped in a shim matching the ``kernel[(grid,)](...)``
         call pattern.
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

    # Inject all spec_utils Triton kernel fallbacks. These cover:
    # - assign_draft_cache_locs: called from eagle_worker._draft_preprocess_decode
    # - assign_req_to_token_pool / assign_req_to_token_pool_func: called from eagle_info
    # - create_extend_after_decode_spec_info: called from eagle_info
    # - get_target_cache_loc: called from eagle_info
    # - filter_finished_cache_loc_kernel: called from eagle_info
    #
    # We patch both the spec_utils module (for future imports) and any modules
    # that have already imported these names via ``from spec_utils import ...``.
    try:
        from sglang.srt.speculative import spec_utils

        # assign_draft_cache_locs: Triton kernel called via kernel[(grid,)]()
        shim_assign_draft = _TritonKernelShim(_tt_assign_draft_cache_locs)
        spec_utils.assign_draft_cache_locs = shim_assign_draft

        # assign_req_to_token_pool: Triton kernel + wrapper function
        shim_assign_pool = _TritonKernelShim(_tt_assign_req_to_token_pool)
        spec_utils.assign_req_to_token_pool = shim_assign_pool
        spec_utils.assign_req_to_token_pool_func = _tt_assign_req_to_token_pool_func

        # create_extend_after_decode_spec_info: Triton kernel
        shim_create_ext = _TritonKernelShim(_tt_create_extend_after_decode_spec_info)
        spec_utils.create_extend_after_decode_spec_info = shim_create_ext

        # get_target_cache_loc: Triton kernel
        shim_get_target = _TritonKernelShim(_tt_get_target_cache_loc)
        spec_utils.get_target_cache_loc = shim_get_target

        # filter_finished_cache_loc_kernel: Triton kernel
        shim_filter = _TritonKernelShim(_tt_filter_finished_cache_loc_kernel)
        spec_utils.filter_finished_cache_loc_kernel = shim_filter

        # Patch already-imported bindings in eagle_worker and eagle_info
        for mod_name in ("eagle_worker", "eagle_info"):
            try:
                mod = __import__(
                    f"sglang.srt.speculative.{mod_name}", fromlist=[mod_name]
                )
                if hasattr(mod, "assign_draft_cache_locs"):
                    mod.assign_draft_cache_locs = shim_assign_draft
                if hasattr(mod, "assign_req_to_token_pool"):
                    mod.assign_req_to_token_pool = shim_assign_pool
                if hasattr(mod, "assign_req_to_token_pool_func"):
                    mod.assign_req_to_token_pool_func = _tt_assign_req_to_token_pool_func
                if hasattr(mod, "create_extend_after_decode_spec_info"):
                    mod.create_extend_after_decode_spec_info = shim_create_ext
                if hasattr(mod, "get_target_cache_loc"):
                    mod.get_target_cache_loc = shim_get_target
                if hasattr(mod, "filter_finished_cache_loc_kernel"):
                    mod.filter_finished_cache_loc_kernel = shim_filter
            except Exception:
                pass

        # align_evict_mask_to_page_size: called from eagle_info.verify
        shim_align = _TritonKernelShim(_tt_align_evict_mask_to_page_size)
        spec_utils.align_evict_mask_to_page_size = shim_align

        logger.info(
            "[TT-Plugin] Installed torch fallbacks for all spec_utils Triton kernels"
        )
    except Exception as exc:
        logger.warning(
            f"spec_utils fallback install failed: {exc!r}"
        )

    # Inject create_flashinfer_kv_indices_triton fallback into attention utils
    try:
        from sglang.srt.layers.attention import utils as attn_utils
        shim_kv_indices = _TritonKernelShim(_tt_create_flashinfer_kv_indices)
        attn_utils.create_flashinfer_kv_indices_triton = shim_kv_indices
        logger.info(
            "[TT-Plugin] Installed torch fallback for create_flashinfer_kv_indices_triton"
        )
    except Exception as exc:
        logger.warning(
            f"create_flashinfer_kv_indices fallback failed: {exc!r}"
        )

    # Patch all already-imported bindings in eagle_info
    try:
        from sglang.srt.speculative import eagle_info
        for name, replacement in [
            ("assign_req_to_token_pool_func", _tt_assign_req_to_token_pool_func),
            ("create_extend_after_decode_spec_info", _TritonKernelShim(_tt_create_extend_after_decode_spec_info)),
            ("align_evict_mask_to_page_size", _TritonKernelShim(_tt_align_evict_mask_to_page_size)),
            ("get_target_cache_loc", _TritonKernelShim(_tt_get_target_cache_loc)),
            ("filter_finished_cache_loc_kernel", _TritonKernelShim(_tt_filter_finished_cache_loc_kernel)),
            ("create_flashinfer_kv_indices_triton", _TritonKernelShim(_tt_create_flashinfer_kv_indices)),
        ]:
            if hasattr(eagle_info, name):
                setattr(eagle_info, name, replacement)
        logger.info(
            "[TT-Plugin] Patched eagle_info imported bindings"
        )
    except Exception as exc:
        logger.warning(
            f"eagle_info binding patch failed: {exc!r}"
        )

    eagle_utils._tt_eagle_kernels_installed = True
    logger.info("[TT-Plugin] Installed torch fallbacks for EAGLE tree-build + verify_tree_greedy")
