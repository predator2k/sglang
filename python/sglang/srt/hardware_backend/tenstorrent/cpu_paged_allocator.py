# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: © 2026 predator2k <pr3dat0r2000@icloud.com>
"""Pure-CPU paged-KV-pool allocator for the Tenstorrent backend.

SGLang's stock `PagedTokenToKVPoolAllocator.alloc_extend` /
`alloc_decode` are Triton kernels (`alloc_extend_kernel`,
`alloc_decode_kernel` in `mem_cache/allocator.py`). On a TT host
Triton has no active driver (no CUDA / HIP / XPU), so the kernel
launch raises `RuntimeError: 0 active drivers ([])`.

The work the kernels do is pure bookkeeping over integer slot
indices — no actual KV bytes move through this code. KV bytes live
on TT device, owned by `tt_transformers.allocate_sglang_kv_cache`.
A direct port to torch CPU ops is correct, behaviour-equivalent,
and fast enough for the small batches the bookkeeping operates
on (microsecond-level work).

The class subclasses `PagedTokenToKVPoolAllocator` and overrides
only the two Triton call sites. Everything else (`alloc`, `free`,
`clear`, sorting, free-page lifecycle) is inherited unchanged.
"""
from __future__ import annotations

import torch

from sglang.srt.mem_cache.allocator import (
    PagedTokenToKVPoolAllocator,
    get_num_new_pages,
)


class TTCpuPagedTokenToKVPoolAllocator(PagedTokenToKVPoolAllocator):
    def alloc_extend(
        self,
        prefix_lens: torch.Tensor,
        prefix_lens_cpu: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
        extend_num_tokens: int,
        num_new_pages: int = None,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 1) % self.page_size == prefix_lens % self.page_size
            )

        bs = len(prefix_lens)
        page_size = self.page_size

        if self.need_sort and extend_num_tokens // page_size + bs + 1 > len(
            self.free_pages
        ):
            self.merge_and_sort_free()

        out_indices = torch.empty(
            (extend_num_tokens,), dtype=torch.int64, device=self.device
        )

        # Vectorise per-request derived quantities so the inner per-pid loop
        # only needs to scatter index ranges. All inputs are tiny (bs <= 32),
        # so the Python loop overhead is acceptable.
        extend_lens = seq_lens - prefix_lens
        pages_after = (seq_lens + page_size - 1) // page_size
        pages_before = (prefix_lens + page_size - 1) // page_size
        num_new_pages_per_req = pages_after - pages_before

        output_starts = torch.cumsum(extend_lens, dim=0) - extend_lens
        new_page_starts = (
            torch.cumsum(num_new_pages_per_req, dim=0) - num_new_pages_per_req
        )

        # Pull everything to host ints once; the kernel reads each value
        # via `tl.load` (one element per access), so a single materialisation
        # here is strictly fewer device→host syncs than the kernel.
        seq_lens_h = seq_lens.tolist()
        prefix_lens_h = prefix_lens.tolist()
        last_loc_h = last_loc.tolist()
        out_starts_h = output_starts.tolist()
        new_page_starts_h = new_page_starts.tolist()
        n_new_self_h = num_new_pages_per_req.tolist()

        for pid in range(bs):
            seq_len = int(seq_lens_h[pid])
            pre_len = int(prefix_lens_h[pid])
            out_start = int(out_starts_h[pid])
            new_page_start = int(new_page_starts_h[pid])
            n_new_self = int(n_new_self_h[pid])
            ll = int(last_loc_h[pid])

            # Part 1 — fill the partial old page.
            pre_page_end = ((pre_len + page_size - 1) // page_size) * page_size
            num_part1 = min(seq_len, pre_page_end) - pre_len
            if num_part1 > 0:
                out_indices[out_start : out_start + num_part1] = torch.arange(
                    ll + 1,
                    ll + 1 + num_part1,
                    dtype=torch.int64,
                    device=self.device,
                )
            if pre_len + num_part1 == seq_len:
                continue

            # Part 2 — fill the new full pages, one page worth of slots
            # per source free-page entry.
            num_part2 = (seq_len // page_size) * page_size - pre_page_end
            if num_part2 > 0:
                offset = torch.arange(
                    num_part2, dtype=torch.int64, device=self.device
                )
                page_ids = self.free_pages[new_page_start + offset // page_size]
                slots = page_ids * page_size + (offset % page_size)
                out_indices[
                    out_start + num_part1 : out_start + num_part1 + num_part2
                ] = slots
            if pre_len + num_part1 + num_part2 == seq_len:
                continue

            # Part 3 — fill the new partial page (the tail).
            num_part3 = seq_len - (seq_len // page_size) * page_size
            if num_part3 > 0:
                page_start_idx = new_page_start + n_new_self - 1
                page_start = int(self.free_pages[page_start_idx].item())
                out_indices[
                    out_start
                    + num_part1
                    + num_part2 : out_start
                    + num_part1
                    + num_part2
                    + num_part3
                ] = torch.arange(
                    page_start * page_size,
                    page_start * page_size + num_part3,
                    dtype=torch.int64,
                    device=self.device,
                )

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        if num_new_pages is None:
            num_new_pages = get_num_new_pages(
                seq_lens=seq_lens_cpu,
                page_size=page_size,
                prefix_lens=prefix_lens_cpu,
            )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices

    def alloc_decode(
        self,
        seq_lens: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        last_loc: torch.Tensor,
    ):
        if self.debug_mode:
            assert torch.all(
                (last_loc + 2) % self.page_size == seq_lens % self.page_size
            )

        bs = len(seq_lens)
        page_size = self.page_size

        if self.need_sort and bs > len(self.free_pages):
            self.merge_and_sort_free()

        out_indices = torch.empty((bs,), dtype=torch.int64, device=self.device)

        # Vectorise as much as possible; the only branchy work is "did this
        # request cross a page boundary?" — a binary mask.
        pre_lens = seq_lens - 1
        pages_after = (seq_lens + page_size - 1) // page_size
        pages_before = (pre_lens + page_size - 1) // page_size
        num_new_pages_per_req = pages_after - pages_before  # 0 or 1
        new_page_starts = (
            torch.cumsum(num_new_pages_per_req, dim=0) - num_new_pages_per_req
        )

        n_new_h = num_new_pages_per_req.tolist()
        new_starts_h = new_page_starts.tolist()
        last_loc_h = last_loc.tolist()

        for pid in range(bs):
            if int(n_new_h[pid]) == 0:
                out_indices[pid] = int(last_loc_h[pid]) + 1
            else:
                page = int(self.free_pages[int(new_starts_h[pid])].item())
                out_indices[pid] = page * page_size

        if self.debug_mode:
            assert len(torch.unique(out_indices)) == len(out_indices)

        num_new_pages = get_num_new_pages(
            seq_lens=seq_lens_cpu,
            page_size=page_size,
            decode=True,
        )
        if num_new_pages > len(self.free_pages):
            return None

        self.free_pages = self.free_pages[num_new_pages:]
        return out_indices
