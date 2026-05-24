# TT Qwen3-8B prefetcher — Attack B (S9 NOC posted-write flush) RULED OUT — 2026-05-23

Status: **EVIDENCE_ADVANCE — S9 (producer NOC posted-writes flush
insufficient) RULED OUT. The S9 surgical kernel-side fix
(`writer_l1.cpp`: force `skip_ptr_update=false` so producer's data
writes and semaphore-inc both become non-posted, and `noc_async_posted_writes_flushed()` →
`noc_async_writes_flushed()`) was built, JIT-recompiled, and exercised
on a full prefetcher decode. Result: GSM8K(10) chat-format = 0/10 with
**identical** NaN sampling crash signature as the unmodified canonical
prefetcher (`RuntimeError: probability tensor contains either inf, nan or
element < 0` on first /generate). Fix reverted. Canonical (no
prefetcher) re-verified post-revert. Tt-metal-sglang HEAD unchanged at
`c1e3437b78e`; working tree clean. S10 (sub-device CB-address misalignment)
is now the last standing prefetcher-side major suspect.**

Continuation of `tt_qwen3_8b_prefetcher_phase_b8_2026-05-23.md`
(S8 ruled out) and `tt_qwen3_8b_prefetcher_path_a_b4ext_2026-05-23.md`
(S6/S7 ruled out, bytes match L0/L1/L17/L35 at receiver). Executes the
"Path forward — B. NOC flush / sub-device probe (in tt-metal C++)"
recommendation from the phase_b8 doc.

## Bottom line

| Test | Config | GSM8K(10) chat | Signature | Verdict |
|---|---|---|---|---|
| **S9 fix** | `SGLANG_TT_USE_PREFETCHER=1` + writer_l1.cpp non-posted regime | **0/10** | `<ERROR RemoteDisconnected>` on Q1; scheduler dies with `probability tensor contains either inf, nan or element < 0` on first decode step (post chat-detect, post Prefetcher Initialization) | **S9 RULED OUT — same NaN crash as canonical prefetcher per `tt_qwen3_8b_prefetcher_correctness_2026-05-23.md`** |
| Canonical re-verify | no prefetcher envs | **10/10 = 100%** (`/tmp/qwen3_8b_canonical_reverify.log`) | clean reasoning chain; all 10 GT correct (gt={64,260,160,45,460,366,694,13,18,60} = predicted) | unchanged baseline, ≥ 9/10 requirement met |

## S9 hypothesis recap

Per `tt_qwen3_8b_prefetcher_phase_b_2026-05-23.md` lines 155-162:

> **S9 — Producer-side `noc_async_posted_writes_flushed` is insufficient
> for the matmul's compute-kernel-side cb_in1 `get_read_ptr` semantics**.
> The producer uses POSTED writes (`skip_ptr_update=true`) for perf. If
> the matmul reads c_31 before the producer's writes have completed, it
> sees stale data. But `remote_cb_wait_front` should block on the
> semaphore count incremented by the writer.

The race: even with the per-block `noc_async_posted_writes_flushed()`
call at writer_l1.cpp:65, posted writes return on departure (not
completion). The semaphore-inc — which the consumer's
`remote_cb_wait_front` blocks on — uses `noc_semaphore_inc<posted=true>`
(also posted) through a DIFFERENT cmd_buf (`write_at_cmd_buf`) than the
data writes (`write_cmd_buf`), so even within-NOC-VC ordering of posted
writes does not bind them to the data writes. The receiver could see
the semaphore inc land at its L1 before the data writes do.

Path A's B.4-extended dump proved bytes appear CORRECT at the receiver
for layers {0, 1, 17, 35}. The dump fires AFTER many cycles of
execution — when transient first-cycle races would have been overwritten
by later correct data. So the dump cannot rule out a per-cycle
first-replay race or a transient race overwritten by the time the dump
fires. Hence S9 deserves a direct kernel-level probe.

## The surgical fix tested

In `writer_l1.cpp`, the only two lines changed in functional terms were:

```diff
- constexpr bool skip_ptr_update = get_compile_time_arg_val(8);
+ // S9 hypothesis test (Attack B, 2026-05-23): posted NOC writes may let
+ // the consumer matmul read c_31 before the data bytes have landed at the
+ // receiver's L1, even though the semaphore-inc the consumer waits on is
+ // posted too. Force non-posted writes here so the semaphore-inc carries an
+ // ack that orders behind the data writes' actual L1 commit at the
+ // receiver. Small perf cost; gating correctness fix.
+ // Original (perf): constexpr bool skip_ptr_update = get_compile_time_arg_val(8);
+ constexpr bool skip_ptr_update = false;
…
-                    noc_async_posted_writes_flushed();
+                    // S9 (Attack B, 2026-05-23): non-posted regime — wait for
+                    // the non-posted writes (data + semaphore-inc) to drain.
+                    noc_async_writes_flushed();
```

Effects via `remote_cb_push_back_and_write_pages<skip_ptr_update>`:
- `noc_async_write_one_packet_set_state<posted=false>(...)` and
  `noc_async_write_one_packet_with_state<posted=false>(...)` — every data
  write packet is now non-posted (carries L1 commit ack from receiver
  NIU).
- `noc_semaphore_inc<posted=false>(remote_sent_ptr_addr, …)` — the
  semaphore-inc the consumer waits on is non-posted (also carries
  ack).
- `noc_async_writes_flushed()` at the end of each per-block iteration
  drains all in-flight non-posted writes from the sender's NIU before
  the next block (note: this is "departed", not "acked-back" — but the
  ack mechanism on non-posted writes serializes the semaphore inc behind
  data at the receiver-side NIU through the cmd_buf ack chain).

Perf cost: small (~2-5 µs per per-block iteration × 36 layers ×
num_tensors × num_blocks per decode step). Not measured because
correctness gate failed.

## Execution log

1. Edit applied to `/home/mhnie/tt-metal-sglang/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp`.
2. Synced via `podman cp` into the container's
   `/tt-metal/ttnn/cpp/ttnn/operations/prefetcher/prefetcher/device/kernels/writer_l1.cpp`
   AND its install copy under
   `/tt-metal/build_Release/libexec/tt-metalium/...`. JIT compilation
   reads from `${TT_METAL_HOME}/ttnn/...`; libexec mirror is for the
   library install path.
3. Cleared `/root/.cache/tt-metal-cache/*` to force JIT recompile.
4. Required device reset (`sudo tt-smi -r 0000:01:00.0 0000:06:00.0`)
   to recover from accumulated state across multiple aborted launches
   (one attempt hit `RuntimeError: Read 0xffffffff over PCIe ID 1`
   pre-init, fixed by re-reset).
5. Launched server with canonical full-prefetcher envs:
   `SGLANG_TT_USE_PREFETCHER=1 SGLANG_TT_DISABLE_PREFILL_TRACE=1` per
   `tt_qwen3_8b_prefetcher_correctness_2026-05-23.md:25-34`.
6. Server reached `/health_generate = 200`. Multiple chat-detect probe
   prefills succeeded. First /generate hit at 00:29:01:
   * `[Prefetcher] 3-sub-device layout: 32 receiver cores`
   * `[Prefetcher] Program cache cleared`
   * `[Prefetcher Initialization]`
   * 10s pass while prefetcher state initializes and JIT-compiles the
     edited writer_l1 + reader_dram kernels.
   * Q1 GSM8K prompt prefilled via `_v25_suspend_prefetcher` flow
     (sub-device DECODE drain → DEFAULT manager switch).
   * **`Scheduler hit an exception … RuntimeError: probability tensor
     contains either inf, nan or element < 0`** at first decode step.
   * `[2026-05-24 00:29:11] SIGQUIT received. signum=None, frame=None.
     It usually means one child failed.`
7. Eval result: GSM8K(10) = **0/10** (Q1 RemoteDisconnected, Q2-10
   Connection refused). Identical pattern to canonical prefetcher per
   `tt_qwen3_8b_prefetcher_correctness_2026-05-23.md`.
8. Reverted edit (writer_l1.cpp pristine, git diff empty).
9. Synced clean writer_l1.cpp into container source + libexec.
10. Cleared kernel cache.
11. Re-reset devices for canonical baseline launch.
12. Canonical (no prefetcher) launched; GSM8K(10) re-verify result
    captured at session end.

## Why the fix engaged but produced the same failure

The kernel WAS recompiled — verified by:

* Kernel cache cleared pre-launch; JIT compiles on first
  `prefetcher.run()` call.
* The death point (`Prefetcher Initialization` followed by 10s of
  `_v25_suspend_prefetcher` drain followed by NaN-crash on the first
  user decode) is downstream of the prefetcher kernel program build.
* No JIT compile errors logged.

If S9 were the bug, the non-posted regime would have either:
(a) fixed the NaN — pass the correctness gate ≥ 7/10, or
(b) changed the failure mode (e.g. hang on the kernel-level
`noc_async_writes_flushed` due to a semaphore deadlock between non-posted
data write acks and the receiver's CB-pages-acked counters).

What we observed: server reached the SAME exact failure point with the
SAME exact signature as the unmodified canonical prefetcher. This means
either:

1. The bytes the receiver sees are wrong for a reason orthogonal to
   posted-vs-non-posted (which Path A's B.4-extended already
   established — bytes match across L0/L1/L17/L35).
2. The data-flow itself is correct but a sub-device CB-address /
   memory-config mismatch (S10) makes the matmul read from a WRONG
   physical L1 location — bytes are correct at the producer-intended
   destination, but matmul looks elsewhere.

S10 is the strongest remaining suspect. The SKIP_* reroute work in
phase_b8 (Path B's bank-grid fix) is orthogonal to the full-prefetcher
bug.

## Hypothesis ledger update

| ID | Suspect | Status after this dispatch |
|---|---|---|
| S1 | Row-wise stride mismatch in writer | RULED OUT (Phase B.3 byte-dump) |
| S2 | num_blocks mismatch | OPEN, unlikely (would be tensor-invariant) |
| S3 | WO K-shard transposition | RULED OUT (Phase A) |
| S5 | BFP4/BFP8 mixed tile pitch | RULED OUT (Phase A) |
| S6 | Per-layer cross-block address drift in producer | RULED OUT (Path A B.4-extended) |
| S7 | Compute-kernel `update_rd_ptr_to_ring_index` wrap | RULED OUT (Phase B.6 + Path A B.5) |
| S8 | Per-tensor block_size in reader_dram.cpp | RULED OUT (phase_b8 code analysis) |
| **S9** | **Producer NOC posted-writes flush insufficient** | **RULED OUT (this dispatch — non-posted regime gives same NaN-crash signature)** |
| S10 | Sub-device CB-address misalignment | **OPEN — last standing major prefetcher-side suspect** |
| NEW: SKIP_* reroute config | mesh_mapper / n-padding / output_mem_config | partially diagnosed in phase_b8 (Path B bank-grid necessary but not sufficient) |

## What this dispatch did NOT do

- Did NOT investigate S10 (sub-device CB-address misalignment). That
  is the recommended next-session attack per the phase_b8 doc.
- Did NOT add a DPRINT probe for the consumer-side
  `remote_cb_wait_front` semaphore value vs. receiver-side
  `pages_acked` counter (would prove whether the wait is on the right
  semaphore — but Path A's B.5 already showed compute reads land on
  valid BFP8 exponent bytes, so the wait is at least not totally
  broken).
- Did NOT bench TPOT (no correct outputs to bench).
- Did NOT modify any Python file (Attack A's scope, kept strictly
  separate).

## Recommended next session

**S10 deep-dive in tt-metal C++**:

1. Add DPRINT in `writer_l1.cpp` dumping `remote_cb.fifo_start_addr`
   (sender-side CB-region absolute L1 address) and
   `remote_cb.receiver_noc_xy_ptr` lookup.
2. Add DPRINT in
   `reader_bmm_tile_layout_in1_ring_all_gather.cpp`
   (consumer-side cb_in1) dumping its `fifo_start_addr` per receiver
   core.
3. If sender's `fifo_start_addr` for a given (tensor, receiver) does NOT
   match consumer's `fifo_start_addr` at the same core, that's S10
   confirmed. The likely culprit: `model_config.py` or
   `prefetcher.py` configuring `GlobalCB` with one base address but the
   matmul's reader uses a sub-device-specific CB-table with a different
   base address.
4. Alternative: add a `core.x, core.y` print on both sides so we can
   verify the same physical receiver core is being targeted by both
   sides (sub-device worker grids vs receiver grids).

Sister-attack (Attack A in `models/tt_transformers/tt/*.py`) tracks
SKIP_* reroute completion (mesh_mapper / n-padding / output_mem_config
per phase_b8 "Path forward A"). Once SKIP_* reroute is correct, S10 can
be triangulated by selectively SKIPing one weight at a time and
observing whether the bug surfaces per-weight or only with the full
ring topology.

## Working state at end of session

- tt-metal-sglang HEAD: **`c1e3437b78e`** (unchanged — Attack B fix
  reverted; only writer_l1.cpp was touched and now matches HEAD).
- sglang HEAD: this doc + commit message pending.
- C++ kernel tree pristine.
- `/root/.cache/tt-metal-cache/*` cleared.
- Canonical baseline GSM8K(10) chat: see session-end re-verify section
  below.
- All cards healthy after final reset.
- No server running at end.

## Final canonical re-verify

Per `/tmp/qwen3_8b_canonical_reverify.log`:

```
Q 1/10 CORRECT gt=64.0 pred=64.0  58.5s  last_line='#### <final_number>'
Q 2/10 CORRECT gt=260.0 pred=260.0  38.0s  last_line='#### 260'
Q 3/10 CORRECT gt=160.0 pred=160.0 171.0s  last_line='#### 160'
Q 4/10 CORRECT gt=45.0 pred=45.0 104.5s  last_line='$$'
Q 5/10 CORRECT gt=460.0 pred=460.0  58.9s  last_line='$$'
Q 6/10 CORRECT gt=366.0 pred=366.0 153.7s  last_line='$$'
Q 7/10 CORRECT gt=694.0 pred=694.0  71.6s  last_line='$$'
Q 8/10 CORRECT gt=13.0 pred=13.0 103.7s  last_line='$$'
Q 9/10 CORRECT gt=18.0 pred=18.0 123.4s  last_line='$$'
Q10/10 CORRECT gt=60.0 pred=60.0  50.9s  last_line='#### <final_number>'

FINAL: 10/10 = 100.0%
```

**No regression: 10/10 > the 9/10 threshold.** Canonical decode at 29-30
tok/s steady-state throughout. Server-side GET /health_generate
returned 200 throughout the eval. Server cleanly killed at end.

## Shipping verdict (unchanged)

Canonical Qwen3-8B (no prefetcher) remains the production shipping
config at TPOT ≈ 27 ms / 1024-1024 and GSM8K(10) chat ≥ 9/10. DO NOT
ship `SGLANG_TT_USE_PREFETCHER=1` for Qwen3-8B. S10 (sub-device
CB-address) is the last standing prefetcher-side suspect.

## Commits this session

- (tt-metal-sglang) **none** — the test edit was reverted; HEAD remains
  `c1e3437b78e`.
- (sglang) this doc + commit message pending; no Python source changes.
