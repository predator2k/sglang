# tt-mlir-sglang local fork setup

Date created: 2026-05-17
Base commit: `eb9005fa360a80e44607e2dfd4404137b510092e` (upstream main as of 2026-05-13)
Path: `/home/mhnie/tt-mlir-sglang/`
Branch: `tenstorrent-p1`
Origin (for future rebases): `https://github.com/tenstorrent/tt-mlir.git`

## Why this fork exists

Spec `docs/superpowers/specs/2026-05-17-tt-xla-tpot-full-stack-design.md` v5.3
workstream B: two compiler-level fixes are needed in tt-mlir that aren't
expected to land upstream in time for this work:

- **B.2** — `CacheFillUpdatePattern` in `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp`
  matches autoregressive-decode scatter ops and lowers them via paths that
  eventually require the cache tensor to have exactly one user. In SGLang
  autoregressive decode the cache is also read by attention (2 users), so
  legalization fails. The fix is a `hasOneUse()` guard that lets multi-user
  scatter fall through to the generic `ttnn.scatter` path.

- **B.1** — `aten::index_put` dim-mismatch crash in SGLang's built-in
  server warmup. Pattern + fix TBD in Phase 3a.

## Layout

This fork is OUTSIDE the sglang repo (matches the `tt-metal-sglang` pattern at
`/home/mhnie/tt-metal-sglang/` documented in user memory `tenstorrent-tt-metal-fork`).

## Build / rebuild

See Phase 1b.3-1b.4 of the implementation plan. Once a build env exists
(`cmake`, `clang`, LLVM toolchain), the rebuild + install cycle is:

```bash
# Future scripts/build_and_install.sh will wrap this:
ninja -C /home/mhnie/tt-mlir-sglang/build
ninja -C /home/mhnie/tt-xla/build-local
PLUGIN=$(find /home/mhnie/tt-xla/build-local -name "pjrt_plugin_tt.so" | head -1)
docker cp "$PLUGIN" tt-xla-eval:/usr/local/lib/python3.12/dist-packages/pjrt_plugin_tt/pjrt_plugin_tt.so
```

## Rebase policy

Pin upstream periodically (~monthly) for security/correctness fixes. Each
rebase MUST be followed by `regression_check_v145.py` PASS.

## Status

- **2026-05-17 03:47** — fork created at pin `eb9005fa`, branch `tenstorrent-p1`
- **2026-05-17 03:49** — `scripts/build_and_install.sh` committed (tt-mlir-sglang commit `04ae33328`)
- **2026-05-17 ~04:00** — **B.2 patch applied** (tt-mlir-sglang commit `c1bae0bf2`): 14-line guard in `CacheFillUpdatePattern::matchAndRewrite` at `lib/Conversion/StableHLOToTTIR/StableHLOToTTIRPatterns.cpp:6124` — returns `failure()` when `scatterOp.getInputs()[0]` has more than one user, letting the autoregressive cache scatter fall through to the generic `ttnn.scatter` path.
- **2026-05-17 04:03** — LLVM/MLIR toolchain built at `/opt/tt-mlir-toolchain/` (5.1 GB, 35/35 ExternalProject steps).
- **2026-05-17 ~04:20** — tt-mlir-sglang top-level built (`build/lib/libTTMLIRCompiler.so`, `build/runtime/lib/libTTMLIRRuntime.so`).
- **2026-05-17 ~04:25** — tt-xla built against local override (`build-local/pjrt_implementation/src/pjrt_plugin_tt.so`, 2 MB).

## Known integration issue (open)

The host-built `pjrt_plugin_tt.so` cannot be dropped directly into the slim container's `pjrt_plugin_tt/lib64/` and remain working. Symptoms while integrating:
- `libprotobuf.so.32` missing (host v32 vs container's bundled v25). Fixed by `apt install libprotobuf32t64`.
- `undefined symbol: _ZN4mlir2tt4ttnn17symbolizeBFPDtypeEN4llvm9StringRefE` — bundled `libTTMLIRCompiler.so` was older than what the new plugin expects. Replaced from local build.
- `libtt-umd.so.0`, `libfmt.so.11`, `libmpi.so.40` chain — bulk-copied tt-metal `build_Release/lib/*.so*` + apt-installed `libopenmpi3`.
- After all that: plugin LOADS but compiled `forward()` FAILS without a clean error, suggesting ABI drift between host-built libs and the container's torch / torch_xla (locked at 2.9.1+cpu / 2.9.0+git44ecef3 per memory `tenstorrent-tt-transformers-constraints`).

To validate B.2 end-to-end, one of:
1. **Build pjrt-plugin-tt INSIDE the slim container** so it links against the container's exact torch / nanobind / protobuf. Requires cmake + clang + lld toolchain inside the container (~5-10 GB image growth).
2. **Match versions on the host**: install torch 2.9.1+cpu + torch_xla 2.9.0+git44ecef3 + matching nanobind in a clean Python env on the host, point the tt-xla build at it.
3. **Build a new `tt-xla-slim` image** with our pjrt plugin baked in, auditwheel-bundled.

Rebuild + reinstall cycle (after the ABI integration is solved):

```bash
bash /home/mhnie/tt-mlir-sglang/scripts/build_and_install.sh
```

## What's preserved

- `/home/mhnie/tt-mlir-sglang/` — fork with B.2 patch on `tenstorrent-p1` branch
- `/opt/tt-mlir-toolchain/` — built toolchain (LLVM, FlatBuffers, StableHLO, Shardy)
- `/home/mhnie/tt-xla/build-local/` — built tt-xla with new `pjrt_plugin_tt.so`
- Container `tt-xla-eval` restored to canonical state (original `pjrt-plugin-tt 1.1.0` wheel) — Workstream A still passes the bit-exact CI test.
