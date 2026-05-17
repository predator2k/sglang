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

- 2026-05-17: fork created at pin `eb9005fa`, branch `tenstorrent-p1`. No patches applied yet.
