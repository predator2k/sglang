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

- `/home/mhnie/tt-mlir-sglang/` — fork with B.2 patch on `tenstorrent-p1` branch (`c1bae0bf2`)
- `/opt/tt-mlir-toolchain/` — built toolchain (LLVM, FlatBuffers, StableHLO, Shardy)
- `/home/mhnie/tt-xla/build-local/` — built tt-xla with new `pjrt_plugin_tt.so` (host-built, ABI-drifted)
- Container `tt-xla-eval` — recreated with extra mounts (`/tt-mlir-sglang`, `/tt-xla`, `/opt/tt-mlir-toolchain`) + build tools (cmake, clang-17, lld, ninja, ccache, libzstd-dev, libprotobuf-dev, etc.) + sglang server runtime deps. Workstream A still passes the bit-exact CI test on this configuration.

## Container-rebuild full pipeline attempt (2026-05-17 19:25-20:30)

Two passes attempted Path 1 ("build inside container"):

### Pass A — manual stepwise build (19:25-19:50)

Built `tt-mlir-sglang` directly inside container at `/tt-mlir-sglang/build`. Hit cascading missing-generated-headers chain (TableGen `.h.inc` files for shardy + stablehlo).

### Pass B — tt-xla's own ExternalProject pipeline (20:00-20:30)

Let tt-xla's `ExternalProject_Add(tt-mlir ...)` build everything end-to-end with `TTMLIR_SOURCE_DIR_OVERRIDE=/tt-mlir-sglang`. This auto-handles TableGen generation. Steps:

- ✓ Installed all build deps in container: cmake/clang-17/lld/ninja/ccache, libzstd-dev, libprotobuf-dev, patchelf, libcapstone-dev, libxxhash-dev, libnuma-dev, libhwloc-dev, libopenmpi-dev, openmpi-bin, openmpi-common, libgtest-dev, libssl-dev, zlib1g-dev, libffi-dev
- ✓ Installed sglang server runtime deps
- ✓ Provisioned SFPI 7.47.0 (downgrade from 7.48 — tt-metal pin)
- ✓ Patched `tt-mlir-sglang/CMakeLists.txt` to `TT_RUNTIME_ENABLE_DISTRIBUTED OFF` by default (committed at `571063ab0`) — to avoid OpenMPI build/runtime fragility
- ✓ Full pipeline build succeeded: tt-mlir-sglang → install at `/tt-xla/third_party/tt-mlir/install/` → tt-xla → `pjrt_plugin_tt.so` (1.4 MB, links cleanly against `/lib/x86_64-linux-gnu/libprotobuf.so.32` and our `/tt-xla/third_party/tt-mlir/install/lib/libTTMLIR{Compiler,Runtime}.so`)
- ✓ Tiny `torch.compile(backend="tt")` smoke test PASSES inside container
- ✗ `test_t2_1_functional_cache.py` and `probe_t2_1_v3_writeops.py` FAIL at runtime:
  - editable-install wrapper → `torch._dynamo.exc.InternalTorchDynamoError: TypeError: eval()` (guard creation crash)
  - canonical-wrapper + swap-only-libs → `RuntimeError: TT_THROW @ /tt-mlir-sglang/.../kernel.cpp:89` during prefill execute, then `LogMessageFatal: ComputationClient already initialized` during cleanup

### Root cause analysis

The new build picks up tt-metal commit `d5a16537336229bee54fd4a6d8bd54c492abc7d1` (newer; what the eb9005fa tt-mlir pin specifies). The slim container's canonical `pjrt-plugin-tt 1.1.0` wheel was built against tt-metal commit `90c914ef258b5cc92ad172f3604b784ec77253ca` (older). The container's `torch_xla 2.9.0+git44ecef3` was validated against the OLDER tt-metal. The newer tt-metal pulls in kernel-level changes that haven't been validated against this torch_xla; baseline prefills now `TT_THROW` even without B.2-relevant code paths.

### What works at this snapshot

- `/home/mhnie/tt-mlir-sglang/` — fork at branch `tenstorrent-p1` with B.2 patch (`c1bae0bf2`) + distributed-runtime-off patch (`571063ab0`)
- `/opt/tt-mlir-toolchain/` — built LLVM/MLIR toolchain (5.1 GB)
- `/home/mhnie/tt-xla/build-local/pjrt_implementation/src/pjrt_plugin_tt.so` — newly-built plugin (1.4 MB, ABI-matched to container's torch_xla per ldd, NOT runtime-validated)
- `/home/mhnie/tt-xla/third_party/tt-mlir/install/` — tt-mlir install tree consumed by tt-xla build

### Next attempt should

1. **Use a Tenstorrent development container** (`tt-xla` non-slim) instead of `tt-xla-slim`. The dev image has matching dev environment + the tt-metal pin tt-xla's pin expects, fully built end-to-end.
2. **Pin tt-mlir-sglang to a commit whose tt-metal pin is the SAME as `tt-xla-slim:latest`'s canonical**: pull tt-metal commit `90c914ef258b5cc92ad172f3604b784ec77253ca` instead of the newer one. This is a tt-mlir rebase to an older base that includes B.2 patches behaviorally.
3. **Build a custom slim image** — start FROM the canonical slim, install build deps, build only `libTTMLIRCompiler.so` and `libTTMLIRRuntime.so` (NOT tt-metal — keep that canonical), and replace those libs.

Path 3 is the most surgical — it doesn't pull in a new tt-metal. But it requires careful CMake config of tt-mlir-sglang to consume the canonical tt-metal install rather than building its own. This is multi-day work.

For this work cycle, **Workstream A is the shippable outcome.**

- ✓ Container recreated with bind-mounts for `tt-mlir-sglang`, `tt-xla`, `tt-mlir-toolchain`
- ✓ Build tools installed: cmake 3.28, clang/clang++-17, ninja, ccache, libzstd-dev, libprotobuf-dev, patchelf, libfmt9, libopenmpi3
- ✓ tt-mlir-sglang rebuilt inside container at `/tt-mlir-sglang/build/` (790/790 steps)
- ✓ tt-mlir-sglang manual install to `/tt-xla/third_party/tt-mlir/install/`
- ✓ Headers copied from `runtime/include/tt/` and `build/runtime/include/tt/`
- ✓ tt-metal source symlinked at `/tt-xla/third_party/tt-mlir/install/tt-metal/{tools,ttnn}`
- ✗ **Blocked**: tt-xla compilation needs TableGen-generated `.h.inc` files from shardy and stablehlo (`shardy/dialect/sdy/ir/dialect.h.inc`, `stablehlo/dialect/BaseAttrInterfaces.h.inc`). These are normally generated as part of tt-mlir's nested build under tt-xla's ExternalProject pipeline. Bypassing via `TTMLIR_SOURCE_DIR_OVERRIDE` skips this auto-handling.

**Next attempt should**: either (a) Dockerfile that starts from `tt-xla-slim`, patches the relevant source, rebuilds via tt-xla's standard ExternalProject pipeline (which auto-generates the .inc files), or (b) add an install rule in `tt-mlir-sglang` that exports the generated .inc files to a location tt-xla can find. Both are multi-hour follow-up work.
