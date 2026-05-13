# R5 (HIGH severity, RadixAttention paged-evict consistency) — mitigation acceptance reduction

Original spec §5.1 promised three orthogonal verifications:
  1. §9.11 eviction-replay (end-to-end) — DONE in T3.7
  2. §9.b5 24h+ long-run KV pool free-list consistency — DEFERRED to P3
  3. §9.15 free-list invariant unit test — DROPPED (no TTPagedKVAdapter)

R5 now relies on §9.11 alone. Documented acceptance reduction.
If R5 reproduces in P3+ workloads, the spec should be re-amended.
