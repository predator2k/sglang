#!/bin/bash
# Benchmark Qwen3-8B decode TPOT on 2× P150a Blackhole.
#
# Measures output tokens per second (tok/s) and time per output token (TPOT)
# across N runs of 100 output tokens each, with timing instrumentation enabled.
#
# Usage:
#   bash bench_decode_perf.sh          # 5 runs, default settings
#   bash bench_decode_perf.sh N        # N runs
#
# Prerequisites:
#   - Container 'p3a-ngram' running with sglang installed
#   - Fork patches copied to container
#   - Server launched with SGLANG_TT_DECODE_TIMING=1
#
# Environment:
#   CONTAINER    Container name (default: p3a-ngram)
#   PORT         Server port (default: 30000)
#   MAX_TOKENS   Tokens per run (default: 100)
#   PRECISION    Precision mode (default: lofi_hifi2_sdpa)

set -uo pipefail

CONTAINER="${CONTAINER:-p3a-ngram}"
PORT="${PORT:-30000}"
MAX_TOKENS="${MAX_TOKENS:-100}"
PRECISION="${PRECISION:-lofi_hifi2_sdpa}"
N_RUNS="${1:-5}"

PROMPT='Translate the following English text to French. English: "The quantum computer achieved a breakthrough in solving complex optimization problems that would take classical computers millions of years." French:'

echo "=== Qwen3-8B Decode Perf Benchmark ==="
echo "Container: $CONTAINER"
echo "Port: $PORT"
echo "Max tokens: $MAX_TOKENS"
echo "Precision: $PRECISION"
echo "Runs: $N_RUNS"
echo ""

# ---- Helper: copy fork patches ----
copy_patches() {
    echo "[SETUP] Copying fork patches..."
    for f in model_config.py generator_sglang.py distributed_norm.py attention.py; do
        podman cp "/home/mhnie/tt-metal-sglang/models/tt_transformers/tt/$f" \
            "$CONTAINER:/tt-metal/models/tt_transformers/tt/$f"
    done
    echo "[SETUP] Fork patches copied."
}

# ---- Helper: launch server ----
launch_server() {
    echo "[SETUP] Launching SGLang server with decode timing..."
    podman exec -d "$CONTAINER" bash -lc "
        export SGLANG_TT_QWEN3_PRECISION=$PRECISION
        export SGLANG_TT_DECODE_TIMING=1
        cd /home/container_app_user/sglang
        python -m sglang.launch_server \
            --model-path /models/Qwen3-8B \
            --port $PORT \
            --tp 1 \
            --max-running-requests 32 \
            --context-length 8192 \
            --mem-fraction-static 0.85 \
            --chunked-prefill-size 2048 \
            --page-size 64 \
            --sampling-backend pytorch \
            --chat-template qwen3-chat-notools \
            --attention-backend torch_native \
            --disable-cuda-graph \
            --log-level info 2>&1 | tee /tmp/sglang_bench.log
    " 2>/dev/null
    echo "[SETUP] Waiting for server to be ready..."
    for i in $(seq 1 120); do
        if podman exec "$CONTAINER" curl -s "http://localhost:$PORT/health" | grep -q '"status"'; then
            echo "[SETUP] Server ready after ${i}s"
            return 0
        fi
        sleep 1
    done
    echo "[ERROR] Server failed to start within 120s"
    return 1
}

# ---- Helper: kill server ----
kill_server() {
    podman exec "$CONTAINER" bash -lc 'pkill -9 -f sglang.launch_server 2>/dev/null; sleep 2; pkill -9 -f "sglang::" 2>/dev/null' || true
    sleep 3
}

# ---- Single benchmark run ----
bench_run() {
    local run_num=$1
    local result
    result=$(podman exec "$CONTAINER" curl -s "http://localhost:$PORT/generate" \
        -H "Content-Type: application/json" \
        -d "{
            \"text\": \"$PROMPT\",
            \"sampling_params\": {
                \"max_new_tokens\": $MAX_TOKENS,
                \"temperature\": 0,
                \"top_p\": 1.0
            }
        }" 2>/dev/null)

    local meta
    meta=$(echo "$result" | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    m = d.get('meta_info', {})
    comp_tok = m.get('completion_tokens', 0)
    e2e = m.get('e2e_request_latency', 0)
    ttft = m.get('prompt_tokens_details', {}).get('ttft', 0)
    if comp_tok > 0 and e2e > 0:
        decode_time = e2e - ttft
        tpot_ms = (decode_time / (comp_tok - 1)) * 1000 if comp_tok > 1 else 0
        tps = comp_tok / e2e
        print(f'{comp_tok},{e2e:.3f},{ttft:.3f},{tpot_ms:.2f},{tps:.1f}')
    else:
        print('0,0,0,0,0')
except Exception:
    print('0,0,0,0,0')
" 2>/dev/null)

    echo "$meta"
}

# ---- Main benchmark loop ----
echo ""
echo "run,tokens,e2e_s,ttft_s,tpot_ms,tok_s"

total_tps=0
total_tpot=0
n_ok=0

for i in $(seq 1 "$N_RUNS"); do
    line=$(bench_run "$i")
    IFS=',' read -r tokens e2e ttft tpot tps <<< "$line"
    echo "$i,$tokens,$e2e,$ttft,$tpot,$tps"
    if [ "$tps" != "0" ] && [ "$tps" != "0.0" ]; then
        total_tps=$(python3 -c "print($total_tps + $tps)")
        total_tpot=$(python3 -c "print($total_tpot + $tpot)")
        n_ok=$((n_ok + 1))
    fi
done

echo ""
if [ "$n_ok" -gt 0 ]; then
    avg_tps=$(python3 -c "print(f'{$total_tps / $n_ok:.1f}')")
    avg_tpot=$(python3 -c "print(f'{$total_tpot / $n_ok:.2f}')")
    echo "=== RESULTS ($n_ok successful runs) ==="
    echo "Average tok/s: $avg_tps"
    echo "Average TPOT: ${avg_tpot}ms"
    echo ""
    echo "Target: >= 32.0 tok/s (TT vLLM baseline: 32.6 tok/s)"
    python3 -c "
tps = $total_tps / $n_ok
gap = 32.6 - tps
pct = (gap / 32.6) * 100
if gap > 0:
    print(f'Gap remaining: {gap:.1f} tok/s ({pct:.1f}%)')
else:
    print(f'TARGET MET! {-gap:.1f} tok/s above vLLM baseline')
"
else
    echo "=== ERROR: All runs failed ==="
fi

echo ""
echo "Decode timing details (from server logs):"
podman exec "$CONTAINER" grep -o '\[TT-SGLANG\] decode timing.*' /tmp/sglang_bench.log 2>/dev/null | tail -3
