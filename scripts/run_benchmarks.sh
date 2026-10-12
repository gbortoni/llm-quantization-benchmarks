#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

HARNESS_REPO="/Users/gabrielbortoni/llama-cpp-inference"
HARNESS="$HARNESS_REPO/build-p3/llama-cpp-inference"

LLAMA_REPO="/Users/gabrielbortoni/dev/edge-inference/llama.cpp"

EXPECTED_HARNESS_SOURCE_COMMIT="38f9a54d74bf2b7433cc4dbda5598c09b72a5a63"
EXPECTED_LLAMA_COMMIT="e71b80510c848c00175924ecf3c40333ccae8eb5"

MODEL_DIR="$ROOT/models/gguf"
RAW_DIR="$ROOT/results/raw"

PREFILL_PROMPT_FILE="$ROOT/prompts/prefill_256.txt"
DECODE_PROMPT_FILE="$ROOT/prompts/decode_32.txt"

CONTEXT=512
BATCH=256
THREADS=6
MEASURED_RUNS=5

mkdir -p "$RAW_DIR"

# -----------------------------
# Reproducibility checks
# -----------------------------

HARNESS_SOURCE_COMMIT="$(
    git -C "$HARNESS_REPO" log -1 --format=%H -- src/main.cpp
)"

LLAMA_COMMIT="$(
    git -C "$LLAMA_REPO" rev-parse HEAD
)"

if [[ "$HARNESS_SOURCE_COMMIT" != "$EXPECTED_HARNESS_SOURCE_COMMIT" ]]; then
    echo "Error: unexpected benchmark harness source commit:"
    echo "$HARNESS_SOURCE_COMMIT"
    exit 1
fi

if [[ "$LLAMA_COMMIT" != "$EXPECTED_LLAMA_COMMIT" ]]; then
    echo "Error: unexpected llama.cpp commit:"
    echo "$LLAMA_COMMIT"
    exit 1
fi

if ! git -C "$HARNESS_REPO" diff --quiet HEAD --; then
    echo "Error: benchmark harness repository has uncommitted tracked changes."
    exit 1
fi

if ! git -C "$LLAMA_REPO" diff --quiet HEAD --; then
    echo "Error: llama.cpp has uncommitted tracked changes."
    exit 1
fi

if [[ ! -x "$HARNESS" ]]; then
    echo "Error: benchmark harness not found:"
    echo "$HARNESS"
    exit 1
fi

if [[ ! -f "$PREFILL_PROMPT_FILE" || ! -f "$DECODE_PROMPT_FILE" ]]; then
    echo "Error: benchmark prompt file missing."
    exit 1
fi

PREFILL_PROMPT="$(cat "$PREFILL_PROMPT_FILE")"
DECODE_PROMPT="$(cat "$DECODE_PROMPT_FILE")"

echo "P3 quantization benchmark"
echo "Harness source commit: $HARNESS_SOURCE_COMMIT"
echo "llama.cpp commit: $LLAMA_COMMIT"
echo "Context: $CONTEXT"
echo "Batch: $BATCH"
echo "Threads: $THREADS"
echo "Measured runs: $MEASURED_RUNS"

validate_runtime() {
    FILE="$1"
    FORMAT="$2"
    WORKLOAD="$3"

    if ! grep -q 'Effective context: 512' "$FILE"; then
        echo "Error: unexpected context for $FORMAT $WORKLOAD."
        exit 1
    fi

    if ! grep -q 'Effective batch: 256' "$FILE"; then
        echo "Error: unexpected batch for $FORMAT $WORKLOAD."
        exit 1
    fi

    if ! grep -q 'Effective generation threads: 6' "$FILE"; then
        echo "Error: unexpected generation threads for $FORMAT $WORKLOAD."
        exit 1
    fi

    if ! grep -q 'Effective batch threads : 6' "$FILE"; then
        echo "Error: unexpected batch threads for $FORMAT $WORKLOAD."
        exit 1
    fi
}

run_format() {
    FORMAT="$1"
    MODEL="$2"

    echo
    echo "========================================"
    echo "Running $FORMAT"
    echo "========================================"

    if [[ ! -f "$MODEL" ]]; then
        echo "Error: model not found: $MODEL"
        exit 1
    fi

    # -----------------------------
    # Load-time benchmark
    # -----------------------------

    echo "Running load-time trials..."

    LOAD_DIR="$RAW_DIR/${FORMAT}_load"
    rm -rf "$LOAD_DIR"
    mkdir -p "$LOAD_DIR"

    for RUN in $(seq 1 "$MEASURED_RUNS"); do
        LOAD_FILE="$LOAD_DIR/run_${RUN}.txt"

        "$HARNESS" \
            --model "$MODEL" \
            --prompt "load" \
            --mode benchmark \
            --benchmark-workload load \
            --tokens 1 \
            > "$LOAD_FILE" 2>&1

        LOAD_RESULTS="$(
            grep -c '^P3_RESULT workload=load' "$LOAD_FILE" || true
        )"

        if [[ "$LOAD_RESULTS" -ne 1 ]]; then
            echo "Error: invalid load result for $FORMAT run $RUN."
            exit 1
        fi
    done

    # -----------------------------
    # Prefill benchmark
    # -----------------------------

    echo "Running prefill..."

    PREFILL_FILE="$RAW_DIR/${FORMAT}_prefill.txt"

    "$HARNESS" \
        --model "$MODEL" \
        --prompt "$PREFILL_PROMPT" \
        --mode benchmark \
        --benchmark-workload prefill \
        --tokens 1 \
        --context "$CONTEXT" \
        --batch "$BATCH" \
        --threads "$THREADS" \
        > "$PREFILL_FILE" 2>&1

    if ! grep -q 'Prompt Tokens: 256' "$PREFILL_FILE"; then
        echo "Error: $FORMAT prefill prompt was not 256 tokens."
        exit 1
    fi

    validate_runtime "$PREFILL_FILE" "$FORMAT" "prefill"

    PREFILL_RESULTS="$(
        grep -c '^P3_RESULT workload=prefill' "$PREFILL_FILE" || true
    )"

    if [[ "$PREFILL_RESULTS" -ne "$MEASURED_RUNS" ]]; then
        echo "Error: expected $MEASURED_RUNS prefill results for $FORMAT; got $PREFILL_RESULTS."
        exit 1
    fi

    # -----------------------------
    # Decode benchmark
    # -----------------------------

    echo "Running decode..."

    DECODE_FILE="$RAW_DIR/${FORMAT}_decode.txt"

    "$HARNESS" \
        --model "$MODEL" \
        --prompt "$DECODE_PROMPT" \
        --mode benchmark \
        --benchmark-workload decode \
        --tokens 128 \
        --context "$CONTEXT" \
        --batch "$BATCH" \
        --threads "$THREADS" \
        > "$DECODE_FILE" 2>&1

    if ! grep -q 'Prompt Tokens: 32' "$DECODE_FILE"; then
        echo "Error: $FORMAT decode prompt was not 32 tokens."
        exit 1
    fi

    validate_runtime "$DECODE_FILE" "$FORMAT" "decode"

    DECODE_RESULTS="$(
        grep -c '^P3_RESULT workload=decode' "$DECODE_FILE" || true
    )"

    if [[ "$DECODE_RESULTS" -ne "$MEASURED_RUNS" ]]; then
        echo "Error: expected $MEASURED_RUNS decode results for $FORMAT; got $DECODE_RESULTS."
        exit 1
    fi

    echo "$FORMAT complete."
}

run_format "F16"    "$MODEL_DIR/Qwen2.5-1.5B-F16.gguf"
run_format "Q8_0"   "$MODEL_DIR/Qwen2.5-1.5B-Q8_0.gguf"
run_format "Q6_K"   "$MODEL_DIR/Qwen2.5-1.5B-Q6_K.gguf"
run_format "Q4_K_M" "$MODEL_DIR/Qwen2.5-1.5B-Q4_K_M.gguf"
run_format "Q4_0"   "$MODEL_DIR/Qwen2.5-1.5B-Q4_0.gguf"

echo
echo "All performance benchmarks complete."
