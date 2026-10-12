#!/bin/bash

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

LLAMA_REPO="/Users/gabrielbortoni/dev/edge-inference/llama.cpp"
PPL="$LLAMA_REPO/build-p3/bin/llama-perplexity"

EXPECTED_LLAMA_COMMIT="e71b80510c848c00175924ecf3c40333ccae8eb5"
EXPECTED_CORPUS_SHA="173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08"

CORPUS="$ROOT/data/wiki.test.raw"
MODEL_DIR="$ROOT/models/gguf"
RAW_DIR="$ROOT/results/raw"

CONTEXT=512
BATCH=256
UBATCH=256
THREADS=6

mkdir -p "$RAW_DIR"

# -----------------------------
# Reproducibility checks
# -----------------------------

LLAMA_COMMIT="$(git -C "$LLAMA_REPO" rev-parse HEAD)"

if [[ "$LLAMA_COMMIT" != "$EXPECTED_LLAMA_COMMIT" ]]; then
    echo "Error: unexpected llama.cpp commit:"
    echo "$LLAMA_COMMIT"
    exit 1
fi

if ! git -C "$LLAMA_REPO" diff --quiet HEAD --; then
    echo "Error: llama.cpp has uncommitted tracked changes."
    exit 1
fi

if [[ ! -x "$PPL" ]]; then
    echo "Error: llama-perplexity binary not found:"
    echo "$PPL"
    exit 1
fi

if [[ ! -f "$CORPUS" ]]; then
    echo "Error: corpus not found:"
    echo "$CORPUS"
    exit 1
fi

CORPUS_SHA="$(shasum -a 256 "$CORPUS" | awk '{print $1}')"

if [[ "$CORPUS_SHA" != "$EXPECTED_CORPUS_SHA" ]]; then
    echo "Error: WikiText-2 corpus SHA-256 mismatch."
    echo "Expected: $EXPECTED_CORPUS_SHA"
    echo "Actual:   $CORPUS_SHA"
    exit 1
fi

echo "P3 perplexity benchmark"
echo "llama.cpp commit: $LLAMA_COMMIT"
echo "Corpus SHA-256:  $CORPUS_SHA"
echo "Context: $CONTEXT"
echo "Batch: $BATCH"
echo "Ubatch: $UBATCH"
echo "Threads: $THREADS"

run_format() {
    FORMAT="$1"
    MODEL="$2"

    echo
    echo "========================================"
    echo "Running $FORMAT perplexity"
    echo "========================================"

    if [[ ! -f "$MODEL" ]]; then
        echo "Error: model not found: $MODEL"
        exit 1
    fi

    # -----------------------------------
    # Placement preflight
    # -----------------------------------

    PLACEMENT_FILE="$RAW_DIR/${FORMAT}_perplexity_placement.txt"

    "$PPL" \
        -m "$MODEL" \
        -f "$CORPUS" \
        -c "$CONTEXT" \
        -b "$BATCH" \
        -ub "$UBATCH" \
        -t "$THREADS" \
        -tb "$THREADS" \
        --gpu-layers all \
        --chunks 1 \
        --ppl-stride 0 \
        --ppl-output-type 0 \
        --warmup \
        --verbosity 5 \
        > "$PLACEMENT_FILE" 2>&1

    if ! grep -q 'using device MTL0 (Apple M2 Pro)' "$PLACEMENT_FILE"; then
        echo "Error: $FORMAT did not report Apple M2 Pro Metal placement."
        exit 1
    fi

    if ! grep -q 'offloaded 29/29 layers to GPU' "$PLACEMENT_FILE"; then
        echo "Error: $FORMAT did not report full layer offload."
        exit 1
    fi

    echo "$FORMAT placement verified."

    # -----------------------------------
    # Full WikiText-2 perplexity
    # -----------------------------------

    RESULT_FILE="$RAW_DIR/${FORMAT}_perplexity.txt"

    "$PPL" \
        -m "$MODEL" \
        -f "$CORPUS" \
        -c "$CONTEXT" \
        -b "$BATCH" \
        -ub "$UBATCH" \
        -t "$THREADS" \
        -tb "$THREADS" \
        --gpu-layers all \
        --chunks -1 \
        --ppl-stride 0 \
        --ppl-output-type 0 \
        --warmup \
        > "$RESULT_FILE" 2>&1

    if ! grep -q 'Final estimate: PPL =' "$RESULT_FILE"; then
        echo "Error: no final perplexity estimate for $FORMAT."
        exit 1
    fi

    PPL_VALUE="$(
        grep 'Final estimate: PPL =' "$RESULT_FILE" |
        tail -1 |
        sed -E 's/.*PPL = ([0-9.]+).*/\1/'
    )"

    echo "$FORMAT final PPL: $PPL_VALUE"
}

run_format "F16"    "$MODEL_DIR/Qwen2.5-1.5B-F16.gguf"
run_format "Q8_0"   "$MODEL_DIR/Qwen2.5-1.5B-Q8_0.gguf"
run_format "Q6_K"   "$MODEL_DIR/Qwen2.5-1.5B-Q6_K.gguf"
run_format "Q4_K_M" "$MODEL_DIR/Qwen2.5-1.5B-Q4_K_M.gguf"
run_format "Q4_0"   "$MODEL_DIR/Qwen2.5-1.5B-Q4_0.gguf"

echo
echo "All perplexity benchmarks complete."
