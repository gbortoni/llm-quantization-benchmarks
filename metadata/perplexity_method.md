# Perplexity Evaluation Method

## Corpus

Dataset: WikiText-2 raw v1 test split

Local evaluation file:

`data/wiki.test.raw`

Downloaded using the helper from the pinned llama.cpp checkout:

`scripts/get-wikitext-2.sh`

Upstream archive:

`wikitext-2-raw-v1.zip`

Upstream source:

`https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip`

Corpus identity:

- SHA-256: `173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`
- Size: `1290590` bytes
- Lines: `4358`

The upstream URL points to `main`, so reproducibility is based on the
recorded SHA-256 rather than assuming the URL remains immutable.

## Evaluation Tool

llama.cpp commit:

`e71b80510c848c00175924ecf3c40333ccae8eb5`

Tool:

`llama-perplexity`

## Final Evaluation Configuration

- context size: 512
- logical batch size: 256
- physical batch size: 256
- generation threads: 6
- batch threads: 6
- GPU layers: all
- chunks: all
- perplexity stride: 0
- perplexity output type: 0
- warmup: enabled

The same corpus and evaluation configuration are used for:

- F16
- Q8_0
- Q6_K
- Q4_K_M
- Q4_0

Perplexity is treated only as a measure of next-token predictive fit on
this corpus. It is not treated as a complete measure of model quality.
