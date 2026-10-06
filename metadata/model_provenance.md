# Model Provenance

## Source model

- Hugging Face repository: `Qwen/Qwen2.5-1.5B`
- Pinned revision: `8faed761d45a263340a0528343f099c05c9a4323`
- Architecture: `Qwen2ForCausalLM`
- Model type: `qwen2`
- Hidden size: `1536`
- Layers: `28`
- Vocabulary size: `151936`

## llama.cpp

- Pinned commit: `e71b80510c848c00175924ecf3c40333ccae8eb5`
- Build type: `Release`
- Backend enabled: `Metal`

## Conversion environment

- Python: `3.12.7`
- torch: `2.11.0`
- numpy: `1.26.4`
- transformers: `4.57.6`
- sentencepiece: `0.2.2`
- protobuf: `4.25.9`

## GGUF artifact lineage

All quantized GGUF files were produced directly from the same F16 GGUF reference.

| Format | Quantizer-reported size | Effective BPW | SHA-256 |
|---|---:|---:|---|
| F16 | 2944.68 MiB | 16.00 | `5fd2f789361b174b9ee441bc6ff0e559f67897ce0e04e076fe24f9b8e5fea571` |
| Q8_0 | 1564.62 MiB | 8.50 | `cd19237a5bdfdb9fd236b68723a584c449cb5a6d9f50d1b2db83ada731297f3f` |
| Q6_K | 1208.10 MiB | 6.56 | `fb8070e0a200100bf2cf92d8f6caa02d949908d427b513d6d3cbd87307aa5faa` |
| Q4_K_M | 934.69 MiB | 5.08 | `f97a321eeccfa912c2a0258d75aa804eaba4c8fc625a9bfbe8ff91e7c0daa8ef` |
| Q4_0 | 885.97 MiB | 4.81 | `99298ebc6a12a61167a4a19d6f2206475b8171da8dad58ebc8e7b5aaa3bece03` |

## Pipeline

```text
Pinned Hugging Face checkpoint
        ↓
convert_hf_to_gguf.py --outtype f16
        ↓
Qwen2.5-1.5B-F16.gguf
        ↓
llama-quantize
        ├── Q8_0
        ├── Q6_K
        ├── Q4_K_M
        └── Q4_0

The quantized formats were not requantized from one another.
