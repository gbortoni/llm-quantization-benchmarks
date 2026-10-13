from pathlib import Path
import csv
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "results" / "performance_summary.csv"
PLOT_DIR = ROOT / "results" / "plots"

PLOT_DIR.mkdir(parents=True, exist_ok=True)

with SUMMARY.open() as f:
    rows = list(csv.DictReader(f))

formats = [r["format"] for r in rows]

size_mib = [
    float(r["file_size_bytes"]) / (1024 ** 2)
    for r in rows
]

prefill = [
    float(r["prefill_tok_s_median"])
    for r in rows
]

decode = [
    float(r["decode_tok_s_median"])
    for r in rows
]

perplexity = [
    float(r["perplexity"])
    for r in rows
]


# --------------------------------------------------
# 1. GGUF storage size
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(formats, size_mib)

plt.ylabel("GGUF size (MiB)")
plt.title("Qwen2.5-1.5B GGUF Storage by Quantization Format")

plt.tight_layout()
plt.savefig(
    PLOT_DIR / "storage_size.png",
    dpi=180,
)
plt.close()


# --------------------------------------------------
# 2. Prefill throughput
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(formats, prefill)

plt.ylabel("Median throughput (tokens/s)")
plt.title("Prefill Throughput by Quantization Format")

plt.tight_layout()
plt.savefig(
    PLOT_DIR / "prefill_throughput.png",
    dpi=180,
)
plt.close()


# --------------------------------------------------
# 3. Decode throughput
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.bar(formats, decode)

plt.ylabel("Median throughput (tokens/s)")
plt.title("Autoregressive Decode Throughput by Quantization Format")

plt.tight_layout()
plt.savefig(
    PLOT_DIR / "decode_throughput.png",
    dpi=180,
)
plt.close()


# --------------------------------------------------
# 4. Size vs quality
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(size_mib, perplexity)

for fmt, x, y in zip(formats, size_mib, perplexity):
    plt.annotate(
        fmt,
        (x, y),
        xytext=(6, 5),
        textcoords="offset points",
    )

plt.xlabel("GGUF size (MiB)")
plt.ylabel("WikiText-2 perplexity (lower is better)")
plt.title("Storage vs Predictive Quality")

plt.tight_layout()
plt.savefig(
    PLOT_DIR / "size_vs_perplexity.png",
    dpi=180,
)
plt.close()


# --------------------------------------------------
# 5. Decode speed vs quality
# --------------------------------------------------

plt.figure(figsize=(8, 5))
plt.scatter(decode, perplexity)

for fmt, x, y in zip(formats, decode, perplexity):
    plt.annotate(
        fmt,
        (x, y),
        xytext=(6, 5),
        textcoords="offset points",
    )

plt.xlabel("Median decode throughput (tokens/s)")
plt.ylabel("WikiText-2 perplexity (lower is better)")
plt.title("Decode Speed vs Predictive Quality")

plt.tight_layout()
plt.savefig(
    PLOT_DIR / "decode_vs_perplexity.png",
    dpi=180,
)
plt.close()


print("Wrote:")
for path in sorted(PLOT_DIR.glob("*.png")):
    print(f"  {path}")
