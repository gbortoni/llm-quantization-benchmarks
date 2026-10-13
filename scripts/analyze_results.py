from pathlib import Path
import csv
import re
import statistics

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "results" / "raw"
MODEL_DIR = ROOT / "models" / "gguf"
RESULTS_DIR = ROOT / "results"

FORMATS = {
    "F16": "Qwen2.5-1.5B-F16.gguf",
    "Q8_0": "Qwen2.5-1.5B-Q8_0.gguf",
    "Q6_K": "Qwen2.5-1.5B-Q6_K.gguf",
    "Q4_K_M": "Qwen2.5-1.5B-Q4_K_M.gguf",
    "Q4_0": "Qwen2.5-1.5B-Q4_0.gguf",
}


def parse_result_line(line):
    if not line.startswith("P3_RESULT"):
        return None

    result = {}

    for item in line.strip().split()[1:]:
        key, value = item.split("=", 1)
        result[key] = value

    return result


def parse_perplexity(path):
    text = path.read_text()

    match = re.search(
        r"Final estimate: PPL = ([0-9.]+)",
        text,
    )

    if not match:
        raise RuntimeError(
            f"Could not find final perplexity in {path}"
        )

    return float(match.group(1))


rows = []

# --------------------------------------------------
# Parse raw performance results
# --------------------------------------------------

for fmt, model_name in FORMATS.items():

    # -------------------------
    # Load-time trials
    # -------------------------

    load_dir = RAW_DIR / f"{fmt}_load"

    for run_file in sorted(load_dir.glob("run_*.txt")):
        for line in run_file.read_text().splitlines():
            result = parse_result_line(line)

            if result and result["workload"] == "load":
                run_match = re.search(r"run_(\d+)", run_file.name)

                if not run_match:
                    raise RuntimeError(
                        f"Could not determine run number from {run_file}"
                    )

                rows.append({
                    "format": fmt,
                    "workload": "load",
                    "run": int(run_match.group(1)),
                    "load_time_ms": float(result["load_time_ms"]),
                    "time_s": "",
                    "tok_s": "",
                    "prefill_time_s": "",
                    "decode_time_s": "",
                    "decode_tok_s": "",
                    "e2e_tok_s": "",
                })

    # -------------------------
    # Prefill trials
    # -------------------------

    prefill_file = RAW_DIR / f"{fmt}_prefill.txt"

    for line in prefill_file.read_text().splitlines():
        result = parse_result_line(line)

        if result and result["workload"] == "prefill":
            rows.append({
                "format": fmt,
                "workload": "prefill",
                "run": int(result["run"]),
                "load_time_ms": "",
                "time_s": float(result["time_s"]),
                "tok_s": float(result["tok_s"]),
                "prefill_time_s": "",
                "decode_time_s": "",
                "decode_tok_s": "",
                "e2e_tok_s": "",
            })

    # -------------------------
    # Decode trials
    # -------------------------

    decode_file = RAW_DIR / f"{fmt}_decode.txt"

    for line in decode_file.read_text().splitlines():
        result = parse_result_line(line)

        if result and result["workload"] == "decode":
            rows.append({
                "format": fmt,
                "workload": "decode",
                "run": int(result["run"]),
                "load_time_ms": "",
                "time_s": "",
                "tok_s": "",
                "prefill_time_s": float(result["prefill_time_s"]),
                "decode_time_s": float(result["decode_time_s"]),
                "decode_tok_s": float(result["decode_tok_s"]),
                "e2e_tok_s": float(result["e2e_tok_s"]),
            })


# --------------------------------------------------
# Preserve parsed run-level data
# --------------------------------------------------

parsed_path = RESULTS_DIR / "parsed_runs.csv"

fieldnames = [
    "format",
    "workload",
    "run",
    "load_time_ms",
    "time_s",
    "tok_s",
    "prefill_time_s",
    "decode_time_s",
    "decode_tok_s",
    "e2e_tok_s",
]

with parsed_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames,
    )
    writer.writeheader()
    writer.writerows(rows)


# --------------------------------------------------
# Build unified performance + quality summary
# --------------------------------------------------

summary_rows = []

for fmt, model_name in FORMATS.items():

    model_path = MODEL_DIR / model_name
    file_size_bytes = model_path.stat().st_size

    load_values = [
        row["load_time_ms"]
        for row in rows
        if row["format"] == fmt
        and row["workload"] == "load"
    ]

    prefill_values = [
        row["tok_s"]
        for row in rows
        if row["format"] == fmt
        and row["workload"] == "prefill"
    ]

    decode_values = [
        row["decode_tok_s"]
        for row in rows
        if row["format"] == fmt
        and row["workload"] == "decode"
    ]

    e2e_values = [
        row["e2e_tok_s"]
        for row in rows
        if row["format"] == fmt
        and row["workload"] == "decode"
    ]

    if not all(
        len(values) == 5
        for values in [
            load_values,
            prefill_values,
            decode_values,
            e2e_values,
        ]
    ):
        raise RuntimeError(
            f"{fmt}: expected exactly 5 measurements per metric"
        )

    perplexity_file = RAW_DIR / f"{fmt}_perplexity.txt"

    if not perplexity_file.exists():
        raise RuntimeError(
            f"Missing perplexity result: {perplexity_file}"
        )

    perplexity = parse_perplexity(perplexity_file)

    summary_rows.append({
        "format": fmt,
        "file_size_bytes": file_size_bytes,
        "load_ms_median": statistics.median(load_values),
        "load_ms_stdev": statistics.stdev(load_values),
        "prefill_tok_s_median": statistics.median(prefill_values),
        "prefill_tok_s_stdev": statistics.stdev(prefill_values),
        "decode_tok_s_median": statistics.median(decode_values),
        "decode_tok_s_stdev": statistics.stdev(decode_values),
        "e2e_tok_s_median": statistics.median(e2e_values),
        "perplexity": perplexity,
    })


# --------------------------------------------------
# Write summary CSV
# --------------------------------------------------

summary_path = RESULTS_DIR / "performance_summary.csv"

with summary_path.open("w", newline="") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=summary_rows[0].keys(),
    )
    writer.writeheader()
    writer.writerows(summary_rows)


# --------------------------------------------------
# Human-readable output
# --------------------------------------------------

print()

print(
    f"{'Format':<10}"
    f"{'Size MiB':>12}"
    f"{'Load ms':>12}"
    f"{'Prefill':>12}"
    f"{'Decode':>12}"
    f"{'E2E':>12}"
    f"{'PPL':>12}"
)

print("-" * 82)

for row in summary_rows:
    size_mib = row["file_size_bytes"] / (1024 ** 2)

    print(
        f"{row['format']:<10}"
        f"{size_mib:>12.2f}"
        f"{row['load_ms_median']:>12.2f}"
        f"{row['prefill_tok_s_median']:>12.2f}"
        f"{row['decode_tok_s_median']:>12.2f}"
        f"{row['e2e_tok_s_median']:>12.2f}"
        f"{row['perplexity']:>12.4f}"
    )

print()
print(f"Wrote {parsed_path}")
print(f"Wrote {summary_path}")
