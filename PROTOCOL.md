# Quantization Benchmark Protocol

## Status

This protocol is defined before collecting quantization benchmark results.

After the protocol is frozen, changes to the benchmark methodology must be documented explicitly rather than silently modifying the experiment based on observed results.

## Research Question

How does quantization format affect the storage, inference performance, and model quality of the same small language model under controlled conditions on an Apple M2 Pro?

## Model

Source model:

- `Qwen/Qwen2.5-1.5B`
- Source revision: `8faed761d45a263340a0528343f099c05c9a4323`

The source checkpoint will be converted to an F16 GGUF using the pinned llama.cpp revision.

The F16 GGUF will serve as the higher-precision reference and as the common source for all tested quantizations.

### Configurations

Reference:

- F16

Quantized formats:

- Q8_0
- Q6_K
- Q4_K_M
- Q4_0

## Experimental Variables

### Independent variable

The independent variable is:

- quantization format

### Controlled variables

The experiment will hold constant, where applicable:

- source model and revision
- llama.cpp revision
- hardware
- backend
- context size
- batch size
- thread count
- prompts
- decode-step count
- sampling configuration
- warm-up procedure
- number of measured runs
- timing methodology

### Dependent variables

The experiment will measure:

- GGUF file size
- model load time
- prefill throughput
- autoregressive decode throughput
- end-to-end throughput
- memory usage, if practical
- perplexity on the fixed quality corpus

## Benchmark Environment

### Hardware

Benchmarks will be run on:

- Apple M2 Pro
- 16 GB unified memory
- arm64 architecture

### Software

- Operating system: exact macOS version will be recorded at benchmark time
- llama.cpp commit:
  `e71b80510c848c00175924ecf3c40333ccae8eb5`
- Backend: Metal
- Compiler and build configuration will be recorded with the benchmark artifacts

The same llama.cpp revision will be used for model conversion, quantization, and inference benchmarking unless a deviation is explicitly documented.

### Runtime Configuration

The performance benchmark will use the following requested settings:

- Context size: 512 tokens
- Batch size: 256 tokens
- CPU threads: 6
- Backend: Metal

Requested and effective runtime settings will both be recorded when llama.cpp reports an effective value different from the requested value.

Backend placement/offload behavior must remain consistent across tested formats. Effective model placement will be verified so that backend differences do not become an uncontrolled variable.

### Sampling Configuration

Performance benchmarks will use fixed sampling settings:

- Seed: 42
- Top-k: 40
- Top-p: 0.9
- Temperature: 0.8

Sampling settings are held constant because sampling behavior is not the independent variable in this experiment.

## Performance Workloads

Performance benchmarking uses separate workloads for prompt processing and autoregressive decoding.

The workloads are intentionally separated so that prefill and decode performance can be measured independently rather than being hidden inside one combined generation result.

### Workload A — Prefill

Purpose:

Measure prompt-processing throughput under an identical multi-token input workload.

Configuration:

- Prompt file: `prompts/prefill_256.txt`
- Prompt length: exactly 256 tokens
- Decode output is not part of the measured workload
- The exact prompt text will remain unchanged after the protocol is frozen

Metric:

- prefill tokens/second

The same tokenizer/model family is used for all formats. The prompt token count will be verified before benchmark results are collected.

### Workload B — Autoregressive Decode

Purpose:

Measure single-token autoregressive generation throughput under an identical amount of decode work.

Configuration:

- Prompt file: `prompts/decode_32.txt`
- Prompt length: exactly 32 tokens
- Decode steps: exactly 128
- Context size: 512
- Sampling configuration: fixed according to the benchmark configuration

The decode prompt should naturally encourage a response longer than 128 tokens, but the benchmark does not rely on the model naturally reaching that length.

For the performance-only workload, EOG does not terminate the benchmark before 128 decode steps have been completed. This ensures that every tested format performs the same number of autoregressive decode iterations.

Tokens produced after a natural EOG are not interpreted as meaningful generated text and are not used for quality evaluation.

Metric:

- autoregressive decode tokens/second

### Separation From Quality Evaluation

The performance workloads are not used to judge model quality.

Quality is evaluated separately using the fixed perplexity corpus so that differences in generated length, EOG behavior, or instruction following do not change the amount of timed work.

## Benchmark Procedure

### General Rules

Each model format is benchmarked using the same procedure.

Performance measurements are collected only after the relevant execution path has been warmed up.

All raw measured runs are preserved. Summary statistics do not replace the underlying run-level data.

During benchmark execution, unnecessary background workloads should be avoided where practical.

### Model Loading

For inference-throughput benchmarks:

1. Load the model once.
2. Create the required runtime/context.
3. Perform warm-up.
4. Execute the measured trials.
5. Destroy the model/runtime after that format's benchmark is complete.

Model loading is not included in prefill or decode timing.

Load time is measured separately.

### Prefill Warm-up

Before measured prefill trials:

1. Run the fixed prefill workload once.
2. Synchronize the backend so asynchronous Metal work has completed.
3. Discard the warm-up timing.
4. Clear runtime/KV state.

The warm-up uses the same type of multi-token workload as the measured prefill benchmark.

### Prefill Measurement

For each format:

1. Begin from cleared runtime/KV state.
2. Process the fixed 256-token prompt.
3. Synchronize the backend before ending the timer.
4. Record elapsed time and throughput.
5. Clear runtime/KV state.
6. Repeat for a total of 5 measured runs.

All five raw results are retained.

### Decode Warm-up

Before measured decode trials:

1. Process the fixed decode prompt.
2. Execute at least one single-token autoregressive decode step.
3. Synchronize the backend.
4. Discard the warm-up timing.
5. Clear runtime/KV state.

This ensures that the autoregressive decode execution path is exercised before measurement.

### Decode Measurement

For each format:

1. Begin from cleared runtime/KV state.
2. Prefill the fixed 32-token decode prompt.
3. Begin decode timing after prompt processing is complete.
4. Execute exactly 128 autoregressive decode steps.
5. Do not terminate the performance workload early because of EOG.
6. Synchronize the backend before ending the timer.
7. Record decode elapsed time and decode throughput.
8. Clear runtime/KV state.
9. Repeat for a total of 5 measured runs.

All five raw results are retained.

### Run Summary

Performance summaries will primarily report the median of the 5 measured runs.

Raw measurements will remain available so runtime variability and noisy measurements can be inspected during Week 9.

### Load-Time Measurement

Model load time is measured separately from inference throughput.

Each load-time trial begins with the model not loaded and ends only after the model is ready for inference according to the benchmark implementation.

The exact load-time timing boundary will be documented in the benchmark harness before results are collected.

#### Harness Timing Boundary

Before load-time results are collected, the benchmark harness defines model
load time as the wall-clock duration of `llama_model_load_from_file()`.

The timer:

- starts immediately before calling `llama_model_load_from_file()`
- ends immediately after the call returns successfully
- excludes `ggml_backend_load_all()`
- excludes tokenization
- excludes context creation
- excludes warm-up and inference

Five load-time trials are collected per format using five separate process
invocations. Each trial therefore begins with no model loaded in that process.

The operating-system file cache is not explicitly flushed between trials.
All five raw measurements are retained, and the median is used for the
summary.

This metric describes llama.cpp model-loading latency under the benchmark
environment. It should not be interpreted as the time required to physically
read every model byte from storage.

## Metrics

All metrics must use the definitions below. Metric definitions will not be changed after benchmark results are observed unless the change is documented as a protocol revision.

### GGUF File Size

Definition:

The size in bytes of the final GGUF file used for benchmarking.

Recorded as:

- `file_size_bytes`

Human-readable MB/GB values may be derived for presentation, but the raw byte count is retained.

File size measures storage requirements of the complete GGUF artifact, not theoretical bits per parameter.

### Model Load Time

Definition:

Elapsed wall-clock time required to load the GGUF model and reach the benchmark-defined ready-for-inference state.

Recorded as:

- `load_time_ms`

Load time is measured separately from prefill and decode throughput.

The exact start and end points used by the harness must remain identical across all model formats.

### Prefill Throughput

Definition:

The number of prompt tokens processed during the measured prefill interval divided by elapsed prefill time.

For the fixed prefill workload:

- prompt tokens = 256

Calculated as:

`prefill_tok_s = prompt_tokens / prefill_time_seconds`

The timer covers prompt processing only.

Backend synchronization must complete before the timer stops.

### Autoregressive Decode Throughput

Definition:

The number of measured autoregressive decode steps divided by elapsed decode time.

For the fixed decode workload:

- measured decode steps = 128

Calculated as:

`decode_tok_s = decode_steps / decode_time_seconds`

Prompt prefill is completed before the decode timer begins.

Backend synchronization must complete before the timer stops.

The benchmark executes exactly 128 decode steps even if EOG is encountered earlier.

### End-to-End Generation Throughput

Definition:

A secondary metric for the fixed decode workload that includes both prompt prefill and autoregressive generation.

Calculated as:

`e2e_tok_s = decode_steps / (prefill_time_seconds + decode_time_seconds)`

This metric is reported separately from pure decode throughput.

Because it combines two execution phases, it must not replace the individual prefill and decode measurements when analyzing quantization behavior.

### Memory Usage

Memory usage will be collected only if a practical and reproducible measurement method is available.

If collected, the protocol must document:

- what memory quantity is being measured
- the measurement tool/API
- the measurement point
- the units

Memory measurements will not be presented without a clearly defined methodology.

### Quality — Perplexity

Quality will be evaluated using perplexity on the fixed WikiText-2 test corpus.

All tested formats will evaluate the same corpus using the same tokenizer and evaluation procedure.

Recorded as:

- `perplexity`

Interpretation:

- lower perplexity indicates better next-token predictive fit to the evaluation corpus
- higher perplexity indicates worse predictive fit to the evaluation corpus

The F16 model serves as the higher-precision reference.

Quantization quality degradation may be reported relative to the F16 perplexity.

Perplexity is not treated as a complete measure of:

- reasoning ability
- factuality
- instruction following
- human preference
- general model quality across every task

## Interpretation Rules

Benchmark conclusions must remain within the scope of the experiment.

### Direct Measurements

The following may be reported directly from the collected data:

- GGUF file size
- model load time
- prefill throughput
- autoregressive decode throughput
- end-to-end throughput
- memory usage, if collected using a documented method
- perplexity on the fixed evaluation corpus

Raw measurements must be preserved alongside summary statistics.

### Comparisons

Comparisons between formats must identify the metric being compared.

For example:

- "Q4_K_M had higher prefill throughput than Q6_K under this benchmark."
- "Q6_K had higher decode throughput than Q4_K_M under this benchmark."
- "Q8_0 produced perplexity closer to the F16 reference than Q4_0 on WikiText-2."

Avoid statements such as "Format X is faster" when different benchmark phases show different behavior.

### Quality Claims

Perplexity results describe next-token predictive fit on the fixed WikiText-2 evaluation corpus.

They do not by themselves establish differences in:

- human preference
- reasoning ability
- factuality
- instruction following
- quality on unrelated datasets or tasks

### Causal Explanations

Observed performance differences do not automatically establish their underlying cause.

Possible explanations such as:

- reduced memory traffic
- dequantization overhead
- kernel implementation differences
- cache behavior
- memory-bandwidth pressure

must be described as hypotheses unless supported by additional profiling or system-level evidence.

### Generalization

Results apply directly only to the tested combination of:

- Qwen2.5-1.5B
- tested quantization formats
- pinned llama.cpp revision
- Apple M2 Pro
- Metal backend
- frozen runtime configuration
- frozen benchmark workloads

The experiment does not establish that a particular quantization format is universally faster, smaller in the same proportion, or higher quality across other models, runtimes, hardware, or workloads.

### No Universal "Best" Format

No format will be labeled the universally best quantization.

Instead, conclusions will describe the measured tradeoffs between:

- storage
- performance
- quality

A preferred configuration can only be discussed relative to an explicitly stated objective or workload.

### Token-Count Verification

The frozen prompt files contain:

- `prefill_256.txt`: 256 tokens
- `decode_32.txt`: 32 tokens

These counts were verified using the tokenizer files from the pinned
`Qwen/Qwen2.5-1.5B` source revision with special-token insertion disabled.

Before benchmark execution, the counts will be cross-checked using the
pinned llama.cpp inference path. Any difference caused by BOS or other
runtime tokenization behavior will be documented before results are collected.