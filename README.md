# vramfit - "Can I run this model?" fitter

An offline command-line tool that answers the question everyone asks
when they first meet local AI: **can my machine actually run this
model?**

It reads a small, human-editable TOML database of common open models
(parameter counts, layer/attention layout) plus per-quantization
overhead, does simple honest math, and tells you:

- whether the model fits on a GPU (VRAM) or in CPU mode (RAM + swap)
- the largest context length that fits
- a suggested quantization (fp16/q8/q6/q5/q4/q3/q2)
- a rough speed estimate with a plain-English warning when it will hurt

No network. No accounts. No telemetry. No API keys. Python standard
library only (3.11+ for `tomllib`). Part of the beginner-friendly AI
toolkit from **Asdale Dynamics, LLC**.

## Install

Nothing to install if you have Python 3.11+:

```sh
python3 -m vramfit check llama-3.1-8b --gpu rtx-3060
```

Or, if you prefer a real entry point:

```sh
pip install -e .
vramfit check llama-3.1-8b --gpu rtx-3060
```

## How the math works (plain English)

Three things consume memory when you run a model locally:

1. **Weights** - the model itself. Rule of thumb:
   `GB = billions of parameters x bits-per-weight / 8`. A 7B model at
   q4 (about 4.7 bits per weight) is therefore about 7 x 4.7 / 8, or
   roughly 4.1 GB.
2. **KV cache** - working memory for the conversation. Grows linearly
   with context length and depends on how many attention layers the
   model has (`2 x layers x kv_head_dim x bytes` per token).
3. **Runtime overhead** - the driver, compute buffers, the kitchen
   sink. A small fixed reserve (1.2 GB for GPU, 0.4 GB for CPU mode).

The tool adds those up, applies a safety headroom (95% of VRAM, 90% of
RAM+swap), and walks a ladder of common context lengths to find the
longest one that fits. Speed is estimated from your memory bandwidth,
because token generation is mostly a memory-bandwidth exercise:
roughly `tokens/sec = 0.8 x bandwidth / weights-GB`. If CPU mode would
push part of the model onto swap, it says so loudly, because swapping
LLMs is near-frozen.

## Examples

Fit check with a short, plain verdict:

```sh
$ vramfit check llama-3.1-8b --gpu rtx-3060

Llama 3.1 8B on NVIDIA RTX 3060 (12 GB)
  quant:      q8 (near-lossless)
  context:    8192 tokens (model tops out at 131072)
  needed:     10.8 GB (weights 8.5 + kv cache 1.1 + overhead 1.2)
  available:  11.4 GB VRAM with safety headroom

  Verdict: YES - it fits with about 0.6 GB to spare.
  Speed (estimate): ~33.8 tokens/sec - snappy for chat-style use.
```

CPU mode (RAM plus swap), with the swap warning doing its job:

```sh
$ vramfit check qwen2.5-32b --cpu cpu-16gb

  Warning: speed is painful; consider a smaller quant or model
  Speed (estimate): ~3.5 tokens/sec - grinding for chat-style use.
```

Handy extras:

```sh
vramfit size mistral-7b --quant q4 --context 8192   # just the numbers
vramfit check llama-3.1-8b --vram 10 --bandwidth 500  # generic GPU
vramfit check llama-3.2-3b --ram 16 --swap 8 --bandwidth 55  # generic CPU
vramfit check llama-3.1-8b --gpu rtx-3060 --kv q8   # halve the KV cache
vramfit check llama-3.1-8b --cpu cpu-16gb --json    # machine-readable
vramfit list models|gpus|cpus|quants
```

## Extending: your own models.toml / hardware.toml

Everything is data. To add a model or a GPU, either:

- drop a `models.toml` / `hardware.toml` in the folder you run from
  (entries there override built-ins), or
- point `--data-dir /some/folder` at your own set.

A model entry looks like:

```toml
[models."my-llama-13b"]
name = "My Llama 13B"
family = "llama"
params_b = 13.02          # billions of parameters
layers = 40               # transformer blocks
kv_head_dim = 1024        # KV heads x head dimension (GQA-aware)
max_context = 8192
```

Hardware entries look like:

```toml
[gpus."my-card"]
name = "My Card"
vram_gb = 10.0
memory_bandwidth_gbs = 500.0

[cpus."my-laptop"]
ram_gb = 16.0
swap_gb = 4.0
memory_bandwidth_gbs = 45.0
```

## Tests

The math is covered by pytest (stdlib app, pytest as the only dev
dependency):

```sh
python -m pytest
```

## The family these tools live in

- **vramfit** (this tool) - answers "can I run it?" (memory fit)
- **ollama-disk-doctor-cli** - answers "where did all my disk space go?"
  (disk space)
- **mini-model-bench-cli** - answers "how fast is it on my machine?"
  (speed)

Use them together: bench first if you have it, fitter for the "will it
even load" question.

## License

MIT License. Copyright (c) 2026 Asdale Dynamics, LLC. See
[LICENSE](LICENSE).

Made with plain English in mind - like a friend would explain it.
