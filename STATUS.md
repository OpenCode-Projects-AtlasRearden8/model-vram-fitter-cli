# STATUS.md - vramfit

## What was built

**vramfit** - an offline CLI that answers "can I run this model?" for
local LLMs. It reads a small editable TOML database of common open
models plus per-quantization overhead, does simple, honest memory
math, and reports:

- a plain-English fit verdict for a GPU (VRAM) or CPU mode (RAM + swap)
- the largest context length that fits (walks a ladder of common sizes)
- a suggested quantization (auto mode picks best quality that fits)
- a bandwidth-based speed estimate (tokens/sec) with warnings, including
  a strong warning when the model would land on disk swap

The formula, plainly: `total = weights + kv_cache + runtime_overhead`,
where `weights = params_b x bits_per_weight / 8` and the KV cache scales
linearly with context and depends on layers/KV-heads (GQA-aware).
Safety headroom: 95% of VRAM, 90% of RAM+swap, 85% of swap usable.

Stack: Python 3.11+ standard library only (`tomllib`, `argparse`,
`dataclasses`, `json`, `pathlib`). pytest is the only dev dependency
and is used exclusively for the math tests.

## Layout

- `vramfit/calculator.py` - all fitting math (weights, KV cache, verdicts,
  context suggestion, speed/swap reports)
- `vramfit/database.py` - offline TOML loading with user-file override
- `vramfit/cli.py` - argparse CLI (`list`, `size`, `check`, `version`)
- `vramfit/data/models.toml` - 15 built-in models (Llama 3.x, Qwen 2.5,
  Mistral, Gemma 2, Phi 3, DeepSeek R1 distill) plus quant table and
  overhead defaults
- `vramfit/data/hardware.toml` - 10 GPUs and 4 CPU presets
- `tests/test_math.py`, `tests/test_cli.py` - 43 offline tests
- `README.md`, `LICENSE` (MIT, Copyright (c) 2026 Asdale Dynamics, LLC)

## How to run it

```sh
python3 -m vramfit check llama-3.1-8b --gpu rtx-3060     # GPU fit
python3 -m vramfit check llama-3.1-8b --cpu cpu-16gb     # CPU mode
python3 -m vramfit check llama-3.1-8b --vram 10 --bandwidth 500   # generic GPU
python3 -m vramfit check llama-3.2-3b --ram 16 --swap 8           # generic CPU
python3 -m vramfit size mistral-7b --quant q4 --context 8192      # raw numbers
python3 -m vramfit list models|gpus|cpus|quants
```

Installer optional: `pip install -e .` gives a `vramfit` command.

Extending: drop a `models.toml` and/or `hardware.toml` in the working
directory (entries override built-ins) or pass `--data-dir DIR`.
Required model fields: `params_b`, `layers`, `kv_head_dim`,
`max_context`.

## Verification results (2026-11-05)

- `python3 -m pytest`: **43 passed** (math, data integrity, CLI smoke,
  user-override merge, JSON output, verdicts, error paths).
- `pip install -e .` with system setuptools: builds and installs; the
  `vramfit` console script and `python -m vramfit` both work; package
  data (TOML databases) resolves correctly through the installed path.
- Manual CLI smoke runs verified: YES verdicts (Llama 8B q8 on RTX
  3060, ~33.8 tok/s), swap warning path (Qwen 32B on 16 GB CPU),
  NO verdict + suggestions (fp16 8B on RTX 3060; Llama 70B on RTX
  3060 exits 1 with "nothing fits"), q8-KV cache halving, `--json`
  output, unknown model/GPU errors (exit 2).
- LICENSE copyright line verified byte-for-byte as
  `Copyright (c) 2026 Asdale Dynamics, LLC`; README credits
  Asdale Dynamics, LLC.
- Secret scan: no API keys, secrets, or personal data in any file.

## Known gaps

- OpenAI-compatible partial GPU offload (split layers between VRAM and
  RAM) is not modeled; it reports all-GPU or all-CPU verdicts only.
- Prefill speed, batching, vision/multimodal components, and MoE
  expert-offloading are not modeled; estimates assume chat-style
  single-stream decode.
- Bits-per-weight figures are effective approximations of GGUF-style
  quants; exact file sizes vary by runtime (llama.cpp, ollama, etc.).
- Speed is bandwidth-derived, not benchmarked (by design - that is
  mini-model-bench-cli's job); numbers have a rough +/-30% character.
- Auto quant pick on CPU avoids swap first; a user could still prefer
  a higher quality quant with swap via explicit `--quant`.
- Only a curated model list ships; anything not in models.toml must be
  added by hand (that is the intended extension point).
