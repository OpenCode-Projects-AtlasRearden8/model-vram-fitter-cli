"""Offline CLI answering "can I run this model?" - plain English, no
network, no accounts, no keys.

Usage highlights:
  python -m vramfit list models|gpus|cpus|quants
  python -m vramfit check llama-3.1-8b --gpu rtx-3060
  python -m vramfit check llama-3.1-8b --cpu cpu-16gb
  python -m vramfit check llama-3.1-8b --vram 12 --bandwidth 360
  python -m vramfit check llama-3.1-8b --ram 16 --swap 8 --bandwidth 55
  python -m vramfit size llama-3.1-8b --quant q4 --context 8192
"""

import argparse
import json
import sys
from pathlib import Path

from . import __version__
from .database import DatabaseError, load_database
from .calculator import (
    KV_BITS,
    UnknownKvDtype,
    clamp_context,
    cpu_from_spec,
    estimate,
    gpu_from_spec,
    max_context_that_fits,
    pick_quant_and_context,
    speed_report,
    swap_needed,
)


def _fmt(gb: float) -> str:
    return f"{gb:.1f}"


def _get_model(db, model_id: str):
    if model_id in db.models:
        return db.models[model_id]
    raise DatabaseError(
        f"unknown model '{model_id}'. Try 'python -m vramfit list models' "
        "or add it to models.toml."
    )


def _resolve_hardware(db, args):
    picked = sum(1 for value in
                 (args.gpu, args.cpu, args.vram, args.ram) if value)
    if picked > 1:
        raise DatabaseError("pick ONE of: --gpu, --cpu, --vram, or --ram")
    if args.gpu:
        if args.gpu not in db.gpus:
            raise DatabaseError(
                f"unknown GPU '{args.gpu}'. Try 'python -m vramfit list gpus'."
            )
        gpu = db.gpus[args.gpu]
        return gpu_from_spec(gpu.name, gpu.vram_gb, gpu.memory_bandwidth_gbs,
                             db.headroom.gpu_fraction)
    if args.cpu:
        if args.cpu not in db.cpus:
            raise DatabaseError(
                f"unknown CPU preset '{args.cpu}'. Try 'python -m vramfit list cpus'."
            )
        cpu = db.cpus[args.cpu]
        return cpu_from_spec(cpu.name, cpu.ram_gb, cpu.swap_gb, cpu.memory_bandwidth_gbs,
                             db.headroom.cpu_fraction, db.headroom.swap_fraction)
    if args.vram:
        bw = args.bandwidth if args.bandwidth else 0.0
        return gpu_from_spec("generic GPU", args.vram, bw, db.headroom.gpu_fraction)
    if args.ram or args.swap:
        bw = args.bandwidth if args.bandwidth else 0.0
        ram = args.ram if args.ram else 8.0
        sw = args.swap if args.swap else 0.0
        return cpu_from_spec("your machine", ram, sw, bw,
                             db.headroom.cpu_fraction, db.headroom.swap_fraction)
    raise DatabaseError(
        "tell me what to fit on: --gpu ID, --cpu ID, --vram N, or --ram N"
    )


def _resolve_quant(db, args):
    if args.quant == "auto":
        return None
    if args.quant not in db.quants_by_id:
        raise DatabaseError(
            f"unknown quant '{args.quant}'. Known: {', '.join(q.id for q in db.quants)}"
        )
    return db.quants_by_id[args.quant]


def _resolve_context(db, model, args):
    if args.context is None:
        return None
    if args.context <= 0:
        raise DatabaseError("--context must be a positive number of tokens")
    return clamp_context(model, args.context)


def _kv_dtype(args):
    if args.kv not in KV_BITS:
        raise UnknownKvDtype(args.kv)
    return args.kv


def _overhead_for(hw, db):
    return db.overheads.gpu_gb if hw.kind == "gpu" else db.overheads.cpu_gb


def cmd_list(args) -> int:
    db = load_database(args.data_dir)
    what = args.what
    if what in ("models", "model"):
        print(f"Models in the offline database ({len(db.models)}):")
        for m in sorted(db.models.values(), key=lambda x: x.params_b):
            print(f"  {m.id:<28} {m.name:<28} {m.params_b:>5.2f} B"
                  f"  ctx up to {m.max_context}")
    elif what == "gpus":
        print("Known GPUs (add yours in hardware.toml):")
        for gpu in db.gpus.values():
            print(f"  {gpu.id:<18} {_fmt(gpu.vram_gb):>4} GB  {gpu.name}")
    elif what == "cpus":
        print("CPU presets (add yours in hardware.toml):")
        for cpu in db.cpus.values():
            print(f"  {cpu.id:<18} {_fmt(cpu.ram_gb):>4} GB RAM + "
                  f"{_fmt(cpu.swap_gb)} GB swap  {cpu.name}")
    elif what == "quants":
        print("Quant sizes (bits per weight):")
        for quant in db.quants:
            print(f"  {quant.id:<4} {quant.bits_per_weight:>4.1f} bits "
                  f"- {quant.quality}")
    else:
        raise DatabaseError(f"don't know how to list '{what}'")
    return 0


def cmd_size(args) -> int:
    db = load_database(args.data_dir)
    model = _get_model(db, args.model)
    quant = _resolve_quant(db, args)
    if quant is None:
        quant = db.quants_by_id.get("q4", db.quants[0])
    context = _resolve_context(db, model, args) or model.max_context
    kv_dtype = _kv_dtype(args)
    ovh = db.overheads.gpu_gb
    est = estimate(model, quant, context, kv_dtype, ovh)
    if args.json:
        print(json.dumps({"model": model.id, "quant": quant.id,
                          "context": context, "kv_dtype": kv_dtype,
                          **est.as_dict()}, indent=2))
        return 0
    print(f"{model.name} at {quant.id} ({quant.quality})")
    print(f"  weights:        {_fmt(est.weights_gb)} GB")
    print(f"  kv cache:       {_fmt(est.kv_gb)} GB at {context} tokens "
          f"({_fmt(est.kv_gb_per_token * 1e6)} MB/token, {kv_dtype} KV)")
    print(f"  runtime extra:  {_fmt(est.overhead_gb)} GB "
          "(driver, compute buffers)")
    print(f"  total needed:   {_fmt(est.total_gb)} GB")
    return 0


def cmd_check(args) -> int:
    db = load_database(args.data_dir)
    model = _get_model(db, args.model)
    hw = _resolve_hardware(db, args)
    quant = _resolve_quant(db, args)
    kv_dtype = _kv_dtype(args)
    ovh = _overhead_for(hw, db)
    pinned_context = _resolve_context(db, model, args)
    ladder = db.context_ladder

    if quant is None:
        quant, suggested = pick_quant_and_context(
            model, db.quants, hw, ladder, kv_dtype, ovh, args.min_context,
            hard_limit_gb=hw.ram_budget_gb if hw.uses_swap else None)
        if suggested is None:
            print(f"Honestly? Nothing in the book fits {model.name} on this "
                  "hardware, even at the smallest quant and smallest context "
                  "in the ladder.")
            return 1
        context = suggested
        auto_note = True
    else:
        context = pinned_context if pinned_context is not None else model.max_context
        auto_note = False

    est = estimate(model, quant, context, kv_dtype, ovh)
    spill = swap_needed(hw, est)
    speed = speed_report(hw, est, spill if hw.uses_swap else None)
    verdict = est.total_gb <= hw.usable_gb

    if args.json:
        print(json.dumps({
            "model": model.id, "quant": quant.id, "context": context,
            "kv_dtype": kv_dtype, "hardware": hw.name, "kind": hw.kind,
            "needed_gb": est.total_gb, "usable_gb": hw.usable_gb,
            "fits": verdict, "suggested_quant": quant.id,
            "suggested_context": context,
            "swap_used_gb": spill, "tokens_per_second": round(speed.tokens_per_second, 2),
            "speed_grade": speed.grade, "warnings": list(speed.warnings),
            **est.as_dict(),
        }, indent=2))
        return 0 if verdict else 1

    hw_line = f"{hw.name}"
    quality = quant.quality
    print()
    print(f"{model.name} on {hw_line}")
    print(f"  quant:      {quant.id} ({quality})")
    print(f"  context:    {context} tokens"
          + (f" (model tops out at {model.max_context})" if context < model.max_context else ""))
    print(f"  needed:     {_fmt(est.total_gb)} GB "
          f"(weights {_fmt(est.weights_gb)} + kv cache {_fmt(est.kv_gb)}"
          f" + overhead {_fmt(est.overhead_gb)})")
    print(f"  available:  {_fmt(hw.usable_gb)} GB "
          f"{'VRAM' if hw.kind == 'gpu' else 'RAM+swap'} with safety headroom")
    print()
    if hw.kind == "gpu":
        if verdict:
            spare = hw.usable_gb - est.total_gb
            print(f"  Verdict: YES - it fits with about {_fmt(spare)} GB to spare.")
        else:
            print("  Verdict: NO - this setup does not fit in VRAM as-is.")
            print("  Try: a smaller quant (python -m vramfit check "
                  f"{model.id} --quant q4 --gpu <id>), a shorter "
                  "--context, or CPU mode:")
            print(f"  python -m vramfit check {model.id} --cpu cpu-32gb")
    else:
        if verdict:
            if spill > 0:
                print(f"  Verdict: YES, with a catch - about {_fmt(spill)} GB "
                      "spills past RAM and lives on swap.")
            else:
                print("  Verdict: YES - it fits in RAM with room to breathe.")
        else:
            print("  Verdict: NO - even RAM plus swap is not enough here.")
            print("  Try a smaller model, quant, or context.")
    print()
    if speed.warnings:
        for warning in speed.warnings:
            print(f"  Warning: {warning}")
    if speed.tokens_per_second > 0:
        print(f"  Speed (estimate): ~{speed.tokens_per_second:.1f} tokens/sec "
              f"- {speed.grade} for chat-style use.")
    else:
        print("  Speed (estimate): unknown - no bandwidth on file for "
              "this hardware. The fit math above still stands.")
    if auto_note:
        print()
        print(f"  (quant and context auto-picked; pin your own with "
              f"--quant and --context, min context {args.min_context})")
    print()
    return 0 if verdict else 1


def cmd_version(args) -> int:
    print(f"vramfit {__version__} - offline model-RAM fitter")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vramfit",
        description="Offline 'can I run this model?' fitter. Pure local math: "
                    "weights + kv cache + overhead vs your VRAM/RAM, plus a "
                    "suggested context and a plain-English speed warning.",
    )
    parser.add_argument("--data-dir", type=Path, default=None,
                        help="folder holding your own models.toml / hardware.toml")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list", help="show the offline database")
    list_p.add_argument("what", help="models | gpus | cpus | quants")
    list_p.set_defaults(func=cmd_list)

    size_p = sub.add_parser("size", help="show a model's size at a quant/context")
    size_p.add_argument("model")
    size_p.add_argument("--quant", default="q4")
    size_p.add_argument("--context", type=int, default=None)
    size_p.add_argument("--kv", default="fp16",
                        help="key/value cache dtype: fp16 (default) or q8")
    size_p.add_argument("--json", action="store_true")
    size_p.set_defaults(func=cmd_size)

    check_p = sub.add_parser("check", help="answer: can I run this model?")
    check_p.add_argument("model")
    hw_group = check_p.add_argument_group("hardware (pick one)")
    hw_group.add_argument("--gpu", help="GPU id from hardware.toml")
    hw_group.add_argument("--cpu", help="CPU preset id from hardware.toml")
    hw_group.add_argument("--vram", type=float, help="generic GPU: VRAM in GB")
    hw_group.add_argument("--ram", type=float, help="generic CPU-mode: RAM in GB")
    hw_group.add_argument("--swap", type=float, default=None,
                          help="generic CPU-mode: swap in GB")
    hw_group.add_argument("--bandwidth", type=float, default=None,
                          help="memory bandwidth in GB/s for the speed estimate")
    check_p.add_argument("--quant", default="auto",
                         help="fp16|q8|q6|q5|q4|q3|q2, or 'auto'")
    check_p.add_argument("--context", type=int, default=None,
                         help="pin a context length (defaults to what fits)")
    check_p.add_argument("--min-context", type=int, default=2048,
                         help="auto mode: stop searching once this context fits")
    check_p.add_argument("--kv", default="fp16",
                         help="key/value cache dtype: fp16 (default) or q8")
    check_p.add_argument("--json", action="store_true")
    check_p.set_defaults(func=cmd_check)

    ver_p = sub.add_parser("version", help="show the version")
    ver_p.set_defaults(func=cmd_version)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except DatabaseError as err:
        print(f"vramfit: {err}", file=sys.stderr)
        return 2
    except UnknownKvDtype:
        print("vramfit: unknown KV dtype (use fp16 or q8)", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
