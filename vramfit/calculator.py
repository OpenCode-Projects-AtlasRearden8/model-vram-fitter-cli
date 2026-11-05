"""The fitting math: weights, KV cache, overheads, fit verdicts,
context suggestion, and a bandwidth-based speed estimate.

All sizes are decimal GB. The model is deliberately simple so a
beginner can follow it:

  total_needed = weights + kv_cache + runtime_overhead

  weights_gb   = params_b * bits_per_weight / 8
  kv_gb/token  = 2 (K and V) * layers * kv_head_dim * kv_bits / 8 / 1e9
  overhead     = fixed reserve for the driver, CUDA context, and
                 compute buffers
"""

from dataclasses import dataclass
from typing import Optional

KV_BITS = {"fp16": 16.0, "q8": 8.0}

GPU_HEADROOM_FRACTION = 0.95
CPU_HEADROOM_FRACTION = 0.90
SWAP_HEADROOM_FRACTION = 0.85

DECODE_EFFICIENCY = 0.80


class UnknownQuant(Exception):
    pass


class UnknownKvDtype(Exception):
    pass


@dataclass(frozen=True)
class Estimate:
    weights_gb: float
    kv_gb_per_token: float
    kv_gb: float
    overhead_gb: float
    total_gb: float

    def as_dict(self):
        return dict(self.__dict__)


@dataclass(frozen=True)
class Hardware:
    kind: str
    name: str
    usable_gb: float
    ram_gb: float = 0.0
    swap_gb: float = 0.0
    memory_bandwidth_gbs: float = 0.0

    @property
    def uses_swap(self) -> bool:
        return self.kind == "cpu"

    @property
    def ram_budget_gb(self) -> float:
        """Smallest safe budget: for CPU mode, physical RAM only."""
        if self.kind == "cpu":
            return self.ram_gb * CPU_HEADROOM_FRACTION
        return self.usable_gb


def gpu_from_spec(name: str, vram_gb: float, bandwidth_gbs: float,
                  headroom: float = GPU_HEADROOM_FRACTION) -> Hardware:
    return Hardware(
        kind="gpu",
        name=name,
        usable_gb=vram_gb * headroom,
        ram_gb=vram_gb,
        memory_bandwidth_gbs=bandwidth_gbs,
    )


def cpu_from_spec(name: str, ram_gb: float, swap_gb: float, bandwidth_gbs: float,
                  headroom: float = CPU_HEADROOM_FRACTION,
                  swap_headroom: float = SWAP_HEADROOM_FRACTION) -> Hardware:
    return Hardware(
        kind="cpu",
        name=name,
        usable_gb=(ram_gb + swap_gb * swap_headroom) * headroom,
        ram_gb=ram_gb,
        swap_gb=swap_gb,
        memory_bandwidth_gbs=bandwidth_gbs,
    )


def weights_gb(params_b: float, quant) -> float:
    """Billions of parameters at ``quant.bits_per_weight`` -> GB."""
    return params_b * quant.bits_per_weight / 8.0


def kv_gb_per_token(model, kv_dtype: str = "fp16") -> float:
    """Key/value cache bytes for one context token, in GB."""
    if kv_dtype not in KV_BITS:
        raise UnknownKvDtype(kv_dtype)
    bytes_per_token = 2 * model.layers * model.kv_head_dim * (KV_BITS[kv_dtype] / 8.0)
    return bytes_per_token / 1e9


def kv_cache_gb(model, context: int, kv_dtype: str = "fp16") -> float:
    return kv_gb_per_token(model, kv_dtype) * context


def estimate(model, quant, context: int, kv_dtype: str = "fp16",
             overhead_gb: float = 1.2) -> Estimate:
    if isinstance(quant, str):
        raise UnknownQuant(quant)
    w = weights_gb(model.params_b, quant)
    kv = kv_cache_gb(model, context, kv_dtype)
    total = w + kv + overhead_gb
    return Estimate(
        weights_gb=w,
        kv_gb_per_token=kv_gb_per_token(model, kv_dtype),
        kv_gb=kv,
        overhead_gb=overhead_gb,
        total_gb=total,
    )


def fits(est: Estimate, hardware: Hardware) -> bool:
    return est.total_gb <= hardware.usable_gb


def clamp_context(model, context: Optional[int]) -> int:
    if context is None:
        return model.max_context
    return max(1, min(int(context), model.max_context))


def max_context_that_fits(model, quant, hardware: Hardware, ladder,
                          kv_dtype: str = "fp16", overhead_gb: float = 1.2,
                          start_context: Optional[int] = None,
                          budget_gb: Optional[float] = None) -> Optional[int]:
    """Largest context in ``ladder`` whose total fits, else None.

    ``budget_gb`` can tighten the fit check (e.g. physical RAM only so
    the model never lands on swap) without loosening it.
    """
    budget = clamp_context(model, start_context)
    usable = hardware.usable_gb
    if budget_gb is not None:
        usable = min(usable, budget_gb)
    options = sorted(ctx for ctx in set(ladder) | {budget} if ctx <= budget)
    for ctx in reversed(options):
        est = estimate(model, quant, ctx, kv_dtype, overhead_gb)
        if est.total_gb <= usable:
            return ctx
    return None


def pick_quant_and_context(model, quants, hardware: Hardware, ladder,
                           kv_dtype: str = "fp16", overhead_gb: float = 1.2,
                           min_context: int = 2048,
                           hard_limit_gb: Optional[float] = None) -> tuple:
    """Best (quant, context) for the hardware, best quality first.

    Walks qualities from best to smallest; a quant wins as soon as it
    fits at least ``min_context`` tokens. If a ``hard_limit_gb`` is
    given (physical RAM in CPU mode), setups that stay inside it are
    preferred first so the model does not land on swap; only when no
    quant manages that do we relax to the full usable budget.

    If nothing reaches ``min_context``, falls back to the setup with
    the longest suggested context.
    """
    def best_ctx(q, budget):
        return max_context_that_fits(model, q, hardware, ladder,
                                     kv_dtype, overhead_gb,
                                     budget_gb=budget)

    def budget_ok(budget):
        return budget is None or budget <= hardware.usable_gb

    if hard_limit_gb is not None and budget_ok(hard_limit_gb):
        for quant in quants:
            ctx = best_ctx(quant, hard_limit_gb)
            if ctx is not None and ctx >= min_context:
                return quant, ctx

    best = (None, None)
    for quant in quants:
        ctx = best_ctx(quant, None)
        if ctx is not None and ctx >= min_context:
            return quant, ctx
        if ctx is not None and (best[1] is None or ctx > best[1]):
            best = (quant, ctx)
    if best[0] is not None:
        return best
    smallest = quants[-1]
    return smallest, best_ctx(smallest, None)


def tokens_per_second(hardware: Hardware, est: Estimate) -> float:
    """Rough decode speed: tokens/s ~ bandwidth / bytes read per token.

    Decode is memory-bound: each token re-reads the weights once. The
    efficiency factor covers imperfect bandwidth utilization, the KV
    cache, embeddings, and small variable buffers.
    """
    if hardware.memory_bandwidth_gbs <= 0 or est.weights_gb <= 0:
        return 0.0
    return DECODE_EFFICIENCY * hardware.memory_bandwidth_gbs / est.weights_gb


@dataclass(frozen=True)
class SpeedReport:
    tokens_per_second: float
    grade: str
    warnings: tuple = ()

    @property
    def slow(self) -> bool:
        return self.grade in ("slow", "grinding")


def speed_report(hardware: Hardware, est: Estimate,
                 swap_used_gb: Optional[float] = None) -> SpeedReport:
    tok_s = tokens_per_second(hardware, est)
    warnings = []
    if hardware.kind == "cpu" and swap_used_gb is not None and swap_used_gb > 0:
        tok_s *= 0.05
        spill = swap_used_gb
        warnings.append(
            f"needs ~{spill:.1f} GB of swap; the model spills into disk swap, "
            "so expect near-frozen generation. Drop the quant or context instead."
        )
    if tok_s >= 20:
        grade = "snappy"
    elif tok_s >= 10:
        grade = "usable"
    elif tok_s >= 4:
        grade = "slow but readable"
    elif tok_s > 0:
        grade = "grinding"
        if not warnings:
            warnings.append("speed is painful; consider a smaller quant or model")
    else:
        grade = "unknown"
        warnings.append("no bandwidth figure for this hardware; speed unknown")
    return SpeedReport(tok_s, grade, tuple(warnings))


def swap_needed(hardware: Hardware, est: Estimate,
                swap_fraction: float = SWAP_HEADROOM_FRACTION) -> float:
    """How much of the model spills past usable physical RAM (GB)."""
    if not hardware.uses_swap:
        return 0.0
    overflow = est.total_gb - hardware.ram_gb * CPU_HEADROOM_FRACTION
    if overflow <= 0:
        return 0.0
    return min(overflow, hardware.swap_gb * swap_fraction)
