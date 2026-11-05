"""Math and data-integrity tests for vramfit.

Everything here runs offline. The formula under test, in plain
English:

  total_needed = weights + kv_cache + runtime_overhead
  weights_gb   = params_b * bits_per_weight / 8
  kv_cache     = 2 * layers * kv_head_dim * kv_bytes * context
"""

import pytest

from vramfit.calculator import (
    DECODE_EFFICIENCY,
    UnknownKvDtype,
    cpu_from_spec,
    estimate,
    fits,
    gpu_from_spec,
    kv_cache_gb,
    kv_gb_per_token,
    max_context_that_fits,
    pick_quant_and_context,
    speed_report,
    swap_needed,
    tokens_per_second,
    weights_gb,
)
from vramfit.database import load_database

DB = load_database()


# --- weight math ---------------------------------------------------

def test_weights_gb_uses_bits_per_weight():
    llama = DB.models["llama-3.1-8b"]
    q4 = DB.quants_by_id["q4"]
    assert weights_gb(llama.params_b, q4) == pytest.approx(8.03 * 4.7 / 8)


def test_fp16_is_two_bytes_per_weight():
    quant = DB.quants_by_id["fp16"]
    assert weights_gb(1.0, quant) == pytest.approx(2.0)


def test_quant_order_strictly_larger_means_smaller_weights():
    sizes = [
        weights_gb(7.0, q) for q in DB.quants
    ]
    assert sizes == sorted(sizes, reverse=True)


def test_unknown_kv_dtype_raises():
    llama = DB.models["llama-3.1-8b"]
    with pytest.raises(UnknownKvDtype):
        kv_gb_per_token(llama, "q3-bogus")


# --- kv cache math -------------------------------------------------

def test_kv_cache_per_token_is_linear_in_layers_and_kv_dim():
    llama = DB.models["llama-3.1-8b"]
    per_token = kv_gb_per_token(llama)
    expected_bytes = 2 * 32 * 1024 * 2
    assert per_token == pytest.approx(expected_bytes / 1e9)


def test_kv_cache_scales_linearly_with_context():
    llama = DB.models["llama-3.1-8b"]
    small = kv_cache_gb(llama, 1024)
    big = kv_cache_gb(llama, 8192)
    assert big == pytest.approx(small * 8)


def test_q8_kv_cache_is_half_of_fp16():
    llama = DB.models["llama-3.1-8b"]
    assert kv_gb_per_token(llama, "q8") == pytest.approx(
        kv_gb_per_token(llama, "fp16") / 2)


# --- fit verdict ---------------------------------------------------

def make_model(params_b=8.0, layers=32, kv_head_dim=1024, max_context=4096):
    from vramfit.database import Model
    return Model(
        id="toy", name="Toy", family="toy", params_b=params_b,
        layers=layers, kv_head_dim=kv_head_dim, max_context=max_context,
    )


def test_fit_at_exactly_usable_memory_counts_as_fitting():
    model = make_model(params_b=8.0, layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=4.8)
    hardware = gpu_from_spec("test gpu", 10.0, 0.0)
    assert hardware.usable_gb == pytest.approx(9.5)
    assert est.total_gb == pytest.approx(hardware.usable_gb)
    assert fits(est, hardware) is True


def test_fit_rejects_one_byte_over():
    model = make_model(params_b=8.0, layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=4.8 + 1e-7)
    assert not fits(est, gpu_from_spec("test gpu", 10.0, 0.0))


def test_overhead_is_always_part_of_the_total():
    model = make_model(layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    tiny = estimate(model, quant, 0, "fp16", overhead_gb=0.5)
    bigger = estimate(model, quant, 0, "fp16", overhead_gb=2.5)
    assert pytest.approx(bigger.total_gb - tiny.total_gb) == 2.0


def test_gpu_headroom_leaves_safety_margin():
    hardware = gpu_from_spec("test gpu", 12.0, 0.0)
    assert hardware.usable_gb == pytest.approx(12.0 * 0.95)


def test_swap_counts_but_at_a_discount():
    ram_cpu = cpu_from_spec("ram only", 16.0, 0.0, 50.0)
    swappy = cpu_from_spec("with swap", 16.0, 8.0, 50.0)
    assert swappy.usable_gb > ram_cpu.usable_gb
    assert swappy.usable_gb == pytest.approx((16 + 8 * 0.85) * 0.9)


# --- suggested context ---------------------------------------------

def test_suggested_context_is_the_longest_that_fits():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    ladder = [512, 2048, 8192, 32768]
    hardware = gpu_from_spec("test gpu", 6.1 / 0.95, 0.0)
    context = max_context_that_fits(model, quant, hardware, ladder,
                                    overhead_gb=1.2)
    assert context == 512


def test_suggested_context_never_exceeds_model_max():
    model = DB.models["mistral-7b"]
    quant = DB.quants_by_id["q4"]
    ladder = [512, 32768, 131072]
    huge_gpu = gpu_from_spec("monster", 1000.0, 0.0)
    assert max_context_that_fits(model, quant, huge_gpu, ladder) == 32768


def test_pinned_context_acts_as_ceiling():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    ladder = [512, 2048, 8192, 32768]
    hardware = gpu_from_spec("test gpu", 100.0, 0.0)
    context = max_context_that_fits(model, quant, hardware, ladder,
                                    start_context=2048)
    assert context == 2048


def test_returns_none_when_nothing_fits():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    hardware = gpu_from_spec("tiny gpu", 0.5, 0.0)
    assert max_context_that_fits(model, quant, hardware, [512, 2048]) is None


def test_auto_pick_prefers_the_best_quant_reaching_min_context():
    model = DB.models["llama-3.1-8b"]
    hardware = gpu_from_spec("test gpu", 11.7, 0.0)
    quant, context = pick_quant_and_context(
        model, DB.quants, hardware, [512, 2048, 8192], min_context=2048)
    assert quant.id == "q8"
    assert context >= 2048


def test_auto_pick_falls_back_to_longest_context():
    model = DB.models["llama-3.1-8b"]
    tiny = cpu_from_spec("tiny", 1.0, 0.0, 10.0)
    quant, context = pick_quant_and_context(
        model, DB.quants, tiny, [512, 1024], min_context=4096)
    assert quant.id == DB.quants[-1].id
    assert context is None


# --- speed estimate -------------------------------------------------

def test_speed_scales_with_bandwidth_and_inverse_weights():
    model = make_model(layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=0.0)
    hw = gpu_from_spec("test gpu", 100.0, 400.0)
    assert tokens_per_second(hw, est) == pytest.approx(
        DECODE_EFFICIENCY * 400.0 / est.weights_gb)


def test_zero_bandwidth_zero_speed():
    model = make_model(layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=0.0)
    hw = gpu_from_spec("no report", 100.0, 0.0)
    assert tokens_per_second(hw, est) == 0.0
    assert speed_report(hw, est).grade == "unknown"


def test_grinding_grade_warns():
    model = make_model(params_b=100.0, layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=0.0)
    hw = cpu_from_spec("slow box", 64.0, 0.0, 1.0)
    report = speed_report(hw, est)
    assert report.grade == "grinding"
    assert report.slow is True
    assert report.warnings


def test_swap_used_only_for_cpu_kind():
    model = make_model(layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=0.0)
    big_gpu = gpu_from_spec("big gpu", 1000.0, 100.0)
    big_cpu = cpu_from_spec("big cpu", 1000.0, 1000.0, 100.0)
    assert swap_needed(big_gpu, est) == 0.0
    assert swap_needed(big_cpu, est) == 0.0


def test_swap_needed_counts_overflow():
    model = make_model(params_b=40.0, layers=0, kv_head_dim=0)
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 0, "fp16", overhead_gb=0.0)
    hw = cpu_from_spec("test box", 16.0, 64.0, 50.0)
    spill = swap_needed(hw, est)
    assert spill > 0
    assert spill <= hw.swap_gb * 0.85


# --- real database integration --------------------------------------

def test_every_model_estimates_positive_weight_and_kv():
    for model in DB.models.values():
        quant = DB.quants_by_id["q4"]
        est = estimate(model, quant, model.max_context)
        assert est.weights_gb > 0
        assert est.kv_gb > 0
        assert est.total_gb > est.weights_gb


def test_llama_8b_q4_8k_context_known_value():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 8192, "fp16", 1.2)
    assert est.weights_gb == pytest.approx(8.03 * 4.7 / 8)
    assert est.kv_gb == pytest.approx(8192 * 131072 / 1e9)
    assert est.total_gb == pytest.approx(8.03 * 4.7 / 8 + 1.073741824 + 1.2)


def test_llama_8b_q4_fits_on_rtx_3060_and_4060():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    for gpu_id in ("rtx-3060", "rtx-4060"):
        gpu = DB.gpus[gpu_id]
        hw = gpu_from_spec(gpu.name, gpu.vram_gb, gpu.memory_bandwidth_gbs)
        est = estimate(model, quant, 8192, "fp16", DB.overheads.gpu_gb)
        assert fits(est, hw)


def test_llama_8b_fp16_at_full_context_blows_past_24_gb():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["fp16"]
    est = estimate(model, quant, 131072, "fp16", DB.overheads.gpu_gb)
    hw = gpu_from_spec("24gb", 24.0, 0.0)
    assert not fits(est, hw)


def test_database_realistic_smoke():
    assert len(DB.models) >= 12
    assert "llama-3.1-8b" in DB.models
    assert any(q.id == "q4" for q in DB.quants)
    ladder = DB.context_ladder
    assert ladder == sorted(ladder)
    assert 4096 in ladder


def test_speed_for_rtx_3060_q4_8k_is_fast_enough():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 8192, "fp16", DB.overheads.gpu_gb)
    gpu = DB.gpus["rtx-3060"]
    hw = gpu_from_spec(gpu.name, gpu.vram_gb, gpu.memory_bandwidth_gbs)
    report = speed_report(hw, est)
    assert report.tokens_per_second > 10
    assert not report.slow


def test_cpu_16gb_runs_8b_q4_without_swap():
    model = DB.models["llama-3.1-8b"]
    quant = DB.quants_by_id["q4"]
    est = estimate(model, quant, 8192, "fp16", DB.overheads.cpu_gb)
    cpu = DB.cpus["cpu-16gb"]
    hw = cpu_from_spec(cpu.name, cpu.ram_gb, cpu.swap_gb,
                       cpu.memory_bandwidth_gbs)
    assert fits(est, hw)
    assert swap_needed(hw, est) == 0.0
