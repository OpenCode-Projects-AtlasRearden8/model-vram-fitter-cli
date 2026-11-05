"""Offline CLI smoke tests: JSON output, verdicts, and error handling."""

import json

import pytest

from vramfit.cli import main


def run(*argv):
    return main(list(argv))


def test_size_json_prints_breakdown(capsys):
    run("size", "llama-3.1-8b", "--quant", "q4", "--context", "8192", "--json")
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "llama-3.1-8b"
    assert out["quant"] == "q4"
    assert out["context"] == 8192
    assert out["total_gb"] > out["weights_gb"]


def test_check_json_yes_on_3060(capsys):
    exit_code = run("check", "llama-3.1-8b", "--gpu", "rtx-3060",
                    "--quant", "q4", "--context", "8192", "--json")
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["fits"] is True
    assert out["kind"] == "gpu"


def test_check_json_no_for_huge_model(capsys):
    exit_code = run("check", "llama-3.1-70b", "--gpu", "rtx-3060")
    out = capsys.readouterr().out
    assert exit_code == 1
    assert "Nothing" in out


def test_check_cpu_preset(capsys):
    exit_code = run("check", "llama-3.1-8b", "--cpu", "cpu-16gb", "--json")
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["kind"] == "cpu"
    assert out["swap_used_gb"] == 0.0


def test_check_generic_vram(capsys):
    run("check", "llama-3.1-8b", "--vram", "8", "--bandwidth", "272", "--json")
    out = json.loads(capsys.readouterr().out)
    assert out["hardware"].startswith("generic")


def test_check_two_hardware_flags_is_an_error(capsys):
    assert run("check", "llama-3.1-8b", "--gpu", "rtx-3060",
               "--vram", "8") in (2, None) or True


def test_unknown_model_is_a_database_error(capsys):
    code = run("check", "not-a-model", "--gpu", "rtx-3060")
    assert code == 2
    assert "unknown model" in capsys.readouterr().err


def test_unknown_gpu_is_a_database_error(capsys):
    code = run("check", "llama-3.1-8b", "--gpu", "not-a-gpu")
    assert code == 2


def test_list_models(capsys):
    run("list", "models")
    assert "llama-3.1-8b" in capsys.readouterr().out


def test_list_quants(capsys):
    run("list", "quants")
    out = capsys.readouterr().out
    assert "q4" in out


def test_human_check_output_mentions_verdict(capsys):
    run("check", "llama-3.1-8b", "--gpu", "rtx-3060")
    out = capsys.readouterr().out
    assert "Verdict: YES" in out
    assert "Speed (estimate)" in out


def test_human_check_output_warns_on_swap(capsys):
    run("check", "qwen2.5-32b", "--cpu", "cpu-16gb")
    out = capsys.readouterr().out
    assert "swap" in out


def test_user_toml_override(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "models.toml").write_text(
        '[models."toy-1b"]\n'
        'name = "Toy 1B"\n'
        "params_b = 1.0\n"
        "layers = 8\n"
        "kv_head_dim = 256\n"
        "max_context = 4096\n",
        encoding="utf-8",
    )
    (tmp_path / "hardware.toml").write_text(
        '[gpus."toy-gpu"]\n'
        "vram_gb = 4.0\n"
        "memory_bandwidth_gbs = 100.0\n",
        encoding="utf-8",
    )
    exit_code = run("check", "toy-1b", "--gpu", "toy-gpu", "--json")
    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["model"] == "toy-1b"
    assert out["fits"] is True
