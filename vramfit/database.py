"""Load the offline TOML databases (models, quants, hardware).

Built-in data lives next to this module in ``vramfit/data``. Users can
extend or override it by dropping ``models.toml`` and/or
``hardware.toml`` in their working directory, or by pointing
``load_database`` at another folder with ``--data-dir``.
"""

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DATA_DIR = Path(__file__).parent / "data"

REQUIRED_MODEL_FIELDS = ("params_b", "layers", "kv_head_dim", "max_context")


class DatabaseError(Exception):
    """Raised when a database file is missing fields or malformed."""


@dataclass(frozen=True)
class Quant:
    id: str
    bits_per_weight: float
    quality: str


@dataclass(frozen=True)
class Model:
    id: str
    name: str
    params_b: float
    layers: int
    kv_head_dim: int
    max_context: int
    family: str


@dataclass(frozen=True)
class Gpu:
    id: str
    name: str
    vram_gb: float
    memory_bandwidth_gbs: float


@dataclass(frozen=True)
class CpuPreset:
    id: str
    name: str
    ram_gb: float
    swap_gb: float
    memory_bandwidth_gbs: float


@dataclass(frozen=True)
class Overheads:
    gpu_gb: float
    cpu_gb: float


@dataclass(frozen=True)
class Headroom:
    gpu_fraction: float
    cpu_fraction: float
    swap_fraction: float


@dataclass(frozen=True)
class Database:
    models: dict
    quants: list
    quants_by_id: dict
    gpus: dict
    cpus: dict
    overheads: Overheads
    headroom: Headroom
    context_ladder: list
    sources: list


def _read(path: Path):
    with path.open("rb") as fh:
        return tomllib.load(fh)


def _required_float(section: dict, path: Path, name: str, fields):
    missing = [f for f in fields if f not in section]
    if missing:
        raise DatabaseError(
            f"{path}: '{name}' is missing required field(s): {', '.join(missing)}"
        )


def _load_models(paths):
    schema = None
    merged = {}
    for path in paths:
        if not path.exists():
            continue
        doc = _read(path)
        schema = doc.get("schema_version", schema)
        quants = doc.get("quants") or {}
        for quant_id, spec in quants.items():
            _required_float(spec, path, f"quants.{quant_id}", ("bits_per_weight",))
            merged.setdefault("_quants", {})[quant_id] = Quant(
                id=quant_id,
                bits_per_weight=float(spec["bits_per_weight"]),
                quality=str(spec.get("quality", "")),
            )
        for model_id, spec in (doc.get("models") or {}).items():
            _required_float(spec, path, f"models.{model_id}", REQUIRED_MODEL_FIELDS)
            merged[model_id] = Model(
                id=model_id,
                name=str(spec.get("name", model_id)),
                family=str(spec.get("family", "")),
                params_b=float(spec["params_b"]),
                layers=int(spec["layers"]),
                kv_head_dim=int(spec["kv_head_dim"]),
                max_context=int(spec["max_context"]),
            )
        merged.setdefault("_quants_seen_paths", []).append(path)
        merged["_schema"] = schema
    return merged


def _load_hardware(paths):
    merged = {}
    for path in paths:
        if not path.exists():
            continue
        doc = _read(path)
        for gpu_id, spec in (doc.get("gpus") or {}).items():
            _required_float(spec, path, f"gpus.{gpu_id}", ("vram_gb",))
            merged["gpus", gpu_id] = Gpu(
                id=gpu_id,
                name=str(spec.get("name", gpu_id)),
                vram_gb=float(spec["vram_gb"]),
                memory_bandwidth_gbs=float(spec.get("memory_bandwidth_gbs", 0.0)),
            )
        for cpu_id, spec in (doc.get("cpus") or {}).items():
            _required_float(spec, path, f"cpus.{cpu_id}", ("ram_gb",))
            merged["cpus", cpu_id] = CpuPreset(
                id=cpu_id,
                name=str(spec.get("name", cpu_id)),
                ram_gb=float(spec["ram_gb"]),
                swap_gb=float(spec.get("swap_gb", 0.0)),
                memory_bandwidth_gbs=float(spec.get("memory_bandwidth_gbs", 0.0)),
            )
        if "overheads" in doc:
            merged["overheads"] = Overheads(
                gpu_gb=float(doc["overheads"].get("gpu_gb", 1.2)),
                cpu_gb=float(doc["overheads"].get("cpu_gb", 0.4)),
            )
        if "headroom" in doc:
            merged["headroom"] = Headroom(
                gpu_fraction=float(doc["headroom"].get("gpu_fraction", 0.95)),
                cpu_fraction=float(doc["headroom"].get("cpu_fraction", 0.90)),
                swap_fraction=float(doc["headroom"].get("swap_fraction", 0.85)),
            )
        if "context_ladder" in doc:
            ladder = [int(ctx) for ctx in doc["context_ladder"]]
            merged["context_ladder"] = sorted(set(ladder))
        merged.setdefault("_paths", []).append(path)
        merged["schema"] = doc.get("schema_version", merged.get("schema"))
    return merged


def load_user_files(extra_dir: Optional[Path]) -> list:
    if extra_dir is None:
        cwd = Path.cwd()
        return [cwd / "models.toml", cwd / "hardware.toml"]
    folder = Path(extra_dir)
    return [folder / "models.toml", folder / "hardware.toml"]


def load_database(user_dir: Optional[Path] = None) -> Database:
    user_paths = load_user_files(user_dir)
    models_doc = _load_models([DATA_DIR / "models.toml", *user_paths])
    hw_doc = _load_hardware([DATA_DIR / "hardware.toml", *user_paths])

    quants = list(models_doc.get("_quants", {}).values())
    if not quants:
        raise DatabaseError("no [quants] section found in any models.toml")

    base_overheads = Overheads(gpu_gb=1.2, cpu_gb=0.4)
    base_headroom = Headroom(0.95, 0.90, 0.85)
    sources = [str(DATA_DIR / "models.toml"), str(DATA_DIR / "hardware.toml")] + [
        str(p) for p in user_paths if p.exists()
    ]

    def unwrap(prefix, items):
        return {k[1]: v for k, v in items
                if isinstance(k, tuple) and k[0] == prefix}

    return Database(
        models={k: v for k, v in models_doc.items() if isinstance(v, Model)},
        quants=quants,
        quants_by_id={q.id: q for q in quants},
        gpus=unwrap("gpus", hw_doc.items()),
        cpus=unwrap("cpus", hw_doc.items()),
        overheads=hw_doc.get("overheads", base_overheads),
        headroom=hw_doc.get("headroom", base_headroom),
        context_ladder=hw_doc.get("context_ladder", [512, 1024, 2048, 4096, 8192, 16384, 32768]),
        sources=sources,
    )
