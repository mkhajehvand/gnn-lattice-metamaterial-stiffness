# load and validate the experiment configuration

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable

REL_DENSITY = 0.01 # Lumpe and Stankovic (2021) used this value
STRUT_RADIUS_MM = 0.1 # Lumpe and Stankovic (2021) used this value
OUTPUTS_DIRNAME = "Outputs"
DATALOADER_NUM_WORKERS = 2
TRAIN_LOG_EVERY = 10
CPU_THREADS = 4


DATASET_FILTER = "All"

# Grega et al. (2025) filtering criteria
DATA_FILTER_USE_MIN_NODE_SEPARATION = True
DATA_FILTER_USE_MAX_JOINTS = True
DATA_FILTER_MAX_JOINTS = 60
DATA_FILTER_USE_MAX_STRUTS = True
DATA_FILTER_MAX_STRUTS = 150
DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO = True
DATA_FILTER_MAX_RADIUS_LENGTH_RATIO = 0.2

_DEFAULT_CONFIG_PATHS = [
    Path.cwd() / "config.json",
    Path(__file__).resolve().parent / "config.json",
    Path(__file__).resolve().parents[1] / "config.json",
]

_REQUIRED_KEYS: Iterable[str] = [
    "EXPERIMENT_NAME",
    # dataset fraction; 1 uses the full dataset
    "FRACTION",

    # maximum spherical harmonic degree for strut features
    "SH_LMAX",

    # entity feature choices
    "STRUT_USE_SH",
    "STRUT_USE_LENGTH",
    "JOINT_USE_ANGLES",

    # centralities included in centrality-guided pooling
    "INCLUDE_DEGREE_CENTRALITY",
    "INCLUDE_CLOSENESS_CENTRALITY",
    "INCLUDE_BETWEENNESS_CENTRALITY",
    "INCLUDE_EIGENVECTOR_CENTRALITY",

    # include unit cell parameters in the graph level features
    "INCLUDE_CELL_PARAMS",

    # include joint and strut counts in the graph level features
    "INCLUDE_GRAPH_COUNTS",

    # training settings
    "SPLIT_SEED",
    "TRAIN_SEED",
    "FINAL_TRAIN_SEEDS",
    "SWEEP_EPOCHS",
    "SWEEP_PATIENCE",
    "FINAL_EPOCHS",
    "FINAL_PATIENCE",
    "BATCH_SIZE",

    # enable axis permutation augmentation
    "TRAIN_AXIS_PERMUTATION_AUGMENTATION",

    # hyperparameter search grid
    "DEPTHS",
    "HIDDEN_SIZES",
    "LRS",
    "WDS",

]

# resolve an explicit config path before package relative defaults
def _find_config_path() -> Path:
    env_path = os.environ.get("GNN_CONFIG_PATH", "").strip()
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
    for p in _DEFAULT_CONFIG_PATHS:
        if p.exists():
            return p
    return Path.cwd() / "config.json"


# reject incomplete experiment configurations before data loading
def _ensure_required_keys(cfg: Dict[str, Any]) -> None:
    missing = [k for k in _REQUIRED_KEYS if k not in cfg]
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise KeyError(f"missing required config keys: {missing_str}")

# require every hyperparameter grid to be a nonempty list
def _ensure_grid_types(cfg: Dict[str, Any]) -> None:
    for k in ["DEPTHS", "HIDDEN_SIZES", "LRS", "WDS"]:
        if k not in cfg or not isinstance(cfg[k], list):
            raise TypeError(f'"{k}" must be a list in config.json')
        if len(cfg[k]) == 0:
            raise ValueError(f'"{k}" list cannot be empty')

# validate ranges that would otherwise fail late during training
def _validate_values(cfg: Dict[str, Any]) -> None:
    if not 0.0 < float(cfg["FRACTION"]) <= 1.0:
        raise ValueError('"FRACTION" must be in (0, 1]')
    for key in ("SWEEP_EPOCHS", "SWEEP_PATIENCE", "FINAL_EPOCHS", "FINAL_PATIENCE"):
        if int(cfg[key]) <= 0:
            raise ValueError(f'"{key}" must be positive')
    seeds = cfg["FINAL_TRAIN_SEEDS"]
    if not isinstance(seeds, list) or len(seeds) != 5:
        raise ValueError('"FINAL_TRAIN_SEEDS" must contain exactly five seeds')
    if len({int(seed) for seed in seeds}) != 5:
        raise ValueError('"FINAL_TRAIN_SEEDS" must contain five distinct seeds')
    if int(cfg["BATCH_SIZE"]) <= 0:
        raise ValueError('"BATCH_SIZE" must be positive')
    if int(cfg["SH_LMAX"]) < 0:
        raise ValueError('"SH_LMAX" cannot be negative')
    if any(int(value) < 1 for value in cfg["DEPTHS"]):
        raise ValueError('all "DEPTHS" values must be at least 1')
    if any(int(value) < 2 for value in cfg["HIDDEN_SIZES"]):
        raise ValueError('all "HIDDEN_SIZES" values must be at least 2')
    if any(float(value) <= 0.0 for value in cfg["LRS"]):
        raise ValueError('all "LRS" values must be positive')
    if any(float(value) < 0.0 for value in cfg["WDS"]):
        raise ValueError('all "WDS" values must be nonnegative')
    train_frac = float(cfg.get("TRAIN_FRAC", 0.8))
    val_frac = float(cfg.get("VAL_FRAC", 0.1))
    if train_frac <= 0.0 or val_frac <= 0.0 or train_frac + val_frac >= 1.0:
        raise ValueError('TRAIN_FRAC and VAL_FRAC must be positive and sum to less than 1')

# apply fixed thread limits and a stable Python hash seed
def _apply_repro_env(cfg: Dict[str, Any]) -> None:
    threads = str(CPU_THREADS)
    os.environ.setdefault("PYTHONHASHSEED", str(cfg["TRAIN_SEED"]))
    os.environ.setdefault("OMP_NUM_THREADS", threads)
    os.environ.setdefault("MKL_NUM_THREADS", threads)
    os.environ.setdefault("OPENBLAS_NUM_THREADS", threads)
    os.environ.setdefault("NUMEXPR_NUM_THREADS", threads)
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", threads)

# load user settings and attach fixed physical and filtering settings
def load_config(path: str | Path | None = None) -> Dict[str, Any]:
    cfg_path = Path(path) if path else _find_config_path()
    if not cfg_path.exists():
        raise FileNotFoundError(
            f"could not find config.json at {cfg_path}. "
            f"set GNN_CONFIG_PATH or place a json next to the code."
        )
    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    cfg = dict(user_cfg)
    cfg.update({
        "DATASET_FILTER": DATASET_FILTER,
        "DATA_FILTER_USE_MIN_NODE_SEPARATION":
            DATA_FILTER_USE_MIN_NODE_SEPARATION,
        "DATA_FILTER_USE_MAX_JOINTS": DATA_FILTER_USE_MAX_JOINTS,
        "DATA_FILTER_MAX_JOINTS": DATA_FILTER_MAX_JOINTS,
        "DATA_FILTER_USE_MAX_STRUTS": DATA_FILTER_USE_MAX_STRUTS,
        "DATA_FILTER_MAX_STRUTS": DATA_FILTER_MAX_STRUTS,
        "DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO":
            DATA_FILTER_USE_MAX_RADIUS_LENGTH_RATIO,
        "DATA_FILTER_MAX_RADIUS_LENGTH_RATIO":
            DATA_FILTER_MAX_RADIUS_LENGTH_RATIO,
        "CPU_THREADS": CPU_THREADS,
    })
    _ensure_required_keys(cfg)
    _ensure_grid_types(cfg)
    _validate_values(cfg)
    _apply_repro_env(cfg)

    # normalize booleans
    # centrality is always cached and is ablated automatically during a sweep
    cfg["STRUT_USE_CENTRALITY"] = True
    cfg["JOINT_USE_CENTRALITY"] = True
    for k in [
        "STRUT_USE_SH", "STRUT_USE_LENGTH",
        "STRUT_USE_CENTRALITY",
        "JOINT_USE_ANGLES", "JOINT_USE_CENTRALITY",
        "INCLUDE_DEGREE_CENTRALITY",
        "INCLUDE_CLOSENESS_CENTRALITY",
        "INCLUDE_BETWEENNESS_CENTRALITY",
        "INCLUDE_EIGENVECTOR_CENTRALITY",
        "INCLUDE_CELL_PARAMS", "INCLUDE_GRAPH_COUNTS",
        "TRAIN_AXIS_PERMUTATION_AUGMENTATION",
        "DATA_FILTER_USE_MIN_NODE_SEPARATION",
    ]:
        cfg[k] = bool(cfg[k])

    if cfg["TRAIN_AXIS_PERMUTATION_AUGMENTATION"]:
        if not cfg["STRUT_USE_SH"] or int(cfg["SH_LMAX"]) < 2:
            raise ValueError(
                '"TRAIN_AXIS_PERMUTATION_AUGMENTATION" requires '
                '"STRUT_USE_SH": true and "SH_LMAX" >= 2'
            )

    # coerce numerics
    cfg["SH_LMAX"]   = int(cfg["SH_LMAX"])
    cfg["SPLIT_SEED"] = int(cfg["SPLIT_SEED"])
    cfg["TRAIN_SEED"] = int(cfg["TRAIN_SEED"])
    cfg["FINAL_TRAIN_SEEDS"] = [int(seed) for seed in cfg["FINAL_TRAIN_SEEDS"]]
    cfg["SWEEP_EPOCHS"] = int(cfg["SWEEP_EPOCHS"])
    cfg["SWEEP_PATIENCE"] = int(cfg["SWEEP_PATIENCE"])
    cfg["FINAL_EPOCHS"] = int(cfg["FINAL_EPOCHS"])
    cfg["FINAL_PATIENCE"] = int(cfg["FINAL_PATIENCE"])
    cfg["BATCH_SIZE"]= int(cfg["BATCH_SIZE"])
    cfg["FRACTION"]  = float(cfg["FRACTION"])
    # keep outputs beside the package even when the active config is a snapshot
    package_root = Path(__file__).resolve().parents[1]
    cfg["OUTPUTS_DIR"] = str(package_root / OUTPUTS_DIRNAME)

    return cfg

# expose as module-level dict
CFG: Dict[str, Any] = load_config()

# return feature choices recorded in dataset and experiment manifests
def feature_flags(cfg: Dict[str, Any] | None = None) -> Dict[str, bool]:
    c = cfg or CFG
    return {
        "strut_use_sh": bool(c.get("STRUT_USE_SH", True)),
        "strut_use_length": bool(c.get("STRUT_USE_LENGTH", True)),
        "strut_use_centrality": bool(c.get("STRUT_USE_CENTRALITY", True)),
        "joint_use_angles": bool(c.get("JOINT_USE_ANGLES", True)),
        "joint_use_centrality": bool(c.get("JOINT_USE_CENTRALITY", True)),
        "include_degree_centrality": bool(c.get("INCLUDE_DEGREE_CENTRALITY", True)),
        "include_closeness_centrality": bool(c.get("INCLUDE_CLOSENESS_CENTRALITY", True)),
        "include_betweenness_centrality": bool(c.get("INCLUDE_BETWEENNESS_CENTRALITY", True)),
        "include_eigenvector_centrality": bool(c.get("INCLUDE_EIGENVECTOR_CENTRALITY", True)),
        "include_cell_params": bool(c.get("INCLUDE_CELL_PARAMS", True)),
        "include_graph_counts": bool(c.get("INCLUDE_GRAPH_COUNTS", False)),
        "standardize_log_targets": True,
    }
