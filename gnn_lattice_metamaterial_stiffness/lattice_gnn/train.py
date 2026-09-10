# train, checkpoint, and evaluate one model configuration reproducibly

from __future__ import annotations

import csv
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import matplotlib
matplotlib.use("Agg")
from .plot_style import configure_matplotlib
configure_matplotlib()
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, NullFormatter

import numpy as np
import torch
from torch import nn
from torch_geometric.loader import DataLoader

from .config import (
    CFG,
    REL_DENSITY,
    STRUT_RADIUS_MM,
    TRAIN_LOG_EVERY,
    feature_flags,
)
from .builders import make_loaders, reset_loader_generator
from .models import (
    BipartiteGNN,
    StaticEdgeHomogeneousGNN,
    DynamicEdgeHomogeneousGNN,
    NodeEncoderANN,
    RawFeatureMeanPoolingANN,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TARGET_NAMES = ("Ex", "Ey", "Ez", "Gyz", "Gxz", "Gxy")
TARGET_LABELS = (
    r"$E_x$", r"$E_y$", r"$E_z$",
    r"$G_{yz}$", r"$G_{xz}$", r"$G_{xy}$",
)
PRIMARY_MODEL_NAMES = (
    "BipartiteGNN",
    "DynamicEdgeHomogeneousGNN",
    "StaticEdgeHomogeneousGNN",
    "NodeEncoderANN",
    "RawFeatureMeanPoolingANN",
)
MODEL_COMPARISON_ORDER = (
    "BipartiteGNN",
    "DynamicEdgeHomogeneousGNN",
    "StaticEdgeHomogeneousGNN",
    "NodeEncoderANN",
    "RawFeatureMeanPoolingANN",
)

# format UTC wall time for training progress messages
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())

# seed python, numpy, and pytorch for repeatable training
def _seed_everything(seed: int, threads: int):
    import os
    import random
    import numpy as np
    import torch

    # python hash and blas/openmp threads
    os.environ["PYTHONHASHSEED"] = str(int(seed) & 0xFFFFFFFF)
    os.environ.setdefault("OMP_NUM_THREADS", str(threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    os.environ.setdefault("NUMEXPR_NUM_THREADS", str(threads))
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", str(threads))

    # base rngs
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass

    # clamp intra-op & inter-op threads
    try:
        torch.set_num_threads(int(threads))
    except Exception:
        pass
    try:
        torch.set_num_interop_threads(max(1, int(threads) // 2))
    except Exception:
        pass

    # request deterministic algorithm selection
    # disable tf32 to remain numerically comparable with earlier runs
    try:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    except Exception:
        pass

# loss
# mean squared error in standardized log10 stiffness space
def loss(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - true) ** 2)

# metrics
# compute scalar mean absolute error for tensors
def _mae(y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
    return float(torch.mean(torch.abs(y_true - y_pred)).item())

# compute coefficient of determination with a constant target safeguard
def _r2_np(y_pred: np.ndarray, y_true: np.ndarray, tol: float = 1e-12) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    ss_tot = float(((y_true - y_true.mean()) ** 2).sum())
    ss_res = float(((y_true - y_pred) ** 2).sum())
    if ss_tot < tol:
        return 1.0 if ss_res < tol else 0.0
    return float(1.0 - ss_res / ss_tot)

# compute scalar mean absolute error for NumPy arrays
def _mae_np(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_true - y_pred)))

# format a finite value in scientific notation
def _format_scientific(value: float, decimal_places: int = 2) -> str:
    value = float(value)
    if not np.isfinite(value):
        return str(value)
    return f"{value:.{decimal_places}e}"

# format model and hyperparameter labels consistently across plots
def _model_result_label(result: Dict, multiline: bool = False) -> str:
    display_names = {
        "BipartiteGNN": "Bipartite GNN",
        "StaticEdgeHomogeneousGNN": "Static-Edge Homogeneous GNN",
        "DynamicEdgeHomogeneousGNN": "Dynamic-Edge Homogeneous GNN",
        "NodeEncoderANN": "Node-Encoder ANN",
        "RawFeatureMeanPoolingANN": "Raw-Feature Mean-Pooling ANN",
    }
    model_name = str(result.get("model_name", ""))
    display_name = display_names.get(model_name, model_name)

    def _compact(value) -> str:
        try:
            return f"{float(value):g}"
        except (TypeError, ValueError):
            return str(value)

    width_depth = (
        f"width={_compact(result.get('hidden', ''))}, "
        f"depth={_compact(result.get('depth', ''))}"
    )
    optimizer = (
        f"lr={_compact(result.get('lr', ''))}, "
        f"wd={_compact(result.get('wd', ''))}"
    )
    if multiline:
        return f"{display_name}\n{width_depth}\n{optimizer}"
    return f"{display_name} ({width_depth}, {optimizer})"

# retain every epoch loss so figures can be regenerated later
def _write_loss_history(
    outdir: Path,
    train_hist: List[float],
    val_hist: List[float],
) -> None:
    with (outdir / "loss_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=("epoch", "training_loss", "validation_loss"),
        )
        writer.writeheader()
        for index, (train_value, val_value) in enumerate(
            zip(train_hist, val_hist),
            start=1,
        ):
            writer.writerow({
                "epoch": index,
                "training_loss": float(train_value),
                "validation_loss": float(val_value),
            })

# write names and physical true/predicted stiffness values to CSV
def _write_test_prediction_data(
    outdir: Path,
    structure_names: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    filename: str = "test_predictions.csv",
) -> None:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.shape != y_pred.shape or y_true.ndim != 2:
        raise ValueError("test prediction arrays must have matching 2D shapes")
    if y_true.shape[1] != len(TARGET_NAMES):
        raise ValueError("test prediction arrays must contain six targets")

    fieldnames = ["sample_index", "structure_name"]
    for target in TARGET_NAMES:
        fieldnames.extend((f"true_{target}_mpa", f"predicted_{target}_mpa"))

    with (outdir / filename).open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for index in range(y_true.shape[0]):
            row = {
                "sample_index": index,
                "structure_name": (
                    structure_names[index]
                    if index < len(structure_names)
                    else ""
                ),
            }
            for target_index, target in enumerate(TARGET_NAMES):
                row[f"true_{target}_mpa"] = float(y_true[index, target_index])
                row[f"predicted_{target}_mpa"] = float(
                    y_pred[index, target_index]
                )
            writer.writerow(row)

# convert log10 stiffness targets back to physical MPa values
def _targets_for_plot(y: torch.Tensor) -> np.ndarray:
    arr = y.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.power(10.0, arr)

# invert target z-scoring while retaining log10 units
def _targets_to_log10_scale(y: torch.Tensor, normalizers: Dict) -> torch.Tensor:
    stats = (normalizers or {}).get("y")
    if not stats:
        raise ValueError("standardized targets require training-set target statistics")
    mean = torch.as_tensor(stats["mean"], dtype=y.dtype, device=y.device)
    std = torch.as_tensor(stats["std"], dtype=y.dtype, device=y.device)
    return y * std + mean

# save the six-panel physical-space parity plot
def _plot_test_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str,
    outdir: Path,
    filename: str = "test_predictions.pdf",
) -> Tuple[List[float], List[float]]:
    outdir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    r2_vals = [_r2_np(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[1])]
    mae_vals = [_mae_np(y_pred[:, i], y_true[:, i]) for i in range(y_true.shape[1])]

    positive = np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)])
    positive = positive[np.isfinite(positive) & (positive > 0)]
    if positive.size:
        gmin = float(positive.min() * 0.9)
        gmax = float(positive.max() * 1.1)
    else:
        finite = np.concatenate([y_true.reshape(-1), y_pred.reshape(-1)])
        finite = finite[np.isfinite(finite)]
        gmin = float(finite.min()) if finite.size else -1.0
        gmax = float(finite.max()) if finite.size else 1.0
        pad = max(1e-6, 0.05 * (gmax - gmin if gmax > gmin else 1.0))
        gmin -= pad
        gmax += pad

    fig, axes = plt.subplots(2, 3, figsize=(12, 8), dpi=180)
    for idx, ax in enumerate(axes.ravel()):
        ax.scatter(y_true[:, idx], y_pred[:, idx], s=12, color="#007d85", alpha=0.40)
        ax.plot([gmin, gmax], [gmin, gmax], "--", color="0.5", linewidth=1.0)
        if gmin > 0 and gmax > 0:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.0,)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlim(gmin, gmax)
        ax.set_ylim(gmin, gmax)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(f"True {TARGET_LABELS[idx]} (MPa)")
        ax.set_ylabel(f"Predicted {TARGET_LABELS[idx]} (MPa)")
        ax.set_title(
            rf"$R^2$ = {r2_vals[idx]:.3f}, "
            f"MAE = {_format_scientific(mae_vals[idx])} MPa",
            fontsize=10,
        )
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(outdir / filename)
    plt.close(fig)
    return r2_vals, mae_vals

# dataloader cache
# avoids reloading dataset for every trial in a sweep
_LOADER_CACHE: Dict[str, Dict[str, DataLoader]] = {}

# build once per process and reuse across train_one() calls.
def _get_loaders_cached(
    include_centrality: bool = True,
) -> Dict[str, DataLoader]:
    key = "centrality_on" if include_centrality else "centrality_off"
    if key not in _LOADER_CACHE:
        t0 = time.time()
        print(f"[{_now()}] [train] building loaders …")
        ld = make_loaders(include_centrality=include_centrality)
        _LOADER_CACHE[key] = ld
        dt = time.time() - t0
        # report the fixed split sizes and feature dimensions once
        try:
            het = ld["het"]; hom = ld["hom"]
            print(f"[{_now()}] [train] loaders ready in {dt:.1f}s "
                  f"(het: tr={len(het['train'].dataset)} va={len(het['val'].dataset)} te={len(het['test'].dataset)}; "
                  f"hom: tr={len(hom['train'].dataset)} va={len(hom['val'].dataset)} te={len(hom['test'].dataset)})")
        except Exception:
            print(f"[{_now()}] [train] loaders ready in {dt:.1f}s")
    return _LOADER_CACHE[key]

# batch helpers
# ensure batch has .u tensor; create empty if missing
def _ensure_u(batch):
    if hasattr(batch, "u"):
        return batch
    # infer number of graphs in batch
    n_graphs = getattr(batch, "num_graphs", None)
    if n_graphs is None:
        try:
            if hasattr(batch, "batch"):
                n_graphs = int(batch.batch.max().item()) + 1
            elif "joint" in batch and hasattr(batch["joint"], "batch"):
                n_graphs = int(batch["joint"].batch.max().item()) + 1
            else:
                n_graphs = 1
        except Exception:
            n_graphs = 1
    batch.u = torch.zeros((n_graphs, 0), dtype=torch.float32, device=DEVICE)
    return batch

# move one graph batch to the selected device and evaluate the model
def _forward_on_device(model, batch):
    batch = batch.to(DEVICE, non_blocking=DEVICE.type == "cuda")
    batch = _ensure_u(batch)
    pred = model(batch)
    # accept targets stored at the root or under the graph_y key
    try:
        true = batch.y
    except AttributeError:
        try:
            true = batch["graph_y"]
        except Exception as _e:
            raise AttributeError("batch has neither .y nor ['graph_y'] target tensors") from _e
    true = true.to(DEVICE)
    return pred, true

# return one batch loss together with detached predictions and targets
def _forward_batch(model, batch):
    pred, true = _forward_on_device(model, batch)
    return loss(pred, true), pred.detach().cpu(), true.detach().cpu()

# calculate a sample weighted standardized target MSE for one epoch
def _run_epoch(model, loader: DataLoader, optimizer=None) -> float:
    is_train = optimizer is not None
    model.train(is_train)
    batch_losses = []
    batch_sizes = []
    n = 0
    grad_context = torch.enable_grad() if is_train else torch.inference_mode()
    with grad_context:
        for batch in loader:
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            pred, true = _forward_on_device(model, batch)
            batch_loss = loss(pred, true)
            if is_train:
                batch_loss.backward()
                optimizer.step()
            batch_size = int(true.size(0))
            batch_losses.append(batch_loss.detach())
            batch_sizes.append(batch_size)
            n += batch_size
    if not batch_losses:
        return 0.0
    loss_values = torch.stack(batch_losses).cpu().to(dtype=torch.float64)
    weights = torch.tensor(batch_sizes, dtype=torch.float64)
    total = torch.dot(loss_values, weights).item()
    return total / max(1, n)

# collect standardized predictions and targets without shuffling
def _predict_loader(model, loader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            _, p, t = _forward_batch(model, batch)
            preds.append(p)
            trues.append(t)
    if len(preds) == 0:
        return torch.zeros((0, 6), dtype=torch.float32), torch.zeros((0, 6), dtype=torch.float32)
    return torch.cat(preds, dim=0), torch.cat(trues, dim=0)


# synchronize CUDA only when wall-clock timing requires it
def _synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


# measure steady state full loader inference after worker/model warm-up
def _timed_predict_loader(model, loader: DataLoader,
                          repeats: int = 3) -> Tuple[torch.Tensor, torch.Tensor, float]:
    repeats = max(1, int(repeats))
    _predict_loader(model, loader)
    _synchronize_device()

    elapsed = []
    predictions = truths = None
    for _ in range(repeats):
        _synchronize_device()
        start = time.perf_counter()
        predictions, truths = _predict_loader(model, loader)
        _synchronize_device()
        elapsed.append(time.perf_counter() - start)

    return predictions, truths, float(np.median(np.asarray(elapsed, dtype=np.float64)))

# recover structure names in deterministic validation or test order
def _names_for_split(loaders: Dict, split: str) -> List[str]:
    normalizers = loaders.get("normalizers", {}) if isinstance(loaders, dict) else {}
    names = list((normalizers or {}).get("names", []))
    idxs = list((normalizers or {}).get(f"idx_{split}", []))
    if names and idxs:
        return [str(names[i]) for i in idxs if 0 <= int(i) < len(names)]
    return []

# infer input feature dimensions for model creation
# be robust to older caches and accidental kind mismatches
# returns encoder and fixed pooling dimensions
def _infer_dims_from_loader(loader, kind: str):
    if len(loader.dataset) == 0:
        raise ValueError("cannot infer model dimensions from an empty training split")
    batch = loader.dataset[0]

    # try to auto detect hetero vs homo regardless of 'kind'
    is_hetero = False
    try:
        is_hetero = ("joint" in batch) and ("strut" in batch)
    except Exception:
        is_hetero = False

    if kind == "het" or is_hetero:
        # hetero graph: expect per-type .x
        j_x = getattr(batch["joint"], "x", None)
        s_x = getattr(batch["strut"], "x", None)
        j_dim = int(j_x.size(1)) if (j_x is not None and j_x.numel() > 0) else 0
        s_dim = int(s_x.size(1)) if (s_x is not None and s_x.numel() > 0) else 0
        j_centrality = getattr(batch["joint"], "pooling_centrality", None)
        s_centrality = getattr(batch["strut"], "pooling_centrality", None)
    else:
        # homogeneous fallback
        x = getattr(batch, "x", None)
        edge_attr = getattr(batch, "edge_attr", None)
        j_dim = int(x.size(1)) if (x is not None and x.numel() > 0) else 0
        s_dim = int(edge_attr.size(1)) if (edge_attr is not None and edge_attr.numel() > 0) else 0
        j_centrality = getattr(batch, "joint_pooling_centrality", None)
        s_centrality = getattr(batch, "strut_pooling_centrality", None)

    j_centrality_dim = (
        int(j_centrality.size(1)) if j_centrality is not None else 0
    )
    s_centrality_dim = (
        int(s_centrality.size(1)) if s_centrality is not None else 0
    )
    if j_centrality_dim != s_centrality_dim:
        raise ValueError("joint and strut pooling centrality widths differ")

    u = getattr(batch, "u", None)
    u_dim = int(u.size(1)) if (u is not None and u.numel() > 0) else 0
    return {
        "s_dim": int(s_dim),
        "j_dim": int(j_dim),
        "u_dim": int(u_dim),
        "centrality_dim": int(j_centrality_dim),
    }

# accept feature dimensions as either a dictionary or a tuple/list
# convert all dimensions to python integers before creating the model
def _build_model(model_name: str,
                 dims,
                 hidden: int, depth: int) -> nn.Module:
    if isinstance(dims, dict):
        in_dim_strut = int(dims.get("s_dim", 0))
        in_dim_joint = int(dims.get("j_dim", 0))
        u_dim        = int(dims.get("u_dim", 0))
        centrality_dim = int(dims.get("centrality_dim", 0))
    else:
        try:
            if len(dims) == 3:
                s_dim, j_dim, u_dim = dims
                centrality_dim = 0
            else:
                s_dim, j_dim, u_dim, centrality_dim = dims
        except Exception as e:
            raise TypeError(
                f"dims must be dict or 3/4-tuple, got: {type(dims)}"
            ) from e
        in_dim_strut = int(s_dim)
        in_dim_joint = int(j_dim)
        u_dim        = int(u_dim)
        centrality_dim = int(centrality_dim)

    hidden = int(hidden)
    depth  = int(depth)
    include_graph_counts = bool(CFG.get("INCLUDE_GRAPH_COUNTS", False))

    if model_name == "NodeEncoderANN":
        return NodeEncoderANN(
            in_dim_strut, in_dim_joint, u_dim, hidden, depth,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        ).to(DEVICE)
    if model_name == "RawFeatureMeanPoolingANN":
        return RawFeatureMeanPoolingANN(
            in_dim_strut, in_dim_joint, u_dim, hidden,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        ).to(DEVICE)
    if model_name == "BipartiteGNN":
        return BipartiteGNN(
            in_dim_strut, in_dim_joint, u_dim, hidden, depth,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        ).to(DEVICE)
    if model_name == "StaticEdgeHomogeneousGNN":
        return StaticEdgeHomogeneousGNN(
            in_dim_joint, in_dim_strut, u_dim, hidden, depth,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        ).to(DEVICE)
    if model_name == "DynamicEdgeHomogeneousGNN":
        return DynamicEdgeHomogeneousGNN(
            in_dim_joint, in_dim_strut, u_dim, hidden, depth,
            include_graph_counts=include_graph_counts,
            centrality_dim=centrality_dim,
        ).to(DEVICE)
    raise ValueError(f"unknown model: {model_name}")

# select the heterogeneous or homogeneous loader for a model family
def _model_kind(model_name: str) -> str:
    # decide which loader family to use
    # return "hom" for homogeneous models, "het" for everything else
    if model_name in (
        "StaticEdgeHomogeneousGNN",
        "DynamicEdgeHomogeneousGNN",
    ):
        return "hom"
    return "het"  # anns and bipartite use hetero graphs


# count only parameters optimized during fitting
def _trainable_parameter_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


# apply the grid depth except for the encoder-free raw feature baseline
def _primary_depths_for_model(
    model_name: str,
    default_depths: List[int],
) -> List[int]:
    if model_name == "RawFeatureMeanPoolingANN":
        return [0]
    return [int(depth) for depth in default_depths]


# save the information needed to reproduce an experiment
def _write_experiment_manifest(loaders: Dict, mode: str) -> None:
    import hashlib
    import platform
    import sys
    from importlib import metadata

    normalizers = dict(loaders.get("normalizers", {}) or {})
    names = list(normalizers.get("names", []))

    def split_record(split: str) -> Dict[str, object]:
        idxs = [int(i) for i in normalizers.get(f"idx_{split}", [])]
        split_names = [str(names[i]) for i in idxs if 0 <= i < len(names)]
        digest = hashlib.sha256("\n".join(split_names).encode("utf-8")).hexdigest()
        return {"count": len(idxs), "ordered_names_sha256": digest}

    source_hash = hashlib.sha256()
    source_root = Path(__file__).resolve().parent
    source_files = sorted(source_root.glob("*.py")) + [source_root.parent / "script.py"]
    for path in source_files:
        if path.is_file():
            source_hash.update(path.name.encode("utf-8"))
            source_hash.update(path.read_bytes())

    package_versions = {}
    for package in (
        "numpy", "networkx", "matplotlib", "torch", "torch-geometric",
        "torch-scatter", "e3nn",
    ):
        try:
            package_versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            package_versions[package] = "not-installed"

    manifest = {
        "created_utc": _now(),
        "mode": str(mode),
        "device": str(DEVICE),
        "python": sys.version,
        "platform": platform.platform(),
        "packages": package_versions,
        "source_sha256": source_hash.hexdigest(),
        "config": dict(CFG),
        "fixed_physical_constants": {
            "relative_density": REL_DENSITY,
            "strut_radius_mm": STRUT_RADIUS_MM,
        },
        "feature_flags": feature_flags(),
        "centrality_usage": normalizers.get("centrality_usage", "unknown"),
        "centrality_pooling_mode": normalizers.get(
            "centrality_pooling_mode", "unknown"
        ),
        "split_strategy": normalizers.get("split_strategy", "unknown"),
        "duplicate_input_groups": int(normalizers.get("duplicate_input_groups", 0)),
        "training_axis_permutation_augmentation": normalizers.get(
            "training_axis_permutation_augmentation",
            {
                "enabled": False,
                "added_training_samples": 0,
            },
        ),
        "splits": {split: split_record(split) for split in ("train", "val", "test")},
    }
    path = Path(CFG["OUTPUTS_DIR"]) / f"experiment_manifest_{mode}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


# train one model configuration and retain its lowest validation loss checkpoint
# screening records validation results without evaluating the test set or saving fit artifacts
# final evaluation saves the checkpoint, loss history, raw predictions, parity plot, timings, and physical space metrics for each stiffness component
def train_one(model_name: str,
              hidden: int, depth: int, lr: float, wd: float,
              tag_suffix: str = "",
              loaders_override: Dict[str, Dict[str, DataLoader]] | None = None,
              prediction_title: str | None = None,
              evaluate_test: bool = True,
              train_seed: int | None = None,
              epochs: int | None = None,
              patience: int | None = None,
              save_artifacts: bool = True,
              save_plots: bool = True) -> Dict[str, float]:
    # cache-aware seeding and threads
    run_seed = int(CFG["TRAIN_SEED"] if train_seed is None else train_seed)
    _seed_everything(seed=run_seed, threads=int(CFG["CPU_THREADS"]))

    loaders = loaders_override if loaders_override is not None else _get_loaders_cached()
    kind = _model_kind(model_name)
    train_loader = loaders[kind]["train"]
    val_loader   = loaders[kind]["val"]
    test_loader  = loaders[kind]["test"]

    dims = _infer_dims_from_loader(train_loader, kind)
    # ensure int-typed dims
    dims = {
        "s_dim": int(dims.get("s_dim", 0)),
        "j_dim": int(dims.get("j_dim", 0)),
        "u_dim": int(dims.get("u_dim", 0)),
        "centrality_dim": int(dims.get("centrality_dim", 0)),
    }

    model = _build_model(model_name, dims, hidden, depth)
    num_parameters = _trainable_parameter_count(model)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # loaders are memoized across trials
    # reset their generators so trial order cannot change the minibatch sequence used by a model configuration
    for split_offset, loader in enumerate((train_loader, val_loader, test_loader)):
        reset_loader_generator(loader, run_seed + split_offset)
    epochs = int(CFG["FINAL_EPOCHS"] if epochs is None else epochs)
    patience = int(CFG["FINAL_PATIENCE"] if patience is None else patience)
    log_every = TRAIN_LOG_EVERY

    plots_root = Path(CFG["OUTPUTS_DIR"])
    tag = f"{model_name}-h{hidden}-d{depth}-lr{lr:g}-wd{wd:g}"
    if tag_suffix:
        tag = f"{tag}-{tag_suffix}"
    outdir = plots_root / tag
    if save_artifacts:
        outdir.mkdir(parents=True, exist_ok=True)

    # track best in memory; save once at the end
    import copy as _copy
    best_val = float("inf")
    best_epoch = 0
    time_to_best = 0.0
    best_state = None
    bad_epochs = 0

    train_losses: List[float] = []
    val_losses: List[float] = []
    train_epoch_time = 0.0
    val_epoch_time = 0.0

    print(f"[{_now()}] [train] start {tag} on {DEVICE} (epochs={epochs})")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        t_train = time.time()
        tr = _run_epoch(model, train_loader, optimizer=opt)
        train_epoch_time += time.time() - t_train
        t_val = time.time()
        va = _run_epoch(model, val_loader, optimizer=None)
        val_epoch_time += time.time() - t_val
        train_losses.append(tr)
        val_losses.append(va)

        if (epoch % log_every) == 0 or epoch == 1:
            print(f"[{_now()}] [train] {tag} ep {epoch:04d} train={tr:.5f} val={va:.5f}")

        # early stopping on val; store best weights in memory only
        if va + 1e-9 < best_val:
            best_val = va
            best_epoch = epoch
            time_to_best = time.time() - t0
            best_state = _copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[{_now()}] [train] early stop {tag} at ep {epoch:04d} (patience={patience})")
                break

    dt = time.time() - t0
    print(f"[{_now()}] [train] finished {tag} in {dt:.1f}s (best_val={best_val:.5f} @ ep {best_epoch})")

    # restore and save the checkpoint with minimum validation mse
    ckpt = outdir / "best.pt"
    if best_state is not None:
        model.load_state_dict(best_state)
    if save_artifacts:
        torch.save(model.state_dict(), ckpt)
        _write_loss_history(outdir, train_losses, val_losses)

    # evaluate the best validation checkpoint
    model.eval()
    val_eval_t0 = time.time()
    P_val, T_val = _predict_loader(model, val_loader)
    normalizers = loaders.get("normalizers", {}) if isinstance(loaders, dict) else {}
    P_val = _targets_to_log10_scale(P_val, normalizers)
    T_val = _targets_to_log10_scale(T_val, normalizers)
    mae_val_log10 = _mae(P_val, T_val)
    true_val_physical = _targets_for_plot(T_val)
    predicted_val_physical = _targets_for_plot(P_val)
    val_r2_by_target = [
        _r2_np(predicted_val_physical[:, index], true_val_physical[:, index])
        for index in range(len(TARGET_NAMES))
    ]
    val_mae_by_target = [
        _mae_np(predicted_val_physical[:, index], true_val_physical[:, index])
        for index in range(len(TARGET_NAMES))
    ]
    if save_artifacts:
        _write_test_prediction_data(
            outdir,
            _names_for_split(loaders, "val"),
            true_val_physical,
            predicted_val_physical,
            filename="validation_predictions.csv",
        )
    val_eval_time = time.time() - val_eval_t0
    inference_timing_repeats = 3 if evaluate_test else 0
    test_eval_time = float("nan")
    mae_test_log10 = float("nan")
    test_r2_by_target: List[float] = []
    test_mae_by_target: List[float] = []
    r2_test_physical_mean = float("nan")
    mae_test_physical_mean = float("nan")
    plot_time = 0.0
    if evaluate_test:
        P_test, T_test, test_eval_time = _timed_predict_loader(
            model, test_loader, repeats=inference_timing_repeats
        )
        P_test = _targets_to_log10_scale(P_test, normalizers)
        T_test = _targets_to_log10_scale(T_test, normalizers)
        mae_test_log10 = _mae(P_test, T_test)
        plot_t0 = time.time()
        prediction_label = _model_result_label({
            "model_name": model_name,
            "hidden": hidden,
            "depth": depth,
            "lr": lr,
            "wd": wd,
        })
        if prediction_title is None:
            prediction_title = f"Test Predictions: {prediction_label}"
        true_test_physical = _targets_for_plot(T_test)
        predicted_test_physical = _targets_for_plot(P_test)
        if save_artifacts:
            _write_test_prediction_data(
                outdir,
                _names_for_split(loaders, "test"),
                true_test_physical,
                predicted_test_physical,
            )
        test_r2_by_target = [
            _r2_np(predicted_test_physical[:, index], true_test_physical[:, index])
            for index in range(len(TARGET_NAMES))
        ]
        test_mae_by_target = [
            _mae_np(predicted_test_physical[:, index], true_test_physical[:, index])
            for index in range(len(TARGET_NAMES))
        ]
        if save_artifacts and save_plots:
            _plot_test_predictions(
                true_test_physical,
                predicted_test_physical,
                title=prediction_title,
                outdir=outdir,
            )
        r2_test_physical_mean = float(
            np.nanmean(np.asarray(test_r2_by_target, dtype=np.float64))
        )
        mae_test_physical_mean = float(
            np.nanmean(np.asarray(test_mae_by_target, dtype=np.float64))
        )
        plot_time = time.time() - plot_t0

    flags = feature_flags()
    flags["centrality_guided_pooling"] = (
        str(normalizers.get("centrality_pooling_mode", ""))
        == "centrality_augmented"
    )
    result = {
        "val_loss": best_val,
        "mae_val_log10": mae_val_log10,
        "val_r2_by_target": val_r2_by_target,
        "val_mae_by_target": val_mae_by_target,
        "r2_val_physical_mean": float(np.nanmean(val_r2_by_target)),
        "mae_val_physical_mean": float(np.nanmean(val_mae_by_target)),
        "mae_test_log10": mae_test_log10,
        "test_r2_by_target": test_r2_by_target,
        "test_mae_by_target": test_mae_by_target,
        "r2_test_physical_mean": r2_test_physical_mean,
        "mae_test_physical_mean": mae_test_physical_mean,
        "fit_time_sec": dt,
        "training_time_sec": dt,
        "time_to_best_val_sec": time_to_best,
        "train_epoch_time_sec": train_epoch_time,
        "val_epoch_time_sec": val_epoch_time,
        "val_eval_time_sec": val_eval_time,
        "test_eval_time_sec": test_eval_time,
        "inference_time_test_sec": test_eval_time,
        "inference_timing_repeats": inference_timing_repeats,
        "plot_time_sec": plot_time,
        "epochs_ran": len(train_losses),
        "best_epoch": best_epoch,
        "train_loss_at_best": float(train_losses[best_epoch - 1]),
        "train_loss_history": [float(value) for value in train_losses],
        "validation_loss_history": [float(value) for value in val_losses],
        "num_parameters": num_parameters,
        "split_seed": int(CFG["SPLIT_SEED"]),
        "train_seed": run_seed,
        "split_strategy": str(normalizers.get("split_strategy", "unknown")),
        "num_train": int(len(train_loader.dataset)),
        "num_val": int(len(val_loader.dataset)),
        "num_test": int(len(test_loader.dataset)),
        **flags,
        "tag": tag,
        "ckpt": str(ckpt) if save_artifacts else "",
        }
    centrality_names = [
        name for name, key in (
            ("degree", "INCLUDE_DEGREE_CENTRALITY"),
            ("closeness", "INCLUDE_CLOSENESS_CENTRALITY"),
            ("betweenness", "INCLUDE_BETWEENNESS_CENTRALITY"),
            ("eigenvector", "INCLUDE_EIGENVECTOR_CENTRALITY"),
        ) if bool(CFG.get(key, True))
    ]
    if flags["centrality_guided_pooling"]:
        learned_weights = {}
        for entity, module_name in (("joint", "joint_pool"), ("strut", "strut_pool")):
            module = getattr(model, module_name, None)
            logits = getattr(module, "logits", None)
            if logits is not None and logits.numel() == len(centrality_names):
                values = torch.softmax(logits.detach().cpu(), dim=0).tolist()
                learned_weights[entity] = {
                    name: float(value)
                    for name, value in zip(centrality_names, values)
                }
        result["learned_centrality_mixture_weights"] = learned_weights
    else:
        result["learned_centrality_mixture_weights"] = None
    return result
