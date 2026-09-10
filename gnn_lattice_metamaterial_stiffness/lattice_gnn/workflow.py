# publication workflow: sharded screening, merge, and seeded evaluation

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, List

import numpy as np

from .config import CFG
from . import train


MODEL_SHARDS = {
    0: ("BipartiteGNN",),
    1: ("DynamicEdgeHomogeneousGNN",),
    2: ("StaticEdgeHomogeneousGNN",),
    3: ("NodeEncoderANN", "RawFeatureMeanPoolingANN"),
}
CONDITIONS = (
    ("with_centrality_features", True),
    ("without_centrality_features", False),
)


# identify the code and screening settings shared by all shards
def _screening_signature() -> str:
    excluded = {
        "OUTPUTS_DIR", "OUTPUT_DIR", "STRUT_USE_CENTRALITY",
        "JOINT_USE_CENTRALITY", "FINAL_EPOCHS", "FINAL_PATIENCE",
        "FINAL_TRAIN_SEEDS",
    }
    payload = {key: value for key, value in CFG.items() if key not in excluded}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8"))
    source_root = Path(__file__).resolve().parent
    for path in sorted(source_root.glob("*.py")):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


# place every stage of one experiment under one safe folder name
def experiment_root(outputs_base: Path) -> Path:
    name = str(CFG["EXPERIMENT_NAME"]).strip()
    if not name or name in {".", ".."} or "/" in name or "\\" in name:
        raise ValueError("EXPERIMENT_NAME must be one safe folder name")
    return outputs_base / "experiments" / name


# synchronize output aliases used by training and plotting code
def _set_output(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    CFG["OUTPUTS_DIR"] = str(path)
    CFG["OUTPUT_DIR"] = str(path)


# convert numpy and path objects before writing reproducibility metadata
def _json_safe(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


# write one rectangular result table with a stable column order
def _write_csv(path: Path, rows: List[Dict]) -> None:
    if not rows:
        raise ValueError(f"cannot write an empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# read a screening shard without changing its stored values
def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


# retain selection and timing fields while discarding trial artifacts
def _screening_row(model_name: str, hidden: int, depth: int,
                   lr: float, wd: float, condition: str,
                   result: Dict) -> Dict:
    return {
        "condition": condition,
        "model_name": model_name,
        "hidden": int(hidden),
        "depth": int(depth),
        "lr": float(lr),
        "wd": float(wd),
        "val_loss": float(result["val_loss"]),
        "train_loss_at_best": float(result["train_loss_at_best"]),
        "mae_val_log10": float(result["mae_val_log10"]),
        "mean_validation_r2": float(result["r2_val_physical_mean"]),
        "mean_validation_mae_mpa": float(result["mae_val_physical_mean"]),
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "num_parameters": int(result["num_parameters"]),
        "training_time_sec": float(result["training_time_sec"]),
        "time_to_best_val_sec": float(result["time_to_best_val_sec"]),
        "split_seed": int(result["split_seed"]),
        "train_seed": int(result["train_seed"]),
    }


# screen one disjoint model group under both pooling conditions
def run_sweep_shard(shard_id: int, outputs_base: Path) -> Dict:
    if shard_id not in MODEL_SHARDS:
        raise ValueError(f"shard id must be one of {sorted(MODEL_SHARDS)}")
    shard_root = experiment_root(outputs_base) / "sweep_shards" / f"shard_{shard_id}"
    shard_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    try:
        for condition, enabled in CONDITIONS:
            CFG["STRUT_USE_CENTRALITY"] = enabled
            CFG["JOINT_USE_CENTRALITY"] = enabled
            train._LOADER_CACHE.clear()
            loaders = train._get_loaders_cached(include_centrality=enabled)
            for model_name in MODEL_SHARDS[shard_id]:
                depths = train._primary_depths_for_model(model_name, list(CFG["DEPTHS"]))
                for depth in depths:
                    for hidden in CFG["HIDDEN_SIZES"]:
                        for lr in CFG["LRS"]:
                            for wd in CFG["WDS"]:
                                result = train.train_one(
                                    model_name, int(hidden), int(depth),
                                    float(lr), float(wd),
                                    loaders_override=loaders,
                                    evaluate_test=False,
                                    train_seed=int(CFG["TRAIN_SEED"]),
                                    epochs=int(CFG["SWEEP_EPOCHS"]),
                                    patience=int(CFG["SWEEP_PATIENCE"]),
                                    save_artifacts=False,
                                    save_plots=False,
                                )
                                rows.append(_screening_row(
                                    model_name, hidden, depth, lr, wd,
                                    condition, result,
                                ))
        _write_csv(shard_root / "screening_results.csv", rows)
        manifest = {
            "shard_id": shard_id,
            "models": list(MODEL_SHARDS[shard_id]),
            "conditions": [name for name, _ in CONDITIONS],
            "num_trials": len(rows),
            "sweep_epochs": int(CFG["SWEEP_EPOCHS"]),
            "sweep_patience": int(CFG["SWEEP_PATIENCE"]),
            "split_seed": int(CFG["SPLIT_SEED"]),
            "train_seed": int(CFG["TRAIN_SEED"]),
            "screening_signature": _screening_signature(),
        }
        (shard_root / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        return manifest
    finally:
        train._LOADER_CACHE.clear()
        CFG["STRUT_USE_CENTRALITY"] = True
        CFG["JOINT_USE_CENTRALITY"] = True


# validate all shards and select each condition/model by validation mse
def merge_sweep_shards(outputs_base: Path) -> Dict:
    root = experiment_root(outputs_base)
    rows: List[Dict[str, str]] = []
    signatures = set()
    for shard_id in sorted(MODEL_SHARDS):
        path = root / "sweep_shards" / f"shard_{shard_id}" / "screening_results.csv"
        if not path.is_file():
            raise FileNotFoundError(f"missing completed sweep shard: {path}")
        manifest_path = path.with_name("manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        signatures.add(str(manifest["screening_signature"]))
        rows.extend(_read_csv(path))
    if len(signatures) != 1:
        raise RuntimeError("sweep shards were produced by different code or configurations")
    expected = {
        (condition, model)
        for condition, _ in CONDITIONS
        for models in MODEL_SHARDS.values()
        for model in models
    }
    present = {(row["condition"], row["model_name"]) for row in rows}
    if expected != present:
        raise RuntimeError(f"screening combinations differ from expected: {sorted(expected - present)}")
    selected: Dict[str, Dict[str, Dict]] = {}
    for condition, model_name in sorted(expected):
        candidates = [row for row in rows if row["condition"] == condition and row["model_name"] == model_name]
        winner = min(candidates, key=lambda row: float(row["val_loss"]))
        selected.setdefault(condition, {})[model_name] = {
            "hidden": int(winner["hidden"]),
            "depth": int(winner["depth"]),
            "lr": float(winner["lr"]),
            "wd": float(winner["wd"]),
            "screening_val_loss": float(winner["val_loss"]),
            "screening_best_epoch": int(winner["best_epoch"]),
            "screening_num_parameters": int(winner["num_parameters"]),
        }
    merged_root = root / "merged_sweep"
    merged_root.mkdir(parents=True, exist_ok=True)
    normalized = []
    float_keys = {
        "lr", "wd", "val_loss", "train_loss_at_best", "mae_val_log10",
        "mean_validation_r2", "mean_validation_mae_mpa",
        "training_time_sec", "time_to_best_val_sec",
    }
    int_keys = {"hidden", "depth", "best_epoch", "epochs_ran", "num_parameters", "split_seed", "train_seed"}
    for row in rows:
        normalized.append({
            key: float(value) if key in float_keys else int(value) if key in int_keys else value
            for key, value in row.items()
        })
    _write_csv(merged_root / "all_screening_results.csv", normalized)
    selection = {
        "selection_metric": "minimum validation MSE",
        "screening_signature": next(iter(signatures)),
        "selected": selected,
    }
    (merged_root / "selected_configurations.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )
    return selection


# store one final fit as a complete analysis ready table row
def _flatten_final_result(condition: str, model_name: str, seed: int,
                          cfg: Dict, result: Dict) -> Dict:
    row = {
        "condition": condition, "model_name": model_name,
        "train_seed": int(seed), "split_seed": int(result["split_seed"]),
        "hidden": int(cfg["hidden"]), "depth": int(cfg["depth"]),
        "lr": float(cfg["lr"]), "wd": float(cfg["wd"]),
        "val_loss": float(result["val_loss"]),
        "train_loss_at_best": float(result["train_loss_at_best"]),
        "mae_val_log10": float(result["mae_val_log10"]),
        "mean_validation_r2": float(result["r2_val_physical_mean"]),
        "mean_validation_mae_mpa": float(result["mae_val_physical_mean"]),
        "mae_test_log10": float(result["mae_test_log10"]),
        "mean_test_r2": float(result["r2_test_physical_mean"]),
        "mean_test_mae_mpa": float(result["mae_test_physical_mean"]),
        "best_epoch": int(result["best_epoch"]),
        "epochs_ran": int(result["epochs_ran"]),
        "num_parameters": int(result["num_parameters"]),
        "training_time_sec": float(result["training_time_sec"]),
        "time_to_best_val_sec": float(result["time_to_best_val_sec"]),
        "train_epoch_time_sec": float(result["train_epoch_time_sec"]),
        "val_epoch_time_sec": float(result["val_epoch_time_sec"]),
        "validation_inference_time_sec": float(result["val_eval_time_sec"]),
        "test_inference_time_sec": float(result["inference_time_test_sec"]),
        "inference_timing_repeats": int(result["inference_timing_repeats"]),
        "checkpoint": str(
            Path(condition) / model_name / f"seed_{seed}"
            / Path(result["ckpt"]).parent.name / "best.pt"
        ),
    }
    for index, target in enumerate(train.TARGET_NAMES):
        row[f"validation_r2_{target}"] = float(result["val_r2_by_target"][index])
        row[f"validation_mae_{target}_mpa"] = float(result["val_mae_by_target"][index])
        row[f"test_r2_{target}"] = float(result["test_r2_by_target"][index])
        row[f"test_mae_{target}_mpa"] = float(result["test_mae_by_target"][index])
    weights = result.get("learned_centrality_mixture_weights") or {}
    for entity in ("joint", "strut"):
        for centrality in ("degree", "closeness", "betweenness", "eigenvector"):
            row[f"{entity}_{centrality}_mixture_weight"] = weights.get(entity, {}).get(centrality, "")
    return row


# calculate five-seed means and sample standard deviations
def _summary_rows(rows: List[Dict]) -> List[Dict]:
    metrics = [
        "val_loss", "train_loss_at_best", "mae_val_log10", "mean_validation_r2",
        "mean_validation_mae_mpa", "mae_test_log10", "mean_test_r2",
        "mean_test_mae_mpa", "best_epoch", "epochs_ran", "training_time_sec",
        "time_to_best_val_sec", "test_inference_time_sec",
    ]
    metrics += [f"test_r2_{target}" for target in train.TARGET_NAMES]
    metrics += [f"test_mae_{target}_mpa" for target in train.TARGET_NAMES]
    metrics += [f"validation_r2_{target}" for target in train.TARGET_NAMES]
    metrics += [f"validation_mae_{target}_mpa" for target in train.TARGET_NAMES]
    metrics += [
        f"{entity}_{centrality}_mixture_weight"
        for entity in ("joint", "strut")
        for centrality in ("degree", "closeness", "betweenness", "eigenvector")
    ]
    summaries = []
    for condition, model_name in sorted({(row["condition"], row["model_name"]) for row in rows}):
        group = [row for row in rows if row["condition"] == condition and row["model_name"] == model_name]
        summary = {
            "condition": condition, "model_name": model_name,
            "num_seeds": len(group), "hidden": group[0]["hidden"],
            "depth": group[0]["depth"], "lr": group[0]["lr"],
            "wd": group[0]["wd"], "num_parameters": group[0]["num_parameters"],
        }
        for metric in metrics:
            values = [float(row[metric]) for row in group if row.get(metric, "") != ""]
            summary[f"{metric}_mean"] = mean(values) if values else ""
            summary[f"{metric}_std"] = stdev(values) if len(values) > 1 else (0.0 if values else "")
        summaries.append(summary)
    return summaries


# plot five seed mean test metrics for every stiffness component
def _write_final_test_metric_comparison(rows: List[Dict], out_root: Path) -> None:
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    model_names = list(train.PRIMARY_MODEL_NAMES)
    display_names = {
        "BipartiteGNN": "Bipartite GNN",
        "DynamicEdgeHomogeneousGNN": "Dynamic-edge\nhomogeneous GNN",
        "StaticEdgeHomogeneousGNN": "Static-edge\nhomogeneous GNN",
        "NodeEncoderANN": "Node-encoder ANN",
        "RawFeatureMeanPoolingANN": "Raw-feature mean-\npooling ANN",
    }
    condition_styles = {
        "with_centrality_features": {
            "label": "Centrality-guided",
            "color": "#007d85",
            "r2_marker": "o",
            "mae_marker": "s",
            "capsize": 6,
            "elinewidth": 1.8,
            "zorder": 3,
        },
        "without_centrality_features": {
            "label": "Uniform",
            "color": "#c2410c",
            "r2_marker": "D",
            "mae_marker": "X",
            "capsize": 3,
            "elinewidth": 1.1,
            "zorder": 4,
        },
    }
    x = np.arange(len(model_names), dtype=np.float64)
    all_r2 = np.asarray([
        float(row[f"test_r2_{target}"])
        for row in rows for target in train.TARGET_NAMES
    ], dtype=np.float64)
    finite_r2 = all_r2[np.isfinite(all_r2)]
    r2_floor = max(
        -1.0,
        float(np.percentile(finite_r2, 10) - 0.1) if finite_r2.size else -0.1,
    )

    fig, axes = train.plt.subplots(2, 3, figsize=(18, 10), dpi=180)
    legend_handles = []
    legend_labels = []
    for target_index, (target, target_label) in enumerate(
        zip(train.TARGET_NAMES, train.TARGET_LABELS)
    ):
        ax_r2 = axes.ravel()[target_index]
        ax_mae = ax_r2.twinx()
        for condition, style in condition_styles.items():
            condition_label = style["label"]
            color = style["color"]
            r2_means, r2_stds, mae_means, mae_stds = [], [], [], []
            for model_name in model_names:
                group = [
                    row for row in rows
                    if row["condition"] == condition
                    and row["model_name"] == model_name
                ]
                r2_values = [float(row[f"test_r2_{target}"]) for row in group]
                mae_values = [float(row[f"test_mae_{target}_mpa"]) for row in group]
                r2_means.append(mean(r2_values))
                r2_stds.append(stdev(r2_values) if len(r2_values) > 1 else 0.0)
                mae_means.append(mean(mae_values))
                mae_stds.append(stdev(mae_values) if len(mae_values) > 1 else 0.0)

            r2_means_array = np.asarray(r2_means, dtype=np.float64)
            clipped_r2 = np.maximum(r2_means_array, r2_floor)
            r2_line = ax_r2.errorbar(
                x, clipped_r2, yerr=r2_stds,
                color=color, marker=style["r2_marker"], linestyle="-",
                linewidth=1.5, markersize=6, capsize=style["capsize"],
                elinewidth=style["elinewidth"], zorder=style["zorder"],
                label=rf"$R^2$ - {condition_label} pooling",
            )
            mae_line = ax_mae.errorbar(
                x, mae_means, yerr=mae_stds,
                color=color, marker=style["mae_marker"],
                markerfacecolor="white", markeredgewidth=1.2,
                linestyle="--", linewidth=1.3, markersize=6,
                capsize=style["capsize"], elinewidth=style["elinewidth"],
                zorder=style["zorder"],
                label=f"MAE - {condition_label} pooling",
            )
            if target_index == 0:
                legend_handles.extend((r2_line, mae_line))
                legend_labels.extend((
                    rf"$R^2$ - {condition_label} pooling",
                    f"MAE - {condition_label} pooling",
                ))
            for model_index, actual in enumerate(r2_means_array):
                if actual < r2_floor:
                    ax_r2.scatter(
                        x[model_index], r2_floor,
                        marker="v", color=color, zorder=5,
                    )
                    ax_r2.annotate(
                        f"{actual:.2f}",
                        (x[model_index], r2_floor),
                        xytext=(0, 5), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7, color=color,
                    )

        ax_r2.axhline(0.0, color="0.55", linewidth=0.8)
        ax_r2.set_ylim(r2_floor, 1.05)
        ax_r2.set_ylabel(r"$R^2$")
        ax_mae.set_ylabel("MAE (MPa)")
        ax_mae.yaxis.set_major_locator(MaxNLocator(nbins=5))
        mae_formatter = ScalarFormatter(useMathText=True)
        mae_formatter.set_scientific(True)
        mae_formatter.set_powerlimits((-2, 2))
        ax_mae.yaxis.set_major_formatter(mae_formatter)
        ax_r2.set_title(target_label)
        ax_r2.set_xticks(x)
        ax_r2.set_xticklabels(
            [display_names[name] for name in model_names],
            rotation=20, ha="right",
        )
        ax_r2.grid(axis="y", alpha=0.25)

    fig.suptitle("Test Performance by Stiffness Component", y=0.99)
    fig.legend(
        legend_handles, legend_labels, loc="upper center", ncol=4,
        bbox_to_anchor=(0.5, 0.955), frameon=False,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    comparison_root = out_root / "comparison"
    comparison_root.mkdir(parents=True, exist_ok=True)
    fig.savefig(comparison_root / "test_metrics_by_stiffness.pdf")
    train.plt.close(fig)


# reuse a fit only when every required raw artifact is complete
def _completed_final_row(seed_root: Path) -> Dict | None:
    metric_paths = sorted(seed_root.glob("*/metrics.json"))
    if len(metric_paths) != 1:
        return None
    run_dir = metric_paths[0].parent
    required = (
        run_dir / "best.pt",
        run_dir / "loss_history.csv",
        run_dir / "test_predictions.csv",
        run_dir / "validation_predictions.csv",
        run_dir / "test_predictions.pdf",
        run_dir / "metrics.json",
    )
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return None
    row = json.loads(metric_paths[0].read_text(encoding="utf-8"))
    row.pop("learned_centrality_mixture_weights", None)
    required_keys = {
        "condition", "model_name", "train_seed", "val_loss",
        "mean_test_r2", "mean_test_mae_mpa", "checkpoint",
    }
    return row if required_keys.issubset(row) else None


# retrain selected configurations across five seeds and report test data
def run_final_evaluation(outputs_base: Path) -> Dict:
    root = experiment_root(outputs_base)
    selection_path = root / "merged_sweep" / "selected_configurations.json"
    if not selection_path.is_file():
        raise FileNotFoundError(f"run merge-sweeps first: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("screening_signature") != _screening_signature():
        raise RuntimeError(
            "code or screening configuration changed after merge-sweeps; "
            "use a new EXPERIMENT_NAME and rerun all sweep shards"
        )
    selected = selection["selected"]
    final_root = root / "final_evaluation"
    aggregate_path = final_root / "all_seed_results.csv"
    manifest_path = final_root / "manifest.json"
    if manifest_path.is_file() and json.loads(
        manifest_path.read_text(encoding="utf-8")
    ).get("complete") is True:
        raise FileExistsError(
            f"final evaluation is already complete: {manifest_path}. "
            "change EXPERIMENT_NAME before starting a new publication run."
        )
    final_root.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    try:
        for condition, enabled in CONDITIONS:
            CFG["STRUT_USE_CENTRALITY"] = enabled
            CFG["JOINT_USE_CENTRALITY"] = enabled
            train._LOADER_CACHE.clear()
            loaders = train._get_loaders_cached(include_centrality=enabled)
            condition_root = final_root / condition
            _set_output(condition_root)
            train._write_experiment_manifest(loaders, f"final_{condition}")
            for model_name in train.PRIMARY_MODEL_NAMES:
                cfg = selected[condition][model_name]
                for seed in CFG["FINAL_TRAIN_SEEDS"]:
                    run_root = final_root / condition / model_name / f"seed_{seed}"
                    completed_row = _completed_final_row(run_root)
                    if completed_row is not None:
                        print(f"[final] reusing complete result: {run_root}")
                        rows.append(completed_row)
                        _write_csv(aggregate_path, rows)
                        continue
                    _set_output(run_root)
                    result = train.train_one(
                        model_name, int(cfg["hidden"]), int(cfg["depth"]),
                        float(cfg["lr"]), float(cfg["wd"]),
                        tag_suffix=f"seed{seed}", loaders_override=loaders,
                        evaluate_test=True, train_seed=int(seed),
                        epochs=int(CFG["FINAL_EPOCHS"]),
                        patience=int(CFG["FINAL_PATIENCE"]),
                        save_artifacts=True,
                        save_plots=True,
                    )
                    row = _flatten_final_result(condition, model_name, seed, cfg, result)
                    rows.append(row)
                    run_dir = Path(result["ckpt"]).parent
                    (run_dir / "metrics.json").write_text(
                        json.dumps(_json_safe({
                            **row,
                            "learned_centrality_mixture_weights": result.get("learned_centrality_mixture_weights"),
                        }), indent=2), encoding="utf-8"
                    )
                    _write_csv(aggregate_path, rows)
        expected_fits = len(train.PRIMARY_MODEL_NAMES) * len(CONDITIONS) * len(CFG["FINAL_TRAIN_SEEDS"])
        run_keys = {
            (row["condition"], row["model_name"], int(row["train_seed"]))
            for row in rows
        }
        if len(rows) != expected_fits or len(run_keys) != expected_fits:
            raise RuntimeError(
                f"final evaluation is incomplete: found {len(run_keys)} of {expected_fits} unique fits"
            )
        _write_csv(final_root / "mean_and_std_across_seeds.csv", _summary_rows(rows))
        _write_final_test_metric_comparison(rows, final_root)
        manifest = {
            "complete": True,
            "selection_metric": "minimum screening validation MSE",
            "final_train_seeds": list(CFG["FINAL_TRAIN_SEEDS"]),
            "num_models": len(train.PRIMARY_MODEL_NAMES),
            "num_conditions": len(CONDITIONS),
            "num_final_fits": len(rows),
            "final_epochs": int(CFG["FINAL_EPOCHS"]),
            "final_patience": int(CFG["FINAL_PATIENCE"]),
            "selected_configurations": selected,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest
    finally:
        train._LOADER_CACHE.clear()
        CFG["STRUT_USE_CENTRALITY"] = True
        CFG["JOINT_USE_CENTRALITY"] = True
        _set_output(final_root)
