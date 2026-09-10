# command line entry point for cache, sweep, merge, and final evaluation

# available workflow modes used directly or by the Slurm scripts
# python -m gnn_lattice_metamaterial_stiffness.script --mode build-dataset-cache
# python -m gnn_lattice_metamaterial_stiffness.script --mode sweep-shard --shard-id 0, 1, 2, or 3
# python -m gnn_lattice_metamaterial_stiffness.script --mode merge-sweeps
# python -m gnn_lattice_metamaterial_stiffness.script --mode final-evaluation

import os
import sys
import argparse
import json
import time
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr

# use a non-interactive backend on laptops and compute nodes
os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib
matplotlib.use("Agg", force=True)

from .lattice_gnn.config import CFG
from .lattice_gnn.plot_style import configure_matplotlib
configure_matplotlib()

_OUTPUTS_BASE = Path(CFG["OUTPUTS_DIR"])
_MODE_OUTPUT_NAMES = {
    "build-dataset-cache": "example_structure_plots",
}

# write each message to both the terminal and the experiment log file
class _ConsoleTee:

    def __init__(self, *streams):
        self._streams = streams

    def write(self, data):
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self):
        for stream in self._streams:
            stream.flush()

# format local wall time for progress logs
def timestamp():
    return time.strftime("%Y-%m-%d %H:%M:%S")


# apply the same 4 thread policy locally and under Slurm
def _set_threads_from_cfg():
    # apply the same four-thread limit in local and slurm runs
    v = int(CFG["CPU_THREADS"])
    if v > 0:
        for k in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS","NUMEXPR_NUM_THREADS","VECLIB_MAXIMUM_THREADS"):
            os.environ.setdefault(k, str(v))
        try:
            import torch
            torch.set_num_threads(v)
            torch.set_num_interop_threads(min(2, v))
        except Exception:
            pass

# create and synchronize output paths used by all modules
def _outputs_root():
    root = CFG["OUTPUTS_DIR"]
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    # keep both output aliases synchronized for plotting helpers
    CFG["OUTPUTS_DIR"] = str(p)
    CFG["OUTPUT_DIR"] = str(p)
    Path(CFG["OUTPUT_DIR"]).mkdir(parents=True, exist_ok=True)
    return p

# resolve the dedicated output directory for cache construction
def _mode_output_dir(mode: str) -> Path:
    return _OUTPUTS_BASE / _MODE_OUTPUT_NAMES[mode]

# give each workflow stage an independent persistent text log
def _terminal_log_path(mode: str, shard_id: int | None = None) -> Path:
    if mode == "build-dataset-cache":
        return _OUTPUTS_BASE / "build_dataset_cache.log"
    name = str(CFG["EXPERIMENT_NAME"])
    suffix = f"_{shard_id}" if shard_id is not None else ""
    return _OUTPUTS_BASE / "experiments" / name / "logs" / f"{mode}{suffix}.log"

# activate a mode-specific output directory before importing workers
def _activate_mode_output(mode: str) -> Path:
    p = _mode_output_dir(mode)
    CFG["OUTPUTS_DIR"] = str(p)
    CFG["OUTPUT_DIR"] = str(p)
    return _outputs_root()

# build the reusable graph cache and render example structures
def _mode_build_dataset_cache():
    print(f"[{timestamp()}] [script] building dataset cache …")

    # load or build the cache and populate visualization structures
    from .lattice_gnn import data
    data.load_datasets()
    print(f"[{timestamp()}] [script] dataset cache ready.")

    # build mode writes the example structure plots
    out_pre = Path(data.CFG["OUTPUTS_DIR"])
    out_pre.mkdir(parents=True, exist_ok=True)

    # use a fixed representative set unless STRUCTURES is configured
    # plot these structures as examples
    struct_names = data.CFG.get(
        "STRUCTURES",
        ["cub_Z06.0_E1", "cub_Z08.0_E3", "cub_Z14.0_E38",
         "cub_Z12.0_E19", "cub_Z04.0_E970", "ort_Z04.3_E319"]
    )

    print(f"[{timestamp()}] [script] rendering structures → {out_pre}")

    # intercept plt.show so the visualization code saves deterministic pdfs
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # preserve structure order in readable file names
    def _save_show():
        # file name mirrors the structure order
        idx = getattr(_save_show, "_i", 0)
        fname = f"{idx:02d}_{struct_names[idx] if idx < len(struct_names) else 'struct'}.pdf"
        plt.savefig(out_pre / fname, bbox_inches="tight")
        plt.close()
        _save_show._i = idx + 1
    _save_show._i = 0
    plt.show = _save_show

    # import after loading so viz sees the populated structure collection
    from .lattice_gnn import viz
    viz.plot_structures(struct_names)

    print(f"[{timestamp()}] [script] saved {len(struct_names)} structure figures.")
    print(f"[{timestamp()}] [script] build-dataset-cache finished.")

# run one disjoint model shard across both pooling conditions
def _mode_sweep_shard(shard_id: int):
    from .lattice_gnn.workflow import run_sweep_shard
    print(f"[{timestamp()}] [script] starting sweep shard {shard_id} …")
    result = run_sweep_shard(shard_id, _OUTPUTS_BASE)
    print(f"[{timestamp()}] [script] sweep shard finished: {result}")


# validate sweep shards and select configurations by validation MSE
def _mode_merge_sweeps():
    from .lattice_gnn.workflow import merge_sweep_shards
    print(f"[{timestamp()}] [script] merging sweep shards …")
    result = merge_sweep_shards(_OUTPUTS_BASE)
    print(f"[{timestamp()}] [script] selected configurations: {json.dumps(result, indent=2)}")


# retrain selected configurations across the configured final seeds
def _mode_final_evaluation():
    from .lattice_gnn.workflow import run_final_evaluation
    print(f"[{timestamp()}] [script] starting final evaluation …")
    result = run_final_evaluation(_OUTPUTS_BASE)
    print(f"[{timestamp()}] [script] final evaluation finished: {result}")

# parse the workflow mode and tee console output to a persistent log
def main():
    parser = argparse.ArgumentParser(description="lattice gnn pipeline")
    parser.add_argument(
        "--mode",
        choices=["build-dataset-cache", "sweep-shard", "merge-sweeps", "final-evaluation"],
        required=True,
        help="build the cache, run one sweep shard, merge shards, or evaluate selected models"
    )
    parser.add_argument("--shard-id", type=int, choices=range(4))
    args = parser.parse_args()

    _set_threads_from_cfg()
    if args.mode == "sweep-shard" and args.shard_id is None:
        parser.error("--shard-id is required for sweep-shard")
    if args.mode == "build-dataset-cache":
        _activate_mode_output(args.mode)
    log_path = _terminal_log_path(args.mode, args.shard_id)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as log_file:
        tee_out = _ConsoleTee(sys.stdout, log_file)
        tee_err = _ConsoleTee(sys.stderr, log_file)
        with redirect_stdout(tee_out), redirect_stderr(tee_err):
            print(f"[{timestamp()}] [script] logging stdout/err → {log_path}")
            if args.mode == "build-dataset-cache":
                _mode_build_dataset_cache()
            elif args.mode == "sweep-shard":
                _mode_sweep_shard(args.shard_id)
            elif args.mode == "merge-sweeps":
                _mode_merge_sweeps()
            elif args.mode == "final-evaluation":
                _mode_final_evaluation()

if __name__ == "__main__":
    main()
