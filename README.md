# Centrality-Guided Joint–Strut Bipartite Graph Neural Networks for Stiffness Prediction of Lattice Metamaterials

This repository contains the code and data [to be added] associated with the manuscript.

## Repository structure

```text
gnn_lattice_metamaterial_stiffness/
  config.json
  data/
    README.md
  fonts/
    README.md
  lattice_gnn/
  script.py
  submit_full_pipeline.sh
  sbatch_build_dataset_cache.sh
  sbatch_sweep_shards.sh
  sbatch_final_evaluation.sh
README.md
LICENSE
THIRD_PARTY_NOTICES.md
requirements.txt
requirements-hpcc.txt
results/
  README.md
```

## Usage

1. Download the unit cell catalog published by Lumpe and Stankovic (2021) from the
   [ETH Zurich Research Collection](https://doi.org/10.3929/ethz-b-000457598).
   The associated article is available at
   [Exploring the property space of periodic cellular structures based on crystal networks](https://doi.org/10.1073/pnas.2003504118). Place the downloaded `Unit_Cell_Catalog.txt` file inside
   `gnn_lattice_metamaterial_stiffness/data/`.

2. Install the required dependencies using `requirements.txt` or `requirements-hpcc.txt`.

3. Set the desired experiment parameters in
   `gnn_lattice_metamaterial_stiffness/config.json`. Assign a unique `EXPERIMENT_NAME`.

### HPCC

The supplied Slurm submission scripts are configured for UCR HPCC and may require adaptation for other computing clusters. Enter the package directory and submit the complete workflow.

```bash
cd gnn_lattice_metamaterial_stiffness
bash submit_full_pipeline.sh
```

This creates three dependent Slurm jobs:

- sbatch_build_dataset_cache.sh: Reads the unit cell catalog, processes the structures, and stores the reusable dataset.
- sbatch_sweep_shards.sh: The four-task sweep array evaluates every applicable hyperparameter combination for all five model families with centrality-guided and uniform pooling.
- sbatch_final_evaluation.sh: Merges the sweep results, selects the lowest-validation-MSE configuration for every model and pooling method, retrains each selected configuration using the five final training seeds, and evaluates the retained checkpoints on the test set.

Logs, sweep summaries, checkpoints, predictions, metrics, and figures are written under `gnn_lattice_metamaterial_stiffness/Outputs/experiments/<EXPERIMENT_NAME>/`.

### Local

For a local run, reduce the hyperparameter grid and training settings in `gnn_lattice_metamaterial_stiffness/config.json`. Then, from the repository root, sequentially run:

```bash
python -m gnn_lattice_metamaterial_stiffness.script --mode build-dataset-cache
python -m gnn_lattice_metamaterial_stiffness.script --mode sweep-shard --shard-id 0
python -m gnn_lattice_metamaterial_stiffness.script --mode sweep-shard --shard-id 1
python -m gnn_lattice_metamaterial_stiffness.script --mode sweep-shard --shard-id 2
python -m gnn_lattice_metamaterial_stiffness.script --mode sweep-shard --shard-id 3
python -m gnn_lattice_metamaterial_stiffness.script --mode merge-sweeps
python -m gnn_lattice_metamaterial_stiffness.script --mode final-evaluation
```

## Results

[to be added]
