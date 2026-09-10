#!/bin/bash -l

#SBATCH --job-name=gnn_sweep
#SBATCH --output=gnn_sweep_%A_%a.log
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --ntasks=1
#SBATCH --array=0-3
#SBATCH --mail-user=milad.khajehvand@ucr.edu
#SBATCH --mail-type=ALL

set -euo pipefail

module load gcc/12.2.0
module load cuda/12.4
module load miniconda3/py39_4.12.0
conda activate gnn_project

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR/.."

python -m gnn_lattice_metamaterial_stiffness.script \
  --mode sweep-shard \
  --shard-id "$SLURM_ARRAY_TASK_ID"
