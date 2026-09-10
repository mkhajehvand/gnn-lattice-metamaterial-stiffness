#!/bin/bash

set -euo pipefail

# snapshot the active configuration so queued jobs cannot observe later edits.
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"
config_source="${GNN_CONFIG_PATH:-$script_dir/config.json}"
if [[ ! -f "$config_source" ]]; then
  echo "configuration file not found: $config_source" >&2
  exit 1
fi

snapshot_dir="$script_dir/Outputs/config_snapshots"
mkdir -p "$snapshot_dir"
snapshot_stamp="$(date -u +%Y%m%dT%H%M%SZ)_$$"
config_snapshot="$snapshot_dir/config_${snapshot_stamp}.json"
cp "$config_source" "$config_snapshot"

# submit cache construction first, then release the array only after it succeeds.
cache_job=$(sbatch \
  --parsable \
  --export="ALL,GNN_CONFIG_PATH=$config_snapshot" \
  "$script_dir/sbatch_build_dataset_cache.sh")
sweep_job=$(sbatch \
  --parsable \
  --dependency="afterok:${cache_job}" \
  --export="ALL,GNN_CONFIG_PATH=$config_snapshot" \
  "$script_dir/sbatch_sweep_shards.sh")
final_job=$(sbatch \
  --parsable \
  --dependency="afterok:${sweep_job}" \
  --export="ALL,GNN_CONFIG_PATH=$config_snapshot" \
  "$script_dir/sbatch_final_evaluation.sh")

echo "saved configuration snapshot: ${config_snapshot}"
echo "submitted dataset-cache job: ${cache_job}"
echo "submitted dependent sweep array: ${sweep_job}"
echo "submitted dependent final-evaluation job: ${final_job}"
