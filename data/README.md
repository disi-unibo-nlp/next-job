# Data Notes

This repository contains metadata for the full exported dataset, not the full raw JobHop data.

Included files:

- `full_metadata/export_summary.json`
- `full_metadata/export_summary.csv`
- `full_metadata/export_config.json`
- `full_metadata/export_config.csv`
- `full_metadata/dataset_statistics.csv`

The final formal split sizes used by the paper are:

- Train retained trajectories: 256,593
- Validation evaluation cases: 32,123
- Test evaluation cases: 32,089
- Train candidate occupation vocabulary size: 2,982

The full raw/exported data are intentionally not committed here because they are large and may have separate access requirements. To recreate the full working data directory, use the export pipeline in `src/export_jobhop_training_data.py` and the training handoff instructions under `training/`.

