# ENSO-X

ENSO-X is an independently designed ENSO forecasting model for long-lead Nino3.4 prediction.
The model takes the previous 12 months of ocean-atmosphere fields as input and predicts the
next 24 months of the Nino3.4 index.

## Overview

ENSO-X uses a hybrid spatiotemporal forecasting framework with three main components:

- A field encoder branch built on 3D CNNs and Transformer layers for large-scale
  spatiotemporal feature extraction.
- A memory branch that combines seasonal state-space dynamics with a physically inspired
  dual-memory mechanism to represent recharge, wind forcing, and persistence effects.
- A lead-repair branch that improves difficult lead windows and stabilizes long-range
  forecasts through local bridge, interpolation, and patch modules.

## Data Setup

The current ENSO-X release was trained and evaluated with the following setup:

- Training: `GODAS 1980-2014`
- Replay augmentation: `ORAS5 1980-2014`
- Main validation: `GODAS 2015-2021`
- External generalization test: `CMIP6 2015-2023`
- Long-range extrapolation test: `CMIP6 2015-2100`

The predictor set contains 9 ocean-atmosphere variables:

- `thetao_5`
- `thetao_wmean`
- `tauu`
- `tauv`
- `uo_5`
- `vo_5`
- `psl`
- `mlotst`
- `sos`

The memory branch uses three derived memory features:

- `wwv_proxy`
- `trade_wind`
- `sst_basin_mean`

## Performance

The released ENSO-X checkpoint achieves stable 24-month prediction skill on the main setup.

- `24/24` lead months have `corr > 0.5`
- Minimum monthly correlation: `0.5089`
- The barrier segment around lead `9-11` is also repaired above `0.5`

On external CMIP6 evaluation:

- `CMIP6 2015-2023`: the model remains above `0.5` through `24` months and above `0.2`
  through at least `48` months
- `CMIP6 2015-2100`: the model remains above `0.5` through at least `48` months in the
  current zero-shot extrapolation test

Practical summary for the released model:

- Effective forecast skill: `24 months`
- Conservative extrapolation capability: `48 months`

## Repository Layout

- `train.py`: training entry point
- `src/ensox/`: core ENSO-X package
- `configs/enso_x_24_final.yaml`: final reproducible training configuration
- `preprocess/`: selected preprocessing scripts used by the released model
- `scripts/evaluate_limit_enso_x.py`: long-range extrapolation evaluator
- `scripts/run_train_enso_x.sh`: training launcher
- `scripts/run_limit_eval_enso_x.sh`: extrapolation-evaluation launcher
- `checkpoints/`: checkpoint manifest and deployment notes
- `results/`: saved release summaries

## Environment

Two environment files are included:

- `requirements.txt`: minimal runtime dependencies
- `requirements-preprocess.txt`: minimal preprocessing dependencies
- `environment.yml`: exported server environment used for the released ENSO-X package

Example setup with conda:

```bash
conda env create -f environment.yml
conda activate enso_x
```

## Data Preparation

ENSO-X expects processed files under:

```text
data/ctefnet_data/
  CMIP6var/
  ReanalysisVar/
    GODAS/
    ORAS5/
```

The preprocessing scripts kept in `preprocess/` are the exact script family matched to the
released data products on the server.

- `preprocess_cmip6_to_ensox.py`
- `preprocess_godas_to_ensox.py`
- `preprocess_oras5_to_ensox.py`

See `preprocess/README.md` for:

- expected raw-data layout
- variable-specific preprocessing rules
- example preprocessing commands
- notes about ERA5 `msl`, weighted-mean variables, and derived `nino34`

## Checkpoints

The clean release keeps three checkpoint groups:

- `final_24_complete`: final released ENSO-X checkpoint
- `seed_24_run`: direct seed checkpoint used to reproduce the final 24-month run
- `lead23_baseline`: earlier high-lead baseline for comparison

See `checkpoints/MANIFEST.json` for exact names, roles, metrics, and deployed server paths.

In the deployed server package, `checkpoints/` also contains symlinks with the same alias names.

## Reproducing The Release

### 1. Set paths

If your processed data are not stored under `./data/ctefnet_data`, export these variables:

```bash
export ENSOX_DATA_ROOT=/path/to/ctefnet_data
export ENSOX_INIT_CKPT=/path/to/seed_24_run/best_frontier.ckpt
export ENSOX_OUTPUT_ROOT=/path/to/save/checkpoints
```

The final config supports `${VAR}` and `${VAR:-default}` environment expansion.

### 2. Reproduce the final 24-month run

```bash
bash scripts/run_train_enso_x.sh
```

Equivalent manual command:

```bash
python train.py --config ./configs/enso_x_24_final.yaml
```

### 3. Evaluate the released final checkpoint

```bash
bash scripts/run_limit_eval_enso_x.sh ./checkpoints/final_24_complete/best.ckpt ./results/enso_x_limit_eval.json
```

Equivalent manual command:

```bash
python scripts/evaluate_limit_enso_x.py \
  --base-config ./configs/enso_x_24_final.yaml \
  --ckpt ./checkpoints/final_24_complete/best.ckpt \
  --data-root "${ENSOX_DATA_ROOT:-./data/ctefnet_data}" \
  --output-json ./results/enso_x_limit_eval.json
```

## Release Artifacts

- `results/enso_x_summary.json`: short release summary
- `results/enso_x_generalization_20260425.json`: external generalization summary
- `results/enso_x_limit_eval_20260425.json`: extrapolation summary used in the release notes
