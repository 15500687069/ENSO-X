# ENSO-X

ENSO-X is an independently designed model for long-lead ENSO prediction. It takes the previous 12 months of ocean-atmosphere fields as input and predicts the next 24 months of Niño3.4 evolution.

## Model Overview

ENSO-X uses a hybrid spatiotemporal framework with three main parts:

- Field encoder branch  
  Uses 3D CNN and Transformer blocks to extract large-scale spatiotemporal features.
- Memory branch  
  Uses seasonal state-space dynamics and a physically inspired dual-memory mechanism to represent recharge, wind forcing, and persistence.
- Lead repair branch  
  Uses local bridge, interpolation, and patch modules to repair difficult lead windows and improve long-lead stability.

## Data Setup

The current ENSO-X release uses the following setup:

- Training: `GODAS 1980-2014`
- Replay augmentation: `ORAS5 1980-2014`
- Main validation: `GODAS 2015-2021`
- External generalization test: `CMIP6 2015-2023`
- Long-range extrapolation test: `CMIP6 2015-2100`

The model uses 9 predictor variables:

- `thetao_5`
- `thetao_wmean`
- `tauu`
- `tauv`
- `uo_5`
- `vo_5`
- `psl`
- `mlotst`
- `sos`

The memory branch uses 3 derived features:

- `wwv_proxy`
- `trade_wind`
- `sst_basin_mean`

## Main Results

The released ENSO-X checkpoint achieves stable 24-month prediction skill in the main setting:

- `24/24` lead months with `corr > 0.5`

External CMIP6 evaluation:

- `CMIP6 2015-2023`: `corr > 0.5` through the first 24 months, and at least `corr > 0.2` through 48 months
- `CMIP6 2015-2100`: in the current zero-shot extrapolation test, `corr > 0.5` remains valid through at least 48 months

## Repository Layout

- `train.py`: training entry point
- `src/ensox/`: ENSO-X core code
- `configs/enso_x_24_final.yaml`: final reproducible training configuration
- `preprocess/`: preprocessing scripts used by the released package
- `scripts/evaluate_limit_enso_x.py`: long-range extrapolation evaluation
- `scripts/run_train_enso_x.sh`: training launcher
- `scripts/run_limit_eval_enso_x.sh`: extrapolation evaluation launcher
- `results/`: release result summaries

## Environment

The repository keeps three environment references:

- `requirements.txt`: minimal runtime dependencies
- `requirements-preprocess.txt`: preprocessing dependencies
- `environment.yml`: full environment file used for the release

## Data Preparation

ENSO-X expects processed data under:

```text
data/ctefnet_data/
  CMIP6var/
  ReanalysisVar/
    GODAS/
    ORAS5/
```

Preprocessing converts raw CMIP6, GODAS, and ORAS5 monthly fields into normalized `npz` files on a unified grid. Monthly anomalies are computed after detrending and removing climatology. The kept scripts are:

- `preprocess_cmip6_to_ensox.py`
- `preprocess_godas_to_ensox.py`
- `preprocess_oras5_to_ensox.py`

The CMIP6 list follows Table S1 after removing the two unavailable models. The released package uses:

- `ACCESS-CM2`
- `ACCESS-ESM1-5`
- `CanESM5`
- `CAS-ESM2-0`
- `CESM2`
- `CESM2-WACCM`
- `CNRM-CM6-1`
- `E3SM-1-0`
- `EC-Earth3`
- `FGOALS-g3`
- `IPSL-CM5A2-INCA`
- `IPSL-CM6A-LR`
- `MIROC6`
- `MRI-ESM2-0`
- `NorESM2-MM`
- `UKESM1-0-LL`

The two missing CMIP6 models are:

- `CMCC-ESM2`
- `FGOALS-f3-L`

Additional raw data used in preprocessing:

- `GODAS`
- `ORAS5`
- `ERA5 mean sea level pressure (msl)`

## Checkpoints

Core released checkpoints used in the experiments:

- `final_24_complete`
- `seed_24_run`

## Release Files

- `results/enso_x_summary.json`: main result summary
- `results/enso_x_generalization_20260425.json`: external generalization summary
- `results/enso_x_limit_eval_20260425.json`: extrapolation summary

## References

- Chen, Q., Cui, Y., Hong, G. et al. *Toward long-range ENSO prediction with an explainable deep learning model*. npj Climate and Atmospheric Science 8, 259 (2025). DOI: `10.1038/s41612-025-01159-w`
- Zhou, L., Zhang, R.-H. & Tao, L. *AI-Enabled conditional nonlinear optimal perturbation enhances ensemble prediction of extreme El Niño events*. npj Climate and Atmospheric Science 9, 30 (2026). DOI: `10.1038/s41612-025-01303-6`
- Zhang, Z., Meng, J., Qiu, Z. et al. *Enhancing the predictability limits of ENSO with physics-guided deep echo state networks*. npj Climate and Atmospheric Science 9, 92 (2026). DOI: `10.1038/s41612-026-01360-5`
