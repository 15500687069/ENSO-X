# ENSO-X

ENSO-X is an independently designed model for long-lead ENSO prediction under a small-sample reanalysis setting. The model uses the previous 12 months of ocean-atmosphere fields to predict the next 24 months of the Niño3.4 index.

## Highlights

- **Small-sample long-lead prediction**: ENSO-X is trained mainly on GODAS reanalysis data and achieves stable 24-month prediction skill on a strictly later GODAS validation period.
- **Continuous 24-month skill**: the released GODAS checkpoint reaches `24/24` lead months with `corr > 0.5` on `GODAS 2015-2021`.
- **Physical memory design**: the memory branch uses warm-water-volume, trade-wind, and basin-mean SST proxies to represent recharge, wind forcing, and persistence.
- **Extreme-event improvement**: strong warm-event amplitude is no longer systematically underestimated in the main GODAS validation, and an optional extreme-event calibration experiment further improves warm/cold event recall while keeping 24-month skill.


## Model Overview

ENSO-X uses a hybrid spatiotemporal forecasting framework:

- **Field encoder branch**: 3D CNN and Transformer blocks extract large-scale spatiotemporal features from ocean-atmosphere fields.
- **Memory branch**: a physics-inspired dual-memory mechanism models recharge memory, wind forcing, and persistence effects.
- **Lead repair branch**: local bridge, interpolation, and patch modules stabilize difficult lead windows and improve long-lead continuity.

## Data Setup

Current release setting:

- Training set: `GODAS 1980-2014`
- Replay augmentation: `ORAS5 1958-1978`
- Main validation set: `GODAS 2015-2021`

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

The memory branch uses 3 derived memory features:

- `wwv_proxy`
- `trade_wind`
- `sst_basin_mean`

## Main Results

Released GODAS checkpoint on `GODAS 2015-2021`:

- `24/24` lead months with `corr > 0.5`
- minimum lead correlation: `0.5129`
- mean lead correlation: `0.6784`

Extreme-event summary on `GODAS 2015-2021`:

- main checkpoint warm-event amplitude bias: `+0.031`
- warm-event recall: `0.667`
- optional calibration keeps `24/24` lead skill and improves warm-event recall to `0.889`

## Repository Layout

- `train.py`: training entry point
- `src/ensox/`: ENSO-X model, loss, metrics, and utilities
- `configs/enso_x_24_final.yaml`: main reproducible configuration
- `preprocess/preprocess_godas_to_ensox.py`: GODAS preprocessing
- `preprocess/preprocess_oras5_to_ensox.py`: ORAS5 preprocessing
- `scripts/evaluate_extreme_enso_x.py`: extreme-event evaluation
- `scripts/evaluate_probabilistic_enso_x.py`: probabilistic pilot evaluation
- `scripts/run_train_enso_x.sh`: training launcher
- `results/`: sanitized release summaries

## Data Preparation

ENSO-X expects processed reanalysis data under:

```text
data/ctefnet_data/
  ReanalysisVar/
    GODAS/
    ORAS5/
```

The preprocessing scripts convert raw monthly reanalysis fields into normalized `npz` files on a unified grid. The expected output includes predictor variables and the Niño3.4 target index for GODAS and ORAS5.

Variable notes:

- GODAS variables are mapped to the common ENSO-X variable names used in the configuration.
- ORAS5 variables are mapped to the same common names.
- `thetao_wmean` is treated as the warm-water-volume-related thermal memory field.
- The memory features are computed from selected tropical Pacific regions after preprocessing.


## Contact

- Contact: Yuzhi Wang (wangyzh267@mail2.sysu.edu.cn)
- Institution: College of Atmospheric Sciences, Sun Yat-sen University

## References

- Chen, Q., Cui, Y., Hong, G. et al. *Toward long-range ENSO prediction with an explainable deep learning model*. npj Climate and Atmospheric Science, 2025. DOI: `10.1038/s41612-025-01159-w`
- Zhou, L., Zhang, R.-H. & Tao, L. *AI-enabled conditional nonlinear optimal perturbation enhances ensemble prediction of extreme El Niño events*. npj Climate and Atmospheric Science, 2026. DOI: `10.1038/s41612-025-01303-6`
- Zhang, Z., Meng, J., Qiu, Z. et al. *Enhancing the predictability limits of ENSO with physics-guided deep echo state networks*. npj Climate and Atmospheric Science, 2026. DOI: `10.1038/s41612-026-01360-5`
