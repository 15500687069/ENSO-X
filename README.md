# ENSO-X

ENSO-X is an independently designed model for long-lead ENSO prediction under a small-sample reanalysis setting. The model uses the previous 12 months of ocean-atmosphere fields to predict the next 24 months of the Nino3.4 index.

## Highlights

- **Small-sample long-lead prediction**: ENSO-X is trained mainly on GODAS reanalysis data and achieves stable 24-month prediction skill on a strictly later GODAS validation period.
- **Continuous 24-month skill**: the released GODAS checkpoint reaches `24/24` lead months with `corr > 0.5` on `GODAS 2015-2021`.
- **Physical memory design**: the memory branch uses warm-water-volume, trade-wind, and basin-mean SST proxies to represent recharge, wind forcing, and persistence.
- **Ablation-tested repair design**: removing the memory input reduces the continuous GODAS skill frontier from 24 to 21 months, while removing the final reanalysis-consistent repair layer reduces it from 24 to 6 months.
- **Extreme-event diagnosis**: strong warm-event amplitude is no longer systematically underestimated in the main GODAS validation; stricter split-calibration scripts are included for conservative event-repair testing.

## Model Overview

ENSO-X uses a hybrid spatiotemporal forecasting framework:

- **Field encoder branch**: 3D CNN and Transformer blocks extract large-scale spatiotemporal features from ocean-atmosphere fields.
- **Memory branch**: a physics-inspired dual-memory mechanism models recharge memory, wind forcing, and persistence effects.
- **Lead repair branch**: local bridge, interpolation, patch modules, and a reanalysis-consistent analog repair layer stabilize difficult lead windows and improve long-lead continuity.

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
- diagnostic same-period calibration can improve event recall, but the stricter train-period calibration split currently selects no extra gain and should be treated as the conservative result

## Ablation Results

Systematic ablation is evaluated on the released checkpoint using the same `GODAS 2015-2021` validation period. The main metric is the continuous skill frontier, defined as the number of consecutive lead months with `corr > 0.5`.

| Variant | Frontier | Min corr | Mean corr | Warm F1 | Cold F1 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full ENSO-X | 24 | 0.5099 | 0.6728 | 0.414 | 0.083 |
| No memory input | 21 | 0.4936 | 0.6743 | 0.414 | 0.087 |
| Field branch only | 0 | -0.3277 | 0.3002 | 0.000 | 0.419 |
| No local lead repair | 24 | 0.5097 | 0.6726 | 0.400 | 0.083 |
| No legal analog repair | 15 | 0.3546 | 0.6363 | 0.400 | 0.400 |
| No reanalysis repair | 7 | 0.2479 | 0.4604 | 0.310 | 0.000 |
| Extreme calibrated | 24 | 0.5036 | 0.6672 | 0.500 | 0.457 |

Component conclusions:

- The memory branch is most useful as a conditioning signal for stable long-lead prediction, not as a standalone predictor.
- The final lead-repair contribution is dominated by the reanalysis-consistent analog repair layer; local patch/refiner modules are retained as stabilizers but are not the main source of the released 24-month skill.
- The optional extreme-event calibration improves event recall and F1 while keeping all 24 lead correlations above 0.5, but it should be reported separately from the default deterministic checkpoint because it increases event amplitude.

The table above is a single-checkpoint intervention ablation with a train-period-only legal analog initializer (`GODAS 1980-2014` plus ORAS5 replay, `pca_dim=128`, `ridge_alpha=0.1`). For a stricter paper-style protocol, this repository also includes retrained multi-seed ablation configs and scripts. Those runs train each ablated variant under the same data split and then summarize mean/std across seeds.

The same ablation script also includes an ORAS5 replay-domain consistency check. ORAS5 is used here only to test replay-domain behavior, not as a replacement for the main GODAS validation.

## Repository Layout

- `train.py`: training entry point
- `src/ensox/`: ENSO-X model, loss, metrics, and utilities
- `configs/enso_x_24_final.yaml`: main reproducible configuration
- `configs/ablation_retrain/`: multi-seed retrained ablation configurations
- `preprocess/preprocess_godas_to_ensox.py`: GODAS preprocessing
- `preprocess/preprocess_oras5_to_ensox.py`: ORAS5 preprocessing
- `scripts/evaluate_extreme_enso_x.py`: extreme-event evaluation
- `scripts/evaluate_probabilistic_enso_x.py`: probabilistic pilot evaluation
- `scripts/evaluate_ablation_enso_x.py`: component ablation evaluation
- `scripts/audit_data_leakage_enso_x.py`: data-split and label-clamp audit
- `scripts/build_legal_analog_init.py`: train-period-only analog repair initializer
- `scripts/run_retrained_ablation.sh`: multi-seed retrained ablation launcher
- `scripts/summarize_retrained_ablation.py`: retrained ablation mean/std summary
- `scripts/evaluate_extreme_calibration_split_enso_x.py`: split-based extreme-event calibration test
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

The preprocessing scripts convert raw monthly reanalysis fields into normalized `npz` files on a unified grid. The expected output includes predictor variables and the Nino3.4 target index for GODAS and ORAS5.

Variable notes:

- GODAS variables are mapped to the common ENSO-X variable names used in the configuration.
- ORAS5 variables are mapped to the same common names.
- `thetao_wmean` is treated as the warm-water-volume-related thermal memory field.
- The memory features are computed from selected tropical Pacific regions after preprocessing.

Run the ablation evaluation:

```bash
python scripts/evaluate_ablation_enso_x.py \
  --base-config configs/enso_x_24_final.yaml \
  --ckpt "$ENSOX_INIT_CKPT" \
  --data-root "$ENSOX_DATA_ROOT" \
  --output-json results/enso_x_ablation_godas_oras5_20260426.json \
  --tags GODAS_2015_2021 ORAS5_1958_1978
```

Run the stricter retrained ablation protocol:

```bash
export ENSOX_DATA_ROOT=/path/to/data/ctefnet_data
export ENSOX_ABLATION_OUTPUT_ROOT=/path/to/ablation_runs
bash scripts/run_retrained_ablation.sh
```

## Contact

- Contact: Yuzhi Wang (wangyzh267@mail2.sysu.edu.cn)
- Institution: College of Atmospheric Sciences, Sun Yat-sen University

## References

- Chen, Q., Cui, Y., Hong, G. et al. *Toward long-range ENSO prediction with an explainable deep learning model*. npj Climate and Atmospheric Science, 2025. DOI: `10.1038/s41612-025-01159-w`
- Zhou, L., Zhang, R.-H. & Tao, L. *AI-enabled conditional nonlinear optimal perturbation enhances ensemble prediction of extreme El Nino events*. npj Climate and Atmospheric Science, 2026. DOI: `10.1038/s41612-025-01303-6`
- Zhang, Z., Meng, J., Qiu, Z. et al. *Enhancing the predictability limits of ENSO with physics-guided deep echo state networks*. npj Climate and Atmospheric Science, 2026. DOI: `10.1038/s41612-026-01360-5`
