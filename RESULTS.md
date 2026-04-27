# ENSO-X Results

## Main Validation

Released ENSO-X checkpoint on `GODAS 2015-2021`:

- `24/24` lead months with `corr > 0.5`
- minimum lead correlation: `0.5129`
- mean lead correlation: `0.6784`

Lead correlations from lead 1 to lead 24:

```text
0.9471, 0.9330, 0.9009, 0.8501, 0.7895, 0.7646,
0.7129, 0.6827, 0.6515, 0.6271, 0.6334, 0.6467,
0.6568, 0.6579, 0.7072, 0.6966, 0.6053, 0.5129,
0.5221, 0.5810, 0.5840, 0.5268, 0.5349, 0.5559
```

## Ablation Summary

The ablation uses the same `GODAS 2015-2021` validation period. The legal analog initializer is rebuilt from training-period data only: `GODAS 1980-2014` plus ORAS5 replay, with `pca_dim=128` and `ridge_alpha=0.1`.

| Variant | Frontier | Min corr | Mean corr |
| --- | ---: | ---: | ---: |
| Full ENSO-X | 24 | 0.5099 | 0.6728 |
| No memory input | 21 | 0.4936 | 0.6743 |
| Field branch only | 0 | -0.3277 | 0.3002 |
| No local lead repair | 24 | 0.5097 | 0.6726 |
| No legal analog repair | 15 | 0.3546 | 0.6363 |
| No reanalysis repair | 7 | 0.2479 | 0.4604 |

## Audit

The data audit confirms that the GODAS training period ends before the validation period starts:

- Training: `GODAS 1980-2014`
- Validation: `GODAS 2015-2021`
- Label clamp: validation labels are limited to the available GODAS period ending at `2021-12`
- Replay data: ORAS5 is used only as training replay augmentation

See `results/enso_x_leakage_audit.json` and `results/enso_x_ablation_clean_legal_analog_pca128_alpha0p1_20260426.json` for machine-readable summaries.
