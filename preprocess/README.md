# ENSO-X Preprocessing

This folder contains the preprocessing scripts selected from the `jiangshr` server that
match the data layout used by the released ENSO-X model.

## Included Pipelines

- `preprocess_cmip6_to_ensox.py`
  Builds `data/ctefnet_data/CMIP6var` from raw CMIP6 NetCDF files.
  This is the CMIP6 pipeline used to generate the predictor and `nino34` npz files
  consumed by ENSO-X for external testing and extrapolation analysis.

- `preprocess_godas_to_ensox.py`
  Builds `data/ctefnet_data/ReanalysisVar/GODAS` from raw GODAS files together with
  ERA5 mean sea level pressure.
  This is the GODAS-side preprocessing path corresponding to the released ENSO-X setup.

- `preprocess_oras5_to_ensox.py`
  Builds `data/ctefnet_data/ReanalysisVar/ORAS5` from raw ORAS5 files together with
  ERA5 mean sea level pressure.
  This matches the ORAS5 replay data format used by ENSO-X.

## Auxiliary Utilities

- `check_cmip6_raw_integrity.py`
  Performs raw CMIP6 NetCDF integrity checks before long preprocessing runs.

- `run_cmip6_preprocess_guard.sh`
  A retry-safe guard runner for long CMIP6 preprocessing jobs.

## Notes

- ENSO-X keeps the same on-disk npz layout as the earlier training pipeline for
  compatibility with the released loaders.
- The released ENSO-X model uses the following processed variables:
  `thetao_5`, `thetao_wmean`, `tauu`, `tauv`, `uo_5`, `vo_5`, `psl`, `mlotst`, `sos`,
  plus `nino34`.
- The reanalysis pipelines therefore also generate the mapped forms used internally by
  ENSO-X, such as `pottmp_5`, `pottmp_wmean`, `ucur_5`, `vcur_5`, `tau_x`, `tau_y`,
  `dbss_obml`, `salt`, `msl` for GODAS, and the ORAS5 equivalents.

## Variable-Specific Rules

Some variables are intentionally preprocessed differently from the rest. These rules are
part of the released ENSO-X data pipeline and should be preserved when reproducing the
model.

### CMIP6

- `thetao_5`
  Extracted with `surface5` mode from the upper-ocean temperature field.
- `thetao_wmean`
  Computed as an upper-ocean weighted mean. If weighted-mean extraction fails, the
  script can fall back to `thetao_5` via `--wmean-fallback thetao_5`.
- `uo_5`, `vo_5`
  Extracted with `surface5` mode rather than direct 2D loading.
- `psl`, `tauu`, `tauv`, `mlotst`, `sos`
  Treated as direct 2D variables.
- `psl`, `tauu`, `tauv`
  Listed in `NO_FILL_VARS`, so they keep their original missing-value behavior and are
  not zero-filled like the other variables.
- `nino34`
  Not read directly from raw files. It is derived from the anomaly of `thetao_5` and
  then min-max normalized before saving.

### GODAS

- `pottmp_5`, `ucur_5`, `vcur_5`, `salt`
  Extracted with `surface5` mode from 3D ocean fields.
- `pottmp_wmean`
  Computed as an upper-ocean weighted mean.
- `tau_x`, `tau_y`, `dbss_obml`
  Treated as direct variables.
- `msl`
  Not taken from GODAS itself. It is imported from ERA5 mean sea level pressure.
- `msl`, `tau_x`, `tau_y`
  Listed in `NO_FILL_VARS`, so they keep original missing behavior and are not
  zero-filled.
- `nino34`
  Derived from the anomaly of `pottmp_5`, then min-max normalized.

### ORAS5

- `votemper_5`, `vozocrtx_5`, `vomecrty_5`
  Extracted with `surface5` mode from 3D ocean fields.
- `votemper_wmean`
  Computed as an upper-ocean weighted mean.
- `sosaline`, `tau_x`, `tau_y`, `somxl030`
  Treated as direct variables.
- `msl`
  Imported from ERA5 mean sea level pressure.
- `msl`, `tau_x`, `tau_y`
  Listed in `NO_FILL_VARS`, so they keep original missing behavior and are not
  zero-filled.
- `nino34`
  Derived from the anomaly of `votemper_5`, then min-max normalized.

## Expected Raw Layout

The scripts are designed to write into the released ENSO-X processed layout:

```text
data/ctefnet_data/
  CMIP6var/
  ReanalysisVar/
    GODAS/
    ORAS5/
```

Typical raw-data layout used by these scripts:

```text
raw/
  cmip6/
    ACCESS-CM2/
    ACCESS-ESM1-5/
    ...
  godas/
    *.nc
  oras5/
    *.nc
  era5_msl/
    *.nc
```

The exact CMIP6 per-model file names can vary, but each model directory must contain the raw
NetCDF files needed by the variable tokens expected in `preprocess_cmip6_to_ensox.py`.

## Example Commands

### CMIP6

```bash
python preprocess/preprocess_cmip6_to_ensox.py \
  --raw-root ./raw/cmip6 \
  --out-root ./data/ctefnet_data/CMIP6var \
  --wmean-fallback thetao_5
```

For long CMIP6 runs you can use the guard wrapper:

```bash
bash preprocess/run_cmip6_preprocess_guard.sh
```

### GODAS

```bash
python preprocess/preprocess_godas_to_ensox.py \
  --godas-root ./raw/godas \
  --era5-root ./raw/era5_msl \
  --out-root ./data/ctefnet_data/ReanalysisVar/GODAS
```

### ORAS5

```bash
python preprocess/preprocess_oras5_to_ensox.py \
  --oras5-root ./raw/oras5 \
  --era5-root ./raw/era5_msl \
  --out-root ./data/ctefnet_data/ReanalysisVar/ORAS5
```

## Output Signatures

The released processed files on the server match this script family by:

- directory layout
- fixed file names such as `GODAS_reanalysis_198001-202312.npz` and
  `ORAS5_reanalysis_195801-202312.npz`
- `npz` keys `data` and `mean_map` for predictor fields
- `npz` key `data` for `nino34`
- predictor value range normalized to `0..1`
