#!/usr/bin/env python3
"""
Preprocess GODAS + ERA5-MSL raw NetCDF files into ENSO-X ReanalysisVar/GODAS npz files.

Output layout:
  <out_root>/pottmp_5/GODAS_reanalysis_198001-202312.npz
  <out_root>/pottmp_wmean/GODAS_reanalysis_198001-202312.npz
  <out_root>/ucur_5/GODAS_reanalysis_198001-202312.npz
  <out_root>/vcur_5/GODAS_reanalysis_198001-202312.npz
  <out_root>/tau_x/GODAS_reanalysis_198001-202312.npz
  <out_root>/tau_y/GODAS_reanalysis_198001-202312.npz
  <out_root>/dbss_obml/GODAS_reanalysis_198001-202312.npz
  <out_root>/salt/GODAS_reanalysis_198001-202312.npz
  <out_root>/msl/GODAS_reanalysis_198001-202312.npz
  <out_root>/nino34/GODAS_reanalysis_198001-202312.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr


TARGET_LATS = np.arange(-59.5, 60.0, 1.0, dtype=np.float32)   # 120
TARGET_LONS = np.arange(0.0, 360.0, 2.0, dtype=np.float32)    # 180
FULL_TIME_GODAS = pd.date_range("1980-01-01", "2023-12-01", freq="MS")

TIME_DIM_CANDS = ("time", "valid_time")
DEPTH_DIM_CANDS = ("level", "lev", "depth", "olevel", "z_t", "st_ocean")
LAT_DIM_CANDS = ("lat", "latitude", "y")
LON_DIM_CANDS = ("lon", "longitude", "x")

# Keep same behavior style as CMIP preprocessing.
NO_FILL_VARS = {"msl", "tau_x", "tau_y"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess GODAS + ERA5 MSL to ENSO-X npz.")
    parser.add_argument("--godas-root", type=str, required=True, help="Raw GODAS dir containing *.year.nc")
    parser.add_argument("--era5-root", type=str, required=True, help="Raw ERA5 msl dir containing monthly nc")
    parser.add_argument("--out-root", type=str, required=True, help="Output dir: .../ReanalysisVar/GODAS")
    parser.add_argument("--depth-max", type=float, default=300.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _lower_map(names: Iterable[str]) -> Dict[str, str]:
    return {str(n).lower(): str(n) for n in names}


def _find_name(candidates: Iterable[str], names: Iterable[str], kind: str) -> str:
    mp = _lower_map(names)
    for c in candidates:
        if c.lower() in mp:
            return mp[c.lower()]
    raise ValueError(f"Cannot find {kind} in {tuple(names)}")


def _open_dataset_with_fallback(fp: Path) -> xr.Dataset:
    last_err = None
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            kwargs = dict(decode_times=True, use_cftime=True, mask_and_scale=False)
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(fp, **kwargs)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"open failed: {fp}: {last_err}")


def _to_time_index(da: xr.DataArray) -> pd.DatetimeIndex:
    tname = _find_name(TIME_DIM_CANDS, da.dims, "time dim")
    if tname != "time":
        da = da.rename({tname: "time"})
    idx = da.indexes["time"]
    if hasattr(idx, "to_datetimeindex"):
        try:
            dt = idx.to_datetimeindex()
        except Exception:
            dt = pd.to_datetime([str(x) for x in idx])
    else:
        dt = pd.to_datetime(idx)
    return pd.DatetimeIndex([pd.Timestamp(int(d.year), int(d.month), 1) for d in dt])


def _normalize_lon(da: xr.DataArray) -> xr.DataArray:
    lon = np.mod(da["lon"].values.astype(np.float64), 360.0)
    da = da.assign_coords(lon=lon).sortby("lon")
    vals = da["lon"].values
    _, uniq_idx = np.unique(np.round(vals, 6), return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return da.isel(lon=uniq_idx)


def _standardize_2d(da: xr.DataArray) -> xr.DataArray:
    tname = _find_name(TIME_DIM_CANDS, da.dims, "time dim")
    lat_name = _find_name(LAT_DIM_CANDS, da.dims, "lat dim")
    lon_name = _find_name(LON_DIM_CANDS, da.dims, "lon dim")

    rename_map = {}
    if tname != "time":
        rename_map[tname] = "time"
    if lat_name != "lat":
        rename_map[lat_name] = "lat"
    if lon_name != "lon":
        rename_map[lon_name] = "lon"
    if rename_map:
        da = da.rename(rename_map)

    if np.any(np.diff(da["lat"].values.astype(np.float64)) < 0):
        da = da.sortby("lat")
    da = _normalize_lon(da)

    da = da.sel(lat=slice(-60.0, 60.0))
    da = da.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="linear")
    return da.transpose("time", "lat", "lon")


def _select_depth_dim(da: xr.DataArray) -> str:
    return _find_name(DEPTH_DIM_CANDS, da.dims, "depth dim")


def _surface_at_depth(da: xr.DataArray, target_depth: float = 5.0) -> xr.DataArray:
    depth_dim = _select_depth_dim(da)
    out = da.sel({depth_dim: target_depth}, method="nearest")
    return _standardize_2d(out)


def _weighted_mean_upper(da: xr.DataArray, depth_max: float) -> xr.DataArray:
    depth_dim = _select_depth_dim(da)
    depth_vals = np.asarray(da[depth_dim].values, dtype=np.float64)
    valid = np.isfinite(depth_vals) & (depth_vals >= 0.0) & (depth_vals <= float(depth_max))
    if not np.any(valid):
        return _surface_at_depth(da, 5.0)

    idx = np.where(valid)[0]
    da = da.isel({depth_dim: idx})
    sel_levels = depth_vals[idx]

    if int(sel_levels.size) == 1:
        da = da.isel({depth_dim: 0})
        return _standardize_2d(da)

    bounds = np.empty(int(sel_levels.size) + 1, dtype=np.float64)
    bounds[1:-1] = 0.5 * (sel_levels[:-1] + sel_levels[1:])
    first_gap = sel_levels[1] - sel_levels[0]
    last_gap = sel_levels[-1] - sel_levels[-2]
    bounds[0] = max(0.0, sel_levels[0] - 0.5 * first_gap)
    bounds[-1] = sel_levels[-1] + 0.5 * last_gap
    dz = np.diff(bounds).astype(np.float32)

    w = xr.DataArray(dz, dims=(depth_dim,), coords={depth_dim: da[depth_dim]})
    out = (da * w).sum(dim=depth_dim) / w.sum()
    return _standardize_2d(out)


def _finalize_concat(
    all_time: List[pd.DatetimeIndex], all_data: List[np.ndarray]
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    time = pd.DatetimeIndex(np.concatenate([t.values for t in all_time]))
    data = np.concatenate(all_data, axis=0)
    order = np.argsort(time.values)
    time = pd.DatetimeIndex(time.values[order])
    data = data[order]
    _, uniq_idx = np.unique(time.values, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return pd.DatetimeIndex(time.values[uniq_idx]), data[uniq_idx]


def _load_concat_godas_var(
    root: Path, var_name: str, mode: str, depth_max: float
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    files = sorted(root.glob(f"{var_name}.*.nc"))
    if not files:
        raise FileNotFoundError(f"{root}/{var_name}.*.nc not found")

    all_time: List[pd.DatetimeIndex] = []
    all_data: List[np.ndarray] = []
    bad: List[str] = []

    for fp in files:
        try:
            with _open_dataset_with_fallback(fp) as ds:
                if var_name not in ds:
                    bad.append(f"{fp.name}: var_missing")
                    continue
                da = ds[var_name]
                if mode == "direct":
                    out = _standardize_2d(da)
                elif mode == "surface5":
                    out = _surface_at_depth(da, 5.0)
                elif mode == "wmean":
                    out = _weighted_mean_upper(da, depth_max)
                else:
                    raise ValueError(f"Unknown mode: {mode}")
                all_time.append(_to_time_index(out))
                all_data.append(out.values.astype(np.float32))
        except Exception as e:
            bad.append(f"{fp.name}: {e}")

    if not all_data:
        msg = "; ".join(bad[:3])
        raise RuntimeError(f"no readable files for {var_name}; examples: {msg}")
    if bad:
        print(f"[WARN] {var_name}: skipped {len(bad)} bad files")
    return _finalize_concat(all_time, all_data)


def _load_era5_msl(root: Path) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    files = sorted(root.glob("*.nc"))
    if not files:
        raise FileNotFoundError(f"{root}/*.nc not found")

    all_time: List[pd.DatetimeIndex] = []
    all_data: List[np.ndarray] = []

    for fp in files:
        with _open_dataset_with_fallback(fp) as ds:
            if "msl" not in ds:
                raise RuntimeError(f"msl not found in {fp}")
            da = ds["msl"]
            out = _standardize_2d(da)
            all_time.append(_to_time_index(out))
            all_data.append(out.values.astype(np.float32))
    return _finalize_concat(all_time, all_data)


def _detrend_linear_nan(arr: np.ndarray) -> np.ndarray:
    y = arr.astype(np.float64, copy=False)
    t = np.arange(y.shape[0], dtype=np.float64)[:, None, None]
    mask = np.isfinite(y)
    n = mask.sum(axis=0).astype(np.float64)

    t_mask = np.where(mask, t, 0.0)
    y_mask = np.where(mask, y, 0.0)
    t_mean = np.divide(t_mask.sum(axis=0), n, out=np.zeros_like(n), where=n > 0)
    y_mean = np.divide(y_mask.sum(axis=0), n, out=np.zeros_like(n), where=n > 0)

    cov = np.where(mask, (t - t_mean) * (y - y_mean), 0.0).sum(axis=0)
    var = np.where(mask, (t - t_mean) ** 2, 0.0).sum(axis=0)
    slope = np.divide(cov, var, out=np.zeros_like(cov), where=var > 0)
    intercept = y_mean - slope * t_mean
    trend = slope[None] * t + intercept[None]
    out = y - trend
    out[~mask] = np.nan
    return out.astype(np.float32)


def _remove_monthly_climatology(arr: np.ndarray, months: np.ndarray) -> np.ndarray:
    out = arr.astype(np.float32, copy=True)
    clim = np.full((12, out.shape[1], out.shape[2]), np.nan, dtype=np.float32)
    for m in range(1, 13):
        idx = months == m
        if np.any(idx):
            clim[m - 1] = np.nanmean(out[idx], axis=0).astype(np.float32)
    out = out - clim[months - 1]
    return out


def _minmax_norm(arr: np.ndarray) -> np.ndarray:
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if (not np.isfinite(vmin)) or (not np.isfinite(vmax)) or (vmax - vmin < 1e-8):
        return np.zeros_like(arr, dtype=np.float32)
    return ((arr - vmin) / (vmax - vmin)).astype(np.float32)


def _preprocess_field(data: np.ndarray, months: np.ndarray, fill_zero: bool) -> Tuple[np.ndarray, np.ndarray]:
    x = data.astype(np.float32, copy=True)
    if fill_zero:
        x = np.nan_to_num(x, nan=0.0)
    anom = _remove_monthly_climatology(_detrend_linear_nan(x), months)
    norm = _minmax_norm(anom)
    return anom, norm


def _align_to_common_time(
    series_map: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]]
) -> Tuple[pd.DatetimeIndex, Dict[str, np.ndarray]]:
    names = list(series_map.keys())
    common = series_map[names[0]][0]
    for n in names[1:]:
        common = common.intersection(series_map[n][0])
    if len(common) == 0:
        raise RuntimeError("No overlapping time among predictors")

    aligned: Dict[str, np.ndarray] = {}
    for n in names:
        t, d = series_map[n]
        idx = t.get_indexer(common)
        if np.any(idx < 0):
            raise RuntimeError(f"Failed time alignment: {n}")
        aligned[n] = d[idx]
    return common, aligned


def _pad_to_full_time(data: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros((len(FULL_TIME_GODAS),) + data.shape[1:], dtype=np.float32)
    idx = FULL_TIME_GODAS.get_indexer(time)
    ok = idx >= 0
    out[idx[ok]] = data[ok]
    return out


def _pad_series_to_full_time(series: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros((len(FULL_TIME_GODAS),), dtype=np.float32)
    idx = FULL_TIME_GODAS.get_indexer(time)
    ok = idx >= 0
    out[idx[ok]] = series[ok]
    return out


def _nino34_from_pottmp_anom(pottmp_anom: np.ndarray) -> np.ndarray:
    lat_mask = (TARGET_LATS >= -5.0) & (TARGET_LATS <= 5.0)
    lon_mask = (TARGET_LONS >= 190.0) & (TARGET_LONS <= 240.0)
    reg = pottmp_anom[:, lat_mask][:, :, lon_mask]
    w = np.cos(np.deg2rad(TARGET_LATS[lat_mask])).astype(np.float32)
    w2 = w[None, :, None]
    num = np.nansum(reg * w2, axis=(1, 2))
    den = np.nansum(np.where(np.isfinite(reg), w2, 0.0), axis=(1, 2))
    den = np.where(den == 0, np.nan, den)
    return (num / den).astype(np.float32)


def _save_feature_npz(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mean_map = np.nanmean(data, axis=0).astype(np.float32)
    np.savez_compressed(path, data=data.astype(np.float32), mean_map=mean_map)


def _save_index_npz(path: Path, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, data=data.astype(np.float32))


def main() -> int:
    args = parse_args()
    godas_root = Path(args.godas_root).resolve()
    era5_root = Path(args.era5_root).resolve()
    out_root = Path(args.out_root).resolve()

    if not godas_root.exists():
        print(f"[ERROR] GODAS root not found: {godas_root}")
        return 2
    if not era5_root.exists():
        print(f"[ERROR] ERA5 root not found: {era5_root}")
        return 2

    out_name = "GODAS_reanalysis_198001-202312.npz"
    outputs = {
        "pottmp_5": out_root / "pottmp_5" / out_name,
        "pottmp_wmean": out_root / "pottmp_wmean" / out_name,
        "ucur_5": out_root / "ucur_5" / out_name,
        "vcur_5": out_root / "vcur_5" / out_name,
        "tau_x": out_root / "tau_x" / out_name,
        "tau_y": out_root / "tau_y" / out_name,
        "dbss_obml": out_root / "dbss_obml" / out_name,
        "salt": out_root / "salt" / out_name,
        "msl": out_root / "msl" / out_name,
        "nino34": out_root / "nino34" / out_name,
    }

    if (not args.overwrite) and all(p.exists() for p in outputs.values()):
        print("[SKIP] all outputs already exist")
        return 0

    print(f"[INFO] GODAS root: {godas_root}")
    print(f"[INFO] ERA5 root: {era5_root}")
    print(f"[INFO] out root: {out_root}")
    print(f"[INFO] target grid: {len(TARGET_LATS)}x{len(TARGET_LONS)}")

    series_map: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]] = {}
    series_map["pottmp_5"] = _load_concat_godas_var(godas_root, "pottmp", "surface5", args.depth_max)
    series_map["pottmp_wmean"] = _load_concat_godas_var(godas_root, "pottmp", "wmean", args.depth_max)
    series_map["ucur_5"] = _load_concat_godas_var(godas_root, "ucur", "surface5", args.depth_max)
    series_map["vcur_5"] = _load_concat_godas_var(godas_root, "vcur", "surface5", args.depth_max)
    series_map["tau_x"] = _load_concat_godas_var(godas_root, "uflx", "direct", args.depth_max)
    series_map["tau_y"] = _load_concat_godas_var(godas_root, "vflx", "direct", args.depth_max)
    series_map["dbss_obml"] = _load_concat_godas_var(godas_root, "dbss_obml", "direct", args.depth_max)
    series_map["salt"] = _load_concat_godas_var(godas_root, "salt", "surface5", args.depth_max)
    series_map["msl"] = _load_era5_msl(era5_root)

    common_time, aligned = _align_to_common_time(series_map)
    # Keep GODAS period only; pad to 1980-2023 later.
    common_mask = (common_time >= FULL_TIME_GODAS[0]) & (common_time <= FULL_TIME_GODAS[-1])
    common_time = common_time[common_mask]
    aligned = {k: v[common_mask] for k, v in aligned.items()}
    months = common_time.month.values

    norm_fields: Dict[str, np.ndarray] = {}
    pottmp_anom: Optional[np.ndarray] = None
    for var, arr in aligned.items():
        fill_zero = var not in NO_FILL_VARS
        anom, norm = _preprocess_field(arr, months, fill_zero)
        norm_fields[var] = norm
        if var == "pottmp_5":
            pottmp_anom = anom

    if pottmp_anom is None:
        raise RuntimeError("pottmp_5 anomaly missing")
    nino = _minmax_norm(_nino34_from_pottmp_anom(pottmp_anom))

    for var, data in norm_fields.items():
        _save_feature_npz(outputs[var], _pad_to_full_time(data, common_time))
        print(f"[SAVE] {outputs[var]}")
    _save_index_npz(outputs["nino34"], _pad_series_to_full_time(nino, common_time))
    print(f"[SAVE] {outputs['nino34']}")

    print("[DONE] GODAS + ERA5 msl preprocessing complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
