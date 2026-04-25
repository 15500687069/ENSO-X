#!/usr/bin/env python3
"""
Build ENSO-X-ready CMIP6 npz files from raw CMIP6 NetCDF files.

Pipeline:
1) Read CMIP6 raw nc files by model/variable.
2) Convert to a common 1x2 grid on 60S-60N, 0-360.
3) Detrend + remove monthly climatology (anomaly).
4) Min-max normalize per model/variable.
5) Save to data/ctefnet_data/CMIP6var/<var>/*.npz and nino34/*.npz.

ENSO-X keeps the same on-disk npz layout as the earlier CTEFNet-style pipeline
so the training code can read CMIP6, GODAS, and ORAS5 consistently.

This script is "robust mode":
- Tries multiple xarray engines.
- Skips broken nc files instead of hard fail.
- Supports common ocean-grid dimension names (lat/lon, nlat/nlon, j/i, rho).
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from tqdm import tqdm


TARGET_LATS = np.arange(-59.5, 60.0, 1.0, dtype=np.float32)   # 120
TARGET_LONS = np.arange(0.0, 360.0, 2.0, dtype=np.float32)    # 180
FULL_TIME = pd.date_range("1850-01-01", "2100-12-01", freq="MS")

DIRECT_VARS = ("psl", "tauu", "tauv", "mlotst", "sos")
NO_FILL_VARS = {"psl", "tauu", "tauv"}  # keep original missing behavior

LAT_DIM_CANDS = ("lat", "latitude", "y")
LON_DIM_CANDS = ("lon", "longitude", "x")
TIME_DIM_CANDS = ("time",)
DEPTH_DIM_CANDS = ("lev", "olevel", "depth", "depthu", "depthv", "z_t", "st_ocean", "rho")
LAT_VAR_CANDS = ("lat", "latitude", "nav_lat", "tlat", "geolat", "ulat", "clat", "y")
LON_VAR_CANDS = ("lon", "longitude", "nav_lon", "tlon", "geolon", "ulon", "clon", "x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw CMIP6 files for ENSO-X.")
    parser.add_argument("--raw-root", type=str, required=True)
    parser.add_argument("--out-root", type=str, required=True)
    parser.add_argument("--models", type=str, default="", help="Comma-separated model list")
    parser.add_argument("--depth-max", type=float, default=300.0)
    parser.add_argument(
        "--wmean-fallback",
        type=str,
        default="thetao_5",
        choices=("none", "thetao_5"),
        help="Fallback when thetao_wmean cannot be computed (e.g. OOM)",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _lower_map(names: Iterable[str]) -> Dict[str, str]:
    return {str(n).lower(): str(n) for n in names}


def _find_dim_name(da: xr.DataArray, candidates: Iterable[str], kind: str) -> str:
    mp = _lower_map(da.dims)
    for c in candidates:
        if c.lower() in mp:
            return mp[c.lower()]
    raise ValueError(f"Cannot find {kind} dim in {tuple(da.dims)}")


def _find_coord_name(da: xr.DataArray, candidates: Iterable[str]) -> Optional[str]:
    mp = _lower_map(da.coords.keys())
    for c in candidates:
        if c.lower() in mp:
            return mp[c.lower()]
    return None


def _attach_geo_coords(da: xr.DataArray, ds: xr.Dataset) -> xr.DataArray:
    # Some files keep lat/lon as data vars, not coords. Promote those.
    all_names = set(ds.variables.keys())
    for c in LAT_VAR_CANDS + LON_VAR_CANDS:
        hit = [n for n in all_names if n.lower() == c.lower()]
        for name in hit:
            v = ds[name]
            if set(v.dims).issubset(set(da.dims)):
                da = da.assign_coords({name: v})
    return da


def _to_time_index(da: xr.DataArray) -> pd.DatetimeIndex:
    idx = da.indexes["time"]
    if hasattr(idx, "to_datetimeindex"):
        try:
            dt = idx.to_datetimeindex()
        except Exception:
            dt = pd.to_datetime([str(x) for x in idx])
    else:
        dt = pd.to_datetime(idx)
    # force month-start timestamps without using to_timestamp("MS")
    return pd.DatetimeIndex([pd.Timestamp(int(d.year), int(d.month), 1) for d in dt])


def _normalize_lon(da: xr.DataArray) -> xr.DataArray:
    lon = np.mod(da["lon"].values.astype(np.float64), 360.0)
    da = da.assign_coords(lon=lon)
    da = da.sortby("lon")
    vals = da["lon"].values
    _, uniq_idx = np.unique(np.round(vals, 6), return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return da.isel(lon=uniq_idx)


def _infer_yx_dims(da: xr.DataArray) -> Tuple[str, str]:
    non_spatial = {d for d in da.dims if d.lower() in TIME_DIM_CANDS or d.lower() in DEPTH_DIM_CANDS}
    cand = [d for d in da.dims if d not in non_spatial]
    if len(cand) < 2:
        raise ValueError(f"Cannot infer 2D spatial dims from {tuple(da.dims)}")
    return cand[-2], cand[-1]


def _coerce_curvilinear_to_rect_coords(
    da: xr.DataArray, ydim: str, xdim: str, lat_name: str, lon_name: str
) -> xr.DataArray:
    latv = da.coords[lat_name]
    lonv = da.coords[lon_name]
    if tuple(latv.dims) != (ydim, xdim) or tuple(lonv.dims) != (ydim, xdim):
        raise ValueError(f"Lat/Lon coords dims do not match spatial dims: {latv.dims}, {lonv.dims}, expected {(ydim, xdim)}")

    lat2d = latv.values.astype(np.float64)
    lon2d = np.mod(lonv.values.astype(np.float64), 360.0)

    lat1d = np.nanmedian(lat2d, axis=1)
    lon1d = np.nanmedian(lon2d, axis=0)
    if np.all(~np.isfinite(lat1d)) or np.all(~np.isfinite(lon1d)):
        raise ValueError("Cannot derive 1D lat/lon from curvilinear coords")

    # fill possible holes
    if np.any(~np.isfinite(lat1d)):
        good = np.where(np.isfinite(lat1d))[0]
        lat1d = np.interp(np.arange(lat1d.size), good, lat1d[good])
    if np.any(~np.isfinite(lon1d)):
        good = np.where(np.isfinite(lon1d))[0]
        lon1d = np.interp(np.arange(lon1d.size), good, lon1d[good])

    # enforce monotonic order
    iy = np.argsort(lat1d)
    ix = np.argsort(lon1d)
    da = da.isel({ydim: iy, xdim: ix})
    # avoid rename conflict when 2D coords already use names "lat"/"lon"
    conflict = [n for n in ("lat", "lon") if (n in da.coords and n not in (ydim, xdim))]
    if conflict:
        da = da.drop_vars(conflict, errors="ignore")
    da = da.assign_coords({ydim: lat1d[iy], xdim: lon1d[ix]})
    da = da.rename({ydim: "lat", xdim: "lon"})
    return da


def _standardize_2d(da: xr.DataArray) -> xr.DataArray:
    # 1) time dim
    tname = _find_dim_name(da, TIME_DIM_CANDS, "time")
    if tname != "time":
        da = da.rename({tname: "time"})

    # 2) already has lat/lon dims
    dim_map = _lower_map(da.dims)
    if "lat" in dim_map and "lon" in dim_map:
        lat_dim = dim_map["lat"]
        lon_dim = dim_map["lon"]
        if lat_dim != "lat" or lon_dim != "lon":
            da = da.rename({lat_dim: "lat", lon_dim: "lon"})
    else:
        # 3) try to use y/x with 1D or 2D lat/lon coordinates
        ydim, xdim = _infer_yx_dims(da)
        lat_name = _find_coord_name(da, LAT_VAR_CANDS)
        lon_name = _find_coord_name(da, LON_VAR_CANDS)
        if lat_name is None or lon_name is None:
            raise ValueError(f"Cannot find lat/lon coordinates in coords={tuple(da.coords.keys())}, dims={tuple(da.dims)}")

        latc = da.coords[lat_name]
        lonc = da.coords[lon_name]
        if latc.ndim == 1 and lonc.ndim == 1:
            # likely mapped on y/x
            lat_dim = latc.dims[0]
            lon_dim = lonc.dims[0]
            conflict = [n for n in ("lat", "lon") if (n in da.coords and n not in (lat_dim, lon_dim))]
            if conflict:
                da = da.drop_vars(conflict, errors="ignore")
            da = da.rename({lat_dim: "lat", lon_dim: "lon"})
            da = da.assign_coords(lat=latc.values, lon=lonc.values)
        elif latc.ndim == 2 and lonc.ndim == 2:
            da = _coerce_curvilinear_to_rect_coords(da, ydim, xdim, lat_name, lon_name)
        else:
            raise ValueError(f"Unsupported lat/lon coord shapes: {latc.shape}, {lonc.shape}")

    # 4) clean/sort/interp
    if np.any(np.diff(da["lat"].values.astype(np.float64)) < 0):
        da = da.sortby("lat")
    da = _normalize_lon(da)
    da = da.sel(lat=slice(-60.0, 60.0))
    da = da.interp(lat=TARGET_LATS, lon=TARGET_LONS, method="linear")
    return da.transpose("time", "lat", "lon")


def _select_depth_dim(da: xr.DataArray) -> str:
    return _find_dim_name(da, DEPTH_DIM_CANDS, "depth")


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


def _open_dataset_with_fallback(fp: Path) -> xr.Dataset:
    last_err = None
    for engine in ("netcdf4", "h5netcdf", None):
        try:
            # mask_and_scale=False significantly reduces memory peaks on huge Omon files.
            kwargs = dict(decode_times=True, use_cftime=True, mask_and_scale=False)
            if engine is not None:
                kwargs["engine"] = engine
            return xr.open_dataset(fp, **kwargs)
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"all engines failed for {fp}: {last_err}")


def _finalize_concat(
    all_time: List[pd.DatetimeIndex],
    all_data: List[np.ndarray],
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    time = pd.DatetimeIndex(np.concatenate([t.values for t in all_time]))
    data = np.concatenate(all_data, axis=0)
    order = np.argsort(time.values)
    time = pd.DatetimeIndex(time.values[order])
    data = data[order]
    _, uniq_idx = np.unique(time.values, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    return pd.DatetimeIndex(time.values[uniq_idx]), data[uniq_idx]


def _load_concat_series(
    model_dir: Path,
    source_var: str,
    mode: str,
    depth_max: float,
) -> Tuple[pd.DatetimeIndex, np.ndarray]:
    files = sorted(model_dir.glob(f"{source_var}_*.nc"))
    if not files:
        raise FileNotFoundError(f"{model_dir}/{source_var}_*.nc not found")

    all_time: List[pd.DatetimeIndex] = []
    all_data: List[np.ndarray] = []
    bad_files: List[str] = []

    for fp in files:
        try:
            with _open_dataset_with_fallback(fp) as ds:
                if source_var not in ds:
                    bad_files.append(f"{fp.name}: var_missing")
                    continue
                da = _attach_geo_coords(ds[source_var], ds)
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
            bad_files.append(f"{fp.name}: {e}")

    if not all_data:
        msg = "; ".join(bad_files[:3])
        raise RuntimeError(f"no readable files for {source_var}; examples: {msg}")

    if bad_files:
        print(f"[WARN] {model_dir.name}/{source_var}: skipped {len(bad_files)} bad files")

    return _finalize_concat(all_time, all_data)


def _load_concat_thetao_dual_series(
    model_dir: Path,
    depth_max: float,
    wmean_fallback: str,
) -> Tuple[Tuple[pd.DatetimeIndex, np.ndarray], Tuple[pd.DatetimeIndex, np.ndarray]]:
    """
    Read thetao files once, then derive both:
    - thetao_5 (surface5)
    - thetao_wmean (upper-ocean weighted mean)
    This keeps strict preprocessing behavior but avoids duplicated file I/O.
    """
    files = sorted(model_dir.glob("thetao_*.nc"))
    if not files:
        raise FileNotFoundError(f"{model_dir}/thetao_*.nc not found")

    all_time_5: List[pd.DatetimeIndex] = []
    all_data_5: List[np.ndarray] = []
    all_time_w: List[pd.DatetimeIndex] = []
    all_data_w: List[np.ndarray] = []
    bad_files_5: List[str] = []
    bad_files_w: List[str] = []
    wmean_err: Optional[str] = None

    for fp in files:
        try:
            with _open_dataset_with_fallback(fp) as ds:
                if "thetao" not in ds:
                    bad_files_5.append(f"{fp.name}: var_missing")
                    bad_files_w.append(f"{fp.name}: var_missing")
                    continue

                da = _attach_geo_coords(ds["thetao"], ds)
                out5 = _surface_at_depth(da, 5.0)
                all_time_5.append(_to_time_index(out5))
                all_data_5.append(out5.values.astype(np.float32))

                if wmean_err is None:
                    try:
                        outw = _weighted_mean_upper(da, depth_max)
                        all_time_w.append(_to_time_index(outw))
                        all_data_w.append(outw.values.astype(np.float32))
                    except Exception as e:
                        wmean_err = f"{fp.name}: {e}"
        except Exception as e:
            bad_files_5.append(f"{fp.name}: {e}")
            bad_files_w.append(f"{fp.name}: {e}")

    if not all_data_5:
        msg = "; ".join(bad_files_5[:3])
        raise RuntimeError(f"no readable files for thetao(surface5); examples: {msg}")

    if bad_files_5:
        print(f"[WARN] {model_dir.name}/thetao_5: skipped {len(bad_files_5)} bad files")

    t5, d5 = _finalize_concat(all_time_5, all_data_5)

    if (wmean_err is None) and all_data_w:
        if bad_files_w:
            print(f"[WARN] {model_dir.name}/thetao_wmean: skipped {len(bad_files_w)} bad files")
        tw, dw = _finalize_concat(all_time_w, all_data_w)
    else:
        if wmean_fallback == "thetao_5":
            msg = wmean_err if wmean_err is not None else "no valid thetao_wmean segments"
            print(f"[WARN] {model_dir.name}/thetao_wmean fallback -> thetao_5 ({msg})")
            tw, dw = t5, d5
        else:
            msg = wmean_err if wmean_err is not None else "no valid thetao_wmean segments"
            raise RuntimeError(f"thetao_wmean failed: {msg}")

    return (t5, d5), (tw, dw)


def _detrend_linear_nan(arr: np.ndarray) -> np.ndarray:
    y = arr.astype(np.float64, copy=False)  # [T,H,W]
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
    out = np.zeros((len(FULL_TIME),) + data.shape[1:], dtype=np.float32)
    idx = FULL_TIME.get_indexer(time)
    ok = idx >= 0
    out[idx[ok]] = data[ok]
    return out


def _pad_series_to_full_time(series: np.ndarray, time: pd.DatetimeIndex) -> np.ndarray:
    out = np.zeros((len(FULL_TIME),), dtype=np.float32)
    idx = FULL_TIME.get_indexer(time)
    ok = idx >= 0
    out[idx[ok]] = series[ok]
    return out


def _nino34_from_thetao_anom(thetao_anom: np.ndarray) -> np.ndarray:
    lat_mask = (TARGET_LATS >= -5.0) & (TARGET_LATS <= 5.0)
    lon_mask = (TARGET_LONS >= 190.0) & (TARGET_LONS <= 240.0)
    reg = thetao_anom[:, lat_mask][:, :, lon_mask]
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


def _discover_models(raw_root: Path, models_arg: str) -> List[str]:
    if models_arg.strip():
        return [x.strip() for x in models_arg.split(",") if x.strip()]
    return sorted([p.name for p in raw_root.iterdir() if p.is_dir()])


def main() -> int:
    args = parse_args()
    raw_root = Path(args.raw_root).resolve()
    out_root = Path(args.out_root).resolve()
    models = _discover_models(raw_root, args.models)

    if not raw_root.exists():
        print(f"[ERROR] raw-root not found: {raw_root}")
        return 2
    if not models:
        print("[ERROR] no models found")
        return 2

    print(f"[INFO] raw_root={raw_root}")
    print(f"[INFO] out_root={out_root}")
    print(f"[INFO] models={models}")
    print(f"[INFO] target_grid={len(TARGET_LATS)}x{len(TARGET_LONS)} full_time={len(FULL_TIME)}")

    file_tag = "185001-210012"
    processed = 0
    skipped: List[str] = []

    for model in tqdm(models, desc="Models"):
        model_dir = raw_root / model
        outputs = {
            "thetao_5": out_root / "thetao_5" / f"{model}_ssp370_{file_tag}.npz",
            "thetao_wmean": out_root / "thetao_wmean" / f"{model}_ssp370_{file_tag}.npz",
            "uo_5": out_root / "uo_5" / f"{model}_ssp370_{file_tag}.npz",
            "vo_5": out_root / "vo_5" / f"{model}_ssp370_{file_tag}.npz",
            "psl": out_root / "psl" / f"{model}_ssp370_{file_tag}.npz",
            "tauu": out_root / "tauu" / f"{model}_ssp370_{file_tag}.npz",
            "tauv": out_root / "tauv" / f"{model}_ssp370_{file_tag}.npz",
            "mlotst": out_root / "mlotst" / f"{model}_ssp370_{file_tag}.npz",
            "sos": out_root / "sos" / f"{model}_ssp370_{file_tag}.npz",
            "nino34": out_root / "nino34" / f"{model}_ssp370_{file_tag}.npz",
        }
        if (not args.overwrite) and all(p.exists() for p in outputs.values()):
            print(f"[SKIP] {model}: all outputs exist")
            continue

        print(f"\n[MODEL] {model}")
        if args.dry_run:
            for k, p in outputs.items():
                print(f"  would_write {k}: {p}")
            processed += 1
            continue

        try:
            series_map: Dict[str, Tuple[pd.DatetimeIndex, np.ndarray]] = {}
            series_map["thetao_5"], series_map["thetao_wmean"] = _load_concat_thetao_dual_series(
                model_dir,
                args.depth_max,
                args.wmean_fallback,
            )
            series_map["uo_5"] = _load_concat_series(model_dir, "uo", "surface5", args.depth_max)
            series_map["vo_5"] = _load_concat_series(model_dir, "vo", "surface5", args.depth_max)
            for var in DIRECT_VARS:
                series_map[var] = _load_concat_series(model_dir, var, "direct", args.depth_max)

            common_time, aligned = _align_to_common_time(series_map)
            months = common_time.month.values
            thetao_anom = None
            norm_fields: Dict[str, np.ndarray] = {}

            for var, arr in aligned.items():
                fill_zero = var not in NO_FILL_VARS
                anom, norm = _preprocess_field(arr, months, fill_zero)
                norm_fields[var] = norm
                if var == "thetao_5":
                    thetao_anom = anom

            if thetao_anom is None:
                raise RuntimeError("thetao_5 anomaly missing")

            nino = _minmax_norm(_nino34_from_thetao_anom(thetao_anom))

            for var, data in norm_fields.items():
                _save_feature_npz(outputs[var], _pad_to_full_time(data, common_time))
            _save_index_npz(outputs["nino34"], _pad_series_to_full_time(nino, common_time))
            processed += 1
        except Exception as e:
            skipped.append(f"{model}: {e}")
            print(f"[WARN] {model} failed -> {e}")
            print(traceback.format_exc(limit=1))

    print("\n=== Summary ===")
    print(f"processed_models: {processed}")
    print(f"skipped_models: {len(skipped)}")
    for s in skipped:
        print(f"  - {s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
