#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import xarray as xr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check raw CMIP6 NetCDF integrity for ENSO-X preprocessing.")
    parser.add_argument("--raw-root", type=str, required=True, help="Directory containing raw CMIP6 nc files")
    parser.add_argument("--log-dir", type=str, required=True, help="Directory to write integrity reports")
    return parser.parse_args()


def ym_to_idx(ym: str) -> int:
    y = int(ym[:4])
    m = int(ym[4:])
    return y * 12 + (m - 1)


def idx_to_ym(idx: int) -> str:
    y, m0 = divmod(idx, 12)
    return "{:04d}{:02d}".format(y, m0 + 1)


def check_nc_file(fp: Path):
    errs = []
    for eng in ("netcdf4", "h5netcdf", None):
        try:
            kw = {"decode_times": False, "mask_and_scale": False, "cache": False}
            if eng is not None:
                kw["engine"] = eng
            with xr.open_dataset(fp, **kw) as ds:
                if len(ds.data_vars) == 0:
                    raise RuntimeError("no_data_vars")
                first_var = list(ds.data_vars)[0]
                da = ds[first_var]
                if da.ndim > 0:
                    idx = {d: 0 for d in da.dims}
                    _ = da.isel(idx).values
                else:
                    _ = da.values
            return True, ""
        except Exception as e:
            msg = str(e).replace("\n", " ")
            errs.append("{}::{}::{}".format(eng or "auto", type(e).__name__, msg[:180]))
    return False, " || ".join(errs[:3])


def main():
    args = parse_args()
    raw_root = Path(args.raw_root)
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    nc_files = sorted(raw_root.rglob("*.nc"))
    part_files = sorted(raw_root.rglob("*.part"))

    zero_nc = [p for p in nc_files if p.stat().st_size == 0]
    zero_set = set(zero_nc)

    corrupt = []
    for i, fp in enumerate(nc_files, 1):
        if fp in zero_set:
            continue
        ok, err = check_nc_file(fp)
        if not ok:
            corrupt.append((str(fp), str(fp.stat().st_size), err))
        if i % 100 == 0 or i == len(nc_files):
            print("[scan] {}/{} done, corrupt={}".format(i, len(nc_files), len(corrupt)))

    corrupt_set = set(x[0] for x in corrupt)
    orphan_parts = [p for p in part_files if not p.with_suffix("").exists()]

    pat = re.compile(
        r"^(?P<var>[^_]+)_(?P<table>[^_]+)_(?P<model>[^_]+)_ssp370_(?P<variant>r\d+i\d+p\d+f\d+)_[^_]+_(?P<s>\d{6})-(?P<e>\d{6})\.nc$"
    )

    groups = defaultdict(list)
    for fp in nc_files:
        if fp in zero_set:
            continue
        if str(fp) in corrupt_set:
            continue
        m = pat.match(fp.name)
        if not m:
            continue
        key = (m.group("model"), m.group("variant"), m.group("var"), m.group("table"))
        groups[key].append((m.group("s"), m.group("e"), str(fp)))

    gaps = []
    for key, segs in groups.items():
        segs = sorted(segs, key=lambda x: ym_to_idx(x[0]))
        if len(segs) <= 1:
            continue
        prev_end = ym_to_idx(segs[0][1])
        prev_file = segs[0][2]
        for s, e, cur_file in segs[1:]:
            cur_start = ym_to_idx(s)
            if cur_start > prev_end + 1:
                gaps.append(
                    (
                        key[0],
                        key[1],
                        key[2],
                        key[3],
                        idx_to_ym(prev_end + 1),
                        idx_to_ym(cur_start - 1),
                        prev_file,
                        cur_file,
                    )
                )
            prev_end = max(prev_end, ym_to_idx(e))
            prev_file = cur_file

    summary = {
        "nc_total": len(nc_files),
        "part_total": len(part_files),
        "part_orphan": len(orphan_parts),
        "zero_size_nc": len(zero_nc),
        "corrupt_nc": len(corrupt),
        "time_gap_groups": len(gaps),
    }

    (log_dir / "raw_check_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with open(log_dir / "raw_check_corrupt_nc.tsv", "w", encoding="utf-8") as f:
        f.write("path\tsize_bytes\terror\n")
        for row in corrupt:
            f.write("\t".join(row) + "\n")

    with open(log_dir / "raw_check_part_orphan.txt", "w", encoding="utf-8") as f:
        for p in orphan_parts:
            f.write(str(p) + "\n")

    with open(log_dir / "raw_check_zero_nc.txt", "w", encoding="utf-8") as f:
        for p in zero_nc:
            f.write(str(p) + "\n")

    with open(log_dir / "raw_check_time_gaps.tsv", "w", encoding="utf-8") as f:
        f.write("model\tvariant\tvar\ttable\tgap_start\tgap_end\tprev_file\tnext_file\n")
        for g in gaps:
            f.write("\t".join(g) + "\n")

    print("[done] {}".format(log_dir / "raw_check_summary.json"))
    print("[done] {}".format(log_dir / "raw_check_corrupt_nc.tsv"))
    print("[done] {}".format(log_dir / "raw_check_part_orphan.txt"))
    print("[done] {}".format(log_dir / "raw_check_time_gaps.tsv"))


if __name__ == "__main__":
    main()
