#!/usr/bin/env python3
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="./outputs/retrained_ablation")
    parser.add_argument("--output-json", default="results/enso_x_retrained_ablation_summary.json")
    return parser.parse_args()


def safe_float(value, default=np.nan):
    try:
        return float(value)
    except Exception:
        return float(default)


def row_from_summary(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    exp_name = os.path.basename(os.path.dirname(path))
    match = re.match(r"(.+)_seed(\d+)$", exp_name)
    variant = match.group(1) if match else exp_name
    seed = int(match.group(2)) if match else None
    corr = data.get("best_frontier_corr") or data.get("best_monitor_corr") or data.get("best_lead_corr") or []
    return {
        "variant": variant,
        "seed": seed,
        "exp_name": exp_name,
        "summary_path": path,
        "epochs_completed": int(data.get("epochs_completed", 0)),
        "best_frontier_prefix": int(data.get("best_frontier_prefix", -1)),
        "best_frontier_min": safe_float(data.get("best_frontier_target_min")),
        "best_frontier_mean": safe_float(data.get("best_frontier_target_mean")),
        "best_frontier_shortfall": safe_float(data.get("best_frontier_target_shortfall")),
        "best_frontier_score": safe_float(data.get("best_frontier_score")),
        "best_leading": int(data.get("best_leading", -1)),
        "corr": corr,
    }


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    out = {}
    for variant, items in sorted(grouped.items()):
        fields = [
            "best_frontier_prefix",
            "best_frontier_min",
            "best_frontier_mean",
            "best_frontier_shortfall",
            "best_frontier_score",
            "best_leading",
        ]
        stats = {"n": len(items), "seeds": [item["seed"] for item in items]}
        for field in fields:
            values = np.asarray([safe_float(item.get(field)) for item in items], dtype=np.float32)
            stats[field] = {
                "mean": float(np.nanmean(values)) if values.size else None,
                "std": float(np.nanstd(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.nanmin(values)) if values.size else None,
                "max": float(np.nanmax(values)) if values.size else None,
            }
        out[variant] = stats
    return out


def main():
    args = parse_args()
    pattern = os.path.join(args.output_root, "*", "training_summary.json")
    rows = [row_from_summary(path) for path in sorted(glob.glob(pattern))]
    payload = {
        "status": "done" if rows else "empty",
        "output_root": args.output_root,
        "num_runs": len(rows),
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload["aggregate"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
