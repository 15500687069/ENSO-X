#!/usr/bin/env python3
import argparse
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.evaluate_ablation_enso_x import collect_variant, event_metrics, skill_metrics
from train import build_dataloaders, build_model, load_init_checkpoint, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--calib-model", default="GODAS")
    parser.add_argument("--calib-period", nargs=2, type=int, default=[1980, 2014])
    parser.add_argument("--test-model", default="GODAS")
    parser.add_argument("--test-period", nargs=2, type=int, default=[2015, 2021])
    parser.add_argument("--gains", nargs="+", type=float, default=[1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    parser.add_argument("--min-frontier", type=int, default=24)
    parser.add_argument("--min-corr", type=float, default=0.5)
    parser.add_argument("--event-quantile", type=float, default=0.8)
    parser.add_argument("--timing-tolerance", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def runtime_cfg(base_cfg, args, model_name, period):
    cfg = deepcopy(base_cfg)
    cfg["model"]["init_ckpt"] = args.ckpt
    cfg["model"]["init_ckpt_optional"] = False
    cfg["model"]["init_ckpt_strict"] = False
    cfg["data"]["data_root"] = args.data_root
    cfg["data"]["num_workers"] = 0
    cfg["data"]["source_replay"] = {"enabled": False}
    cfg["data"]["valid_models"] = [model_name]
    cfg["data"]["valid_period"] = [int(period[0]), int(period[1])]
    cfg["data"]["valid_batch_size"] = int(args.batch_size)
    return cfg


def metadata(dataset):
    fn = getattr(dataset, "metadata", None)
    if callable(fn):
        return fn()
    meta = getattr(dataset, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def scaled(pred, gain):
    center = pred.mean(axis=1, keepdims=True)
    return center + float(gain) * (pred - center)


def evaluate_gain(pred, true, gain, args):
    pred_g = scaled(pred, gain)
    skill = skill_metrics(pred_g, true)
    event = event_metrics(pred_g, true, args.event_quantile, args.timing_tolerance)
    warm = event["warm_event"]
    cold = event["cold_event"]
    return {
        "gain": float(gain),
        "skill": skill,
        "event": event,
        "selection_score": float(warm["f1"] + cold["f1"] + 0.25 * (warm["recall"] + cold["recall"])),
    }


def main():
    args = parse_args()
    base_cfg = load_yaml_config(args.base_config)
    device = torch.device(args.device)

    calib_cfg = runtime_cfg(base_cfg, args, args.calib_model, args.calib_period)
    _, calib_loader, train_dataset = build_dataloaders(calib_cfg)
    model = build_model(calib_cfg, memory_dim=train_dataset.memory_dim).to(device)
    load_init_checkpoint(model, calib_cfg.get("model", {}))
    model.eval()

    calib_pred, calib_true = collect_variant(model, calib_loader, device, "full", args)
    calib_rows = [evaluate_gain(calib_pred, calib_true, gain, args) for gain in args.gains]
    eligible = [
        row
        for row in calib_rows
        if row["skill"]["frontier_prefix"] >= int(args.min_frontier) and row["skill"]["min_corr"] >= float(args.min_corr)
    ]
    candidates = eligible if eligible else calib_rows
    best = max(candidates, key=lambda row: (row["selection_score"], row["skill"]["mean_corr"]))

    test_cfg = runtime_cfg(base_cfg, args, args.test_model, args.test_period)
    _, test_loader, _ = build_dataloaders(test_cfg)
    test_pred, test_true = collect_variant(model, test_loader, device, "full", args)
    test_base = evaluate_gain(test_pred, test_true, 1.0, args)
    test_best = evaluate_gain(test_pred, test_true, best["gain"], args)

    payload = {
        "status": "done",
        "policy": "Gain is selected on calibration data only, then applied once to the held-out test period.",
        "calibration": {
            "model": args.calib_model,
            "period": [int(args.calib_period[0]), int(args.calib_period[1])],
            "dataset_meta": metadata(calib_loader.dataset),
            "rows": calib_rows,
            "selected_gain": float(best["gain"]),
        },
        "test": {
            "model": args.test_model,
            "period": [int(args.test_period[0]), int(args.test_period[1])],
            "dataset_meta": metadata(test_loader.dataset),
            "baseline": test_base,
            "calibrated": test_best,
        },
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps({"selected_gain": best["gain"], "test": test_best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
