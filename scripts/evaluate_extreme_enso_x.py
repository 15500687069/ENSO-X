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

from train import build_dataloaders, build_model, load_init_checkpoint, load_yaml_config, weighted_skill_np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--tags", nargs="+", default=["GODAS_2015_2021"])
    parser.add_argument("--event-quantile", type=float, default=0.8)
    parser.add_argument("--timing-tolerance", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_eval_specs():
    return {
        "GODAS_2015_2021": {"valid_models": ["GODAS"], "valid_period": [2015, 2021]},
        "ORAS5_1958_1978": {"valid_models": ["ORAS5"], "valid_period": [1958, 1978]},
    }


def dataset_metadata(dataset):
    meta_fn = getattr(dataset, "metadata", None)
    if callable(meta_fn):
        try:
            return meta_fn()
        except Exception:
            return {}
    meta = getattr(dataset, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def runtime_cfg(base_cfg, args, spec):
    cfg = deepcopy(base_cfg)
    cfg["model"]["init_ckpt"] = args.ckpt
    cfg["model"]["init_ckpt_optional"] = False
    cfg["model"]["init_ckpt_strict"] = False
    cfg["data"]["data_root"] = args.data_root
    cfg["data"]["num_workers"] = 0
    cfg["data"]["source_replay"] = {"enabled": False}
    cfg["data"]["valid_models"] = spec["valid_models"]
    cfg["data"]["valid_period"] = spec["valid_period"]
    cfg["data"]["valid_batch_size"] = int(args.batch_size)
    return cfg


def prefix_len(values, threshold=0.5):
    n = 0
    for value in values:
        if float(value) < float(threshold):
            break
        n += 1
    return int(n)


def corr_stats(pred, true):
    score, corr = weighted_skill_np(pred, true, pred_time=pred.shape[1])
    corr = np.asarray(corr, dtype=np.float32)
    return {
        "score": float(score),
        "corr": [float(x) for x in corr.tolist()],
        "leading": int(np.sum(corr >= 0.5)),
        "frontier_prefix": prefix_len(corr, 0.5),
        "min_corr": float(corr.min()),
        "mean_corr": float(corr.mean()),
    }


def binary_metrics(true_mask, pred_mask):
    true_mask = np.asarray(true_mask, dtype=bool)
    pred_mask = np.asarray(pred_mask, dtype=bool)
    tp = int(np.logical_and(true_mask, pred_mask).sum())
    fp = int(np.logical_and(~true_mask, pred_mask).sum())
    fn = int(np.logical_and(true_mask, ~pred_mask).sum())
    tn = int(np.logical_and(~true_mask, ~pred_mask).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "support": int(true_mask.sum()),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def dominant_window_metrics(pred, true, event_q=0.8, timing_tolerance=2):
    pred = np.asarray(pred, dtype=np.float32)
    true = np.asarray(true, dtype=np.float32)
    center = float(np.median(true))

    peak_true = true.max(axis=1)
    peak_pred = pred.max(axis=1)
    peak_true_lead = true.argmax(axis=1) + 1
    peak_pred_lead = pred.argmax(axis=1) + 1
    trough_true = true.min(axis=1)
    trough_pred = pred.min(axis=1)
    trough_true_lead = true.argmin(axis=1) + 1
    trough_pred_lead = pred.argmin(axis=1) + 1

    warm_amp_true = np.maximum(peak_true - center, 0.0)
    warm_amp_pred = np.maximum(peak_pred - center, 0.0)
    cold_amp_true = np.maximum(center - trough_true, 0.0)
    cold_amp_pred = np.maximum(center - trough_pred, 0.0)

    warm_thr = float(np.quantile(warm_amp_true, event_q))
    cold_thr = float(np.quantile(cold_amp_true, event_q))
    warm_true = np.logical_and(warm_amp_true >= warm_thr, warm_amp_true >= cold_amp_true)
    cold_true = np.logical_and(cold_amp_true >= cold_thr, cold_amp_true >= warm_amp_true)
    warm_pred = np.logical_and(warm_amp_pred >= warm_thr, warm_amp_pred >= cold_amp_pred)
    cold_pred = np.logical_and(cold_amp_pred >= cold_thr, cold_amp_pred >= warm_amp_pred)

    def side(mask, pred_mask, pred_amp, true_amp, pred_lead, true_lead, threshold):
        out = binary_metrics(mask, pred_mask)
        if int(mask.sum()) == 0:
            out.update(
                {
                    "threshold": float(threshold),
                    "mean_true_amplitude": None,
                    "mean_pred_amplitude": None,
                    "mean_amplitude_bias": None,
                    "timing_hit_rate": None,
                }
            )
            return out
        lead_error = pred_lead[mask] - true_lead[mask]
        timing_hit = np.logical_and(pred_mask[mask], np.abs(lead_error) <= int(timing_tolerance))
        out.update(
            {
                "threshold": float(threshold),
                "mean_true_amplitude": float(np.mean(true_amp[mask])),
                "mean_pred_amplitude": float(np.mean(pred_amp[mask])),
                "mean_amplitude_bias": float(np.mean(pred_amp[mask] - true_amp[mask])),
                "mean_abs_timing_error": float(np.mean(np.abs(lead_error))),
                "timing_hit_rate": float(np.mean(timing_hit.astype(np.float32))),
            }
        )
        return out

    return {
        "center": center,
        "warm_event": side(warm_true, warm_pred, warm_amp_pred, warm_amp_true, peak_pred_lead, peak_true_lead, warm_thr),
        "cold_event": side(cold_true, cold_pred, cold_amp_pred, cold_amp_true, trough_pred_lead, trough_true_lead, cold_thr),
    }


def run_inference(model, loader, device):
    preds = []
    trues = []
    with torch.no_grad():
        for x_field, _, y_index, m_hist, _, init_month in loader:
            x_field = x_field.to(device, non_blocking=True)
            m_hist = m_hist.to(device, non_blocking=True)
            init_month = init_month.to(device, non_blocking=True)
            out = model(x_field, m_hist, init_month)["index_pred"].detach().cpu().numpy()
            preds.append(out[:, 0, :])
            trues.append(y_index.numpy()[:, 0, :])
    return np.concatenate(preds, axis=0), np.concatenate(trues, axis=0)


def main():
    args = parse_args()
    base_cfg = load_yaml_config(args.base_config)
    specs = build_eval_specs()
    device = torch.device(args.device)
    results = []

    for tag in args.tags:
        if tag not in specs:
            raise ValueError("Unknown tag {}. Available tags: {}".format(tag, sorted(specs)))
        spec = dict(specs[tag])
        cfg = runtime_cfg(base_cfg, args, spec)
        _, valid_loader, train_dataset = build_dataloaders(cfg)
        model = build_model(cfg, memory_dim=train_dataset.memory_dim).to(device)
        load_init_checkpoint(model, cfg.get("model", {}))
        model.eval()
        pred, true = run_inference(model, valid_loader, device)
        results.append(
            {
                "tag": tag,
                "valid_models": spec["valid_models"],
                "valid_period": spec["valid_period"],
                "samples": int(true.shape[0]),
                "valid_dataset_meta": dataset_metadata(valid_loader.dataset),
                "skill": corr_stats(pred, true),
                "window_event": dominant_window_metrics(pred, true, args.event_quantile, args.timing_tolerance),
            }
        )

    payload = {
        "status": "done",
        "base_config": args.base_config,
        "event_quantile": float(args.event_quantile),
        "timing_tolerance": int(args.timing_tolerance),
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("OUT", args.output_json, flush=True)


if __name__ == "__main__":
    main()
