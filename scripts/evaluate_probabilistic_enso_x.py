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
    parser.add_argument("--members", type=int, default=21)
    parser.add_argument("--field-sigma", type=float, default=0.025)
    parser.add_argument("--memory-sigma", type=float, default=0.08)
    parser.add_argument("--ic-steps", type=int, default=12)
    parser.add_argument("--event-quantile", type=float, default=0.8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_eval_specs():
    return {
        "GODAS_2015_2021": {"valid_models": ["GODAS"], "valid_period": [2015, 2021]},
        "ORAS5_1958_1978": {"valid_models": ["ORAS5"], "valid_period": [1958, 1978]},
    }


def patch_extended_queries(model, trained_horizon=24):
    model_ref = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        if hasattr(model_ref, "query_tokens"):
            query_tokens = model_ref.query_tokens
            if int(query_tokens.shape[0]) > trained_horizon:
                fill = query_tokens[trained_horizon - 1].detach().clone()
                query_tokens[trained_horizon:].copy_(fill.unsqueeze(0).expand(query_tokens.shape[0] - trained_horizon, -1))
        if getattr(model_ref, "use_lead_embedding", False) and hasattr(model_ref, "lead_embedding"):
            lead_weight = model_ref.lead_embedding.weight
            if int(lead_weight.shape[0]) > trained_horizon:
                fill = lead_weight[trained_horizon - 1].detach().clone()
                lead_weight[trained_horizon:].copy_(fill.unsqueeze(0).expand(lead_weight.shape[0] - trained_horizon, -1))


def dataset_metadata(dataset):
    meta_fn = getattr(dataset, "metadata", None)
    if callable(meta_fn):
        try:
            return meta_fn()
        except Exception:
            return {}
    meta = getattr(dataset, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def build_runtime_cfg(base_cfg, args, spec):
    cfg = deepcopy(base_cfg)
    cfg["model"]["init_ckpt"] = args.ckpt
    cfg["model"]["init_ckpt_optional"] = False
    cfg["model"]["init_ckpt_strict"] = False
    cfg["data"]["data_root"] = args.data_root
    cfg["data"]["num_workers"] = 0
    cfg["data"]["valid_models"] = spec["valid_models"]
    cfg["data"]["valid_period"] = spec["valid_period"]
    cfg["data"]["valid_batch_size"] = int(args.batch_size)
    cfg["data"]["source_replay"] = {"enabled": False}
    return cfg


def collect_train_targets(loader):
    ys = []
    for _, _, y_index, _, _, _ in loader:
        ys.append(y_index.numpy()[:, 0, :])
    return np.concatenate(ys, axis=0).astype(np.float32)


def build_thresholds(train_true, event_q):
    center = float(np.median(train_true))
    warm_lead = np.quantile(train_true, float(event_q), axis=0).astype(np.float32)
    cold_lead = np.quantile(train_true, 1.0 - float(event_q), axis=0).astype(np.float32)
    warm_amp = np.maximum(train_true.max(axis=1) - center, 0.0)
    cold_amp = np.maximum(center - train_true.min(axis=1), 0.0)
    return {
        "center": center,
        "warm_lead": warm_lead,
        "cold_lead": cold_lead,
        "warm_window_amp": float(np.quantile(warm_amp, float(event_q))),
        "cold_window_amp": float(np.quantile(cold_amp, float(event_q))),
    }


def perturb_inputs(x_field, m_hist, member_idx, args):
    if int(member_idx) == 0:
        return x_field, m_hist
    xf = x_field.clone()
    mh = m_hist.clone()
    t_cut = min(int(args.ic_steps), int(xf.size(1)))
    if float(args.field_sigma) > 0.0 and t_cut > 0:
        f0 = xf[:, :t_cut]
        f_scale = f0.std(dim=(1, 3, 4), keepdim=True).clamp_min(1.0e-4)
        xf[:, :t_cut] = (f0 + torch.randn_like(f0) * f_scale * float(args.field_sigma)).clamp(0.0, 1.0)
    if float(args.memory_sigma) > 0.0 and t_cut > 0:
        m0 = mh[:, :t_cut]
        m_scale = m0.std(dim=1, keepdim=True).clamp_min(1.0e-4)
        mh[:, :t_cut] = m0 + torch.randn_like(m0) * m_scale * float(args.memory_sigma)
    return xf, mh


def collect_ensemble(model, loader, device, args):
    members = []
    true_chunks = []
    with torch.no_grad():
        for x_field, _, y_index, m_hist, _, init_month in loader:
            x_field = x_field.to(device, non_blocking=True)
            m_hist = m_hist.to(device, non_blocking=True)
            init_month = init_month.to(device, non_blocking=True)
            batch_members = []
            for member_idx in range(int(args.members)):
                xf, mh = perturb_inputs(x_field, m_hist, member_idx, args)
                pred = model(xf, mh, init_month)["index_pred"].detach().cpu().numpy()[:, 0, :]
                batch_members.append(pred)
            members.append(np.stack(batch_members, axis=0))
            true_chunks.append(y_index.numpy()[:, 0, :])
    ens = np.concatenate(members, axis=1).astype(np.float32)
    true = np.concatenate(true_chunks, axis=0).astype(np.float32)
    return ens, true


def prefix_len(corr, threshold=0.5):
    out = 0
    for value in corr:
        if float(value) < float(threshold):
            break
        out += 1
    return int(out)


def corr_summary(pred, true):
    score, corr = weighted_skill_np(pred, true, pred_time=pred.shape[1])
    corr = np.asarray(corr, dtype=np.float32)
    return {
        "score": float(score),
        "corr": [float(x) for x in corr.tolist()],
        "leading": int(np.sum(corr >= 0.5)),
        "frontier": prefix_len(corr, 0.5),
        "min_corr": float(corr.min()),
        "mean_corr": float(corr.mean()),
    }


def brier(prob, event):
    prob = np.asarray(prob, dtype=np.float32)
    event = np.asarray(event, dtype=np.float32)
    return float(np.mean((prob - event) ** 2))


def reliability(prob, event, bins=5):
    prob = np.asarray(prob, dtype=np.float32).reshape(-1)
    event = np.asarray(event, dtype=np.float32).reshape(-1)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    rows = []
    ece = 0.0
    n = max(int(prob.size), 1)
    for i in range(int(bins)):
        lo, hi = float(edges[i]), float(edges[i + 1])
        mask = (prob >= lo) & (prob <= hi if i == int(bins) - 1 else prob < hi)
        count = int(mask.sum())
        if count == 0:
            rows.append({"bin": [lo, hi], "count": 0, "confidence": None, "observed_frequency": None})
            continue
        conf = float(prob[mask].mean())
        obs = float(event[mask].mean())
        ece += (count / n) * abs(conf - obs)
        rows.append({"bin": [lo, hi], "count": count, "confidence": conf, "observed_frequency": obs})
    return {"ece": float(ece), "bins": rows}


def interval_metrics(ens, true, levels=(0.5, 0.8, 0.9)):
    out = {}
    for level in levels:
        lo_q = (1.0 - float(level)) / 2.0
        hi_q = 1.0 - lo_q
        lo = np.quantile(ens, lo_q, axis=0)
        hi = np.quantile(ens, hi_q, axis=0)
        cover = np.logical_and(true >= lo, true <= hi).astype(np.float32)
        out[str(level)] = {
            "coverage": float(cover.mean()),
            "mean_width": float(np.mean(hi - lo)),
            "coverage_by_lead": [float(x) for x in cover.mean(axis=0).tolist()],
        }
    return out


def crps_ensemble(ens, true):
    term1 = np.mean(np.abs(ens - true[None, :, :]), axis=0)
    pair = np.mean(np.abs(ens[:, None, :, :] - ens[None, :, :, :]), axis=(0, 1))
    crps = term1 - 0.5 * pair
    return {
        "mean": float(np.mean(crps)),
        "by_lead": [float(x) for x in np.mean(crps, axis=0).tolist()],
    }


def spread_skill(ens, true):
    mean = ens.mean(axis=0)
    spread = ens.std(axis=0)
    err = mean - true
    rmse_by_lead = np.sqrt(np.mean(err * err, axis=0))
    spread_by_lead = np.mean(spread, axis=0)
    corr = []
    for lead in range(true.shape[1]):
        a = spread[:, lead]
        b = np.abs(err[:, lead])
        if np.std(a) < 1.0e-8 or np.std(b) < 1.0e-8:
            corr.append(0.0)
        else:
            corr.append(float(np.corrcoef(a, b)[0, 1]))
    return {
        "rmse_mean": float(np.mean(rmse_by_lead)),
        "spread_mean": float(np.mean(spread_by_lead)),
        "spread_rmse_ratio": float(np.mean(spread_by_lead) / max(np.mean(rmse_by_lead), 1.0e-8)),
        "rmse_by_lead": [float(x) for x in rmse_by_lead.tolist()],
        "spread_by_lead": [float(x) for x in spread_by_lead.tolist()],
        "spread_abs_error_corr_by_lead": corr,
        "spread_abs_error_corr_mean": float(np.mean(corr)),
    }


def deterministic_event_prob(pred, threshold, mode):
    if mode == "warm":
        return (pred >= threshold[None, :]).astype(np.float32)
    return (pred <= threshold[None, :]).astype(np.float32)


def lead_event_metrics(ens, true, thresholds):
    mean_pred = ens.mean(axis=0)
    warm_event = (true >= thresholds["warm_lead"][None, :]).astype(np.float32)
    cold_event = (true <= thresholds["cold_lead"][None, :]).astype(np.float32)
    warm_prob = (ens >= thresholds["warm_lead"][None, None, :]).mean(axis=0)
    cold_prob = (ens <= thresholds["cold_lead"][None, None, :]).mean(axis=0)
    warm_det = deterministic_event_prob(mean_pred, thresholds["warm_lead"], "warm")
    cold_det = deterministic_event_prob(mean_pred, thresholds["cold_lead"], "cold")
    warm_clim = np.repeat(warm_event.mean(axis=0, keepdims=True), warm_event.shape[0], axis=0)
    cold_clim = np.repeat(cold_event.mean(axis=0, keepdims=True), cold_event.shape[0], axis=0)
    return {
        "warm": {
            "threshold_by_lead": [float(x) for x in thresholds["warm_lead"].tolist()],
            "event_rate_by_lead": [float(x) for x in warm_event.mean(axis=0).tolist()],
            "brier": brier(warm_prob, warm_event),
            "deterministic_brier": brier(warm_det, warm_event),
            "climatology_brier": brier(warm_clim, warm_event),
            "reliability": reliability(warm_prob, warm_event),
        },
        "cold": {
            "threshold_by_lead": [float(x) for x in thresholds["cold_lead"].tolist()],
            "event_rate_by_lead": [float(x) for x in cold_event.mean(axis=0).tolist()],
            "brier": brier(cold_prob, cold_event),
            "deterministic_brier": brier(cold_det, cold_event),
            "climatology_brier": brier(cold_clim, cold_event),
            "reliability": reliability(cold_prob, cold_event),
        },
    }


def window_event_metrics(ens, true, thresholds):
    center = float(thresholds["center"])
    warm_thr = float(thresholds["warm_window_amp"])
    cold_thr = float(thresholds["cold_window_amp"])

    true_warm_amp = np.maximum(true.max(axis=1) - center, 0.0)
    true_cold_amp = np.maximum(center - true.min(axis=1), 0.0)
    true_warm = np.logical_and(true_warm_amp >= warm_thr, true_warm_amp >= true_cold_amp).astype(np.float32)
    true_cold = np.logical_and(true_cold_amp >= cold_thr, true_cold_amp >= true_warm_amp).astype(np.float32)

    pred_warm_amp = np.maximum(ens.max(axis=2) - center, 0.0)
    pred_cold_amp = np.maximum(center - ens.min(axis=2), 0.0)
    pred_warm = np.logical_and(pred_warm_amp >= warm_thr, pred_warm_amp >= pred_cold_amp)
    pred_cold = np.logical_and(pred_cold_amp >= cold_thr, pred_cold_amp >= pred_warm_amp)
    warm_prob = pred_warm.mean(axis=0).astype(np.float32)
    cold_prob = pred_cold.mean(axis=0).astype(np.float32)
    return {
        "center": center,
        "warm_threshold_amp": warm_thr,
        "cold_threshold_amp": cold_thr,
        "warm_support": int(true_warm.sum()),
        "cold_support": int(true_cold.sum()),
        "warm_brier": brier(warm_prob, true_warm),
        "cold_brier": brier(cold_prob, true_cold),
        "warm_reliability": reliability(warm_prob, true_warm),
        "cold_reliability": reliability(cold_prob, true_cold),
        "warm_prob_mean": float(warm_prob.mean()),
        "cold_prob_mean": float(cold_prob.mean()),
        "warm_hit_at_0_5": int(np.logical_and(warm_prob >= 0.5, true_warm > 0.5).sum()),
        "cold_hit_at_0_5": int(np.logical_and(cold_prob >= 0.5, true_cold > 0.5).sum()),
    }


def run_one_tag(base_cfg, args, spec):
    cfg = build_runtime_cfg(base_cfg, args, spec)
    train_loader, valid_loader, train_dataset = build_dataloaders(cfg)
    train_true = collect_train_targets(train_loader)
    thresholds = build_thresholds(train_true, float(args.event_quantile))

    device = torch.device(args.device)
    model = build_model(cfg, memory_dim=train_dataset.memory_dim).to(device)
    load_init_checkpoint(model, cfg.get("model", {}))
    patch_extended_queries(model, trained_horizon=24)
    model.eval()

    torch.manual_seed(int(args.seed))
    np.random.seed(int(args.seed))
    ens, true = collect_ensemble(model, valid_loader, device, args)
    ens_mean = ens.mean(axis=0)
    deterministic = ens[0]

    return {
        "tag": spec["tag"],
        "valid_models": spec["valid_models"],
        "valid_period": spec["valid_period"],
        "samples": int(true.shape[0]),
        "members": int(args.members),
        "perturbation": {
            "field_sigma": float(args.field_sigma),
            "memory_sigma": float(args.memory_sigma),
            "ic_steps": int(args.ic_steps),
            "seed": int(args.seed),
        },
        "train_dataset_meta": dataset_metadata(train_dataset),
        "valid_dataset_meta": dataset_metadata(valid_loader.dataset),
        "deterministic": corr_summary(deterministic, true),
        "ensemble_mean": corr_summary(ens_mean, true),
        "spread_skill": spread_skill(ens, true),
        "intervals": interval_metrics(ens, true),
        "crps": crps_ensemble(ens, true),
        "lead_event": lead_event_metrics(ens, true, thresholds),
        "window_event": window_event_metrics(ens, true, thresholds),
    }


def main():
    args = parse_args()
    base_cfg = load_yaml_config(args.base_config)
    specs = build_eval_specs()
    wanted = []
    for tag in args.tags:
        if tag not in specs:
            raise ValueError("Unknown tag {}. Available: {}".format(tag, sorted(specs)))
        spec = dict(specs[tag])
        spec["tag"] = tag
        wanted.append(spec)

    results = []
    for spec in wanted:
        print("[Prob] running", spec["tag"], flush=True)
        result = run_one_tag(base_cfg, args, spec)
        results.append(result)
        print(
            json.dumps(
                {
                    "tag": result["tag"],
                    "det_frontier": result["deterministic"]["frontier"],
                    "ens_frontier": result["ensemble_mean"]["frontier"],
                    "warm_brier": result["lead_event"]["warm"]["brier"],
                    "cold_brier": result["lead_event"]["cold"]["brier"],
                    "coverage_80": result["intervals"]["0.8"]["coverage"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    payload = {
        "status": "done",
        "base_config": args.base_config,
        "ckpt": args.ckpt,
        "data_root": args.data_root,
        "event_quantile": float(args.event_quantile),
        "method": "initial-condition perturbation ensemble using one fixed ENSO-X checkpoint",
        "results": results,
    }
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("[Prob] OUT", args.output_json, flush=True)


if __name__ == "__main__":
    main()
