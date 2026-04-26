#!/usr/bin/env python3
import argparse
import json
import os
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import build_dataloaders, build_model, load_yaml_config, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--ridge-alpha", type=float, default=1.0e-2)
    parser.add_argument("--pca-dim", type=int, default=64)
    parser.add_argument("--zero-memory-input", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def metadata(dataset):
    fn = getattr(dataset, "metadata", None)
    if callable(fn):
        return fn()
    meta = getattr(dataset, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def fit_ridge(z, y, alpha):
    y_mean = y.mean(axis=0, keepdims=True)
    yc = y - y_mean
    lhs = z.T @ z
    lhs.flat[:: lhs.shape[0] + 1] += float(alpha)
    rhs = z.T @ yc
    weight = np.linalg.solve(lhs, rhs)
    bias = y_mean.reshape(-1)
    return weight.astype(np.float32), bias.astype(np.float32)


def fit_pca(z, pca_dim):
    max_dim = min(int(pca_dim), z.shape[0], z.shape[1])
    if max_dim <= 0:
        raise ValueError("Invalid PCA dimension for z shape {}".format(z.shape))
    _, _, vt = np.linalg.svd(z, full_matrices=False)
    vt = vt[:max_dim].astype(np.float32)
    proto = z @ vt.T
    return vt, proto.astype(np.float32)


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    set_seed(int(cfg.get("seed", 0)))
    if args.data_root:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    cfg.setdefault("data", {})["num_workers"] = 0
    cfg.setdefault("data", {})["train_batch_size"] = int(args.batch_size)
    cfg.setdefault("model", {})["legal_analog_enabled"] = True

    train_loader, _, train_dataset = build_dataloaders(cfg)
    device = torch.device(args.device)
    model = build_model(cfg, memory_dim=train_dataset.memory_dim).to(device)
    model.eval()
    model_ref = model.module if hasattr(model, "module") else model
    if not getattr(model_ref, "legal_analog_enabled", False):
        raise RuntimeError("legal_analog_enabled must be true to build legal analog init")

    feats = []
    targets = []
    with torch.no_grad():
        for x_field, _, y_index, m_hist, _, init_month in train_loader:
            x_field = x_field.to(device, non_blocking=True)
            m_hist = m_hist.to(device, non_blocking=True)
            init_month = init_month.to(device, non_blocking=True)
            if args.zero_memory_input:
                m_hist = torch.zeros_like(m_hist)
            feat = model_ref._legal_analog_features(x_field, m_hist, init_month)
            feats.append(feat.detach().cpu().numpy().astype(np.float32))
            targets.append(y_index[:, 0, :].numpy().astype(np.float32))

    feat = np.concatenate(feats, axis=0).astype(np.float32)
    target = np.concatenate(targets, axis=0).astype(np.float32)
    mean = feat.mean(axis=0).astype(np.float32)
    std = feat.std(axis=0).astype(np.float32)
    std[std < 1.0e-6] = 1.0
    z = np.clip((feat - mean[None, :]) / std[None, :], -5.0, 5.0).astype(np.float32)

    ridge_weight, ridge_bias = fit_ridge(z, target, args.ridge_alpha)
    pca_vt, analog_proto = fit_pca(z, args.pca_dim)
    meta = {
        "config": os.path.relpath(os.path.abspath(args.config), ROOT),
        "zero_memory_input": bool(args.zero_memory_input),
        "ridge_alpha": float(args.ridge_alpha),
        "pca_dim": int(pca_vt.shape[0]),
        "feature_dim": int(feat.shape[1]),
        "num_samples": int(feat.shape[0]),
        "target_dim": int(target.shape[1]),
        "train_dataset_meta": metadata(train_dataset),
        "leakage_policy": "Built from the configured training loader only; validation data are not used.",
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    np.savez_compressed(
        args.output,
        mean=mean,
        std=std,
        ridge_weight=ridge_weight,
        ridge_bias=ridge_bias,
        pca_vt=pca_vt,
        analog_proto=analog_proto,
        analog_targets=target,
        blend=np.asarray(float(cfg.get("model", {}).get("legal_analog_blend", 0.35)), dtype=np.float32),
        power=np.asarray(float(cfg.get("model", {}).get("legal_analog_power", 0.5)), dtype=np.float32),
        topk=np.asarray(int(cfg.get("model", {}).get("legal_analog_topk", 3)), dtype=np.int64),
        meta_json=np.asarray(json.dumps(meta, ensure_ascii=False)),
    )
    print(json.dumps({"status": "done", "output": args.output, "meta": meta}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
