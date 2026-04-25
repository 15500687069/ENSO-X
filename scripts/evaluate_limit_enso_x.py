import argparse
import gc
import json
import os
import sys
from copy import deepcopy

import numpy as np
import torch

THRESHOLDS = [
    ("0_5", 0.5),
    ("0_2", 0.2),
    ("0_0", 0.0),
]


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
    parser.add_argument("--horizons", nargs="+", type=int, default=[24, 32, 40, 48, 60, 72, 84, 96, 108, 120])
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--min-samples", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def cmip6_models(data_root):
    path = os.path.join(data_root, "CMIP6var", "nino34")
    out = []
    for file_name in sorted(os.listdir(path)):
        if file_name.endswith(".npz"):
            out.append(file_name.split("_")[0])
    return out


def patch_extended_queries(model, trained_horizon=24):
    model_ref = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        if hasattr(model_ref, "query_tokens"):
            query_tokens = model_ref.query_tokens
            cur_horizon = int(query_tokens.shape[0])
            if cur_horizon > trained_horizon:
                fill = query_tokens[trained_horizon - 1].detach().clone()
                query_tokens[trained_horizon:cur_horizon].copy_(
                    fill.unsqueeze(0).expand(cur_horizon - trained_horizon, -1)
                )

        if getattr(model_ref, "use_lead_embedding", False) and hasattr(model_ref, "lead_embedding"):
            lead_weight = model_ref.lead_embedding.weight
            cur_horizon = int(lead_weight.shape[0])
            if cur_horizon > trained_horizon:
                fill = lead_weight[trained_horizon - 1].detach().clone()
                lead_weight[trained_horizon:cur_horizon].copy_(
                    fill.unsqueeze(0).expand(cur_horizon - trained_horizon, -1)
                )


def infer_limit_stats(corr):
    corr = [float(x) for x in corr]
    stats = {
        "min_corr": float(min(corr)),
        "mean_corr": float(sum(corr) / len(corr)),
    }
    for label, threshold in THRESHOLDS:
        first_below = next((i + 1 for i, x in enumerate(corr) if x < threshold), None)
        stats["first_below_{}_lead".format(label)] = None if first_below is None else int(first_below)
        stats["last_ge_{}_lead".format(label)] = int(len(corr) if first_below is None else first_below - 1)

    stats["first_negative_lead"] = stats["first_below_0_0_lead"]
    stats["last_nonnegative_lead"] = stats["last_ge_0_0_lead"]
    return stats


def write_progress(out_path, payload):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_or_init_progress(args):
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            progress = json.load(f)
        if not isinstance(progress.get("results"), list):
            progress["results"] = []
    else:
        progress = {"results": []}

    progress["base_config"] = args.base_config
    progress["ckpt"] = args.ckpt
    progress["data_root"] = args.data_root
    progress["trained_horizon"] = 24
    progress["horizons"] = [int(x) for x in args.horizons]
    progress["min_samples"] = int(args.min_samples)
    progress["status"] = "running"
    return progress


def enrich_existing_results(progress):
    changed = False
    for result in progress.get("results", []):
        corr = result.get("corr")
        if corr is None:
            continue
        stats = infer_limit_stats(corr)
        for key, value in stats.items():
            if result.get(key) != value:
                result[key] = value
                changed = True
    return changed


def main():
    args = parse_args()
    device = torch.device(args.device)
    base_cfg = load_yaml_config(args.base_config)
    cmip_models = cmip6_models(args.data_root)

    eval_specs = [
        {
            "tag": "GODAS_2015_2021",
            "valid_models": ["GODAS"],
            "valid_period": [2015, 2021],
        },
        {
            "tag": "CMIP6_2015_2023",
            "valid_models": cmip_models,
            "valid_period": [2015, 2023],
        },
        {
            "tag": "CMIP6_2015_2100",
            "valid_models": cmip_models,
            "valid_period": [2015, 2100],
        },
    ]
    if args.tags:
        want_tags = set(args.tags)
        eval_specs = [spec for spec in eval_specs if spec["tag"] in want_tags]
        if not eval_specs:
            raise ValueError("No eval_specs matched requested tags: {}".format(sorted(want_tags)))

    progress = load_or_init_progress(args)
    if enrich_existing_results(progress):
        write_progress(args.output_json, progress)

    active_tags = {spec["tag"] for spec in eval_specs}
    completed_pairs = set()
    for result in progress.get("results", []):
        tag = result.get("tag")
        horizon = result.get("horizon")
        if tag in active_tags and horizon is not None:
            completed_pairs.add((tag, int(horizon)))
            if result.get("first_below_0_0_lead") is not None or result.get("first_negative_lead") is not None:
                active_tags.discard(tag)

    write_progress(args.output_json, progress)

    for horizon in args.horizons:
        horizon = int(horizon)
        pending_specs = [
            spec
            for spec in eval_specs
            if spec["tag"] in active_tags and (spec["tag"], horizon) not in completed_pairs
        ]
        if not pending_specs:
            print(
                json.dumps({"horizon": horizon, "status": "resume_skip_horizon"}, ensure_ascii=False),
                flush=True,
            )
            continue

        train_cfg = deepcopy(base_cfg)
        train_cfg["model"]["init_ckpt"] = args.ckpt
        train_cfg["model"]["init_ckpt_optional"] = False
        train_cfg["model"]["init_ckpt_strict"] = False
        train_cfg["data"]["data_root"] = args.data_root
        train_cfg["data"]["pred_time"] = horizon
        train_cfg["data"]["num_workers"] = 0
        train_cfg["data"]["valid_models"] = ["GODAS"]
        train_cfg["data"]["valid_period"] = [2015, 2021]
        train_cfg["data"]["valid_batch_size"] = 16
        train_cfg["data"]["source_replay"] = {"enabled": False}
        train_cfg.setdefault("summary", {})["frontier_target"] = horizon
        train_cfg.setdefault("summary", {})["positive_target"] = horizon

        _, _, train_dataset = build_dataloaders(train_cfg)
        model = build_model(train_cfg, memory_dim=train_dataset.memory_dim).to(device)
        load_init_checkpoint(model, train_cfg.get("model", {}))
        patch_extended_queries(model, trained_horizon=24)
        model.eval()

        for spec in pending_specs:
            cfg = deepcopy(train_cfg)
            cfg["data"]["valid_models"] = spec["valid_models"]
            cfg["data"]["valid_period"] = spec["valid_period"]
            _, valid_loader, _ = build_dataloaders(cfg)
            sample_count = len(valid_loader.dataset)

            if sample_count < int(args.min_samples):
                result = {
                    "tag": spec["tag"],
                    "horizon": horizon,
                    "valid_models": spec["valid_models"],
                    "valid_period": spec["valid_period"],
                    "samples": int(sample_count),
                    "status": "skipped_insufficient_samples",
                }
                print(json.dumps(result, ensure_ascii=False), flush=True)
                progress["results"].append(result)
                completed_pairs.add((spec["tag"], horizon))
                write_progress(args.output_json, progress)
                continue

            preds = []
            trues = []
            with torch.no_grad():
                for x_field, _, y_index, m_hist, _, init_month in valid_loader:
                    x_field = x_field.to(device, non_blocking=True)
                    m_hist = m_hist.to(device, non_blocking=True)
                    init_month = init_month.to(device, non_blocking=True)
                    out = model(x_field, m_hist, init_month)["index_pred"].detach().cpu().numpy()
                    preds.append(out[:, 0, :])
                    trues.append(y_index.numpy()[:, 0, :])

            pred = np.concatenate(preds, axis=0)
            true = np.concatenate(trues, axis=0)
            score, corr = weighted_skill_np(pred, true, pred_time=horizon)
            corr = [float(x) for x in corr]
            stats = infer_limit_stats(corr)
            result = {
                "tag": spec["tag"],
                "horizon": horizon,
                "valid_models": spec["valid_models"],
                "valid_period": spec["valid_period"],
                "samples": int(sample_count),
                "status": "ok",
                "score": float(score),
                "corr": corr,
                **stats,
            }
            print(json.dumps(result, ensure_ascii=False), flush=True)
            progress["results"].append(result)
            completed_pairs.add((spec["tag"], horizon))
            write_progress(args.output_json, progress)

            if result["first_negative_lead"] is not None:
                active_tags.discard(spec["tag"])

            del valid_loader, preds, trues, pred, true
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        del model, train_dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if not active_tags:
            break

    progress["status"] = "completed"
    progress["unfinished_tags"] = sorted(active_tags)
    write_progress(args.output_json, progress)
    print("OUT", args.output_json, flush=True)


if __name__ == "__main__":
    main()
