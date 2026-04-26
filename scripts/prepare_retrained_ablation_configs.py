#!/usr/bin/env python3
import argparse
import os
import sys
from copy import deepcopy

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import load_yaml_config


LEAD_REPAIR_FLAGS = [
    "long_head_enabled",
    "barrier_head_enabled",
    "lead_refiner_enabled",
    "ms_refiner_enabled",
    "rollout_refiner_enabled",
    "tail_booster_enabled",
    "barrier_booster_enabled",
    "barrier_bridge_enabled",
    "prefix_bridge_enabled",
    "prefix_chain_enabled",
    "prefix_band_enabled",
    "prefix_direct_enabled",
    "hole_interp_enabled",
    "hole_patch_enabled",
    "frontier_refiner_enabled",
    "lead_mixer_enabled",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", default="configs/enso_x_24_final.yaml")
    parser.add_argument("--output-dir", default="configs/ablation_retrain")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["full", "no_memory", "no_local_lead_repair", "no_legal_analog", "no_reanalysis_repair"],
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--valid-batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1.0e-4)
    parser.add_argument("--min-lr", type=float, default=1.0e-6)
    parser.add_argument("--legal-analog-init", default="${ENSOX_LEGAL_ANALOG_INIT:-./outputs/ablation_init/legal_analog_train_only.npz}")
    parser.add_argument(
        "--legal-analog-init-no-memory",
        default="${ENSOX_LEGAL_ANALOG_INIT_NO_MEMORY:-./outputs/ablation_init/legal_analog_train_only_no_memory.npz}",
    )
    return parser.parse_args()


def scratch_base(cfg, args, variant, seed):
    out = deepcopy(cfg)
    out["seed"] = int(seed)
    out.setdefault("model", {})
    out.setdefault("optimizer", {})
    out.setdefault("summary", {})
    out.setdefault("data", {})
    out.setdefault("loss", {})

    out["model"]["init_ckpt"] = ""
    out["model"]["init_ckpt_optional"] = True
    out["model"]["init_ckpt_strict"] = False
    out["model"]["legal_analog_init"] = args.legal_analog_init
    # The released checkpoint uses a saturated repair scale after fine-tuning.
    # For from-scratch ablations, start repair layers unsaturated so the field
    # encoder and memory branch can still learn instead of being locked by the
    # fixed analog/ridge predictor at epoch 0.
    if "legal_analog_scale" in out["model"]:
        out["model"]["legal_analog_scale"] = -2.0
    if "regional_ridge_scale" in out["model"]:
        out["model"]["regional_ridge_scale"] = 0.0

    opt = out["optimizer"]
    opt["epoch"] = int(args.epochs)
    opt["lr"] = float(args.lr)
    opt["lr_range"] = [float(args.min_lr), float(args.lr) * 3.0]
    opt["min_lr"] = float(args.min_lr)
    opt["l2sp_lambda"] = 0.0
    opt.pop("train_patterns", None)
    opt.pop("freeze_patterns", None)
    opt.pop("lr_multipliers", None)

    data = out["data"]
    data["data_root"] = "${ENSOX_DATA_ROOT:-./data/ctefnet_data}"
    data["train_batch_size"] = int(args.train_batch_size)
    data["valid_batch_size"] = int(args.valid_batch_size)
    data["num_workers"] = 0

    summary = out["summary"]
    summary["save_dir"] = "${ENSOX_ABLATION_OUTPUT_ROOT:-./outputs/retrained_ablation}"
    summary["exp_name"] = "{}_seed{}".format(variant, seed)
    summary["save_last"] = True
    summary["monitor_metric"] = "frontier"
    summary["frontier_target"] = int(data.get("pred_time", 24))
    summary["positive_target"] = int(data.get("pred_time", 24))
    summary["early_stop_patience"] = 15

    return out


def apply_variant(cfg, variant, args):
    model = cfg["model"]
    loss = cfg["loss"]
    if variant == "full":
        return
    if variant == "no_memory":
        cfg["ablation"] = {"zero_memory_input": True}
        model["legal_analog_init"] = args.legal_analog_init_no_memory
        model["memory_cross_attn_enabled"] = False
        model["memory_residual_scale"] = 0.0
        model["memory_bridge_scale"] = 0.0
        model["memory_fusion_alpha"] = 0.0
        loss["lambda_memory"] = 0.0
        loss["lambda_whm_coupling"] = 0.0
        return
    if variant == "no_local_lead_repair":
        for name in LEAD_REPAIR_FLAGS:
            if name in model:
                model[name] = False
        return
    if variant == "no_legal_analog":
        model["legal_analog_enabled"] = False
        model.pop("legal_analog_init", None)
        return
    if variant == "no_reanalysis_repair":
        model["legal_analog_enabled"] = False
        model["regional_ridge_enabled"] = False
        model.pop("legal_analog_init", None)
        model.pop("regional_ridge_init", None)
        return
    raise ValueError("Unknown variant: {}".format(variant))


def dump_yaml(cfg, path):
    with open(path, "w", encoding="utf-8") as f:
        try:
            yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=False, default_flow_style=False)
        except TypeError:
            yaml.safe_dump(cfg, f, allow_unicode=False, default_flow_style=False)


def main():
    args = parse_args()
    base_cfg = load_yaml_config(args.base_config)
    out_dir = os.path.abspath(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    written = []
    for variant in args.variants:
        for seed in args.seeds:
            cfg = scratch_base(base_cfg, args, variant, seed)
            apply_variant(cfg, variant, args)
            path = os.path.join(out_dir, "{}_seed{}.yaml".format(variant, seed))
            dump_yaml(cfg, path)
            written.append(os.path.relpath(path, ROOT))
    manifest = os.path.join(out_dir, "manifest.txt")
    with open(manifest, "w", encoding="utf-8") as f:
        for path in written:
            f.write(path.replace("\\", "/") + "\n")
    print("Wrote {} configs to {}".format(len(written), out_dir))
    print("Manifest:", os.path.relpath(manifest, ROOT))


if __name__ == "__main__":
    main()
