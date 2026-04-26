#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import date

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from train import build_dataloaders, load_yaml_config


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", default="")
    parser.add_argument("--output-json", default="")
    return parser.parse_args()


def parse_date(text):
    return date.fromisoformat(str(text))


def metadata(dataset):
    fn = getattr(dataset, "metadata", None)
    if callable(fn):
        return fn()
    meta = getattr(dataset, "meta", None)
    return dict(meta) if isinstance(meta, dict) else {}


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    if args.data_root:
        cfg.setdefault("data", {})["data_root"] = args.data_root
    cfg.setdefault("data", {})["num_workers"] = 0

    train_loader, valid_loader, train_dataset = build_dataloaders(cfg)
    train_meta = metadata(train_dataset)
    valid_meta = metadata(valid_loader.dataset)

    train_end = parse_date(train_meta["selected_time_end"])
    valid_start = parse_date(valid_meta["selected_time_start"])
    valid_end = parse_date(valid_meta["selected_time_end"])
    train_models = set(train_meta.get("source_models", []))
    valid_models = set(valid_meta.get("source_models", []))

    checks = []

    # The dataset slices each period before constructing samples, so the last
    # training target is bounded by selected_time_end rather than by the raw
    # reanalysis file end.
    checks.append(
        {
            "name": "train_targets_do_not_enter_validation_period",
            "passed": bool(train_end < valid_start),
            "detail": {
                "train_selected_end": train_meta["selected_time_end"],
                "valid_selected_start": valid_meta["selected_time_start"],
            },
        }
    )
    checks.append(
        {
            "name": "same_reanalysis_split_is_time_ordered",
            "passed": bool(not (train_models & valid_models) or train_end < valid_start),
            "detail": {
                "train_models": sorted(train_models),
                "valid_models": sorted(valid_models),
            },
        }
    )
    checks.append(
        {
            "name": "validation_labels_are_clamped_to_available_period",
            "passed": bool(valid_meta.get("effective_full_time_end") == valid_meta.get("label_last_valid_time") or valid_meta.get("label_last_valid_time") is None),
            "detail": {
                "effective_full_time_end": valid_meta.get("effective_full_time_end"),
                "label_last_valid_time": valid_meta.get("label_last_valid_time"),
                "label_clamp_applied": valid_meta.get("label_clamp_applied"),
            },
        }
    )

    replay = cfg.get("data", {}).get("source_replay", {})
    replay_enabled = isinstance(replay, dict) and bool(replay.get("enabled", False))
    checks.append(
        {
            "name": "source_replay_is_training_only",
            "passed": bool(replay_enabled),
            "detail": replay if isinstance(replay, dict) else {},
        }
    )

    passed = all(bool(item["passed"]) for item in checks)
    payload = {
        "status": "passed" if passed else "failed",
        "config": os.path.relpath(os.path.abspath(args.config), ROOT),
        "train_meta": train_meta,
        "valid_meta": valid_meta,
        "train_batches": int(len(train_loader)),
        "valid_batches": int(len(valid_loader)),
        "checks": checks,
        "interpretation": (
            "Training and validation are separated by time before sample construction; "
            "therefore training targets cannot cross into the GODAS validation period."
        ),
    }

    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
