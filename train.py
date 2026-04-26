import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from ensox.data import build_dataloaders
from ensox.models import ENSOX, build_loss
from ensox.utils import load_yaml_config, set_seed, weighted_skill_np


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.sum += float(val) * n
        self.count += n

    @property
    def avg(self):
        return self.sum / max(self.count, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/enso_x_24_final.yaml")
    return parser.parse_args()


def _as_bool(v, default=False):
    if v is None:
        return bool(default)
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return bool(v)


def _prefix_len(values, threshold, inclusive=True):
    n = 0
    for x in values:
        if inclusive:
            ok = float(x) >= float(threshold)
        else:
            ok = float(x) > float(threshold)
        if not ok:
            break
        n += 1
    return int(n)


def _segment_stats(values, threshold):
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    if values.size == 0:
        return {
            "min": -1.0,
            "mean": -1.0,
            "shortfall": 0.0,
        }
    return {
        "min": float(values.min()),
        "mean": float(values.mean()),
        "shortfall": float(np.maximum(float(threshold) - values, 0.0).sum()),
    }


def _corr_metrics(corr, lead_threshold=0.5, positive_threshold=0.0, frontier_target=None, positive_target=None):
    corr = np.asarray(corr, dtype=np.float32).reshape(-1)
    n = int(corr.size)
    frontier_requested = n if frontier_target is None or int(frontier_target) <= 0 else int(frontier_target)
    positive_requested = n if positive_target is None or int(positive_target) <= 0 else int(positive_target)
    frontier_eval = min(n, frontier_requested)
    positive_eval = min(n, positive_requested)

    frontier_seg = corr[:frontier_eval]
    positive_seg = corr[:positive_eval]
    frontier_stats = _segment_stats(frontier_seg, lead_threshold)
    positive_stats = _segment_stats(positive_seg, positive_threshold)

    leading = int(np.sum(corr >= float(lead_threshold)))
    frontier_prefix = _prefix_len(corr, lead_threshold, inclusive=True)
    positive_count = int(np.sum(corr > float(positive_threshold)))
    positive_prefix = _prefix_len(corr, positive_threshold, inclusive=False)

    return {
        "lead_threshold": float(lead_threshold),
        "positive_threshold": float(positive_threshold),
        "leading": leading,
        "frontier_prefix": frontier_prefix,
        "leading_prefix": frontier_prefix,
        "positive_count": positive_count,
        "positive_prefix": positive_prefix,
        "frontier_target_requested": int(frontier_requested),
        "frontier_target_eval": int(frontier_eval),
        "frontier_target_min": float(frontier_stats["min"]),
        "frontier_target_mean": float(frontier_stats["mean"]),
        "frontier_target_shortfall": float(frontier_stats["shortfall"]),
        "frontier_hit": int(frontier_eval > 0 and frontier_prefix >= frontier_eval),
        "positive_target_requested": int(positive_requested),
        "positive_target_eval": int(positive_eval),
        "positive_target_min": float(positive_stats["min"]),
        "positive_target_mean": float(positive_stats["mean"]),
        "positive_target_shortfall": float(positive_stats["shortfall"]),
        "positive_hit": int(positive_eval > 0 and positive_prefix >= positive_eval),
    }


def _frontier_monitor_key(corr_stats, score):
    return (
        int(corr_stats.get("frontier_prefix", -1)),
        float(corr_stats.get("frontier_target_min", -1e9)),
        float(-corr_stats.get("frontier_target_shortfall", 1e9)),
        float(corr_stats.get("frontier_target_mean", -1e9)),
        int(corr_stats.get("positive_prefix", -1)),
        float(corr_stats.get("positive_target_min", -1e9)),
        float(-corr_stats.get("positive_target_shortfall", 1e9)),
        float(corr_stats.get("positive_target_mean", -1e9)),
        float(score),
    )


def _frontier_monitor_value(corr_stats, score):
    return (
        1_000_000.0 * float(corr_stats.get("frontier_prefix", 0))
        + 10_000.0 * float(corr_stats.get("positive_prefix", 0))
        + 600.0 * float(corr_stats.get("frontier_target_min", -1.0))
        + 120.0 * float(corr_stats.get("frontier_target_mean", -1.0))
        + 80.0 * float(corr_stats.get("positive_target_min", -1.0))
        + 20.0 * float(corr_stats.get("positive_target_mean", -1.0))
        - 600.0 * float(corr_stats.get("frontier_target_shortfall", 0.0))
        - 80.0 * float(corr_stats.get("positive_target_shortfall", 0.0))
        + float(score)
    )


def _positive_monitor_key(corr_stats, score):
    return (
        int(corr_stats.get("positive_prefix", -1)),
        float(corr_stats.get("positive_target_min", -1e9)),
        float(-corr_stats.get("positive_target_shortfall", 1e9)),
        float(corr_stats.get("positive_target_mean", -1e9)),
        int(corr_stats.get("frontier_prefix", -1)),
        float(corr_stats.get("frontier_target_min", -1e9)),
        float(-corr_stats.get("frontier_target_shortfall", 1e9)),
        float(score),
    )


def _positive_monitor_value(corr_stats, score):
    return (
        1_000_000.0 * float(corr_stats.get("positive_prefix", 0))
        + 4_000.0 * float(corr_stats.get("frontier_prefix", 0))
        + 200.0 * float(corr_stats.get("positive_target_min", -1.0))
        + 60.0 * float(corr_stats.get("positive_target_mean", -1.0))
        + 30.0 * float(corr_stats.get("frontier_target_min", -1.0))
        - 200.0 * float(corr_stats.get("positive_target_shortfall", 0.0))
        - 30.0 * float(corr_stats.get("frontier_target_shortfall", 0.0))
        + float(score)
    )


def build_model(cfg, memory_dim):
    model_cfg = cfg.get("model", {})
    data_cfg = cfg.get("data", {})
    long_head_enabled = _as_bool(model_cfg.get("long_head_enabled", False), default=False)
    model = ENSOX(
        in_channels=len(data_cfg.get("predictor")),
        memory_in_dim=memory_dim,
        dim=int(model_cfg.get("dim", 256)),
        head=int(model_cfg.get("head", 4)),
        depth=int(model_cfg.get("depth", 6)),
        decoder_depth=model_cfg.get("decoder_depth"),
        dim_feedforward=int(model_cfg.get("dim_feedforward", 512)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        obs_time=int(data_cfg.get("obs_time", 12)),
        pred_time=int(data_cfg.get("pred_time", 24)),
        num_index=len(data_cfg.get("predictand")),
        memory_dim=int(model_cfg.get("memory_dim", 128)),
        long_head_enabled=long_head_enabled,
        long_head_start=int(model_cfg.get("long_head_start", 11)),
        long_head_end=int(model_cfg.get("long_head_end", 18)),
        long_head_scale=float(model_cfg.get("long_head_scale", 0.25)),
        barrier_head_enabled=_as_bool(model_cfg.get("barrier_head_enabled", False), default=False),
        barrier_head_start=int(model_cfg.get("barrier_head_start", 8)),
        barrier_head_end=int(model_cfg.get("barrier_head_end", 12)),
        barrier_head_scale=float(model_cfg.get("barrier_head_scale", -2.0)),
        lead_refiner_enabled=_as_bool(model_cfg.get("lead_refiner_enabled", False), default=False),
        lead_refiner_start=int(model_cfg.get("lead_refiner_start", 9)),
        lead_refiner_layers=int(model_cfg.get("lead_refiner_layers", 2)),
        lead_refiner_heads=int(model_cfg.get("lead_refiner_heads", model_cfg.get("head", 4))),
        lead_refiner_scale=float(model_cfg.get("lead_refiner_scale", -2.2)),
        lead_refiner_iters=int(model_cfg.get("lead_refiner_iters", 2)),
        lead_refiner_ffn=int(model_cfg.get("lead_refiner_ffn", 0)),
        ms_refiner_enabled=_as_bool(model_cfg.get("ms_refiner_enabled", False), default=False),
        ms_refiner_start=int(model_cfg.get("ms_refiner_start", 11)),
        ms_refiner_layers=int(model_cfg.get("ms_refiner_layers", 2)),
        ms_refiner_heads=int(model_cfg.get("ms_refiner_heads", model_cfg.get("head", 4))),
        ms_refiner_scale=float(model_cfg.get("ms_refiner_scale", -2.8)),
        ms_refiner_iters=int(model_cfg.get("ms_refiner_iters", 1)),
        ms_refiner_kernel=int(model_cfg.get("ms_refiner_kernel", 3)),
        rollout_refiner_enabled=_as_bool(model_cfg.get("rollout_refiner_enabled", False), default=False),
        rollout_refiner_start=int(model_cfg.get("rollout_refiner_start", 12)),
        rollout_refiner_hidden=int(model_cfg.get("rollout_refiner_hidden", 0)),
        rollout_refiner_scale=float(model_cfg.get("rollout_refiner_scale", -2.0)),
        rollout_refiner_detach_prev=_as_bool(model_cfg.get("rollout_refiner_detach_prev", True), default=True),
        frontier_refiner_enabled=_as_bool(model_cfg.get("frontier_refiner_enabled", False), default=False),
        frontier_refiner_start=int(model_cfg.get("frontier_refiner_start", 5)),
        frontier_refiner_end=int(model_cfg.get("frontier_refiner_end", 16)),
        frontier_refiner_hidden=int(model_cfg.get("frontier_refiner_hidden", 0)),
        frontier_refiner_scale=float(model_cfg.get("frontier_refiner_scale", -1.2)),
        frontier_refiner_detach_prev=_as_bool(model_cfg.get("frontier_refiner_detach_prev", True), default=True),
        tail_booster_enabled=_as_bool(model_cfg.get("tail_booster_enabled", False), default=False),
        tail_booster_start=int(model_cfg.get("tail_booster_start", 21)),
        tail_booster_hidden=int(model_cfg.get("tail_booster_hidden", 0)),
        tail_booster_scale=float(model_cfg.get("tail_booster_scale", -2.0)),
        barrier_booster_enabled=_as_bool(model_cfg.get("barrier_booster_enabled", False), default=False),
        barrier_booster_start=int(model_cfg.get("barrier_booster_start", 8)),
        barrier_booster_end=int(model_cfg.get("barrier_booster_end", 10)),
        barrier_booster_hidden=int(model_cfg.get("barrier_booster_hidden", 0)),
        barrier_booster_scale=float(model_cfg.get("barrier_booster_scale", -0.8)),
        barrier_bridge_enabled=_as_bool(model_cfg.get("barrier_bridge_enabled", False), default=False),
        barrier_bridge_start=int(model_cfg.get("barrier_bridge_start", 8)),
        barrier_bridge_end=int(model_cfg.get("barrier_bridge_end", 11)),
        barrier_bridge_context=int(model_cfg.get("barrier_bridge_context", 2)),
        barrier_bridge_hidden=int(model_cfg.get("barrier_bridge_hidden", 0)),
        barrier_bridge_layers=int(model_cfg.get("barrier_bridge_layers", 1)),
        barrier_bridge_scale=float(model_cfg.get("barrier_bridge_scale", -0.9)),
        prefix_bridge_enabled=_as_bool(model_cfg.get("prefix_bridge_enabled", False), default=False),
        prefix_bridge_start=int(model_cfg.get("prefix_bridge_start", 5)),
        prefix_bridge_end=int(model_cfg.get("prefix_bridge_end", 16)),
        prefix_bridge_context=int(model_cfg.get("prefix_bridge_context", 3)),
        prefix_bridge_hidden=int(model_cfg.get("prefix_bridge_hidden", 0)),
        prefix_bridge_layers=int(model_cfg.get("prefix_bridge_layers", 1)),
        prefix_bridge_scale=float(model_cfg.get("prefix_bridge_scale", -0.7)),
        hole_interp_enabled=_as_bool(model_cfg.get("hole_interp_enabled", False), default=False),
        hole_interp_lead=int(model_cfg.get("hole_interp_lead", 9)),
        hole_interp_start=int(model_cfg.get("hole_interp_start", 0)),
        hole_interp_end=int(model_cfg.get("hole_interp_end", 0)),
        hole_interp_context=int(model_cfg.get("hole_interp_context", 2)),
        hole_interp_scale=float(model_cfg.get("hole_interp_scale", 0.45)),
        hole_patch_enabled=_as_bool(model_cfg.get("hole_patch_enabled", False), default=False),
        hole_patch_lead=int(model_cfg.get("hole_patch_lead", 9)),
        hole_patch_context=int(model_cfg.get("hole_patch_context", 1)),
        hole_patch_hidden=int(model_cfg.get("hole_patch_hidden", 0)),
        hole_patch_scale=float(model_cfg.get("hole_patch_scale", -0.4)),
        memory_fusion_alpha=float(model_cfg.get("memory_fusion_alpha", 1.0)),
        gate_mode=str(model_cfg.get("gate_mode", "learned")),
        use_month_embedding=_as_bool(model_cfg.get("use_month_embedding", True), default=True),
        use_lead_embedding=_as_bool(model_cfg.get("use_lead_embedding", True), default=True),
        legacy_skip_enabled=_as_bool(model_cfg.get("legacy_skip_enabled", False), default=False),
        legacy_skip_alpha=float(model_cfg.get("legacy_skip_alpha", 1.0)),
        memory_mode=str(model_cfg.get("memory_mode", "legacy_ssm")),
        dual_memory_hidden=int(model_cfg.get("dual_memory_hidden", 0)),
        memory_driver_wind_idx=int(model_cfg.get("memory_driver_wind_idx", 1)),
        memory_driver_wwv_idx=int(model_cfg.get("memory_driver_wwv_idx", 0)),
        memory_driver_sst_idx=int(model_cfg.get("memory_driver_sst_idx", 2)),
        memory_cross_attn_enabled=_as_bool(model_cfg.get("memory_cross_attn_enabled", False), default=False),
        memory_cross_attn_heads=int(model_cfg.get("memory_cross_attn_heads", model_cfg.get("head", 4))),
        memory_residual_scale=float(model_cfg.get("memory_residual_scale", 0.8)),
        memory_bridge_scale=float(model_cfg.get("memory_bridge_scale", 0.2)),
        warm_growth_enabled=_as_bool(model_cfg.get("warm_growth_enabled", False), default=False),
        warm_growth_start=int(model_cfg.get("warm_growth_start", 8)),
        warm_growth_end=int(model_cfg.get("warm_growth_end", data_cfg.get("pred_time", 24))),
        warm_growth_context=int(model_cfg.get("warm_growth_context", 2)),
        warm_growth_hidden=int(model_cfg.get("warm_growth_hidden", 0)),
        warm_growth_layers=int(model_cfg.get("warm_growth_layers", 1)),
        warm_growth_scale=float(model_cfg.get("warm_growth_scale", -1.2)),
        prefix_chain_enabled=_as_bool(model_cfg.get("prefix_chain_enabled", False), default=False),
        prefix_chain_start=int(model_cfg.get("prefix_chain_start", 5)),
        prefix_chain_end=int(model_cfg.get("prefix_chain_end", 12)),
        prefix_chain_hidden=int(model_cfg.get("prefix_chain_hidden", 0)),
        prefix_chain_scale=float(model_cfg.get("prefix_chain_scale", -0.8)),
        prefix_chain_detach_prev=_as_bool(model_cfg.get("prefix_chain_detach_prev", False), default=False),
        prefix_band_enabled=_as_bool(model_cfg.get("prefix_band_enabled", False), default=False),
        prefix_band_start=int(model_cfg.get("prefix_band_start", 5)),
        prefix_band_end=int(model_cfg.get("prefix_band_end", 12)),
        prefix_band_hidden=int(model_cfg.get("prefix_band_hidden", 0)),
        prefix_band_layers=int(model_cfg.get("prefix_band_layers", 2)),
        prefix_band_scale=float(model_cfg.get("prefix_band_scale", -0.7)),
        prefix_direct_enabled=_as_bool(model_cfg.get("prefix_direct_enabled", False), default=False),
        prefix_direct_start=int(model_cfg.get("prefix_direct_start", 7)),
        prefix_direct_end=int(model_cfg.get("prefix_direct_end", 12)),
        prefix_direct_hidden=int(model_cfg.get("prefix_direct_hidden", 0)),
        prefix_direct_layers=int(model_cfg.get("prefix_direct_layers", 2)),
        prefix_direct_scale=float(model_cfg.get("prefix_direct_scale", -0.4)),
        prefix_direct_mode=str(model_cfg.get("prefix_direct_mode", "residual")),
        lead_mixer_enabled=_as_bool(model_cfg.get("lead_mixer_enabled", False), default=False),
        lead_mixer_start=int(model_cfg.get("lead_mixer_start", 8)),
        lead_mixer_end=int(model_cfg.get("lead_mixer_end", 24)),
        lead_mixer_hidden=int(model_cfg.get("lead_mixer_hidden", 16)),
        lead_mixer_kernel=int(model_cfg.get("lead_mixer_kernel", 7)),
        lead_mixer_scale=float(model_cfg.get("lead_mixer_scale", -0.7)),
        regional_ridge_enabled=_as_bool(model_cfg.get("regional_ridge_enabled", False), default=False),
        regional_ridge_scale=float(model_cfg.get("regional_ridge_scale", 4.0)),
        legal_analog_enabled=_as_bool(model_cfg.get("legal_analog_enabled", False), default=False),
        legal_analog_scale=float(model_cfg.get("legal_analog_scale", 8.0)),
        legal_analog_topk=int(model_cfg.get("legal_analog_topk", 3)),
        legal_analog_blend=float(model_cfg.get("legal_analog_blend", 0.35)),
        legal_analog_power=float(model_cfg.get("legal_analog_power", 0.5)),
        legal_analog_distance_gate_enabled=_as_bool(
            model_cfg.get("legal_analog_distance_gate_enabled", False), default=False
        ),
        legal_analog_distance_threshold=float(model_cfg.get("legal_analog_distance_threshold", 0.0)),
        legal_analog_distance_temperature=float(model_cfg.get("legal_analog_distance_temperature", 1.0)),
    )
    return model

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    clip_grad=0.0,
    anchor_params=None,
    l2sp_lambda=0.0,
    augment_cfg=None,
):
    model.train()
    meter = AverageMeter()
    for x_field, _, y_index, m_hist, m_future, init_month in loader:
        x_field = x_field.to(device, non_blocking=True)
        y_index = y_index.to(device, non_blocking=True)
        m_hist = m_hist.to(device, non_blocking=True)
        m_future = m_future.to(device, non_blocking=True)
        init_month = init_month.to(device, non_blocking=True)
        x_field, m_hist = apply_train_augmentation(x_field, m_hist, augment_cfg)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=(device.type == "cuda")):
            outputs = model(x_field, m_hist, init_month)
            loss, _ = criterion(outputs, y_index, m_future, init_month)
            if l2sp_lambda > 0.0 and anchor_params:
                model_ref = model.module if hasattr(model, "module") else model
                l2sp_reg = torch.zeros((), device=device)
                reg_count = 0
                for name, param in model_ref.named_parameters():
                    if not param.requires_grad:
                        continue
                    anchor = anchor_params.get(name)
                    if anchor is None:
                        continue
                    l2sp_reg = l2sp_reg + torch.sum((param - anchor) ** 2)
                    reg_count += param.numel()
                if reg_count > 0:
                    loss = loss + l2sp_lambda * (l2sp_reg / float(reg_count))

        scaler.scale(loss).backward()
        if clip_grad > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
        scaler.step(optimizer)
        scaler.update()
        meter.update(loss.item(), x_field.size(0))
    return meter.avg


def _randn_like_shape(ref, shape):
    return torch.randn(shape, device=ref.device, dtype=ref.dtype)


def apply_train_augmentation(x_field, m_hist, augment_cfg):
    """Train-only input perturbation for reanalysis robustness."""
    if not isinstance(augment_cfg, dict) or not _as_bool(augment_cfg.get("enabled", False), default=False):
        return x_field, m_hist

    b, _, c, _, _ = x_field.shape
    field_gain_std = float(augment_cfg.get("field_channel_gain_std", 0.0))
    field_bias_std = float(augment_cfg.get("field_channel_bias_std", 0.0))
    field_noise_std = float(augment_cfg.get("field_noise_std", 0.0))
    field_dropout_prob = float(augment_cfg.get("field_dropout_prob", 0.0))
    memory_gain_std = float(augment_cfg.get("memory_gain_std", 0.0))
    memory_bias_std = float(augment_cfg.get("memory_bias_std", 0.0))
    memory_noise_std = float(augment_cfg.get("memory_noise_std", 0.0))

    if field_gain_std > 0.0:
        gain = 1.0 + _randn_like_shape(x_field, (b, 1, c, 1, 1)) * field_gain_std
        x_field = x_field * gain
    if field_bias_std > 0.0:
        scale = x_field.std(dim=(1, 3, 4), keepdim=True).clamp_min(1.0e-4)
        bias = _randn_like_shape(x_field, (b, 1, c, 1, 1)) * scale * field_bias_std
        x_field = x_field + bias
    if field_noise_std > 0.0:
        scale = x_field.std(dim=(1, 3, 4), keepdim=True).clamp_min(1.0e-4)
        x_field = x_field + torch.randn_like(x_field) * scale * field_noise_std
    if field_dropout_prob > 0.0:
        keep = (torch.rand((b, 1, c, 1, 1), device=x_field.device) >= field_dropout_prob).to(dtype=x_field.dtype)
        channel_mean = x_field.mean(dim=(1, 3, 4), keepdim=True)
        x_field = keep * x_field + (1.0 - keep) * channel_mean

    if _as_bool(augment_cfg.get("clamp_field", True), default=True):
        lo = float(augment_cfg.get("field_clamp_min", 0.0))
        hi = float(augment_cfg.get("field_clamp_max", 1.0))
        x_field = x_field.clamp(lo, hi)

    if memory_gain_std > 0.0:
        gain = 1.0 + _randn_like_shape(m_hist, (m_hist.size(0), 1, m_hist.size(2))) * memory_gain_std
        m_hist = m_hist * gain
    if memory_bias_std > 0.0:
        scale = m_hist.std(dim=1, keepdim=True).clamp_min(1.0e-4)
        bias = _randn_like_shape(m_hist, (m_hist.size(0), 1, m_hist.size(2))) * scale * memory_bias_std
        m_hist = m_hist + bias
    if memory_noise_std > 0.0:
        scale = m_hist.std(dim=1, keepdim=True).clamp_min(1.0e-4)
        m_hist = m_hist + torch.randn_like(m_hist) * scale * memory_noise_std

    return x_field, m_hist


def resolve_ckpt_path(root, ckpt_path):
    if not ckpt_path:
        return None
    if os.path.isabs(ckpt_path):
        return ckpt_path
    return os.path.normpath(os.path.join(root, ckpt_path))


def _dataset_metadata(dataset):
    if dataset is None:
        return {}
    meta_fn = getattr(dataset, "metadata", None)
    if callable(meta_fn):
        try:
            return meta_fn()
        except Exception:
            return {}
    meta = getattr(dataset, "meta", None)
    if isinstance(meta, dict):
        return dict(meta)
    return {}


def _remap_legacy_ctefnet_state(state, model_state):
    mapped = {}
    remap_hits = 0
    duplicate_hits = 0

    def assign(dst_key, src_tensor, src_key):
        nonlocal remap_hits, duplicate_hits
        cur = model_state.get(dst_key)
        if cur is None:
            return False
        if tuple(cur.shape) != tuple(src_tensor.shape):
            return False
        if dst_key in mapped:
            duplicate_hits += 1
            return True
        mapped[dst_key] = src_tensor
        if dst_key != src_key:
            remap_hits += 1
        return True

    for k, v in state.items():
        hit = False

        # Legacy single norm -> ENSO-X encode/decode norms.
        if k.startswith("norm."):
            suffix = k.split(".", 1)[1]
            hit = assign("field_norm." + suffix, v, k) or hit
            hit = assign("decoder_norm." + suffix, v, k) or hit
            if hit:
                continue

        # Legacy decoder head typo key.
        if k.startswith("deocder_head.0."):
            hit = assign(k.replace("deocder_head.0.", "deep_head."), v, k) or hit

        # Legacy skip branch.
        if k.startswith("encoder_head.0."):
            hit = assign(k, v, k) or hit
        if k.startswith("res_norm."):
            hit = assign(k, v, k) or hit
            # Optional fallback when legacy branch is disabled.
            suffix = k.split(".", 1)[1]
            hit = assign("field_norm." + suffix, v, k) or hit
        if k == "res":
            hit = assign("res", v, k) or hit

        # Legacy lead query token.
        if k == "lead_queries":
            hit = assign("query_tokens", v, k) or hit

        # Legacy conv encoder remap.
        if k.startswith("conv.15."):
            hit = assign(k.replace("conv.15.", "field_encoder.head."), v, k) or hit
        if k.startswith("conv."):
            k2 = "field_encoder." + k
            k2 = k2.replace(".bn1.", ".norm1.").replace(".bn2.", ".norm2.")
            hit = assign(k2, v, k) or hit

        # Direct key match last.
        hit = assign(k, v, k) or hit

    print(
        "Applied legacy remap: matched={} remapped_keys={} duplicate_hits={}".format(
            len(mapped), remap_hits, duplicate_hits
        )
    )
    return mapped


def load_init_checkpoint(model, model_cfg):
    init_ckpt = resolve_ckpt_path(ROOT, model_cfg.get("init_ckpt"))
    if not init_ckpt:
        return None
    if not os.path.exists(init_ckpt):
        if bool(model_cfg.get("init_ckpt_optional", False)):
            print("Warning: init_ckpt not found, continue without init: {}".format(init_ckpt))
            return None
        raise FileNotFoundError("init_ckpt not found: {}".format(init_ckpt))
    ckpt = torch.load(init_ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model_ref = model.module if hasattr(model, "module") else model
    skip_patterns = [str(x) for x in model_cfg.get("init_ckpt_skip_patterns", []) if str(x).strip()]
    if skip_patterns:
        state = {
            k: v
            for k, v in state.items()
            if not any(pat in k for pat in skip_patterns)
        }
        print("Init ckpt skip patterns applied:", skip_patterns)

    remap_mode = str(model_cfg.get("init_ckpt_remap", "")).lower().strip()
    if remap_mode in ("legacy_ctefnet", "ctefnet_v1"):
        state = _remap_legacy_ctefnet_state(state, model_ref.state_dict())

    strict = bool(model_cfg.get("init_ckpt_strict", True))
    if strict:
        model_ref.load_state_dict(state, strict=True)
        loaded_state = state
    else:
        current_state = model_ref.state_dict()
        filtered_state = {}
        skipped_shape = []
        partial_shape = []
        dynamic_buffers = []
        for k, v in state.items():
            cur = current_state.get(k)
            if cur is None:
                continue
            if tuple(cur.shape) != tuple(v.shape):
                if k.startswith("legal_analog_") and cur.numel() == 0:
                    setattr(model_ref, k, v.detach().clone().to(device=cur.device, dtype=cur.dtype))
                    dynamic_buffers.append((k, tuple(v.shape)))
                    continue
                if cur.ndim == v.ndim and cur.ndim > 0:
                    overlap = tuple(min(int(a), int(b)) for a, b in zip(cur.shape, v.shape))
                    if all(x > 0 for x in overlap):
                        patched = cur.detach().clone()
                        sl = tuple(slice(0, x) for x in overlap)
                        patched[sl] = v[sl]
                        filtered_state[k] = patched
                        partial_shape.append((k, tuple(v.shape), tuple(cur.shape), overlap))
                        continue
                skipped_shape.append((k, tuple(v.shape), tuple(cur.shape)))
                continue
            filtered_state[k] = v
        model_ref.load_state_dict(filtered_state, strict=False)
        loaded_state = filtered_state
        print(
            "Loaded init_ckpt (non-strict) from {}: matched={} partial_shape={} skipped_shape={}".format(
                init_ckpt, len(filtered_state), len(partial_shape), len(skipped_shape)
            )
        )
        if dynamic_buffers:
            preview = dynamic_buffers[:8]
            print("  dynamic-buffer loads:", preview)
        if partial_shape:
            preview = partial_shape[:8]
            print("  partial-shape samples:", preview)
        if skipped_shape:
            preview = skipped_shape[:8]
            print("  shape-mismatch samples:", preview)
    src_epoch = ckpt.get("epoch", "unknown") if isinstance(ckpt, dict) else "unknown"
    src_score = ckpt.get("score", "unknown") if isinstance(ckpt, dict) else "unknown"
    print("Loaded init_ckpt from {} (epoch={}, score={})".format(init_ckpt, src_epoch, src_score))
    anchor_state = {k: v.detach().clone() for k, v in loaded_state.items()}
    return anchor_state


def load_regional_ridge_init(model, model_cfg):
    init_path = model_cfg.get("regional_ridge_init")
    if not init_path:
        return
    init_path = resolve_ckpt_path(ROOT, init_path)
    if not os.path.exists(init_path):
        raise FileNotFoundError("regional_ridge_init not found: {}".format(init_path))
    model_ref = model.module if hasattr(model, "module") else model
    if not getattr(model_ref, "regional_ridge_enabled", False):
        print("Warning: regional_ridge_init is set but regional_ridge is disabled")
        return
    payload = np.load(init_path)
    weight = torch.tensor(payload["weight"], dtype=model_ref.regional_ridge_head.weight.dtype)
    bias = torch.tensor(payload["bias"], dtype=model_ref.regional_ridge_head.bias.dtype)
    if tuple(weight.shape) != tuple(model_ref.regional_ridge_head.weight.shape):
        raise ValueError(
            "regional ridge weight shape mismatch: {} vs {}".format(
                tuple(weight.shape),
                tuple(model_ref.regional_ridge_head.weight.shape),
            )
        )
    if tuple(bias.shape) != tuple(model_ref.regional_ridge_head.bias.shape):
        raise ValueError(
            "regional ridge bias shape mismatch: {} vs {}".format(
                tuple(bias.shape),
                tuple(model_ref.regional_ridge_head.bias.shape),
            )
        )
    with torch.no_grad():
        model_ref.regional_ridge_head.weight.copy_(weight.to(model_ref.regional_ridge_head.weight.device))
        model_ref.regional_ridge_head.bias.copy_(bias.to(model_ref.regional_ridge_head.bias.device))
    print("Loaded regional_ridge_init from {}".format(init_path))


def load_legal_analog_init(model, model_cfg):
    init_path = model_cfg.get("legal_analog_init")
    if not init_path:
        return
    init_path = resolve_ckpt_path(ROOT, init_path)
    if not os.path.exists(init_path):
        raise FileNotFoundError("legal_analog_init not found: {}".format(init_path))
    model_ref = model.module if hasattr(model, "module") else model
    if not getattr(model_ref, "legal_analog_enabled", False):
        print("Warning: legal_analog_init is set but legal_analog is disabled")
        return

    payload = np.load(init_path, allow_pickle=True)
    required = [
        "mean",
        "std",
        "ridge_weight",
        "ridge_bias",
        "pca_vt",
        "analog_proto",
        "analog_targets",
    ]
    missing = [k for k in required if k not in payload]
    if missing:
        raise KeyError("legal_analog_init missing keys: {}".format(missing))

    device = next(model_ref.parameters()).device

    def set_buffer(name, value):
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if name not in model_ref._buffers:
            model_ref.register_buffer(name, tensor, persistent=True)
        else:
            setattr(model_ref, name, tensor)

    set_buffer("legal_analog_mean", payload["mean"])
    set_buffer("legal_analog_std", payload["std"])
    set_buffer("legal_analog_ridge_weight", payload["ridge_weight"])
    set_buffer("legal_analog_ridge_bias", payload["ridge_bias"])
    set_buffer("legal_analog_pca_vt", payload["pca_vt"])
    set_buffer("legal_analog_proto", payload["analog_proto"])
    set_buffer("legal_analog_targets", payload["analog_targets"])
    if "blend" in payload:
        set_buffer("legal_analog_blend", np.asarray(payload["blend"], dtype=np.float32))
    if "power" in payload:
        set_buffer("legal_analog_power", np.asarray(payload["power"], dtype=np.float32))
    if "topk" in payload:
        model_ref.legal_analog_topk = int(np.asarray(payload["topk"]).reshape(-1)[0])

    print(
        "Loaded legal_analog_init from {} feature_dim={} proto={} pca_dim={} topk={}".format(
            init_path,
            int(model_ref.legal_analog_mean.numel()),
            tuple(model_ref.legal_analog_proto.shape),
            tuple(model_ref.legal_analog_pca_vt.shape),
            int(model_ref.legal_analog_topk),
        )
    )


def _as_pattern_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value else []
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            if v is None:
                continue
            s = str(v)
            if s:
                out.append(s)
        return out
    s = str(value)
    return [s] if s else []


def build_trainable_param_groups(model, opt_cfg, base_lr, weight_decay):
    model_ref = model.module if hasattr(model, "module") else model
    train_patterns = _as_pattern_list(opt_cfg.get("train_patterns"))
    freeze_patterns = _as_pattern_list(opt_cfg.get("freeze_patterns"))

    lr_mult_cfg = opt_cfg.get("lr_multipliers", {})
    lr_mult_items = []
    if isinstance(lr_mult_cfg, dict):
        for k, v in lr_mult_cfg.items():
            if k is None:
                continue
            ks = str(k)
            if not ks:
                continue
            try:
                vv = float(v)
            except Exception:
                continue
            lr_mult_items.append((ks, vv))
    lr_mult_items.sort(key=lambda x: len(x[0]), reverse=True)

    group_map = {}
    trainable_names = []
    frozen_names = []

    for name, param in model_ref.named_parameters():
        allow_train = True
        if train_patterns:
            allow_train = any(pat in name for pat in train_patterns)
        if freeze_patterns and any(pat in name for pat in freeze_patterns):
            allow_train = False

        param.requires_grad = allow_train
        if not allow_train:
            frozen_names.append(name)
            continue

        mult = 1.0
        tag = "base"
        for pat, val in lr_mult_items:
            if pat in name:
                mult = float(val)
                tag = pat
                break

        key = (mult, tag)
        group_map.setdefault(key, []).append(param)
        trainable_names.append(name)

    if not trainable_names:
        raise ValueError("No trainable parameters after applying train_patterns/freeze_patterns")

    groups = []
    for (mult, tag), params in sorted(group_map.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        g = {"params": params, "weight_decay": weight_decay}
        lr_i = base_lr * float(mult)
        if abs(mult - 1.0) > 1e-12:
            g["lr"] = lr_i
        groups.append(g)
        param_count = int(sum(p.numel() for p in params))
        print("[Opt] group tag={} lr={:.6e} params={}".format(tag, lr_i, param_count))

    print(
        "[Opt] trainable_tensors={} frozen_tensors={} train_patterns={} freeze_patterns={}".format(
            len(trainable_names), len(frozen_names), train_patterns, freeze_patterns
        )
    )
    if trainable_names:
        print("[Opt] trainable sample:", trainable_names[:12])

    return groups


@torch.no_grad()
def evaluate(model, loader, criterion, device, pred_time, eval_cfg=None):
    model.eval()
    meter = AverageMeter()
    preds = []
    trues = []
    eval_cfg = eval_cfg or {}
    ens_enabled = _as_bool(eval_cfg.get("enabled", False), default=False)
    ens_members = int(eval_cfg.get("members", 1))
    if not ens_enabled:
        ens_members = 1
    ens_members = max(1, ens_members)
    field_sigma = float(eval_cfg.get("field_sigma", 0.0))
    memory_sigma = float(eval_cfg.get("memory_sigma", 0.0))
    ic_steps = int(eval_cfg.get("ic_steps", 4))

    for x_field, _, y_index, m_hist, m_future, init_month in loader:
        x_field = x_field.to(device, non_blocking=True)
        y_index = y_index.to(device, non_blocking=True)
        m_hist = m_hist.to(device, non_blocking=True)
        m_future = m_future.to(device, non_blocking=True)
        init_month = init_month.to(device, non_blocking=True)

        if ens_members == 1:
            outputs = model(x_field, m_hist, init_month)
        else:
            pred_acc = None
            for member in range(ens_members):
                if member == 0:
                    xf = x_field
                    mh = m_hist
                else:
                    xf = x_field.clone()
                    mh = m_hist.clone()
                    t_cut = min(ic_steps, xf.size(1))
                    if field_sigma > 0.0 and t_cut > 0:
                        f0 = xf[:, :t_cut]
                        f_scale = f0.std(dim=(1, 3, 4), keepdim=True).clamp_min(1e-4)
                        xf[:, :t_cut] = f0 + torch.randn_like(f0) * f_scale * field_sigma
                    if memory_sigma > 0.0 and t_cut > 0:
                        m0 = mh[:, :t_cut]
                        m_scale = m0.std(dim=1, keepdim=True).clamp_min(1e-4)
                        mh[:, :t_cut] = m0 + torch.randn_like(m0) * m_scale * memory_sigma

                out_m = model(xf, mh, init_month)
                if pred_acc is None:
                    pred_acc = out_m["index_pred"]
                else:
                    pred_acc = pred_acc + out_m["index_pred"]
            outputs = {"index_pred": pred_acc / float(ens_members), "memory_feature_pred": None}

        loss, _ = criterion(outputs, y_index, m_future, init_month)
        meter.update(loss.item(), x_field.size(0))

        preds.append(outputs["index_pred"].detach().cpu())
        trues.append(y_index.detach().cpu())

    pred = torch.cat(preds, dim=0).numpy()
    true = torch.cat(trues, dim=0).numpy()
    # First predictand
    score, corr = weighted_skill_np(pred[:, 0, :], true[:, 0, :], pred_time=pred_time)
    leading = int(np.sum(corr >= 0.5))
    return meter.avg, score, corr, leading


def main():
    args = parse_args()
    cfg = load_yaml_config(args.config)
    seed = int(cfg.get("seed", 0))
    set_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    train_loader, valid_loader, train_dataset = build_dataloaders(cfg)
    model = build_model(cfg, memory_dim=train_dataset.memory_dim).to(device)
    train_dataset_meta = _dataset_metadata(train_dataset)
    valid_dataset_meta = _dataset_metadata(valid_loader.dataset)
    if train_dataset_meta:
        print("[Data] train dataset meta:", json.dumps(train_dataset_meta, ensure_ascii=False))
    if valid_dataset_meta:
        print("[Data] valid dataset meta:", json.dumps(valid_dataset_meta, ensure_ascii=False))

    if device.type == "cuda" and torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    model_cfg = cfg.get("model", {})
    anchor_state = load_init_checkpoint(model, model_cfg)
    load_regional_ridge_init(model, model_cfg)
    load_legal_analog_init(model, model_cfg)

    criterion = build_loss(cfg).to(device)
    opt_cfg = cfg.get("optimizer", {})
    lr = float(opt_cfg.get("lr", 2e-4))
    lr_range = opt_cfg.get("lr_range")
    if isinstance(lr_range, (list, tuple)) and len(lr_range) == 2:
        lr_lo = float(lr_range[0])
        lr_hi = float(lr_range[1])
        if not (lr_lo <= lr <= lr_hi):
            print("Warning: lr={} is outside configured lr_range=[{}, {}]".format(lr, lr_lo, lr_hi))

    weight_decay = float(opt_cfg.get("weight_decay", 1e-4))
    param_groups = build_trainable_param_groups(model, opt_cfg, lr, weight_decay)
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler_name = str(opt_cfg.get("scheduler", "cosine")).lower()
    min_lr = float(opt_cfg.get("min_lr", 1e-6))
    if scheduler_name == "cosine_restart":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=int(opt_cfg.get("restart_t0", 10)),
            T_mult=int(opt_cfg.get("restart_tmult", 2)),
            eta_min=min_lr,
        )
    elif scheduler_name == "none":
        scheduler = None
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=int(opt_cfg.get("epoch", 40)), eta_min=min_lr
        )
    scaler = GradScaler(enabled=(device.type == "cuda"))

    l2sp_lambda = float(opt_cfg.get("l2sp_lambda", 0.0))
    anchor_params = None
    if l2sp_lambda > 0.0:
        if anchor_state is None:
            print("Warning: l2sp_lambda>0 but init_ckpt is not set, skipping L2-SP.")
        else:
            model_ref = model.module if hasattr(model, "module") else model
            anchor_params = {}
            for name, param in model_ref.named_parameters():
                anchor = anchor_state.get(name)
                if anchor is None or anchor.shape != param.shape:
                    continue
                anchor_params[name] = anchor.to(device=device, dtype=param.dtype)
            print("L2-SP enabled: lambda={} anchor_params={}".format(l2sp_lambda, len(anchor_params)))

    out_cfg = cfg.get("summary", {})
    eval_cfg = cfg.get("eval_ensemble", {})
    augment_cfg = cfg.get("augmentation", {})
    if isinstance(augment_cfg, dict) and _as_bool(augment_cfg.get("enabled", False), default=False):
        print("[Train] augmentation enabled:", json.dumps(augment_cfg, ensure_ascii=False, sort_keys=True))
    save_dir = os.path.join(ROOT, out_cfg.get("save_dir", "checkpoints"), out_cfg.get("exp_name", "enso_x"))
    os.makedirs(save_dir, exist_ok=True)

    best_score = -1e9
    best_epoch = -1
    best_leading = -1
    best_lead_score = -1e9
    best_lead_epoch = -1
    best_lead_corr = None
    num_epochs = int(opt_cfg.get("epoch", 40))
    pred_time = int(cfg.get("data", {}).get("pred_time", 24))
    early_stop_patience = int(out_cfg.get("early_stop_patience", 0))
    early_stop_min_improve = float(out_cfg.get("early_stop_min_improve", 1e-6))
    monitor_metric = str(out_cfg.get("monitor_metric", "score")).lower()
    if monitor_metric == "lead":
        monitor_metric = "leading"
    if monitor_metric in ("prefix", "milestone"):
        monitor_metric = "frontier"
    if monitor_metric in ("positive", "sign"):
        monitor_metric = "positive_prefix"
    if monitor_metric not in ("score", "leading", "hybrid", "frontier", "positive_prefix"):
        print("Warning: unknown monitor_metric={}, fallback to score".format(monitor_metric))
        monitor_metric = "score"
    lead_bonus = float(out_cfg.get("lead_bonus", 3.0))
    monitor_min_improve = float(out_cfg.get("monitor_min_improve", early_stop_min_improve))
    lead_tie_score_min_improve = float(out_cfg.get("lead_tie_score_min_improve", early_stop_min_improve))
    lead_threshold = float(out_cfg.get("lead_threshold", 0.5))
    positive_threshold = float(out_cfg.get("positive_threshold", 0.0))
    frontier_target = int(out_cfg.get("frontier_target", pred_time))
    positive_target = int(out_cfg.get("positive_target", pred_time))

    best_monitor_epoch = -1
    best_monitor_score = -1e9
    best_monitor_leading = -1
    best_monitor_value = -1e18
    best_monitor_key = None
    best_monitor_stats = None
    best_monitor_corr = None
    best_frontier_epoch = -1
    best_frontier_score = -1e9
    best_frontier_corr = None
    best_frontier_key = None
    best_frontier_stats = None
    no_improve = 0

    def make_ckpt_payload(epoch_, score_, leading_, corr_, corr_stats_):
        model_ref = model.module if hasattr(model, "module") else model
        return {
            "model": model_ref.state_dict(),
            "cfg": cfg,
            "epoch": epoch_,
            "score": score_,
            "leading": leading_,
            "corr": corr_,
            "corr_metrics": corr_stats_,
            "train_data_meta": train_dataset_meta,
            "valid_data_meta": valid_dataset_meta,
        }

    def monitor_key(score_, leading_, corr_stats_):
        if monitor_metric == "score":
            return (float(score_),)
        if monitor_metric == "hybrid":
            return (float(score_) + lead_bonus * float(leading_), float(score_))
        if monitor_metric == "frontier":
            return _frontier_monitor_key(corr_stats_, score_)
        if monitor_metric == "positive_prefix":
            return _positive_monitor_key(corr_stats_, score_)
        return (int(leading_), float(score_))

    def monitor_value(score_, leading_, corr_stats_):
        if monitor_metric == "score":
            return float(score_)
        if monitor_metric == "hybrid":
            return float(score_) + lead_bonus * float(leading_)
        if monitor_metric == "frontier":
            return _frontier_monitor_value(corr_stats_, score_)
        if monitor_metric == "positive_prefix":
            return _positive_monitor_value(corr_stats_, score_)
        return float(leading_)

    def monitor_improved(score_, leading_, corr_stats_):
        if best_monitor_epoch < 0:
            return True
        key_ = monitor_key(score_, leading_, corr_stats_)
        if key_ > best_monitor_key:
            if monitor_metric in ("score", "hybrid"):
                return monitor_value(score_, leading_, corr_stats_) > best_monitor_value + monitor_min_improve
            if monitor_metric == "leading":
                if leading_ > best_monitor_leading:
                    return True
                return score_ > best_monitor_score + lead_tie_score_min_improve
            return True
        return False

    def frontier_improved(score_, corr_stats_):
        key_ = _frontier_monitor_key(corr_stats_, score_)
        return best_frontier_key is None or key_ > best_frontier_key

    for epoch in range(num_epochs):
        start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            clip_grad=float(opt_cfg.get("grad_clip", 0.0)),
            anchor_params=anchor_params,
            l2sp_lambda=l2sp_lambda,
            augment_cfg=augment_cfg,
        )
        valid_loss, score, corr, leading = evaluate(
            model,
            valid_loader,
            criterion,
            device,
            pred_time,
            eval_cfg=eval_cfg,
        )
        corr_stats = _corr_metrics(
            corr,
            lead_threshold=lead_threshold,
            positive_threshold=positive_threshold,
            frontier_target=frontier_target,
            positive_target=positive_target,
        )
        leading = int(corr_stats["leading"])
        if scheduler is not None:
            scheduler.step()

        if score > best_score + early_stop_min_improve:
            best_score = score
            best_epoch = epoch
            torch.save(
                make_ckpt_payload(epoch, score, leading, corr, corr_stats),
                os.path.join(save_dir, "best_score.ckpt"),
            )

        if monitor_improved(score, leading, corr_stats):
            best_monitor_epoch = epoch
            best_monitor_score = score
            best_monitor_leading = leading
            best_monitor_value = monitor_value(score, leading, corr_stats)
            best_monitor_key = monitor_key(score, leading, corr_stats)
            best_monitor_stats = dict(corr_stats)
            best_monitor_corr = corr.copy()
            no_improve = 0
            torch.save(
                make_ckpt_payload(epoch, score, leading, corr, corr_stats),
                os.path.join(save_dir, "best.ckpt"),
            )
        else:
            no_improve += 1

        if (leading > best_leading) or (leading == best_leading and score > best_lead_score + lead_tie_score_min_improve):
            best_leading = leading
            best_lead_score = score
            best_lead_epoch = epoch
            best_lead_corr = corr.copy()
            torch.save(
                make_ckpt_payload(epoch, score, leading, corr, corr_stats),
                os.path.join(save_dir, "best_lead.ckpt"),
            )

        if frontier_improved(score, corr_stats):
            best_frontier_epoch = epoch
            best_frontier_score = score
            best_frontier_corr = corr.copy()
            best_frontier_key = _frontier_monitor_key(corr_stats, score)
            best_frontier_stats = dict(corr_stats)
            torch.save(
                make_ckpt_payload(epoch, score, leading, corr, corr_stats),
                os.path.join(save_dir, "best_frontier.ckpt"),
            )

        if bool(out_cfg.get("save_last", True)):
            torch.save(
                make_ckpt_payload(epoch, score, leading, corr, corr_stats),
                os.path.join(save_dir, "last.ckpt"),
            )

        lr = optimizer.param_groups[0]["lr"]
        dur = time.time() - start
        print(
            "[{}/{}] lr={:.6e} train_loss={:.4f} valid_loss={:.4f} score={:.4f} leading={} frontier={}/{} positive={}/{} best_score={:.4f}@{} best_lead={} (score {:.4f})@{} best_frontier={} (score {:.4f})@{} t={:.1f}s".format(
                epoch + 1,
                num_epochs,
                lr,
                train_loss,
                valid_loss,
                score,
                leading,
                corr_stats["frontier_prefix"],
                corr_stats["frontier_target_eval"],
                corr_stats["positive_prefix"],
                corr_stats["positive_target_eval"],
                best_score,
                best_epoch + 1,
                best_leading,
                best_lead_score,
                best_lead_epoch + 1,
                -1 if best_frontier_stats is None else best_frontier_stats["frontier_prefix"],
                best_frontier_score,
                best_frontier_epoch + 1,
                dur,
            )
        )
        print("  corr:", np.round(corr, 4))
        print(
            "  frontier@{:.2f}: prefix={}/{} count={} min={:.4f} mean={:.4f} shortfall={:.4f}".format(
                lead_threshold,
                corr_stats["frontier_prefix"],
                corr_stats["frontier_target_eval"],
                leading,
                corr_stats["frontier_target_min"],
                corr_stats["frontier_target_mean"],
                corr_stats["frontier_target_shortfall"],
            )
        )
        print(
            "  positive>{:.2f}: prefix={}/{} count={} min={:.4f} mean={:.4f} shortfall={:.4f}".format(
                positive_threshold,
                corr_stats["positive_prefix"],
                corr_stats["positive_target_eval"],
                corr_stats["positive_count"],
                corr_stats["positive_target_min"],
                corr_stats["positive_target_mean"],
                corr_stats["positive_target_shortfall"],
            )
        )
        print(
            "  monitor_metric={} best_monitor_value={:.4f} best_monitor_leading={} best_monitor_score={:.4f}@{} best_monitor_frontier={}".format(
                monitor_metric,
                best_monitor_value,
                best_monitor_leading,
                best_monitor_score,
                best_monitor_epoch + 1,
                -1 if best_monitor_stats is None else best_monitor_stats["frontier_prefix"],
            )
        )

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(
                "Early stop at epoch {}: no {} improvement for {} epochs.".format(
                    epoch + 1, monitor_metric, no_improve
                )
            )
            break

    summary = {
        "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "init_ckpt": resolve_ckpt_path(ROOT, model_cfg.get("init_ckpt")),
        "best_score": float(best_score),
        "best_score_epoch": int(best_epoch + 1),
        "best_leading": int(best_leading),
        "best_lead_epoch": int(best_lead_epoch + 1),
        "best_lead_score": float(best_lead_score),
        "monitor_metric": monitor_metric,
        "best_monitor_epoch": int(best_monitor_epoch + 1),
        "best_monitor_score": float(best_monitor_score),
        "best_monitor_leading": int(best_monitor_leading),
        "best_monitor_value": float(best_monitor_value),
        "lead_bonus": float(lead_bonus),
        "lead_threshold": float(lead_threshold),
        "positive_threshold": float(positive_threshold),
        "frontier_target_requested": int(frontier_target),
        "positive_target_requested": int(positive_target),
        "epochs_planned": int(num_epochs),
        "epochs_completed": int(epoch + 1),
        "early_stop_patience": int(early_stop_patience),
        "train_data_meta": train_dataset_meta,
        "valid_data_meta": valid_dataset_meta,
    }
    if best_lead_corr is not None:
        summary["best_lead_corr"] = np.asarray(best_lead_corr).tolist()
    if best_frontier_stats is not None:
        summary.update(
            {
                "best_frontier_epoch": int(best_frontier_epoch + 1),
                "best_frontier_score": float(best_frontier_score),
                "best_frontier_leading": int(best_frontier_stats["leading"]),
                "best_frontier_prefix": int(best_frontier_stats["frontier_prefix"]),
                "best_positive_prefix": int(best_frontier_stats["positive_prefix"]),
                "best_positive_count": int(best_frontier_stats["positive_count"]),
                "best_frontier_target_eval": int(best_frontier_stats["frontier_target_eval"]),
                "best_frontier_target_min": float(best_frontier_stats["frontier_target_min"]),
                "best_frontier_target_mean": float(best_frontier_stats["frontier_target_mean"]),
                "best_frontier_target_shortfall": float(best_frontier_stats["frontier_target_shortfall"]),
                "best_positive_target_eval": int(best_frontier_stats["positive_target_eval"]),
                "best_positive_target_min": float(best_frontier_stats["positive_target_min"]),
                "best_positive_target_mean": float(best_frontier_stats["positive_target_mean"]),
                "best_positive_target_shortfall": float(best_frontier_stats["positive_target_shortfall"]),
            }
        )
    if best_frontier_corr is not None:
        summary["best_frontier_corr"] = np.asarray(best_frontier_corr).tolist()
    if best_monitor_key is not None:
        summary["best_monitor_key"] = [float(x) for x in best_monitor_key]
    if best_monitor_stats is not None:
        summary["best_monitor_corr"] = np.asarray(best_monitor_corr).tolist() if best_monitor_corr is not None else []
        summary["best_monitor_frontier_prefix"] = int(best_monitor_stats["frontier_prefix"])
        summary["best_monitor_positive_prefix"] = int(best_monitor_stats["positive_prefix"])
        summary["best_monitor_positive_count"] = int(best_monitor_stats["positive_count"])
        summary["best_monitor_frontier_target_eval"] = int(best_monitor_stats["frontier_target_eval"])
        summary["best_monitor_frontier_target_min"] = float(best_monitor_stats["frontier_target_min"])
        summary["best_monitor_frontier_target_mean"] = float(best_monitor_stats["frontier_target_mean"])
        summary["best_monitor_frontier_target_shortfall"] = float(best_monitor_stats["frontier_target_shortfall"])
        summary["best_monitor_positive_target_eval"] = int(best_monitor_stats["positive_target_eval"])
        summary["best_monitor_positive_target_min"] = float(best_monitor_stats["positive_target_min"])
        summary["best_monitor_positive_target_mean"] = float(best_monitor_stats["positive_target_mean"])
        summary["best_monitor_positive_target_shortfall"] = float(best_monitor_stats["positive_target_shortfall"])
    with open(os.path.join(save_dir, "training_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Training finished. Best score={:.4f} at epoch {}".format(best_score, best_epoch + 1))
    print(
        "Best lead={} at epoch {} (score={:.4f})".format(
            best_leading, best_lead_epoch + 1, best_lead_score
        )
    )
    if best_frontier_stats is not None:
        print(
            "Best frontier@{:.2f} prefix={} / {} at epoch {} (score={:.4f}, min={:.4f}, positive_prefix={})".format(
                lead_threshold,
                best_frontier_stats["frontier_prefix"],
                best_frontier_stats["frontier_target_eval"],
                best_frontier_epoch + 1,
                best_frontier_score,
                best_frontier_stats["frontier_target_min"],
                best_frontier_stats["positive_prefix"],
            )
        )
    print(
        "Best monitor({})={} at epoch {} (score={:.4f}, lead={})".format(
            monitor_metric,
            best_monitor_value,
            best_monitor_epoch + 1,
            best_monitor_score,
            best_monitor_leading,
        )
    )
    print("Checkpoint (monitor metric):", os.path.join(save_dir, "best.ckpt"))


if __name__ == "__main__":
    main()
