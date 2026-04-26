import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ENSOXLoss(nn.Module):
    def __init__(
        self,
        pred_time: int,
        obs_time: int,
        lead_weights=None,
        spring_months=None,
        spring_boost: float = 1.4,
        lambda_memory: float = 0.2,
        lambda_corr: float = 0.05,
        lambda_tail_corr: float = 0.0,
        tail_start: int = 11,
        tail_end: int = 0,
        tail_target: float = 0.5,
        tail_power: float = 1.0,
        tail_hinge_power: float = 1.0,
        corr_focus_start: int = 0,
        corr_focus_end: int = 0,
        corr_focus_boost: float = 1.0,
        lambda_lead_proxy: float = 0.0,
        lead_threshold: float = 0.5,
        lead_proxy_temp: float = 0.08,
        lead_proxy_start: int = 1,
        lead_proxy_end: int = 0,
        lambda_prefix_floor: float = 0.0,
        prefix_floor_start: int = 1,
        prefix_floor_end: int = 0,
        prefix_floor_target: float = 0.5,
        prefix_floor_temp: float = 0.03,
        lambda_prefix_chain: float = 0.0,
        prefix_chain_start: int = 1,
        prefix_chain_end: int = 0,
        prefix_chain_target: float = 0.5,
        prefix_chain_temp: float = 0.03,
        lambda_whm_coupling: float = 0.0,
        whm_target_abs_corr: float = 0.20,
        whm_focus_start: int = 8,
        whm_focus_end: int = 12,
        whm_wwv_idx: int = 0,
        whm_wind_idx: int = 1,
        whm_sst_idx: int = 2,
        whm_alpha: float = 0.7,
        lambda_warm_event: float = 0.0,
        lambda_warm_peak: float = 0.0,
        lambda_warm_timing: float = 0.0,
        lambda_warm_non_event_reg: float = 0.0,
        warm_event_quantile: float = 0.8,
        warm_event_min_amp: float = 0.10,
        warm_event_margin: float = 0.02,
    ):
        super().__init__()
        self.pred_time = pred_time
        self.obs_time = obs_time
        self.spring_months = spring_months if spring_months is not None else [2, 3, 4]
        self.spring_boost = spring_boost
        self.lambda_memory = lambda_memory
        self.lambda_corr = lambda_corr
        self.lambda_tail_corr = lambda_tail_corr
        self.tail_start = max(1, int(tail_start))
        self.tail_end = int(tail_end)
        self.tail_target = float(tail_target)
        self.tail_power = float(tail_power)
        self.tail_hinge_power = float(tail_hinge_power)
        self.corr_focus_start = int(corr_focus_start)
        self.corr_focus_end = int(corr_focus_end)
        self.corr_focus_boost = float(corr_focus_boost)
        self.lambda_lead_proxy = float(lambda_lead_proxy)
        self.lead_threshold = float(lead_threshold)
        self.lead_proxy_temp = float(lead_proxy_temp)
        self.lead_proxy_start = int(lead_proxy_start)
        self.lead_proxy_end = int(lead_proxy_end)
        self.lambda_prefix_floor = float(lambda_prefix_floor)
        self.prefix_floor_start = int(prefix_floor_start)
        self.prefix_floor_end = int(prefix_floor_end)
        self.prefix_floor_target = float(prefix_floor_target)
        self.prefix_floor_temp = float(prefix_floor_temp)
        self.lambda_prefix_chain = float(lambda_prefix_chain)
        self.prefix_chain_start = int(prefix_chain_start)
        self.prefix_chain_end = int(prefix_chain_end)
        self.prefix_chain_target = float(prefix_chain_target)
        self.prefix_chain_temp = float(prefix_chain_temp)
        self.lambda_whm_coupling = float(lambda_whm_coupling)
        self.whm_target_abs_corr = float(whm_target_abs_corr)
        self.whm_focus_start = int(whm_focus_start)
        self.whm_focus_end = int(whm_focus_end)
        self.whm_wwv_idx = int(whm_wwv_idx)
        self.whm_wind_idx = int(whm_wind_idx)
        self.whm_sst_idx = int(whm_sst_idx)
        self.whm_alpha = float(whm_alpha)
        self.lambda_warm_event = float(lambda_warm_event)
        self.lambda_warm_peak = float(lambda_warm_peak)
        self.lambda_warm_timing = float(lambda_warm_timing)
        self.lambda_warm_non_event_reg = float(lambda_warm_non_event_reg)
        self.warm_event_quantile = float(warm_event_quantile)
        self.warm_event_min_amp = float(warm_event_min_amp)
        self.warm_event_margin = float(warm_event_margin)
        if lead_weights is None:
            lead_weights = np.array([1.2] * 6 + [1.8] * 6 + [2.2] * 6 + [2.8] * 24, dtype=np.float32)[:pred_time]
        self.register_buffer("lead_weights", torch.tensor(lead_weights, dtype=torch.float32), persistent=False)

    def _build_weights(self, init_month, device):
        bsz = init_month.shape[0]
        lead_ids = torch.arange(self.pred_time, device=device, dtype=torch.long)
        target_month = (init_month[:, None] + self.obs_time + lead_ids[None, :]) % 12
        seasonal = torch.ones((bsz, self.pred_time), device=device, dtype=torch.float32)
        for m in self.spring_months:
            seasonal = seasonal + (target_month == int(m)).float() * (self.spring_boost - 1.0)
        weights = seasonal * self.lead_weights[None, :]
        return weights

    def forward(self, outputs, target_index, memory_future, init_month):
        pred = outputs["index_pred"]
        # pred/target: [B, K, P]
        weights = self._build_weights(init_month, pred.device).unsqueeze(1)
        weighted_mse = ((pred - target_index) ** 2) * weights
        index_loss = torch.sqrt(weighted_mse.mean(dim=0) + 1e-8).mean()

        memory_loss = torch.tensor(0.0, device=pred.device)
        if memory_future is not None and outputs.get("memory_feature_pred", None) is not None:
            mem_pred = outputs["memory_feature_pred"]
            memory_loss = F.mse_loss(mem_pred, memory_future)

        pred_ = pred - pred.mean(dim=0, keepdim=True)
        true_ = target_index - target_index.mean(dim=0, keepdim=True)
        corr = F.cosine_similarity(pred_, true_, dim=0)  # [K, P]
        corr_per_lead = corr.mean(dim=0) if corr.ndim == 2 else corr

        lead_corr_weights = self.lead_weights.clone()
        if self.corr_focus_boost > 1.0 and self.corr_focus_start > 0:
            focus_start = max(self.corr_focus_start - 1, 0)
            focus_end = self.pred_time if self.corr_focus_end <= 0 else min(self.corr_focus_end, self.pred_time)
            if focus_start < focus_end:
                lead_corr_weights[focus_start:focus_end] = lead_corr_weights[focus_start:focus_end] * self.corr_focus_boost
        lead_corr_weights = lead_corr_weights / (lead_corr_weights.sum() + 1e-8)
        corr_loss = ((1.0 - corr_per_lead) * lead_corr_weights).sum()

        tail_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_tail_corr > 0.0:
            start_idx = min(max(self.tail_start - 1, 0), self.pred_time - 1)
            end_idx = self.pred_time if self.tail_end <= 0 else min(max(self.tail_end, self.tail_start), self.pred_time)
            tail_corr = corr_per_lead[start_idx:end_idx]
            if tail_corr.numel() > 0:
                if self.tail_power != 1.0:
                    tail_steps = torch.arange(1, tail_corr.numel() + 1, device=pred.device, dtype=torch.float32)
                    tail_weights = tail_steps.pow(self.tail_power)
                    tail_weights = tail_weights / (tail_weights.sum() + 1e-8)
                else:
                    tail_weights = torch.full_like(tail_corr, 1.0 / float(tail_corr.numel()))
                gap = F.relu(self.tail_target - tail_corr)
                if self.tail_hinge_power != 1.0:
                    gap = gap.pow(self.tail_hinge_power)
                tail_loss = (gap * tail_weights).sum()

        lead_proxy_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_lead_proxy > 0.0:
            lp_start = min(max(self.lead_proxy_start - 1, 0), self.pred_time - 1)
            lp_end = self.pred_time if self.lead_proxy_end <= 0 else min(max(self.lead_proxy_end, self.lead_proxy_start), self.pred_time)
            lp_corr = corr_per_lead[lp_start:lp_end]
            if lp_corr.numel() > 0:
                lp_weights = lead_corr_weights[lp_start:lp_end]
                lp_weights = lp_weights / (lp_weights.sum() + 1e-8)
                temp = max(self.lead_proxy_temp, 1e-4)
                margin = (self.lead_threshold - lp_corr) / temp
                lead_proxy_loss = (F.softplus(margin) * lp_weights).sum()

        prefix_floor_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_prefix_floor > 0.0:
            pf_start = min(max(self.prefix_floor_start - 1, 0), self.pred_time - 1)
            pf_end = self.pred_time if self.prefix_floor_end <= 0 else min(max(self.prefix_floor_end, self.prefix_floor_start), self.pred_time)
            pf_corr = corr_per_lead[pf_start:pf_end]
            if pf_corr.numel() > 0:
                temp = max(self.prefix_floor_temp, 1e-4)
                soft_min = -temp * torch.logsumexp(-pf_corr / temp, dim=0)
                prefix_floor_loss = F.relu(self.prefix_floor_target - soft_min)

        prefix_chain_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_prefix_chain > 0.0:
            pc_start = min(max(self.prefix_chain_start - 1, 0), self.pred_time - 1)
            pc_end = self.pred_time if self.prefix_chain_end <= 0 else min(max(self.prefix_chain_end, self.prefix_chain_start), self.pred_time)
            pc_corr = corr_per_lead[pc_start:pc_end]
            if pc_corr.numel() > 0:
                temp = max(self.prefix_chain_temp, 1e-4)
                prefix_softmins = []
                for i in range(int(pc_corr.numel())):
                    prefix_softmins.append(-temp * torch.logsumexp(-pc_corr[: i + 1] / temp, dim=0))
                prefix_softmins = torch.stack(prefix_softmins, dim=0)
                chain_weights = torch.arange(
                    1, prefix_softmins.numel() + 1, device=pred.device, dtype=pred.dtype
                )
                chain_weights = chain_weights / (chain_weights.sum() + 1e-8)
                prefix_chain_loss = (F.relu(self.prefix_chain_target - prefix_softmins) * chain_weights).sum()

        whm_coupling_loss = torch.tensor(0.0, device=pred.device)
        if self.lambda_whm_coupling > 0.0 and memory_future is not None:
            mem = memory_future
            mem_dim = int(mem.size(-1))
            if mem_dim > 0:
                i_wwv = min(max(self.whm_wwv_idx, 0), mem_dim - 1)
                i_wind = min(max(self.whm_wind_idx, 0), mem_dim - 1)
                i_sst = min(max(self.whm_sst_idx, 0), mem_dim - 1)
                wwv = mem[:, :, i_wwv]
                wind = mem[:, :, i_wind]
                sst = mem[:, :, i_sst]

                # Physics-guided Wyrtki-Hasselmann proxy: recharge x wind forcing + SST persistence.
                combo = torch.tanh(self.whm_alpha * (wwv * wind) + (1.0 - self.whm_alpha) * sst)
                pred_main = pred[:, 0, :] if pred.ndim == 3 else pred

                c_start = min(max(self.whm_focus_start - 1, 0), self.pred_time - 1)
                c_end = self.pred_time if self.whm_focus_end <= 0 else min(max(self.whm_focus_end, self.whm_focus_start), self.pred_time)
                p_seg = pred_main[:, c_start:c_end]
                c_seg = combo[:, c_start:c_end]
                if p_seg.numel() > 0 and c_seg.numel() > 0:
                    p_seg = p_seg - p_seg.mean(dim=0, keepdim=True)
                    c_seg = c_seg - c_seg.mean(dim=0, keepdim=True)
                    corr_abs = torch.abs(F.cosine_similarity(p_seg, c_seg, dim=0))
                    c_weights = lead_corr_weights[c_start:c_end]
                    c_weights = c_weights / (c_weights.sum() + 1e-8)
                    whm_coupling_loss = (F.relu(self.whm_target_abs_corr - corr_abs) * c_weights).sum()

        warm_event_loss = torch.tensor(0.0, device=pred.device)
        warm_peak_loss = torch.tensor(0.0, device=pred.device)
        warm_timing_loss = torch.tensor(0.0, device=pred.device)
        warm_non_event_reg = torch.tensor(0.0, device=pred.device)
        if (
            (self.lambda_warm_event > 0.0 or self.lambda_warm_peak > 0.0 or self.lambda_warm_timing > 0.0 or self.lambda_warm_non_event_reg > 0.0)
            and outputs.get("warm_event_logit", None) is not None
        ):
            target_main = target_index[:, 0, :] if target_index.ndim == 3 else target_index
            center = target_main.median(dim=-1, keepdim=True).values
            warm_amp_true = torch.clamp(target_main.max(dim=-1).values - center.squeeze(-1), min=0.0)
            cold_amp_true = torch.clamp(center.squeeze(-1) - target_main.min(dim=-1).values, min=0.0)
            if warm_amp_true.numel() > 1:
                warm_q = torch.quantile(warm_amp_true.detach(), self.warm_event_quantile)
            else:
                warm_q = warm_amp_true.max().detach()
            warm_thr = torch.maximum(warm_q, pred.new_tensor(self.warm_event_min_amp))
            warm_mask = torch.logical_and(
                warm_amp_true >= warm_thr,
                warm_amp_true >= cold_amp_true + float(self.warm_event_margin),
            )
            event_target = warm_mask.float().view(-1, 1)
            warm_event_loss = F.binary_cross_entropy_with_logits(outputs["warm_event_logit"], event_target)

            if bool(warm_mask.any().item()):
                peak_target = warm_amp_true.unsqueeze(-1).expand(-1, pred.shape[1])
                warm_peak_loss = F.smooth_l1_loss(outputs["warm_peak_pred"][warm_mask], peak_target[warm_mask])
                timing_target = target_main.argmax(dim=-1)
                warm_timing_loss = F.cross_entropy(outputs["warm_timing_logit"][warm_mask], timing_target[warm_mask])

            if outputs.get("base_index_pred", None) is not None:
                delta = outputs["index_pred"] - outputs["base_index_pred"]
                nonwarm_mask = ~warm_mask
                if bool(nonwarm_mask.any().item()):
                    warm_non_event_reg = delta[nonwarm_mask].abs().mean()

        total = (
            index_loss
            + self.lambda_memory * memory_loss
            + self.lambda_corr * corr_loss
            + self.lambda_tail_corr * tail_loss
            + self.lambda_lead_proxy * lead_proxy_loss
            + self.lambda_prefix_floor * prefix_floor_loss
            + self.lambda_prefix_chain * prefix_chain_loss
            + self.lambda_whm_coupling * whm_coupling_loss
            + self.lambda_warm_event * warm_event_loss
            + self.lambda_warm_peak * warm_peak_loss
            + self.lambda_warm_timing * warm_timing_loss
            + self.lambda_warm_non_event_reg * warm_non_event_reg
        )
        return total, {
            "index_loss": float(index_loss.detach().cpu()),
            "memory_loss": float(memory_loss.detach().cpu()),
            "corr_loss": float(corr_loss.detach().cpu()),
            "tail_loss": float(tail_loss.detach().cpu()),
            "lead_proxy_loss": float(lead_proxy_loss.detach().cpu()),
            "prefix_floor_loss": float(prefix_floor_loss.detach().cpu()),
            "prefix_chain_loss": float(prefix_chain_loss.detach().cpu()),
            "whm_coupling_loss": float(whm_coupling_loss.detach().cpu()),
            "warm_event_loss": float(warm_event_loss.detach().cpu()),
            "warm_peak_loss": float(warm_peak_loss.detach().cpu()),
            "warm_timing_loss": float(warm_timing_loss.detach().cpu()),
            "warm_non_event_reg": float(warm_non_event_reg.detach().cpu()),
        }


def build_loss(cfg):
    data_cfg = cfg.get("data", {})
    loss_cfg = cfg.get("loss", {})
    return ENSOXLoss(
        pred_time=int(data_cfg.get("pred_time", 24)),
        obs_time=int(data_cfg.get("obs_time", 12)),
        lead_weights=loss_cfg.get("lead_weights"),
        spring_months=loss_cfg.get("spring_months", [2, 3, 4]),
        spring_boost=float(loss_cfg.get("spring_boost", 1.4)),
        lambda_memory=float(loss_cfg.get("lambda_memory", 0.2)),
        lambda_corr=float(loss_cfg.get("lambda_corr", 0.05)),
        lambda_tail_corr=float(loss_cfg.get("lambda_tail_corr", 0.0)),
        tail_start=int(loss_cfg.get("tail_start", 11)),
        tail_end=int(loss_cfg.get("tail_end", 0)),
        tail_target=float(loss_cfg.get("tail_target", 0.5)),
        tail_power=float(loss_cfg.get("tail_power", 1.0)),
        tail_hinge_power=float(loss_cfg.get("tail_hinge_power", 1.0)),
        corr_focus_start=int(loss_cfg.get("corr_focus_start", 0)),
        corr_focus_end=int(loss_cfg.get("corr_focus_end", 0)),
        corr_focus_boost=float(loss_cfg.get("corr_focus_boost", 1.0)),
        lambda_lead_proxy=float(loss_cfg.get("lambda_lead_proxy", 0.0)),
        lead_threshold=float(loss_cfg.get("lead_threshold", 0.5)),
        lead_proxy_temp=float(loss_cfg.get("lead_proxy_temp", 0.08)),
        lead_proxy_start=int(loss_cfg.get("lead_proxy_start", 1)),
        lead_proxy_end=int(loss_cfg.get("lead_proxy_end", 0)),
        lambda_prefix_floor=float(loss_cfg.get("lambda_prefix_floor", 0.0)),
        prefix_floor_start=int(loss_cfg.get("prefix_floor_start", 1)),
        prefix_floor_end=int(loss_cfg.get("prefix_floor_end", 0)),
        prefix_floor_target=float(loss_cfg.get("prefix_floor_target", 0.5)),
        prefix_floor_temp=float(loss_cfg.get("prefix_floor_temp", 0.03)),
        lambda_prefix_chain=float(loss_cfg.get("lambda_prefix_chain", 0.0)),
        prefix_chain_start=int(loss_cfg.get("prefix_chain_start", 1)),
        prefix_chain_end=int(loss_cfg.get("prefix_chain_end", 0)),
        prefix_chain_target=float(loss_cfg.get("prefix_chain_target", 0.5)),
        prefix_chain_temp=float(loss_cfg.get("prefix_chain_temp", 0.03)),
        lambda_whm_coupling=float(loss_cfg.get("lambda_whm_coupling", 0.0)),
        whm_target_abs_corr=float(loss_cfg.get("whm_target_abs_corr", 0.20)),
        whm_focus_start=int(loss_cfg.get("whm_focus_start", 8)),
        whm_focus_end=int(loss_cfg.get("whm_focus_end", 12)),
        whm_wwv_idx=int(loss_cfg.get("whm_wwv_idx", 0)),
        whm_wind_idx=int(loss_cfg.get("whm_wind_idx", 1)),
        whm_sst_idx=int(loss_cfg.get("whm_sst_idx", 2)),
        whm_alpha=float(loss_cfg.get("whm_alpha", 0.7)),
        lambda_warm_event=float(loss_cfg.get("lambda_warm_event", 0.0)),
        lambda_warm_peak=float(loss_cfg.get("lambda_warm_peak", 0.0)),
        lambda_warm_timing=float(loss_cfg.get("lambda_warm_timing", 0.0)),
        lambda_warm_non_event_reg=float(loss_cfg.get("lambda_warm_non_event_reg", 0.0)),
        warm_event_quantile=float(loss_cfg.get("warm_event_quantile", 0.8)),
        warm_event_min_amp=float(loss_cfg.get("warm_event_min_amp", 0.10)),
        warm_event_margin=float(loss_cfg.get("warm_event_margin", 0.02)),
    )
