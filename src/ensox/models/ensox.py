import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(length: int, dim: int) -> torch.Tensor:
    pe = torch.zeros(length, dim, dtype=torch.float32)
    position = torch.arange(0, length, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)


def _make_group_norm(channels: int, preferred_groups: int = 16) -> nn.GroupNorm:
    groups = min(preferred_groups, channels)
    while groups > 1 and (channels % groups != 0):
        groups -= 1
    return nn.GroupNorm(groups, channels)


def _resolve_attn_heads(embed_dim: int, requested: int) -> int:
    heads = max(1, int(requested))
    while heads > 1 and (embed_dim % heads != 0):
        heads -= 1
    return heads


class BasicBlock3D(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        self.conv1 = nn.Conv3d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=(3, 5, 5),
            stride=stride,
            padding=(1, 2, 2),
            bias=False,
        )
        self.norm1 = _make_group_norm(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv3d(
            in_channels=out_channel,
            out_channels=out_channel,
            kernel_size=(3, 5, 5),
            stride=1,
            padding=(1, 2, 2),
            bias=False,
        )
        self.norm2 = _make_group_norm(out_channel)
        self.downsample = nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=1, bias=False)

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out = out + identity
        out = self.relu(out)
        return out


class FieldEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels=in_channels, out_channels=64, kernel_size=(3, 4, 8), padding="same"),
            _make_group_norm(64),
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            BasicBlock3D(64, 64),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            BasicBlock3D(64, 64),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            BasicBlock3D(64, 128),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            BasicBlock3D(128, 256),
            nn.AvgPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
        )
        self.head = nn.Linear(256, dim)

    def forward(self, x):
        # x: [B, T, C, H, W]
        x = x.permute(0, 2, 1, 3, 4)
        x = self.conv(x)
        x = x.permute(0, 2, 1, 3, 4).flatten(2)
        x = self.head(x)
        return x


class SeasonalStateSpace(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.in_proj = nn.Linear(in_dim, hidden_dim)
        self.month_transition = nn.Parameter(torch.randn(12, hidden_dim, hidden_dim) * 0.02)
        self.norm = nn.LayerNorm(hidden_dim)

    def _step(self, h, month_idx, x_t=None):
        # h: [B, D], month_idx: [B]
        a = self.month_transition[month_idx]  # [B, D, D]
        h_next = torch.bmm(a, h.unsqueeze(-1)).squeeze(-1)
        if x_t is not None:
            h_next = h_next + self.in_proj(x_t)
        h_next = torch.tanh(self.norm(h_next))
        return h_next

    def forward(self, x, init_month):
        # x: [B, T, in_dim], init_month: [B]
        bsz, tlen, _ = x.shape
        h = torch.zeros(bsz, self.hidden_dim, device=x.device, dtype=x.dtype)
        outs = []
        for t in range(tlen):
            month = (init_month + t) % 12
            h = self._step(h, month, x[:, t])
            outs.append(h)
        return torch.stack(outs, dim=1), h

    def rollout(self, h, start_month, pred_time, forcing=None):
        outs = []
        for t in range(pred_time):
            month = (start_month + t) % 12
            x_t = None if forcing is None else forcing[:, t]
            h = self._step(h, month, x_t)
            outs.append(h)
        return torch.stack(outs, dim=1)


class WyrtkiHasselmannMemory(nn.Module):
    def __init__(
        self,
        memory_in_dim: int,
        hidden_dim: int,
        wind_idx: int = 1,
        wwv_idx: int = 0,
        sst_idx: int = 2,
    ):
        super().__init__()
        self.memory_in_dim = int(memory_in_dim)
        self.hidden_dim = int(hidden_dim)
        self.wind_idx = int(wind_idx)
        self.wwv_idx = int(wwv_idx)
        self.sst_idx = int(sst_idx)

        self.in_proj = nn.Linear(memory_in_dim, hidden_dim)
        self.driver_proj = nn.Linear(3, hidden_dim)
        self.feature_feedback = nn.Linear(hidden_dim * 2, memory_in_dim)
        self.token_out = nn.Sequential(
            nn.Linear(hidden_dim * 2 + memory_in_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
        )

        # Month-conditioned Wyrtki recharge dynamics.
        self.wyr_decay = nn.Embedding(12, hidden_dim)
        self.wyr_gain = nn.Embedding(12, hidden_dim)
        self.wyr_couple = nn.Embedding(12, hidden_dim)

        # Month-conditioned Hasselmann persistence dynamics.
        self.hass_decay = nn.Embedding(12, hidden_dim)
        self.hass_gain = nn.Embedding(12, hidden_dim)
        self.hass_couple = nn.Embedding(12, hidden_dim)

        self.bias_wyr = nn.Embedding(12, hidden_dim)
        self.bias_hass = nn.Embedding(12, hidden_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.wyr_decay.weight, mean=1.4, std=0.05)
        nn.init.normal_(self.hass_decay.weight, mean=1.8, std=0.05)
        for emb in [
            self.wyr_gain,
            self.wyr_couple,
            self.hass_gain,
            self.hass_couple,
            self.bias_wyr,
            self.bias_hass,
        ]:
            nn.init.normal_(emb.weight, mean=0.0, std=0.02)

    def _safe_index(self, idx: int) -> int:
        if self.memory_in_dim <= 0:
            return 0
        return min(max(int(idx), 0), self.memory_in_dim - 1)

    def _driver_triplet(self, x_t: torch.Tensor) -> torch.Tensor:
        # [B, M] -> [B, 3]
        i_wwv = self._safe_index(self.wwv_idx)
        i_wind = self._safe_index(self.wind_idx)
        i_sst = self._safe_index(self.sst_idx)
        return torch.stack([x_t[:, i_wwv], x_t[:, i_wind], x_t[:, i_sst]], dim=-1)

    def _step(self, wyr, hass, month_idx, forcing=None):
        decay_w = torch.sigmoid(self.wyr_decay(month_idx))
        gain_w = torch.tanh(self.wyr_gain(month_idx))
        couple_w = torch.tanh(self.wyr_couple(month_idx))

        decay_h = torch.sigmoid(self.hass_decay(month_idx))
        gain_h = torch.tanh(self.hass_gain(month_idx))
        couple_h = torch.tanh(self.hass_couple(month_idx))

        if forcing is None:
            forcing = torch.zeros_like(wyr)

        wyr_next = decay_w * wyr + gain_w * forcing + couple_w * hass + 0.1 * self.bias_wyr(month_idx)
        hass_next = decay_h * hass + gain_h * forcing + couple_h * wyr + 0.1 * self.bias_hass(month_idx)
        return wyr_next, hass_next

    def forward(self, x_hist, init_month, pred_time):
        # x_hist: [B, T, M]
        bsz, hist_len, _ = x_hist.shape
        device = x_hist.device
        dtype = x_hist.dtype

        wyr = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)
        hass = torch.zeros(bsz, self.hidden_dim, device=device, dtype=dtype)

        hist_tokens = []
        x_persist = x_hist[:, -1]

        for t in range(hist_len):
            month = (init_month + t) % 12
            x_t = x_hist[:, t]
            forcing = torch.tanh(self.in_proj(x_t) + self.driver_proj(self._driver_triplet(x_t)))
            wyr, hass = self._step(wyr, hass, month, forcing=forcing)
            token = self.token_out(torch.cat([wyr, hass, x_t], dim=-1))
            hist_tokens.append(token)

        hist_tokens = torch.stack(hist_tokens, dim=1)

        fut_tokens = []
        fut_features = []
        for k in range(int(pred_time)):
            month = (init_month + hist_len + k) % 12
            forcing = torch.tanh(self.in_proj(x_persist) + self.driver_proj(self._driver_triplet(x_persist)))
            wyr, hass = self._step(wyr, hass, month, forcing=forcing)

            x_feedback = self.feature_feedback(torch.cat([wyr, hass], dim=-1))
            x_persist = 0.88 * x_persist + 0.12 * x_feedback

            token = self.token_out(torch.cat([wyr, hass, x_persist], dim=-1))
            fut_tokens.append(token)
            fut_features.append(x_persist)

        fut_tokens = torch.stack(fut_tokens, dim=1)
        fut_features = torch.stack(fut_features, dim=1)
        return hist_tokens, fut_tokens, fut_features


class MultiScaleLeadRefiner(nn.Module):
    def __init__(self, dim: int, heads: int, layers: int = 2, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.layers = max(1, int(layers))
        k = max(3, int(kernel_size))
        if k % 2 == 0:
            k += 1

        self.dw_convs = nn.ModuleList()
        self.pw_convs = nn.ModuleList()
        self.conv_norms = nn.ModuleList()
        for i in range(self.layers):
            dilation = 2 ** i
            pad = dilation * (k // 2)
            self.dw_convs.append(
                nn.Conv1d(dim, dim, kernel_size=k, padding=pad, dilation=dilation, groups=dim, bias=False)
            )
            self.pw_convs.append(nn.Conv1d(dim, dim, kernel_size=1, bias=False))
            self.conv_norms.append(nn.LayerNorm(dim))

        attn_heads = _resolve_attn_heads(dim, heads)
        self.attn = nn.MultiheadAttention(dim, num_heads=attn_heads, dropout=dropout, batch_first=True)
        self.attn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.ffn_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.GELU()

    def forward(self, x):
        h = x
        for dw, pw, norm in zip(self.dw_convs, self.pw_convs, self.conv_norms):
            y = dw(h.transpose(1, 2))
            y = pw(self.act(y)).transpose(1, 2)
            h = norm(h + self.dropout(y))

        attn_out, _ = self.attn(h, h, h, need_weights=False)
        h = self.attn_norm(h + self.dropout(attn_out))
        ff = self.ffn(h)
        h = self.ffn_norm(h + self.dropout(ff))
        return h


class ENSOX(nn.Module):
    def __init__(
        self,
        in_channels=9,
        memory_in_dim=3,
        dim=256,
        head=4,
        depth=6,
        decoder_depth=None,
        dim_feedforward=512,
        dropout=0.1,
        obs_time=12,
        pred_time=24,
        num_index=1,
        memory_dim=128,
        long_head_enabled=False,
        long_head_start=11,
        long_head_end=18,
        long_head_scale=0.25,
        barrier_head_enabled=False,
        barrier_head_start=8,
        barrier_head_end=12,
        barrier_head_scale=-2.0,
        lead_refiner_enabled=False,
        lead_refiner_start=9,
        lead_refiner_layers=2,
        lead_refiner_heads=4,
        lead_refiner_scale=-2.2,
        lead_refiner_iters=2,
        lead_refiner_ffn=0,
        ms_refiner_enabled=False,
        ms_refiner_start=11,
        ms_refiner_layers=2,
        ms_refiner_heads=4,
        ms_refiner_scale=-2.8,
        ms_refiner_iters=1,
        ms_refiner_kernel=3,
        rollout_refiner_enabled=False,
        rollout_refiner_start=12,
        rollout_refiner_hidden=0,
        rollout_refiner_scale=-2.0,
        rollout_refiner_detach_prev=True,
        frontier_refiner_enabled=False,
        frontier_refiner_start=5,
        frontier_refiner_end=16,
        frontier_refiner_hidden=0,
        frontier_refiner_scale=-1.2,
        frontier_refiner_detach_prev=True,
        tail_booster_enabled=False,
        tail_booster_start=21,
        tail_booster_hidden=0,
        tail_booster_scale=-2.0,
        barrier_booster_enabled=False,
        barrier_booster_start=8,
        barrier_booster_end=10,
        barrier_booster_hidden=0,
        barrier_booster_scale=-0.8,
        barrier_bridge_enabled=False,
        barrier_bridge_start=8,
        barrier_bridge_end=11,
        barrier_bridge_context=2,
        barrier_bridge_hidden=0,
        barrier_bridge_layers=1,
        barrier_bridge_scale=-0.9,
        prefix_bridge_enabled=False,
        prefix_bridge_start=5,
        prefix_bridge_end=16,
        prefix_bridge_context=3,
        prefix_bridge_hidden=0,
        prefix_bridge_layers=1,
        prefix_bridge_scale=-0.7,
        hole_interp_enabled=False,
        hole_interp_lead=9,
        hole_interp_start=0,
        hole_interp_end=0,
        hole_interp_context=2,
        hole_interp_scale=0.45,
        hole_patch_enabled=False,
        hole_patch_lead=9,
        hole_patch_context=1,
        hole_patch_hidden=0,
        hole_patch_scale=-0.4,
        memory_fusion_alpha=1.0,
        gate_mode="learned",
        use_month_embedding=True,
        use_lead_embedding=True,
        legacy_skip_enabled=False,
        legacy_skip_alpha=1.0,
        memory_mode="legacy_ssm",
        dual_memory_hidden=0,
        memory_driver_wind_idx=1,
        memory_driver_wwv_idx=0,
        memory_driver_sst_idx=2,
        memory_cross_attn_enabled=False,
        memory_cross_attn_heads=4,
        memory_residual_scale=0.8,
        memory_bridge_scale=0.2,
    ):
        super().__init__()
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.num_index = num_index
        self.long_head_enabled = bool(long_head_enabled)
        self.barrier_head_enabled = bool(barrier_head_enabled)
        self.lead_refiner_enabled = bool(lead_refiner_enabled)
        self.lead_refiner_iters = max(1, int(lead_refiner_iters))
        self.ms_refiner_enabled = bool(ms_refiner_enabled)
        self.ms_refiner_iters = max(1, int(ms_refiner_iters))
        self.rollout_refiner_enabled = bool(rollout_refiner_enabled)
        self.rollout_refiner_detach_prev = bool(rollout_refiner_detach_prev)
        self.frontier_refiner_enabled = bool(frontier_refiner_enabled)
        self.frontier_refiner_detach_prev = bool(frontier_refiner_detach_prev)
        self.tail_booster_enabled = bool(tail_booster_enabled)
        self.barrier_booster_enabled = bool(barrier_booster_enabled)
        self.barrier_bridge_enabled = bool(barrier_bridge_enabled)
        self.prefix_bridge_enabled = bool(prefix_bridge_enabled)
        self.hole_interp_enabled = bool(hole_interp_enabled)
        self.hole_patch_enabled = bool(hole_patch_enabled)
        self.memory_fusion_alpha = float(memory_fusion_alpha)
        self.gate_mode = str(gate_mode).lower()
        self.use_month_embedding = bool(use_month_embedding)
        self.use_lead_embedding = bool(use_lead_embedding)
        self.legacy_skip_enabled = bool(legacy_skip_enabled)
        self.legacy_skip_alpha = float(legacy_skip_alpha)
        self.memory_mode = str(memory_mode).lower().strip()
        self.memory_cross_attn_enabled = bool(memory_cross_attn_enabled)

        self.field_encoder = FieldEncoder(in_channels=in_channels, dim=dim)
        self.field_norm = nn.LayerNorm(dim)
        self.decoder_norm = nn.LayerNorm(dim)

        self.register_buffer("time_embedding", sinusoidal_embedding(obs_time + pred_time + 1, dim), persistent=False)
        self.month_embedding = nn.Embedding(12, dim)
        self.lead_embedding = nn.Embedding(pred_time, dim)
        self.query_tokens = nn.Parameter(torch.randn(pred_time, dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=dim, nhead=head, dim_feedforward=dim_feedforward, dropout=dropout, batch_first=True
        )
        dec_layers = int(decoder_depth) if decoder_depth is not None else max(2, depth // 2)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=dec_layers)

        # Baseline seasonal memory branch.
        self.memory_proj = nn.Sequential(
            nn.Linear(memory_in_dim, memory_dim),
            nn.GELU(),
            nn.LayerNorm(memory_dim),
        )
        self.memory_ssm = SeasonalStateSpace(memory_dim, memory_dim)
        self.memory_head = nn.Linear(memory_dim, num_index)
        self.memory_feature_head = nn.Linear(memory_dim, memory_in_dim)

        # Physical dual-memory branch inspired by Wyrtki + Hasselmann memory.
        dual_hidden = int(dual_memory_hidden) if int(dual_memory_hidden) > 0 else int(memory_dim)
        self.physical_memory = WyrtkiHasselmannMemory(
            memory_in_dim=memory_in_dim,
            hidden_dim=dual_hidden,
            wind_idx=memory_driver_wind_idx,
            wwv_idx=memory_driver_wwv_idx,
            sst_idx=memory_driver_sst_idx,
        )
        self.physical_hist_proj = nn.Linear(dual_hidden, memory_dim)
        self.physical_roll_proj = nn.Linear(dual_hidden, memory_dim)
        self.physical_index_head = nn.Linear(memory_dim, num_index)

        self.memory_mix_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid(),
        )
        self.memory_hist_mix_gate = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
            nn.Sigmoid(),
        )
        self.memory_bridge = nn.Sequential(
            nn.Linear(memory_dim * 2, memory_dim),
            nn.GELU(),
            nn.Linear(memory_dim, memory_dim),
        )
        self.memory_bridge_scale = nn.Parameter(torch.tensor(float(memory_bridge_scale), dtype=torch.float32))
        self.hybrid_mem_blend = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.feature_mix_gate = nn.Sequential(
            nn.Linear(memory_in_dim * 2, memory_in_dim),
            nn.Sigmoid(),
        )
        nn.init.constant_(self.memory_mix_gate[2].bias, -2.0)
        nn.init.constant_(self.memory_hist_mix_gate[2].bias, -2.0)
        nn.init.constant_(self.feature_mix_gate[0].bias, -2.0)

        if self.memory_cross_attn_enabled:
            attn_heads = _resolve_attn_heads(memory_dim, memory_cross_attn_heads)
            self.query_to_memory = nn.Linear(dim, memory_dim)
            self.memory_cross_attn = nn.MultiheadAttention(
                embed_dim=memory_dim,
                num_heads=attn_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.memory_to_query = nn.Linear(memory_dim, dim)
            self.memory_residual_scale = nn.Parameter(torch.tensor(float(memory_residual_scale), dtype=torch.float32))
        else:
            self.query_to_memory = None
            self.memory_cross_attn = None
            self.memory_to_query = None
            self.memory_residual_scale = None

        self.deep_head = nn.Linear(dim, num_index)
        self.gate = nn.Sequential(
            nn.Linear(dim + memory_dim + dim, dim),
            nn.GELU(),
            nn.Linear(dim, num_index),
            nn.Sigmoid(),
        )

        # Legacy CTEFNet compatibility branch.
        self.encoder_head = nn.Sequential(
            nn.Linear(obs_time * dim, pred_time * num_index),
        )
        self.res = nn.Parameter(torch.rand(dim, dtype=torch.float32), requires_grad=True)
        self.res_norm = nn.LayerNorm(dim)

        if self.long_head_enabled:
            self.long_head = nn.Sequential(
                nn.Linear(dim + memory_dim + dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_index),
            )
            self.long_head_gate = nn.Sequential(
                nn.Linear(dim, num_index),
                nn.Sigmoid(),
            )
            long_start = max(1, int(long_head_start))
            long_end = max(long_start, int(long_head_end))
            mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            mask[:, :, long_start - 1 : min(long_end, pred_time)] = 1.0
            self.register_buffer("long_mask", mask, persistent=False)
            self.long_scale = nn.Parameter(torch.tensor(float(long_head_scale), dtype=torch.float32))
        else:
            self.long_head = None
            self.long_head_gate = None
            self.long_scale = None

        if self.barrier_head_enabled:
            self.barrier_head = nn.Sequential(
                nn.Linear(dim + memory_dim + dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_index),
            )
            self.barrier_gate = nn.Sequential(
                nn.Linear(dim, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.barrier_head[3].weight)
            nn.init.zeros_(self.barrier_head[3].bias)
            nn.init.constant_(self.barrier_gate[0].bias, -2.0)
            b_start = max(1, int(barrier_head_start))
            b_end = max(b_start, int(barrier_head_end))
            b_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            b_mask[:, :, b_start - 1 : min(b_end, pred_time)] = 1.0
            self.register_buffer("barrier_mask", b_mask, persistent=False)
            self.barrier_scale = nn.Parameter(torch.tensor(float(barrier_head_scale), dtype=torch.float32))
        else:
            self.barrier_head = None
            self.barrier_gate = None
            self.barrier_mask = None
            self.barrier_scale = None

        if self.lead_refiner_enabled:
            refiner_ffn = int(lead_refiner_ffn) if int(lead_refiner_ffn) > 0 else int(dim_feedforward)
            refiner_heads = _resolve_attn_heads(dim, int(lead_refiner_heads))
            refiner_layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=refiner_heads,
                dim_feedforward=refiner_ffn,
                dropout=dropout,
                batch_first=True,
            )
            self.lead_refiner = nn.TransformerEncoder(refiner_layer, num_layers=max(1, int(lead_refiner_layers)))
            self.index_token_proj = nn.Linear(num_index, dim)
            self.memory_token_proj = nn.Sequential(
                nn.Linear(memory_dim + memory_in_dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
            )
            self.refiner_head = nn.Linear(dim, num_index)
            self.refiner_gate = nn.Sequential(
                nn.Linear(dim, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.refiner_head.weight)
            nn.init.zeros_(self.refiner_head.bias)
            nn.init.constant_(self.refiner_gate[0].bias, -2.0)
            ref_start = max(1, int(lead_refiner_start))
            ref_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            ref_mask[:, :, ref_start - 1 :] = 1.0
            self.register_buffer("refiner_mask", ref_mask, persistent=False)
            self.lead_refiner_scale = nn.Parameter(torch.tensor(float(lead_refiner_scale), dtype=torch.float32))
        else:
            self.lead_refiner = None
            self.index_token_proj = None
            self.memory_token_proj = None
            self.refiner_head = None
            self.refiner_gate = None
            self.refiner_mask = None
            self.lead_refiner_scale = None

        if self.ms_refiner_enabled:
            ms_heads = _resolve_attn_heads(dim, int(ms_refiner_heads))
            self.ms_index_proj = nn.Linear(num_index, dim)
            self.ms_memory_proj = nn.Sequential(
                nn.Linear(memory_dim + memory_in_dim, dim),
                nn.GELU(),
                nn.LayerNorm(dim),
            )
            self.ms_refiner = MultiScaleLeadRefiner(
                dim=dim,
                heads=ms_heads,
                layers=max(1, int(ms_refiner_layers)),
                kernel_size=max(3, int(ms_refiner_kernel)),
                dropout=dropout,
            )
            self.ms_head = nn.Linear(dim, num_index)
            self.ms_gate = nn.Sequential(
                nn.Linear(dim, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.ms_head.weight)
            nn.init.zeros_(self.ms_head.bias)
            nn.init.constant_(self.ms_gate[0].bias, -2.0)
            ms_start = max(1, int(ms_refiner_start))
            ms_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            ms_mask[:, :, ms_start - 1 :] = 1.0
            self.register_buffer("ms_refiner_mask", ms_mask, persistent=False)
            self.ms_refiner_scale = nn.Parameter(torch.tensor(float(ms_refiner_scale), dtype=torch.float32))
        else:
            self.ms_index_proj = None
            self.ms_memory_proj = None
            self.ms_refiner = None
            self.ms_head = None
            self.ms_gate = None
            self.ms_refiner_mask = None
            self.ms_refiner_scale = None

        if self.rollout_refiner_enabled:
            rollout_hidden = int(rollout_refiner_hidden) if int(rollout_refiner_hidden) > 0 else int(dim)
            self.rollout_in_proj = nn.Sequential(
                nn.Linear(dim + memory_dim + num_index, rollout_hidden),
                nn.GELU(),
            )
            self.rollout_cell = nn.GRUCell(rollout_hidden, rollout_hidden)
            self.rollout_norm = nn.LayerNorm(rollout_hidden)
            self.rollout_head = nn.Linear(rollout_hidden, num_index)
            self.rollout_gate = nn.Sequential(
                nn.Linear(rollout_hidden, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.rollout_head.weight)
            nn.init.zeros_(self.rollout_head.bias)
            nn.init.constant_(self.rollout_gate[0].bias, -2.0)
            ro_start = max(1, int(rollout_refiner_start))
            ro_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            ro_mask[:, :, ro_start - 1 :] = 1.0
            self.register_buffer("rollout_refiner_mask", ro_mask, persistent=False)
            self.rollout_refiner_scale = nn.Parameter(torch.tensor(float(rollout_refiner_scale), dtype=torch.float32))
        else:
            self.rollout_in_proj = None
            self.rollout_cell = None
            self.rollout_norm = None
            self.rollout_head = None
            self.rollout_gate = None
            self.rollout_refiner_mask = None
            self.rollout_refiner_scale = None

        if self.frontier_refiner_enabled:
            frontier_hidden = int(frontier_refiner_hidden) if int(frontier_refiner_hidden) > 0 else int(dim)
            frontier_in_dim = int(dim + memory_dim + num_index * 3)
            self.frontier_in_proj = nn.Sequential(
                nn.Linear(frontier_in_dim, frontier_hidden),
                nn.GELU(),
            )
            self.frontier_cell = nn.GRUCell(frontier_hidden, frontier_hidden)
            self.frontier_norm = nn.LayerNorm(frontier_hidden)
            self.frontier_head = nn.Linear(frontier_hidden, num_index)
            self.frontier_gate = nn.Sequential(
                nn.Linear(frontier_hidden, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.frontier_head.weight)
            nn.init.zeros_(self.frontier_head.bias)
            nn.init.constant_(self.frontier_gate[0].bias, -1.5)
            fr_start = max(1, int(frontier_refiner_start))
            fr_end = max(fr_start, int(frontier_refiner_end))
            fr_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            fr_mask[:, :, fr_start - 1 : min(fr_end, pred_time)] = 1.0
            self.register_buffer("frontier_refiner_mask", fr_mask, persistent=False)
            self.frontier_refiner_scale = nn.Parameter(torch.tensor(float(frontier_refiner_scale), dtype=torch.float32))
        else:
            self.frontier_in_proj = None
            self.frontier_cell = None
            self.frontier_norm = None
            self.frontier_head = None
            self.frontier_gate = None
            self.frontier_refiner_mask = None
            self.frontier_refiner_scale = None

        if self.tail_booster_enabled:
            tail_hidden = int(tail_booster_hidden) if int(tail_booster_hidden) > 0 else int(dim)
            self.tail_booster_proj = nn.Sequential(
                nn.Linear(dim + memory_dim + num_index, tail_hidden),
                nn.GELU(),
                nn.LayerNorm(tail_hidden),
            )
            self.tail_booster_conv = nn.Sequential(
                nn.Conv1d(tail_hidden, tail_hidden, kernel_size=3, padding=1, groups=1, bias=False),
                nn.GELU(),
                nn.Conv1d(tail_hidden, tail_hidden, kernel_size=1, bias=False),
            )
            self.tail_booster_norm = nn.LayerNorm(tail_hidden)
            self.tail_booster_head = nn.Linear(tail_hidden, num_index)
            self.tail_booster_gate = nn.Sequential(
                nn.Linear(tail_hidden, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.tail_booster_head.weight)
            nn.init.zeros_(self.tail_booster_head.bias)
            nn.init.constant_(self.tail_booster_gate[0].bias, -2.0)
            tb_start = max(1, int(tail_booster_start))
            tb_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            tb_mask[:, :, tb_start - 1 :] = 1.0
            self.register_buffer("tail_booster_mask", tb_mask, persistent=False)
            self.tail_booster_scale = nn.Parameter(torch.tensor(float(tail_booster_scale), dtype=torch.float32))
        else:
            self.tail_booster_proj = None
            self.tail_booster_conv = None
            self.tail_booster_norm = None
            self.tail_booster_head = None
            self.tail_booster_gate = None
            self.tail_booster_mask = None
            self.tail_booster_scale = None

        if self.barrier_booster_enabled:
            barrier_hidden = int(barrier_booster_hidden) if int(barrier_booster_hidden) > 0 else int(dim)
            self.barrier_booster_proj = nn.Sequential(
                nn.Linear(dim + memory_dim + num_index, barrier_hidden),
                nn.GELU(),
                nn.LayerNorm(barrier_hidden),
            )
            self.barrier_booster_conv = nn.Sequential(
                nn.Conv1d(barrier_hidden, barrier_hidden, kernel_size=3, padding=1, groups=1, bias=False),
                nn.GELU(),
                nn.Conv1d(barrier_hidden, barrier_hidden, kernel_size=1, bias=False),
            )
            self.barrier_booster_norm = nn.LayerNorm(barrier_hidden)
            self.barrier_booster_head = nn.Linear(barrier_hidden, num_index)
            self.barrier_booster_gate = nn.Sequential(
                nn.Linear(barrier_hidden, num_index),
                nn.Sigmoid(),
            )
            nn.init.constant_(self.barrier_booster_gate[0].bias, -0.2)
            bb_start = max(1, int(barrier_booster_start))
            bb_end = max(bb_start, int(barrier_booster_end))
            bb_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            bb_mask[:, :, bb_start - 1 : min(bb_end, pred_time)] = 1.0
            self.register_buffer("barrier_booster_mask", bb_mask, persistent=False)
            self.barrier_booster_scale = nn.Parameter(torch.tensor(float(barrier_booster_scale), dtype=torch.float32))
        else:
            self.barrier_booster_proj = None
            self.barrier_booster_conv = None
            self.barrier_booster_norm = None
            self.barrier_booster_head = None
            self.barrier_booster_gate = None
            self.barrier_booster_mask = None
            self.barrier_booster_scale = None

        if self.barrier_bridge_enabled:
            bridge_hidden = int(barrier_bridge_hidden) if int(barrier_bridge_hidden) > 0 else max(int(dim) // 2, int(num_index) * 8)
            bridge_layers = max(1, int(barrier_bridge_layers))
            bridge_in_dim = int(dim + memory_dim + num_index * 5)
            self.barrier_bridge_proj = nn.Sequential(
                nn.Linear(bridge_in_dim, bridge_hidden),
                nn.GELU(),
                nn.LayerNorm(bridge_hidden),
            )
            self.barrier_bridge_rnn = nn.GRU(
                input_size=bridge_hidden,
                hidden_size=bridge_hidden,
                num_layers=bridge_layers,
                batch_first=True,
                dropout=dropout if bridge_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.barrier_bridge_norm = nn.LayerNorm(bridge_hidden * 2)
            self.barrier_bridge_head = nn.Linear(bridge_hidden * 2, num_index)
            self.barrier_bridge_gate = nn.Sequential(
                nn.Linear(bridge_hidden * 2, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.barrier_bridge_head.weight)
            nn.init.zeros_(self.barrier_bridge_head.bias)
            nn.init.constant_(self.barrier_bridge_gate[0].bias, -1.2)
            bridge_start = max(1, int(barrier_bridge_start))
            bridge_end = max(bridge_start, int(barrier_bridge_end))
            bridge_context = max(1, int(barrier_bridge_context))
            bridge_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            bridge_mask[:, :, bridge_start - 1 : min(bridge_end, pred_time)] = 1.0
            self.register_buffer("barrier_bridge_mask", bridge_mask, persistent=False)
            self.barrier_bridge_window_start = max(0, bridge_start - 1 - bridge_context)
            self.barrier_bridge_window_end = min(pred_time, bridge_end + bridge_context)
            self.barrier_bridge_scale = nn.Parameter(torch.tensor(float(barrier_bridge_scale), dtype=torch.float32))
        else:
            self.barrier_bridge_proj = None
            self.barrier_bridge_rnn = None
            self.barrier_bridge_norm = None
            self.barrier_bridge_head = None
            self.barrier_bridge_gate = None
            self.barrier_bridge_mask = None
            self.barrier_bridge_window_start = 0
            self.barrier_bridge_window_end = 0
            self.barrier_bridge_scale = None

        if self.prefix_bridge_enabled:
            prefix_hidden = int(prefix_bridge_hidden) if int(prefix_bridge_hidden) > 0 else max(int(dim) // 2, int(num_index) * 10)
            prefix_layers = max(1, int(prefix_bridge_layers))
            prefix_in_dim = int(dim + memory_dim + num_index * 7)
            self.prefix_bridge_proj = nn.Sequential(
                nn.Linear(prefix_in_dim, prefix_hidden),
                nn.GELU(),
                nn.LayerNorm(prefix_hidden),
            )
            self.prefix_bridge_rnn = nn.GRU(
                input_size=prefix_hidden,
                hidden_size=prefix_hidden,
                num_layers=prefix_layers,
                batch_first=True,
                dropout=dropout if prefix_layers > 1 else 0.0,
                bidirectional=True,
            )
            self.prefix_bridge_norm = nn.LayerNorm(prefix_hidden * 2)
            self.prefix_bridge_head = nn.Linear(prefix_hidden * 2, num_index)
            self.prefix_bridge_gate = nn.Sequential(
                nn.Linear(prefix_hidden * 2, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.prefix_bridge_head.weight)
            nn.init.zeros_(self.prefix_bridge_head.bias)
            nn.init.constant_(self.prefix_bridge_gate[0].bias, -0.9)
            prefix_start = max(1, int(prefix_bridge_start))
            prefix_end = max(prefix_start, int(prefix_bridge_end))
            prefix_context = max(1, int(prefix_bridge_context))
            prefix_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            prefix_mask[:, :, prefix_start - 1 : min(prefix_end, pred_time)] = 1.0
            self.register_buffer("prefix_bridge_mask", prefix_mask, persistent=False)
            self.prefix_bridge_window_start = max(0, prefix_start - 1 - prefix_context)
            self.prefix_bridge_window_end = min(pred_time, prefix_end + prefix_context)
            self.prefix_bridge_scale = nn.Parameter(torch.tensor(float(prefix_bridge_scale), dtype=torch.float32))
        else:
            self.prefix_bridge_proj = None
            self.prefix_bridge_rnn = None
            self.prefix_bridge_norm = None
            self.prefix_bridge_head = None
            self.prefix_bridge_gate = None
            self.prefix_bridge_mask = None
            self.prefix_bridge_window_start = 0
            self.prefix_bridge_window_end = 0
            self.prefix_bridge_scale = None

        if self.hole_interp_enabled:
            interp_start = int(hole_interp_start) if int(hole_interp_start) > 0 else int(hole_interp_lead)
            interp_end = int(hole_interp_end) if int(hole_interp_end) > 0 else int(hole_interp_lead)
            interp_start = min(max(interp_start, 1), pred_time)
            interp_end = min(max(interp_end, interp_start), pred_time)
            interp_context = max(1, int(hole_interp_context))
            interp_offsets = [off for off in range(-interp_context, interp_context + 1) if off != 0]
            interp_weights = [1.0 / float(abs(off)) for off in interp_offsets]
            interp_norm = sum(interp_weights) if interp_weights else 1.0
            interp_weights = [w / interp_norm for w in interp_weights]
            interp_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            interp_mask[:, :, interp_start - 1 : interp_end] = 1.0
            self.register_buffer("hole_interp_mask", interp_mask, persistent=False)
            self.register_buffer(
                "hole_interp_offsets",
                torch.tensor(interp_offsets, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "hole_interp_weights",
                torch.tensor(interp_weights, dtype=torch.float32),
                persistent=False,
            )
            self.hole_interp_window_start = int(interp_start - 1)
            self.hole_interp_window_end = int(interp_end)
            self.hole_interp_context = interp_context
            self.hole_interp_scale = nn.Parameter(
                torch.full((int(num_index),), float(hole_interp_scale), dtype=torch.float32)
            )
        else:
            self.hole_interp_mask = None
            self.hole_interp_offsets = None
            self.hole_interp_weights = None
            self.hole_interp_window_start = 0
            self.hole_interp_window_end = 0
            self.hole_interp_context = 1
            self.hole_interp_scale = None

        if self.hole_patch_enabled:
            hole_hidden = int(hole_patch_hidden) if int(hole_patch_hidden) > 0 else max(int(dim) // 2, int(num_index) * 12)
            hole_in_dim = int(dim + memory_dim + num_index * 10)
            self.hole_patch_proj = nn.Sequential(
                nn.Linear(hole_in_dim, hole_hidden),
                nn.GELU(),
                nn.LayerNorm(hole_hidden),
            )
            self.hole_patch_head = nn.Linear(hole_hidden, num_index)
            self.hole_patch_gate = nn.Sequential(
                nn.Linear(hole_hidden, num_index),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.hole_patch_head.weight)
            nn.init.zeros_(self.hole_patch_head.bias)
            nn.init.constant_(self.hole_patch_gate[0].bias, -0.6)
            hole_idx = min(max(int(hole_patch_lead) - 1, 0), pred_time - 1)
            hole_mask = torch.zeros((1, 1, pred_time), dtype=torch.float32)
            hole_mask[:, :, hole_idx] = 1.0
            self.register_buffer("hole_patch_mask", hole_mask, persistent=False)
            self.hole_patch_index = int(hole_idx)
            self.hole_patch_context = max(1, int(hole_patch_context))
            self.hole_patch_scale = nn.Parameter(torch.tensor(float(hole_patch_scale), dtype=torch.float32))
        else:
            self.hole_patch_proj = None
            self.hole_patch_head = None
            self.hole_patch_gate = None
            self.hole_patch_mask = None
            self.hole_patch_index = 0
            self.hole_patch_context = 1
            self.hole_patch_scale = None

    def _build_memory_branches(self, x_memory, init_month):
        # Baseline SSM branch.
        mem_inputs = self.memory_proj(x_memory)
        mem_hist_ssm, mem_last = self.memory_ssm(mem_inputs, init_month)
        pred_start_month = (init_month + x_memory.shape[1]) % 12
        mem_roll_ssm = self.memory_ssm.rollout(mem_last, pred_start_month, self.pred_time)
        mem_pred_ssm = self.memory_head(mem_roll_ssm).transpose(1, 2)
        mem_feat_ssm = self.memory_feature_head(mem_roll_ssm)

        # Physical Wyrtki-Hasselmann branch.
        phy_hist_raw, phy_roll_raw, phy_feat = self.physical_memory(x_memory, init_month, self.pred_time)
        phy_hist = self.physical_hist_proj(phy_hist_raw)
        phy_roll = self.physical_roll_proj(phy_roll_raw)
        mem_pred_phy = self.physical_index_head(phy_roll).transpose(1, 2)

        if self.memory_mode == "legacy_ssm":
            mem_hist = mem_hist_ssm
            mem_roll = mem_roll_ssm
            mem_pred = mem_pred_ssm
            mem_feat_pred = mem_feat_ssm
        elif self.memory_mode == "dual_physical":
            mem_hist = phy_hist
            mem_roll = phy_roll
            mem_pred = mem_pred_phy
            mem_feat_pred = phy_feat
        else:
            roll_mix = self.memory_mix_gate(torch.cat([mem_roll_ssm, phy_roll], dim=-1))
            hist_mix = self.memory_hist_mix_gate(torch.cat([mem_hist_ssm, phy_hist], dim=-1))
            bridge = self.memory_bridge(torch.cat([mem_roll_ssm, phy_roll], dim=-1))

            mem_roll = roll_mix * phy_roll + (1.0 - roll_mix) * mem_roll_ssm
            mem_roll = mem_roll + torch.tanh(self.memory_bridge_scale) * bridge
            mem_hist = hist_mix * phy_hist + (1.0 - hist_mix) * mem_hist_ssm

            blend = torch.sigmoid(self.hybrid_mem_blend).view(1, 1, 1)
            mem_pred = blend * mem_pred_phy + (1.0 - blend) * mem_pred_ssm

            feat_mix = self.feature_mix_gate(torch.cat([mem_feat_ssm, phy_feat], dim=-1))
            mem_feat_pred = feat_mix * phy_feat + (1.0 - feat_mix) * mem_feat_ssm

        return mem_hist, mem_roll, mem_pred, mem_feat_pred

    def forward(self, x_field, x_memory, init_month):
        # x_field: [B, obs, C, H, W], x_memory: [B, obs, M], init_month: [B]
        bsz, obs, _, _, _ = x_field.shape
        lead_ids = torch.arange(self.pred_time, device=x_field.device, dtype=torch.long)
        obs_ids = torch.arange(obs, device=x_field.device, dtype=torch.long)

        obs_month = (init_month[:, None] + obs_ids[None, :]) % 12
        pred_start_month = (init_month + obs) % 12
        pred_month = (pred_start_month[:, None] + lead_ids[None, :]) % 12

        field_tokens_raw = self.field_encoder(x_field)
        field_tokens = field_tokens_raw + self.time_embedding[:, :obs, :]
        if self.use_month_embedding:
            field_tokens = field_tokens + self.month_embedding(obs_month)
        field_tokens = self.field_norm(field_tokens)
        field_ctx = self.encoder(field_tokens)

        mem_hist, mem_roll, mem_pred, mem_feat_pred = self._build_memory_branches(x_memory, init_month)

        query = self.query_tokens.unsqueeze(0).expand(bsz, -1, -1)
        if self.use_lead_embedding:
            query = query + self.lead_embedding(lead_ids)[None, :, :]
        if self.use_month_embedding:
            query = query + self.month_embedding(pred_month)

        deep_ctx = self.decoder(tgt=self.decoder_norm(query), memory=field_ctx)

        if self.memory_cross_attn_enabled:
            memory_bank = torch.cat([mem_hist, mem_roll], dim=1)
            q_mem = self.query_to_memory(deep_ctx)
            mem_ctx, _ = self.memory_cross_attn(q_mem, memory_bank, memory_bank, need_weights=False)
            deep_ctx = deep_ctx + torch.tanh(self.memory_residual_scale) * self.memory_to_query(mem_ctx)

        deep_pred = self.deep_head(deep_ctx).transpose(1, 2)

        gate_in = torch.cat([deep_ctx, mem_roll, query], dim=-1)
        if self.gate_mode == "fixed_one":
            gate = torch.ones(
                (bsz, self.num_index, self.pred_time),
                device=deep_pred.device,
                dtype=deep_pred.dtype,
            )
        else:
            gate = self.gate(gate_in).transpose(1, 2)

        index_pred = self.memory_fusion_alpha * mem_pred + gate * deep_pred

        if self.legacy_skip_enabled:
            legacy_tokens = self.res_norm(field_ctx + field_tokens_raw * self.res)
            if legacy_tokens.size(1) != self.obs_time:
                legacy_tokens = F.interpolate(
                    legacy_tokens.transpose(1, 2),
                    size=self.obs_time,
                    mode="linear",
                    align_corners=False,
                ).transpose(1, 2)
            skip = self.encoder_head(legacy_tokens.flatten(1)).view(-1, self.num_index, self.pred_time)
            index_pred = index_pred + self.legacy_skip_alpha * skip

        long_res = None
        long_gate = None
        if self.long_head_enabled:
            long_res = self.long_head(gate_in).transpose(1, 2)
            long_gate = self.long_head_gate(deep_ctx).transpose(1, 2)
            long_term = torch.tanh(self.long_scale) * long_res * long_gate * self.long_mask.to(index_pred.dtype)
            index_pred = index_pred + long_term

        barrier_res = None
        barrier_gate = None
        if self.barrier_head_enabled:
            barrier_res = self.barrier_head(gate_in).transpose(1, 2)
            barrier_gate = self.barrier_gate(deep_ctx).transpose(1, 2)
            barrier_term = (
                torch.sigmoid(self.barrier_scale)
                * barrier_res
                * barrier_gate
                * self.barrier_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + barrier_term

        refine_tokens = deep_ctx
        refiner_res = None
        refiner_gate = None
        if self.lead_refiner_enabled:
            idx_tokens = self.index_token_proj(index_pred.transpose(1, 2))
            mem_tokens = self.memory_token_proj(torch.cat([mem_roll, mem_feat_pred], dim=-1))
            refine_tokens = deep_ctx + idx_tokens + mem_tokens
            for _ in range(self.lead_refiner_iters):
                refine_tokens = self.lead_refiner(refine_tokens)
            refiner_res = self.refiner_head(refine_tokens).transpose(1, 2)
            refiner_gate = self.refiner_gate(refine_tokens).transpose(1, 2)
            refiner_term = (
                torch.sigmoid(self.lead_refiner_scale)
                * refiner_res
                * refiner_gate
                * self.refiner_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + refiner_term

        ms_res = None
        ms_gate = None
        if self.ms_refiner_enabled:
            ms_idx = self.ms_index_proj(index_pred.transpose(1, 2))
            ms_mem = self.ms_memory_proj(torch.cat([mem_roll, mem_feat_pred], dim=-1))
            ms_tokens = refine_tokens + ms_idx + ms_mem
            for _ in range(self.ms_refiner_iters):
                ms_tokens = self.ms_refiner(ms_tokens)
            ms_res = self.ms_head(ms_tokens).transpose(1, 2)
            ms_gate = self.ms_gate(ms_tokens).transpose(1, 2)
            ms_term = (
                torch.sigmoid(self.ms_refiner_scale)
                * ms_res
                * ms_gate
                * self.ms_refiner_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + ms_term

        rollout_res = None
        rollout_gate = None
        if self.rollout_refiner_enabled:
            rollout_inputs = torch.cat([deep_ctx, mem_roll], dim=-1)
            h = torch.zeros(
                bsz,
                int(self.rollout_cell.hidden_size),
                device=index_pred.device,
                dtype=index_pred.dtype,
            )
            prev = index_pred[:, :, 0]
            rollout_res_steps = []
            rollout_gate_steps = []
            for t in range(self.pred_time):
                step_in = torch.cat([rollout_inputs[:, t, :], prev], dim=-1)
                h = self.rollout_cell(self.rollout_in_proj(step_in), h)
                h = self.rollout_norm(h)
                step_res = self.rollout_head(h)
                step_gate = self.rollout_gate(h)
                rollout_res_steps.append(step_res)
                rollout_gate_steps.append(step_gate)
                step_term = torch.sigmoid(self.rollout_refiner_scale) * step_res * step_gate
                prev_step = index_pred[:, :, t] + step_term
                prev = prev_step.detach() if self.rollout_refiner_detach_prev else prev_step

            rollout_res = torch.stack(rollout_res_steps, dim=1).transpose(1, 2)
            rollout_gate = torch.stack(rollout_gate_steps, dim=1).transpose(1, 2)
            rollout_term = (
                torch.sigmoid(self.rollout_refiner_scale)
                * rollout_res
                * rollout_gate
                * self.rollout_refiner_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + rollout_term

        tail_res = None
        tail_gate = None
        if self.tail_booster_enabled:
            tail_in = torch.cat([deep_ctx, mem_roll, index_pred.transpose(1, 2)], dim=-1)
            tail_tokens = self.tail_booster_proj(tail_in)
            tail_conv = self.tail_booster_conv(tail_tokens.transpose(1, 2)).transpose(1, 2)
            tail_tokens = self.tail_booster_norm(tail_tokens + tail_conv)
            tail_res = self.tail_booster_head(tail_tokens).transpose(1, 2)
            tail_gate = self.tail_booster_gate(tail_tokens).transpose(1, 2)
            tail_term = (
                torch.sigmoid(self.tail_booster_scale)
                * tail_res
                * tail_gate
                * self.tail_booster_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + tail_term

        barrier_booster_res = None
        barrier_booster_gate = None
        if self.barrier_booster_enabled:
            barrier_in = torch.cat([deep_ctx, mem_roll, index_pred.transpose(1, 2)], dim=-1)
            barrier_tokens = self.barrier_booster_proj(barrier_in)
            barrier_conv = self.barrier_booster_conv(barrier_tokens.transpose(1, 2)).transpose(1, 2)
            barrier_tokens = self.barrier_booster_norm(barrier_tokens + barrier_conv)
            barrier_booster_res = self.barrier_booster_head(barrier_tokens).transpose(1, 2)
            barrier_booster_gate = self.barrier_booster_gate(barrier_tokens).transpose(1, 2)
            barrier_booster_term = (
                torch.sigmoid(self.barrier_booster_scale)
                * barrier_booster_res
                * barrier_booster_gate
                * self.barrier_booster_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + barrier_booster_term

        barrier_bridge_res = None
        barrier_bridge_gate = None
        if self.barrier_bridge_enabled:
            curr_seq = index_pred.transpose(1, 2)
            left_seq = torch.cat([curr_seq[:, :1, :], curr_seq[:, :-1, :]], dim=1)
            right_seq = torch.cat([curr_seq[:, 1:, :], curr_seq[:, -1:, :]], dim=1)
            bridge_in = torch.cat(
                [deep_ctx, mem_roll, curr_seq, left_seq, right_seq, mem_pred.transpose(1, 2), deep_pred.transpose(1, 2)],
                dim=-1,
            )
            ws = int(self.barrier_bridge_window_start)
            we = int(self.barrier_bridge_window_end)
            bridge_tokens = self.barrier_bridge_proj(bridge_in[:, ws:we, :])
            bridge_hidden, _ = self.barrier_bridge_rnn(bridge_tokens)
            bridge_hidden = self.barrier_bridge_norm(bridge_hidden)
            bridge_res_local = self.barrier_bridge_head(bridge_hidden)
            bridge_gate_local = self.barrier_bridge_gate(bridge_hidden)
            barrier_bridge_res = torch.zeros_like(index_pred)
            barrier_bridge_gate = torch.zeros_like(index_pred)
            barrier_bridge_res[:, :, ws:we] = bridge_res_local.transpose(1, 2)
            barrier_bridge_gate[:, :, ws:we] = bridge_gate_local.transpose(1, 2)
            barrier_bridge_term = (
                torch.sigmoid(self.barrier_bridge_scale)
                * barrier_bridge_res
                * barrier_bridge_gate
                * self.barrier_bridge_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + barrier_bridge_term

        prefix_bridge_res = None
        prefix_bridge_gate = None
        if self.prefix_bridge_enabled:
            curr_seq = index_pred.transpose(1, 2)
            left_seq = torch.cat([curr_seq[:, :1, :], curr_seq[:, :-1, :]], dim=1)
            right_seq = torch.cat([curr_seq[:, 1:, :], curr_seq[:, -1:, :]], dim=1)
            prefix_denom = torch.arange(
                1, self.pred_time + 1, device=index_pred.device, dtype=index_pred.dtype
            ).view(1, self.pred_time, 1)
            suffix_denom = torch.arange(
                self.pred_time, 0, -1, device=index_pred.device, dtype=index_pred.dtype
            ).view(1, self.pred_time, 1)
            prefix_mean_seq = torch.cumsum(curr_seq, dim=1) / prefix_denom
            suffix_mean_seq = torch.flip(torch.cumsum(torch.flip(curr_seq, dims=[1]), dim=1), dims=[1]) / suffix_denom
            prefix_in = torch.cat(
                [
                    deep_ctx,
                    mem_roll,
                    curr_seq,
                    left_seq,
                    right_seq,
                    prefix_mean_seq,
                    suffix_mean_seq,
                    mem_pred.transpose(1, 2),
                    deep_pred.transpose(1, 2),
                ],
                dim=-1,
            )
            ws = int(self.prefix_bridge_window_start)
            we = int(self.prefix_bridge_window_end)
            prefix_tokens = self.prefix_bridge_proj(prefix_in[:, ws:we, :])
            prefix_hidden, _ = self.prefix_bridge_rnn(prefix_tokens)
            prefix_hidden = self.prefix_bridge_norm(prefix_hidden)
            prefix_res_local = self.prefix_bridge_head(prefix_hidden)
            prefix_gate_local = self.prefix_bridge_gate(prefix_hidden)
            prefix_bridge_res = torch.zeros_like(index_pred)
            prefix_bridge_gate = torch.zeros_like(index_pred)
            prefix_bridge_res[:, :, ws:we] = prefix_res_local.transpose(1, 2)
            prefix_bridge_gate[:, :, ws:we] = prefix_gate_local.transpose(1, 2)
            prefix_bridge_term = (
                torch.sigmoid(self.prefix_bridge_scale)
                * prefix_bridge_res
                * prefix_bridge_gate
                * self.prefix_bridge_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + prefix_bridge_term

        hole_interp_res = None
        if self.hole_interp_enabled:
            interp_scale = torch.sigmoid(self.hole_interp_scale).to(index_pred.dtype).view(1, -1)
            hole_interp_res = torch.zeros_like(index_pred)
            for t in range(int(self.hole_interp_window_start), int(self.hole_interp_window_end)):
                curr_seq = index_pred.transpose(1, 2)
                interp_terms = []
                interp_weights = []
                used_idx = set()
                for i in range(int(self.hole_interp_offsets.numel())):
                    off = int(self.hole_interp_offsets[i].item())
                    idx = min(max(t + off, 0), self.pred_time - 1)
                    if idx == t or idx in used_idx:
                        continue
                    used_idx.add(idx)
                    interp_terms.append(curr_seq[:, idx, :])
                    interp_weights.append(float(self.hole_interp_weights[i].item()))
                if not interp_terms:
                    continue
                neigh_stack = torch.stack(interp_terms, dim=1)
                weight_tensor = torch.tensor(interp_weights, device=index_pred.device, dtype=index_pred.dtype)
                weight_tensor = weight_tensor / weight_tensor.sum().clamp_min(1.0e-6)
                interp_target = (neigh_stack * weight_tensor.view(1, -1, 1)).sum(dim=1)
                interp_delta = interp_target - curr_seq[:, t, :]
                step_res = interp_scale * interp_delta
                hole_interp_res[:, :, t] = step_res
                index_pred[:, :, t] = index_pred[:, :, t] + step_res

        hole_patch_res = None
        hole_patch_gate = None
        if self.hole_patch_enabled:
            curr_seq = index_pred.transpose(1, 2)
            prefix_denom = torch.arange(
                1, self.pred_time + 1, device=index_pred.device, dtype=index_pred.dtype
            ).view(1, self.pred_time, 1)
            suffix_denom = torch.arange(
                self.pred_time, 0, -1, device=index_pred.device, dtype=index_pred.dtype
            ).view(1, self.pred_time, 1)
            prefix_mean_seq = torch.cumsum(curr_seq, dim=1) / prefix_denom
            suffix_mean_seq = torch.flip(torch.cumsum(torch.flip(curr_seq, dims=[1]), dim=1), dims=[1]) / suffix_denom
            t = int(self.hole_patch_index)
            ctx = int(self.hole_patch_context)
            ws = max(0, t - ctx)
            we = min(self.pred_time, t + ctx + 1)
            left_idx = max(t - 1, 0)
            right_idx = min(t + 1, self.pred_time - 1)
            far_left_idx = max(t - ctx, 0)
            far_right_idx = min(t + ctx, self.pred_time - 1)
            local_mean = curr_seq[:, ws:we, :].mean(dim=1)
            hole_feat = torch.cat(
                [
                    deep_ctx[:, t, :],
                    mem_roll[:, t, :],
                    curr_seq[:, t, :],
                    curr_seq[:, left_idx, :],
                    curr_seq[:, right_idx, :],
                    curr_seq[:, far_left_idx, :],
                    curr_seq[:, far_right_idx, :],
                    local_mean,
                    prefix_mean_seq[:, t, :],
                    suffix_mean_seq[:, t, :],
                    mem_pred[:, :, t],
                    deep_pred[:, :, t],
                ],
                dim=-1,
            )
            hole_hidden = self.hole_patch_proj(hole_feat)
            hole_res_local = self.hole_patch_head(hole_hidden)
            hole_gate_local = self.hole_patch_gate(hole_hidden)
            hole_patch_res = torch.zeros_like(index_pred)
            hole_patch_gate = torch.zeros_like(index_pred)
            hole_patch_res[:, :, t] = hole_res_local
            hole_patch_gate[:, :, t] = hole_gate_local
            hole_patch_term = (
                torch.sigmoid(self.hole_patch_scale)
                * hole_patch_res
                * hole_patch_gate
                * self.hole_patch_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + hole_patch_term

        frontier_res = None
        frontier_gate = None
        if self.frontier_refiner_enabled:
            h = torch.zeros(
                bsz,
                int(self.frontier_cell.hidden_size),
                device=index_pred.device,
                dtype=index_pred.dtype,
            )
            prev = index_pred[:, :, 0]
            prefix_mean = prev
            frontier_res_steps = []
            frontier_gate_steps = []
            for t in range(self.pred_time):
                curr = index_pred[:, :, t]
                step_in = torch.cat([deep_ctx[:, t, :], mem_roll[:, t, :], curr, prev, prefix_mean], dim=-1)
                h = self.frontier_cell(self.frontier_in_proj(step_in), h)
                h = self.frontier_norm(h)
                step_res = self.frontier_head(h)
                step_gate = self.frontier_gate(h)
                frontier_res_steps.append(step_res)
                frontier_gate_steps.append(step_gate)
                step_term = torch.sigmoid(self.frontier_refiner_scale) * step_res * step_gate
                masked_term = step_term * self.frontier_refiner_mask[:, :, t].to(index_pred.dtype)
                curr_out = curr + masked_term
                prev = curr_out.detach() if self.frontier_refiner_detach_prev else curr_out
                prefix_mean = prefix_mean + (curr_out - prefix_mean) / float(t + 2)

            frontier_res = torch.stack(frontier_res_steps, dim=1).transpose(1, 2)
            frontier_gate = torch.stack(frontier_gate_steps, dim=1).transpose(1, 2)
            frontier_term = (
                torch.sigmoid(self.frontier_refiner_scale)
                * frontier_res
                * frontier_gate
                * self.frontier_refiner_mask.to(index_pred.dtype)
            )
            index_pred = index_pred + frontier_term

        return {
            "index_pred": index_pred,
            "memory_index_pred": mem_pred,
            "deep_index_pred": deep_pred,
            "gate": gate,
            "memory_feature_pred": mem_feat_pred,
            "long_residual": long_res,
            "long_gate": long_gate,
            "barrier_residual": barrier_res,
            "barrier_gate": barrier_gate,
            "refiner_residual": refiner_res,
            "refiner_gate": refiner_gate,
            "ms_refiner_residual": ms_res,
            "ms_refiner_gate": ms_gate,
            "rollout_refiner_residual": rollout_res,
            "rollout_refiner_gate": rollout_gate,
            "tail_booster_residual": tail_res,
            "tail_booster_gate": tail_gate,
            "barrier_booster_residual": barrier_booster_res,
            "barrier_booster_gate": barrier_booster_gate,
            "barrier_bridge_residual": barrier_bridge_res,
            "barrier_bridge_gate": barrier_bridge_gate,
            "prefix_bridge_residual": prefix_bridge_res,
            "prefix_bridge_gate": prefix_bridge_gate,
            "hole_interp_residual": hole_interp_res,
            "hole_patch_residual": hole_patch_res,
            "hole_patch_gate": hole_patch_gate,
            "frontier_refiner_residual": frontier_res,
            "frontier_refiner_gate": frontier_gate,
        }


if __name__ == "__main__":
    model = ENSOX(in_channels=9, memory_in_dim=3, obs_time=12, pred_time=24)
    x = torch.randn(2, 12, 9, 120, 180)
    m = torch.randn(2, 12, 3)
    mon = torch.tensor([0, 5], dtype=torch.long)
    out = model(x, m, mon)
    print(out["index_pred"].shape, out["memory_feature_pred"].shape)
