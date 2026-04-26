import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset


def runmean(data: np.ndarray, n_run: int = 3) -> np.ndarray:
    ll = data.shape[0]
    data_run = np.zeros([ll], dtype=np.float32)
    for i in range(ll):
        if i < (n_run - 1):
            data_run[i] = np.nanmean(data[0 : i + 1])
        else:
            data_run[i] = np.nanmean(data[i - n_run + 1 : i + 1])
    return data_run


def _list_sorted_files(path: str) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError("Variable directory not found: {}".format(path))
    files = os.listdir(path)
    files.sort()
    return files


def _match_model_file(file_name: str, model: str) -> bool:
    parts = file_name.split("_")
    if len(parts) < 2:
        return False
    model_name, mode_name = parts[0], parts[1]
    if model_name != model:
        return False
    if model_name in ("GODAS", "ORAS5"):
        return True
    return mode_name == "ssp370"


def load_index(root: str, predictands: List[str], in_models: List[str]):
    uni = []
    last_valid_raw = {}
    if not predictands:
        return uni, last_valid_raw
    for model in in_models:
        for predictand in predictands:
            file_list = _list_sorted_files(os.path.join(root, predictand))
            var_sep = []
            for file_name in file_list:
                if not _match_model_file(file_name, model):
                    continue
                file_path = os.path.join(root, predictand, file_name)
                with np.load(file_path) as npz:
                    raw_model_data = npz["data"].astype(np.float32)
                nonzero = np.where(np.abs(raw_model_data) > 1.0e-8)[0]
                if nonzero.size > 0:
                    last_valid_raw[model] = max(int(nonzero[-1]), int(last_valid_raw.get(model, -1)))
                model_data = runmean(raw_model_data, 3)
                var_sep.append(model_data[None, :])
            if not var_sep:
                raise FileNotFoundError("Index file not found for model={} var={}".format(model, predictand))
            uni.append(var_sep[0])
    return uni, last_valid_raw


def _infer_last_valid_time_index(index_arrays: List[np.ndarray], eps: float = 1.0e-8) -> Optional[int]:
    if not index_arrays:
        return None
    valid_mask = None
    for arr in index_arrays:
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            cur_mask = np.abs(arr) > float(eps)
        else:
            cur_mask = np.any(np.abs(arr) > float(eps), axis=0)
        valid_mask = cur_mask if valid_mask is None else (valid_mask | cur_mask)
    if valid_mask is None or not np.any(valid_mask):
        return None
    return int(np.where(valid_mask)[0][-1])


def union_var(root: str, var_list: List[str], in_models: List[str]) -> List[np.ndarray]:
    data = []
    for model in in_models:
        vdata = []
        for var in var_list:
            load_type = "data"
            var_name = var
            if var.endswith("_mm"):
                load_type = "mean_map"
                var_name = var[:-3]
            file_list = _list_sorted_files(os.path.join(root, var_name))
            var_sep = []
            for file_name in file_list:
                if not _match_model_file(file_name, model):
                    continue
                file_path = os.path.join(root, var_name, file_name)
                with np.load(file_path) as npz:
                    if load_type not in npz:
                        raise KeyError("Key {} missing in {}".format(load_type, file_path))
                    model_data = npz[load_type].astype(np.float32)
                model_data = np.nan_to_num(model_data)
                model_data[np.abs(model_data) > 999] = 0
                var_sep.append(model_data[:, None])
            if not var_sep:
                raise FileNotFoundError("Predictor file not found for model={} var={}".format(model, var_name))
            vdata.append(var_sep[0])
        if not vdata:
            continue
        vdata = np.concatenate(vdata, axis=1)
        data.append(vdata)
        print("Loading model: {} shape={}".format(model, vdata.shape))
    return data


class ENSOMemoryDataset(Dataset):
    GODAS_CHANNEL_DICT = {
        "tos": "pottmp_5",
        "thetao_5": "pottmp_5",
        "thetao_20": "pottmp_20",
        "thetao_40": "pottmp_40",
        "thetao_60": "pottmp_60",
        "thetao_90": "pottmp_90",
        "thetao_120": "pottmp_120",
        "thetao_150": "pottmp_150",
        "uo_5": "ucur_5",
        "vo_5": "vcur_5",
        "wo_10": "dzdt_10",
        "tauuo": "uflx",
        "tauvo": "vflx",
        "zos": "sshg",
        "vor": "vor",
        "hfds": "thflx",
        "thetao_wmean": "pottmp_wmean",
        "sltfl": "sltfl",
        "psl": "msl",
        "mlotst": "dbss_obml",
        "sos": "salt",
        "tauu": "tau_x",
        "tauv": "tau_y",
        "mld_diff": "mld_diff",
    }
    ORAS5_CHANNEL_DICT = {
        "tos": "votemper_5",
        "thetao_5": "votemper_5",
        "thetao_20": "votemper_20",
        "thetao_40": "votemper_40",
        "thetao_60": "votemper_60",
        "thetao_90": "votemper_90",
        "thetao_120": "votemper_120",
        "thetao_150": "votemper_150",
        "uo_5": "vozocrtx_5",
        "vo_5": "vomecrty_5",
        "wo_10": "dzdt_10",
        "tauuo": "sozotaux",
        "tauvo": "sometauy",
        "zos": "sshg",
        "vor": "vor",
        "hfds": "thflx",
        "thetao_wmean": "votemper_wmean",
        "sltfl": "sltfl",
        "psl": "msl",
        "mlotst": "somxl030",
        "sos": "sosaline",
        "tauu": "tau_x",
        "tauv": "tau_y",
        "mld_diff": "mld_diff",
    }

    def __init__(
        self,
        in_channels: List[str],
        out_channels: List[str],
        in_models: List[str],
        time_range: List[int],
        obs_time: int,
        pred_type: str,
        pred_time: int,
        input_region: List[int],
        target_region: List[int],
        output_type: str = "index",
        transform=None,
        data_root: Optional[str] = None,
        memory_features: Optional[List[Dict]] = None,
        memory_stats: Optional[Dict[str, np.ndarray]] = None,
    ):
        self.obs_time = obs_time
        self.pred_time = pred_time
        self.pred_type = pred_type
        self.output_type = output_type
        self.input_region = input_region
        self.target_region = target_region
        self.transform = transform
        self.source_models = list(in_models)
        self.requested_time_range = [int(time_range[0]), int(time_range[1])]

        data_root = data_root or os.environ.get(
            "ENSOX_DATA_ROOT",
            os.environ.get("CTEFNET_DATA_ROOT", "/mnt/disk1/ctefnet_data"),
        )
        data_root = os.path.abspath(data_root)

        channel_mapper = None
        if in_models == ["GODAS"]:
            file_path = os.path.join(data_root, "ReanalysisVar", "GODAS")
            full_time = pd.date_range(start="19800101", end="20231201", freq="MS")
            channel_mapper = self.GODAS_CHANNEL_DICT
        elif in_models == ["ORAS5"]:
            oras5_reanalysis_path = os.path.join(data_root, "ReanalysisVar", "ORAS5")
            oras5_path = os.path.join(data_root, "ORAS5")
            file_path = oras5_reanalysis_path if os.path.exists(oras5_reanalysis_path) else oras5_path
            full_time = pd.date_range(start="19580101", end="20231201", freq="MS")
            channel_mapper = self.ORAS5_CHANNEL_DICT
        else:
            raise ValueError("This ENSO-X release supports GODAS and ORAS5 reanalysis inputs only.")

        if not os.path.exists(file_path):
            raise FileNotFoundError("Data path not found: {}".format(file_path))

        original_full_end = full_time[-1]
        original_full_start = full_time[0]
        clamp_applied = False
        last_valid_time = None

        self.in_channels = self._map_channels(in_channels, channel_mapper)
        memory_features_mapped = self._map_memory_features(memory_features, channel_mapper)

        raw_data = union_var(file_path, self.in_channels, in_models)
        raw_index, raw_index_last_valid = load_index(file_path, out_channels, in_models)

        # Some reanalysis label files are padded with all-zero tails beyond the truly available period.
        # Clamp the usable time axis to the last non-zero predictand month so validation windows remain honest.
        if len(in_models) == 1 and in_models[0] in ("GODAS", "ORAS5"):
            last_valid_idx = raw_index_last_valid.get(in_models[0])
            if last_valid_idx is None:
                last_valid_idx = _infer_last_valid_time_index(raw_index)
            if last_valid_idx is not None and last_valid_idx < (len(full_time) - 1):
                last_valid_time = full_time[last_valid_idx]
                clamp_applied = True
                print(
                    "[Data] clamp {} usable label period to {} (requested full end {})".format(
                        in_models[0],
                        str(last_valid_time.date()),
                        str(full_time[-1].date()),
                    )
                )
                full_time = full_time[: last_valid_idx + 1]
                raw_data = [arr[: last_valid_idx + 1] for arr in raw_data]
                raw_index = [arr[:, : last_valid_idx + 1] for arr in raw_index]

        needed_time = (full_time.year >= time_range[0]) & (full_time.year <= time_range[1])
        if not np.any(needed_time):
            raise ValueError(
                "Requested time range {}-{} yields no usable samples for models {}.".format(
                    int(time_range[0]),
                    int(time_range[1]),
                    in_models,
                )
            )
        selected_months = full_time[needed_time].month.to_numpy(dtype=np.int64)
        selected_time = full_time[needed_time]

        self.data = []
        self.index = []
        self.memory = []
        print("preprocessing data ...")
        for i, model_data in enumerate(raw_data):
            selected_model_data = model_data[needed_time]
            self.data.append(selected_model_data)
            self.index.append(raw_index[i][:, needed_time])
            mem = self._extract_memory_features(selected_model_data, memory_features_mapped)
            self.memory.append(mem)
        print("done")

        self.months = selected_months
        self.memory_dim = self.memory[0].shape[1]
        if memory_stats is None:
            stacked = np.concatenate(self.memory, axis=0)
            mem_mean = stacked.mean(axis=0, keepdims=True).astype(np.float32)
            mem_std = stacked.std(axis=0, keepdims=True).astype(np.float32)
            mem_std[mem_std < 1e-6] = 1.0
            self.memory_stats = {"mean": mem_mean, "std": mem_std}
        else:
            self.memory_stats = memory_stats

        self.memory = [
            ((m - self.memory_stats["mean"]) / self.memory_stats["std"]).astype(np.float32) for m in self.memory
        ]

        self.num_model = len(self.data)
        self.num_mon = self.data[0].shape[0]
        self.model_len = self.num_mon - self.obs_time - self.pred_time + 1
        if self.model_len <= 0:
            raise ValueError("Invalid sequence setup: num_mon={} obs={} pred={}".format(self.num_mon, obs_time, pred_time))

        self.meta = {
            "source_models": list(self.source_models),
            "requested_time_range_years": [int(time_range[0]), int(time_range[1])],
            "requested_full_time_start": str(original_full_start.date()),
            "requested_full_time_end": str(original_full_end.date()),
            "effective_full_time_start": str(full_time[0].date()),
            "effective_full_time_end": str(full_time[-1].date()),
            "label_clamp_applied": bool(clamp_applied),
            "label_last_valid_time": None if last_valid_time is None else str(last_valid_time.date()),
            "selected_time_start": str(selected_time[0].date()),
            "selected_time_end": str(selected_time[-1].date()),
            "selected_month_count": int(selected_time.size),
            "num_model": int(self.num_model),
            "num_mon": int(self.num_mon),
            "model_len": int(self.model_len),
            "obs_time": int(self.obs_time),
            "pred_time": int(self.pred_time),
        }

    @staticmethod
    def _map_channels(channels: List[str], mapper: Optional[Dict[str, str]]) -> List[str]:
        if mapper is None:
            return channels
        mapped = []
        for ch in channels:
            base = ch.rstrip("_m")
            suffix = "_m" if ch.endswith("_m") else ""
            if base not in mapper:
                raise KeyError("Channel {} not in mapper".format(ch))
            mapped.append(mapper[base] + suffix)
        return mapped

    @staticmethod
    def _map_memory_features(memory_features: Optional[List[Dict]], mapper: Optional[Dict[str, str]]) -> Optional[List[Dict]]:
        if not memory_features:
            return None
        if mapper is None:
            return memory_features
        mapped_features = []
        for item in memory_features:
            obj = dict(item)
            var = obj.get("var")
            if var is None:
                mapped_features.append(obj)
                continue
            if var in mapper:
                obj["var"] = mapper[var]
            var2 = obj.get("var2")
            if var2 in mapper:
                obj["var2"] = mapper[var2]
            mapped_features.append(obj)
        return mapped_features

    def _default_memory_specs(self, c: int) -> List[Dict]:
        channel_set = set(self.in_channels)
        specs = []
        wwv_var = None
        for candidate in ("thetao_wmean", "thetao_5", "tos", "pottmp_wmean", "votemper_wmean"):
            if candidate in channel_set:
                wwv_var = candidate
                break
        if wwv_var is None:
            wwv_var = self.in_channels[0]
        specs.append({"name": "wwv_proxy", "var": wwv_var, "region": [54, 66, 80, 130], "op": "mean"})

        if "tauu" in channel_set or "tau_x" in channel_set:
            tau_var = "tauu" if "tauu" in channel_set else "tau_x"
            specs.append({"name": "trade_wind", "var": tau_var, "region": [55, 65, 70, 130], "op": "mean"})

        sst_var = None
        for candidate in ("thetao_5", "tos", "pottmp_5", "votemper_5"):
            if candidate in channel_set:
                sst_var = candidate
                break
        if sst_var is not None:
            specs.append({"name": "sst_basin_mean", "var": sst_var, "region": [0, 120, 0, 180], "op": "mean"})
        else:
            specs.append({"name": "feature0_mean", "var": self.in_channels[0], "region": [0, 120, 0, 180], "op": "mean"})
        return specs

    def _extract_memory_features(self, fields: np.ndarray, memory_features: Optional[List[Dict]]) -> np.ndarray:
        t, c, h, w = fields.shape
        specs = memory_features or self._default_memory_specs(c)
        channel_to_idx = {name: idx for idx, name in enumerate(self.in_channels)}
        out = []

        def _region_patch(var_name: str, region):
            if var_name not in channel_to_idx:
                raise KeyError("Memory feature var {} not found in in_channels".format(var_name))
            idx = channel_to_idx[var_name]
            r = region or [0, h, 0, w]
            lat0, lat1, lon0, lon1 = int(r[0]), int(r[1]), int(r[2]), int(r[3])
            lat0 = max(0, min(lat0, h - 1))
            lat1 = max(lat0 + 1, min(lat1, h))
            lon0 = lon0 % w
            lon1 = lon1 % w
            if lon1 <= lon0:
                patch_left = fields[:, idx, lat0:lat1, lon0:w]
                patch_right = fields[:, idx, lat0:lat1, 0:max(lon1, 1)]
                patch = np.concatenate([patch_left, patch_right], axis=2)
            else:
                patch = fields[:, idx, lat0:lat1, lon0:lon1]
            if patch.size == 0:
                patch = fields[:, idx, :, :]
            return patch

        for spec in specs:
            var = spec["var"]
            op = spec.get("op", "mean")
            if op == "mean":
                patch = _region_patch(var, spec.get("region", [0, h, 0, w]))
                series = np.nanmean(patch, axis=(1, 2))
            elif op == "sum":
                patch = _region_patch(var, spec.get("region", [0, h, 0, w]))
                series = np.nansum(patch, axis=(1, 2))
            elif op == "diff":
                patch1 = _region_patch(var, spec.get("region", [0, h, 0, w]))
                var2 = spec.get("var2", var)
                patch2 = _region_patch(var2, spec.get("region2", [0, h, 0, w]))
                series = np.nanmean(patch1, axis=(1, 2)) - np.nanmean(patch2, axis=(1, 2))
            else:
                raise ValueError("Unsupported memory op: {}".format(op))
            out.append(series.astype(np.float32))
        return np.stack(out, axis=1)

    def __len__(self):
        return self.num_model * self.model_len

    def __getitem__(self, idx):
        model_idx = int(idx / self.model_len)
        month = int(idx % self.model_len)

        x = torch.tensor(
            self.data[model_idx][
                month : month + self.obs_time,
                :,
                self.input_region[0] : self.input_region[1],
                self.input_region[2] : self.input_region[3],
            ],
            dtype=torch.float32,
        )
        y_field = torch.tensor(
            self.data[model_idx][
                month + self.obs_time : month + self.obs_time + self.pred_time,
                :,
                self.input_region[0] : self.input_region[1],
                self.input_region[2] : self.input_region[3],
            ],
            dtype=torch.float32,
        )
        if self.transform is not None:
            x = self.transform(x)

        if self.pred_type == "series":
            y_index = torch.tensor(
                self.index[model_idx][:, month + self.obs_time : month + self.obs_time + self.pred_time],
                dtype=torch.float32,
            )
        else:
            y_index = torch.tensor(
                self.index[model_idx][:, month + self.obs_time + self.pred_time, None],
                dtype=torch.float32,
            )
        m_hist = torch.tensor(self.memory[model_idx][month : month + self.obs_time], dtype=torch.float32)
        m_future = torch.tensor(
            self.memory[model_idx][month + self.obs_time : month + self.obs_time + self.pred_time], dtype=torch.float32
        )
        init_month = torch.tensor(int(self.months[month] - 1), dtype=torch.long)
        return x, y_field, y_index, m_hist, m_future, init_month

    def metadata(self) -> Dict[str, object]:
        return dict(self.meta)


def build_dataloaders(cfg, transforms=None):
    data_params = cfg.get("data", {})
    num_workers = int(data_params.get("num_workers", 0 if os.name == "nt" else 8))
    pin_memory = torch.cuda.is_available()

    train_dataset = ENSOMemoryDataset(
        data_params.get("predictor"),
        data_params.get("predictand"),
        data_params.get("train_models"),
        data_params.get("train_period"),
        data_params.get("obs_time"),
        data_params.get("pred_type"),
        data_params.get("pred_time"),
        data_params.get("input_region"),
        data_params.get("target_region"),
        output_type="index",
        transform=transforms,
        data_root=data_params.get("data_root"),
        memory_features=data_params.get("memory_features"),
    )
    train_dataset_for_loader = train_dataset
    source_replay = data_params.get("source_replay")
    if isinstance(source_replay, dict) and bool(source_replay.get("enabled", False)):
        replay_models = source_replay.get("models")
        replay_period = source_replay.get("period")
        replay_repeat = int(source_replay.get("repeat", 1))
        if replay_models and replay_period:
            replay_dataset = ENSOMemoryDataset(
                data_params.get("predictor"),
                data_params.get("predictand"),
                replay_models,
                replay_period,
                data_params.get("obs_time"),
                data_params.get("pred_type"),
                data_params.get("pred_time"),
                data_params.get("input_region"),
                data_params.get("target_region"),
                output_type="index",
                transform=transforms,
                data_root=data_params.get("data_root"),
                memory_features=data_params.get("memory_features"),
                memory_stats=train_dataset.memory_stats,
            )
            datasets = [train_dataset]
            for _ in range(max(replay_repeat, 1)):
                datasets.append(replay_dataset)
            train_dataset_for_loader = ConcatDataset(datasets)
            train_dataset_for_loader.memory_dim = train_dataset.memory_dim
            print(
                "[Data] source replay enabled: models={} period={} repeat={} target_len={} replay_len={} total_len={}".format(
                    replay_models,
                    replay_period,
                    max(replay_repeat, 1),
                    len(train_dataset),
                    len(replay_dataset),
                    len(train_dataset_for_loader),
                )
            )
        else:
            print("[Data] source replay skipped: models or period missing in source_replay config")

    valid_dataset = ENSOMemoryDataset(
        data_params.get("predictor"),
        data_params.get("predictand"),
        data_params.get("valid_models"),
        data_params.get("valid_period"),
        data_params.get("obs_time"),
        data_params.get("pred_type"),
        data_params.get("pred_time"),
        data_params.get("input_region"),
        data_params.get("target_region"),
        output_type="index",
        transform=None,
        data_root=data_params.get("data_root"),
        memory_features=data_params.get("memory_features"),
        memory_stats=train_dataset.memory_stats,
    )

    train_loader = DataLoader(
        train_dataset_for_loader,
        batch_size=int(data_params.get("train_batch_size", 8)),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=int(data_params.get("valid_batch_size", 16)),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, valid_loader, train_dataset
