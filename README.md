# ENSO-X

ENSO-X 是一个面向 ENSO 长提前期预测的独立设计模型。模型以过去 12 个月的海洋和大气场作为输入，预测未来 24 个月的 Niño3.4 指数演变。

英文版说明见 `README.en.md`。

## 模型概述

ENSO-X 采用混合时空预测框架，主要由三部分组成：

- 场编码分支  
  使用 3D CNN 与 Transformer 提取大尺度海气场的时空特征。
- 记忆分支  
  结合季节状态空间与物理启发的双记忆机制，用于表征再充电过程、风场强迫和持续性效应。
- lead 修复分支  
  通过局部桥接、插值和 patch 模块，修复困难 lead 窗口并提升长 lead 预测稳定性。

## 数据设置

当前发布版 ENSO-X 的训练与评估设置如下：

- 训练集：`GODAS 1980-2014`
- replay 增强：`ORAS5 1980-2014`
- 主验证集：`GODAS 2015-2021`
- 外部泛化测试：`CMIP6 2015-2023`
- 长时段外推测试：`CMIP6 2015-2100`

模型输入使用 9 个海气变量：

- `thetao_5`
- `thetao_wmean`
- `tauu`
- `tauv`
- `uo_5`
- `vo_5`
- `psl`
- `mlotst`
- `sos`

记忆分支使用 3 个派生记忆特征：

- `wwv_proxy`
- `trade_wind`
- `sst_basin_mean`

## 主要效果

当前发布的 ENSO-X checkpoint 在主设置下已经实现稳定的 24 个月预测能力：

- `24/24` 个 lead 月的 `corr > 0.5`
- 最小月相关系数为 `0.5089`
- 原本最困难的 `lead 9-11` 屏障段也已经整体修复到 `0.5` 以上

在外部 `CMIP6` 评估上：

- `CMIP6 2015-2023`：前 `24` 个月保持 `corr > 0.5`，并且至少到 `48` 个月仍保持 `corr > 0.2`
- `CMIP6 2015-2100`：当前 zero-shot 外推测试中，至少到 `48` 个月仍保持 `corr > 0.5`

当前发布版的实用结论可以概括为：

- 有效预测能力：`24` 个月
- 保守外推能力：`48` 个月

## 仓库结构

- `train.py`：训练入口
- `src/ensox/`：ENSO-X 核心代码
- `configs/enso_x_24_final.yaml`：最终可复现训练配置
- `preprocess/`：当前发布版实际使用的预处理脚本
- `scripts/evaluate_limit_enso_x.py`：长时段外推评估脚本
- `scripts/run_train_enso_x.sh`：训练启动脚本
- `scripts/run_limit_eval_enso_x.sh`：外推评估启动脚本
- `checkpoints/`：checkpoint 清单与说明
- `results/`：发布版结果摘要

## 环境

仓库中保留了三类环境说明：

- `requirements.txt`：最小运行依赖
- `requirements-preprocess.txt`：预处理所需依赖
- `environment.yml`：服务器发布环境导出的完整环境文件

示例：

```bash
conda env create -f environment.yml
conda activate enso_x
```

## 数据预处理

ENSO-X 期望的处理后数据目录结构如下：

```text
data/ctefnet_data/
  CMIP6var/
  ReanalysisVar/
    GODAS/
    ORAS5/
```

`preprocess/` 目录中保留的是与当前发布版数据产物相匹配的脚本：

- `preprocess_cmip6_to_ensox.py`
- `preprocess_godas_to_ensox.py`
- `preprocess_oras5_to_ensox.py`

`preprocess/README.md` 中已经补充了：

- 原始数据目录布局
- 哪些变量采用特殊处理
- 预处理示例命令
- `ERA5 msl`、加权平均变量和 `nino34` 派生方式的说明

## Checkpoint

当前发布版保留了 3 组核心 checkpoint：

- `final_24_complete`：最终发布版主 checkpoint
- `seed_24_run`：复现最终 24 个月结果时使用的直接 seed
- `lead23_baseline`：更早期的高 lead baseline，用于对比

具体名称、作用、服务器路径和关键指标见：

- `checkpoints/MANIFEST.json`
- `checkpoints/README.md`

在服务器发布目录中，`checkpoints/` 下也保留了相同别名的软链接。

## 如何复现

### 1. 设置路径

如果你的处理后数据不在默认目录 `./data/ctefnet_data` 下，可以先设置：

```bash
export ENSOX_DATA_ROOT=/path/to/ctefnet_data
export ENSOX_INIT_CKPT=/path/to/seed_24_run/best_frontier.ckpt
export ENSOX_OUTPUT_ROOT=/path/to/save/checkpoints
```

最终配置支持 `${VAR}` 和 `${VAR:-default}` 形式的环境变量展开。

### 2. 复现最终 24 个月训练

```bash
bash scripts/run_train_enso_x.sh
```

等价命令：

```bash
python train.py --config ./configs/enso_x_24_final.yaml
```

### 3. 评估最终 checkpoint

```bash
bash scripts/run_limit_eval_enso_x.sh ./checkpoints/final_24_complete/best.ckpt ./results/enso_x_limit_eval.json
```

等价命令：

```bash
python scripts/evaluate_limit_enso_x.py \
  --base-config ./configs/enso_x_24_final.yaml \
  --ckpt ./checkpoints/final_24_complete/best.ckpt \
  --data-root "${ENSOX_DATA_ROOT:-./data/ctefnet_data}" \
  --output-json ./results/enso_x_limit_eval.json
```

## 发布结果文件

- `results/enso_x_summary.json`：主结果摘要
- `results/enso_x_generalization_20260425.json`：外部泛化结果
- `results/enso_x_limit_eval_20260425.json`：外推评估摘要
