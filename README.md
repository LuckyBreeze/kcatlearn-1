# kcatlearn-1

模型二代码归档：ProtT5 蛋白残基作为 Query，MolT5 序列 token 与一个 MACCS token 作为 Key/Value，预测 log10(kcat)。

本仓库于 2026-10-28 通过 SSH 导出自：

```text
/share/home/u17015/data/GraphKcat/catapro_residue_query_model2_20261027
```

24 个服务器文件保持原始字节；逐文件 SHA256 和大小见 `SOURCE_MANIFEST.json`。本次仅发布代码，没有重新训练、推断或修改实验设置。这是 2026-10-27 的逐批验证 RMSE 选模版本，不包含另一目录中 2026-10-28 的 R² 双选模改动。

## 模型与配置

- 输入：ProtT5 残基特征 `[B,L,1024]`、MolT5 token 特征 `[B,M,768]`、MACCS 指纹 `[B,167]`，以及对应 mask。
- MolT5 token 投影至 1024 维，MACCS 投影成一个 1024 维 token；8 头交叉注意力输出与原蛋白残基逐位置相加。
- masked mean 后使用 `1024 → 256 / BatchNorm / ReLU → 1` 预测头；本次冻结配置的 Dropout 为 0。
- 固定 cluster40 划分，seed 668；Adam、学习率 `1e-5`、weight decay `0.01`、原 RMSE 损失、batch size 8；训练尾部单样本与前批合并。
- 原逐批 valid RMSE 均值选模，最多 150 轮，patience 20、min_delta 0.001。完整实现以 `train.py` 为准。

## 文件入口

| 文件 | 用途 |
| --- | --- |
| `model.py` | 残基与底物交叉注意力及回归头 |
| `prepare.py` | 校验冻结数据与划分，准备 token 查表输入 |
| `extract_tokens.py` | 冻结 ProtT5 / MolT5 提取、缓存及身份校验 |
| `data.py`, `tail_batch.py` | 数据加载、padding、mask 与尾批处理 |
| `train.py`, `author_utils.py` | CUDA 预检、训练、选模锁定、valid 重放和 test |
| `prepare.sbatch`, `protein.sbatch`, `molt5.sbatch`, `train.sbatch` | 原 Slurm 作业脚本 |
| `submit_remote.py`, `snapshot_remote.py` | 提交凭据与状态查询 |
| `cuda_probe.py`, `cuda_probe.sbatch`, `protein_known_gpu.sbatch` | 原 CUDA / 调度诊断文件 |
| `test_*.py` | 原合成检查代码 |
| `checkpoints.json` | 外部预训练模型路径与文件 SHA256，不含权重 |
| `deployment_manifest.json`, `verify_deployment.py` | 原 20 个部署文件的完整性核验 |
| `integrated_validation.json` | 原有 29 项 CPU 合成检查记录，不代表本次重新运行 |

## 外部依赖与现有缺项

这是保留原路径的服务器代码快照。Slurm 脚本仍引用原服务器目录、conda 环境和外部数据，克隆后不能直接在任意机器运行完整训练。

正式运行需要 Linux/Slurm、兼容的 CUDA/PyTorch 环境，以及 NumPy、pandas、SciPy、Transformers、SentencePiece 等依赖。原 conda 环境路径为 `/share/home/u17015/.conda/envs/graphkcat_env`；本导出包没有完整依赖版本锁文件，未猜测或补写版本号。

`prepare.py` 依赖下列外部目录中的五个冻结输入；文件身份由其 `EXPECTED` 字典校验：

```text
/share/home/u17015/data/GraphKcat/catapro_cluster40_six_seed668_20261025/
  inputs/feature_store.json
  inputs/rowspec.json
  inputs/rows.csv
  inputs/feature1959.npy
  assignments.csv
```

还需要 `checkpoints.json` 指定的 ProtT5 和 MolT5 预训练文件。数据、序列/特征缓存、编码器权重、训练 checkpoint、预测结果、日志、提交状态与历史失败目录均未上传。

服务器原目录没有 `example_forward.py`，但 `test_model.py` 导入其中的 `build_review_example` 和 `synthetic_inputs`。因此当前快照不能直接运行完整测试集；本次没有补写该模块。此缺项属于测试辅助文件，正式训练入口不导入它。

## 运行流程与核验

原流程为：准备固定输入 → 分别提取两种 token 缓存 → CUDA 预检 → 从固定初始化训练 → 锁定选模结果 → 完整 valid 重放 → test。正式计算使用 `sbatch`；CPU 准备作业使用 `intel`，GPU 作业使用 `L40`。

迁移或重新运行前，应先核实外部输入、预训练权重、依赖版本以及脚本中的路径。原提交器使用 `submission.json` 防止重复投递，此运行状态文件不包含在本代码归档中；不要把新的克隆副本直接当作原实验的续提入口。

本次发布核验包括：24 个源文件与服务器 SHA256 一致、原部署清单一致、Python 源码语法解析通过。未重新运行训练或模型测试。Git 属性关闭换行转换，以保留这些源码的原始校验值。
