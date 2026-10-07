# RTX 4090 服务器复现操作全流程指南

本文档汇总了在 **RTX 4090 服务器（CUDA 12.0 / Linux）** 上复现本论文各阶段实验的具体命令与说明。

---

## 一、一键脚本与文件说明

当前目录下提供了以下辅助脚本：

1. **`setup_server_4090.sh`**：一键环境配置脚本（支持 `cu118`）。
2. **`run_experiments_4090.sh`**：各实验阶段快捷启动脚本。
3. **`requirements-4090.txt`**：已解决依赖冲突（严格锁定 `numpy==1.26.4` 与 `ml-dtypes<=0.5.1`）的依赖清单。

---

## 二、阶段概览与执行命令

### 阶段 0：环境准备（仅需执行一次）

如果您尚未配置好环境，或者刚才遇到 `numpy`/`scipy` 冲突，在项目根目录下执行：

```bash
# 激活环境
conda activate hardlabel

# 确保无依赖冲突的安装
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements-4090.txt
```

---

### 阶段 1：冒烟测试（Smoke Test）

验证纯黑盒 Oracle 查询和模型加载：

```bash
# 方式 A：使用一键启动器
bash run_experiments_4090.sh smoke

# 方式 B：直接执行 Python 命令
python -c "
import numpy as np, os, sys
sys.path.insert(0, 'full_end_to_end')
import oracle
print('Oracle 标签测试:', oracle.label(np.zeros(32)))
print('累计查询次数:', oracle.query_count())

import tensorflow as tf, keras
from pathlib import Path
for p in Path('data').glob('*.keras'):
    m = tf.keras.models.load_model(p, compile=False)
    print(f'成功加载 {p.name}: 输入维度 {m.input_shape}')
"
```

---

### 阶段 2：完整端到端黑盒网络提取（论文最核心结果）

在完全不访问真实权重的条件下，仅靠硬标签黑盒漫步，完整偷取并还原整个神经网络的参数：

```bash
# 方式 A：使用一键启动器（指定 32 进程）
bash run_experiments_4090.sh end-to-end 32

# 方式 B：手动进入目录执行
cd full_end_to_end
python run.py 32
```

- **说明**：
  - 参数 `32` 为并行 Worker 数（建议根据服务器 CPU 核心数设为 16 或 32）。
  - 总共执行约 4,000 万次 Oracle 查询。
  - 完成后会自动调用 `validate.py` 对比真值，输出每层相对误差及 100% 标签一致率。
  - 输出保存在 `full_end_to_end/out/net.npz` 和 `full_end_to_end/out/stats.json`。

---

### 阶段 3：黑盒符号恢复实验（Unitary 网络）

在已知未定符号特征的前提下，统计恢复各层神经元的激活正负号：

```bash
# 方式 A：使用一键启动器（8 进程并行）
bash run_experiments_4090.sh blackbox 8

# 方式 B：手动进入目录执行
cd sign_recovery
python sign_recovery_blackbox.py --model unitary_32_32x3_10_float64 --layer 1-3 --neuron 0-7 --j 8
python create_tables.py
```

---

### 阶段 4：Leaky ReLU 快速符号恢复

针对 Leaky ReLU 架构的网络进行高效符号恢复：

```bash
# 方式 A：使用一键启动器
bash run_experiments_4090.sh leaky 8

# 方式 B：手动进入目录执行
cd sign_recovery
python sign_recovery_leaky.py --model unitary_leaky_32_32x3_10_float64 --j 8
```

---

### 阶段 5：CIFAR-10 白盒符号恢复（针对深层网络）

需要 CIFAR-10 双点数据（若未下载，可跳过或拉取预计算仓库）：

```bash
# 方式 A：使用一键启动器
bash run_experiments_4090.sh whitebox 10

# 方式 B：手动进入目录执行
cd sign_recovery
python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 1 --neuron 0-15 --j 10
python create_tables.py
```

