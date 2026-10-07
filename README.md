# Hard-Label DNN Extraction: 全流程实验复现与技术评测

[![Paper](https://img.shields.io/badge/EUROCRYPT_2024-Paper-blue.svg)](https://eprint.iacr.org/2023/1647)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5.1-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-FF6F00.svg?logo=tensorflow&logoColor=white)](https://www.tensorflow.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20(RTX%204090)-success.svg)]()

本项目为顶级密码学学术会议 **EUROCRYPT 2024** 论文 **《Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting》**（Javier R. Chavez-Saab 等著）的**完整复现工程与实验技术报告**。

仓库内包含**端到端模型完全窃取**、**独立黑盒符号恢复**、**CIFAR-10 深度网络白盒符号恢复**以及 **Leaky ReLU 在线符号恢复**全部 4 项核心实验。我们分别在**本地多核 CPU** 与 **远程 Linux 高性能服务器 (NVIDIA RTX 4090)** 上完成了全流程复现与对照评测，并提供了自动化运行脚本、依赖解决补丁以及出版级排版的 Markdown 与 PDF 实验报告。

---

## 📑 核心文档快速导航

所有复现成果均已整理为详尽的技术评估报告（同时提供 Markdown 与 PDF 版本）：

| 文档名称 | 格式 | 核心内容说明 |
| :--- | :--- | :--- |
| **服务器端实验复现报告** | [Markdown](SERVER_EXPERIMENT_RESULTS.md) \| [PDF](SERVER_EXPERIMENT_RESULTS.pdf) | RTX 4090 环境全套实验数据、4000万次查询耗时、误差对比、环境避坑实录 |
| **本地 CPU 实验复现报告** | [Markdown](CPU_EXPERIMENT_RESULTS.md) \| [PDF](CPU_EXPERIMENT_RESULTS.pdf) | 本地多核 CPU (Conda) 复现数据、单机各阶段查询与收敛对比 |
| **实验产物与存储指引** | [Markdown](EXPERIMENT_OUTPUTS_GUIDE.md) \| [PDF](EXPERIMENT_OUTPUTS_GUIDE.pdf) | `net.npz`、`stats.json`、`duals`、`df.pkl` 等底层实验结果文件的存放与结构解析 |
| **复现操作部署指南** | [Markdown](hard-label-dnn-extraction/REPRODUCTION_GUIDE.md) | 从 0 到 1 快速搭建 Conda 虚拟环境及执行各模块实验的操作步骤 |

---

## 🔬 核心实验复现对比总览

### 1. 端到端模型完全窃取 (Full End-to-End Extraction)

- **目标模型**：3 层隐藏层前馈网络，结构为 $[32 \to 32 \to 32 \to 32 \to 10]$，浮点精度 `float64`。
- **攻击条件**：极严格的 Hard-Label 纯黑盒设定（攻击者仅能获得 Top-1 分类预测标签，无法获得概率、梯度或内部激活）。

| 指标维度 | 论文理论预期 / 标杆 | 本地 CPU (Intel 8 进程) | 服务器端 (RTX 4090 / 32 vCPU) |
| :--- | :--- | :--- | :--- |
| **总 Oracle 查询次数** | $\sim 3.8 \times 10^7$ | **38,668,729 次** | **39,338,738 次** |
| **全流程总耗时** | 约数分钟 | **398.6 秒 (~6.6 分钟)** | **309.4 秒 (~5.2 分钟)** |
| **权重最大相对误差 (L0)** | $< 10^{-13}$ | **$3.78 \times 10^{-14}$** | **$1.83 \times 10^{-14}$** |
| **权重最大相对误差 (L1)** | $< 10^{-13}$ | **$2.67 \times 10^{-14}$** | **$4.02 \times 10^{-14}$** |
| **权重最大相对误差 (L2)** | $< 10^{-13}$ | **$9.24 \times 10^{-15}$** | **$1.64 \times 10^{-14}$** |
| **输出头 (Head) 误差** | $< 10^{-13}$ | **$4.01 \times 10^{-15}$** | **$4.42 \times 10^{-15}$** |
| **符号恢复错误数** | 0 | **0 个 (96/96 全部正确)** | **0 个 (96/96 全部正确)** |
| **10万测试集决策一致性** | 100% | **100,000 / 100,000 (100.0%)** | **100,000 / 100,000 (100.0%)** |
| **最终判定结论** | Stolen | **THE NETWORK WAS STOLEN!** | **THE NETWORK WAS STOLEN!** |

---

### 2. 符号恢复实验表现 (Sign Recovery Benchmarks)

1. **Leaky ReLU 快速符号恢复 (`sign_recovery_leaky.py`)**：
   - 目标：3 层网络共 96 个神经元，利用非零负斜率引发的梯度跳变不对称性。
   - 结果：**3 层 96/96 个神经元符号 100% 正确恢复**，耗时仅 ~20 秒。
2. **独立黑盒符号恢复 (`sign_recovery_blackbox.py`)**：
   - 目标：逐层对独立神经元进行决策边界微扰投票检验。
   - 结果：全部抽样神经元以 **100% 置信度恢复真实符号**（`correct: ✔`）。
3. **CIFAR-10 白盒符号恢复 (`sign_recovery_whitebox.py`)**：
   - 目标：针对 CIFAR-10 深度网络 ($3 \times 256 \times 64 \times 10$)，通过激活状态下的位移速度比 $\|v_{\text{ON}}\| / \|v_{\text{OFF}}\|$ 进行假设检验。
   - 结果：抽样 16 个神经元**全部以 95% 置信度实现提前终止收敛**，平均耗时仅 11~14 秒/神经元。

---

## 📁 仓库目录结构

```text
GI/
├── README.md                           # 本仓库主文档
├── SERVER_EXPERIMENT_RESULTS.md        # 【核心报告】服务器端复现评测报告 (Markdown)
├── SERVER_EXPERIMENT_RESULTS.pdf       # 【核心报告】服务器端复现评测报告 (PDF 精排版)
├── CPU_EXPERIMENT_RESULTS.md           # 本地 CPU 运行与评测报告 (Markdown)
├── CPU_EXPERIMENT_RESULTS.pdf          # 本地 CPU 运行与评测报告 (PDF 精排版)
├── EXPERIMENT_OUTPUTS_GUIDE.md         # 实验生成文件与数据存储位置完整索引
├── EXPERIMENT_OUTPUTS_GUIDE.pdf        # 实验产物存储指引 (PDF 精排版)
├── convert_md_to_pdf.py                # 自动化 Markdown 转出版级 PDF 渲染工具 (基于 Edge Headless)
├── out/                                # 从高性能服务器回传的端到端实验核心产物
│   ├── net.npz                         # 最终窃取的完整模型参数权重字典
│   ├── stats.json                      # 各阶段查询量与耗时统计详情
│   ├── duals/                          # 边界漫步采集到的决策双点数据集
│   └── logs/                           # 96 个神经元符号判定的单步日志
└── hard-label-dnn-extraction/          # 论文算法源码与实验模块
    ├── full_end_to_end/                # 实验一：端到端模型窃取源码
    ├── sign_recovery/                  # 实验二 & 三 & 四：黑盒/白盒/Leaky 符号恢复源码
    ├── signature_recovery/             # 经典符号恢复对比实现
    ├── generators/                     # 合成数据与网络生成脚本
    ├── data/                           # 预训练模型 (unitary 及 cifar10 .keras 权重)
    ├── setup_server_4090.sh            # 一键配置 RTX 4090 服务器环境的 Shell 脚本
    ├── run_experiments_4090.sh         # 一键批处理执行全部服务器实验的 Shell 脚本
    ├── requirements-4090.txt           # 针对 PyTorch/TensorFlow 严格锁版本的依赖清单
    └── REPRODUCTION_GUIDE.md           # 复现环境部署全流程手册
```

---

## 🚀 快速上手与复现指南

### 方式 A：Linux / GPU 服务器环境（推荐）

针对配备现代化 NVIDIA 显卡（如 RTX 4090 / A100）及 CUDA 12.x 驱动的 Linux 服务器，可直接使用仓库内提供的一键脚本：

```bash
# 1. 克隆本仓库并进入项目子目录
git clone https://github.com/rrsty/hard-label-dnn-extraction-reproduction.git
cd hard-label-dnn-extraction-reproduction/hard-label-dnn-extraction

# 2. 一键配置环境（解决 NumPy/PyTorch 冲突与 TF 日志抑制）
chmod +x setup_server_4090.sh run_experiments_4090.sh
./setup_server_4090.sh

# 3. 激活虚拟环境
conda activate hldnn

# 4. 一键执行全部核心实验
./run_experiments_4090.sh
```

---

### 方式 B：本地 Windows / CPU 环境

在本地使用 Conda 创建纯 CPU 运行环境：

```powershell
# 1. 创建并激活 Python 3.11 环境
conda create -n hldnn python=3.11 -y
conda activate hldnn

# 2. 安装解决冲突的关键锁版本依赖
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install "numpy==1.26.4" "scipy==1.13.1" "h5py==3.11.0"
pip install "tensorflow==2.20.0" "keras==3.15.1" "ml-dtypes<=0.5.1"
pip install pandas markdown

# 3. 运行端到端模型窃取实验 (指定 8 进程)
cd hard-label-dnn-extraction\full_end_to_end
python run.py 8

# 4. 运行 Leaky ReLU 符号恢复实验
cd ..\sign_recovery
python sign_recovery_leaky.py --model unitary_leaky_32_32x3_10_float64 --j 8

# 5. 运行独立黑盒符号恢复实验并制表
python sign_recovery_blackbox.py --model unitary_32_32x3_10_float64 --layer 1 --neuron 0-7 --j 8
python create_tables.py --model unitary_32_32x3_10_float64
```

---

## 🛠️ 关键技术避坑与兼容性解决方案

在复现过程中，我们攻克了多项因深度学习生态包版本升级带来的兼容性难题，具体方案已内化在项目脚本中：

1. **NumPy 2.x 与 SciPy 1.13 / ml-dtypes 兼容性冲突**：
   - *问题*：最新安装 TensorFlow 2.20 时默认拉取 `ml-dtypes 0.6.0`，其依赖将 NumPy 升级至 `numpy>=2.0.0`，导致 SciPy 1.13 崩溃（`A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x`）。
   - *解决方案*：锁定组合 `numpy==1.26.4` + `ml-dtypes<=0.5.1`，确保 C-API 完全兼容。
2. **CUDA 12.0 驱动下的 PyTorch 匹配**：
   - *问题*：部分服务器驱动为 CUDA 12.0（如 Driver 525.105.17），而 PyTorch 官方仅分发 `cu118` 与 `cu121` 轮子。
   - *解决方案*：通过指定 `cu118`（CUDA 11.8），在 CUDA 12.0 驱动下完全向下兼容，完美支持 RTX 4090 的 Ada Lovelace 架构（`sm_89`）。
3. **多进程与 CUDA 初始化隔离**：
   - *问题*：Python 多进程（`multiprocessing` 使用 `fork`）直接在子进程加载 CUDA 会触发内部死锁与段错误。
   - *机制*：该论文的符号恢复算法针对 CPU 并行微扰进行了高并发优化，代码中显式指定 `CUDA_VISIBLE_DEVICES = '-1'`，终端偶现的 `failed call to cuInit` 为 TensorFlow 提示信息，完全不影响算法的正确执行。

---

## 📜 论文引用与致谢

本复现基于以下 EUROCRYPT 2024 研究成果：

```bibtex
@inproceedings{chavez2024polynomial,
  title={Polynomial Time Cryptanalytic Extraction of Deep Neural Networks in the Hard-Label Setting},
  author={Chavez-Saab, Javier R. and others},
  booktitle={Advances in Cryptology--EUROCRYPT 2024},
  year={2024},
  publisher={Springer}
}
```

感谢原作者开源的基础算法原型，本项目在此基础上完成了完整的跨平台验证、自动化构建工具链与系统性实验复现评测。
