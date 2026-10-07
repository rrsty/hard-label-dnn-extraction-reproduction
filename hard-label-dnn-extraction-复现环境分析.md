# hard-label-dnn-extraction 复现环境分析

检查日期：2026-10-07。仓库：https://github.com/Jchavezsaab/hard-label-dnn-extraction

本地目录：C:/Users/rsty/Desktop/GI/hard-label-dnn-extraction

固定提交：4921b679b0533e73a26f64fb522d57db9ac04ed1。

## 结论与验证范围

优先复现 full_end_to_end：CPU 即可，不需要 CUDA，也不需要 TensorFlow。全套独立实验还需要 TensorFlow/Keras；推荐分别建环境。建议 Python 3.11（64 位），完整论文实验优先 Linux 或 WSL2。

已完成克隆、34 个 Python 文件的 AST 语法检查、模型 ZIP 元数据/结构检查、依赖与路径检查。本报告中的版本是候选配置，不是作者提供的锁定环境，也未经过安装、模型加载或完整实验验证。仓库没有 requirements.txt、environment.yml 或依赖锁文件。本次没有安装大型依赖、生成双点或启动完整实验。

## 当前机器

- Windows，系统报告版本 10.0.26200，20 个逻辑处理器。
- PATH 中 Python 为 D:/develop/python-3.14.3/python.exe，版本 3.14.3，64 位。
- 此 Python 未检测到 numpy、scipy、torch、tensorflow、keras、h5py、pandas、matplotlib、tabulate。
- PATH 中没有 conda。
- nvidia-smi 返回 Failed to initialize NVML: Unknown Error；无法据此判断显卡是否存在或可用。
- 当前 Python 不适用于下面的 TensorFlow 2.20 候选环境：PyPI 为该版本提供 Python 3.9–3.13 wheel，没有 3.14 wheel。

## 各模块依赖与前置条件

| 模块 | 直接外部依赖 | 数据/运行条件 |
| --- | --- | --- |
| full_end_to_end | numpy、scipy、torch、h5py | 使用已有 unitary_32_32x3_10_float64.keras；自行生成双点与中间结果；无需 TensorFlow |
| signature_recovery | numpy、scipy、torch、matplotlib、tensorflow/keras | 默认 MODEL32=True、DEVICE=cpu、USE_GRADIENT=False；包含 models/unitary32.pth；oracle 首次调用时加载 Keras 模型 |
| sign_recovery 白盒 | numpy、tensorflow/keras、pandas、matplotlib；制表涉及 tabulate | 需要 data/dual_points_模型名/layer层号_neuron神经元号.npy |
| sign_recovery 黑盒 | numpy、tensorflow/keras、pandas；oracle 还使用 torch | 当前代码在线生成双点，不要求预计算文件；默认以真实权重模拟已恢复签名，支持 --signatures |
| sign_recovery Leaky | numpy、tensorflow/keras | 在线计算，无需预计算双点；仍假设签名已知 |
| generators | numpy、tensorflow/keras | 白盒双点生成器持续运行直到手动终止；网络生成器会覆盖原模型 |

Python 标准库（multiprocessing、pickle、zipfile 等）无需额外安装。

## 候选环境

### A：端到端最小 CPU 环境

Python 3.11，numpy==1.26.4、scipy==1.13.1、torch==2.5.1、h5py==3.11.0。

requirements-end-to-end.txt 包含 numpy/scipy/h5py；PyTorch 通过 CPU wheel 索引单独安装，避免默认 CUDA 包带来额外下载。

### B：全套实验环境

在 A 基础上增加 tensorflow==2.20.0、keras==3.15.0、pandas==2.2.3、matplotlib==3.9.2、tabulate==0.9.0，见 requirements-all.txt。它是复现起点，完成 pip check、加载模型和单神经元实验后才能视为已验证环境。推荐先使用 A 跑通小网络。

当前模型元数据：

| 模型 | 保存 Keras 版本 | 结构 |
| --- | --- | --- |
| cifar10_3x256_64_10_float64.keras | 3.15.0 | 3072 → 256 → 256 → 256 → 64 → 10，输出 softmax |
| unitary_32_32x3_10_float64.keras | 3.11.3 | 32 → 32 → 32 → 32 → 10，隐藏层 ReLU |
| unitary_leaky_32_32x3_10_float64.keras | 3.11.3 | 32 → 32 → 32 → 32 → 10，隐藏层 LeakyReLU |

因此不要直接套用 TensorFlow 2.10/Keras 2 环境。推荐 Keras 3.15.0 是为覆盖当前 CIFAR 模型的序列化配置，仍需实测加载兼容性。所有数值计算应保留 float64；代码中的 1e-14/1e-16 容差不适合改成 float32。独立 TensorFlow 脚本主动禁用 CUDA，并关闭 oneDNN 来减少舍入差异。

## 环境建立与检查

以下命令在 Linux/WSL2、已准备好 Python 3.11 后执行。在 GI 目录创建虚拟环境：

~~~bash
python3.11 -m venv .venv-hldnn
source .venv-hldnn/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-end-to-end.txt
python -m pip check
~~~

全套实验建议另建 .venv-hldnn-all，以 requirements-all.txt 替换最后的依赖文件。

Windows 安装 Python 3.11 后可用 py -3.11 -m venv .venv-hldnn；无需激活也可用 .\.venv-hldnn\Scripts\python.exe 执行以上 pip 命令。

端到端冒烟检查（进入仓库 full_end_to_end 目录）：

~~~bash
python -c "import numpy, scipy, torch, h5py; import oracle; import utils, recover_weights; print(oracle.label(numpy.zeros(32))); print('queries=', oracle.query_count())"
~~~

全套环境模型加载检查（进入仓库根目录）：

~~~bash
python -c "import tensorflow as tf, keras; from pathlib import Path; print(tf.__version__,keras.__version__); [print(p.name,tf.keras.models.load_model(p,compile=False).input_shape) for p in Path('data').glob('*.keras')]"
~~~

## 推荐运行顺序

### 1. 完整小网络提取

~~~bash
cd hard-label-dnn-extraction/full_end_to_end
python run.py 8
~~~

8 表示进程数，可先用 2–4 测试，再根据内存与 CPU 调整；不必使用默认 32。该入口已经针对 Windows 使用 spawn，Linux 使用 fork。最终输出为 out/net.npz、out/stats.json，以及各层恢复参数和日志；自动调用 validate.py。攻击阶段仅调用标签 oracle，最后验证阶段读取真实权重。

作者 README 给出的参考是 32 核机器上几分钟、约 4000 万查询；这是作者估计，不是本机实测，运行时间受进程数、数值库、随机采样影响。验证应查看每层误差和输出标签一致率，而不只看进程退出码。已有 out/ 会被用作断点结果；更换模型时应使用新的输出目录或先备份已有结果。

### 2. 独立黑盒符号恢复（先单神经元）

~~~bash
cd ../sign_recovery
python sign_recovery_blackbox.py --model unitary_32_32x3_10_float64 --layer 1 --neuron 0 --j 1
~~~

当前版本直接调用 findDualPoints，不必先下载双点。默认签名取自真实模型，属于已知签名条件下的符号实验，不能单独作为完整端到端提取的证明。oracle 固定指向 unitary ReLU 模型，所以仅修改 --model 不足以攻击其他目标。

### 3. 白盒符号恢复

双点不在克隆仓库中；先生成或从作者预计算仓库下载。生成命令：

~~~bash
cd ../generators
python dual_generator_whitebox.py cifar10_3x256_64_10_float64
~~~

每 1000 个总双点保存一次，程序不自动停止。文件按隐藏层分配，因此总数达到 10000 不代表每个神经元都拥有 10000 个样本。停止前确认目标神经元文件已经保存且样本数量足够。

~~~bash
cd ../sign_recovery
python sign_recovery_whitebox.py --model cifar10_3x256_64_10_float64 --layer 1 --neuron 0 --j 1
~~~

预计算数据来源：https://github.com/Jchavezsaab/precomputation-for-hard-label-dnn-extraction 。下载后必须保留对应原始模型；重新生成模型会使双点失效。当前模型与预计算数据是否匹配尚未核验。

### 4. Leaky 符号恢复

~~~bash
python sign_recovery_leaky.py --model unitary_leaky_32_32x3_10_float64 --j 1
~~~

该入口会运行所有隐藏层、所有神经元。

## README 与源码不一致及兼容问题

1. signature_recovery/README 写 find_dual_points.py，但实际文件名是 find_duals.py；源码当前 USE_GRADIENT=False，而 README 仍说默认 True。以固定提交的源码为准。
2. generators README 的 unitary 双点命令多写了 .keras；脚本会自动追加扩展名，应传不带后缀的模型名。
3. 白盒参数实际是 --layer、--neuron，README 前半部分出现的 --layerID/--neuronID 不能直接照抄。
4. sign_recovery 黑盒 README 的预计算要求已落后于当前在线生成双点的实现。
5. sign_recovery 多进程任务函数定义在 __main__ 内；Windows spawn 下子进程可能找不到函数或所依赖的全局状态。优先 --j 1，或改造后再并行；Linux 多进程还需注意 Python 版本的默认启动方式。full_end_to_end 已单独处理 spawn/fork。
6. 大模型/TINY 非 MODEL32 路径需要 signature_recovery/data/x_test.npy，仓库未提供；旧 leaky 子目录也缺自己的 data 和 models/deep.pth 等文件。不建议把它们当作开箱即用入口。
7. 独立 signature_recovery/find_duals.py 用 os.mkdir 创建 exp/SEED，但没有递归创建 exp 父目录；运行前应在该子目录创建 exp。大量相对路径要求从各子目录启动。
8. full_end_to_end 的网络维度、宽度和层数为固定 32/32/3；不能只替换模型文件就运行其他结构。
9. Python 3.14 AST 检查发现 create_tables.py 的反斜杠字符串 SyntaxWarning；不是当前语法错误，但进一步支持使用独立 Python 3.11 环境。

## 硬件与存储

最小端到端入口无需 GPU。建议先 4–8 个进程，16 GB RAM 是实验准备建议而非测量结论；32 进程或 CIFAR 大规模双点实验建议更多内存。小网络数万个双点通常可管理；CIFAR 的 3072 维 float64 三元双点每样本仅坐标就约 72 KiB，100 万个约 68.7 GiB，尚不含对象和其他数组开销。不要一次按 README 的千万级目标盲目生成。

## 可追溯性

模型 SHA256：

- CIFAR：144db9955316693800fbee2f3655447a1155f6c7c82ebdd57850208e2af6a4a9
- unitary ReLU：e3929efb05b60e91427752fa1cec48afb5c66dac6048b711116c1a5740aae400
- unitary Leaky：b6bb72caca6d626ade611f090f96b6cb13850f39a74bcf7b53ec0c1cb751a292

成功验证后在所用环境执行 python -m pip freeze > requirements-verified.txt，并保留提交、模型哈希、out/stats.json 和验证输出。该提交已比 2024 原始论文代码更新；若要严格复现论文原版本，还需确认作者当时使用的提交与数据。

参考：

- 仓库各子目录 README 与源码（本地固定提交）
- https://pypi.org/project/tensorflow/2.20.0/ （Python wheel 范围）
- https://pypi.org/project/keras/3.15.0/
- https://www.tensorflow.org/install/pip （Windows 原生 GPU 的旧版本限制；本项目首选 CPU）
