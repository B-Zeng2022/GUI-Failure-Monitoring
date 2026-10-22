# GUI Failure Monitoring

[English](readme.md) | [简体中文](readme_zh.md)

论文 **What Makes a Wrong Click? An Empirical Analysis of Decision Reliability
and Failure Detection in GUI Agents** 的方法推理代码。

给定用户指令、屏幕截图和可选的交互历史，项目先生成下一步操作，再在操作执行前计算三个模型内部信号：推理熵 `H_CoT`、结构感知坐标不确定性 `U_coord`，以及推理与动作的表示耦合度 `C_RA`。可以使用预先校准并冻结的固定权重或逻辑回归模型，将三个信号融合为点击失败风险分数。

支持 **AgentCPM-GUI-8B**、**MAI-UI-8B** 和 **UI-TARS-1.5-7B**。主实验的数据目录映射覆盖 CMGUI、CAGUI、AC-High 和 GUIOdyssey；另外保留 AITZ 与 AC-Low 的可选输入支持。

本项目提供方法的推理实现，不包含数据集、模型权重、论文预测缓存或已经拟合的逻辑回归系数，也不保证复现论文结果表。原脚本中的坐标熵单位、UI-TARS 原生坐标和历史窗口等实现约定，见[方法与实现说明](docs/METHOD.md)。

## 项目结构

```text
gui_failure/
  backends.py          本地模型加载、提示词和生成
  signals.py           从单次生成过程中提取三个信号
  parsing.py           各模型原生推理段与点击坐标的严格解析
  data.py              预处理后 episode 数据加载
  actions.py           历史动作的格式转换
  cmgui.py             官方 CMGUI 测试标注转换
  parallel.py          每卡一个模型实例与分片结果合并
  fusion.py            固定标准化参数与固定权重/逻辑回归融合
  evaluation.py        兼容原规则的 AgentCPM/MAI 动作评测
  cli.py               统一命令行入口
  resources/           模型提示词与 AgentCPM 动作 schema
eval/
  scripts_gui/run_inference.py   可直接运行的推理脚本
  eval_data/                    数据准备说明与脚本；数据保留在本地
requirements/                   两套模型环境和可选评测、融合依赖
docs/                           环境、数据和方法的详细说明
tests/                          合成数据上的数值与集成检查
```

原来复制过来的探索实验脚本及配套依赖，完整保留在本地 `legacy/eval/`。该目录已被 Git 忽略，也不进入 Python 发布包，其中包括遮挡、缩放、注意力、不确定性锚点和多层表示等实验。当前主推理流程不导入这些归档代码。若编辑器隐藏了被忽略的目录，见[原始脚本清单与位置](docs/ORIGINAL_SCRIPTS.md)。

## 1. 配置模型环境与确认入口

使用 Python 3.11，并为 AgentCPM 与另外两个模型分别配置环境。版本固定参考作者实际使用的环境和模型官方项目。CUDA 安装、权重下载和上游链接见[环境配置说明](docs/ENVIRONMENT.md)。

AgentCPM：

```bash
conda create -n gui_agent python=3.11 -y
conda activate gui_agent
python -m pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements/agentcpm.txt
python -m pip install -e .
```

MAI-UI / UI-TARS：

```bash
conda create -n mai python=3.11 -y
conda activate mai
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements/mai-tars.txt
python -m pip install -e .
```

以上命令在项目根目录运行。如果已有匹配的模型环境，可以直接激活它，无需重新创建。

项目统一入口是 `gui_failure/cli.py` 中的 `main()`。推荐通过 `python -m gui_failure` 调用，它由 `gui_failure/__main__.py` 转入统一入口。安装项目后，`gui-failure` 命令与其等价。五个子命令分别是：

| 子命令 | 功能 |
| --- | --- |
| `check-data` | 检查数据结构、指令和图片路径 |
| `infer` | 生成操作并提取三个信号，可选同时应用融合模型 |
| `evaluate` | 沿用原 AgentCPM/MAI 规则评测动作正确性，并导出点击错误标签 |
| `fit-fusion` | 在独立校准集上拟合标准化参数与融合参数 |
| `score` | 对已提取的信号应用冻结的融合模型 |

查看推理参数：

```bash
python -m gui_failure infer --help
```

也可以直接运行 `python eval/scripts_gui/run_inference.py --help`。这个脚本会调用同一个入口，并自动选择 `infer` 子命令，因此参数中不用再写 `infer`。

主推理所需的辅助实现都在 `gui_failure/` 内。原始 AgentCPM 工程根目录中的独立 `utils.py` 不是当前项目的依赖，无需复制到开源仓库。

AgentCPM 使用权重目录提供的 `MiniCPMVTokenizerFast`，不能强制给它的处理器设置 `use_fast=False`。若加载时提示缺少 `tokenization_qwen2.py`，见[环境说明](docs/ENVIRONMENT.md#agentcpm-gui)。

## 2. 准备输入数据

数据来源与预处理参考 [eval/eval_data/readme.md](eval/eval_data/readme.md)，准确的目录层级与 episode JSON 格式见[数据说明](docs/DATA.md)。保持约定的数据目录即可，不需要将数据集上传到项目中。

CMGUI 下载中断或出现超时时，参考[下载排查与重试命令](eval/eval_data/readme.md#download-troubleshooting-xet-errors-and-http-timeouts)。

可以先检查少量样本，无需加载模型或使用 GPU：

```bash
python -m gui_failure check-data --data-name cmgui_test --max-samples 10
```

如果数据不在本项目的 `eval/eval_data/`，添加 `--data-root /path/to/eval_data`。单张截图推理不需要准备数据集，见下一节。

## 3. 生成操作与提取信号

先激活对应模型环境，并准备好权重：

```bash
# AgentCPM：单张截图，不使用历史
python -m gui_failure infer --backend agentcpm --model-path model/AgentCPM-GUI \
  --image /path/to/screen.png --instruction 'Open Settings' \
  --history-n 0 --output results/agentcpm/single.jsonl

# AgentCPM：在预处理后的 CMGUI 上运行少量样本
python -m gui_failure infer --backend agentcpm --model-path model/AgentCPM-GUI \
  --data-name cmgui_test --max-samples 10 --output results/agentcpm/cmgui.jsonl

# MAI-UI：AC-High
python -m gui_failure infer --backend maiui --model-path model/MAI-UI-8B \
  --data-name android_control_high_test --max-samples 10 \
  --output results/maiui/ac_high.jsonl

# UI-TARS：GUIOdyssey
python -m gui_failure infer --backend tars --model-path model/UI-TARS-1.5-7B \
  --data-name gui_odyssey_test --max-samples 10 --output results/tars/odyssey.jsonl
```

删除 `--max-samples` 即可处理所选测试集的全部样本。每条命令处理一种模型和一个数据集；通过 `--data-name` 切换 CAGUI 等数据集。`--model-path` 可以是本地权重目录或 Hugging Face 模型 ID；省略时使用对应的官方 ID。本地离线运行可以添加 `--local-files-only`。

已经下载到 Hugging Face 缓存中的模型不需要复制到项目的 `model/`。`--model_path` 可以直接填写官方模型 ID，也可以省略该参数，让程序使用所选 backend 的默认模型 ID。加上 `--local-files-only` 时，程序会先定位缓存中的快照目录，再加载模型与处理器；缺少缓存时直接报错，不会开始下载。MAI-UI 和 UI-TARS 使用 `mai` 环境。

默认设备为 `cuda:0`，CUDA 下默认使用 bfloat16。代码每个进程加载一个模型到一张 GPU。使用 `--device` 运行单进程，或通过 `--devices cuda:0,cuda:1,cuda:2,cuda:3` 将数据集分给四个模型实例处理。`CUDA_VISIBLE_DEVICES` 控制进程能看到的物理显卡，`--devices` 中的编号以可见设备列表为准。

AgentCPM 默认开启采样（`temperature=0.1`、`top_p=0.3`）；MAI-UI 和 UI-TARS 默认使用贪心解码。所有模型都可以显式添加 `--do-sample`（或 `--do_sample`）开启采样，添加 `--no-sample`（或 `--no_sample`）关闭采样。采样参数可通过 `--temperature`、`--top-p`、`--top-k`、`--seed` 设置，`--max-new-tokens` 控制输出长度。

`--history-n 3` 保留之前所有动作与思维文本，以及最多两张历史截图和当前截图。添加 `--history-steps 3`，可进一步将文本与动作历史限制为最近三步；`--history-n 0` 完全关闭历史。具体语义见[方法说明](docs/METHOD.md)。

结果以 JSONL 格式保存，每个输入步骤对应一行，包括非点击操作：

```json
{
  "sample_id": "cmgui_test/cmgui/episode_001/0",
  "generated_text": "...",
  "action": {"thought": "...", "POINT": [600, 300]},
  "point_native": [600, 300],
  "point_1000": [600, 300],
  "is_valid_click": true,
  "signals": {"H_CoT": 0.8, "U_coord": 0.12, "C_RA": 0.65},
  "signal_errors": {},
  "signal_config": {"backend": "agentcpm", "alpha": 1.0, "digit_weight": 0.7, "coordinate_entropy_unit": "normalized"}
}
```

上述数值仅用于说明输出格式，不是实验结果。其他字段记录坐标分量、生成长度、图片尺寸、样本 ID 与随机种子。无法计算的信号保存为 `null`，原因写入 `signal_errors`。推理与风险计算不读取当前步骤的参考动作或正确性标签。

多卡运行时，输出目录下的文件含义如下：

| 文件 | 用途 |
| --- | --- |
| `predict.jsonl` | 所有进程成功后按数据集顺序合并的最终结果，每个步骤一行 |
| `predict.jsonl.meta.json` | 全局运行参数、GPU 分配与分片文件位置，用于检查断点续跑配置 |
| `predict.shards/worker-NNN.jsonl` | 单个进程负责的样本分片结果 |
| `predict.shards/worker-NNN.jsonl.meta.json` | 单进程的设备、分片设置、Python 与包版本 |
| `predict.shards/worker-NNN.log` | 模型加载、逐样本进度、警告和报错堆栈 |

使用 `--output` 指定其他 JSONL 文件名时，元信息和分片目录随该文件名变化。worker 编号是进程序号，不一定等于 GPU 编号。`is_valid_click` 表示能否将预测解析为有效点击，不代表点击目标正确。未指定 `--fusion` 时，结果不会包含融合后的 `risk` 字段。这些文件保存汇总特征，不保存完整 logits、隐状态矩阵、benchmark 正确性标签或检测指标。

运行环境版本与参数保存到输出旁的 `<output>.meta.json`；多卡运行时，包版本记录在 `predict.shards/` 内每个进程的元信息文件中（使用 `--output` 时，分片目录名为 `<output-stem>.shards/`）。已有输出受到保护，不会直接覆盖。恢复中断运行时，在相同命令中添加 `--resume`：程序会检查配置一致性，并跳过已经保存的样本 ID。每个完整样本写入后都会刷新文件；损坏或不完整的 JSONL 行会明确报错，不会被悄悄忽略。


### AgentCPM + CMGUI 四卡运行示例（兼容旧参数写法）

先按[测试集下载与转换命令](eval/eval_data/readme.md#cmgui-test-split)准备 CMGUI。针对“不使用历史、开启采样”的配置，可以运行：

```bash
conda activate gui_agent
python eval/scripts_gui/run_inference.py \
  --backend agentcpm \
  --model_path ./model/AgentCPM-GUI \
  --data_name cmgui_test \
  --output_dir ./end_results/AgentCPM-GUI/consist_entropy_cot_click_cmgui_his0 \
  --devices cuda:0,cuda:1,cuda:2,cuda:3 \
  --do_sample \
  --history_n 0
```

`--output_dir`（或 `--output-dir`）会在指定目录下保存 `predict.jsonl`。每张 GPU 加载一份独立模型，各自处理互不重复的样本分片，最后按数据集顺序合并；这是按样本分工，不是将模型的不同层切到多张卡上。`predict.shards/` 中保留各进程的 JSONL、元信息和日志。任一进程失败都不会生成一个声称完整的合并结果；可以在相同命令后添加 `--resume` 恢复。

旧参数 `--model_path`、`--data_name`、`--data_root`、`--history_n`、`--do_sample` 与对应的连字符写法都受支持。`--output` 指定完整 JSONL 文件名，`--output_dir` 指定结果目录，二者只能选一个。

如果命令同时出现 `--history_n 3` 和后面的 `--history_n 0`，argparse 最终采用最后一个值，即 0。因此示例只保留 `--history_n 0`；需要历史时，将它改成 3 即可。`--max-samples` 限制的是所有进程合计的样本数。数据在其他位置时，添加 `--data_root /path/to/eval_data`。

## 4. 评测预测动作

新的 JSONL 通过以下入口使用原 AgentCPM/MAI-UI 的正确性判定规则评测，不需要加载模型或使用 GPU：

```bash
python -m pip install -r requirements/evaluation.txt
python -m gui_failure evaluate \
  --backend agentcpm \
  --data-name cmgui_test \
  --input end_results/AgentCPM-GUI/consist_entropy_cot_click_cmgui/predict.jsonl \
  --output-dir end_results/AgentCPM-GUI/consist_entropy_cot_click_cmgui/evaluation \
  --require-complete
```

评测 MAI-UI 时改为 `--backend maiui`，并填写它的输入、输出路径。Android Control 和旧脚本一样，需要添加 `--eval-android-control`。

输出包括逐步骤 `result.json`、沿用旧字段的 `summary.json`，以及含点击错误标签的 `predictions_labeled.jsonl`。正确点击标为 `error_label=0`，错误点击为 `1`，非点击预测为 `null`。动作正确性评测、风险融合和检测 AUROC/AP 是不同环节；`score` 只应用融合参数。旧判定规则中的边界情况、文件用途及一致性范围见[评测说明](docs/EVALUATION.md)。

## 5. 校准与应用风险融合

三个信号的顺序是 `[H_CoT,U_coord,C_RA]`。AgentCPM/UI-TARS 的固定权重为 `(0.25,0.45,0.30)`，MAI-UI 为 `(0.60,0.15,0.25)`。标准化需要参考统计量，应先在**独立校准集**上提取特征，再保存并冻结标准化参数：

```bash
python -m gui_failure fit-fusion --method fixed \
  --input results/calibration/agentcpm.jsonl --output calibration/agentcpm_fixed.json

python -m gui_failure score --input results/agentcpm/cmgui.jsonl \
  --fusion calibration/agentcpm_fixed.json --output results/agentcpm/cmgui_scored.jsonl
```

也可以给 `infer` 添加 `--fusion calibration/agentcpm_fixed.json`，在每个新样本推理时直接输出融合风险。固定权重融合得到的是标准化加权风险分数，不是概率。没有融合参数文件时，仍然可以正常生成操作并输出三个原始信号。

使用监督式融合时，先按照对应 benchmark 的正确性评估规则，在校准集的每行数据中加入 `error_label`：错误点击为 `1`，正确点击为 `0`。

```bash
python -m pip install -r requirements/fusion.txt
python -m gui_failure fit-fusion --method logistic \
  --input results/calibration/agentcpm_labeled.jsonl \
  --output calibration/agentcpm_logistic.json
```

随后同样使用 `score` 或 `infer --fusion` 应用该文件。逻辑回归融合输出估计的错误概率。实际推理时不重新训练标准化器或融合模型，也不需要当前样本的标签。不同模型或信号配置应分别校准，不要使用留出的测试集拟合参数。

`fit-fusion` 会在传入的全部校准样本上拟合用于部署的模型，不会自动执行或复现论文的五折 episode 分组交叉验证。

## 验证与适用范围

```bash
python -m unittest discover -s tests -v
```

检查覆盖坐标不确定性的解析计算、严格输出解析、数据与历史隔离、冻结的融合参数，以及 token 与隐状态的位置对齐。安装 Torch 和 Transformers 后，会使用小型随机初始化模型，对照原始 logits 和末层隐状态验证提取逻辑，包括 Qwen3-VL 的不同隐状态返回约定。测试不需要 benchmark 数据。

完整的十亿参数模型推理仍需相应权重和可用 GPU。已经执行的检查与尚未验证的范围见[验证记录](docs/VALIDATION.md)。隔离执行环境能否访问 GPU，与服务器是否安装显卡是两个问题；排查时应分别确认宿主机的 `nvidia-smi` 和实际推理环境的 CUDA 可见性。

## 许可证与致谢

采用 Apache-2.0，见 [LICENSE](LICENSE) 与[第三方来源说明](THIRD_PARTY_NOTICES.md)。本项目使用了 [AgentCPM-GUI](https://github.com/OpenBMB/AgentCPM-GUI)、[MAI-UI](https://github.com/Tongyi-MAI/MAI-UI) 和 [UI-TARS](https://github.com/bytedance/UI-TARS) 的接口约定与相关资源。研究使用时，请引用本论文以及对应的上游模型与数据集。
