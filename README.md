**简体中文** | [English](README.en.md)

# livecaption

macOS 本地实时英文转录 + 中文翻译的命令行工具，无 UI，输出到终端或文本文件。

![livecaption 演示：实时转录、S1/S2 说话人分离（不同颜色）、two-pass 纠偏（红色删除线为纠掉的词、绿色为纠正后的词）与逐句中文翻译](docs/demo.png)

- **ASR**：mlx-audio 跑 NVIDIA `nemotron-3.5-asr-streaming-0.6b`（cache-aware 真流式 transducer），Apple Silicon GPU / MLX；端点检测用 Silero VAD（mlx 版）。
- **说话人分离**（默认开，`--no-diarize` 关）：NVIDIA Sortformer v2.1 流式模型，最多 4 人；句子按说话人切分、标 S1/S2/…、各自翻译，编号全程稳定。
- **翻译**：mlx-lm 跑 `Hy-MT2-1.8B-8bit`（腾讯混元 MT 第三代），Apple Silicon GPU。
- **音源**：麦克风，或会议系统音频（Zoom/Teams/浏览器扬声器输出），或两者双流分轨。

ASR 和翻译都跑 Apple GPU（统一内存）；VAD 把静音段挡在 encoder 之外，只有说话时才耗 GPU；翻译只处理 ASR 的定稿句，不反压音频。定稿句经过 two-pass 纠偏：实时字幕用低延迟流式解码，句子结束时用最大 look-ahead 整句重解一遍，定稿和翻译都用更准的重解结果。

## 前置要求

- macOS 14.2+（系统音频捕获用到 Core Audio process tap；Tahoe 建议 ≥ 26.1）
- Apple Silicon
- [uv](https://docs.astral.sh/uv/)
- 仅 `system` / `both` 音源需要：Swift 5.9+（编译 audiotee）

## 安装

```bash
uv sync
# 仅在需要捕获会议/系统音频时，编译 audiotee：
bash scripts/build_audiotee.sh
```

首次运行会自动从 Hugging Face 下载 ASR（约 1.2GB）、Silero VAD（很小）、说话人分离模型 Sortformer（约 225MB，默认开启，`--no-diarize` 可关）和翻译模型（约 2GB）。

## 用法

```bash
# 麦克风实时转录 + 翻译，输出到终端
uv run livecaption --source mic

# 转录会议输出（系统音频），写入文件
uv run livecaption --source system --out meeting.md

# 双流分轨：同时转录自己的麦克风和对方的声音
uv run livecaption --source both --out meeting.md

# 转录音频文件（会议录音复盘 / 端到端测试；wav/mp3/m4a，自动重采样，跑完即退出）
uv run livecaption --source file --file recording.m4a --out recording.md

# 播客音频成稿：使用音频进度时间戳，并避免 qwen 词级说话人切分把句子切碎
uv run --extra qwen livecaption --source file --file podcast.mp3 --out podcast.md \
  --asr-model qwen3-1.7b-8bit --asr-lang Chinese --podcast --no-translate

# 翻译默认带前 3 句上下文（提升代词指代/术语连贯）；如需关闭或调整：
uv run livecaption --source system --context 0

# 只转录不翻译
uv run livecaption --source mic --no-translate

# 翻成日语、换更大更准的翻译模型
uv run livecaption --target-lang ja-jp --mt-model hy-mt2-7b-4bit

# 中文 / 中英混说会议：换 Qwen3-ASR 引擎（需 `--extra qwen`），用 8bit 量化省显存
uv run --extra qwen livecaption --asr-model qwen3-1.7b-8bit --asr-lang Chinese

# 非英语会议：指定说话语言（40 个 locale，传错会列出全部可选值）
uv run livecaption --asr-lang de-de --target-lang zh-cn

# 只捕获某个 App（先用 `ps`/活动监视器拿到 Zoom 的 PID）
uv run livecaption --source system --include-pid 12345

# 列出麦克风设备
uv run livecaption --list-devices

# 配色：默认 auto（探测终端背景明暗，探测不到则用高对比保底配色）；
# 浅色背景看不清时显式指定，深色终端同理
uv run livecaption --theme light

# 监控 MLX 统一内存占用（终端底部状态行显示 active/cache/peak；默认关闭，诊断用）
uv run livecaption --source mic --mem
```

终端里：底部灰色行是实时中间结果；定稿句向上滚动为原文，下方是译文（加粗或彩色，取决于主题）。说话人 S1–S4 各用一种颜色便于辨认。

**配色不清楚时**：默认 `--theme auto` 会尝试从 `COLORFGBG` 探测背景明暗，但多数 macOS 终端（Terminal/iTerm/VS Code）不设这个变量，探测不到就回退到“默认前景色 + 加粗”的保底方案（任何背景都和正文一样清楚，但译文不带专属颜色）。想要彩色译文就显式指定 `--theme light`（浅色背景，译文深青蓝）或 `--theme dark`（深色背景，译文亮青）。

## 模型选择

`--asr-model` 和 `--mt-model` 都是**固定列表**（传列表外的值会被拒并列出可选值）；ASR 引擎由所选模型自动决定，不再有单独的后端开关。

**`--asr-model`**（默认 `nemotron-0.6b`）：

| 别名 | 引擎 | 说明 |
|---|---|---|
| `nemotron-0.6b` / `nemotron-0.6b-8bit` | nemotron | 英文强、实时逐词 partial + 说话人分离；8bit 省显存 |
| `qwen3-1.7b` / `qwen3-1.7b-8bit` | qwen3 | 中文 / 中英混说强（需 `--extra qwen`）；fp16 ~4.7GB，8bit ~2.4GB |
| `qwen3-0.6b` / `qwen3-0.6b-8bit` | qwen3 | 更小更快的中文版；8bit ~1.0GB |

**`--mt-model`**（默认 `hy-mt2-1.8b-8bit`）：`hy-mt2-1.8b-8bit`（~2GB）、`hy-mt2-1.8b-4bit`、`hy-mt2-7b-4bit`（~4.2GB，更准）、`hy-mt2-7b-8bit`。

> qwen3 的 8bit 量化由本项目转换并发布在 HuggingFace：`Six666/mlx-qwen3-asr-1.7b-8bit`、`Six666/mlx-qwen3-asr-0.6b-8bit`。

## 权限

- **麦克风**：首次运行系统弹窗，授权落在你的终端 App 上。

- **系统音频（重要）**：audiotee 是裸二进制、又经 Python 子进程启动，macOS 的授权弹窗**经常不出现**。如果跑 `--source system` 看到「正在监听」却完全没有转录（约 8 秒后程序会打印静音警告），几乎可以肯定是「屏幕与系统录制」权限没给——Core Audio 在无权限时会静默返回静音流，不报错也不弹窗。手动授权：

  1. 打开 **系统设置 > 隐私与安全性 > 屏幕与系统录制**
  2. **macOS 15（Sequoia）及以上这一栏分上下两个子区**：往下滚到 **「仅系统音频录制」(System Audio Recording Only)** 子区——**不是**顶部那个「屏幕与系统录制」子区——把运行本工具的终端 App（Terminal / iTerm / VS Code 等）加进去并打开开关；没有就点 `+` 手动添加（如 `/System/Applications/Utilities/Terminal.app`）。audiotee 只做音频 tap、不录屏，**加错子区会照样全 0 静音**。（macOS 14 只有单一列表，无此区分。）
  3. **完全退出并重启该终端 App**（TCC 权限变更必须重启进程才生效），然后重跑
  4. 仍不行就先用 macOS 自带的 Terminal.app（而非 iTerm/VS Code 终端）跑一次，它更容易触发授权弹窗

  > 「上下两个子区」这个细节连 audiotee 自己的 README 都没写，只见于作者在 [audiotee#7](https://github.com/makeusabrew/audiotee/issues/7) 的回复。

  验证权限是否生效：播放任意声音，跑 `uv run python scripts/diag_system_audio.py`，看 `max |amplitude|` 是否 > 0。

## 选型说明

`nemotron-3.5-asr-streaming` 是 `nemotron-speech-streaming-en` 的官方多语言后继（同为 cache-aware 真流式，0.6B 参数预算不变），不是用滑窗模拟离线模型。运行时选 mlx-audio：MLX 原生跑 Apple GPU（sherpa-onnx 在 mac 上只能 CPU，CoreML 对流式 transducer 算子覆盖不全）。mlx-audio 只提供 pull 式接口，本项目把它的 streaming 内核改写成了 push 式实时 stepper（`asr.py`），端点检测由 Silero VAD 按 sherpa 的 rule1/2/3 语义重实现。默认说话语言是 `en-us`（内部映射到模型的 `en-US` key），避免 auto 检测在混杂语流里跳变。语言参数同时支持小写短代码和英文语言名，如 `zh-cn` / `Chinese`、`ja-jp` / `Japanese`、`de-de` / `German`。

## 已知风险与退路

- ASR 流式质量不满意 → 调大 `ASR_ATT_CONTEXT`（如 `[56,13]`，精度最好但 1.12s 一刷、partial 延迟更大）。
- ASR 与翻译同抢 GPU 出现卡顿 → 翻译换更小 / 更低精度的模型，`--no-diarize` 关掉说话人分离，或 `--no-translate` 只转录，让 ASR 独占 GPU。
- 显存吃紧 → ASR 换 8bit 量化版 `--asr-model nemotron-0.6b-8bit`（或 qwen3 用 `qwen3-1.7b-8bit` / `qwen3-0.6b-8bit`）。
- 翻译质量不够 → `--mt-model hy-mt2-7b-4bit`（约 4.2GB，更准）。
- 单独 tap 某个 App 偶发静音 → 默认 tap 全系统输出更稳（不传 `--include-pid`）。
