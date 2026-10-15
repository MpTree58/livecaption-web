# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目定位

livecaption 是 macOS（Apple Silicon）本地实时英文转录 + 中文翻译的命令行工具，无 UI。ASR 用 mlx-audio 跑 NVIDIA `nemotron-3.5-asr-streaming-0.6b`（cache-aware 真流式 transducer），端点检测用 Silero VAD（mlx 版）；翻译用 mlx-lm 跑腾讯混元 `Hy-MT2`。两者都走 Apple GPU/MLX、共享统一内存——VAD 把静音段挡在 encoder 外省 GPU。**全进程 mx 求值都过 `runtime.MLX_LOCK` 串行化**（MLX 多线程并发求值无官方保证）：ASR 侧整段计算持锁（毫秒级），翻译侧用 `stream_generate` 逐 decode step 取锁——既保证线程安全，又不让一次翻译把 partial 冻住数秒。

## 常用命令

```bash
uv sync                                      # 装依赖
uv run livecaption --source mic|system|both  # 运行；首次自动下载模型（ASR ~1.2GB + VAD + 说话人分离 ~225MB + 翻译 ~2GB）
uv run livecaption --source file --file x.wav --out y.md  # 转录音频文件，跑完即退出（全流程端到端测试入口）
uv run livecaption --list-devices            # 列出麦克风设备
bash scripts/build_audiotee.sh               # 编译系统音频捕获二进制（仅 system/both 需要）

# 验证：本项目无正式测试框架，用这些手动冒烟脚本
uv run python scripts/smoke_asr.py           # ASR 端到端：加载模型 + 解码自带 test wav，对照离线结果
uv run python scripts/smoke_translate.py     # 翻译端到端
uv run python scripts/smoke_diff.py          # _inline_diff 纯逻辑测试（无模型，秒级）
uv run python scripts/smoke_mel.py           # 增量 mel 与整句一次性计算的一致性（改 _mel_grow 后必跑）
uv run python scripts/diag_system_audio.py   # 系统音频 / 权限自检（播放声音看能否捕获到非零数据）
uv run --extra qwen python scripts/smoke_model_swap.py  # 网页端切换语言预设后旧模型是否释放（真模型 + say 合成音频冒充麦克风，临时会议库）
uvx ruff check .                             # lint（规则集见 pyproject）
```

## 架构：三阶段线程流水线

音频从采集到上屏经过三个用队列解耦的线程：

```
AudioSource(后台)  ──queue──▶  AsrWorker(线程)  ──final 句──▶  Translator(线程)  ──▶  Renderer / FileWriter
 mic: sounddevice 回调           mlx-audio nemotron-3.5         mlx-lm + Hy-MT2
 system: audiotee 子进程         + Silero VAD 端点切句
                                 partial / final 双事件
```

- **`audio.py`** — `MicSource`(sounddevice 回调)、`SystemAudioSource`(读 audiotee 子进程 stdout)、`FileSource`(解码音频文件)，统一把 16k float32 mono chunk 放进 `self.queue`，流结束放 `SENTINEL`。实时源绝不阻塞（队列满了丢帧）；**FileSource 例外：阻塞 put 不丢帧**（无实时性约束，端点检测数 VAD 帧不依赖墙钟）。`SystemAudioSource` 有 supervisor 线程做断流看门狗：健康的 tap 静音时也持续输出零字节流，**完全无数据 ≥5s（`SYSTEM_AUDIO_STALL_SEC`）= tap 已死**（实测诱因：切换默认输出设备，音频在新设备继续播、tap 还挂在旧设备上 IO 停转），此时杀掉重启 audiotee 重新 tap 当前设备，字幕自动恢复——曾因纯阻塞 read 静默饿死整条流水线且无任何提示。
- **`asr.py`** — `Recognizer`(共享权重) + `OnlineStream`(单流状态机：Silero VAD 判开口/静音，push 式增量 encoder + RNNT 解码) + `AsrWorker`(消费队列、把事件转回调)。发两种事件：`on_partial`(中间结果，随说话变) 和 `on_final`(端点后定稿)。diarize（默认开）时一句可能切成多个 final（每说话人一段），label 带 `#S1` 后缀。
- **`translate.py`** — `Translator` 后台加载 Hy-MT2，**只翻译 final 句**。翻译堵塞绝不反压音频（积压 ≥10 句时向 stderr 告警一次）。`context_size>0` 时用最近 N 句原文做翻译背景。生成用 `stream_generate` 手动驱动、每个 decode step 取一次 `runtime.MLX_LOCK`。模型加载失败会打印 "translation disabled" 并退出线程，转录照常。输出侧有三道防护（都不改官方 prompt 模版）：(1) 源句 < `MT_CONTEXT_MIN_WORDS` 词时不带上下文（碎片输入易让模型转去翻背景块）；(2) `_strip_boilerplate` 剥掉 "根据背景信息，以下是译文：" 这类元文本前缀；(3) 上下文回声检测（译文与上一句译文的字符 bigram 重合 ≥45% = 模型翻了背景而非原文，实测阈值能区分劫持与同话题相邻句），命中则去掉上下文重翻一次。
- **`runtime.py`** — 进程级 `MLX_LOCK`（中立归属，asr 和 translate 都 import 它，谁也不依赖谁）。
- **`render.py`** — `Renderer`(rich.Live 终端，底部活动区刷 partial、上方滚动 final) + `FileWriter`(追加写文件)。配色按 `--theme`(auto/light/dark) 分三套：单套固定色总会在某种背景下隐身，所以**只有译文真正随背景变**(default=默认前景+加粗"不赌颜色"，dark=亮青，light=深青蓝)，其余元素用对两种背景都中性可见的固定样式(grey50 中灰、说话人调色板)——**一律不用 `dim`**(dim 相对背景降亮度，两种背景都偏淡，正是用户反馈"看不清"的根因)。说话人 S1–S4 各一色(`_SPEAKER_PALETTE`，按主题分套且刻意避开青系以防与译文撞色、避开绿系以防与 diff 新词撞色；S1 品红/S2 蓝/S3 橙/S4 红跨主题保持同色相)。diff 配色：纠掉的词用中灰删除线（被取代的内容不该比正文显眼；红色曾与 S4 标签混淆），新词绿色。auto 从 `COLORFGBG` 末段探测背景，探测不到回退 default。
- **`cli.py`** — 装配层：解析参数 → 建音频源 → 每个源一个 recognizer + AsrWorker → 回调接到 renderer/writer/translator → 等 Ctrl-C。
- **`config.py`** — 所有可调参数集中在此（端点 rule、采样参数、上下文句数、prompt 模版）。
- **`models.py`** — 模型下载/文件解析、audiotee 二进制定位。

### 关键设计决策（动代码前必读）

- **只翻 final 不翻 partial**：partial 反复变，翻它浪费算力且闪烁；句子级翻译质量更好。改这个会破坏整个延迟/质量平衡。
- **two-pass 纠偏**：partial 用流式小 look-ahead（`ASR_ATT_CONTEXT=[56,6]`，560ms 延迟）求快；定稿时用 `ASR_FINAL_ATT_CONTEXT=[56,13]`（官方最高精度档）整句重解，final 和翻译输入用重解版（`asr._second_pass`）。token 一旦吐出流式端不回改——纠偏只发生在 final 这一步。有纠正时 final 事件带 `_inline_diff` 的 spans，终端在 final 行内渲染（纠掉的词灰色删除线、新词绿色，`--no-diff` 关）；diff 用 `_diff_key` 归一化比较（忽略大小写和词缘标点）——two-pass 的纠正大多只是标点/大小写微调，逐个渲染成删改对会让满屏红绿噪声，归一化后只有真正的换词才显示；文件和翻译永远用干净的纠正后文本。
- **时间戳 = 句子开始时刻**：`started_at` 在一句话第一个非空 partial 出现时记录（`asr.py`），partial 与 final 共用同一个值，所以两个回调都带它。diarize 把一句切成多段 final 时，第二段起按段首 token 的 80ms 时间戳从锚点推算各自的开始时刻（事件 5 元组的 offset 字段，None=锚点段）。renderer 里**不要**用 `datetime.now()`。
- **回调签名是贯穿三层的契约**：`on_partial(label, text, started_at)` / `on_final(label, text, started_at, diff=None)`（diff 是 two-pass 纠正的 inline spans，只有 renderer 消费；writer/translator 永远拿干净文本）；翻译侧是 `Translator.submit(label, text, started_at)` → `on_translation(label, src, zh, started_at)`（started_at 透传，输出端用它把译文挂回原文——翻译落后时 ZH 行会插在后续 EN 行之后）。改签名必须同步改 asr/translate → cli(`handle_*`) → render，否则运行时 TypeError（import 检查抓不到）。
- **both 模式 = 一份共享 Recognizer 权重 + 每源一个 OnlineStream/AsrWorker，共享一个 translator/renderer/writer**。每源用 `label`("me"/"them") 区分，label 用作 `_partials` 字典 key；终端从不显示它，文件也**只在多源（both）时**写 `[me]`/`[them]` 前缀（`FileWriter(show_label=len(sources)>1)`）——单源时 label 每行都一样，纯噪声，不写。
- **push 式 stepper 是手工移植的**：mlx-audio 只有 pull 式接口（整段音频进 generator 出），`asr._StreamingEncoder` 把它的 `stream_encode` 簿记逐行改写成 push 式。改这块前先读 mlx-audio 的 `stt/models/nemotron_asr/streaming.py` 原文对照。

## 非显而易见的约束（都是踩过的坑）

- **mlx-audio 必须 pin `>=0.4.4,<0.5`**：`asr.py` import 了它的内部实现（`streaming._stream_block`、`_PRE_ENCODE_MEL_CACHE`、alignment/tokenizer 工具），minor 升级可能改内部结构。升级前先跑 `scripts/smoke_asr.py`（它会对照离线 `generate` 验证流式结果一致）。
- **非 final 时 mel 尾部要扣留 `_MEL_HOLDBACK` 帧**：STFT center-padding 让缓冲区末尾约 2 帧 mel 的值还会随后续音频改变，提前喂进 encoder 会与离线结果产生分歧。final flush 时才全部喂完。mel 是增量维护的（`_mel_grow`：稳定前缀缓存 + 只重算尾部，避免持锁热路径上 O(n²) 重算），改它必跑 `scripts/smoke_mel.py`。
- **线程失败路径有三道防线，别拆**：(1) `AsrWorker` 异常会打 traceback 并回调 `on_error`（cli 接到 `stop_event.set`，让流水线优雅退出而不是假活）；(2) 音频源的 SENTINEL 一律走 `_put_sentinel` 非阻塞入队（消费者死了 + 队列满时，阻塞 put 会卡死清理路径）；(3) 第一次 Ctrl-C 优雅停，第二次恢复默认 handler 强制退出；`FileWriter` 对 close 后的迟到写入静默丢弃。
- **端点检测三条 rule 单位都是秒（不是帧）**，OR 关系，基于 Silero VAD 的 32ms 帧语音概率实现。rule1（无文本超时）只重置不出 final；`RULE3_MIN_UTTERANCE_LENGTH` 是"一句最长多少秒强制切"。另有 **soft max 回溯切分**（`RULE2_SOFT_MAX_UTTERANCE`，默认 8s）：快语速连续语流中停顿常不足 rule2 阈值，没有它几乎每句都撞 20s 强切、拦腰断句。实现要点：**不能**用"文本以句号结尾 + 短静音"判断——560ms 解码 look-ahead 意味着句号 token 解出来时短停顿早已过去，且句号常和下一句开头词在同一个 chunk 里一起吐出（"text ends with ." 永远观察不到）。正确做法是扫 hypothesis 里最近的句末 token（`_last_sentence_end`，带 3+ 字母防缩写守卫），按其时间戳回溯切音频：头部定稿（second pass 整段重解所以文本干净），**尾部音频回种为新 utterance 的起始缓冲**（look-ahead 里可能已含下一句开头，直接丢会截字），并把尾部时长记入 `seed_skew_sec` 让 AsrWorker 回拨 `started_at`。
- **语言参数同时支持小写短代码和英文语言名**：默认 `--asr-lang en-us`、`--target-lang zh-cn`；用户也可以传 `English`、`Chinese`、`Japanese`。ASR 层会把 `en-us` / `English` 映射到模型真实 key（如 `en-US`），传错会列出全部支持值，但别默认 `"auto"`：auto 让模型自检语言，会议中英混杂时输出语言会跳变。模型对不认识的 key 会静默回退默认语言，所以 `Recognizer` 里做了显式校验。
- **Sortformer 不要小块流式喂**：这个 checkpoint 的原生工作点是 `chunk_len=188`（15 秒/块），喂 0.6~1.3s 小块会让说话人身份漂移（实测凭空多出说话人）。正确用法是定稿时整句一次 `feed`（state 跨句携带保证编号稳定），partial 的临时标签用只读 peek（state 不持久化，避免定稿 feed 重复消费音频）。
- **说话人切分是定稿时才发生的**：两人无缝接话（间隙 <1.2s）会先合在一个 utterance 里（partial 显示混合文本），到静音端点/20s 上限才按 token 时间戳切开出多个 final。换人间隙 >1.2s 时 rule2 自然先切，无此问题。
- **token 时间戳与 diar 帧同为 80ms 粒度同一时间轴**，直接对位；换人边界的句尾标点按时间戳会落进下一组，`_attribute_speakers` 里有回挂后处理。token 是 subword（词首 piece 带前导空格，整句文本就是 token 文本直接拼接），**说话人分组只允许在词首 token 处开新组**——diar 边界落在词中间时若直接切会把一个词劈进两个 final（实测出过 "used" → "u"|"sed"）。
- **系统音频全静音 = TCC 权限问题，不是 bug**。audiotee 裸二进制经子进程启动，macOS 几乎不弹授权窗；没授权时 Core Audio **静默返回静音流**（有字节、全 0、不报错）。`SystemAudioSource` 连续约 8 秒全 0 会打印警告。解法见 README 权限一节（手动授权 + 重启终端）。**macOS 15 (Sequoia) 起「屏幕与系统录制」分上下两子区**：audiotee 只做音频 tap，必须加到下方 **「仅系统音频录制」(System Audio Recording Only)** 子区，加到顶部那个子区会照样全 0 静音（macOS 14 单列表无此区分）——此细节 audiotee README 没写，出处是作者在 [audiotee#7](https://github.com/makeusabrew/audiotee/issues/7) 的回复。
- **audiotee 需从源码编译**（Swift，无 brew/无预编译）。`scripts/build_audiotee.sh` 克隆 + `swift build` 到 `./bin/audiotee`；`models.resolve_audiotee` 按 显式路径 > `./bin` > PATH 查找。

## 约定

- **运行时面向用户的文案用英文**（console 输出、CLI `--help`、异常消息、脚本 print）；**代码注释和 docstring 也用英文**（2026-06 起为面向国际开源从中文改为英文；本 `CLAUDE.md` 作为内部开发指导保留中文，README 为中文主版 + `README.en.md` 英文版）。
- **翻译 prompt 完全遵循 Hy-MT2 官方 README 模版**（`config.py` 的 `TRANSLATE_PROMPT` / `TRANSLATE_PROMPT_WITH_CONTEXT`），不要自创加约束；采样参数也是官方推荐值，勿随意改。带上下文时前文用**空格**拼接（不是换行），因为端点切出的"句子"常是连续语流的片段。对模型输出缺陷的纠正一律放在**输出后处理**（`translate.py` 的 boilerplate 剥离 / 上下文回声重翻 / 短碎片降级为无上下文模版），不动 prompt 本身。
