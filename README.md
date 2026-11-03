# Horfen Drums

音频 → 鼓点 MIDI 的转录工具。模型是**混合注意力**结构（3× GLA + 1× Anchor + AttnRes），支持**整曲推理**，不需要把歌切成 30 秒块再拼接，因此没有分块边界的假切分。

> 这不是「通用音频转 MIDI」。它只做鼓：9 类打击乐事件 → General MIDI 鼓音高。旋律乐器请用别的工具。

<br>

## 中文

### 为什么这样做

鼓点信号有三个特点：事件是 1–3 帧的窄脉冲、天然贴着节拍网格、同类事件密集重复。Transformer 的 Self-Attention 复杂度是 O(T²)，一首 4 分钟的歌就是 2 万帧以上，显存和耗时都会爆掉——原来只能 30 秒分块，代价是块边界丢事件、拼接处出现假切分。

所以这里换成 **Gated Linear Attention**：固定状态的线性递推，复杂度 O(T·d²)，一首歌一次前向跑完。再叠一层全局 Anchor 注意力和 AttnRes 残差，补回长程上下文。

### 结构

```
mel-spectrogram (B, 1, 128, T)
        │
   ┌────┴────┐
  GLA ×3（多尺度记忆：短记忆抓鼓点，长记忆抓乐句）
        │
   Anchor Attention ×1（全局上下文）
        │
     AttnRes
        │
onset logits (B, T, 9)  → kick / snare / hihat_closed / hihat_open /
                          tom_high / tom_mid / tom_low / crash / ride
```

### 安装与使用

```powershell
pip install -r requirements.txt

# 输入：已分离的鼓声轨
python audio_transcription/drum_transcriber.py .\drums.wav -o .\deliverable --title MySong
```

| 参数 | 说明 |
|---|---|
| `audio` | 鼓声轨 WAV（建议先用分离模型抽出 drum stem，本仓库不含分离模型） |
| `-o/--output` | 输出目录 |
| `--title` | MIDI 文件名 |
| `--threshold` | 兜底阈值；存在逐类校准文件时会被覆盖 |
| `--stats` | 打印每类事件计数 |

输出是 **General MIDI 鼓轨**（36 kick / 38 snare / 42·46 hats / 41·45·47 toms / 49 crash / 51 ride），可直接拖进 DAW 或 MuseScore。

### 阈值：不要用 0.95

模型输出的 sigmoid 概率整体偏低，上限约 0.94。沿用老方案的全局 0.95 阈值会得到 **0 个事件**。正确做法是读 `models/HorfenDrumsHybrid/per_class_thresholds.json` 的逐类阈值（0.5 ~ 0.7），这是在同一验证集上按类网格搜索 + F1 校准得到的，文件里同时保留了每类 `best_f1`。

同一验证集上的帧级 F1 对比（旧 Self-Attention 方案 vs 本方案）：**0.1793 → 0.2914（+62%）**。

### 仓库内容

| 路径 | 说明 |
|---|---|
| `models/HorfenDrumsHybrid.py` | 模型定义与 `load_model` |
| `models/HorfenDrumsHybrid/horfen_drums_hybrid_best.pt` | 推理权重（约 26 MB） |
| `models/HorfenDrumsHybrid/per_class_thresholds.json` | 逐类校准阈值 + F1 摘要 |
| `audio_transcription/drum_transcriber.py` | 整曲推理 → 峰值提取 → GM MIDI |

**不含**：训练脚本、训练数据、数据配比、训练超参。想复现训练请自行准备数据集与方案。

<br>

---

## English

Audio → drum MIDI transcription. The model uses a **hybrid attention** design (3× GLA + 1× Anchor + AttnRes) and performs **whole-song inference** — no 30-second chunking, therefore no boundary artefacts.

> Not a general audio-to-MIDI converter. Drums only: 9 percussion classes mapped to General MIDI pitches.

### Why

Drum onsets are 1–3 frame spikes aligned to a rhythmic grid, and a 4-minute song exceeds 20k frames. Self-Attention is O(T²), so the previous design had to chunk at 30s — losing events at boundaries and producing phantom splits. Gated Linear Attention is O(T·d²) with a fixed recurrent state, so the whole song runs in one forward pass. A global Anchor attention layer plus AttnRes restores long-range context.

### Output

General MIDI drum track (36 kick / 38 snare / 42·46 hats / 41·45·47 toms / 49 crash / 51 ride), ready for a DAW or MuseScore.

### Usage

```powershell
pip install -r requirements.txt
python audio_transcription/drum_transcriber.py .\drums.wav -o .\deliverable --title MySong
```

Input should be an isolated drum stem — **stem separation models are not included** in this repo.

### Thresholds: do not use 0.95

The model's sigmoid outputs are compressed (ceiling ≈ 0.94), so a global 0.95 threshold yields **zero events**. Read the per-class thresholds from `models/HorfenDrumsHybrid/per_class_thresholds.json` (0.5–0.7), calibrated by per-class grid search on the same validation set; per-class F1 is stored alongside.

Frame-level F1 on the same validation set: **0.1793 → 0.2914 (+62%)** vs the previous Self-Attention design.

### What is not here

Training scripts, training data, data mixtures and hyper-parameters are intentionally excluded.

<br>

## License / 许可

**PolyForm Noncommercial License 1.0.0** — 源码与权重公开，允许个人学习、研究、非商业使用与修改；**禁止任何商业用途**。

Source-available, **not OSI open source**. Commercial use is prohibited. See [LICENSE](LICENSE).

商业授权请联系作者。/ For commercial licensing, contact the author.

<br>

## 第三方依赖与许可 / Third-Party Dependencies

本仓库的模型定义、推理代码与权重以 PolyForm Noncommercial 1.0.0 授权。下列运行依赖**不在本仓库内**，由用户通过 pip 自行安装，各自适用其原有许可。

| 组件 | 许可 |
|---|---|
| PyTorch (`torch`) | BSD-3-Clause |
| librosa | ISC |
| pretty_midi | 见其官方仓库 LICENSE |
| NumPy | BSD-3-Clause |

本仓库不分发、不打包上述组件，也不包含任何第三方源码。

All model code and weights here are licensed under PolyForm Noncommercial 1.0.0. The runtime dependencies below are **not bundled**; users install them via pip and each remains under its own license. No third-party source code is included in this repository.
