#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 MrHorfen
# SPDX-License-Identifier: MIT
"""
文件: drum_transcriber.py
用途: 打击乐扒谱模块（HorfenDrumsHybrid 混合注意力全曲推理 + 逐类校准阈值 + librosa 信号处理兜底），
      9类鼓映射GM MIDI。v3.19.0 起主路径由原 HorfenDrums(Self-Attention, 30s分块) 升级为
      HorfenDrumsHybrid(GLA+Anchor, 整曲推理)，阈值由全局 0.95 改为逐类校准 0.5~0.7。
调用方: piano_transcribe.py
输入: 音频文件
输出: 鼓组 MIDI 文件（GM Percussion Map）
依赖: torch, librosa, numpy, pretty_midi, models.HorfenDrumsHybrid
作者: Meandro(菠萝的自我救赎）
创建时间: 2026-07-26
最后更新: 2026-08-15
更新说明: v3.19.0 主路径替换为 HorfenDrumsHybrid 全曲推理 + 逐类校准阈值。
          依据验证集 F1（seed=42）：hybrid mean F1 0.2914 vs 原方案最优 0.1793（+62%），
          每类均提升（tom_mid +219%、ride +382%）。原 HorfenDrums 方案已归档至 archive/。
"""

import torch
import os
import sys
import time
import logging
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import pretty_midi

# 项目根目录（确保能导入 models 模块）
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ============================================================
# 鼓映射
# ============================================================
# 模型输出 9 类 → GM MIDI 音高
MODEL_CLASS_TO_MIDI = [
    36,   # kick
    38,   # snare
    42,   # hihat_closed
    46,   # hihat_open
    50,   # tom_high
    47,   # tom_mid
    43,   # tom_low
    49,   # crash
    51,   # ride
]

MODEL_CLASS_NAMES = [
    "kick", "snare", "hihat_closed", "hihat_open",
    "tom_high", "tom_mid", "tom_low", "crash", "ride",
]

DRUM_MAP = {
    "kick": 36,
    "snare": 38,
    "snare_rim": 37,
    "hh_closed": 42,
    "hh_pedal": 44,
    "hh_open": 46,
    "crash": 49,
    "ride": 51,
    "ride_bell": 53,
    "high_tom": 50,
    "mid_tom": 47,
    "low_tom": 43,
    "floor_tom": 41,
}
DRUM_NAMES = {v: k for k, v in DRUM_MAP.items()}

# 项目路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# 模型 checkpoint 路径（按优先级排序）
MODEL_CANDIDATES = [
    PROJECT_ROOT / "models" / "HorfenDrumsHybrid" / "horfen_drums_hybrid_best.pt",  # HorfenDrumsHybrid（混合注意力整曲推理，v3.19.0 主方案）
]

# 逐类校准阈值 JSON（验证集 seed=42 网格搜索 F1 最大化；hybrid sigmoid 概率整体偏低，
# 原 0.95 全局阈值会 0 事件，必须用本逐类阈值）
PER_CLASS_THRESHOLD_JSON = PROJECT_ROOT / "models" / "HorfenDrumsHybrid" / "per_class_thresholds.json"

# 推理配置
SAMPLE_RATE = 16000
N_MELS = 128
HOP_LENGTH = 160
N_FFT = 2048
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 模型加载（惰性加载，单例）
# ============================================================
_model_instance = None
_model_config = None
_th_per_class_cache = None


def _load_model() -> Tuple[Optional[nn.Module], Optional[Dict]]:
    """加载训练好的鼓点模型（惰性加载，自动选择最佳 checkpoint）"""
    global _model_instance, _model_config

    if _model_instance is not None:
        return _model_instance, _model_config

    # 按优先级查找可用的 checkpoint
    checkpoint_path = None
    for candidate in MODEL_CANDIDATES:
        if candidate.exists():
            checkpoint_path = candidate
            break

    if checkpoint_path is None:
        logger.warning(f"模型文件不存在（搜索路径: {[str(c) for c in MODEL_CANDIDATES]}），将使用信号处理兜底方案")
        return None, None

    try:
        from models.HorfenDrumsHybrid import load_model
        model = load_model(str(checkpoint_path), DEVICE)
        _model_instance = model
        _model_config = {"model_path": str(checkpoint_path)}
        logger.info(f"鼓点模型加载完成 ({checkpoint_path.name}, arch={model.__class__.__name__})")
        return model, _model_config
    except Exception as e:
        logger.warning(f"模型加载失败 ({e})，将使用信号处理兜底方案")
        return None, None


def _load_per_class_thresholds() -> Optional[Dict[str, float]]:
    """惰性加载逐类校准阈值（验证集 F1 最大化）；不存在时返回 None（用全局阈值）"""
    global _th_per_class_cache
    if _th_per_class_cache is not None:
        return _th_per_class_cache
    try:
        if PER_CLASS_THRESHOLD_JSON.exists():
            with open(PER_CLASS_THRESHOLD_JSON, "r", encoding="utf-8") as f:
                _th_per_class_cache = json.load(f)["per_class_thresholds"]
            return _th_per_class_cache
    except Exception as e:
        logger.warning(f"逐类阈值加载失败 ({e})，回退全局阈值")
    return None


# ============================================================
# 深度学习推断
# ============================================================
def _infer_nn(y: np.ndarray, sr: int,
              threshold: float = 0.95,
              min_interval: float = 0.05) -> List[Dict]:
    """
    使用 HorfenDrumsHybrid 模型进行鼓点推断（v3.19.0 整曲推理）

    Hybrid 模型返回 logits，需手动 sigmoid 转换为概率。
    默认使用逐类校准阈值（models/HorfenDrumsHybrid/per_class_thresholds.json），
    全局 threshold 仅作为无校准文件时的兜底。

    Args:
        y: 音频数据 (单声道)
        sr: 采样率
        threshold: onset 检测阈值 [0-1]（兜底，校准文件存在时被逐类覆盖）
        min_interval: 同类鼓最小间隔（秒），用于去重

    Returns:
        List[Dict]: [{time, drum_type, midi_note, confidence}, ...]
    """
    model, config = _load_model()
    if model is None:
        return []

    # 重采样到 16kHz（与训练一致）
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)

    # 使用 librosa 计算 mel（与训练时 extract_features.py 完全一致）
    mel_db = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH,
        n_mels=N_MELS, fmin=30.0, fmax=8000.0,
    )
    mel_np = librosa.power_to_db(mel_db, ref=np.max).astype(np.float32)

    # 整曲推理（Hybrid GLA 线性注意力 O(T·d²)，无 30s 分块边界损耗）
    mel_t = torch.from_numpy(mel_np).unsqueeze(0).unsqueeze(0).float().to(DEVICE)  # (1, 1, 128, T)

    with torch.no_grad():
        output = model(mel_t).squeeze(0).cpu()  # (T', 9)
        probs = torch.sigmoid(output).numpy()

    # 逐类阈值：校准 JSON 优先，否则全局 threshold
    th_per_class = _load_per_class_thresholds()
    binary = np.zeros_like(probs)
    for ci, c in enumerate(MODEL_CLASS_NAMES):
        t_c = (th_per_class or {}).get(c, threshold)
        binary[:, ci] = (probs[:, ci] > t_c).astype(np.float32)

    # binary 边沿检测（对齐 infer.py：取 0→1 跳变，而非峰值）
    events = []
    for cls_idx in range(probs.shape[1]):
        col = binary[:, cls_idx]
        for t in range(1, len(col)):
            if col[t] == 1 and col[t - 1] == 0:
                time_sec = t * HOP_LENGTH / SAMPLE_RATE
                confidence = float(probs[t, cls_idx])
                events.append({
                    "time": round(time_sec, 4),
                    "drum_type": MODEL_CLASS_NAMES[cls_idx],
                    "midi_note": MODEL_CLASS_TO_MIDI[cls_idx],
                    "confidence": round(confidence, 3),
                })

    # 按时间排序
    events.sort(key=lambda e: e["time"])

    # 去重：同一时间（30ms内）同类鼓只保留一个
    if events:
        unique = [events[0]]
        for ev in events[1:]:
            last = unique[-1]
            if (ev["time"] - last["time"] < min_interval
                    and ev["drum_type"] == last["drum_type"]):
                continue
            unique.append(ev)
        events = unique

    return events


def _find_peaks(probs: np.ndarray, threshold: float = 0.5) -> List[int]:
    """简单峰值检测"""
    if len(probs) < 3:
        return []
    peaks = []
    for i in range(1, len(probs) - 1):
        if probs[i] > threshold and probs[i] > probs[i - 1] and probs[i] >= probs[i + 1]:
            peaks.append(i)
    return peaks


# ============================================================
# 信号处理兜底（原方案）
# ============================================================
def extract_onset_features(
    y: np.ndarray,
    sr: int,
    onset_time: float,
    segment_duration: float = 0.15,
) -> Dict[str, float]:
    start_sample = int(onset_time * sr)
    end_sample = int((onset_time + segment_duration) * sr)
    end_sample = min(end_sample, len(y))
    if end_sample - start_sample < sr * 0.02:
        return {}
    segment = y[start_sample:end_sample]
    n_fft = min(2048, len(segment))
    if n_fft < 64:
        return {}
    S = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=n_fft // 4))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
    total_energy = np.sum(S) + 1e-10
    sub_bass = np.sum(S[(freqs >= 20) & (freqs < 60)]) / total_energy
    bass = np.sum(S[(freqs >= 60) & (freqs < 150)]) / total_energy
    low_mid = np.sum(S[(freqs >= 150) & (freqs < 500)]) / total_energy
    mid = np.sum(S[(freqs >= 500) & (freqs < 2000)]) / total_energy
    high_mid = np.sum(S[(freqs >= 2000) & (freqs < 5000)]) / total_energy
    high = np.sum(S[(freqs >= 5000) & (freqs < 12000)]) / total_energy
    ultra_high = np.sum(S[freqs >= 12000]) / total_energy
    centroid = np.sum(freqs * np.sum(S, axis=1)) / total_energy
    centroid_diff = freqs - centroid
    bandwidth = np.sqrt(np.sum((centroid_diff ** 2) * np.sum(S, axis=1)) / total_energy)
    rolloff = np.sum(S[freqs < 2000]) / total_energy
    rms = np.sqrt(np.mean(segment ** 2))
    peak_idx = np.argmax(np.abs(segment))
    decay_samples = 0
    peak_amp = np.abs(segment[peak_idx])
    threshold_val = peak_amp / np.e
    for i in range(peak_idx, len(segment)):
        if np.abs(segment[i]) < threshold_val:
            decay_samples = i - peak_idx
            break
    decay_time = decay_samples / sr if decay_samples > 0 else segment_duration
    return {
        "sub_bass": sub_bass, "bass": bass, "low_mid": low_mid,
        "mid": mid, "high_mid": high_mid, "high": high, "ultra_high": ultra_high,
        "centroid": centroid, "bandwidth": bandwidth, "rolloff": rolloff,
        "rms": rms, "decay_time": decay_time,
    }


def classify_drum(features: Dict[str, float]) -> Tuple[str, float]:
    if not features:
        return "unknown", 0.0
    sub_bass = features.get("sub_bass", 0)
    bass = features.get("bass", 0)
    low_mid = features.get("low_mid", 0)
    mid = features.get("mid", 0)
    high_mid = features.get("high_mid", 0)
    high = features.get("high", 0)
    ultra_high = features.get("ultra_high", 0)
    centroid = features.get("centroid", 0)
    decay = features.get("decay_time", 0.15)
    high_total = high_mid + high + ultra_high
    low_total = sub_bass + bass + low_mid
    # === Hihat / Cymbal 先检测（高频优先）===
    if high > 0.15 and centroid > 3500 and decay < 0.15:
        return "hh_closed", min(high * 2.5, 1.0)
    if (high_mid + high) > 0.3 and centroid > 3000 and 0.10 <= decay < 0.30:
        return "hh_open", min((high_mid + high) * 1.5, 1.0)
    if (high_mid + high + ultra_high) > 0.25 and decay > 0.4:
        return "crash", min((high_mid + high + ultra_high) * 1.5, 1.0)
    if high_mid > 0.2 and high > 0.15 and centroid > 3000 and decay > 0.2:
        return "ride", min((high_mid + high) * 1.2, 1.0)

    # === Kick: 低频短促 ===
    if (sub_bass + bass) > 0.2 and centroid < 500 and decay < 0.12:
        return "kick", min((sub_bass + bass) * 1.8, 1.0)

    # === Snare: 中频宽带 + 一定高频 ===
    if (low_mid > 0.12 and mid > 0.08 and high_total > 0.06 and 200 < centroid < 3000):
        return "snare", min((low_mid + mid) * 1.3, 1.0)
    if (bass + low_mid) > 0.3 and 200 < centroid < 1000 and 0.05 < decay < 0.3:
        if centroid < 500:
            return "low_tom", min((bass + low_mid) * 1.0, 0.9)
        else:
            return "high_tom", min((bass + low_mid) * 1.0, 0.9)
    if low_total > high_total:
        return ("kick", 0.5) if centroid < 500 else ("snare", 0.4)
    else:
        return ("hh_closed", 0.5) if decay < 0.15 else ("crash", 0.4)


def _infer_signal(y: np.ndarray, sr: int,
                  onset_sensitivity: float = 0.5) -> List[Dict]:
    """信号处理兜底方案"""
    onset_env = librosa.onset.onset_strength(
        y=y, sr=sr, hop_length=512, aggregate=np.median)
    onset_frames = librosa.onset.onset_detect(
        onset_envelope=onset_env, sr=sr, hop_length=512,
        backtrack=True, pre_max=3, post_max=3, pre_avg=3, post_avg=5,
        delta=0.15 * (1.0 - onset_sensitivity + 0.3), wait=3)
    onset_times = librosa.frames_to_time(onset_frames, sr=sr, hop_length=512)

    events = []
    for onset_t in onset_times:
        features = extract_onset_features(y, sr, onset_t)
        drum_type, confidence = classify_drum(features)
        midi_note = DRUM_MAP.get(drum_type, 36)
        if drum_type == "unknown":
            continue
        events.append({
            "time": round(onset_t, 4),
            "drum_type": drum_type,
            "midi_note": midi_note,
            "confidence": round(confidence, 3),
        })
    return events


# ============================================================
# 主入口
# ============================================================
def transcribe_drums(
    drums_wav_path: str,
    output_dir: str,
    title: str = "drums",
    sr: int = 16000,
    onset_sensitivity: float = 0.5,
    velocity_default: int = 100,
    nn_threshold: float = 0.95,
    nn_min_interval: float = 0.05,
    time_signature: Tuple[int, int] = (4, 4),
    tempo_bpm: Optional[float] = None,
) -> Path:
    """
    从鼓声轨转录为 GM MIDI

    优先使用深度学习模型，失败则回退到信号处理。

    Args:
        drums_wav_path: Demucs 分离后的 drums.wav 路径
        output_dir: 输出目录
        title: 标题（用于 MIDI 文件名）
        sr: 重采样率
        onset_sensitivity: onset 检测灵敏度（仅信号处理方案）
        velocity_default: 默认力度
        nn_threshold: 模型 onset 检测阈值 [0-1]
        nn_min_interval: 同类鼓最小间隔（秒）
        time_signature: 拍号 (分子, 分母)，默认 4/4
        tempo_bpm: BPM，None 时默认 120

    Returns:
        Path: 生成的 MIDI 文件路径
    """
    logger.info(f"鼓转录开始: {drums_wav_path}")

    # 加载音频（用 librosa 替代 soundfile，支持 MP3 等格式）
    y, orig_sr = librosa.load(drums_wav_path, sr=None, mono=True)
    if orig_sr != sr:
        y = librosa.resample(y, orig_sr=orig_sr, target_sr=sr)

    duration = len(y) / sr
    logger.info(f"  音频时长: {duration:.1f}s, 采样率: {sr}Hz")

    # ===== 方案1：深度学习 HorfenDrumsHybrid（优先）=====
    logger.info("[1/2] 深度学习 HorfenDrumsHybrid 推理...")
    events = []
    method = "signal"

    try:
        t0 = time.time()
        nn_events = _infer_nn(y, sr, threshold=nn_threshold, min_interval=nn_min_interval)
        elapsed = time.time() - t0
        if nn_events and len(nn_events) >= 5:
            logger.info(f"  Hybrid 模型输出 {len(nn_events)} 个事件 (耗时 {elapsed:.1f}s)")
            events = nn_events
            method = "nn_hybrid"
        else:
            logger.info(f"  Hybrid 输出偏少 ({len(nn_events)} 个)，回退信号处理...")
    except Exception as e:
        logger.warning(f"  Hybrid 推理失败 ({e})，回退信号处理...")

    # ===== 方案2：信号处理（兜底）=====
    if method == "signal":
        logger.info("[2/2] 信号处理 onset 检测 & 鼓分类...")
        t0 = time.time()
        events = _infer_signal(y, sr, onset_sensitivity=onset_sensitivity)
        elapsed = time.time() - t0
        logger.info(f"  信号处理耗时: {elapsed:.1f}s, 检测到 {len(events)} 个事件")

    # 统计
    type_counts = {}
    for ev in events:
        dt = ev["drum_type"]
        type_counts[dt] = type_counts.get(dt, 0) + 1
    logger.info(f"  识别结果 ({method}): {type_counts} ({len(events)} 个鼓事件)")

    # ===== 生成 MIDI =====
    logger.info("[2/2] 生成 GM MIDI...")
    _tempo = tempo_bpm if tempo_bpm else 120.0
    midi = pretty_midi.PrettyMIDI(initial_tempo=_tempo, resolution=480)
    # 写入拍号元数据（修复 MuseScore 显示 4/3 的问题）
    midi.time_signature_changes.append(
        pretty_midi.TimeSignature(
            numerator=time_signature[0],
            denominator=time_signature[1],
            time=0.0,
        )
    )
    safe_title = "".join(c if c.isascii() and c.isalnum() or c in "_-" else "_" for c in title)
    drum_track = pretty_midi.Instrument(
        program=0, is_drum=True, name=f"{safe_title}_drums",
    )

    seen: Set[Tuple[float, int]] = set()
    # 量化到 16 分音符网格（~151ms@99BPM），确保 MuseScore 正确识别 4/4 拍
    sixteenth = 60.0 / _tempo / 4  # 16分音符时长
    
    # P0: 力度动态映射（confidence → velocity）
    # 收集所有 confidence，归一化到 velocity 范围 60-120
    confidences = [ev["confidence"] for ev in events]
    if confidences:
        conf_min = min(confidences)
        conf_max = max(confidences)
        conf_range = conf_max - conf_min if conf_max > conf_min else 1.0
    else:
        conf_min, conf_max, conf_range = 0.0, 1.0, 1.0
    
    for ev in events:
        start = round(ev["time"] / sixteenth) * sixteenth
        key = (start, ev["midi_note"])
        if key in seen:
            continue
        seen.add(key)
        
        # 力度映射：confidence 归一化 → velocity 60-120
        conf_norm = (ev["confidence"] - conf_min) / conf_range
        velocity = int(60 + conf_norm * 60)  # 60-120 范围
        velocity = max(60, min(120, velocity))  # 钳制
        
        note = pretty_midi.Note(
            velocity=velocity,
            pitch=ev["midi_note"],
            start=start,
            end=start + 0.1,
        )
        drum_track.notes.append(note)

    midi.instruments.append(drum_track)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / f"{safe_title}_drums.mid"
    midi.write(str(midi_path))
    total_notes = len(drum_track.notes)
    logger.info(f"  鼓 MIDI: {midi_path} ({total_notes} 音符)")
    logger.info(f"鼓转录完成 (method={method}): {total_notes} 个鼓事件")
    return midi_path


# ============================================================
# 命令行入口
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="打击乐扒谱 - 深度学习鼓转录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="示例:\n  python drum_transcriber.py drums.wav -o ./output\n  python drum_transcriber.py drums.wav -o ./output --title Song --threshold 0.5",
    )
    parser.add_argument("audio", help="鼓声轨 WAV 文件")
    parser.add_argument("-o", "--output", default="./deliverable", help="输出目录")
    parser.add_argument("--title", default="drums", help="标题")
    parser.add_argument("--sensitivity", type=float, default=0.5, help="onset 灵敏度 (信号处理)")
    parser.add_argument("--threshold", type=float, default=0.95,
                        help="NN 检测阈值（兜底；存在逐类校准 JSON 时被覆盖，hybrid 用 0.5~0.7 更优）")
    parser.add_argument("--sr", type=int, default=16000, help="重采样率")
    parser.add_argument("--stats", action="store_true", help="输出统计 JSON")

    args = parser.parse_args()
    midi_path = transcribe_drums(
        drums_wav_path=args.audio, output_dir=args.output,
        title=args.title, sr=args.sr,
        onset_sensitivity=args.sensitivity, nn_threshold=args.threshold,
    )
    if args.stats:
        midi = pretty_midi.PrettyMIDI(str(midi_path))
        stats = {}
        for inst in midi.instruments:
            for note in inst.notes:
                name = DRUM_NAMES.get(note.pitch, str(note.pitch))
                stats[name] = stats.get(name, 0) + 1
        print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
