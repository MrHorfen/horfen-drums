#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 MrHorfen
# SPDX-License-Identifier: MIT
"""
文件: HorfenDrumsHybrid.py
用途: HorfenDrumsHybrid —— 混合注意力（3×GLA + 1×Anchor + AttnRes）鼓点转录模型（生产主方案 v3.19.0）
      GLA(Gated Linear Attention) O(T·d²) 线性注意力，支持整曲推理，取代原 HorfenDrums 单层
      Self-Attention 的 O(T²) 瓶颈（原方案因显存限制只能 30s 分块，边界有拼接损耗）。
调用方: audio_transcription/drum_transcriber.py（load_model 惰性加载）
输入: mel-spectrogram (B, 1, 128, T)
输出: 鼓点 onset logits (B, T, 9)
依赖: torch
创建时间: 2026-08-15
最后更新: 2026-08-15
更新说明: 由 temp/scripts/hybrid_model.py（沙箱 S6b 修复版）逐字复制升格为生产模型，架构零改动。
          训练校准结论：hybrid sigmoid 概率整体偏低（上限≈0.94），逐类阈值 0.5~0.7 更优，
          生产原 0.95 全局阈值在 hybrid 上会 0 事件，不得沿用（见 per_class_thresholds.json）。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 常量（与生产 HorfenDrums 对齐）----
DRUM_CLASSES = ["kick", "snare", "hihat_closed", "hihat_open",
                "tom_high", "tom_mid", "tom_low", "crash", "ride"]
NUM_CLASSES = len(DRUM_CLASSES)


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


# ============================================================
# Gated Linear Attention（双向，块式串行递推）
# ============================================================
class GatedLinearAttention(nn.Module):
    """Gated Linear Attention（GLA）—— O(T·d²) 固定状态线性注意力

    递推（单方向）: S_t = decay ⊙ S_{t-1} + k_t ⊗ v_t ;  o_t = q_tᵀ S_t
    - q,k 过 RMSNorm，v 不过；decay = sigmoid(log_decay) ∈ (0,1)^(H×d_h)，逐通道衰减
    - 多尺度初始化：低编号 head 衰减大（短记忆/快节奏），高编号 head 衰减小（长记忆/乐句）
    - 训练：chunk 边界 detach 状态（truncated BPTT），控制计算图深度与显存
    - 双向：正向递推 + 序列翻转反向递推，拼接后 out_proj
    """

    def __init__(self, d: int = 384, h: int = 8, chunk: int = 128, dropout: float = 0.0,
                 learn_decay: bool = False):
        super().__init__()
        self.d, self.h, self.dh = d, h, d // h
        self.chunk = chunk
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(2 * d, d, bias=False)
        self.rms = RMSNorm(d)
        self.dropout = nn.Dropout(dropout)

        # 多尺度衰减初始化：低编号 head 记忆长（decay 大→log_decay 小→慢遗忘），高编号 head 记忆短
        head_idx = torch.linspace(2.0, -2.0, h)  # (h,)  → sigmoid ∈ (0,1)，逐 head 标量衰减
        self.log_decay = nn.Parameter(head_idx.clone())
        # 默认冻结：GLA 分块反向对"强负衰减 head"的 log_decay 梯度产生 NaN（fp64 下亦复现），
        # 保持多尺度初值作为固定记忆画像；learn_decay=True 供后续修复 backward 后开启
        if not learn_decay:
            self.log_decay.requires_grad_(False)

    def _fwd(self, q, k, v, log_decay):
        """单方向 chunkwise 并行递推（与逐帧递推等价，快 ~75x）

        逐 head 标量衰减：decay=sigmoid(param)∈(0,1) → log_decay=log(decay)≤0。
        qlog = chunk 内累计 log_decay（≤0，随时间单调非增）。

        - within-chunk：w[t,s]=exp(qlog_t−qlog_s)（s≤t 因果），因 qlog_t≤qlog_s → w∈(0,1] **无溢出**，
          attn=(Q·Kᵀ)⊙w（(B,H,C,C) 仅 4MB），o_intra=attn·V；s>t 用 tril 屏蔽避免 w>1 溢出
        - cross-chunk：S_abs=chunk 起始绝对状态（Σ_{s<cs} exp(Σ_{u=s+1..cs-1}log) k_s v_s），
          o_inter[t]=exp(qlog_t)·Q_tᵀ·S_abs（qlog_t=chunk 内累计，从 cs 起含 cs 本身）
        - S 更新（chunk 末）：S_next=exp(qlog_end)·S_abs + Σ_s w[end,s]·k_s⊗v_s
        - fp32 全程：指数≤0 永不溢出；极端远史权重下溢为 0（无害，梯度亦为 0 仅影响远古位置）
        - truncated BPTT：chunk 边界 detach 状态，只影响梯度不影响前向
        """
        orig_dtype = q.dtype
        q, k, v = q.float(), k.float(), v.float()
        log_decay = log_decay.float()
        B, H, T, dh = q.shape
        o = torch.empty(B, H, T, dh, device=q.device, dtype=q.dtype)
        S = torch.zeros(B, H, dh, dh, device=q.device, dtype=q.dtype)  # 绝对状态 (i,j)
        c = self.chunk
        for s in range(0, T, c):
            e = min(s + c, T)
            qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]  # (B,H,C,dh)
            lc = log_decay[:, :, s:e]                              # (B,H,C) ≤0
            # truncated BPTT：状态跨 chunk 传递，梯度在 chunk 边界截断
            S_abs = S.detach()
            qlog = torch.cumsum(lc, dim=2)                         # (B,H,C) 累计（≤0）
            # within-chunk 因果衰减注意力 w[t,s]=exp(qlog_t−qlog_s)，s≤t → w∈(0,1]
            w = torch.exp(qlog[:, :, :, None] - qlog[:, :, None, :])  # (B,H,C,C)
            w = torch.tril(w)                                      # 屏蔽 s>t（否则 w>1 溢出）
            qk = qc @ kc.transpose(-1, -2)                         # (B,H,C,C)
            o_intra = (qk * w) @ vc                                # (B,H,C,dh)
            # cross-chunk：S_abs 为 chunk 起始绝对状态，按 exp(qlog_t) 缩放到当前
            o_inter = torch.einsum("bhCi,bhij->bhCj", qc, S_abs) * torch.exp(qlog[:, :, :, None])
            o[:, :, s:e] = o_intra + o_inter
            # S 更新（chunk 末）：衰减旧状态 + 本 chunk 累计（w[end,s]=exp(qlog_end−qlog_s)）
            S = S_abs * torch.exp(qlog[:, :, -1:])[:, :, :, None] \
                + torch.einsum("bhCi,bhCj->bhij", w[:, :, -1, :, None] * kc, vc)
        return o.to(orig_dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,d) → (B,T,d)"""
        B, T, d = x.shape
        h = self.rms(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)  # (B,H,T,dh)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        log_decay = torch.log(torch.sigmoid(self.log_decay).clamp_min(0.05))  # (H,) ≤0
        log_decay = log_decay[None, :, None].expand(B, self.h, T)  # (B,H,T) 逐 head

        o_fwd = self._fwd(q, k, v, log_decay)
        o_bwd = self._fwd(q.flip(2), k.flip(2), v.flip(2), log_decay.flip(2)).flip(2)

        o = torch.cat([o_fwd, o_bwd], -1).transpose(1, 2).reshape(B, T, 2 * d)  # (B,T,2d)
        return self.dropout(self.out(o))  # 无内部输出门（身份初始化下梯度路径干净）


# ============================================================
# Softmax Anchor Attention（局部窗口 + 全局锚）
# ============================================================
class AnchorAttention(nn.Module):
    """Softmax Anchor Block —— 局部窗口双向全 softmax + 全局锚 token 交叉注意力

    - 局部窗口（W=512 ≈ 5.1s，hop=W//2 重叠）：窗口内全秩精确交互，直接覆盖密集击打/滚奏
    - 全局锚（A=256 可学习 token）：token 以精确 softmax 读取全局锚字典，补偿线性注意力状态秩不足
    - 输出门控 σ(g) 兼作噪音抑制
    """

    def __init__(self, d: int = 384, h: int = 8, W: int = 512, A: int = 256, dropout: float = 0.0):
        super().__init__()
        self.d, self.h, self.dh = d, h, d // h
        self.W, self.A = W, A
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.out = nn.Linear(d, d, bias=False)
        self.rms = RMSNorm(d)
        self.dropout = nn.Dropout(dropout)
        # 全局锚字典（可学习 token）
        self.anchors = nn.Parameter(torch.randn(A, d) * 0.02)

    def _local_window(self, q, k, v):
        """重叠窗口内双向全 softmax 注意力，结果累加。q,k,v: (B,H,T,dh)"""
        B, H, T, dh = q.shape
        W, hop = self.W, self.W // 2
        o = torch.zeros_like(q)
        scale = 1.0 / math.sqrt(dh)
        for s in range(0, T, hop):
            e = min(s + W, T)
            qc, kc, vc = q[:, :, s:e], k[:, :, s:e], v[:, :, s:e]
            attn = (qc @ kc.transpose(-1, -2)) * scale  # (B,H,C,C)
            attn = F.softmax(attn, dim=-1)
            o[:, :, s:e] += attn @ vc
        return o

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,T,d) → (B,T,d)"""
        B, T, d = x.shape
        h = self.rms(x)
        q, k, v = self.qkv(h).chunk(3, -1)
        q = q.view(B, T, self.h, self.dh).transpose(1, 2)
        k = k.view(B, T, self.h, self.dh).transpose(1, 2)
        v = v.view(B, T, self.h, self.dh).transpose(1, 2)
        scale = 1.0 / math.sqrt(self.dh)

        # (1) 局部窗口软注意力
        o_local = self._local_window(q, k, v)

        # (2) 全局锚交叉注意力：token 为 q，锚为 k/v
        _, ak, av = self.qkv(self.anchors.unsqueeze(0)).chunk(3, -1)  # (1,A,3d) → 各 (1,A,d)
        ak = ak.view(1, self.A, self.h, self.dh).transpose(1, 2)   # (1,H,A,dh)
        av = av.view(1, self.A, self.h, self.dh).transpose(1, 2)
        attn_g = (q @ ak.transpose(-1, -2)) * scale  # (B,H,T,A)
        attn_g = F.softmax(attn_g, dim=-1)
        o_global = attn_g @ av  # (B,H,T,dh)

        o = (o_local + o_global).transpose(1, 2).reshape(B, T, d)
        return self.dropout(self.out(o))  # 无内部输出门（身份初始化下梯度路径干净）


# ============================================================
# AttnRes（Block Attention Residuals）—— 跨层块摘要路由
# ============================================================
class AttnRes(nn.Module):
    """把前一块输出的块摘要路由到本块输入。gate 初始 0.0 → 训练初期无扰，随训练学开"""

    def __init__(self, d: int = 384, n_seg: int = 4, dropout: float = 0.0):
        super().__init__()
        self.route = nn.Linear(d, d)
        self.gate = nn.Parameter(torch.zeros(1))  # 零初始化
        self.n_seg = n_seg
        self.dropout = nn.Dropout(dropout)

    def forward(self, prev_out: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """prev_out: (B,T,d) 前一块完整输出；x: (B,T,d) 本块输入 → 门控残差注入"""
        B, T, d = x.shape
        # 块摘要：自适应 N 段 mean-pool → (B, n_seg, d)
        z = F.adaptive_avg_pool1d(prev_out.transpose(1, 2), self.n_seg).transpose(1, 2)
        # 上采样回 T（线性插值）——输入 (B, d, n_seg) 为 3D
        z = F.interpolate(z.transpose(1, 2), size=T, mode="linear",
                          align_corners=False).transpose(1, 2)  # (B,T,d)
        z = self.dropout(self.route(z))
        return x + torch.tanh(self.gate) * z


# ============================================================
# HorfenDrumsHybrid（整体：Conv + BiGRU + HybridAttention ×4 + Head）
# ============================================================
class HorfenDrumsHybrid(nn.Module):
    """HorfenDrums 混合注意力升级版（沙箱研究）

    架构：
    - ConvStack：4×Conv2D(1→32→64→128→256, k=3, BN, ReLU, MaxPool(2,1)) → (B,T,2048)【与生产一致】
    - BiGRUStack：2×GRU(192, bidir, dropout=0.3) → (B,T,384)【与生产一致】
    - HybridAttentionStack ×4（身份初始化，取代生产 MHA）：
        Block1: GLA + 标量门 g1
        Block2: GLA + 标量门 g2
        Block3: Anchor(W=512,A=256) + 标量门 g3
        Block4: GLA + 标量门 g4（输入额外注入 Block3 摘要，AttnRes gate 初始 0）
        残差 = x + g_i·block(x)（零初始化残差，无 LN、无内部输出门）
    - Head：Linear(384→192)→ReLU→Dropout(0.3)→Linear(192→9)【与生产一致】
    - 身份初始化（identity_init=True）：各块 out 投影=0 → 初始前向精确等于
      "生产底座"(conv+gru+fc，无 MHA)，注意力栈从零学习不阻塞迁移底座
    """

    def __init__(
        self,
        n_mels: int = 128,
        num_classes: int = NUM_CLASSES,
        conv_channels: list = None,
        gru_hidden: int = 192,
        gru_layers: int = 2,
        attn_heads: int = 8,
        d_model: int = 384,
        gla_chunk: int = 128,
        window: int = 512,
        n_anchors: int = 256,
        dropout: float = 0.3,
        identity_init: bool = True,
        learn_decay: bool = False,
    ):
        super().__init__()
        if conv_channels is None:
            conv_channels = [32, 64, 128, 256]
        self.n_mels = n_mels
        self.num_classes = num_classes
        self.d_model = d_model
        self.identity_init = identity_init
        self.learn_decay = learn_decay
        assert d_model == gru_hidden * 2, "d_model 必须等于 BiGRU 输出维 2*gru_hidden"

        # --- ConvStack（4 层，与生产一致）---
        convs = []
        in_ch = 1
        for ch in conv_channels:
            convs.append(nn.Sequential(
                nn.Conv2d(in_ch, ch, kernel_size=(3, 3), padding=(1, 1)),
                nn.BatchNorm2d(ch),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            ))
            in_ch = ch
        self.conv_stack = nn.ModuleList(convs)
        self._conv_output_dim = conv_channels[-1] * (n_mels // 16)  # 256*8=2048

        # --- BiGRUStack（与生产一致）---
        self.gru = nn.GRU(
            input_size=self._conv_output_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0,
        )

        # --- HybridAttentionStack ---
        self.gla1 = GatedLinearAttention(d_model, attn_heads, chunk=gla_chunk, dropout=dropout,
                                         learn_decay=learn_decay)
        self.gla2 = GatedLinearAttention(d_model, attn_heads, chunk=gla_chunk, dropout=dropout,
                                         learn_decay=learn_decay)
        self.anchor = AnchorAttention(d_model, attn_heads, W=window, A=n_anchors, dropout=dropout)
        self.gla4 = GatedLinearAttention(d_model, attn_heads, chunk=gla_chunk, dropout=dropout,
                                         learn_decay=learn_decay)
        self.attnres = AttnRes(d_model, n_seg=4, dropout=dropout)
        # 每块标量残差门（初始 1.0，随训练学）
        self.res_gate = nn.Parameter(torch.ones(4))

        # --- Head（与生产一致）---
        self.fc = nn.Sequential(
            nn.Linear(d_model, gru_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden, num_classes),
        )

        if identity_init:
            self._apply_identity_init()

    def _apply_identity_init(self):
        """身份初始化：使注意力栈初始输出=0 → 残差恒等

        配合零初始化残差（x = x + g·block(x)，g 为标量门 init=1）：
        - 各注意力块 out 投影零初始化 → block(x) = 0 → 残差恒等
        → 初始前向精确等于"生产底座"(conv+gru+fc，无 MHA)，
          注意力栈在此好底座上从零学习（out 投影先学，qkv 随后渐进解冻）
        """
        for m in (self.gla1, self.gla2, self.anchor, self.gla4):
            m.out.weight.data.zero_()
            if m.out.bias is not None:
                m.out.bias.data.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B,1,n_mels,T) → onset logits (B,T,9)"""
        for conv in self.conv_stack:
            x = conv(x)
        B, _, _, T = x.shape
        x = x.permute(0, 3, 1, 2).reshape(B, T, -1)  # (B,T,2048)

        x, _ = self.gru(x)  # (B,T,384)

        g = self.res_gate
        # 零初始化残差：块输出经标量门 g 加到主干（无 LN，避免输入=0 处梯度放大）
        # 身份初始化下 block(x)=0 → 每个块都是恒等，主干精确 = 生产底座路径
        x = x + g[0] * self.gla1(x)
        x = x + g[1] * self.gla2(x)
        x3 = x
        x = x + g[2] * self.anchor(x)
        x = self.attnres(x3, x)  # 注入 Block3 摘要（gate 初始 0，恒等）
        x = x + g[3] * self.gla4(x)

        return self.fc(x)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(x))


# ============================================================
# 工具：参数量 / 迁移初始化
# ============================================================
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_pretrained_weights(hybrid: "HorfenDrumsHybrid", ckpt_path: str, device="cpu") -> list:
    """从生产 HorfenDrums checkpoint 迁移 ConvStack/BiGRU/Head 权重，返回未迁移的层名列表

    - CNN/BiGRU/FC 头：直接 load_state_dict（strict=False，仅匹配键名）
    - 注意力栈 + AttnRes：不迁移（若 identity_init=True 已零初始化 out 投影，保持身份初始化；
      否则保持随机初始化）。生产 HorfenDrums 的 MHA/attn_norm 键不匹配 → 自动跳过
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    # 生产键名：conv1..conv4 / gru / fc.*
    # hybrid 键名：conv_stack.0..3 / gru / fc.* —— 需要键名映射
    mapping = {}
    for k in state.keys():
        if k.startswith("conv") and "." in k and k.split(".")[0] in {f"conv{i}" for i in range(1, 5)}:
            # conv1.xxx -> conv_stack.0.xxx
            idx = int(k.split(".")[0][4:]) - 1
            new_k = "conv_stack." + str(idx) + "." + ".".join(k.split(".")[1:])
            mapping[k] = new_k
        elif k.startswith("gru") or k.startswith("fc"):
            mapping[k] = k
        elif k.startswith("attn_norm"):
            # 生产 1 层 self-attn 的 LN，不迁移
            continue
        else:
            continue
    new_state = {}
    missed = []
    for k, v in state.items():
        if k in mapping:
            new_state[mapping[k]] = v
        else:
            missed.append(k)
    incompatible = hybrid.load_state_dict(new_state, strict=False)
    return missed



# ============================================================
# 生产加载入口
# ============================================================
def load_model(
    checkpoint_path,
    device=None,
    config: dict = None,
) -> nn.Module:
    """加载训练好的 HorfenDrumsHybrid 模型 checkpoint（生产标准入口）

    Args:
        checkpoint_path: .pt 文件路径（state_dict 或含 model_state_dict）
        device: torch.device（默认 cuda/cpu）
        config: 覆盖模型超参（默认取训练时的 MODEL_CONFIG）

    Returns:
        HorfenDrumsHybrid 实例（已 load_state_dict + eval）
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = dict(
        n_mels=128, num_classes=NUM_CLASSES,
        conv_channels=[32, 64, 128, 256], gru_hidden=192, gru_layers=2,
        attn_heads=8, d_model=384, gla_chunk=128, window=512, n_anchors=256, dropout=0.3,
        identity_init=False, learn_decay=False,
    )
    if config:
        cfg.update(config)

    model = HorfenDrumsHybrid(**cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        raise RuntimeError(f"HorfenDrumsHybrid 权重加载不完整，缺失参数: {missing}")
    model.to(device)
    model.eval()
    return model
