"""模型二的可审阅实现：蛋白残基 Q，MolT5 序列 + MACCS token 作 K/V。

本文件只定义网络，不加载编码器、不提取特征、不读数据、不启动训练。
主要维度：B=批量，L=蛋白残基数，M=MolT5 token 数，H=注意力头数。
mask=True 表示有效内容，False 表示 padding 或已排除的特殊 token。
"""

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class AttentionConfig:
    """必须显式传入；示例参数不代表已经确认的正式实验配置。"""

    num_heads: int
    attention_dropout: float
    output_dropout: float
    linear_bias: bool

    def __post_init__(self):
        if type(self.num_heads) is not int or self.num_heads <= 0:
            raise ValueError("num_heads must be a positive integer")
        if 1024 % self.num_heads != 0:
            raise ValueError("num_heads must divide protein width 1024")
        for name in ("attention_dropout", "output_dropout"):
            if not 0.0 <= getattr(self, name) < 1.0:
                raise ValueError(name + " must be in [0, 1)")
        if type(self.linear_bias) is not bool:
            raise TypeError("linear_bias must be bool")


@dataclass
class InteractionOutput:
    # 核心输出仍保留蛋白位置；无效蛋白位置全部置零。
    residues: Tensor                       # T′: [B, L, 1024]
    delta: Tensor                          # ΔT: [B, L, 1024]
    protein_mask: Tensor                   # [B, L]
    substrate_mask: Tensor                 # [B, M+1]，最后一列为 MACCS
    attention: Optional[Tensor] = None     # [B, H, L, M+1]，Dropout 之前


def _check_mask(mask: Tensor, shape, device, name: str) -> None:
    """不猜测 mask 的含义，也不从零向量推断哪个位置有效。"""
    if mask.dtype != torch.bool or tuple(mask.shape) != tuple(shape):
        raise ValueError(name + " must be bool with shape " + str(tuple(shape)))
    if mask.device != device:
        raise ValueError(name + " must be on the same device as its features")
    if not bool(mask.any(dim=1).all()):
        raise ValueError(name + " must contain at least one valid position per sample")


class ResidueSubstrateCrossAttention(nn.Module):
    """对每个蛋白残基查询底物 memory，然后加回同位置的原始 ProtT5。

    这里显式展开多头注意力，方便逐行检查 Q/K/V 和 softmax 的维度。
    原始残差 T 不做 BatchNorm、池化或投影。仅注意力分支做线性投影。
    不添加额外 FFN、位置编码、类型 embedding、归一化或可学习残差门控。
    Linear 使用 PyTorch 默认初始化；不加载旧权重，也不将末层置零。
    """

    def __init__(self, config: AttentionConfig):
        super().__init__()
        self.config = config
        self.head_dim = 1024 // config.num_heads

        # 两种底物输入的维度不同，先分别投影到共同的 1024 维空间。
        # 同一个 molt5_projection 对每个 token 使用同一组权重，不平均 M。
        self.molt5_projection = nn.Linear(768, 1024, bias=config.linear_bias)
        self.maccs_projection = nn.Linear(167, 1024, bias=config.linear_bias)
        self.query_projection = nn.Linear(1024, 1024, bias=config.linear_bias)
        self.key_projection = nn.Linear(1024, 1024, bias=config.linear_bias)
        self.value_projection = nn.Linear(1024, 1024, bias=config.linear_bias)
        self.output_projection = nn.Linear(1024, 1024, bias=config.linear_bias)
        self.attention_dropout = nn.Dropout(config.attention_dropout)
        self.output_dropout = nn.Dropout(config.output_dropout)

    def _split_heads(self, x: Tensor) -> Tensor:
        # [B, token数, 1024] -> [B, H, token数, 1024/H]
        batch, length, _ = x.shape
        return x.reshape(batch, length, self.config.num_heads, self.head_dim).transpose(1, 2)

    def forward(
        self,
        protein_tokens: Tensor,     # 原始逐残基 ProtT5: [B, L, 1024]
        protein_mask: Tensor,       # 有效蛋白残基: [B, L]
        molt5_tokens: Tensor,       # 平均池化之前的 MolT5: [B, M, 768]
        molt5_mask: Tensor,         # 有效 SMILES 内容 token: [B, M]
        maccs: Tensor,              # 原始 MACCS 位值，转浮点: [B, 167]
        *,
        return_attention: bool = False,
    ) -> InteractionOutput:
        # 1. 输入契约：拒绝把已经池化的 [B,1024]/[B,768] 当作序列。
        if protein_tokens.ndim != 3 or protein_tokens.shape[-1] != 1024:
            raise ValueError("protein_tokens must have shape [B, L, 1024]")
        if molt5_tokens.ndim != 3 or molt5_tokens.shape[-1] != 768:
            raise ValueError("molt5_tokens must have shape [B, M, 768]")
        batch, protein_length, _ = protein_tokens.shape
        if batch == 0 or molt5_tokens.shape[0] != batch or maccs.shape != (batch, 167):
            raise ValueError("batch dimensions must agree; maccs must have shape [B, 167]")
        for name, tensor in (("protein_tokens", protein_tokens),
                             ("molt5_tokens", molt5_tokens), ("maccs", maccs)):
            if not tensor.is_floating_point():
                raise TypeError(name + " must be floating point")
            if tensor.device != protein_tokens.device or tensor.dtype != protein_tokens.dtype:
                raise ValueError("all feature tensors must share device and dtype")
        _check_mask(protein_mask, protein_tokens.shape[:2], protein_tokens.device, "protein_mask")
        _check_mask(molt5_mask, molt5_tokens.shape[:2], molt5_tokens.device, "molt5_mask")

        # 先清零 padding，防止 NaN padding 经过 Linear 后污染有效输出。
        # masked_fill 而非乘 mask：NaN * 0 仍然是 NaN。
        raw_protein = protein_tokens.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        clean_molt5 = molt5_tokens.masked_fill(~molt5_mask.unsqueeze(-1), 0.0)
        for name, tensor in (("protein_tokens", raw_protein),
                             ("molt5_tokens", clean_molt5), ("maccs", maccs)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(name + " contains non-finite valid features")

        # 2. 构造底物 memory。cat(dim=1) 拼接的是 token 位置，不是特征维。
        # MolT5 的 M 个 token 保留；MACCS 整体只增加 1 个全局 token。
        smiles_memory = self.molt5_projection(clean_molt5)   # [B, M, 1024]
        fingerprint_memory = self.maccs_projection(maccs).unsqueeze(1)  # [B, 1, 1024]
        memory = torch.cat([smiles_memory, fingerprint_memory], dim=1)  # [B, M+1, 1024]
        maccs_valid = torch.ones((batch, 1), dtype=torch.bool, device=maccs.device)
        memory_mask = torch.cat([molt5_mask, maccs_valid], dim=1)

        # 3. 蛋白作 Q，底物作 K/V；每个残基都有独立的查询向量。
        q = self._split_heads(self.query_projection(raw_protein))  # [B,H,L,d]
        k = self._split_heads(self.key_projection(memory))        # [B,H,M+1,d]
        v = self._split_heads(self.value_projection(memory))      # [B,H,M+1,d]

        # 4. 每个蛋白位置对底物 token 归一化，绝不能沿蛋白 L 做 softmax。
        # scores/weights: [B,H,L,M+1]。每个样本在自己的 memory 内计算。
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(~memory_mask[:, None, None, :], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        # padding Query 不应显示为有意义的注意力，其整行权重设为0。
        weights = weights.masked_fill(~protein_mask[:, None, :, None], 0.0)

        # 5. 只聚合底物位置，保留所有蛋白位置：AV -> [B,H,L,d]。
        # 返回的诊断权重用 Dropout 之前的 weights；有效行权重和为1。
        context = torch.matmul(self.attention_dropout(weights), v)
        context = context.transpose(1, 2).contiguous().reshape(batch, protein_length, 1024)
        delta = self.output_dropout(self.output_projection(context))
        # output_projection 的 bias 可能让 padding 非零，必须再次清零。
        delta = delta.masked_fill(~protein_mask.unsqueeze(-1), 0.0)

        # 6. 同一残基位置直接相加，不广播全局向量，不提前池化 L。
        updated_protein = raw_protein + delta              # [B, L, 1024]
        return InteractionOutput(
            residues=updated_protein,
            delta=delta,
            protein_mask=protein_mask,
            substrate_mask=memory_mask,
            attention=weights if return_attention else None,
        )


class MaskedMeanKcatHead(nn.Module):
    """可选末端读出：残差完成后才池化 L；并非已冻结的正式选项。

    本模块不接收底物，没有底物绕过交互直接进入预测头的通路。
    预测头首层由原 CataPro 的1959维改为1024维，不能声称原结构未改。
    训练模式下 BatchNorm 需要 B>=2；不会暗中切换 eval 或更换归一化。
    """

    def __init__(self, *, dropout: float, linear_bias: bool):
        super().__init__()
        if not 0.0 <= dropout < 1.0:
            raise ValueError("head dropout must be in [0, 1)")
        if type(linear_bias) is not bool:
            raise TypeError("linear_bias must be bool")
        self.decoder = nn.Sequential(
            nn.Linear(1024, 256, bias=linear_bias),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            nn.ReLU(),
        )
        self.out = nn.Linear(256, 1, bias=linear_bias)

    def forward(self, residues: Tensor, protein_mask: Tensor) -> Tensor:
        if residues.ndim != 3 or residues.shape[-1] != 1024 or residues.shape[0] == 0:
            raise ValueError("residues must have shape [B, L, 1024] with B>0")
        _check_mask(protein_mask, residues.shape[:2], residues.device, "protein_mask")
        clean = residues.masked_fill(~protein_mask.unsqueeze(-1), 0.0)
        if not bool(torch.isfinite(clean).all()):
            raise ValueError("readout contains non-finite valid residues")
        if self.training and residues.shape[0] < 2:
            raise ValueError("training BatchNorm requires B>=2; choose the tail-batch policy explicitly")
        # 按每条蛋白的有效残基数求均值，padding 不进入分子或分母。
        counts = protein_mask.sum(dim=1, keepdim=True).to(residues.dtype)
        pooled = clean.sum(dim=1) / counts                 # [B, 1024]
        return self.out(self.decoder(pooled)).squeeze(-1)  # [B]，仅去掉输出的单例维


class CataProTokenRegressor(nn.Module):
    """显式组合交互层和选定的读出头；核心交互层也可以独立使用。"""

    def __init__(self, interaction: ResidueSubstrateCrossAttention, readout: MaskedMeanKcatHead):
        super().__init__()
        self.interaction = interaction
        self.readout = readout

    def forward(self, protein_tokens, protein_mask, molt5_tokens, molt5_mask, maccs,
                *, return_attention=False):
        interaction = self.interaction(
            protein_tokens, protein_mask, molt5_tokens, molt5_mask, maccs,
            return_attention=return_attention,
        )
        prediction = self.readout(interaction.residues, interaction.protein_mask)
        return {"pred": prediction, "interaction": interaction}
