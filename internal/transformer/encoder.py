import torch
import torch.nn as nn
from attention import MultiHeadAttentionBlock
from layers import FeedForwardBlock, LayerNormalization


class EncoderBlock(nn.Module):
    """
    Transformer 编码器块

    包含多头自注意力层和前馈神经网络层。
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, num_heads, dropout)
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)

        self.norm1 = LayerNormalization(d_model)
        self.norm2 = LayerNormalization(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        编码器块的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, src_seq_len, d_model)
        mask: 源序列的掩码，形状为 (batch_size, 1, 1, src_seq_len)

        返回:
        编码器块的输出，形状为 (batch_size, src_seq_len, d_model)
        """
        # 自注意力层
        attn_output = self.self_attention(x, mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # 前馈神经网络层
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x


class Encoder(nn.Module):
    """
    Transformer 编码器

    由多个编码器块堆叠而成。
    """

    def __init__(self, num_layers: int, d_model: int, num_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.layers = nn.ModuleList([
            EncoderBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.norm = LayerNormalization(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        编码器的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, src_seq_len, d_model)
        mask: 源序列的掩码，形状为 (batch_size, 1, 1, src_seq_len)

        返回:
        编码器的输出，形状为 (batch_size, src_seq_len, d_model)
        """
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
