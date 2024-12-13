import torch
import torch.nn as nn
import math


class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self,
                                     Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor,
                                     mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        """
        计算缩放点积注意力

        参数:
        Q: 查询张量, 形状为 (..., seq_len_q, d_k)
        K: 键张量, 形状为 (..., seq_len_k, d_k)
        V: 值张量, 形状为 (..., seq_len_v, d_k)
        mask: 可选的掩码张量

        返回:
        注意力输出和注意力权重
        """
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        output = torch.matmul(attention_weights, V)

        return output, attention_weights

    def split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将最后一个维度分割成 (num_heads, d_k)
        """
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)

    def concat_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        将多头注意力的输出组合回原始形状
        """
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)

    def forward(self, Q: torch.Tensor, K: torch.Tensor = None, V: torch.Tensor = None,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        多头注意力机制的前向传播

        参数:
        Q: 查询张量, 形状为 (batch_size, seq_len_q, d_model)
        K: 键张量, 形状为 (batch_size, seq_len_k, d_model)，默认为 None（则使用 q），encoder场景
        V: 值张量, 形状为 (batch_size, seq_len_v, d_model)，默认为 None（则使用 q），encoder场景
        mask: 可选的掩码张量

        返回:
        注意力的输出, 形状为 (batch_size, seq_len_q, d_model)
        """
        batch_size, seq_length, _ = Q.size()

        if K is None:
            K = Q
        if V is None:
            V = Q

        # 线性变换
        Q = self.W_q(Q)
        K = self.W_k(K)
        V = self.W_v(V)

        # 分割头
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 缩放点积注意力
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # 组合多头
        output = self.concat_heads(attn_output)

        # 最后的线性层
        output = self.W_o(output)

        return output
