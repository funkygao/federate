import torch
import torch.nn as nn
import torch.nn.functional as F


class FeedForwardBlock(nn.Module):
    """
    前馈神经网络块 MLP

    包含两个线性变换/全连接层，中间有一个ReLU激活函数。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前馈神经网络的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        输出张量，形状与输入相同
        """
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


class LayerNormalization(nn.Module):
    """
    层归一化

    用于归一化每个位置的特征，使其均值为0，方差为1。
    """

    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        层归一化的前向传播

        参数:
        x: 输入张量，形状为 (..., features)

        返回:
        归一化后的张量，形状与输入相同
        """
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class PositionWiseFeedForward(nn.Module):
    """
    位置前馈网络

    这是 FeedForwardBlock 的一个变体，包含了层归一化和残差连接。
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.feed_forward = FeedForwardBlock(d_model, d_ff, dropout)
        self.layer_norm = LayerNormalization(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置前馈网络的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        输出张量，形状与输入相同
        """
        return self.layer_norm(x + self.dropout(self.feed_forward(x)))


class ResidualConnection(nn.Module):
    """
    残差连接

    实现了 Transformer 中的 "Add & Norm" 步骤。
    """

    def __init__(self, features: int, dropout: float = 0.1):
        super().__init__()
        self.layer_norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, sublayer: callable) -> torch.Tensor:
        """
        残差连接的前向传播

        参数:
        x: 输入张量
        sublayer: 一个可调用对象，表示要应用的子层

        返回:
        应用残差连接和层归一化后的张量
        """
        return self.layer_norm(x + self.dropout(sublayer(x)))
