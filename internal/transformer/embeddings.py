import torch
import torch.nn as nn
import math


class TokenEmbedding(nn.Module):
    """
    词元嵌入层

    将输入的词元索引转换为密集向量表示。
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        初始化词元嵌入层

        参数:
        vocab_size: 词汇表大小
        d_model: 嵌入维度
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        词元嵌入的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len)

        返回:
        嵌入后的张量，形状为 (batch_size, seq_len, d_model)
        """
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """
    位置编码

    为序列中的每个位置添加位置信息。
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        初始化位置编码

        参数:
        d_model: 模型的维度
        max_seq_length: 最大序列长度
        dropout: dropout 比率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_length, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        位置编码的前向传播

        参数:
        x: 输入张量，形状为 (seq_len, batch_size, d_model)

        返回:
        添加位置编码后的张量，形状与输入相同
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TransformerEmbedding(nn.Module):
    """
    Transformer 嵌入层

    结合了词元嵌入和位置编码。
    """

    def __init__(self, vocab_size: int, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        初始化 Transformer 嵌入层

        参数:
        vocab_size: 词汇表大小
        d_model: 模型的维度
        max_seq_length: 最大序列长度
        dropout: dropout 比率
        """
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Transformer 嵌入的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len)

        返回:
        嵌入后的张量，形状为 (batch_size, seq_len, d_model)
        """
        return self.positional_encoding(self.token_embedding(x))


class LearnedPositionalEncoding(nn.Module):
    """
    可学习的位置编码

    不同于固定的正弦位置编码，这个编码可以通过训练学习。
    """

    def __init__(self, d_model: int, max_seq_length: int = 5000, dropout: float = 0.1):
        """
        初始化可学习的位置编码

        参数:
        d_model: 模型的维度
        max_seq_length: 最大序列长度
        dropout: dropout 比率
        """
        super().__init__()
        self.embedding = nn.Embedding(max_seq_length, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        可学习位置编码的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        添加位置编码后的张量，形状与输入相同
        """
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).expand(x.size(0), seq_len).contiguous()
        return self.dropout(x + self.embedding(positions))
