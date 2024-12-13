import torch
import torch.nn as nn
from embeddings import InputEmbeddings, PositionalEncoding
from encoder import Encoder
from decoder import Decoder
from projection import ProjectionLayer


class Transformer(nn.Module):

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: InputEmbeddings,
                 tgt_embed: InputEmbeddings,
                 src_pos: PositionalEncoding,
                 tgt_pos: PositionalEncoding,
                 projection_layer: ProjectionLayer) -> None:
        super().__init__()
        self.encoder = encoder  # 编码器
        self.decoder = decoder  # 解码器
        self.src_embed = src_embed  # 源语言词嵌入
        self.tgt_embed = tgt_embed  # 目标语言词嵌入
        self.src_pos = src_pos  # 源语言位置编码
        self.tgt_pos = tgt_pos  # 目标语言位置编码
        self.projection_layer = projection_layer  # 输出投影层

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        对输入序列进行编码

        参数:
        src: 源语言输入序列, 形状为 (batch_size, src_seq_len)
        src_mask: 源语言的掩码, 形状为 (batch_size, 1, 1, src_seq_len)

        返回:
        编码器的输出, 形状为 (batch_size, src_seq_len, d_model)
        """
        # 词嵌入, 将输入从词索引转换为词向量
        # 输出形状: (batch_size, src_seq_len, d_model)
        src = self.src_embed(src)

        # 添加位置编码, 形状不变
        src = self.src_pos(src)

        # 通过编码器, 形状不变
        return self.encoder(src, src_mask)

    def decode(self, encoder_output: torch.Tensor, src_mask: torch.Tensor,
               tgt: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        根据编码器输出和目标序列进行解码

        参数:
        encoder_output: 编码器的输出, 形状为 (batch_size, src_seq_len, d_model)
        src_mask: 源语言的掩码, 形状为 (batch_size, 1, 1, src_seq_len)
        tgt: 目标语言输入序列, 形状为 (batch_size, tgt_seq_len)
        tgt_mask: 目标语言的掩码, 形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)

        返回:
        解码器的输出, 形状为 (batch_size, tgt_seq_len, d_model)
        """
        # 词嵌入, 将输入从词索引转换为词向量
        # 输出形状: (batch_size, tgt_seq_len, d_model)
        tgt = self.tgt_embed(tgt)

        # 添加位置编码, 形状不变
        # 输入/输出形状: (batch_size, tgt_seq_len, d_model)
        tgt = self.tgt_pos(tgt)

        # 通过解码器, 形状不变
        # 输入/输出形状: (batch_size, tgt_seq_len, d_model)
        return self.decoder(tgt, encoder_output, src_mask, tgt_mask)

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        将解码器的输出投影到词汇表空间

        参数:
        x: 解码器输出, 形状为 (batch_size, seq_len, d_model)

        返回:
        投影后的输出, 形状为 (batch_size, seq_len, vocab_size)
        """
        return self.projection_layer(x)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor,
                src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        """
        Transformer 的前向传播

        这个方法串联了整个 Transformer 的工作流程:
        1. 编码输入序列
        2. 解码目标序列
        3. 投影到词汇表空间

        参数:
        src: 源语言输入序列, 形状为 (batch_size, src_seq_len)
        tgt: 目标语言输入序列, 形状为 (batch_size, tgt_seq_len)
        src_mask: 源语言的掩码, 形状为 (batch_size, 1, 1, src_seq_len)
        tgt_mask: 目标语言的掩码, 形状为 (batch_size, 1, tgt_seq_len, tgt_seq_len)

        返回:
        模型的输出, 表示目标语言词汇的概率分布, 形状为 (batch_size, tgt_seq_len, vocab_size)
        """
        # 编码, 输出形状: (batch_size, src_seq_len, d_model)
        encoder_output = self.encode(src, src_mask)

        # 解码, 输出形状: (batch_size, tgt_seq_len, d_model)
        decoder_output = self.decode(encoder_output, src_mask, tgt, tgt_mask)

        # 投影到词汇表空间, 输出形状: (batch_size, tgt_seq_len, vocab_size)
        return self.project(decoder_output)
