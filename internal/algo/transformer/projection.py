import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionLayer(nn.Module):
    """
    投影层

    用于将解码器的输出投影到目标词汇表的大小，并应用 log softmax。
    """

    def __init__(self, d_model: int, vocab_size: int):
        """
        初始化投影层

        参数:
        d_model: 模型的维度
        vocab_size: 目标词汇表的大小
        """
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        投影层的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        输出张量，形状为 (batch_size, seq_len, vocab_size)，
        其中每个元素表示对应词的对数概率
        """
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, vocab_size)
        return F.log_softmax(self.proj(x), dim=-1)


class AdaptiveComputationTime(nn.Module):
    """
    自适应计算时间 (ACT) 层

    实现了 "Adaptive Computation Time for Neural Networks" 论文中描述的 ACT 机制。
    这允许模型动态决定在每个位置上应用多少计算步骤。
    """

    def __init__(self, d_model: int, max_steps: int = 20, threshold: float = 0.99):
        """
        初始化 ACT 层

        参数:
        d_model: 模型的维度
        max_steps: 最大计算步骤数
        threshold: 停止计算的阈值
        """
        super().__init__()
        self.max_steps = max_steps
        self.threshold = threshold
        self.halting_probability = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ACT 层的前向传播

        参数:
        x: 输入张量，形状为 (batch_size, seq_len, d_model)

        返回:
        输出张量，形状与输入相同
        """
        batch_size, seq_len, _ = x.size()

        # 初始化
        halting_probability = torch.zeros(batch_size, seq_len, 1, device=x.device)
        remainders = torch.zeros(batch_size, seq_len, 1, device=x.device)
        n_updates = torch.zeros(batch_size, seq_len, 1, device=x.device)
        previous_state = torch.zeros_like(x)

        for _ in range(self.max_steps):
            # 计算当前步骤的停止概率
            p = torch.sigmoid(self.halting_probability(x))

            # 更新累积概率和剩余值
            still_running = (halting_probability < 1.0).float()
            new_halted = (halting_probability + p * still_running > self.threshold).float() * still_running
            halting_probability += p * still_running
            remainders += new_halted * (1 - halting_probability)
            halting_probability += new_halted * remainders

            # 更新状态
            update_weights = p * still_running + new_halted * remainders
            previous_state = (1 - update_weights) * previous_state + update_weights * x

            # 更新计数器
            n_updates += still_running

            # 如果所有位置都已停止，则提前退出
            if not still_running.any():
                break

        # 计算 ponder time
        ponder_time = n_updates + remainders

        return previous_state, ponder_time


class LabelSmoothing(nn.Module):
    """
    标签平滑

    实现了标签平滑技术，可以提高模型的泛化能力。
    """

    def __init__(self, vocab_size: int, padding_idx: int, smoothing: float = 0.1):
        """
        初始化标签平滑层

        参数:
        vocab_size: 词汇表大小
        padding_idx: 填充标记的索引
        smoothing: 平滑参数
        """
        super().__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.vocab_size = vocab_size

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        标签平滑的前向传播

        参数:
        x: 模型的输出，形状为 (batch_size, seq_len, vocab_size)
        target: 目标标签，形状为 (batch_size, seq_len)

        返回:
        平滑后的损失
        """
        x = x.view(-1, x.size(-1))
        target = target.view(-1)

        with torch.no_grad():
            true_dist = x.clone()
            true_dist.fill_(self.smoothing / (self.vocab_size - 2))
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            true_dist[:, self.padding_idx] = 0
            mask = torch.nonzero(target == self.padding_idx)
            if mask.dim() > 0:
                true_dist.index_fill_(0, mask.squeeze(), 0.0)

        return self.criterion(x, true_dist)
