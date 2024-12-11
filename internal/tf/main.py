#encoding utf8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math

class MultiHeadAttention(nn.Module):
    def __int__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__int__()
        assert num_heads % d_model == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)




