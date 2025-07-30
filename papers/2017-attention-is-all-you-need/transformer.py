import logging

import torch
import torch.nn as nn

logger = logging.getLogger()


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)  # Query transformation
        self.W_k = nn.Linear(d_model, d_model)  # Key transformation
        self.W_v = nn.Linear(d_model, d_model)  # Value transformation
        self.W_o = nn.Linear(d_model, d_model)  # Output transformation

    def split_heads(self, batch_size, seq_len, tensor) -> torch.tensor:
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.W_q(x)  # q.shape (batch_size, seq_len, d_model)
        k = self.W_k(x)  # k.shape (batch_size, seq_len, d_model)
        v = self.W_v(x)  # v.shape (batch_size, seq_len, d_model)

        q = self.split_heads(
            batch_size, seq_len, q
        )  # q.shape (batch_size, num_heads, seq_len, head_dim)
        k = self.split_heads(
            batch_size, seq_len, k
        )  # k.shape (batch_size, num_heads, seq_len, head_dim)
        v = self.split_heads(
            batch_size, seq_len, v
        )  # v.shape (batch_size, num_heads, seq_len, head_dim)

        qk = q @ k.transpose(-2, -1)


if __name__ == "__main__":
    logger.info("Running Transformer")
    d_model = 6
    num_heads = 2
    seq_len = 4
    mh = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(2, seq_len, d_model)
    mh.forward(x)
