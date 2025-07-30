import logging

import torch
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s - %(message)s")


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

    def _split_heads(self, tensor) -> torch.tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape
        q = self.W_q(x)  # q.shape (batch_size, seq_len, d_model)
        k = self.W_k(x)  # k.shape (batch_size, seq_len, d_model)
        v = self.W_v(x)  # v.shape (batch_size, seq_len, d_model)

        q = self._split_heads(q)  # q.shape (batch_size, num_heads, seq_len, head_dim)
        k = self._split_heads(k)  # k.shape (batch_size, num_heads, seq_len, head_dim)
        v = self._split_heads(v)  # v.shape (batch_size, num_heads, seq_len, head_dim)

        # qk.shape (batch_size, num_heads, seq_len, seq_len)
        qk = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        qk = qk.masked_fill(mask == 0, float("-inf"))
        attn_w = torch.softmax(qk, dim=-1)
        attn_o = attn_w @ v  # attn_o.shape (batch_size, num_heads, seq_len, head_dim)

        # attn_c.shape (batch_size, num_heads, d_model)
        attn_c = (
            attn_o.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

        attn_o = self.W_o(attn_c)


if __name__ == "__main__":
    logger.info("Running Transformer")
    d_model = 6
    num_heads = 2
    seq_len = 4
    mh = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(2, seq_len, d_model)
    mh.forward(x)
