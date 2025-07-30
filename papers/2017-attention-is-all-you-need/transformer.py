import logging

import torch
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s - %(message)s")


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def _split_heads(self, tensor: torch.tensor) -> torch.tensor:
        batch_size, seq_len, _ = tensor.shape
        return tensor.view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

    def _combine_heads(self, tensor: torch.tensor) -> torch.tensor:
        """
        input shape: (batch_size, num_heads, seq_len, head_dim)
        output shape: (batch_size, num_heads, d_model)
        """
        batch_size, _, seq_len, _ = tensor.shape
        return (
            tensor.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        )

    def _scaled_dot_product(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        """
        input shape q, k, v: (batch_size, num_heads, seq_len, head_dim)
        output shape: (batch_size, num_heads, seq_len, head_dim)
        """
        # qk shape (batch_size, num_heads, seq_len, seq_len)
        qk = (q @ k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim))
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0)
        qk = qk.masked_fill(mask == 0, float("-inf"))
        attn_w = torch.softmax(qk, dim=-1)
        return attn_w @ v

    def forward(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor
    ) -> torch.tensor:
        """
        input shape q, k, v: (batch_size, seq_len, d_model)
        output shape y: (batch_size, seq_len, d_model)
        """
        q = self.W_q(q)  # q.shape (batch_size, seq_len, d_model)
        k = self.W_k(k)  # k.shape (batch_size, seq_len, d_model)
        v = self.W_v(v)  # v.shape (batch_size, seq_len, d_model)

        q = self._split_heads(q)  # q.shape (batch_size, num_heads, seq_len, head_dim)
        k = self._split_heads(k)  # k.shape (batch_size, num_heads, seq_len, head_dim)
        v = self._split_heads(v)  # v.shape (batch_size, num_heads, seq_len, head_dim)

        attn_output = self._scaled_dot_product(q, k, v)

        # combined_heads.shape (batch_size, num_heads, d_model)
        combined_heads = self._combine_heads(attn_output)

        y = self.W_o(combined_heads)
        return y


class PositionalEncoding(nn.Module):
    """Implements the positional encodings as described in the paper
    "Attention is All You Need" (Vaswani et al., 2017).
        p_i = sin(i / 10000^(2j/d_model)) for j even
        p_i = cos(i / 10000^(2j/d_model)) for j odd
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len=512,
    ):
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = []
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embed(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        max_len=512,
    ):
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = []
        self.norm == nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        x = self.embed(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.out_proj(self.norm(x))


class Transformer(nn.module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers
        )
        self.decoder = TransformerDecoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)


if __name__ == "__main__":
    logger.info("Running Transformer")
    d_model = 6
    num_heads = 2
    seq_len = 4
    mh = MultiHeadAttention(d_model, num_heads)
    x = torch.randn(2, seq_len, d_model)
    mh.forward(x)
