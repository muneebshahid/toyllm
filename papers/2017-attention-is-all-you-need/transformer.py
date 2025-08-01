import logging

import torch
import torch.nn as nn

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format="%(levelname)s - %(message)s")


def generate_padding_mask(seq, pad_idx=0):
    """Generate padding mask for sequences.

    Args:
        seq: Input sequence tensor of shape (batch_size, seq_len)
        pad_idx: Padding token index (default: 0)

    Returns:
        Mask of shape (batch_size, 1, 1, seq_len) where 1 = valid, 0 = padding
    """
    return (seq != pad_idx).unsqueeze(1).unsqueeze(2)


def generate_causal_mask(seq_len, device=None):
    """Generate causal mask to prevent attending to future positions.

    Args:
        seq_len: Sequence length
        device: Device to create mask on

    Returns:
        Mask of shape (1, 1, seq_len, seq_len) where 1 = can attend, 0 = cannot
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)


def generate_decoder_mask(tgt_seq, pad_idx=0):
    """Generate combined causal and padding mask for decoder.

    Args:
        tgt_seq: Target sequence tensor of shape (batch_size, seq_len)
        pad_idx: Padding token index (default: 0)

    Returns:
        Combined mask of shape (batch_size, 1, seq_len, seq_len)
    """
    seq_len = tgt_seq.size(1)

    # Padding mask: (batch_size, 1, 1, seq_len)
    padding_mask = generate_padding_mask(tgt_seq, pad_idx)

    # Causal mask: (1, 1, seq_len, seq_len)
    causal_mask = generate_causal_mask(seq_len, tgt_seq.device)

    # Combine masks: both must be 1 for position to be valid
    # Broadcasting will handle the shape differences
    combined_mask = padding_mask * causal_mask

    return combined_mask


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
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask=None
    ) -> torch.tensor:
        """
        input shape q, k, v: (batch_size, num_heads, seq_len, head_dim)
        output shape: (batch_size, num_heads, seq_len, head_dim)
        """
        # qk shape (batch_size, num_heads, seq_len, seq_len)
        qk = (q @ k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.head_dim, dtype=torch.float32)
        )

        if mask is not None:
            # Expand mask for num_heads dimension if needed
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)
            qk = qk.masked_fill(mask == 0, float("-inf"))

        attn_w = torch.softmax(qk, dim=-1)
        return attn_w @ v

    def forward(
        self, q: torch.tensor, k: torch.tensor, v: torch.tensor, mask=None
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

        attn_output = self._scaled_dot_product(q, k, v, mask)

        # combined_heads.shape (batch_size, num_heads, d_model)
        combined_heads = self._combine_heads(attn_output)

        y = self.W_o(combined_heads)
        return y


class Feedforward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x):
        """
        input dimension x: (batch_size, seq_len, d_model)
        output dimension: (batch_size, seq_len, d_model)
        """
        return self.net(x)


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
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        input dimension x: (batch_size, seq_len, d_model)
        output dimension: (batch_size, seq_len, d_model)
        """
        return x + self.pe[:, : x.size(1)]


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, num_heads)
        self.ff = Feedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        input dimension x: (batch_size, seq_len, d_model)
        input dimension mask: (batch_size, 1, 1, seq_len) or None
        output dimension: (batch_size, seq_len, d_model)
        """
        x = self.norm1(x + self.dropout(self.attn(x, x, x, mask)))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = Feedforward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        """
        input dimension x: (batch_size, tgt_seq_len, d_model)
        input dimension enc_output: (batch_size, src_seq_len, d_model)
        input dimension tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) or None
        input dimension src_mask: (batch_size, 1, 1, src_seq_len) or None
        output dimension: (batch_size, tgt_seq_len, d_model)
        """
        x = self.norm1(x + self.dropout(self.self_attn(x, x, x, tgt_mask)))
        x = self.norm2(
            x + self.dropout(self.cross_attn(x, enc_output, enc_output, src_mask))
        )
        x = self.norm3(x + self.dropout(self.ff(x)))
        return x


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        num_heads: int,
        d_ff: int,
        num_layers: int,
        dropout: float,
        max_len=512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        """
        input dimension x: (batch_size, seq_len) - token indices
        input dimension mask: (batch_size, 1, 1, seq_len) or None
        output dimension: (batch_size, seq_len, d_model)
        """
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
        dropout: float,
        max_len=512,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.out_proj = nn.Linear(d_model, vocab_size)

    def forward(self, tgt, enc_out, tgt_mask=None, src_mask=None):
        """
        input dimension tgt: (batch_size, tgt_seq_len) - token indices
        input dimension enc_out: (batch_size, src_seq_len, d_model)
        input dimension tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) or None
        input dimension src_mask: (batch_size, 1, 1, src_seq_len) or None
        output dimension: (batch_size, tgt_seq_len, vocab_size)
        """
        x = self.embed(tgt)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, enc_out, tgt_mask, src_mask)
        return self.out_proj(self.norm(x))


class Transformer(nn.Module):
    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        d_ff: int,
        dropout: float = 0.0,
        max_len=512,
    ):
        super().__init__()
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len
        )
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, d_ff, num_layers, dropout, max_len
        )

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        input dimension src: (batch_size, src_seq_len) - source token indices
        input dimension tgt: (batch_size, tgt_seq_len) - target token indices
        input dimension src_mask: (batch_size, 1, 1, src_seq_len) or None
        input dimension tgt_mask: (batch_size, 1, tgt_seq_len, tgt_seq_len) or None
        output dimension: (batch_size, tgt_seq_len, tgt_vocab_size)
        """
        enc_out = self.encoder(src, src_mask)
        return self.decoder(tgt, enc_out, tgt_mask, src_mask)
