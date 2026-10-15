"""A character-level, decoder-only Transformer built from basic PyTorch layers."""
from dataclasses import dataclass
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    vocab_size: int
    context: int = 64
    width: int = 96
    heads: int = 4
    layers: int = 3
    dropout: float = 0.1


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.dropout = config.dropout

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [t.view(batch, length, self.heads, width // self.heads)
                   .transpose(1, 2) for t in (q, k, v)]
        # Causal masking prevents each position from reading future characters.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0)
        return self.projection(y.transpose(1, 2).contiguous().view(batch, length, width))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.width)
        self.norm2 = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.feedforward = nn.Sequential(
            nn.Linear(config.width, 4 * config.width), nn.GELU(),
            nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm1(x))
        return x + self.feedforward(self.norm2(x))


class Hope(nn.Module):
    def __init__(self, config):
        super().__init__()
        if min(config.vocab_size, config.context, config.width, config.heads, config.layers) < 1:
            raise ValueError('Model dimensions must be positive.')
        if config.width % config.heads:
            raise ValueError('Width must be divisible by the number of heads.')
        self.config = config
        self.token = nn.Embedding(config.vocab_size, config.width)
        self.position = nn.Embedding(config.context, config.width)
        self.blocks = nn.Sequential(*[Block(config) for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.output = nn.Linear(config.width, config.vocab_size)

    def forward(self, tokens, targets=None):
        length = tokens.shape[1]
        if not 0 < length <= self.config.context:
            raise ValueError('Sequence length must be between 1 and context size.')
        x = self.token(tokens) + self.position(torch.arange(length, device=tokens.device))
        logits = self.output(self.norm(self.blocks(x)))
        loss = None if targets is None else F.cross_entropy(
            logits.reshape(-1, self.config.vocab_size), targets.reshape(-1))
        return logits, loss

    @torch.no_grad()
    def generate(self, tokens, count=200, temperature=0.8, top_k=20):
        if temperature <= 0 or count < 0 or top_k < 0:
            raise ValueError('Temperature must be positive; count and top_k must be nonnegative.')
        self.eval()
        for _ in range(count):
            logits, _ = self(tokens[:, -self.config.context:])
            logits = logits[:, -1, :] / temperature
            if top_k:
                cutoff = logits.topk(min(top_k, logits.shape[-1])).values[:, -1:]
                logits = logits.masked_fill(logits < cutoff, float('-inf'))
            next_token = torch.multinomial(logits.softmax(-1), num_samples=1)
            tokens = torch.cat((tokens, next_token), dim=1)
        return tokens


# Compatibility for scripts written before the model was named Hope.
TinyLM = Hope
