import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class LMConfig:
    vocab_size: int
    d_model: int = 1024
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 1024
    swa_window: int = 256

def build_rope_cache(max_seq_len: int, d_head: int, base: float = 10000.0):
    inv_freq = 1.0 / (base ** (2 * torch.arange(0, d_head/2).float() / d_head))
    t = torch.arange(max_seq_len)
    freqs = t.unsqueeze(1) @ inv_freq.unsqueeze(0)
    angles = torch.cat([freqs, freqs], dim=-1)
    return angles.cos(), angles.sin()

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)

class SWASelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, swa_window):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = int(d_model / n_heads)
        self.window_size = swa_window

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        swa_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        swa_mask = torch.triu(swa_mask, diagonal=-self.window_size + 1)
        self.register_buffer("swa_mask", swa_mask, persistent=False)

        cos, sin = build_rope_cache(max_seq_len, self.d_head)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x) # [batch_size, seq_len, d_model]
        k = self.k_proj(x) # [batch_size, seq_len, d_model]
        v = self.v_proj(x) # [batch_size, seq_len, d_model]

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]

        q = apply_rope(q, self.rope_cos[:seq_len], self.rope_sin[:seq_len])
        k = apply_rope(k, self.rope_cos[:seq_len], self.rope_sin[:seq_len])

        attention_scores = (q @ k.transpose(2, 3))/math.sqrt(self.d_head) # [batch_size, n_head, seq_len, seq_len]
        mask = self.swa_mask[:seq_len, :seq_len]
        attention_scores = attention_scores.masked_fill(~mask, float("-inf"))
        attention_scores = torch.softmax(attention_scores, dim=-1) # [batch_size, n_head, seq_len, seq_len]
        
        output = attention_scores @ v # [batch_size, n_head, seq_len, d_head]
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model) # [batch_size, seq_len, d_model]
        output = self.out_proj(output) # [batch_size, seq_len, d_model]

        return output

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.RMSNorm(cfg.d_model)
        self.attn = SWASelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.swa_window)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class ThirdDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.RMSNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) #[batch_size, seq_len, d_model]
        for block in self.blocks:
            x = block(x) #[batch_size, seq_len, d_model]
        x = self.ln_f(x) #[batch_size, seq_len, d_model]
        x = self.head(x) #[batch_size, seq_len, vocab_size]
        return x