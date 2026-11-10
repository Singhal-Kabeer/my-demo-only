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

def sinusoidal_positional_encoding(max_len, d_model):
    PE = torch.zeros(max_len, d_model)

    for i in range(max_len):
        for j in range(d_model):
            if j%2==0:
                PE[i, j] = math.sin(i / (10000**(j/d_model)))
            else:
                PE[i, j] = math.cos(i / (10000**((j-1)/d_model)))

    return PE

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = int(d_model / n_heads)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        causal_mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("causal_mask", causal_mask, persistent=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x) # [batch_size, seq_len, d_model]
        k = self.k_proj(x) # [batch_size, seq_len, d_model]
        v = self.v_proj(x) # [batch_size, seq_len, d_model]

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2) # [batch_size, seq_len, n_head, d_head] to [batch_size, n_head, seq_len, d_head]

        attention_scores = (q @ k.transpose(2, 3))/math.sqrt(self.d_head) # [batch_size, n_head, seq_len, seq_len]
        mask = self.causal_mask[:seq_len, :seq_len]
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
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
        self.register_buffer("pos_emb", pe, persistent=False)

        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb[:T] #[batch_size, seq_len, d_model]
        for block in self.blocks:
            x = block(x) #[batch_size, seq_len, d_model]
        x = self.ln_f(x) #[batch_size, seq_len, d_model]
        x = self.head(x) #[batch_size, seq_len, vocab_size]
        return x