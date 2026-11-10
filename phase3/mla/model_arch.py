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
    kv_latent_dim: int = 256

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, n_heads, kv_latent_dim, max_seq_len):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = int(d_model/n_heads)
        self.kv_latent_dim = kv_latent_dim

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.kv_down_proj = nn.Linear(d_model, kv_latent_dim, bias=False)
        self.k_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.v_up_proj = nn.Linear(kv_latent_dim, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        pre_mask = torch.arange(0, max_seq_len).reshape(1, max_seq_len) - torch.arange(0, max_seq_len).reshape(max_seq_len, 1)
        pre_mask = pre_mask.float()

        alibi = torch.zeros(n_heads, max_seq_len, max_seq_len)
        for k in range(n_heads):
            weight = 1/(2**(k+1))
            alibi[k] = pre_mask*weight
            alibi[k] = alibi[k].masked_fill(alibi[k]>0, float("-inf"))
            
        self.register_buffer("alibi", alibi, persistent=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        q = self.q_proj(x)
        
        kv = self.kv_down_proj(x)
        k = self.k_up_proj(kv)
        v = self.v_up_proj(kv)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        current_alibi = self.alibi[:, :seq_len, :seq_len].unsqueeze(0) # to add batch_size dimension
        attention_scores = (q @ k.transpose(2, 3))/math.sqrt(self.d_head) + current_alibi # [batch_size, n_head, seq_len, seq_len]
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
        self.attn = MultiHeadLatentAttention(cfg.d_model, cfg.n_heads, cfg.kv_latent_dim, cfg.max_seq_len)
        self.ln2 = nn.RMSNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class FourthDecoderLM(nn.Module):
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
