import math
import torch
import torch.nn as nn
from dataclasses import dataclass

@dataclass
class EncDecConfig:
    vocab_size: int
    d_model: int = 960
    n_layers: int = 4
    n_heads: int = 8
    d_ff: int = 1664
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

class BidirectionalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = int(d_model / n_heads)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attention = q @ k.transpose(2, 3) / math.sqrt(self.d_head)
        attention = torch.softmax(attention, dim=3)

        output = attention @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(output)

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
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attention = q @ k.transpose(2, 3) / math.sqrt(self.d_head)
        mask = self.causal_mask[:seq_len, :seq_len]
        attention = attention.masked_fill(~mask, float("-inf"))
        attention = torch.softmax(attention, dim=3)

        output = attention @ v
        output = output.transpose(1, 2).reshape(batch_size, seq_len, d_model)
        return self.out_proj(output)

class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()

        self.n_heads = n_heads
        self.d_head = int(d_model / n_heads)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x_dec, enc_out):
        batch_size, dec_seq_len, d_model = x_dec.shape
        batch_size, enc_seq_len, d_model = enc_out.shape

        q = self.q_proj(x_dec)
        k = self.k_proj(enc_out)
        v = self.v_proj(enc_out)

        q = q.reshape(batch_size, dec_seq_len, self.n_heads, self.d_head).transpose(1, 2)
        k = k.reshape(batch_size, enc_seq_len, self.n_heads, self.d_head).transpose(1, 2)
        v = v.reshape(batch_size, enc_seq_len, self.n_heads, self.d_head).transpose(1, 2)

        attention = q @ k.transpose(2, 3) / math.sqrt(self.d_head)
        attention = torch.softmax(attention, dim=3)

        output = attention @ v
        output = output.transpose(1, 2).reshape(batch_size, dec_seq_len, d_model)
        return self.out_proj(output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.net(x)

class EncoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = BidirectionalSelfAttention(cfg.d_model, cfg.n_heads)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x

class DecoderBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn1 = CausalSelfAttention(cfg.d_model, cfg.n_heads, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.attn2 = CrossAttention(cfg.d_model, cfg.n_heads)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff)
        self.ln3 = nn.LayerNorm(cfg.d_model)

    def forward(self, x, enc_out):
        x = x + self.attn1(self.ln1(x))
        x = x + self.attn2(self.ln2(x), enc_out)
        x = x + self.ffn(self.ln3(x))
        return x

class EncoderDecoderLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)

        pe = sinusoidal_positional_encoding(cfg.max_seq_len, cfg.d_model)
        self.register_buffer("pos_emb", pe, persistent=False)

        self.encoder_blocks = nn.ModuleList([EncoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.encoder_ln = nn.LayerNorm(cfg.d_model)

        self.decoder_blocks = nn.ModuleList([DecoderBlock(cfg) for _ in range(cfg.n_layers)])
        self.decoder_ln = nn.LayerNorm(cfg.d_model)

        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.head.weight = self.tok_emb.weight

    def encode(self, enc_idx):
        batch_size, seq_len = enc_idx.shape
        x = self.tok_emb(enc_idx) + self.pos_emb[:seq_len]

        for block in self.encoder_blocks:
            x = block(x)

        encoder_out = self.encoder_ln(x)
        return encoder_out

    def decode(self, dec_idx, encoder_out):
        batch_size, seq_len = dec_idx.shape
        x = self.tok_emb(dec_idx) + self.pos_emb[:seq_len]

        for block in self.decoder_blocks:
            x = block(x, encoder_out)

        x = self.decoder_ln(x)
        logits = self.head(x)
        return logits

    def forward(self, enc_idx, dec_idx):
        encoder_out = self.encode(enc_idx)
        logits = self.decode(dec_idx, encoder_out)
        return logits
