import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.transformer import (
    PositionalEncoding, MultiHeadAttention,
    FeedForward, TransformerClassifier
)

# VARIANT 1: NO POSITIONAL ENCODING
class TransformerNoPositionalEncoding(TransformerClassifier):
    def forward(self, input_ids):
        mask = self.make_padding_mask(input_ids)
        # Standard embedding + scale
        x = self.embedding(input_ids) * self.embed_scale
        # ← ABLATION: skip pos_encoding(x)
        x = self.pos_encoding.dropout(x)   # keep dropout for fair comparison
        for layer in self.layers:
            x = layer(x, mask)
        cls_repr = x[:, 0, :]
        return self.classifier(cls_repr)

# VARIANT 2: SINGLE-HEAD ATTENTION
class SingleHeadAttention(nn.Module):
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model          # single head uses full dimension
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.attention_weights = None

    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Single head — no reshape into multiple heads
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.squeeze(1), -1e9)
        attn   = F.softmax(scores, dim=-1)
        self.attention_weights = attn.unsqueeze(1).detach()  # [B,1,S,S]
        out    = torch.matmul(attn, V)
        return self.W_o(out)


class EncoderLayerSingleHead(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = SingleHeadAttention(d_model, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.norm1(x + self.drop(self.self_attn(x, mask)))
        x = self.norm2(x + self.drop(self.ffn(x)))
        return x


class TransformerSingleHead(TransformerClassifier):
    def __init__(self, vocab_size, d_model=256, num_layers=4,
                 d_ff=1024, max_len=256, dropout=0.1, pad_idx=0):
        # Call grandparent (nn.Module) init directly to avoid
        # TransformerClassifier's layer construction
        nn.Module.__init__(self)
        self.d_model    = d_model
        self.num_layers = num_layers
        self.num_heads  = 1           # single head

        self.embedding  = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.embed_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayerSingleHead(d_model, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)
        self._init_weights()


# VARIANT 3: NO RESIDUAL CONNECTIONS
class EncoderLayerNoResidual(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn   = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ← ABLATION: no "+ x" residual
        x = self.norm1(self.drop(self.self_attn(x, mask)))
        x = self.norm2(self.drop(self.ffn(x)))
        return x


class TransformerNoResidual(TransformerClassifier):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4,
                 d_ff=1024, max_len=256, dropout=0.1, pad_idx=0):
        nn.Module.__init__(self)
        self.d_model    = d_model
        self.num_layers = num_layers
        self.num_heads  = num_heads

        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.embed_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayerNoResidual(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)
        self._init_weights()


# VARIANT 4: NO LAYER NORMALIZATION
class EncoderLayerNoLayerNorm(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn  = FeedForward(d_model, d_ff, dropout)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # ← ABLATION: no LayerNorm
        x = x + self.drop(self.self_attn(x, mask))
        x = x + self.drop(self.ffn(x))
        return x


class TransformerNoLayerNorm(TransformerClassifier):
    def __init__(self, vocab_size, d_model=256, num_heads=4, num_layers=4,
                 d_ff=1024, max_len=256, dropout=0.1, pad_idx=0):
        nn.Module.__init__(self)
        self.d_model    = d_model
        self.num_layers = num_layers
        self.num_heads  = num_heads

        self.embedding   = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)
        self.embed_scale = math.sqrt(d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList([
            EncoderLayerNoLayerNorm(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, 2)
        self._init_weights()


# REGISTRY — maps name → constructor for easy iteration
ABLATION_VARIANTS = {
    "full":             TransformerClassifier,
    "no_pos_enc":       TransformerNoPositionalEncoding,
    "single_head":      TransformerSingleHead,
    "no_residual":      TransformerNoResidual,
    "no_layer_norm":    TransformerNoLayerNorm,
}

ABLATION_LABELS = {
    "full":          "Full Model (baseline)",
    "no_pos_enc":    "No Positional Encoding",
    "single_head":   "Single-Head Attention",
    "no_residual":   "No Residual Connections",
    "no_layer_norm": "No Layer Normalization",
}
