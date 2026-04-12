import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# COMPONENT 1: SINUSOIDAL POSITIONAL ENCODING
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create the positional encoding matrix: shape [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[:pe[:, 1::2].shape[1]])

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # x.size(1) = current sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

# COMPONENT 2: SCALED DOT-PRODUCT ATTENTION
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)  # key dimension

    # Step 1: Compute raw scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: Apply padding mask
    if mask is not None:
        scores = scores.masked_fill(mask, -1e9)

    # Step 3: Softmax over the key dimension (last dim)
    attn_weights = F.softmax(scores, dim=-1)  # [batch, heads, seq_q, seq_k]

    # Step 4: Weighted sum of values
    output = torch.matmul(attn_weights, V)

    return output, attn_weights

# COMPONENT 3: MULTI-HEAD ATTENTION
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads  # dimension per head

        self.W_q = nn.Linear(d_model, d_model, bias=False)  # Query projection
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # Key projection
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # Value projection
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # Output projection

        self.dropout = nn.Dropout(dropout)


    def forward(self, x, mask=None):
        batch, seq_len, _ = x.shape

        # PROJECT to Q, K, V 
        # [batch, seq, d_model] → [batch, seq, d_model]
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # SPLIT INTO HEADS
        Q = Q.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        # Q, K, V: [batch, heads, seq, d_k]

        # SCALED DOT-PRODUCT ATTENTION
        attn_output, attn_weights = scaled_dot_product_attention(Q, K, V, mask)

        # Save weights for interpretability (Clark et al. visualization)
        self.attention_weights = attn_weights.detach()

        # CONCATENATE HEADS
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch, seq_len, self.d_model)

        # OUTPUT PROJECTION
        output = self.W_o(attn_output) 
        return output

# COMPONENT 4: POSITION-WISE FEED-FORWARD NETWORK
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# COMPONENT 5: ENCODER LAYER (one full block)
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn  = MultiHeadAttention(d_model, num_heads, dropout)
        self.ffn        = FeedForward(d_model, d_ff, dropout)
        self.norm1      = nn.LayerNorm(d_model)
        self.norm2      = nn.LayerNorm(d_model)
        self.dropout    = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # SUB-LAYER 1: Self-Attention + Residual + LayerNorm
        attn_out = self.self_attn(x, mask)       
        x        = self.norm1(x + self.dropout(attn_out))

        # SUB-LAYER 2: Feed-Forward + Residual + LayerNorm
        ffn_out  = self.ffn(x)                    
        x        = self.norm2(x + self.dropout(ffn_out))

        return x

# COMPONENT 6: FULL TRANSFORMER ENCODER + CLASSIFIER
class TransformerClassifier(nn.Module):
    def __init__(self, vocab_size, d_model=128, num_heads=4, num_layers=4,
                 d_ff=512, max_len=512, dropout=0.1, pad_idx=0):
        super().__init__()

        self.d_model    = d_model
        self.num_layers = num_layers
        self.num_heads  = num_heads

        # EMBEDDING
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_idx)

        self.embed_scale = math.sqrt(d_model)

        # POSITIONAL ENCODING
        self.pos_encoding = PositionalEncoding(d_model, max_len, dropout)

        # ENCODER LAYERS
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # CLASSIFIER HEAD
        self.classifier = nn.Linear(d_model, 2)

        # WEIGHT INITIALIZATION
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0, std=self.d_model ** -0.5)
                with torch.no_grad():
                    module.weight[0].fill_(0)

    def make_padding_mask(self, input_ids, pad_idx=0):
        mask = (input_ids == pad_idx)
        return mask.unsqueeze(1).unsqueeze(2)

    def forward(self, input_ids):
        # PADDING MASK
        mask = self.make_padding_mask(input_ids)

        # EMBED + POSITIONAL ENCODING
        x = self.embedding(input_ids) * self.embed_scale
        x = self.pos_encoding(x)

        # ENCODER STACK
        for layer in self.layers:
            x = layer(x, mask)

        # CLS POOLING
        cls_repr = x[:, 0, :]  # [batch, d_model]

        # CLASSIFY
        logits = self.classifier(cls_repr)  # [batch, 2]

        return logits

    def get_all_attention_weights(self):
        return [layer.self_attn.attention_weights for layer in self.layers]

    def get_intermediate_representations(self, input_ids):
        mask = self.make_padding_mask(input_ids)
        x = self.embedding(input_ids) * self.embed_scale
        x = self.pos_encoding(x)

        reps = []
        for layer in self.layers:
            x = layer(x, mask)
            reps.append(x.detach())

        return reps
