import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class BiLSTMClassifier(nn.Module):

    def __init__(self, vocab_size, embed_dim=128, hidden_dim=256,
                 num_layers=2, dropout=0.3, pad_idx=0):
        super().__init__()

        # EMBEDDING LAYER
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )

        # BILSTM
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        # Dropout for regularization after pooling
        self.dropout = nn.Dropout(dropout)

        # CLASSIFIER HEAD
        self.classifier = nn.Linear(2 * hidden_dim, 2)

    def forward(self, input_ids, lengths):
        # Step 1: Embed tokens
        embedded = self.dropout(self.embedding(input_ids))

        # Step 2: PackedSequence
        packed = pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )

        # Step 3: Run BiLSTM
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        # Step 4: Mean pooling over the TRUE sequence (ignore padding)
        batch_size, seq_len, hidden = output.shape
        mask = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        # mask: [1, seq_len], lengths: [batch, 1]
        mask = mask < lengths.unsqueeze(1)  
        mask = mask.unsqueeze(-1).float()   

        # Zero out padding positions, sum, divide by length
        summed  = (output * mask).sum(dim=1)           
        lengths_f = lengths.unsqueeze(1).float().to(output.device)
        pooled  = summed / lengths_f                  

        # Step 5: Dropout + classify
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)              

        return logits
