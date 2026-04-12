import re
import torch
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from datasets import load_dataset


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
CLS_TOKEN = "<CLS>"

PAD_IDX = 0  # padding index must be 0 so we can mask it easily
UNK_IDX = 1
CLS_IDX = 2


# TOKENIZER
def simple_tokenize(text):
    text = text.lower()
    # Put a space before and after every punctuation character
    text = re.sub(r"([.,!?;:\"\'()\[\]{}/\\])", r" \1 ", text)
    # Collapse multiple spaces into one, then split
    tokens = text.split()
    return tokens


# VOCABULARY
class Vocabulary:
    def __init__(self, max_vocab=20000):
        self.max_vocab = max_vocab
        # token → integer
        self.token2idx = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX, CLS_TOKEN: CLS_IDX}
        # integer → token (for visualization)
        self.idx2token = {PAD_IDX: PAD_TOKEN, UNK_IDX: UNK_TOKEN, CLS_IDX: CLS_TOKEN}

    def build(self, sentences):
        counter = Counter()
        for sent in sentences:
            tokens = simple_tokenize(sent)
            counter.update(tokens)

        # Take the `max_vocab - 3` most common tokens
        # (subtract 3 because PAD, UNK, CLS already occupy indices 0,1,2)
        most_common = counter.most_common(self.max_vocab - 3)

        for token, _ in most_common:
            idx = len(self.token2idx)
            self.token2idx[token] = idx
            self.idx2token[idx] = token

        print(f"[Vocab] Built vocabulary: {len(self.token2idx):,} tokens")

    def encode(self, sentence):
        tokens = simple_tokenize(sentence)
        indices = [CLS_IDX]  # CLS goes first
        for tok in tokens:
            indices.append(self.token2idx.get(tok, UNK_IDX))
        return indices

    def __len__(self):
        return len(self.token2idx)


# PYTORCH DATASET AND DATALOADER 
class SST2Dataset(Dataset):
    def __init__(self, examples, vocab):
        self.data = []
        for ex in examples:
            ids = vocab.encode(ex["sentence"])
            self.data.append((ids, ex["label"]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return ids, label


# COLLATE FUNCTION (PADDING)
def collate_fn(batch):
    ids_list, labels = zip(*batch)

    # Find the longest sentence in this batch
    max_len = max(len(ids) for ids in ids_list)

    # Pad all sequences to max_len with PAD_IDX (= 0)
    padded = []
    lengths = []
    for ids in ids_list:
        lengths.append(len(ids))
        pad_len = max_len - len(ids)
        padded.append(ids + [PAD_IDX] * pad_len)

    input_ids = torch.tensor(padded, dtype=torch.long)   # [B, T]
    lengths   = torch.tensor(lengths, dtype=torch.long)   # [B]
    labels    = torch.tensor(labels,  dtype=torch.long)   # [B]

    return input_ids, lengths, labels


# MAIN LOADER FUNCTION
def get_dataloaders(batch_size=32, max_vocab=20000):
    print("[Data] Loading SST-2 from HuggingFace datasets...")
    raw = load_dataset("glue", "sst2")

    train_examples = [{"sentence": ex["sentence"], "label": ex["label"]}
                      for ex in raw["train"]]
    val_examples   = [{"sentence": ex["sentence"], "label": ex["label"]}
                      for ex in raw["validation"]]

    print(f"[Data] Train: {len(train_examples):,}  |  Val: {len(val_examples):,}")

    # Build vocabulary from training sentences ONLY
    vocab = Vocabulary(max_vocab=max_vocab)
    vocab.build([ex["sentence"] for ex in train_examples])

    # Create Dataset objects
    train_dataset = SST2Dataset(train_examples, vocab)
    val_dataset   = SST2Dataset(val_examples,   vocab)

    # DataLoader wraps the Dataset and handles batching + shuffling
    # shuffle=True for training (randomizes order each epoch → better generalization)
    # shuffle=False for validation (order doesn't matter, we just want accuracy)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    return train_loader, val_loader, vocab


# QUICK TEST
if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders(batch_size=4)
    print(f"\nVocab size: {len(vocab)}")

    # Inspect one batch
    for ids, lengths, labels in train_loader:
        print(f"\nBatch input_ids shape : {ids.shape}")   # [4, max_len]
        print(f"Batch lengths         : {lengths}")
        print(f"Batch labels          : {labels}")
        # Decode first example back to tokens
        tokens = [vocab.idx2token[i.item()] for i in ids[0]]
        print(f"First example tokens  : {tokens}")
        break
