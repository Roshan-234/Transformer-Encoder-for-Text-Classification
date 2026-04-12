import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_CONFIG = {
    "batch_size":  32,
    "max_vocab":   20000,
    "max_seq_len": 256,
}

TRANSFORMER_CONFIG = {
    # Architecture
    "d_model":         256,
    "num_heads":       4,
    "num_layers":      4,
    "d_ff":            1024,
    "max_len":         256,
    "dropout":         0.1,

    # Training
    "warmup_steps":    400,
    "epochs":          25,
    "weight_decay":    1e-4,
    "label_smoothing": 0.1,

    # Anti-overfit — dual early stopping
    "patience":        6,     # val accuracy patience
    "gap_patience":    4,     # gap patience (more forgiving)
    "max_gap":         0.12,  # 12% max train-val gap

    # SWA
    "swa_start":       8,     # start averaging after epoch 8
    "swa_decay":       0.99,

    "device": DEVICE,
}

BILSTM_CONFIG = {
    # Architecture
    "embed_dim":       128,
    "hidden_dim":      256,
    "num_layers":      2,
    "dropout":         0.5,

    # Training
    "lr":              1e-3,
    "lr_min":          1e-5,   # CosineAnnealing floor
    "epochs":          20,
    "weight_decay":    5e-4,
    "label_smoothing": 0.05,

    # Anti-overfit — tighter dual early stopping
    "patience":        3,     # val accuracy patience (was 5)
    "gap_patience":    3,     # gap patience — new
    "max_gap":         0.10,  # 10% max gap — tighter than Transformer

    # SWA
    "swa_start":       5,     # start averaging after epoch 5
    "swa_decay":       0.99,

    "device": DEVICE,
}

TFIDF_CONFIG = {
    "max_features": 20000,
    "ngram_range":  (1, 2),
    "C":            1.0,
}

INTERP_CONFIG = {
    "n_viz_sentences":       5,
    "n_saliency_ex":         20,
    "head_analysis_batches": 20,
    "probe_split":           0.8,
}

PATHS = {
    "checkpoint_dir":   "checkpoints",
    "output_dir":       "outputs",
    "transformer_ckpt": "checkpoints/transformer_best.pt",
    "bilstm_ckpt":      "checkpoints/bilstm_best.pt",
    "results_file":     "outputs/results.txt",
}

VIZ_SENTENCES = [
    "a stirring , funny and finally transporting re-imagining",
    "the film is not good at all",
    "it is very rarely funny or exciting",
    "a beautiful and deeply moving portrait of human resilience",
    "the worst film i have seen all year",
]