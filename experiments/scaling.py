import os, sys, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import TRANSFORMER_CONFIG, DATA_CONFIG, DEVICE
from data.data_loader import get_dataloaders
from train import train_model, evaluate
from metrics import compute_all_metrics
from models.transformer import TransformerClassifier

# SCALING EXPERIMENT
SCALING_CONFIGS = {
    "small":  {"d_model": 64,  "d_ff": 256,  "num_heads": 2, "num_layers": 2},
    "medium": {"d_model": 128, "d_ff": 512,  "num_heads": 4, "num_layers": 4},
    "large":  {"d_model": 256, "d_ff": 1024, "num_heads": 4, "num_layers": 4},
}
SCALING_LABELS = {
    "small":  "Small  (d=64,  L=2, H=2)",
    "medium": "Medium (d=128, L=4, H=4)",
    "large":  "Large  (d=256, L=4, H=4)",
}


def run_scaling_experiment(train_loader, val_loader, vocab, save_dir="outputs/scaling", ckpt_dir="checkpoints/scaling"):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")
    val_labs = [ex["label"] for ex in raw["validation"]]

    base_cfg = {**TRANSFORMER_CONFIG}
    results  = {}

    print("\n" + "="*65)
    print("  SCALING EXPERIMENT (3 model sizes)")
    print("="*65)

    for size_name, arch in SCALING_CONFIGS.items():
        label = SCALING_LABELS[size_name]
        print(f"\n{'─'*60}")
        print(f"  Size: {label}")
        print(f"{'─'*60}")

        torch.manual_seed(42)
        cfg = {**base_cfg, **arch}

        model = TransformerClassifier(
            vocab_size=len(vocab),
            d_model=arch["d_model"],
            num_heads=arch["num_heads"],
            num_layers=arch["num_layers"],
            d_ff=arch["d_ff"],
            max_len=cfg["max_len"],
            dropout=cfg["dropout"],
            pad_idx=0
        )
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        ckpt_path = os.path.join(ckpt_dir, f"{size_name}_best.pt")
        history   = train_model(model, train_loader, val_loader,
                                cfg, ckpt_path, "transformer")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)

        preds, probs = [], []
        model.eval()
        with torch.no_grad():
            for ids, _, _ in val_loader:
                ids = ids.to(DEVICE)
                p = torch.softmax(model(ids), dim=-1)
                preds.extend(p.argmax(dim=-1).cpu().tolist())
                probs.extend(p[:, 1].cpu().tolist())

        m = compute_all_metrics(val_labs, preds, probs, label)
        m["n_params"]  = n_params
        m["best_epoch"] = ckpt["epoch"]
        results[size_name] = {"metrics": m, "history": history,
                               "n_params": n_params}
        print(f"  → Val Acc: {m['accuracy']:.4f}  Params: {n_params:,}")

    _plot_scaling(results, save_dir)
    _save_scaling_json(results, save_dir)
    return results


def _plot_scaling(results, save_dir):
    sizes  = list(SCALING_CONFIGS.keys())
    params = [results[s]["n_params"] for s in sizes]
    accs   = [results[s]["metrics"]["accuracy"] for s in sizes]
    f1s    = [results[s]["metrics"]["f1_macro"] for s in sizes]
    labels = [SCALING_LABELS[s] for s in sizes]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy vs param count (scatter + line)
    ax = axes[0]
    ax.plot([p/1e6 for p in params], accs,
            "o-", color="#2ECC71", linewidth=2, markersize=9)
    for x, y, lbl in zip(params, accs, labels):
        ax.annotate(lbl.split("(")[0].strip(),
                    (x/1e6, y), textcoords="offset points",
                    xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Model Parameters (millions)", fontsize=11)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_title("Scaling: Accuracy vs Parameter Count", fontsize=12)
    ax.grid(True, alpha=0.3)

    # Right: bar chart comparison
    ax = axes[1]
    x = np.arange(len(sizes))
    bars = ax.bar(x, accs, color=["#3498DB","#2ECC71","#E74C3C"],
                  edgecolor="black", linewidth=0.6, width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height()+0.003,
                f"{acc:.4f}", ha="center", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Validation Accuracy", fontsize=11)
    ax.set_title("Scaling: Accuracy by Model Size", fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Transformer Scaling Experiments on SST-2",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "scaling_results.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scaling plot → {path}")


def _save_scaling_json(results, save_dir):
    out = {}
    for k, v in results.items():
        m = v["metrics"]
        out[k] = {
            "n_params": v["n_params"],
            "accuracy": m["accuracy"],
            "f1_macro": m["f1_macro"],
            "mcc":      m["mcc"],
        }
    with open(os.path.join(save_dir, "scaling_results.json"), "w") as f:
        json.dump(out, f, indent=2)


# PRETRAINED EMBEDDINGS EXPERIMENT
def load_glove(glove_path, vocab, d_model):
    print(f"  Loading GloVe from {glove_path} ...")
    glove = {}
    with open(glove_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            word  = parts[0]
            vec   = np.array(parts[1:], dtype=np.float32)
            if len(vec) == d_model:
                glove[word] = vec

    vocab_size = len(vocab)
    matrix = np.random.normal(0, 0.01, (vocab_size, d_model)).astype(np.float32)
    covered = 0
    for token, idx in vocab.token2idx.items():
        if token in glove:
            matrix[idx] = glove[token]
            covered += 1

    coverage = covered / vocab_size
    print(f"  GloVe coverage: {covered}/{vocab_size} tokens ({coverage:.1%})")
    return torch.tensor(matrix), coverage


def run_pretrained_embeddings_experiment(train_loader, val_loader, vocab, glove_path="data/glove.6B.100d.txt", save_dir="outputs/scaling", ckpt_dir="checkpoints/scaling"):
    if not os.path.exists(glove_path):
        print(f"\n[SKIP] GloVe file not found at {glove_path}.")
        print("       Download glove.6B.100d.txt and place at data/glove.6B.100d.txt")
        print("       to run this experiment.")
        return None

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")
    val_labs = [ex["label"] for ex in raw["validation"]]

    d_model = 100   # match GloVe dimension
    cfg = {**TRANSFORMER_CONFIG,
           "d_model": d_model, "d_ff": d_model*4,
           "num_heads": 4}

    results = {}
    print("\n" + "="*65)
    print("  PRETRAINED EMBEDDINGS EXPERIMENT (GloVe 100d)")
    print("="*65)

    for init_name in ["random", "glove"]:
        label = f"{'GloVe init' if init_name=='glove' else 'Random init'} (d={d_model})"
        print(f"\n  Init: {label}")

        torch.manual_seed(42)
        model = TransformerClassifier(
            vocab_size=len(vocab), d_model=d_model,
            num_heads=cfg["num_heads"], num_layers=cfg["num_layers"],
            d_ff=cfg["d_ff"], max_len=cfg["max_len"],
            dropout=cfg["dropout"], pad_idx=0
        )

        if init_name == "glove":
            emb_matrix, coverage = load_glove(glove_path, vocab, d_model)
            with torch.no_grad():
                model.embedding.weight.copy_(emb_matrix)

        ckpt_path = os.path.join(ckpt_dir, f"pretrain_{init_name}_best.pt")
        history   = train_model(model, train_loader, val_loader,
                                cfg, ckpt_path, "transformer")

        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)

        preds, probs = [], []
        model.eval()
        with torch.no_grad():
            for ids, _, _ in val_loader:
                ids = ids.to(DEVICE)
                p = torch.softmax(model(ids), dim=-1)
                preds.extend(p.argmax(dim=-1).cpu().tolist())
                probs.extend(p[:, 1].cpu().tolist())

        m = compute_all_metrics(val_labs, preds, probs, label)
        results[init_name] = {"metrics": m, "history": history}
        print(f"  → Val Acc: {m['accuracy']:.4f}")

    _plot_pretrained_comparison(results, save_dir)
    return results


def _plot_pretrained_comparison(results, save_dir):
    names = ["random", "glove"]
    labels = ["Random Init", "GloVe Init"]
    accs  = [results[n]["metrics"]["accuracy"] for n in names]
    f1s   = [results[n]["metrics"]["f1_macro"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    colors = ["#95A5A6", "#3498DB"]

    for ax, vals, title in zip(axes, [accs, f1s],
                                ["Accuracy", "F1 Macro"]):
        bars = ax.bar(labels, vals, color=colors,
                      edgecolor="black", linewidth=0.7, width=0.4)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x()+bar.get_width()/2,
                    bar.get_height()+0.003,
                    f"{v:.4f}", ha="center", fontsize=11)
        delta = vals[1] - vals[0]
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"Pretrained vs Random Init — {title}\n"
                     f"(GloVe improvement: {delta:+.4f})", fontsize=11)
        ax.set_ylim(0.5, 1.0)
        ax.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Effect of GloVe Embedding Initialization on SST-2",
                 fontsize=13)
    plt.tight_layout()
    path = os.path.join(save_dir, "pretrained_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Pretrained comparison → {path}")


if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders(
        batch_size=DATA_CONFIG["batch_size"],
        max_vocab=DATA_CONFIG["max_vocab"]
    )
    run_scaling_experiment(train_loader, val_loader, vocab)
    run_pretrained_embeddings_experiment(train_loader, val_loader, vocab)
