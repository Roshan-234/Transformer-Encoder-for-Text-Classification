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
from models.transformer_variants import ABLATION_VARIANTS, ABLATION_LABELS


def run_ablation(train_loader, val_loader, vocab, save_dir="outputs/ablation", ckpt_dir="checkpoints/ablation"):
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)

    # Training config — identical for all variants
    cfg = {**TRANSFORMER_CONFIG}   # copy base config

    results   = {}
    histories = {}

    print("\n" + "="*65)
    print("  ABLATION STUDY — 5 variants, identical hyperparameters")
    print("="*65)

    for variant_name, ModelClass in ABLATION_VARIANTS.items():
        label = ABLATION_LABELS[variant_name]
        print(f"\n{'─'*60}")
        print(f"  Variant: {label}")
        print(f"{'─'*60}")

        # Fix random seed for reproducible initialisation
        torch.manual_seed(42)
        if DEVICE == "cuda":
            torch.cuda.manual_seed(42)

        # Build model — single_head variant uses num_heads=1 constructor
        if variant_name == "single_head":
            model = ModelClass(
                vocab_size=len(vocab),
                d_model=cfg["d_model"],
                num_layers=cfg["num_layers"],
                d_ff=cfg["d_ff"],
                max_len=cfg["max_len"],
                dropout=cfg["dropout"],
                pad_idx=0
            )
        else:
            model = ModelClass(
                vocab_size=len(vocab),
                d_model=cfg["d_model"],
                num_heads=cfg["num_heads"],
                num_layers=cfg["num_layers"],
                d_ff=cfg["d_ff"],
                max_len=cfg["max_len"],
                dropout=cfg["dropout"],
                pad_idx=0
            )

        ckpt_path = os.path.join(ckpt_dir, f"{variant_name}_best.pt")
        history   = train_model(
            model, train_loader, val_loader,
            cfg, ckpt_path, "transformer"
        )

        # Load best checkpoint and evaluate
        ckpt = torch.load(ckpt_path, map_location=DEVICE)
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(DEVICE)

        # Collect predictions + probs
        preds, probs = [], []
        model.eval()
        with torch.no_grad():
            for ids, lens, _ in val_loader:
                ids = ids.to(DEVICE)
                logits = model(ids)
                p = torch.softmax(logits, dim=-1)
                preds.extend(p.argmax(dim=-1).cpu().tolist())
                probs.extend(p[:, 1].cpu().tolist())

        from datasets import load_dataset
        raw = load_dataset("glue", "sst2")
        val_labs = [ex["label"] for ex in raw["validation"]]

        m = compute_all_metrics(val_labs, preds, probs, label)
        m["best_epoch"]  = ckpt["epoch"]
        m["stopped_at"]  = len(history["val_acc"])
        m["best_val_acc"] = max(history["val_acc"])

        results[variant_name]   = m
        histories[variant_name] = history
        print(f"  → Val Acc: {m['accuracy']:.4f}  F1: {m['f1_macro']:.4f}  MCC: {m['mcc']:.4f}")

    # Save raw results 
    results_serializable = {
        k: {kk: (vv.tolist() if hasattr(vv, 'tolist') else vv)
            for kk, vv in v.items()
            if kk != "confusion_matrix"}
        for k, v in results.items()
    }
    with open(os.path.join(save_dir, "ablation_results.json"), "w") as f:
        json.dump(results_serializable, f, indent=2)

    # Plots
    _plot_ablation_bar(results, save_dir)
    _plot_ablation_curves(histories, save_dir)
    _write_ablation_table(results, save_dir)

    return results, histories


def _plot_ablation_bar(results, save_dir):
    names  = list(ABLATION_LABELS.values())
    keys   = list(ABLATION_VARIANTS.keys())
    accs   = [results[k]["accuracy"] for k in keys]
    f1s    = [results[k]["f1_macro"] for k in keys]
    full_acc = results["full"]["accuracy"]

    colors = ["#2ECC71" if k == "full" else "#E74C3C" for k in keys]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, vals, title, ylabel in zip(
        axes,
        [accs, f1s],
        ["Validation Accuracy", "F1 Macro"],
        ["Accuracy", "F1 Macro"]
    ):
        bars = ax.barh(names, vals, color=colors,
                       edgecolor="black", linewidth=0.6, height=0.55)
        ax.axvline(x=full_acc, color="gray", linestyle="--",
                   linewidth=1.2, alpha=0.7, label="Full model baseline")
        for bar, val, key in zip(bars, vals, keys):
            delta = val - full_acc
            delta_str = f"{delta:+.3f}" if key != "full" else "baseline"
            ax.text(max(vals) + 0.002, bar.get_y() + bar.get_height()/2,
                    f"{val:.3f} ({delta_str})",
                    va="center", fontsize=9)
        ax.set_xlabel(ylabel, fontsize=11)
        ax.set_xlim(min(vals) - 0.05, max(vals) + 0.12)
        ax.set_title(f"Ablation Study — {title}", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, axis="x", alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_bar.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ablation bar chart → {path}")


def _plot_ablation_curves(histories, save_dir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    colors = {"full": "#2ECC71",
              "no_pos_enc": "#E74C3C",
              "single_head": "#3498DB",
              "no_residual": "#E67E22",
              "no_layer_norm": "#9B59B6"}
    styles = {"full": "-",
              "no_pos_enc": "--",
              "single_head": "-.",
              "no_residual": ":",
              "no_layer_norm": (0, (3, 1, 1, 1))}

    for key, history in histories.items():
        label = ABLATION_LABELS[key]
        epochs = range(1, len(history["val_acc"]) + 1)
        ax1.plot(epochs, history["val_acc"],
                 label=label, color=colors[key],
                 linestyle=styles[key], linewidth=2)
        ax2.plot(epochs, history["val_loss"],
                 label=label, color=colors[key],
                 linestyle=styles[key], linewidth=2)

    for ax, title in zip([ax1, ax2],
                         ["Validation Accuracy", "Validation Loss"]):
        ax.set_xlabel("Epoch", fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(f"Ablation Study — {title}", fontsize=12)
        ax.legend(fontsize=8, loc="lower right" if "Acc" in title else "upper right")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "ablation_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Ablation curves → {path}")


def _write_ablation_table(results, save_dir):
    lines = [
        "ABLATION STUDY RESULTS",
        "="*75,
        f"{'Variant':<32} {'Acc':>7} {'F1 Mac':>8} {'MCC':>7} {'AUC':>7} {'Best Ep':>8}",
        "─"*75,
    ]
    full_acc = results["full"]["accuracy"]
    for k in ABLATION_VARIANTS:
        m = results[k]
        delta = m["accuracy"] - full_acc
        delta_str = f"({delta:+.3f})" if k != "full" else "(baseline)"
        auc = f"{m['auc_roc']:.4f}" if m['auc_roc'] else "  N/A "
        lines.append(
            f"{ABLATION_LABELS[k]:<32} {m['accuracy']:>7.4f} "
            f"{m['f1_macro']:>8.4f} {m['mcc']:>7.4f} {auc:>7} "
            f"{m['best_epoch']:>5} {delta_str:>8}"
        )
    lines += ["─"*75,
              "Delta = difference from full model accuracy."]
    text = "\n".join(lines)
    path = os.path.join(save_dir, "ablation_table.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"  Ablation table → {path}")
    print("\n" + text)


if __name__ == "__main__":
    train_loader, val_loader, vocab = get_dataloaders(
        batch_size=DATA_CONFIG["batch_size"],
        max_vocab=DATA_CONFIG["max_vocab"]
    )
    run_ablation(train_loader, val_loader, vocab)
