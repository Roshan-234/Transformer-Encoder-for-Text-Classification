import matplotlib.pyplot as plt
import numpy as np
import os


def plot_training_curves(history, model_name, save_dir="outputs"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train", color="steelblue")
    ax1.plot(epochs, history["val_loss"],   label="Val",   color="darkorange")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.set_title(f"{model_name} — Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], label="Train", color="steelblue")
    ax2.plot(epochs, history["val_acc"],   label="Val",   color="darkorange")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title(f"{model_name} — Accuracy")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"{model_name.lower().replace(' ','_')}_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Training curves saved → {path}")


def plot_model_comparison(results_dict, save_dir="outputs"):
    models = list(results_dict.keys())
    accs   = [results_dict[m] for m in models]
    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(models, accs, color=colors, edgecolor="black", linewidth=0.8,
                  width=0.5)

    # Label each bar with its value
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    ax.set_ylabel("Validation Accuracy (SST-2)", fontsize=12)
    ax.set_title("Model Comparison: SST-2 Binary Sentiment\n(Research Question 1)",
                 fontsize=12)
    ax.set_ylim(0.5, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Model comparison plot saved → {path}")


def plot_tfidf_vs_attention(tfidf_feature_names, tfidf_coefs, attn_tokens, attn_scores, save_dir="outputs", n=15):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # LEFT: TF-IDF Feature Weights
    # Show top-n positive and top-n negative features
    top_pos = np.argsort(tfidf_coefs)[-n:]
    top_neg = np.argsort(tfidf_coefs)[:n]
    idx     = np.concatenate([top_neg, top_pos])
    names   = tfidf_feature_names[idx]
    coefs   = tfidf_coefs[idx]

    colors = ["#d73027" if c > 0 else "#4575b4" for c in coefs]
    y = np.arange(len(idx))
    ax1.barh(y, coefs, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=8)
    ax1.axvline(x=0, color="black", linewidth=0.8)
    ax1.set_xlabel("LR Coefficient\n(red=positive sentiment, blue=negative)")
    ax1.set_title(f"TF-IDF + LR: Top Feature Weights\n(Classical word importance)")

    # RIGHT: Attention Weights
    x = np.arange(len(attn_tokens))
    colors_attn = plt.cm.YlOrRd(attn_scores / (attn_scores.max() + 1e-9))
    ax2.bar(x, attn_scores, color=colors_attn, edgecolor="black", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(attn_tokens, rotation=45, ha="right", fontsize=9)
    ax2.set_ylabel("Average Attention Weight")
    ax2.set_title("Transformer: Avg Attention per Token\n(Neural word importance)")

    plt.suptitle("RQ1: TF-IDF Feature Weights vs. Learned Attention", fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "tfidf_vs_attention.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"TF-IDF vs attention comparison saved → {path}")
