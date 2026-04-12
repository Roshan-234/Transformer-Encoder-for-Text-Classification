import os, sys, json, torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from interpretability.saliency import compute_saliency


def compute_attention_scores(model, input_ids):
    """
    Average CLS attention across all layers and heads.
    Returns normalised per-token attention array [seq_len].
    """
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)
    all_attn = model.get_all_attention_weights()
    seq_len  = input_ids.shape[1]
    avg = np.zeros(seq_len)
    valid = 0
    for la in all_attn:
        if la is None: continue
        # [B, heads, S, S] → mean over heads → [S,S] → row 0 (CLS query)
        avg += la[0].cpu().numpy().mean(axis=0)[0, :seq_len]
        valid += 1
    if valid: avg /= valid
    # Normalise to sum to 1
    if avg.sum() > 0: avg /= avg.sum()
    return avg


def run_correlation_experiment(model, val_loader, vocab, device,n_examples=200, save_dir="outputs/attention_validation"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    num_layers = model.num_layers
    layer_corrs = {i: [] for i in range(num_layers)}
    all_corrs   = []  

    count = 0
    print(f"[Corr] Computing attention-saliency correlation "
          f"({n_examples} examples)...")

    for input_ids, lengths, labels in val_loader:
        if count >= n_examples:
            break
        for b in range(input_ids.size(0)):
            if count >= n_examples:
                break
            true_len = lengths[b].item()
            if true_len < 3:   # skip very short sentences
                continue

            single = input_ids[b:b+1].to(device)

            # Global average attention (all layers averaged)
            avg_attn = compute_attention_scores(model, single)[:true_len]

            # Gradient saliency
            sal, pred = compute_saliency(model, single)
            sal = sal[:true_len]

            # Spearman ρ — global
            if len(avg_attn) >= 3:
                rho, _ = spearmanr(avg_attn, sal)
                if not np.isnan(rho):
                    all_corrs.append(rho)

            # ── Per-layer correlation ─────────────────────────────────────
            with torch.no_grad():
                _ = model(single)
            all_attn = model.get_all_attention_weights()

            for layer_idx, la in enumerate(all_attn):
                if la is None: continue
                layer_avg = la[0].cpu().numpy().mean(axis=0)[0, :true_len]
                if layer_avg.sum() > 0:
                    layer_avg = layer_avg / layer_avg.sum()
                if len(layer_avg) >= 3:
                    rho_l, _ = spearmanr(layer_avg, sal)
                    if not np.isnan(rho_l):
                        layer_corrs[layer_idx].append(rho_l)

            count += 1

    avg_corrs = {i: float(np.mean(v)) if v else 0.0
                 for i, v in layer_corrs.items()}
    overall   = float(np.mean(all_corrs)) if all_corrs else 0.0

    print(f"  Overall mean Spearman ρ (all layers avg): {overall:.4f}")
    for i, rho in avg_corrs.items():
        print(f"  Layer {i+1}: mean ρ = {rho:.4f}  "
              f"(n={len(layer_corrs[i])})")

    # ── Plot 1: histogram of ρ values ──────────────────────────────────────
    fig, axes = plt.subplots(1, num_layers, figsize=(4*num_layers, 4),
                              sharey=True)
    if num_layers == 1: axes = [axes]
    for i, ax in enumerate(axes):
        vals = layer_corrs[i]
        ax.hist(vals, bins=20, color="#3498DB", edgecolor="black",
                linewidth=0.5, alpha=0.8)
        ax.axvline(avg_corrs[i], color="red", linestyle="--",
                   linewidth=1.5, label=f"mean={avg_corrs[i]:.3f}")
        ax.axvline(0, color="gray", linestyle=":", linewidth=1)
        ax.set_xlabel("Spearman ρ", fontsize=10)
        ax.set_title(f"Layer {i+1}", fontsize=11)
        if i == 0: ax.set_ylabel("Count", fontsize=10)
        ax.legend(fontsize=8)
    plt.suptitle("Attention-Saliency Correlation per Layer\n"
                 "(Spearman ρ: 1.0 = perfect agreement, 0 = no agreement)",
                 fontsize=12)
    plt.tight_layout()
    p1 = os.path.join(save_dir, "corr_by_layer.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()

    # ── Plot 2: mean ρ per layer bar chart ─────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = [f"Layer {i+1}" for i in range(num_layers)]
    means  = [avg_corrs[i] for i in range(num_layers)]
    stds   = [float(np.std(layer_corrs[i])) if layer_corrs[i] else 0
              for i in range(num_layers)]
    bars = ax.bar(layers, means, yerr=stds, capsize=5,
                  color="#3498DB", edgecolor="black", linewidth=0.7, alpha=0.85)
    ax.axhline(0, color="gray", linewidth=0.8)
    ax.axhline(overall, color="red", linestyle="--",
               linewidth=1.5, label=f"Overall mean ρ={overall:.3f}")
    for bar, v in zip(bars, means):
        ax.text(bar.get_x()+bar.get_width()/2,
                bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("Mean Spearman ρ", fontsize=11)
    ax.set_title("Attention-Saliency Correlation by Layer\n"
                 "(higher = attention agrees more with gradient importance)",
                 fontsize=12)
    ax.set_ylim(-0.1, 1.05)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    p2 = os.path.join(save_dir, "corr_mean_by_layer.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Correlation plots → {save_dir}/")

    return layer_corrs, avg_corrs, overall


def run_token_removal_experiment(model, val_loader, vocab, device,
                                  n_examples=200, k_values=(1, 2, 3),
                                  save_dir="outputs/attention_validation"):
    """
    For each example:
      1. Remove the top-K attention tokens (zero their embeddings)
      2. Remove K random tokens (repeated 10 times, averaged)
      3. Record accuracy drop for both strategies

    Returns:
        results : dict mapping k → {"attention": drop, "random": drop}
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    print(f"\n[Token Removal] Testing K = {k_values} ...")

    removal_results = {k: {"attn_correct": 0, "rand_correct": 0,
                            "orig_correct": 0, "n": 0}
                       for k in k_values}

    from datasets import load_dataset
    raw = load_dataset("glue", "sst2")
    val_labs = [ex["label"] for ex in raw["validation"]]

    count = 0
    lab_idx = 0
    for input_ids, lengths, labels in val_loader:
        if count >= n_examples:
            break
        for b in range(input_ids.size(0)):
            if count >= n_examples:
                break
            true_len = lengths[b].item()
            true_label = labels[b].item()
            if true_len < max(k_values) + 2:
                lab_idx += 1
                continue

            single = input_ids[b:b+1].to(device)

            # Original prediction
            with torch.no_grad():
                logits_orig = model(single)
            orig_pred = logits_orig.argmax(dim=-1).item()
            orig_correct = int(orig_pred == true_label)

            # Get attention scores
            attn = compute_attention_scores(model, single)[:true_len]
            # Skip CLS token (position 0) — always has high attention
            attn[0] = 0.0
            top_k_idx = np.argsort(attn)[::-1]

            for k in k_values:
                removal_results[k]["orig_correct"] += orig_correct
                removal_results[k]["n"] += 1

                # Attention-guided removal
                mask_attn = torch.ones(true_len, dtype=torch.long).to(device)
                for pos in top_k_idx[:k]:
                    mask_attn[int(pos)] = 0
                ids_masked = single.clone()
                ids_masked[0, :true_len] *= mask_attn

                with torch.no_grad():
                    logits_attn = model(ids_masked)
                attn_pred = logits_attn.argmax(dim=-1).item()
                removal_results[k]["attn_correct"] += int(attn_pred == true_label)

                # Random removal (average over 10 trials)
                rand_correct_sum = 0
                for _ in range(10):
                    rand_positions = np.random.choice(
                        range(1, true_len), size=min(k, true_len-1), replace=False
                    )
                    ids_rand = single.clone()
                    for pos in rand_positions:
                        ids_rand[0, int(pos)] = 0
                    with torch.no_grad():
                        logits_rand = model(ids_rand)
                    rand_pred = logits_rand.argmax(dim=-1).item()
                    rand_correct_sum += int(rand_pred == true_label)
                removal_results[k]["rand_correct"] += rand_correct_sum / 10

            count += 1
            lab_idx += 1

    # Compute accuracy and drop
    summary = {}
    for k, r in removal_results.items():
        n = r["n"]
        if n == 0: continue
        orig_acc = r["orig_correct"] / n
        attn_acc = r["attn_correct"] / n
        rand_acc = r["rand_correct"] / n
        summary[k] = {
            "n":             n,
            "original_acc":  orig_acc,
            "attn_removal_acc": attn_acc,
            "rand_removal_acc": rand_acc,
            "attn_drop":     orig_acc - attn_acc,
            "rand_drop":     orig_acc - rand_acc,
        }
        print(f"  K={k}: orig={orig_acc:.4f}  "
              f"attn_removal={attn_acc:.4f} (drop={orig_acc-attn_acc:+.4f})  "
              f"rand_removal={rand_acc:.4f} (drop={orig_acc-rand_acc:+.4f})")

    # ── Plot ──────────────────────────────────────────────────────────────────
    ks     = sorted(summary.keys())
    attn_drops = [summary[k]["attn_drop"] for k in ks]
    rand_drops = [summary[k]["rand_drop"] for k in ks]
    attn_accs  = [summary[k]["attn_removal_acc"] for k in ks]
    rand_accs  = [summary[k]["rand_removal_acc"] for k in ks]
    orig_accs  = [summary[k]["original_acc"] for k in ks]

    x     = np.arange(len(ks))
    width = 0.3

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: accuracy after removal
    ax1.bar(x - width/2, attn_accs, width, label="Attention-guided removal",
            color="#E74C3C", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax1.bar(x + width/2, rand_accs, width, label="Random removal",
            color="#95A5A6", edgecolor="black", linewidth=0.6, alpha=0.85)
    for i, (aa, ra, oa) in enumerate(zip(attn_accs, rand_accs, orig_accs)):
        ax1.axhline(oa, color="#2ECC71", linestyle="--",
                    linewidth=1.2, alpha=0.6,
                    label="Original acc" if i == 0 else "")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"K={k}" for k in ks])
    ax1.set_ylabel("Accuracy after removal", fontsize=11)
    ax1.set_title("Token Removal: Accuracy\n"
                  "(if attention is causal → red bars much lower)", fontsize=11)
    ax1.legend(fontsize=9)
    ax1.set_ylim(0.4, 1.0)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right: accuracy drop
    ax2.bar(x - width/2, attn_drops, width, label="Attention-guided drop",
            color="#E74C3C", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.bar(x + width/2, rand_drops, width, label="Random drop",
            color="#95A5A6", edgecolor="black", linewidth=0.6, alpha=0.85)
    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"K={k}" for k in ks])
    ax2.set_ylabel("Accuracy drop (original − removed)", fontsize=11)
    ax2.set_title("Token Removal: Accuracy Drop\n"
                  "(higher = those tokens were more important)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(True, axis="y", alpha=0.3)

    plt.suptitle("Attention Faithfulness Test: Top-K Token Removal",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "token_removal.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Token removal plot → {path}")

    # Save summary
    with open(os.path.join(save_dir, "token_removal_results.json"),
              "w") as f:
        json.dump({str(k): v for k, v in summary.items()}, f, indent=2)

    return summary


def run_all_attention_validation(model, val_loader, vocab, device,
                                  save_dir="outputs/attention_validation"):
    """Run both experiments and write a summary file."""
    os.makedirs(save_dir, exist_ok=True)

    print("\n" + "="*65)
    print("  ATTENTION VALIDATION EXPERIMENTS")
    print("="*65)

    # Experiment 1: Correlation
    layer_corrs, avg_corrs, overall_rho = run_correlation_experiment(
        model, val_loader, vocab, device,
        n_examples=200, save_dir=save_dir
    )

    # Experiment 2: Token removal
    removal_summary = run_token_removal_experiment(
        model, val_loader, vocab, device,
        n_examples=200, k_values=(1, 2, 3),
        save_dir=save_dir
    )

    # Write summary
    lines = [
        "="*65,
        "  ATTENTION VALIDATION SUMMARY",
        "="*65,
        "",
        "1. Attention-Saliency Correlation (Spearman rho)",
        "─"*50,
        f"   Overall mean rho (all layers): {overall_rho:.4f}",
    ]
    for i, rho in avg_corrs.items():
        n = len(layer_corrs[i])
        lines.append(f"   Layer {i+1}: mean rho = {rho:.4f}  (n={n})")

    lines += [
        "",
        "2. Top-K Token Removal",
        "─"*50,
        f"   {'K':<5} {'Orig Acc':>10} {'Attn Drop':>12} {'Rand Drop':>12} {'Causal?':>10}",
        "   " + "─"*50,
    ]
    for k, r in sorted(removal_summary.items()):
        causal = "YES" if r["attn_drop"] > r["rand_drop"] * 1.2 else "PARTIAL"
        lines.append(
            f"   {k:<5} {r['original_acc']:>10.4f} "
            f"{r['attn_drop']:>12.4f} {r['rand_drop']:>12.4f} {causal:>10}"
        )

    lines += ["", "="*65]
    text = "\n".join(lines)
    path = os.path.join(save_dir, "validation_summary.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"\n  Summary → {path}")
    print("\n" + text)

    return {"correlation": avg_corrs, "overall_rho": overall_rho,
            "token_removal": removal_summary}