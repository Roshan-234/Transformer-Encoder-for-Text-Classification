import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import os
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix,
    classification_report, roc_auc_score
)


def compute_all_metrics(y_true, y_pred, y_prob=None, model_name="Model"):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc      = accuracy_score(y_true, y_pred)
    prec_neg = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    prec_pos = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec_neg  = recall_score(y_true, y_pred,    pos_label=0, zero_division=0)
    rec_pos  = recall_score(y_true, y_pred,    pos_label=1, zero_division=0)
    f1_neg   = f1_score(y_true, y_pred,        pos_label=0, zero_division=0)
    f1_pos   = f1_score(y_true, y_pred,        pos_label=1, zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wtd   = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    mcc      = matthews_corrcoef(y_true, y_pred)
    cm       = confusion_matrix(y_true, y_pred)

    auc = None
    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob)
        except Exception:
            auc = None

    metrics = {
        "model":           model_name,
        "accuracy":        acc,
        "precision_neg":   prec_neg,
        "precision_pos":   prec_pos,
        "recall_neg":      rec_neg,
        "recall_pos":      rec_pos,
        "f1_neg":          f1_neg,
        "f1_pos":          f1_pos,
        "f1_macro":        f1_macro,
        "f1_weighted":     f1_wtd,
        "mcc":             mcc,
        "auc_roc":         auc,
        "confusion_matrix": cm,
        "support_neg":     int((y_true == 0).sum()),
        "support_pos":     int((y_true == 1).sum()),
    }

    _print_metrics(metrics)
    return metrics


def _print_metrics(m):
    auc_str = f"{m['auc_roc']:.4f}" if m['auc_roc'] is not None else "N/A"
    print(f"\n{'─'*55}")
    print(f"  {m['model']} — Evaluation Metrics")
    print(f"{'─'*55}")
    print(f"  Accuracy          : {m['accuracy']:.4f}  ({m['accuracy']*100:.2f}%)")
    print(f"  F1 Macro          : {m['f1_macro']:.4f}")
    print(f"  F1 Weighted       : {m['f1_weighted']:.4f}")
    print(f"  MCC               : {m['mcc']:.4f}")
    print(f"  AUC-ROC           : {auc_str}")
    print(f"\n  Per-Class Results:")
    print(f"  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print(f"  {'-'*52}")
    print(f"  {'Negative':<12} {m['precision_neg']:>10.4f} {m['recall_neg']:>10.4f} "
          f"{m['f1_neg']:>10.4f} {m['support_neg']:>10}")
    print(f"  {'Positive':<12} {m['precision_pos']:>10.4f} {m['recall_pos']:>10.4f} "
          f"{m['f1_pos']:>10.4f} {m['support_pos']:>10}")
    print(f"\n  Confusion Matrix:")
    print(f"                  Pred Neg   Pred Pos")
    print(f"  Actual Neg   {m['confusion_matrix'][0,0]:>9}  {m['confusion_matrix'][0,1]:>9}")
    print(f"  Actual Pos   {m['confusion_matrix'][1,0]:>9}  {m['confusion_matrix'][1,1]:>9}")
    print(f"{'─'*55}")


def plot_confusion_matrix(cm, model_name, save_dir):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")

    # Annotate each cell with the count and percentage
    total = cm.sum()
    for i in range(2):
        for j in range(2):
            pct = cm[i, j] / total * 100
            ax.text(j, i, f"{cm[i,j]}\n({pct:.1f}%)",
                    ha="center", va="center", fontsize=12,
                    color="white" if cm[i,j] > cm.max()*0.6 else "black")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"])
    ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"])
    ax.set_title(f"{model_name}\nConfusion Matrix", fontsize=12, pad=12)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    safe = model_name.lower().replace(" ", "_").replace("+", "plus")
    path = os.path.join(save_dir, f"confusion_{safe}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix saved → {path}")
    return path


def plot_metrics_comparison(all_metrics, save_dir):
    metric_keys  = ["accuracy", "f1_macro", "f1_weighted", "mcc",
                    "precision_pos", "recall_pos"]
    metric_labels = ["Accuracy", "F1 Macro", "F1 Weighted", "MCC",
                     "Precision\n(Positive)", "Recall\n(Positive)"]

    models = [m["model"] for m in all_metrics]
    colors = ["#5B9BD5", "#ED7D31", "#70AD47"]

    x     = np.arange(len(metric_keys))
    width = 0.25

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, (m, color) in enumerate(zip(all_metrics, colors)):
        vals = [m[k] if m[k] is not None else 0.0 for k in metric_keys]
        bars = ax.bar(x + i*width, vals, width, label=m["model"],
                      color=color, edgecolor="black", linewidth=0.6)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2,
                    bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom",
                    fontsize=7, rotation=45)

    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels, fontsize=10)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_ylim(0, 1.12)
    ax.set_title("Full Metrics Comparison — All Three Models (SST-2 Validation Set)",
                 fontsize=12)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, "metrics_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Full metrics comparison saved → {path}")
    return path


def save_metrics_txt(all_metrics, probe_results, path):
    lines = [
        "=" * 65,
        "  CS 5100 Final Project — Complete Results",
        "  Roshan Ghimire | SST-2 Binary Sentiment Classification",
        "=" * 65,
        "",
        "─" * 65,
        "  CLASSIFICATION ACCURACY (Validation Set)",
        "─" * 65,
        f"  {'Model':<35} {'Accuracy':>10} {'F1 Macro':>10}",
        "  " + "-" * 57,
    ]
    for m in all_metrics:
        lines.append(
            f"  {m['model']:<35} {m['accuracy']:>10.4f} {m['f1_macro']:>10.4f}"
        )

    lines += [
        "",
        "─" * 65,
        "  PER-MODEL FULL METRICS",
        "─" * 65,
    ]
    for m in all_metrics:
        auc_str = f"{m['auc_roc']:.4f}" if m['auc_roc'] is not None else "N/A"
        lines += [
            "",
            f"  {m['model']}",
            f"    Accuracy          : {m['accuracy']:.4f}",
            f"    F1 Macro          : {m['f1_macro']:.4f}",
            f"    F1 Weighted       : {m['f1_weighted']:.4f}",
            f"    MCC               : {m['mcc']:.4f}",
            f"    AUC-ROC           : {auc_str}",
            f"    Precision (Neg/Pos): {m['precision_neg']:.4f} / {m['precision_pos']:.4f}",
            f"    Recall    (Neg/Pos): {m['recall_neg']:.4f} / {m['recall_pos']:.4f}",
            f"    F1        (Neg/Pos): {m['f1_neg']:.4f} / {m['f1_pos']:.4f}",
            f"    Confusion Matrix:",
            f"                          Pred Neg   Pred Pos",
            f"      Actual Neg    {m['confusion_matrix'][0,0]:>9}  {m['confusion_matrix'][0,1]:>9}",
            f"      Actual Pos    {m['confusion_matrix'][1,0]:>9}  {m['confusion_matrix'][1,1]:>9}",
        ]

    if probe_results:
        lines += [
            "",
            "─" * 65,
            "  PROBING CLASSIFIER RESULTS (Accuracy per Encoder Layer)",
            "─" * 65,
            f"  {'Probe':<20} " + " | ".join([f"Layer {i+1}" for i in range(4)]),
            "  " + "-" * 55,
        ]
        for probe_name, accs in probe_results.items():
            row = f"  {probe_name:<20} " + " | ".join([f"{a:.4f}" for a in accs])
            lines.append(row)

    lines += [
        "",
        "─" * 65,
        "  OUTPUT FILES",
        "─" * 65,
        "  outputs/results.txt                 <- this file",
        "  outputs/model_comparison.png        <- accuracy bar chart",
        "  outputs/metrics_comparison.png      <- full metrics bar chart",
        "  outputs/tfidf_vs_attention.png      <- RQ1 word importance",
        "  outputs/bilstm_curves.png",
        "  outputs/transformer_curves.png",
        "  outputs/confusion_tfidf*.png",
        "  outputs/confusion_bilstm*.png",
        "  outputs/confusion_transformer*.png",
        "  outputs/attention/                  <- RQ2 heat maps",
        "  outputs/saliency/                   <- RQ3 gradient saliency",
        "  outputs/probing/                    <- RQ3 probe plots",
        "=" * 65,
    ]

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Results written -> {path}")
