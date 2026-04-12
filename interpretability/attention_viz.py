import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


def plot_attention_head(attn_matrix, tokens, layer_idx, head_idx, save_path=None):
    fig, ax = plt.subplots(figsize=(max(6, len(tokens) * 0.5),
                                    max(6, len(tokens) * 0.5)))

    # Plot heat map: rows = query positions, cols = key positions
    im = ax.imshow(attn_matrix, cmap="Blues", aspect="auto",
                   vmin=0.0, vmax=attn_matrix.max())

    # Axis labels: token strings
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tokens, fontsize=8)

    ax.set_xlabel("Key (attended-to position)", fontsize=10)
    ax.set_ylabel("Query (attending position)", fontsize=10)
    ax.set_title(f"Layer {layer_idx + 1}, Head {head_idx + 1}\n"
                 f"Attention Weight Distribution", fontsize=11)

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved → {save_path}")
    plt.close()


def plot_all_heads(model, input_ids, tokens, sentence_text, save_dir="outputs/attention"):
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)  

    all_attn = model.get_all_attention_weights()
    # all_attn: list of [1, num_heads, seq, seq] tensors

    print(f"\nPlotting attention for: \"{sentence_text[:50]}\"")
    os.makedirs(save_dir, exist_ok=True)

    for layer_idx, layer_attn in enumerate(all_attn):
        if layer_attn is None:
            continue
        # layer_attn: [1, num_heads, seq, seq] → [num_heads, seq, seq]
        layer_attn_np = layer_attn[0].cpu().numpy()  # [heads, seq, seq]

        for head_idx in range(layer_attn_np.shape[0]):
            head_matrix = layer_attn_np[head_idx]  # [seq, seq]
            save_path = os.path.join(save_dir, f"layer{layer_idx}_head{head_idx}.png")
            plot_attention_head(
                head_matrix, tokens, layer_idx, head_idx, save_path
            )

    print(f"  Saved {len(all_attn) * all_attn[0].shape[1]} attention plots → {save_dir}/")


def plot_layer_summary(model, input_ids, tokens, save_dir="outputs/attention"):
    model.eval()
    with torch.no_grad():
        _ = model(input_ids)

    all_attn = model.get_all_attention_weights()
    num_layers = len(all_attn)
    num_heads  = all_attn[0].shape[1]

    fig, axes = plt.subplots(
        num_layers, num_heads,
        figsize=(num_heads * 3, num_layers * 3)
    )
    # Handle case of single layer or single head
    if num_layers == 1:
        axes = [axes]
    if num_heads == 1:
        axes = [[ax] for ax in axes]

    for layer_idx, layer_attn in enumerate(all_attn):
        if layer_attn is None:
            continue
        layer_np = layer_attn[0].cpu().numpy()  # [heads, seq, seq]
        for head_idx in range(num_heads):
            ax = axes[layer_idx][head_idx]
            im = ax.imshow(layer_np[head_idx], cmap="Blues",
                           vmin=0.0, vmax=layer_np[head_idx].max(),
                           aspect="auto")
            # Only label the first column and first row to avoid clutter
            if head_idx == 0:
                ax.set_ylabel(f"L{layer_idx+1}", fontsize=8)
            if layer_idx == 0:
                ax.set_title(f"H{head_idx+1}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

    plt.suptitle("All Attention Heads (rows=layers, cols=heads)", fontsize=12)
    plt.tight_layout()

    save_path = os.path.join(save_dir, "all_heads_grid.png")
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  All-heads grid saved → {save_path}")


def compute_head_entropy(attn_matrix):
    # Clip to avoid log(0)
    p = np.clip(attn_matrix, 1e-9, 1.0)
    entropy = -np.sum(p * np.log(p), axis=-1)  # [seq]
    return entropy


def analyze_head_patterns(model, data_loader, vocab, device, n_batches=10, save_dir="outputs/attention"):
    model.eval()

    num_layers = model.num_layers
    num_heads  = model.num_heads

    # Accumulation arrays
    cls_attn_sum  = np.zeros((num_layers, num_heads))
    adj_attn_sum  = np.zeros((num_layers, num_heads))
    entropy_sum   = np.zeros((num_layers, num_heads))
    count         = 0

    with torch.no_grad():
        for batch_idx, (input_ids, lengths, labels) in enumerate(data_loader):
            if batch_idx >= n_batches:
                break

            input_ids = input_ids.to(device)
            _ = model(input_ids)
            all_attn = model.get_all_attention_weights()

            # all_attn[layer]: [batch, heads, seq, seq]
            for layer_idx, layer_attn in enumerate(all_attn):
                if layer_attn is None:
                    continue
                batch_np = layer_attn.cpu().numpy()  # [B, H, S, S]
                B, H, S, _ = batch_np.shape

                for h in range(H):
                    for b in range(B):
                        mat = batch_np[b, h]  # [S, S]
                        seq_len = lengths[b].item()
                        mat_real = mat[:seq_len, :seq_len]

                        # Avg attention to CLS (position 0) from all positions
                        cls_attn_sum[layer_idx, h]  += mat_real[:, 0].mean()

                        # Avg attention to adjacent tokens
                        adj = 0.0
                        for i in range(seq_len):
                            for j in [i-1, i+1]:
                                if 0 <= j < seq_len:
                                    adj += mat_real[i, j]
                        adj_attn_sum[layer_idx, h] += adj / seq_len

                        # Average entropy
                        ent = compute_head_entropy(mat_real).mean()
                        entropy_sum[layer_idx, h] += ent

                count += B

    # Normalize
    cls_attn = cls_attn_sum  / count
    adj_attn = adj_attn_sum  / count
    avg_ent  = entropy_sum   / count

    # ── PLOT SUMMARY ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, data, title in zip(
        axes,
        [cls_attn, adj_attn, avg_ent],
        ["Avg Attention to CLS", "Avg Attention to Adjacent", "Avg Entropy"]
    ):
        im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
        ax.set_xlabel("Head")
        ax.set_ylabel("Layer")
        ax.set_xticks(range(num_heads))
        ax.set_yticks(range(num_layers))
        ax.set_xticklabels([f"H{i+1}" for i in range(num_heads)])
        ax.set_yticklabels([f"L{i+1}" for i in range(num_layers)])
        ax.set_title(title)
        plt.colorbar(im, ax=ax)

    plt.suptitle("Head Pattern Analysis (Clark et al. Method)", fontsize=12)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "head_pattern_summary.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Head pattern analysis saved → {save_path}")

    return {"cls_attn": cls_attn, "adj_attn": adj_attn, "entropy": avg_ent}
