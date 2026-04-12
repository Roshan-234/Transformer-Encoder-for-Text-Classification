import torch
import numpy as np
import matplotlib.pyplot as plt
import os


def compute_saliency(model, input_ids, target_class=None):
    model.eval()

    # Step 1: Get embeddings for the input_ids
    embeddings = model.embedding(input_ids)          
    embeddings = embeddings * model.embed_scale       # apply scale

    embeddings = embeddings.detach().requires_grad_(True)

    # Step 2: Forward pass using the embedding directly (bypass the Embedding layer)
    mask = model.make_padding_mask(input_ids)
    x = model.pos_encoding(embeddings)

    for layer in model.layers:
        x = layer(x, mask)

    cls_repr = x[:, 0, :]
    logits   = model.classifier(cls_repr)             

    # Step 3: Determine target class
    if target_class is None:
        target_class = logits.argmax(dim=-1).item()

    # Step 4: Backward pass — compute ∂logit_c / ∂embedding
    logit_c = logits[0, target_class]
    model.zero_grad()
    logit_c.backward()

    # embeddings.grad: [1, seq, d_model]
    grad = embeddings.grad.detach()[0]                
    emb  = embeddings.detach()[0]                     

    # Step 5: Embedding × Gradient saliency
    # For each token, sum the elementwise product over the d_model dimension
    saliency = (emb * grad).sum(dim=-1)               
    saliency = saliency.cpu().numpy()

    # Step 6: Normalize to [0, 1] for visualization
    saliency = np.abs(saliency)  # take absolute value (sign doesn't matter)
    if saliency.max() > 0:
        saliency = saliency / saliency.max()

    return saliency, target_class


def plot_saliency(tokens, saliency_scores, sentence, predicted_label, attn_scores=None, save_path=None):
    x = np.arange(len(tokens))

    if attn_scores is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(max(8, len(tokens)*0.6), 7))
    else:
        fig, ax1 = plt.subplots(figsize=(max(8, len(tokens)*0.6), 4))

    label_str = "Positive" if predicted_label == 1 else "Negative"

    # Gradient Saliency 
    colors = plt.cm.Reds(saliency_scores)
    ax1.bar(x, saliency_scores, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
    ax1.set_ylabel("Gradient Saliency", fontsize=10)
    ax1.set_title(f"Gradient Saliency Map\n\"{sentence[:60]}...\"\nPrediction: {label_str}",
                  fontsize=10)
    ax1.set_ylim(0, 1.1)

    # Attention Weights (for comparison)
    if attn_scores is not None:
        colors_attn = plt.cm.Blues(attn_scores / (attn_scores.max() + 1e-9))
        ax2.bar(x, attn_scores, color=colors_attn, edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(tokens, rotation=45, ha="right", fontsize=9)
        ax2.set_ylabel("Avg Attention (CLS)", fontsize=10)
        ax2.set_title("CLS Attention Weights (for comparison)", fontsize=10)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saliency plot saved → {save_path}")
    plt.close()


def run_saliency_analysis(model, data_loader, vocab, device, n_examples=10, save_dir="outputs/saliency"):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    count = 0

    for input_ids, lengths, labels in data_loader:
        if count >= n_examples:
            break

        # Process one example at a time
        for i in range(input_ids.size(0)):
            if count >= n_examples:
                break

            single_ids = input_ids[i:i+1].to(device)     # [1, seq]
            true_len   = lengths[i].item()
            true_label = labels[i].item()

            # Decode tokens (trim to actual length)
            tokens = [vocab.idx2token[idx.item()]
                      for idx in single_ids[0, :true_len]]

            # Compute gradient saliency
            saliency, pred_class = compute_saliency(model, single_ids)
            saliency = saliency[:true_len]

            # Get CLS attention weights for comparison
            # CLS is at position 0; we look at what CLS attends to
            model.eval()
            with torch.no_grad():
                _ = model(single_ids)
            all_attn = model.get_all_attention_weights()
            # Average CLS attention across all layers and heads
            cls_attn_avg = np.zeros(true_len)
            valid_layers = 0
            for layer_attn in all_attn:
                if layer_attn is None:
                    continue
                # [1, heads, seq, seq] → average over heads → [seq, seq]
                layer_np = layer_attn[0].cpu().numpy().mean(axis=0)
                # CLS is query position 0 → layer_np[0, :] = what CLS attends to
                cls_attn_avg += layer_np[0, :true_len]
                valid_layers += 1
            if valid_layers > 0:
                cls_attn_avg /= valid_layers

            sentence = " ".join(tokens[1:])  # skip <CLS> for display
            save_path = os.path.join(save_dir, f"example_{count:03d}_"
                                                f"{'pos' if pred_class==1 else 'neg'}.png")

            plot_saliency(tokens, saliency, sentence, pred_class,
                          attn_scores=cls_attn_avg, save_path=save_path)
            count += 1

    print(f"\nSaliency analysis complete. {count} plots saved → {save_dir}/")
