import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import os
import matplotlib.pyplot as plt


# WORD LISTS FOR PROBING
NEGATION_WORDS = {
    "not", "never", "no", "nobody", "nothing", "neither", "nor",
    "n't", "nt", "without", "hardly", "barely", "scarcely"
}

INTENSIFIER_WORDS = {
    "very", "really", "extremely", "incredibly", "absolutely", "utterly",
    "quite", "rather", "pretty", "highly", "deeply", "truly", "so", "too"
}


# PROBE 1: Sentiment from CLS hidden states at each layer
def probe_sentiment_per_layer(model, data_loader, device, save_dir="outputs/probing"):
    model.eval()
    num_layers = model.num_layers

    # Collect hidden states and labels for all sentences
    # reps_by_layer[i] = list of CLS vectors from layer i
    reps_by_layer = [[] for _ in range(num_layers)]
    all_labels    = []

    print("[Probing] Collecting hidden states for sentiment probe...")
    with torch.no_grad():
        for input_ids, lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            # get_intermediate_representations returns a list of
            # [batch, seq, d_model] tensors, one per layer
            layer_reps = model.get_intermediate_representations(input_ids)

            for layer_idx, rep in enumerate(layer_reps):
                # CLS token is always at position 0
                cls_rep = rep[:, 0, :].cpu().numpy()    # [batch, d_model]
                reps_by_layer[layer_idx].append(cls_rep)

            all_labels.extend(labels.numpy())

    # Split 80/20 for probing train/test
    n = len(all_labels)
    split = int(0.8 * n)
    labels_arr  = np.array(all_labels)
    train_labels = labels_arr[:split]
    test_labels  = labels_arr[split:]

    layer_accuracies = []
    print("[Probing] Training sentiment probes per layer...")

    for layer_idx in range(num_layers):
        X = np.concatenate(reps_by_layer[layer_idx], axis=0)  # [n, d_model]
        X_train, X_test = X[:split], X[split:]

        probe = LogisticRegression(max_iter=500, C=1.0, random_state=42)
        probe.fit(X_train, train_labels)
        acc = accuracy_score(test_labels, probe.predict(X_test))
        layer_accuracies.append(acc)
        print(f"  Layer {layer_idx+1}: probe accuracy = {acc:.4f}")

    # ── PLOT ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = list(range(1, num_layers + 1))
    ax.plot(layers, layer_accuracies, marker="o", linewidth=2, color="steelblue")
    ax.axhline(y=max(sum(labels_arr) / len(labels_arr),
                     1 - sum(labels_arr) / len(labels_arr)),
               color="gray", linestyle="--", label="Majority baseline")
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Sentiment Probe: Accuracy by Layer\n(RQ3: Does attention encode sentiment?)")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "sentiment_probe_by_layer.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Sentiment probe plot saved → {save_path}")

    return layer_accuracies

# PROBE 2: Token-level property probing (negation / intensifier)
def probe_token_property(model, data_loader, vocab, device, property_name="negation", save_dir="outputs/probing"):
    if property_name == "negation":
        target_words = NEGATION_WORDS
    elif property_name == "intensifier":
        target_words = INTENSIFIER_WORDS
    else:
        raise ValueError(f"Unknown property: {property_name}")

    model.eval()
    num_layers = model.num_layers

    # For each layer, collect (hidden_state, is_target_word) pairs
    reps_by_layer  = [[] for _ in range(num_layers)]
    all_token_labels = []

    print(f"[Probing] Collecting token reps for {property_name} probe...")
    with torch.no_grad():
        for input_ids, lengths, labels in data_loader:
            input_ids = input_ids.to(device)
            layer_reps = model.get_intermediate_representations(input_ids)

            batch_size, seq_len = input_ids.shape
            for b in range(batch_size):
                true_len = lengths[b].item()

                # Create token-level binary labels for this sentence
                tok_labels = []
                for pos in range(true_len):
                    token_str = vocab.idx2token.get(input_ids[b, pos].item(), "")
                    tok_labels.append(1 if token_str in target_words else 0)

                # Collect hidden state at each real position for each layer
                for layer_idx, rep in enumerate(layer_reps):
                    # rep: [batch, seq, d_model]
                    token_reps = rep[b, :true_len, :].cpu().numpy()  # [true_len, d_model]
                    reps_by_layer[layer_idx].append(token_reps)

                all_token_labels.extend(tok_labels)

    # Split
    n = len(all_token_labels)
    split = int(0.8 * n)
    labels_arr   = np.array(all_token_labels)
    train_labels = labels_arr[:split]
    test_labels  = labels_arr[split:]

    # Count class balance (positive rate)
    pos_rate = labels_arr.mean()
    majority_baseline = max(pos_rate, 1 - pos_rate)
    print(f"  {property_name} token rate: {pos_rate:.3f} | Majority baseline: {majority_baseline:.4f}")

    layer_accuracies = []
    print(f"[Probing] Training {property_name} token probes per layer...")

    for layer_idx in range(num_layers):
        X = np.concatenate(reps_by_layer[layer_idx], axis=0)  # [n_tokens, d_model]
        X_train, X_test = X[:split], X[split:]

        probe = LogisticRegression(
            max_iter=500, C=1.0, random_state=42,
            class_weight="balanced"  # handles class imbalance (few negation words)
        )
        probe.fit(X_train, train_labels)
        acc = accuracy_score(test_labels, probe.predict(X_test))
        layer_accuracies.append(acc)
        print(f"  Layer {layer_idx+1}: {property_name} probe acc = {acc:.4f}")

    # PLOT
    fig, ax = plt.subplots(figsize=(7, 4))
    layers = list(range(1, num_layers + 1))
    ax.plot(layers, layer_accuracies, marker="o", linewidth=2, color="darkorange")
    ax.axhline(y=majority_baseline, color="gray", linestyle="--",
               label=f"Majority baseline ({majority_baseline:.3f})")
    ax.set_xlabel("Encoder Layer")
    ax.set_ylabel("Probe Accuracy")
    ax.set_title(f"{property_name.capitalize()} Token Probe: Accuracy by Layer\n"
                 f"(Does encoder encode {property_name} words differently?)")
    ax.set_xticks(layers)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{property_name}_probe_by_layer.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  {property_name} probe plot saved → {save_path}")

    return layer_accuracies

# RUN ALL PROBES
def run_all_probes(model, val_loader, vocab, device, save_dir="outputs/probing"):
    print("\n" + "="*60)
    print("INTERPRETABILITY: PROBING CLASSIFIERS (RQ3)")
    print("="*60)

    results = {}

    # Probe 1: Sentiment at CLS per layer
    results["sentiment"] = probe_sentiment_per_layer(
        model, val_loader, device, save_dir
    )

    # Probe 2: Negation token detection
    results["negation"] = probe_token_property(
        model, val_loader, vocab, device, "negation", save_dir
    )

    # Probe 3: Intensifier token detection
    results["intensifier"] = probe_token_property(
        model, val_loader, vocab, device, "intensifier", save_dir
    )

    # SUMMARY TABLE
    print(f"{'Probe':<20} | " + " | ".join(
        [f"L{i+1}" for i in range(model.num_layers)]
    ))
    print("─"*55)
    for probe_name, accs in results.items():
        row = f"{probe_name:<20} | " + " | ".join([f"{a:.3f}" for a in accs])
        print(row)
    print("─"*55)

    return results
