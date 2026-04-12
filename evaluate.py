import os, sys, argparse, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (TRANSFORMER_CONFIG, BILSTM_CONFIG, DATA_CONFIG,
                    PATHS, INTERP_CONFIG, VIZ_SENTENCES, DEVICE)
from data.data_loader            import get_dataloaders
from models.transformer          import TransformerClassifier
from models.bilstm               import BiLSTMClassifier
from train                       import evaluate as eval_loop
from metrics                     import (compute_all_metrics,
                                          plot_confusion_matrix,
                                          plot_metrics_comparison,
                                          save_metrics_txt)
from interpretability.attention_viz  import (plot_all_heads, plot_layer_summary,
                                               analyze_head_patterns)
from interpretability.saliency       import (run_saliency_analysis,
                                               compute_saliency, plot_saliency)
from interpretability.probing        import run_all_probes


def load_transformer(vocab_size, device):
    p = PATHS["transformer_ckpt"]
    if not os.path.exists(p):
        sys.exit(f"[ERROR] No checkpoint at {p}. Run main.py first.")
    m = TransformerClassifier(
        vocab_size=vocab_size,
        d_model=TRANSFORMER_CONFIG["d_model"],
        num_heads=TRANSFORMER_CONFIG["num_heads"],
        num_layers=TRANSFORMER_CONFIG["num_layers"],
        d_ff=TRANSFORMER_CONFIG["d_ff"],
        max_len=TRANSFORMER_CONFIG["max_len"],
        dropout=0.0, pad_idx=0
    )
    ckpt = torch.load(p, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m = m.to(device); m.eval()
    print(f"[Loaded] Transformer  epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f}")
    return m


def load_bilstm(vocab_size, device):
    p = PATHS["bilstm_ckpt"]
    if not os.path.exists(p):
        print(f"[WARNING] No BiLSTM checkpoint at {p}.")
        return None
    m = BiLSTMClassifier(
        vocab_size=vocab_size,
        embed_dim=BILSTM_CONFIG["embed_dim"],
        hidden_dim=BILSTM_CONFIG["hidden_dim"],
        num_layers=BILSTM_CONFIG["num_layers"],
        dropout=0.0, pad_idx=0
    )
    ckpt = torch.load(p, map_location=device)
    m.load_state_dict(ckpt["model_state_dict"])
    m = m.to(device); m.eval()
    print(f"[Loaded] BiLSTM       epoch={ckpt['epoch']}  "
          f"val_acc={ckpt['val_acc']:.4f}")
    return m


@torch.no_grad()
def _get_preds_probs(model, loader, device, mtype):
    model.eval()
    preds, probs = [], []
    for ids, lens, _ in loader:
        ids  = ids.to(device); lens = lens.to(device)
        logits = model(ids) if mtype == "transformer" else model(ids, lens)
        p = torch.softmax(logits, dim=-1)
        preds.extend(p.argmax(dim=-1).cpu().tolist())
        probs.extend(p[:, 1].cpu().tolist())
    return preds, probs


def analyze_sentence(transformer, vocab, sentence, device):
    """Print token-level attention and saliency for a custom sentence."""
    print(f"\n{'─'*60}")
    print(f"Sentence: \"{sentence}\"")
    print(f"{'─'*60}")
    ids   = vocab.encode(sentence)
    ids_t = torch.tensor([ids], dtype=torch.long).to(device)
    toks  = [vocab.idx2token[i] for i in ids]

    transformer.eval()
    with torch.no_grad():
        logits = transformer(ids_t)
    probs     = torch.softmax(logits, dim=-1)[0]
    pred      = probs.argmax().item()
    print(f"\nPrediction  : {'POSITIVE' if pred==1 else 'NEGATIVE'}  "
          f"({probs[pred].item():.2%} confidence)")
    print(f"P(negative) : {probs[0].item():.4f}")
    print(f"P(positive) : {probs[1].item():.4f}")

    # Average CLS attention across all layers and heads
    all_attn = transformer.get_all_attention_weights()
    cls_attn = np.zeros(len(toks))
    valid = 0
    for la in all_attn:
        if la is None: continue
        cls_attn += la[0].cpu().numpy().mean(axis=0)[0, :len(toks)]
        valid += 1
    if valid: cls_attn /= valid

    print("\nAvg CLS Attention (across all layers and heads):")
    for tok, sc in zip(toks, cls_attn):
        bar = "█" * int(sc * 40)
        print(f"  {tok:<20} {bar} {sc:.4f}")

    # Gradient saliency
    sal, _ = compute_saliency(transformer, ids_t, target_class=pred)
    sal    = sal[:len(toks)]
    print("\nGradient Saliency (normalised to [0,1]):")
    for tok, sc in zip(toks, sal):
        bar = "█" * int(sc * 40)
        print(f"  {tok:<20} {bar} {sc:.4f}")

    # Save plots
    safe    = sentence[:30].replace(" ", "_")
    sdir    = os.path.join(PATHS["output_dir"], "custom_analysis")
    plot_all_heads(transformer, ids_t, toks, sentence,
                   os.path.join(sdir, safe))
    plot_saliency(toks, sal, sentence, pred, attn_scores=cls_attn,
                  save_path=os.path.join(sdir, f"{safe}_saliency.png"))
    print(f"\nPlots saved → {sdir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentence", type=str, default=None)
    parser.add_argument("--attention", action="store_true")
    parser.add_argument("--saliency",  action="store_true")
    parser.add_argument("--probing",   action="store_true")
    parser.add_argument("--metrics",   action="store_true")
    args = parser.parse_args()
    run_all = not any([args.attention, args.saliency,
                       args.probing, args.metrics, args.sentence])

    print(f"\nDevice: {DEVICE}")
    train_loader, val_loader, vocab = get_dataloaders(
        batch_size=DATA_CONFIG["batch_size"],
        max_vocab=DATA_CONFIG["max_vocab"]
    )
    transformer = load_transformer(len(vocab), DEVICE)
    bilstm      = load_bilstm(len(vocab), DEVICE)

    if args.sentence:
        analyze_sentence(transformer, vocab, args.sentence, DEVICE)
        return

    out = PATHS["output_dir"]

    if run_all or args.metrics:
        print("\n── Recomputing metrics ──")
        from datasets import load_dataset
        raw      = load_dataset("glue", "sst2")
        val_labs = [ex["label"] for ex in raw["validation"]]
        tf_preds, tf_probs = _get_preds_probs(
            transformer, val_loader, DEVICE, "transformer")
        tf_m = compute_all_metrics(val_labs, tf_preds, tf_probs,
                                   "Transformer (from scratch)")
        plot_confusion_matrix(tf_m["confusion_matrix"],
                              "Transformer (from scratch)", out)
        all_m = [tf_m]
        if bilstm:
            bl_preds, bl_probs = _get_preds_probs(
                bilstm, val_loader, DEVICE, "bilstm")
            bl_m = compute_all_metrics(val_labs, bl_preds, bl_probs, "BiLSTM")
            plot_confusion_matrix(bl_m["confusion_matrix"], "BiLSTM", out)
            all_m.append(bl_m)
        plot_metrics_comparison(all_m, out)

    if run_all or args.attention:
        print("\n── Attention visualization ──")
        adir = os.path.join(out, "attention")
        for i, sent in enumerate(VIZ_SENTENCES):
            ids   = vocab.encode(sent)
            ids_t = torch.tensor([ids], dtype=torch.long).to(DEVICE)
            toks  = [vocab.idx2token[x] for x in ids]
            plot_all_heads(transformer, ids_t, toks, sent,
                           os.path.join(adir, f"example_{i:02d}"))
            plot_layer_summary(transformer, ids_t, toks,
                               os.path.join(adir, f"example_{i:02d}"))
        analyze_head_patterns(
            transformer, val_loader, vocab, DEVICE,
            n_batches=INTERP_CONFIG["head_analysis_batches"],
            save_dir=adir
        )

    if run_all or args.saliency:
        print("\n── Saliency maps ──")
        run_saliency_analysis(
            transformer, val_loader, vocab, DEVICE,
            n_examples=INTERP_CONFIG["n_saliency_ex"],
            save_dir=os.path.join(out, "saliency")
        )

    if run_all or args.probing:
        print("\n── Probing classifiers ──")
        run_all_probes(transformer, val_loader, vocab, DEVICE,
                       save_dir=os.path.join(out, "probing"))

    print(f"\nDone. All outputs in {out}/")


if __name__ == "__main__":
    main()
