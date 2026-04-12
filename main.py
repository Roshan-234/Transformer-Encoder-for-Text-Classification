import os, sys, argparse, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import (TRANSFORMER_CONFIG, BILSTM_CONFIG, DATA_CONFIG,
                    TFIDF_CONFIG, INTERP_CONFIG, PATHS, VIZ_SENTENCES, DEVICE)
from data.data_loader         import get_dataloaders
from models.tfidf_lr          import TFIDFBaseline
from models.bilstm            import BiLSTMClassifier
from models.transformer       import TransformerClassifier
from train                    import train_model, evaluate
from metrics                  import (compute_all_metrics, plot_confusion_matrix,
                                       plot_metrics_comparison, save_metrics_txt)
from interpretability.attention_viz  import (plot_all_heads, plot_layer_summary,
                                               analyze_head_patterns)
from interpretability.saliency       import run_saliency_analysis
from interpretability.probing        import run_all_probes
from interpretability.results_viz    import (plot_training_curves,
                                               plot_model_comparison,
                                               plot_tfidf_vs_attention)
from experiments.ablation            import run_ablation
from experiments.attention_validation import run_all_attention_validation
from experiments.scaling             import (run_scaling_experiment,
                                              run_pretrained_embeddings_experiment)

for d in [PATHS["checkpoint_dir"], PATHS["output_dir"]]:
    os.makedirs(d, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-ablation",  action="store_true")
    parser.add_argument("--skip-scaling",   action="store_true")
    parser.add_argument("--skip-pretrained",action="store_true")
    parser.add_argument("--only-ablation",  action="store_true")
    parser.add_argument("--only-validation",action="store_true")
    args = parser.parse_args()

    _banner()
    print(f"\n  Device : {DEVICE}")
    if DEVICE == "cuda":
        print(f"  GPU    : {torch.cuda.get_device_name(0)}")

    # ── Load data (shared across all experiments) ─────────────────────────────
    _h("STEP 1: Loading SST-2 Dataset")
    train_loader, val_loader, vocab = get_dataloaders(
        batch_size=DATA_CONFIG["batch_size"],
        max_vocab=DATA_CONFIG["max_vocab"]
    )
    print(f"  Vocab size    : {len(vocab):,} tokens")
    print(f"  Train batches : {len(train_loader)}  |  Val : {len(val_loader)}")

    from datasets import load_dataset
    raw      = load_dataset("glue", "sst2")
    val_labs = [ex["label"] for ex in raw["validation"]]
    tr_sents = [ex["sentence"] for ex in raw["train"]]
    tr_labs  = [ex["label"]    for ex in raw["train"]]
    val_sents= [ex["sentence"] for ex in raw["validation"]]

    all_metrics  = []
    probe_results = {}

    if args.only_ablation:
        _run_ablation(train_loader, val_loader, vocab)
        return
    if args.only_validation:
        transformer = _load_best_transformer(vocab)
        run_all_attention_validation(
            transformer, val_loader, vocab, DEVICE)
        return


    # STEP 2: TF-IDF
    _h("STEP 2: Baseline 1 — TF-IDF + Logistic Regression")
    tfidf = TFIDFBaseline(**TFIDF_CONFIG)
    tfidf.fit(tr_sents, tr_labs)
    tfidf_preds = tfidf.classifier.predict(
        tfidf.vectorizer.transform(val_sents))
    tfidf_probs = tfidf.classifier.predict_proba(
        tfidf.vectorizer.transform(val_sents))[:, 1]
    feat_names, coefs = tfidf.get_top_features(n=20)
    tm = compute_all_metrics(val_labs, tfidf_preds, tfidf_probs, "TF-IDF + LR")
    plot_confusion_matrix(tm["confusion_matrix"], "TF-IDF + LR",
                          PATHS["output_dir"])
    all_metrics.append(tm)

    # STEP 3: BiLSTM
    _h("STEP 3: Baseline 2 — Bidirectional LSTM")
    bilstm = BiLSTMClassifier(
        vocab_size=len(vocab), embed_dim=BILSTM_CONFIG["embed_dim"],
        hidden_dim=BILSTM_CONFIG["hidden_dim"],
        num_layers=BILSTM_CONFIG["num_layers"],
        dropout=BILSTM_CONFIG["dropout"], pad_idx=0)
    bilstm_history = train_model(
        bilstm, train_loader, val_loader,
        BILSTM_CONFIG, PATHS["bilstm_ckpt"], "bilstm")
    ckpt = torch.load(PATHS["bilstm_ckpt"], map_location=DEVICE)
    bilstm.load_state_dict(ckpt["model_state_dict"])
    bilstm = bilstm.to(DEVICE)
    bl_preds, bl_probs = _get_preds_probs(bilstm, val_loader, DEVICE, "bilstm")
    bm = compute_all_metrics(val_labs, bl_preds, bl_probs, "BiLSTM")
    plot_confusion_matrix(bm["confusion_matrix"], "BiLSTM", PATHS["output_dir"])
    plot_training_curves(bilstm_history, "BiLSTM", PATHS["output_dir"])
    all_metrics.append(bm)

    # STEP 4: Transformer
    _h("STEP 4: Primary Model — Transformer Encoder (from scratch)")
    transformer = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=TRANSFORMER_CONFIG["d_model"],
        num_heads=TRANSFORMER_CONFIG["num_heads"],
        num_layers=TRANSFORMER_CONFIG["num_layers"],
        d_ff=TRANSFORMER_CONFIG["d_ff"],
        max_len=TRANSFORMER_CONFIG["max_len"],
        dropout=TRANSFORMER_CONFIG["dropout"], pad_idx=0)
    tf_history = train_model(
        transformer, train_loader, val_loader,
        TRANSFORMER_CONFIG, PATHS["transformer_ckpt"], "transformer")
    ckpt = torch.load(PATHS["transformer_ckpt"], map_location=DEVICE)
    transformer.load_state_dict(ckpt["model_state_dict"])
    transformer = transformer.to(DEVICE)
    tf_preds, tf_probs = _get_preds_probs(
        transformer, val_loader, DEVICE, "transformer")
    tfm = compute_all_metrics(
        val_labs, tf_preds, tf_probs, "Transformer (from scratch)")
    plot_confusion_matrix(tfm["confusion_matrix"],
                          "Transformer (from scratch)", PATHS["output_dir"])
    plot_training_curves(tf_history, "Transformer", PATHS["output_dir"])
    all_metrics.append(tfm)

    # STEP 5: Metrics comparison
    _h("STEP 5: Full Metrics Comparison")
    plot_metrics_comparison(all_metrics, PATHS["output_dir"])
    plot_model_comparison(
        {m["model"]: m["accuracy"] for m in all_metrics},
        PATHS["output_dir"])
    _print_table(all_metrics)

    # STEPS 6–7: Attention visualization
    _h("STEP 6: Attention Visualization — Research Question 2")
    attn_dir = os.path.join(PATHS["output_dir"], "attention")
    for i, sent in enumerate(VIZ_SENTENCES):
        ids   = vocab.encode(sent)
        ids_t = torch.tensor([ids], dtype=torch.long).to(DEVICE)
        toks  = [vocab.idx2token[x] for x in ids]
        sdir  = os.path.join(attn_dir, f"example_{i:02d}")
        plot_all_heads(transformer, ids_t, toks, sent, sdir)
        plot_layer_summary(transformer, ids_t, toks, sdir)

    _h("STEP 7: Head Pattern Analysis — Clark et al. (2019)")
    analyze_head_patterns(
        transformer, val_loader, vocab, DEVICE,
        n_batches=INTERP_CONFIG["head_analysis_batches"],
        save_dir=attn_dir)

    # STEPS 8–9: Saliency + Probing
    _h("STEP 8: Gradient Saliency Maps")
    run_saliency_analysis(
        transformer, val_loader, vocab, DEVICE,
        n_examples=INTERP_CONFIG["n_saliency_ex"],
        save_dir=os.path.join(PATHS["output_dir"], "saliency"))

    _h("STEP 9: Probing Classifiers")
    probe_results = run_all_probes(
        transformer, val_loader, vocab, DEVICE,
        save_dir=os.path.join(PATHS["output_dir"], "probing"))

    # STEP 10: TF-IDF vs Attention
    _h("STEP 10: TF-IDF vs Attention Comparison")
    sent  = VIZ_SENTENCES[3]
    ids   = vocab.encode(sent)
    ids_t = torch.tensor([ids], dtype=torch.long).to(DEVICE)
    toks  = [vocab.idx2token[x] for x in ids]
    transformer.eval()
    with torch.no_grad():
        _ = transformer(ids_t)
    all_attn = transformer.get_all_attention_weights()
    cls_attn = np.zeros(len(toks))
    n = 0
    for la in all_attn:
        if la is None: continue
        cls_attn += la[0].cpu().numpy().mean(axis=0)[0, :len(toks)]
        n += 1
    if n: cls_attn /= n
    plot_tfidf_vs_attention(feat_names, coefs, toks, cls_attn,
                            PATHS["output_dir"])

    # STEP 11: Results file
    _h("STEP 11: Writing Complete Results")
    save_metrics_txt(all_metrics, probe_results, PATHS["results_file"])


    # STEP 12: Ablation study
    if not args.skip_ablation:
        _h("STEP 12: Ablation Study — 4 Component Variants")
        print("  NOTE: This trains 5 models. Use --skip-ablation to skip.")
        run_ablation(train_loader, val_loader, vocab)
    else:
        print("\n[SKIP] Ablation study (--skip-ablation flag set)")

    # STEP 13: Attention validation
    _h("STEP 13: Attention Validation — Correlation + Token Removal")
    run_all_attention_validation(
        transformer, val_loader, vocab, DEVICE,
        save_dir=os.path.join(PATHS["output_dir"], "attention_validation"))

    # STEP 14: Scaling
    if not args.skip_scaling:
        _h("STEP 14: Scaling Experiment — 3 Model Sizes")
        run_scaling_experiment(train_loader, val_loader, vocab)
    else:
        print("\n[SKIP] Scaling experiment (--skip-scaling flag set)")

    # STEP 15: Pretrained embeddings
    if not args.skip_pretrained:
        _h("STEP 15: Pretrained Embeddings (GloVe vs Random Init)")
        run_pretrained_embeddings_experiment(
            train_loader, val_loader, vocab,
            glove_path="data/glove.6B.100d.txt")
    else:
        print("\n[SKIP] Pretrained embeddings (--skip-pretrained flag set)")

    _h("ALL EXPERIMENTS COMPLETE")
    _print_output_tree()

# HELPERS
@torch.no_grad()
def _get_preds_probs(model, loader, device, mtype):
    model.eval()
    preds, probs = [], []
    for ids, lens, _ in loader:
        ids = ids.to(device); lens = lens.to(device)
        logits = model(ids) if mtype == "transformer" else model(ids, lens)
        p = torch.softmax(logits, dim=-1)
        preds.extend(p.argmax(dim=-1).cpu().tolist())
        probs.extend(p[:, 1].cpu().tolist())
    return preds, probs


def _load_best_transformer(vocab):
    ckpt = torch.load(PATHS["transformer_ckpt"], map_location=DEVICE)
    model = TransformerClassifier(
        vocab_size=len(vocab),
        d_model=TRANSFORMER_CONFIG["d_model"],
        num_heads=TRANSFORMER_CONFIG["num_heads"],
        num_layers=TRANSFORMER_CONFIG["num_layers"],
        d_ff=TRANSFORMER_CONFIG["d_ff"],
        max_len=TRANSFORMER_CONFIG["max_len"],
        dropout=0.0, pad_idx=0)
    model.load_state_dict(ckpt["model_state_dict"])
    return model.to(DEVICE)


def _banner():
    print("\n" + "="*70)
    print("  CS 5100 Final Project: Analyzing Self-Attention in Transformers")
    print("  Roshan Ghimire | SST-2 Sentiment Classification")
    print("="*70)

def _h(title):
    print(f"\n{'─'*60}\n  {title}\n{'─'*60}")

def _print_table(all_metrics):
    print(f"\n  {'Model':<35} {'Acc':>8} {'F1 Mac':>8} {'MCC':>8}")
    print("  " + "─"*62)
    for m in all_metrics:
        print(f"  {m['model']:<35} {m['accuracy']:>8.4f} "
              f"{m['f1_macro']:>8.4f} {m['mcc']:>8.4f}")

def _print_output_tree():
    print("""
  ─────────────────────────────────────────────────────────────
  OUTPUTS
  ─────────────────────────────────────────────────────────────
  outputs/
  ├── results.txt                    full metrics
  ├── model_comparison.png           accuracy bar chart
  ├── metrics_comparison.png         6-metric comparison
  ├── tfidf_vs_attention.png         RQ1
  ├── confusion_*.png                confusion matrices
  ├── *_curves.png                   training curves
  ├── attention/                     RQ2 heat maps
  ├── saliency/                      RQ3 saliency
  ├── probing/                       RQ3 probes
  ├── attention_validation/          NEW: corr + token removal
  ├── ablation/                      NEW: component ablations
  └── scaling/                       NEW: model sizes + GloVe
  ─────────────────────────────────────────────────────────────
  Quick re-run flags:
    python main.py --skip-ablation --skip-scaling
    python main.py --only-ablation
    python main.py --only-validation
    python evaluate.py --sentence "your text here"
""")


if __name__ == "__main__":
    main()
