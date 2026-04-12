from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
import numpy as np


class TFIDFBaseline:
    def __init__(self, max_features=20000, ngram_range=(1, 2), C=1.0):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            min_df=2 
        )

        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            solver="lbfgs",
            random_state=42
        )

    def fit(self, sentences, labels):
        print("[TF-IDF] Fitting vectorizer and computing TF-IDF features...")
        X_train = self.vectorizer.fit_transform(sentences)
        print(f"[TF-IDF] Feature matrix shape: {X_train.shape}")

        print("[TF-IDF] Training Logistic Regression...")
        self.classifier.fit(X_train, labels)
        train_preds = self.classifier.predict(X_train)
        train_acc = accuracy_score(labels, train_preds)
        print(f"[TF-IDF] Train accuracy: {train_acc:.4f}")

    def evaluate(self, sentences, labels, split_name="Val"):
        X = self.vectorizer.transform(sentences)
        preds = self.classifier.predict(X)
        acc = accuracy_score(labels, preds)
        print(f"\n[TF-IDF] {split_name} Accuracy: {acc:.4f}")
        print(classification_report(labels, preds,
                                    target_names=["Negative", "Positive"]))
        return acc, preds

    def get_top_features(self, n=20):
        feature_names = self.vectorizer.get_feature_names_out()
        coef = self.classifier.coef_[0]  # shape [n_features]

        # Top positive features (most predictive of positive sentiment)
        top_pos_idx = np.argsort(coef)[-n:][::-1]
        top_neg_idx = np.argsort(coef)[:n]

        print(f"\nTop {n} positive-sentiment features:")
        for i in top_pos_idx:
            print(f"  {feature_names[i]:<25} weight={coef[i]:+.4f}")

        print(f"\nTop {n} negative-sentiment features:")
        for i in top_neg_idx:
            print(f"  {feature_names[i]:<25} weight={coef[i]:+.4f}")

        return feature_names, coef


# RUNNER
def run_tfidf_baseline():
    """Load SST-2, run TF-IDF baseline, print results."""
    print("=" * 60)
    print("BASELINE 1: TF-IDF + Logistic Regression")
    print("=" * 60)

    raw = load_dataset("glue", "sst2")
    train_sentences = [ex["sentence"] for ex in raw["train"]]
    train_labels    = [ex["label"]    for ex in raw["train"]]
    val_sentences   = [ex["sentence"] for ex in raw["validation"]]
    val_labels      = [ex["label"]    for ex in raw["validation"]]

    model = TFIDFBaseline()
    model.fit(train_sentences, train_labels)
    val_acc, _ = model.evaluate(val_sentences, val_labels)
    model.get_top_features(n=15)

    return model, val_acc


if __name__ == "__main__":
    run_tfidf_baseline()
