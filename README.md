# Transformer Encoder for Text Classification

**Analyzing Self-Attention in Transformers: Implementation, Ablation and Attention Validation on SST-2**

Roshan Ghimire · CS 5100: Foundations of Artificial Intelligence · Northeastern University · April 2026

---

## What This Project Does

The goal of this project is to understand how the Transformer architecture works by building it from scratch. Of using a pre-trained model every part of the model is implemented using PyTorch tensor operations. This includes the tokenizer, positional encodings, attention mechanism and layer normalization. No pre-built transformer libraries are used.

The project focuses on binary sentiment classification on the Stanford Sentiment Treebank SST-2 dataset. This means predicting whether a movie review phrase is positive or negative. The from-scratch Transformer model is compared to a TF-IDF + Logistic Regression baseline and a Bidirectional LSTM model. The project also explores what the attention mechanism learns and whether its weights are meaningful for the models predictions.

---

## Results at a Glance

The three models were evaluated on the SST-2 validation set, which has 872 examples. The results are as follows:

| Model | Accuracy | F1 Macro | MCC | AUC-ROC |
|---|---|---|---|---|
| TF-IDF + Logistic Regression | 80.96% | 0.8088 | 0.6211 | 0.9065 |
| Bidirectional LSTM | 82.57% | 0.8254 | 0.6516 | 0.9012 |
| Transformer (from scratch) | 80.85% | 0.8085 | 0.6181 | 0.8869 |

The Transformer model performs similarly to the TF-IDF model in terms of accuracy. It produces more informative representations. It learns to pay attention to words that're important for the sentiment, such as negation words.

**Ablation study**. This is a study where each variant of the model removes one component at a time:

| Variant | Accuracy | Change | What It Shows |
|---|---|---|---|
| Full Model | 80.96% | — | Baseline |
| No Positional Encoding | 81.42% | +0.46% | Word order is largely irrelevant for short SST-2 phrases |
| Single-Head Attention | 80.39% | −0.57% | Multiple heads primarily speed up convergence |
| No Residual Connections | 50.92% | −30.04% | Complete training collapse — model predicts everything as positive |
| No Layer Normalization | 81.31% | +0.35% | Stability costs only; accuracy recovers |
The most surprising result is that the model fails to train without residual connections. This shows that residual connections are essential for the model to learn.

**Attention faithfulness test**. This test shows how important high-attention tokens are compared to ones:

| Tokens Removed (K) | Attention-Guided Drop | Random Drop | Ratio |
|---|---|---|---|
| 1 | −6.50% | −0.30% | 21.67× |
| 2 | −13.50% | −2.10% | 6.43× |
| 3 | −10.50% | −2.45% | 4.29× |

---

## How the Code Is Organised

The code is organized into files and folders:

```

project_final/

│

├── main.py                          # This is the file that runs the project

├── evaluate.py                      # This file is used to re-run analysis from saved checkpoints

├── train.py                         # This file contains the training loop

├── config.py                        # This file has all the hyperparameters

├── metrics.py                       # This file has functions to calculate metrics

│

├── data/

│   └── data_loader.py               # This file loads the SST-2 dataset

│

├── models/

│   ├── transformer.py               # This file implements the Transformer model

│   ├── bilstm.py                    # This file implements the Bidirectional LSTM model

│   ├── tfidf_lr.py                  # This file implements the TF-IDF + Logistic Regression model

│   └── transformer_variants.py      # This file implements the ablation variants

│

├── experiments/

│   ├── ablation.py                  # This file runs the ablation study

│   ├── attention_validation.py      # This file runs the attention validation test

│   └── scaling.py                   # This file runs the scaling experiments

│

└── interpretability/

├── attention_viz.py             # This file visualizes the attention weights

├── saliency.py                  # This file calculates the saliency scores

├── probing.py                   # This file runs the probing experiments

└── results_viz.py               # This file visualizes the results

```

The checkpoints are saved to `checkpoints/` and the figures and results are saved to `outputs/`.

---

## Getting Started

To run the project you need to install the required packages:

```bash

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install datasets scikit-learn matplotlib seaborn scipy numpy

```

You can then run the project using:

```bash

python main.py

```

This will run all the experiments and save the results to `outputs/`.

If you want to run a part of the project you can use the following commands:

```bash

python main.py --skip-ablation --skip-scaling

python main.py --only-ablation

python main.py --only-validation

```

You can also analyze your own sentences using:

```bash

python evaluate.py --sentence "the film is not good at all"

```

---

## The Architecture

The Transformer encoder follows the Vaswani et al. (2017) Design. The input is embedded into 256- vectors, scaled by √256 and then combined with sinusoidal positional encodings. The encoder has four stacked layers, each with -head self-attention and a position-wise feed-forward network. The output is passed to a two-class linear classifier.

The core attention operation is:

```

Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) * V

```

The positional encodings are:

```

PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))

PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

```

The feed-forward sublayer applies two transformations with a ReLU between them:

```

FFN(x) = max(0, x*W1. B1)*w2 + b2

```

The model has 6,927,618 trainable parameters. All linear layers are initialized with Xavier uniform initialization.

---


## How Overfitting Is Controlled

To prevent overfitting several things work together. The important thing is the dual-criterion early stopping rule. This rule stops the training when the validation accuracy does not improve for a number of epochs or when the difference between training and validation accuracy becomes too big.

The things that help prevent overfitting are:

- Label smoothing which makes the model less confident in its predictions

- criterion early stopping which stops the training when the validation accuracy does not improve or when the training-validation gap becomes too big

- Cosine annealing which reduces the learning rate over time

- Stochastic Weight Averaging which keeps an average of the models weights

- Gradient norm clipping which prevents the model from making big updates



## What Gets Generated

After running the model we get some files in the outputs folder.

These files include the metrics, model comparison metrics comparison, TF-IDF vs attention, confusion matrices, training and validation curves attention heat maps, saliency maps, probing plots, ablation plots and scaling plots.

## Optional: GloVe Embeddings

We can also use GloVe embeddings in our model.

To do this we need to download the GloVe file extract the 100d embeddings and put them in the data folder.

Then we can run the model as normal.

## Research Questions

We had three research questions:

1. Does attention-based representation outperform TF-IDF?

The answer is that the Transformer matches TF-IDF in accuracy but avoids the positive-class recall bias.

2. Do attention heads specialise?

The answer is that they do. They specialise in a consistent pattern.

3. Are attention weights causally important?

The answer is that they are partially but not completely.

## Hardware and Training Time

We ran our experiments on an NVIDIA GeForce RTX 4090 Laptop GPU.

The training time was 30-45 minutes for the three main models and 4-6 hours for the full pipeline.

## References

We used some papers in our research:

- Vaswani et al. (2017). *Attention Is All You Need.*

- Clark et al. (2019). *What Does BERT Look At? An Analysis of BERTs Attention.*

*Roshan Ghimire. MSAI, Khoury College of Computer Sciences, Northeastern University*