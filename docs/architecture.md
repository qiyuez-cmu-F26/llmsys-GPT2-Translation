# Architecture and Design Notes

This document describes the model, the data format and the training setup used in this project. It explains the design and the reasoning behind it, and deliberately contains no implementation code (this was a graded course assignment).

## 1. Overview

The model is a decoder-only Transformer in the style of GPT-2. It treats translation as language modelling: the German source sentence is a prefix, and the model continues it with the English translation.

```
input token ids  (B, S)
      │
      ├── token embedding      (B, S, 256)
      └── position embedding   (1, S, 256)      broadcast over the batch
      │
      ▼  add, then dropout
      │
      ▼  ┌──────────────────────────────────────────────┐
         │  Transformer layer  ×4  (pre-LN)             │
         │                                              │
         │   x ── LayerNorm ── Masked MHA ──┐           │
         │   │                              ▼           │
         │   └──────────────────────────── (+) ── x'    │
         │                                              │
         │   x' ── LayerNorm ── FeedForward ─┐          │
         │   │                               ▼          │
         │   └──────────────────────────── (+) ── x''   │
         └──────────────────────────────────────────────┘
      ▼
   final LayerNorm
      ▼
   Linear (256 → 10,000)
      ▼
   logits  (B, S, 10,000)
```

B is batch size and S is sequence length (up to 39 inputs per sequence, since the 40-token window is shifted by one for next-token prediction).

## 2. Configuration

| Item | Value |
|---|---|
| Layers | 4 |
| Attention heads | 8 (32 dims per head) |
| Embedding / hidden dim | 256 |
| Feed-forward hidden dim | 256 |
| Activation | GELU (tanh approximation) |
| Vocabulary | 10,000 (byte-level BPE) |
| Max positions | 40 |
| Dropout | 0.1 |
| LayerNorm epsilon | 1e-5 |
| Parameters | about 6.7M |

Rough parameter breakdown: token embeddings 2.56M, output head 2.57M, and about 0.4M per Transformer layer (1.6M for four). The embedding and output matrices dominate because the vocabulary is large relative to the hidden size.

## 3. Components

### 3.1 Embeddings

Tokens are mapped to vectors by a learned embedding table. Positions get their own learned embedding table (GPT-2 style, not sinusoidal). The two are summed and passed through dropout. The lookup is expressed as a one-hot matrix multiplied by the embedding table, which fits the tensor operations the framework provides and keeps the backward pass automatic.

### 3.2 Multi-head causal self-attention

1. The input `(B, S, D)` is projected by three separate linear layers into queries, keys and values, each `(B, S, D)`.
2. Each is reshaped to split `D` into `h` heads of size `D_h = D / h`, then permuted to `(B, h, S, D_h)`, so every head attends independently.
3. Keys are transposed on their last two axes so that a batched matrix multiply gives scores of shape `(B, h, S, S)`.
4. Scores are divided by `sqrt(D_h)`, and a causal mask is added: positions above the diagonal receive a very large negative value, so after the softmax each token attends only to itself and earlier tokens. A large finite constant is used rather than negative infinity to avoid NaNs.
5. Softmax over the last axis gives attention weights, followed by dropout.
6. The weights multiply the values, giving `(B, h, S, D_h)`. Heads are permuted back and merged into `(B, S, D)`.
7. An output projection mixes information across heads.

The mask has shape `(1, 1, S, S)` and relies on implicit broadcasting across batch and heads, so it is built once per forward pass rather than per example.

### 3.3 Pre-LN Transformer layer

Each layer normalizes *before* each sub-block and adds the sub-block output back onto the residual stream:

```
x = x + Attention(LayerNorm(x))
x = x + FeedForward(LayerNorm(x))
```

**Why pre-LN rather than post-LN.** In the post-LN design (normalize after the residual add), gradients near the output layers are larger at initialization, which typically forces a careful learning-rate warmup. Pre-LN keeps a clean identity path through the residual stream, gives better-behaved gradients and trains stably without warmup in most cases (Xiong et al., 2020). GPT-2 also uses pre-LN, with one extra LayerNorm after the last block, which is included here.

### 3.4 Feed-forward network

Two linear layers with a GELU in between, applied independently at every position, followed by dropout. Because the linear layer operates on 2D input, the `(B, S, D)` tensor is flattened to `(B·S, D)` for the matrix multiply and reshaped back afterwards.

### 3.5 LayerNorm

Normalizes each token vector across the feature dimension (mean and variance over the 256 features), then applies a learned per-feature scale (initialized to 1) and shift (initialized to 0). Statistics are computed per token, so the result does not depend on batch size or on other tokens in the sequence.

### 3.6 Dropout

Inverted dropout: during training each element is zeroed with probability `p`, and the survivors are scaled by `1 / (1 - p)` so the expected activation is unchanged. In evaluation mode dropout does nothing, so no rescaling is needed at inference. A bug worth remembering: without the `1/(1-p)` scaling, activations shrink by half in expectation at `p = 0.5`, and results differ from the reference.

### 3.7 Output head and loss

A final linear layer maps the 256-dim hidden state to 10,000 vocabulary logits at each position.

The loss is softmax cross-entropy, computed in its stable form:

```
loss(z, y) = log( Σ_i exp(z_i) ) − z_y
```

The `logsumexp` term is computed by subtracting the row maximum before exponentiating and adding it back afterwards, which avoids overflow for large logits without changing the value. The target logit `z_y` is selected with a one-hot mask. The loss is then multiplied by a per-token weight (0 for source and padding tokens, 1 for English target tokens) and normalized by the number of target tokens, so the model is only penalized for the translation, not for "predicting" the German input.

## 4. Data pipeline

**Tokenizer.** A byte-level BPE tokenizer is trained from scratch on both languages jointly, with vocabulary size 10,000 and three special tokens: `<eos_de>`, `<eos_en>` and `<pad>`. A shared vocabulary lets the model reuse subwords such as names and numbers across languages.

**Sequence layout.**

```
<German tokens> <eos_de> <English tokens> <eos_en> <pad> <pad> …
```

Sequences are truncated or padded to 40 tokens. Inputs are the first 39 tokens and labels are the same sequence shifted left by one, so at each position the model predicts the next token. The loss weight is 0 on the German part and padding, 1 on the English part.

**Filtering.** Pairs whose combined word count is 40 or more are removed so that most examples fit in the context window.

## 5. Decoding

Generation is greedy and processes one sentence at a time:

1. Tokenize the German sentence and append `<eos_de>`.
2. Run the model on the current token sequence and take the logits at the **last** position.
3. Pick the highest-scoring token (argmax) and append it.
4. Stop on `<eos_en>` or when the sequence reaches the context limit.
5. Decode the generated tokens back to text.

Because the model has no key/value cache, step 2 recomputes attention over the whole prefix every time. That is simple and correct, but the cost per sentence grows quadratically with output length. This is why generating 100 sentences takes several minutes while an epoch of batched training takes tens of minutes for 20,000 sentences.

## 6. Training procedure

| Item | Value |
|---|---|
| Optimizer | Adam |
| Learning rate | 0.001 (0.02 did not converge) |
| Batch size | 128 |
| Samples per epoch | 20,000 (randomly drawn) |
| Epochs | 20 |
| Gradient steps | about 3,140 |
| Precision | float32 |
| Hardware | 1× V100 16 GB |

Each epoch runs training, computes validation loss on the whole validation split, then generates translations for the 100 test sentences and scores BLEU.

Approximate timing per batch of 128 sequences: forward 5 s, backward 10 s, optimizer step 1 s. Backward being about twice the forward pass is the usual ratio.

## 7. Results summary

| Metric | Epoch 0 | Epoch 9 | Epoch 19 |
|---|---:|---:|---:|
| Validation loss | 4.68 | 2.93 | 2.50 |
| BLEU (100 sentences) | 8.1 | 16.5 | 20.6 |

Best BLEU was 21.2 at epoch 17. See the [main README](../README.md) for the learning curve and the learning-rate comparison.

## 8. Design decisions and what I learned

- **Learning rate mattered more than anything else.** The same model and code went from "does not learn" (lr 0.02) to about 20 BLEU (lr 0.001). Checking the loss trend after the first couple of epochs would have saved most of a wasted run.
- **Separate Q/K/V projections vs. one fused projection.** A single projection split three ways is faster; three separate ones are simpler and closer to the textbook description. At this model size the difference is negligible.
- **Masking the loss to target tokens** makes the objective match the task. Training on the source side as well would spend capacity on modelling German.
- **Test the whole pipeline on a tiny run first.** My decoding bug only surfaced after a full epoch of training. A 20-example, 1-epoch dry run through train, validate, generate and score would have caught it in a minute.
- **Evaluate with enough data.** With 100 test sentences BLEU jumps by about ±1 between epochs. Validation loss was the more trustworthy progress signal.

## 9. Possible extensions

- Key/value caching to make decoding linear in output length.
- Beam search instead of greedy decoding.
- Learning-rate warmup with cosine decay, label smoothing and weight tying between the input embedding and the output head.
- A larger evaluation set (or the full test split) for a less noisy BLEU.
- Mixed-precision training to speed up the 48-minute epochs.
