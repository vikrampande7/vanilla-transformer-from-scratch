# 🤖 Transformer from Scratch — *Attention Is All You Need*

> A clean, from-scratch PyTorch implementation of the original Transformer architecture, trained on the **IITB English→Hindi** translation dataset, with **attention visualization** support.

📄 **Paper**: [Attention Is All You Need — Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)

---

## 📋 Table of Contents

- [Motivation: Why Not RNNs?](#-motivation-why-not-rnns)
- [How Attention Works](#-how-attention-works)
- [Architecture Deep Dive](#-architecture-deep-dive)
  - [Input Embeddings](#1-input-embeddings)
  - [Positional Encoding](#2-positional-encoding-sinusoidal)
  - [Self-Attention Block](#3-self-attention-block)
  - [Multi-Head Attention](#4-multi-head-attention)
  - [Positionwise Feed-Forward Network](#5-positionwise-feed-forward-network)
  - [Add & Norm (Residual + LayerNorm)](#6-add--norm-residual-connection--layernorm)
  - [Encoder Architecture](#7-encoder-architecture)
  - [Decoder Architecture](#8-decoder-architecture)
  - [Cross-Attention](#9-cross-attention)
  - [Softmax & Output Projection](#10-softmax--output-projection)
- [Training](#-training)
- [Greedy Decoding](#-greedy-decoding)
- [Dataset](#-dataset-iitb-english--hindi)
- [Attention Visualization](#-attention-visualization)
- [Results](#-results)
- [Usage](#-usage)

---

## 🧠 Motivation: Why Not RNNs?

Before Transformers, sequence-to-sequence tasks (like translation) were dominated by **Recurrent Neural Networks (RNNs)** and their variants (LSTMs, GRUs). However, they suffered from fundamental architectural limitations:

| Problem | Description |
|---|---|
| **Sequential Bottleneck** | RNNs process tokens one-by-one. Token `t` cannot be computed until token `t-1` is done. This makes training on long sequences very slow and prevents parallelization across time steps. |
| **Vanishing Gradients** | Gradients must flow back through every time step. In long sequences, they vanish (or explode), making it hard for the model to learn long-range dependencies. |
| **Information Bottleneck** | In encoder-decoder RNNs, the entire input sequence is compressed into a single fixed-size context vector — a severe bottleneck for long sentences. |
| **Memory of distant tokens** | Even with LSTMs, remembering information from many steps ago is unreliable; the hidden state has a fixed capacity. |

The Transformer solves all of these by **eliminating recurrence entirely** and replacing it with **self-attention**, which directly connects every token to every other token in a single step — enabling full parallelism and O(1) path length between any two positions.

---

## How Attention Works

The core idea is simple: *when encoding a word, look at all other words in the sequence and decide how much each one matters.*

Given a sequence of tokens, we project each token into three vectors:
- **Query (Q)** — *"What am I looking for?"*
- **Key (K)** — *"What do I contain?"*
- **Value (V)** — *"What will I contribute if selected?"*

Attention is then computed as:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- The dot product $QK^T$ scores how relevant each key is to each query.
- Dividing by $\sqrt{d_k}$ (dimension of keys) prevents softmax from saturating in high dimensions — keeping gradients healthy.
- The softmax turns scores into a probability distribution (weights that sum to 1).
- These weights multiply the **Values**, producing a weighted sum — the attended output.

Every token attends to every other token in **parallel**, solving both the bottleneck and the sequential processing problem.

---

## Architecture Deep Dive

```
Input Tokens
     │
     ▼
[Input Embeddings] + [Positional Encoding]
     │
     ▼
┌─────────────────────────────┐
│        ENCODER (×N)         │
│  ┌─────────────────────┐    │
│  │  Multi-Head          │    │
│  │  Self-Attention      │    │
│  └────────┬────────────┘    │
│           │ Add & Norm       │
│  ┌────────▼────────────┐    │
│  │  Feed-Forward Net   │    │
│  └────────┬────────────┘    │
│           │ Add & Norm       │
└───────────┼─────────────────┘
            │ (Memory)
            ▼
┌─────────────────────────────┐
│        DECODER (×N)         │
│  ┌─────────────────────┐    │
│  │  Masked Multi-Head   │    │
│  │  Self-Attention      │    │
│  └────────┬────────────┘    │
│           │ Add & Norm       │
│  ┌────────▼────────────┐    │
│  │  Cross-Attention     │◄───┤ (from Encoder)
│  └────────┬────────────┘    │
│           │ Add & Norm       │
│  ┌────────▼────────────┐    │
│  │  Feed-Forward Net   │    │
│  └────────┬────────────┘    │
│           │ Add & Norm       │
└───────────┼─────────────────┘
            │
     [Linear Projection]
            │
       [Softmax]
            │
     Output Token Probabilities
```

---

### 1. Input Embeddings

**What:** A learned lookup table that maps each integer token ID to a dense vector of size `d_model`.

**Why:** Neural networks can't process raw integers or one-hot vectors efficiently. Embeddings place tokens in a continuous vector space where semantically similar words are geometrically close. The paper scales embeddings by $\sqrt{d_{model}}$ to keep their magnitude compatible with the positional encodings added to them.

```python
class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)
```

---

### 2. Positional Encoding (Sinusoidal)

**What:** A fixed (non-learned) vector added to each token embedding that encodes its **position** in the sequence.

**Why:** Attention is inherently **permutation-invariant** — it doesn't naturally know if token A comes before token B. Without positional encoding, shuffling the input would give the same output. The paper uses sinusoids at different frequencies:

$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

**Why sinusoids specifically?**
- They allow the model to generalize to sequence lengths longer than those seen during training.
- The relative position between tokens can be expressed as a linear function of any fixed offset — the model can easily learn to attend by relative position.
- Each dimension oscillates at a different frequency, giving a unique fingerprint to each position.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)
        return self.dropout(x)
```

---

### 3. Self-Attention Block

**What:** The scaled dot-product attention mechanism applied within a single sequence (each token attending to all others in the same sequence).

**Why:** Self-attention allows each position in the encoder to relate to all other positions — capturing syntactic and semantic dependencies regardless of distance. In the decoder, **masked** self-attention prevents each position from attending to future tokens (preserving autoregressive causality during training).

The `mask` parameter serves two roles:
- **Encoder padding mask**: Prevents attending to `[PAD]` tokens.
- **Decoder causal mask**: An upper-triangular mask of `-inf` that blocks future token visibility.

```python
  @staticmethod
  def attention(query, key, value, mask, dropout: nn.Dropout):
    d_k = query.shape[-1]
    attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
      attention_scores.masked_fill_(mask == 0, -1e9)
    attention_scores = attention_scores.softmax(dim = -1)
    if dropout is not None:
      attention_scores = dropout(attention_scores)
    return (attention_scores @ value), attention_scores
```

---

### 4. Multi-Head Attention

**What:** Instead of computing attention once with full `d_model` dimensions, the model runs `h` parallel attention heads, each working in a smaller subspace of dimension $d_k = d_{model}/h$. Their outputs are concatenated and linearly projected.

**Why:** Different heads can specialize in different types of relationships simultaneously — one head might learn syntactic agreement, another coreference, another local context. A single attention head would have to average all these patterns into one set of weights, losing nuance. The paper uses `h=8` heads.

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$
$$\text{where head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

```python
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        Q = self.w_q(q)
        K = self.w_k(k)
        V = self.w_v(v)
        # Split into heads: (batch, h, seq_len, d_k)
        B = Q.size(0)
        Q = Q.view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.h, self.d_k).transpose(1, 2)
        x, self.attention_scores = MultiHeadAttentionBlock.attention(Q, K, V, mask, self.dropout)
        # Concat heads: (batch, seq_len, d_model)
        x = x.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.w_o(x)
```

---

### 5. Positionwise Feed-Forward Network

**What:** A two-layer MLP applied **independently and identically** to each position:

$$\text{FFN}(x) = \max(0,\ xW_1 + b_1)W_2 + b_2$$

The inner dimension is expanded to `d_ff = 2048` (4× `d_model = 512`), then projected back.

**Why:** The attention sublayer is great at *mixing information across positions*, but it's largely linear. The FFN introduces **non-linearity** and increases the model's representational capacity. It acts as a per-position "memory" that can store and transform pattern information. Because it's applied identically to each position, it still doesn't mix positions — the two sublayers have a clean division of labour.

```python
class PWFFN(nn.Module):

  def __init__(self, d_model: int, dff: int, dropout: float):
    super().__init__() # Get attributes of Parent Class nn.Module
    self.linear_1 = nn.Linear(d_model, dff) # W1 and B1, Bias is already defined
    self.dropout = nn.Dropout(dropout)
    self.linear_2 = nn.Linear(dff, d_model) # W2 and B2
    self.activation = nn.ReLU()

  def forward(self, x):
    # d_model = 512 and dff = 2048 in paper
    return self.linear_2(self.dropout(self.activation(self.linear_1(x))))
```

---

### 6. Add & Norm (Residual Connection + LayerNorm)

**What:** After every sub-layer (attention or FFN), the input is **added back** (residual) and then **layer-normalized**:

$$\text{Output} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

**Why — Residual Connections:**
Deep networks suffer from vanishing gradients. Residual connections (from ResNet) create a *highway* for gradients to flow directly through layers unchanged, making it possible to stack many layers (the paper uses N=6). They also allow each layer to focus on learning a *residual correction* rather than a full transformation.

**Why — Layer Normalization:**
Unlike BatchNorm (which normalizes across the batch), LayerNorm normalizes across the **feature dimension** for each token independently. This is critical because:
- Sequence lengths vary (batches aren't uniform).
- It works well even with batch size 1.
- It stabilizes activations, enabling higher learning rates and faster convergence.

```python
class LayerNormalization(nn.Module):
    def __init__(self, features: int, eps: float = 1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class ResidualConnection(nn.Module):
    def __init__(self, features: int, dropout: float):
        super().__init__()
        self.norm = LayerNormalization(features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
```

> **Note:** The paper applies LayerNorm *after* the residual (`Post-LN`). This implementation uses `Pre-LN` (norm before sublayer), which is now more commonly preferred for training stability.

---

### 7. Encoder Architecture

**What:** A stack of `N=6` identical encoder layers, each containing:
1. Multi-Head **Self**-Attention (all tokens attend to all tokens)
2. Positionwise FFN
Each with its own Add & Norm wrapper.

**Why:** Stacking layers allows the model to build progressively more abstract representations. Lower layers tend to capture local syntactic patterns; higher layers capture longer-range semantic relationships. The encoder reads the entire source sequence and produces a rich, context-aware representation for every position — this becomes the "memory" the decoder draws from.

```python
class EncoderBlock(nn.Module):

  def __init__(self, self_attention_block: MultiHeadAttentionBlock, ffn: PWFFN, dropout: float):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.ffn = ffn
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

  def forward(self, x, src_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
    x = self.residual_connection[1](x, self.ffn)
    return x
```

---

### 8. Decoder Architecture

**What:** A stack of `N=6` identical decoder layers, each containing **three** sublayers:
1. **Masked** Multi-Head Self-Attention (causal — can only attend to past/current tokens)
2. **Cross-Attention** (attends to the encoder output)
3. Positionwise FFN

**Why three sublayers?** The decoder must do two things: (1) model the target sequence autoregressively, and (2) condition on the source sequence. The masked self-attention handles (1); cross-attention handles (2); the FFN handles non-linear processing of the combined information.

The **causal mask** (look-ahead mask) ensures that during training, position `i` in the decoder only attends to positions `≤ i`, preserving the autoregressive property — the model cannot "cheat" by looking at future target tokens.

```python
class DecoderBlock(nn.Module):

  def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, ffn: PWFFN, dropout: float):
    super().__init__()
    self.self_attention_block = self_attention_block
    self.cross_attention_block = cross_attention_block
    self.ffn = ffn
    self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

  def forward(self, x, encoder_output, src_mask, tgt_mask):
    x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
    x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
    x = self.residual_connection[2](x, self.ffn)
    return x
```

---

### 9. Cross-Attention

**What:** The second attention sublayer inside each decoder block. The **Query** comes from the decoder's current state; the **Keys** and **Values** come from the encoder output.

**Why:** This is the bridge between source and target. For each target token being generated, cross-attention asks: *"Which source positions are most relevant to generating this target word?"* The encoder memory (K, V) is fixed — only the decoder query changes at each step. This is how the model "reads" the source sentence while generating each target word, replacing the fixed-size context vector bottleneck of RNN seq2seq models.

```
Decoder state (Q) ──────┐
                         ▼
Encoder output (K, V) ──► Cross-Attention ──► context-aware decoder state
```

---

### 10. Softmax & Output Projection

**What:** After the final decoder layer, a **linear projection** maps from `d_model` to `vocab_size`, followed by a **softmax** to produce a probability distribution over the target vocabulary.

**Why:** The decoder produces a `d_model`-dimensional vector per position. The linear layer (whose weights are often tied to the input embedding matrix — *weight tying*) projects this to vocabulary logits. Softmax then converts logits to probabilities summing to 1, from which we can either sample or take the argmax.

**Label Smoothing** is applied during training (ε = 0.1): instead of a hard 1 for the correct token, the target distribution spreads 0.1 uniformly across all tokens. This prevents the model from becoming overconfident and improves generalization.

```python
class Projection(nn.Module):

  def __init__(self, vocab_size: int, d_model: int) :
    super().__init__()
    self.projection = nn.Linear(d_model, vocab_size)

  def forward(self, x):
    return torch.log_softmax(self.projection(x), dim=-1)
```

---

## Training

### Loss Function

**Cross-Entropy Loss** with **Label Smoothing** (ε = 0.1), as used in the paper:

```python
loss_fn = nn.CrossEntropyLoss(
    ignore_index=tokenizer.token_to_id('[PAD]'),
    label_smoothing=0.1
)
```

- `ignore_index`: PAD tokens don't contribute to the loss.
- `label_smoothing`: Softens target distribution to improve generalization and calibration.

### Optimizer

**Adam** optimizer with the paper's custom **learning rate schedule** (warmup + inverse square root decay):

<!-- $$lr = d_{model}^{-0.5} \cdot \min\left(\text{step}^{-0.5},\ \text{step} \cdot \text{warmup\_steps}^{-1.5}\right)$$ -->

```python
optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'], eps=1e-9)
```

The warmup phase (4000 steps) linearly increases LR before the decay begins. This gives the model time to establish stable gradients before large updates destabilize training.

### Teacher Forcing

During training, the decoder receives the **ground-truth target tokens** as input (shifted right), rather than its own predictions. This stabilizes training but creates an exposure bias — at inference, the model sees its own (possibly wrong) outputs. This is standard practice for Transformers.

### Hyperparameters

| Parameter | Value |
|---|---|
| `d_model` | 512 |
| `N` (layers) | 6 |
| `h` (heads) | 8 |
| `d_ff` | 2048 |
| `dropout` | 0.1 |
| `warmup_steps` | 4000 |
| `batch_size` | 32 |
| `seq_len` | 512 |

---

## Greedy Decoding

**What:** At inference time, generate the translation token-by-token. At each step, pick the single highest-probability token from the softmax output (argmax).

**Why greedy?** It's simple and fast. The alternative — **beam search** (keeping the top-k partial sequences at each step) — produces higher BLEU scores but is significantly slower. Greedy decoding often works well enough for demonstration and is the baseline decoding strategy.

```python
def greedy_decode(model, source, source_mask, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.tensor([[sos_idx]], device=device)

    while decoder_input.size(1) < max_len:
        decoder_mask = causal_mask(decoder_input.size(1)).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])          # logits for last token
        next_token = prob.argmax(dim=-1).item()   # greedy pick
        decoder_input = torch.cat([
            decoder_input,
            torch.tensor([[next_token]], device=device)
        ], dim=1)
        if next_token == eos_idx:
            break

    return decoder_input.squeeze(0)
```

---

##  Dataset: IITB English → Hindi

This model is trained on the **IIT Bombay English-Hindi Parallel Corpus** — one of the largest publicly available En-Hi translation datasets.

- **Source**: [IITB English-Hindi Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/)
- **HuggingFace**: [`cfilt/iitb-english-hindi`](https://huggingface.co/datasets/cfilt/iitb-english-hindi)
- **Size**: ~1.6 million sentence pairs (train), with dev and test splits
- **Tokenization**: Trained separate **WordPiece / BPE** tokenizers for English and Hindi using the `tokenizers` library

```python
from datasets import load_dataset
dataset = load_dataset("cfilt/iitb-english-hindi")
```

---

##  Attention Visualization

Attention weights from each head in each layer are extracted and visualized as heatmaps. This provides interpretability into what the model has learned:

- **Encoder self-attention**: Which source words attend to which?
- **Decoder self-attention**: How does the decoder model target-side dependencies?
- **Cross-attention**: Which source words does the decoder look at when generating each target word?

```python
# After a forward pass, attention scores are stored in each MHA block
attention_weights = model.decoder.layers[layer_idx]\
                         .cross_attention_block.attention_scores
# Shape: (batch, heads, tgt_len, src_len)
```

Example visualization (cross-attention):

```
Source:  The  cat  sat  on   the  mat  .
         ┌────────────────────────────────┐
बिल्ली  │ 0.02 0.91 0.01 0.01 0.02 0.01 0.02│
बैठी    │ 0.01 0.03 0.88 0.04 0.01 0.02 0.01│
चटाई   │ 0.01 0.01 0.02 0.02 0.05 0.87 0.02│
पर      │ 0.01 0.02 0.02 0.89 0.03 0.01 0.02│
         └────────────────────────────────┘
```

---

## Results

| Metric | Value |
|---|---|
| Dataset | IITB En→Hi |
| Epochs trained | 50 |
| BLEU Score | - |
| Decoding | Greedy |


## References

A great thanks to Umar Jamil for Transformer Demonstration.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al., 2017
- [Coding Transformer from Scratch](https://www.youtube.com/watch?v=ISNdQcPhsts) - Umar Jamil
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) - Jay Alammar
- [IITB English-Hindi Corpus](http://www.cfilt.iitb.ac.in/iitb_parallel/) - IIT Bombay
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) - Harvard NLP

---

<p align="center">
  Built with ❤️ and a lot of matrix multiplications.
</p>