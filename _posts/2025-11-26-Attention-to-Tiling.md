
## What You Will Learn

This document provides a ground-up explanation of the attention mechanism and Flash Attention, with fully worked numerical examples you can verify by hand.

### 1. How Attention Works (Forward Pass)
- The intuition behind Query, Key, Value matrices
- Step-by-step computation: S = QKᵀ → P = softmax(S) → O = PV
- A complete 4×4 numerical example with every multiplication shown

### 2. How Backpropagation Works (Backward Pass)
- Computing gradients dV, dP, dS, dQ, dK from the loss
- The tricky softmax gradient derivation
- How weight matrices W_Q, W_K, W_V actually get updated
- Concrete numbers showing weights before and after one gradient step

### 3. The Memory Problem
- Why standard attention requires O(N²) memory
- The GPU memory hierarchy: HBM vs SRAM
- Why memory bandwidth (not compute) is the bottleneck

### 4. Flash Attention: The Solution
- The key insight: never materialize the full N × N matrices
- Online softmax: computing softmax incrementally with running statistics
- The rescaling trick: why we never need to "go back" and update previous values
- Tiled computation: processing small blocks that fit in SRAM
- Complete numerical trace showing the algorithm tile-by-tile
- Backward pass: recomputing P instead of storing it

> **The Core Insight:** Flash Attention achieves the *exact same result* as standard attention, but avoids storing the N × N attention matrix. Instead of O(N²) intermediate memory, it only needs O(N) extra storage (the running statistics m and ℓ) by processing small tiles in fast SRAM and deferring normalization until the end.

---

# Part I: Standard Attention

## 1. What is Attention?

### The Intuition

Attention is a mechanism that allows a model to focus on relevant parts of its input when producing output. Given a sequence of tokens, attention answers: *"For each token, how much should it look at every other token?"*

For example, in the sentence "The cat sat on the mat because **it** was tired":
- The word "it" needs to attend strongly to "cat" to understand what "it" refers to
- It should attend weakly to "mat" (less relevant)

### The Three Matrices: Q, K, V

Attention uses three projections of the input:

- **Query (Q)**: "What am I looking for?" - Each token's search vector
- **Key (K)**: "What do I contain?" - Each token's identifier vector
- **Value (V)**: "What information do I provide?" - The actual content to retrieve

The attention mechanism:
1. Compare each Query against all Keys (dot product) → relevance scores
2. Convert scores to probabilities (softmax) → attention weights
3. Use weights to take weighted sum of Values → output

### The Mathematical Formulation

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

Where:
- Q ∈ ℝ^(N × d) — N tokens, d-dimensional queries
- K ∈ ℝ^(N × d) — N tokens, d-dimensional keys
- V ∈ ℝ^(N × d) — N tokens, d-dimensional values
- √d_k is a scaling factor to prevent large dot products

---

## 2. A Concrete Example: 4×4 Attention

Let's work through attention step-by-step with a small example we can compute by hand.

### Setup

- Sequence length N = 4 (four tokens)
- Embedding dimension d = 4
- Scaling factor √d_k = 1 (ignored for simplicity)

### Input Matrices

```
Q = | 1  0  1  0 |    K = | 1  0  0  0 |    V = | 1   2   3   4  |
    | 0  1  0  1 |        | 0  1  0  0 |        | 5   6   7   8  |
    | 1  0  0  0 |        | 1  0  1  0 |        | 9   10  11  12 |
    | 0  1  0  0 |        | 0  1  0  1 |        | 13  14  15  16 |
```

We chose V with distinct values so we can trace which tokens contribute to the output.

---

## 3. Forward Pass: Step by Step

### Step 1: Compute Score Matrix S = QKᵀ

Each element S_ij measures how much token i's query matches token j's key:

$$
S_{ij} = Q_i \cdot K_j = \sum_{k} Q_{ik} \cdot K_{jk}
$$

**Computing Row 0 of S** (Query = [1,0,1,0]):

```
S₀₀ = [1,0,1,0] · [1,0,0,0] = 1·1 + 0·0 + 1·0 + 0·0 = 1
S₀₁ = [1,0,1,0] · [0,1,0,0] = 1·0 + 0·1 + 1·0 + 0·0 = 0
S₀₂ = [1,0,1,0] · [1,0,1,0] = 1·1 + 0·0 + 1·1 + 0·0 = 2
S₀₃ = [1,0,1,0] · [0,1,0,1] = 1·0 + 0·1 + 1·0 + 0·1 = 0
```

**Result:**

```
S = QKᵀ = | 1  0  2  0 |
          | 0  1  0  2 |
          | 1  0  1  0 |
          | 0  1  0  1 |
```

**Interpretation:** Token 0 has highest score (2) with Token 2—they will attend strongly to each other.

### Step 2: Apply Softmax to Get Attention Weights P

Softmax converts scores to probabilities (each row sums to 1):

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_{k} e^{S_{ik}}}
$$

Using e⁰ = 1, e¹ ≈ 2.718, e² ≈ 7.389:

**Row 0:** scores [1, 0, 2, 0]
```
exponentials = [e¹, e⁰, e², e⁰] = [2.718, 1, 7.389, 1]
sum = 12.107
P₀ = [2.718/12.107, 1/12.107, 7.389/12.107, 1/12.107]
   = [0.224, 0.083, 0.610, 0.083]
```

Token 0 attends 61% to Token 2, 22% to itself, and only 8% each to Tokens 1 and 3.

**All rows:**

```
P = | 0.224  0.083  0.610  0.083 |
    | 0.083  0.224  0.083  0.610 |
    | 0.366  0.134  0.366  0.134 |
    | 0.134  0.366  0.134  0.366 |
```

### Step 3: Compute Output O = PV

Each output row is a weighted combination of value rows:

$$
O_i = \sum_{j} P_{ij} \cdot V_j
$$

**Row 0:**
```
O₀ = 0.224·[1,2,3,4] + 0.083·[5,6,7,8] + 0.610·[9,10,11,12] + 0.083·[13,14,15,16]
   = [7.20, 8.20, 9.20, 10.20]
```

The output is weighted toward V₂ = [9,10,11,12] because Token 0 attended most strongly (61%) to Token 2.

**All rows:**

```
O = | 7.20   8.20   9.20   10.20 |
    | 9.88   10.88  11.88  12.88 |
    | 6.08   7.08   8.08   9.08  |
    | 7.92   8.92   9.92   10.92 |
```

---

## 4. Backward Pass: Computing Gradients

For training, we need gradients of the loss L with respect to Q, K, and V. Let dX = ∂L/∂X denote gradients.

### The Computation Graph

```
Q, K  →  S = QKᵀ  →  P = softmax(S)  →  O = PV
              ↑                              ↑
              K                              V
```

Backward pass flows gradients in reverse: dO → dP → dS → dQ, dK, and dO → dV.

### Gradient Formulas

**Step 1: Gradient w.r.t. V**

$$
dV = P^T \cdot dO
$$

**Step 2: Gradient w.r.t. P**

$$
dP = dO \cdot V^T
$$

**Step 3: Gradient w.r.t. S** (through softmax)

$$
D_i = \sum_j P_{ij} \cdot dP_{ij}
$$

$$
dS_{ij} = P_{ij} \cdot (dP_{ij} - D_i)
$$

**Step 4: Gradients w.r.t. Q and K**

$$
dQ = dS \cdot K
$$

$$
dK = dS^T \cdot Q
$$

### Concrete Example: Computing All Gradients

Assume the loss gradient dO arrives from the next layer:

```
dO = | 1  1  1  1 |
     | 0  0  0  0 |
     | 1  1  1  1 |
     | 0  0  0  0 |
```

(Rows 0 and 2 receive gradient signal; rows 1 and 3 don't contribute to loss.)

#### Step 1: Compute dV = Pᵀ · dO

```
dV = | 0.590  0.590  0.590  0.590 |
     | 0.217  0.217  0.217  0.217 |
     | 0.976  0.976  0.976  0.976 |
     | 0.217  0.217  0.217  0.217 |
```

**Interpretation:** Token 2's value embedding receives the largest gradient (0.976) because tokens 0 and 2 attended heavily to it.

#### Step 2: Compute dP = dO · Vᵀ

```
dP = | 10  26  42  58 |
     | 0   0   0   0  |
     | 10  26  42  58 |
     | 0   0   0   0  |
```

#### Step 3: Compute dS (softmax backward)

D₀ = 34.83, D₂ = 30.30, D₁ = D₃ = 0

```
dS = | -5.57  -0.73   4.38   1.91 |
     |  0      0      0      0    |
     | -7.42  -0.58   4.28   3.72 |
     |  0      0      0      0    |
```

**Interpretation:** Token 0 wants to *increase* attention to Tokens 2 and 3 (dS > 0) and *decrease* attention to Tokens 0 and 1 (dS < 0).

#### Step 4: Compute dQ and dK

```
dQ = | -1.19   1.18   4.38   1.91 |
     |  0      0      0      0    |
     | -3.14   3.14   4.28   3.72 |
     |  0      0      0      0    |

dK = | -12.99  0  -5.57  0    |
     | -1.31   0  -0.73  0    |
     |  8.66   0   4.38  0    |
     |  5.64   0   1.91  0    |
```

### What Gets Updated During Training

> **Key Insight:** Q, K, V are **not directly trainable**—they're computed from the input X via learned weight matrices:
>
> Q = X·W_Q,  K = X·W_K,  V = X·W_V
>
> The gradients flow back to update the weights:
>
> dW_Q = Xᵀ·dQ,  dW_K = Xᵀ·dK,  dW_V = Xᵀ·dV

#### Example: Weight Updates (learning rate η = 0.1)

With X = I (identity), W = Q, K, V directly.

**Update W_V:**
```
W_V_new = | 0.94   1.94   2.94   3.94  |
          | 4.98   5.98   6.98   7.98  |
          | 8.90   9.90   10.90  11.90 |
          | 12.98  13.98  14.98  15.98 |
```

Token 2's value decreased most (by 0.098) because it received the largest gradient.

**Update W_Q[0]:**
```
W_Q_new[0] = [1, 0, 1, 0] - 0.1·[-1.19, 1.18, 4.38, 1.91] = [1.12, -0.12, 0.56, -0.19]
```

**Update W_K[0]:**
```
W_K_new[0] = [1, 0, 0, 0] - 0.1·[-12.99, 0, -5.57, 0] = [2.30, 0, 0.56, 0]
```

| Gradient | Effect on Weights |
|----------|-------------------|
| dV large | Value embedding shrinks (attended-to tokens) |
| dQ positive | Query shifts away from matching certain keys |
| dK negative | Key becomes more attractive to queries |

### What We Need to Store

For the backward pass, we must keep in memory:
- Q, K, V (the inputs) — O(Nd) each
- P (the attention weights) — O(N²) **← This is the problem!**

---

# Part II: The Memory Problem

## Why Standard Attention Doesn't Scale

### The Quadratic Bottleneck

| Sequence Length N | Matrix Size N² | Memory (FP16) |
|-------------------|----------------|---------------|
| 512 | 262K | 0.5 MB |
| 2,048 | 4.2M | 8 MB |
| 8,192 | 67M | 134 MB |
| 32,768 | 1.07B | 2.1 GB |
| 131,072 | 17.2B | 34 GB |

Modern LLMs want context lengths of 100K+ tokens. The N² matrices alone would exceed GPU memory!

### The Memory Hierarchy Problem

```
┌─────────────────┐     SLOW      ┌─────────────────┐     FAST      ┌─────────────────┐
│  HBM (Main)     │ ───────────→  │  SRAM (On-chip) │ ───────────→  │    Compute      │
│  40-80 GB       │  Bottleneck!  │  ~20 MB total   │               │  312 TFLOPS     │
│  1.5-2.0 TB/s   │               │  ~19 TB/s       │               │                 │
└─────────────────┘               └─────────────────┘               └─────────────────┘
```

> **Key Insight:** The GPU can compute 312 trillion operations per second, but can only transfer 2 trillion bytes per second from HBM. **Memory bandwidth, not compute, is the bottleneck.**

### Standard Attention Memory Access Pattern

1. **Read** Q, K from HBM → compute S → **Write** S to HBM
2. **Read** S from HBM → compute P → **Write** P to HBM
3. **Read** P, V from HBM → compute O → **Write** O to HBM

That's **4 round-trips** to HBM for N × N matrices—O(N²) memory traffic!

---

# Part III: Flash Attention: The Solution

## The Core Idea

> **Key Insight:** **Never materialize the full N × N matrices.** Instead:
> 1. Process small **tiles** that fit in SRAM
> 2. Compute softmax **incrementally** using running statistics
> 3. Only write the final output O to HBM

But wait—softmax needs to see *all* scores in a row to compute the normalizing denominator. How can we compute it tile-by-tile?

---

## The Online Softmax Trick

### The Challenge

Softmax requires the sum over *all* elements:

$$
P_{ij} = \frac{e^{S_{ij}}}{\sum_{k=1}^{N} e^{S_{ik}}}
$$

If we only see scores [S_i,0, S_i,1] first, then later [S_i,2, S_i,3], how can we get the correct result?

### The Key Insight: Defer Normalization

We want to compute:

$$
O_i = \frac{\sum_{j} e^{S_{ij}} \cdot V_j}{\sum_k e^{S_{ik}}} = \frac{\text{numerator}}{\text{denominator}}
$$

> **Key Insight:** **Both numerator and denominator can be accumulated incrementally!**
> - Track running sum: ℓ = Σⱼ exp(S_ij - m) (denominator, shifted by max)
> - Track running weighted sum: O_unnorm = Σⱼ exp(S_ij - m) · Vⱼ (numerator)
> - At the end: O = O_unnorm / ℓ

### The Rescaling Trick

When we see new scores with a larger max, we must **rescale** our accumulated sums:

> **When processing a new tile with max m̃:**
> 1. Update max: m_new = max(m_old, m̃)
> 2. Compute correction: c = exp(m_old - m_new)  (Note: c ≤ 1)
> 3. Rescale old sums: ℓ_new = ℓ_old · c + (new exponentials)
> 4. Rescale old output: O_new = O_old · c + (new weighted terms)

### Why We Never Go Back

> **The Crucial Insight:** **We never revisit or update previous values.** We only:
> 1. Maintain running sums (ℓ and O_unnorm)
> 2. Scale the *entire accumulated sum* when max changes
> 3. Normalize once at the very end
>
> The correction factor c applied to the sum is equivalent to applying it to each term individually (distributive property).

---

## Tiled Computation

### Dividing into Tiles

With block size B=2, we divide our 4 × 4 matrices into 2 × 2 tiles:

```
         K rows 0-1    K rows 2-3
        ┌───────────┬───────────┐
Q rows  │   S_00    │   S_01    │
 0-1    │    ①      │    ②      │
        ├───────────┼───────────┤
Q rows  │   S_10    │   S_11    │
 2-3    │    ③      │    ④      │
        └───────────┴───────────┘
```

**Key:** We never store this full matrix. We compute one tile at a time in SRAM.

---

## Flash Attention Forward: Complete Trace

### Initialization

**In HBM (slow memory):**
- m = [-∞, -∞, -∞, -∞]ᵀ (max score per row)
- ℓ = [0, 0, 0, 0]ᵀ (sum of exponentials per row)
- O = 0₄ₓ₄ (output accumulator, unnormalized)

### Tile (0,0): Q Rows 0-1 × K Rows 0-1

**Load into SRAM:**
```
Q₀₋₂ = | 1  0  1  0 |    K₀₋₂ = | 1  0  0  0 |    V₀₋₂ = | 1  2  3  4 |
       | 0  1  0  1 |           | 0  1  0  0 |           | 5  6  7  8 |
```

**Step 1: Compute tile scores**
```
S₀₀ = Q₀₋₂ · K₀₋₂ᵀ = | 1  0 |
                      | 0  1 |
```

**Step 2: Find tile max (per row):** m̃₀ = 1, m̃₁ = 1

**Step 3: Update global max:** m₀ = 1, m₁ = 1

**Step 4: Correction factors:** c₀ = exp(-∞ - 1) = 0, c₁ = 0 (first tile: old values zeroed out)

**Step 5: Local exponentials**
```
P̃ = | exp(1-1)  exp(0-1) | = | 1      0.368 |
    | exp(0-1)  exp(1-1) |   | 0.368  1     |
```

**Step 6: Update ℓ:** ℓ₀ = 1.368, ℓ₁ = 1.368

**Step 7: Update O**
```
O₀ = 1·[1,2,3,4] + 0.368·[5,6,7,8] = [2.84, 4.21, 5.57, 6.94]
O₁ = 0.368·[1,2,3,4] + 1·[5,6,7,8] = [5.37, 6.74, 8.10, 9.47]
```

### Tile (0,1): Q Rows 0-1 × K Rows 2-3

**Load:** K₂₋₄, V₂₋₄ = [[9,10,11,12], [13,14,15,16]]

**Step 1: Tile scores**
```
S₀₁ = | 2  0 |
      | 0  2 |
```

**Step 2-3: Update global max:** m₀ = 2, m₁ = 2 **(max increased!)**

> **Key Insight:** The max increased! Our previous exponentials were exp(s-1), but should be exp(s-2). The correction factor fixes this without revisiting old data.

**Step 4: Correction factors:** c = exp(1-2) = 0.368

**Step 5: Local exponentials**
```
P̃ = | 1      0.135 |
    | 0.135  1     |
```

**Step 6: Update ℓ (with rescaling)**
```
ℓ₀ = 1.368 · 0.368 + (1 + 0.135) = 1.638
```

**Step 7: Update O (with rescaling)**
```
O₀ = [2.84, 4.21, 5.57, 6.94]·0.368 + 1·[9,10,11,12] + 0.135·[13,14,15,16]
   = [11.81, 13.44, 15.08, 16.71]
```

**Normalize (end of Q block 0):**
```
O₀_final = [11.81, 13.44, 15.08, 16.71] / 1.638 = [7.20, 8.20, 9.20, 10.20]
O₁_final = [16.20, 17.83, 19.47, 21.10] / 1.638 = [9.88, 10.88, 11.88, 12.88]
```

**These match our standard attention output!**

---

## Flash Attention Backward Pass

### The Challenge

Standard backprop needs the P matrix. But we didn't store it!

### The Solution: Recomputation

> **Key Insight:** We stored m and ℓ (only O(N) memory). From these, we can **recompute** any tile:
>
> $$
> P_{ij} = \frac{e^{S_{ij} - m_i}}{\ell_i}
> $$
>
> We trade extra compute for memory savings. Since GPUs are memory-bound, this is a good trade!

### Backward Algorithm

**Saved from forward:** Q, K, V, O, m, ℓ

**Input:** dO

**Precompute:** D_i = Σⱼ dO_ij · O_ij (row-wise)

**For each** Q block i, **for each** K/V block j:
1. Load Q_i, K_j, V_j, m_i, ℓ_i, dO_i into SRAM
2. **Recompute:** S_ij = Q_i · K_jᵀ, then P_ij = exp(S_ij - m_i) / ℓ_i
3. Compute gradients:
   - dV_j += P_ijᵀ · dO_i
   - dP_ij = dO_i · V_jᵀ
   - dS_ij = P_ij ⊙ (dP_ij - D_i)
   - dQ_i += dS_ij · K_j
   - dK_j += dS_ijᵀ · Q_i

---

# Part IV: Summary

## Comparison

| Aspect | Standard Attention | Flash Attention |
|--------|-------------------|-----------------|
| Score matrix S | Store N × N in HBM | Compute tiles in SRAM, discard |
| Attention matrix P | Store N × N in HBM | **Never stored** |
| Backward pass | Read P from HBM | Recompute P tiles from m, ℓ |
| HBM reads/writes | O(N²d) for S, P | O(N²d²/M) where M = SRAM |
| Extra storage | O(N²) for S, P | O(N) for m, ℓ only |

The HBM traffic reduction comes from reading Q, K, V in tiles that fit in SRAM, avoiding repeated round-trips for the N × N intermediate matrices.

---

## Code Verification

```python
import torch
import torch.nn.functional as F

def flash_attention(Q, K, V, block_size=2):
    N, d = Q.shape
    m = torch.full((N,), float('-inf'))
    l = torch.zeros(N)
    O = torch.zeros(N, d)

    for i in range(0, N, block_size):
        Qi = Q[i:i+block_size]
        mi, li, Oi = m[i:i+block_size], l[i:i+block_size], O[i:i+block_size]

        for j in range(0, N, block_size):
            Kj, Vj = K[j:j+block_size], V[j:j+block_size]
            Sij = Qi @ Kj.T
            m_tilde = Sij.max(dim=1).values
            m_new = torch.maximum(mi, m_tilde)
            c = torch.exp(mi - m_new)
            P_tilde = torch.exp(Sij - m_new.unsqueeze(1))
            li = li * c + P_tilde.sum(dim=1)
            Oi = Oi * c.unsqueeze(1) + P_tilde @ Vj
            mi = m_new

        O[i:i+block_size] = Oi / li.unsqueeze(1)
    return O

# Test
Q = torch.tensor([[1.,0.,1.,0.], [0.,1.,0.,1.], [1.,0.,0.,0.], [0.,1.,0.,0.]])
K = torch.tensor([[1.,0.,0.,0.], [0.,1.,0.,0.], [1.,0.,1.,0.], [0.,1.,0.,1.]])
V = torch.tensor([[1.,2.,3.,4.], [5.,6.,7.,8.], [9.,10.,11.,12.], [13.,14.,15.,16.]])

O_std = F.softmax(Q @ K.T, dim=-1) @ V
O_flash = flash_attention(Q, K, V)
print("Match:", torch.allclose(O_std, O_flash, atol=1e-4))  # True
```

---

## Key Takeaways

1. **Attention** computes weighted sums of values based on query-key similarity
2. **Standard attention** stores O(N²) matrices, causing memory bottlenecks
3. **Flash Attention** processes tiles that fit in fast SRAM memory
4. **Online softmax** allows incremental computation by:
   - Tracking running max m and sum ℓ
   - Rescaling accumulated values when max increases
   - Normalizing once at the end
5. **Backward pass** recomputes P tiles using saved m, ℓ statistics (only O(N) extra storage)
6. **Result:** Avoids storing N × N matrices, significantly reducing HBM traffic and enabling 2-4× speedups

---

## References

- Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). *FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness*. NeurIPS. [arXiv:2205.14135](https://arxiv.org/abs/2205.14135)

- Dao, T. (2023). *FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning*. [arXiv:2307.08691](https://arxiv.org/abs/2307.08691)
