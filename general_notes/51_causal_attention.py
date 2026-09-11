import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    1. **Self-Attention**:
      <span style="color:cyan">
      $$
      \text{Output} = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V
      $$
      </span>

    2. **Causal Attention**:
      <span style="color:cyan">
      $$
     \quad \text{Output} = \text{softmax}\left(\frac{Q K^T + M}{\sqrt{d_k}}\right) V
      $$
      </span>

    3. **Cross-Attention**:
      <span style="color:cyan">
      $$
      \text{Output} = \text{softmax}\left(\frac{Q_{\text{decoder}} K_{\text{encoder}}^T}{\sqrt{d_k}}\right) V_{\text{encoder}}
      $$
      </span>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 1. **Self-Attention (Scaled Dot-Product Attention)**

    - **Input**: $X \in \mathbb{R}^{n \times d}$, where $n$ is the sequence length and $d$ is the embedding dimension.

    - **Weight Matrices**:
      - $W_q \in \mathbb{R}^{d \times d_k}$, $W_k \in \mathbb{R}^{d \times d_k}$, $W_v \in \mathbb{R}^{d \times d_v}$

    - **Queries, Keys, Values**:
      - $Q = X W_q$, $K = X W_k$, $V = X W_v$

    - **Attention Scores**:
      $$
      A = \frac{Q K^T}{\sqrt{d_k}}
      $$

    - **Softmax**:
      $$
      \text{Attention Weights} = \text{softmax}(A)
      $$

    - **Output**:
      $$
      \text{Output} = \text{softmax}(A) V
      $$

    ---

    #### 2. **Causal Attention (Masked Self-Attention)**

    - **Mask**: $M \in \mathbb{R}^{n \times n}$, $M_{ij} = -\infty$ for $i < j$, $M_{ij} = 0$ for $i \geq j$

      *The mask ensures that each token can only attend to itself and previous tokens, preventing future information from influencing the current token.*

    - **Attention Scores with Mask**:
      $$
      A' = \frac{Q K^T + M}{\sqrt{d_k}}
      $$

    - **Softmax**:
      $$
      \text{Attention Weights} = \text{softmax}(A')
      $$

    - **Output**:
      $$
      \text{Output} = \text{softmax}(A') V
      $$

    ---

    #### 3. **Cross-Attention (Encoder-Decoder Attention)**

    - **Queries (Decoder)**: $Q_{\text{decoder}} = X_{\text{decoder}} W_q$

    - **Keys, Values (Encoder)**: $K_{\text{encoder}} = X_{\text{encoder}} W_k$, $V_{\text{encoder}} = X_{\text{encoder}} W_v$

    - **Attention Scores**:
      $$
      A = \frac{Q_{\text{decoder}} K_{\text{encoder}}^T}{\sqrt{d_k}}
      $$

    - **Softmax**:
      $$
      \text{Attention Weights} = \text{softmax}(A)
      $$

    - **Output**:
      $$
      \text{Output} = \text{softmax}(A) V_{\text{encoder}}
      $$
    ---
    """)
    return


@app.cell
def _():
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    import torch
    import torch.nn as nn

    plt.style.use("dark_background")
    torch.manual_seed(47)
    np.random.seed(47)
    cmap = mcolors.LinearSegmentedColormap.from_list("magenta_cyan", ["magenta", "teal"])

    (vocab_size, context_length, embedding_dim, output_dim, batch_size) = 5000, 4, 16, 5, 2

    batch_of_tokenized_sentences = torch.randint(0, vocab_size, (batch_size, context_length))

    token_embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

    positional_embedding_layer = nn.Embedding(num_embeddings=context_length, embedding_dim=embedding_dim)

    token_embeddings = token_embedding_layer(batch_of_tokenized_sentences)
    position_embeddings = positional_embedding_layer(torch.arange(context_length))

    token_embeddings.shape, position_embeddings.shape

    embeddings = token_embeddings + position_embeddings

    embeddings.shape

    W_query = nn.Parameter(torch.rand(embedding_dim, output_dim), requires_grad=True)
    W_key = nn.Parameter(torch.rand(embedding_dim, output_dim), requires_grad=True)
    W_value = nn.Parameter(torch.rand(embedding_dim, output_dim), requires_grad=True)

    query = torch.einsum("ab,cda->cdb", W_query, embeddings)
    key = torch.einsum("ab,cda->cdb", W_key, embeddings)
    value = torch.einsum("ab,cda->cdb", W_value, embeddings)

    attention_scores = query @ key.permute(0, 2, 1)
    attention_scores.masked_fill_(torch.triu(torch.ones(context_length, context_length), diagonal=1).bool(), -torch.inf)
    attention_weights_torch = torch.softmax(attention_scores / output_dim**0.5, dim=-1)

    attention_weights = attention_weights_torch.detach().numpy()

    attention_weights[1].shape
    return (
        attention_weights,
        attention_weights_torch,
        cmap,
        np,
        plt,
        sns,
        torch,
        value,
    )


@app.cell
def _(attention_weights, cmap, np, plt, sns):
    ticklabels = [f"Token-{i}" for i in range(1, 5)]
    plt.figure(figsize=(5, 5), dpi=300)
    sns.heatmap(
        attention_weights[0, :],
        annot=True,
        mask=1 - np.tril(np.ones_like(attention_weights[0])),
        annot_kws={"size": 10},
        fmt=".4f",
        cmap=cmap,
        linewidths=1,
        linecolor="black",
        square=True,
        xticklabels=ticklabels,
        yticklabels=ticklabels,
        cbar_kws={"label": "Attention Weight"},
    )
    plt.title("Causal (Masked) Attention")
    plt.savefig("causal_masked_attention.png", dpi=300)
    plt.show()
    return


@app.cell
def _(attention_weights_torch, torch, value):
    context_vectors = torch.bmm(attention_weights_torch, value)
    assert torch.allclose(context_vectors[0, :][0, :], value[0, :][0])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    _Causal Attention ensures that tokens are considered in the direction of causality. Token-1 predicts Token-2, Token-1 and Token-2 predict Token-3, Token-1, Token-2, and Token-3 predict Token-4, and so on. This happens because the context vector is a weighted combination of value vectors, and by making attention weights for future tokens zero, we essentially discard them from influencing the prediction. This ensures that the model generates predictions in an autoregressive manner, where each token can only use information from previous tokens, preserving the natural flow of language generation. By masking future tokens, we prevent the model from "cheating" by looking ahead, maintaining a correct dependency structure._
    """)
    return


if __name__ == "__main__":
    app.run()
