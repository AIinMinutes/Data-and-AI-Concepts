import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 49 Focal Loss](49_focal_loss_balanced.py) | [Index](../index.html) | [51 Causal Attention →](51_causal_attention.py)

        # 50. Scaled Dot-Product Attention: The Mathematical Engine of Transformer Architectures

        ### Executive Summary

        The **Scaled Dot-Product Attention** mechanism (Vaswani et al., 2017) represents the foundational computational building block of modern large language models, vision transformers, and multimodal foundational architectures. Departing from recurrence (RNNs) and convolution (CNNs), attention operates as a content-based associative memory that constructs context-aware representations by computing pairwise affinity weights across arbitrary sequence lengths in $\mathcal{O}(1)$ sequential operations.

        At its core, the mechanism maps each token into three distinct linear projections: a **Query** ($Q$, what information is sought), a **Key** ($K$, what information is indexed), and a **Value** ($V$, the actual content retrieved). By scaling raw dot-product affinities by $1/\sqrt{d_k}$, it solves the severe vanishing gradient problem inherent in high-dimensional softmax transformations, ensuring stable optimization across modern deep networks.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Scaling Derivations

        ### 1. Matrix Formulation of Scaled Dot-Product Attention

        Let $X \in \mathbb{R}^{N \times d_{\text{model}}}$ represent an input sequence of $N$ token embeddings. Given learnable projection matrices $W_Q \in \mathbb{R}^{d_{\text{model}} \times d_k}$, $W_K \in \mathbb{R}^{d_{\text{model}} \times d_k}$, and $W_V \in \mathbb{R}^{d_{\text{model}} \times d_v}$, the query, key, and value matrices are:

        $$Q = X W_Q \in \mathbb{R}^{N \times d_k}, \quad K = X W_K \in \mathbb{R}^{M \times d_k}, \quad V = X W_V \in \mathbb{R}^{M \times d_v}$$

        The scaled dot-product attention output $Y \in \mathbb{R}^{N \times d_v}$ is defined by the compact matrix equation:

        $$\operatorname{Attention}(Q, K, V) = \operatorname{Softmax}\left(\frac{Q K^\top}{\sqrt{d_k}}\right) V$$

        where the pre-softmax score matrix $S \in \mathbb{R}^{N \times M}$ and attention weight matrix $A \in \mathbb{R}^{N \times M}$ are:

        $$S = \frac{Q K^\top}{\sqrt{d_k}}, \qquad A_{ij} = \frac{\exp(S_{ij})}{\sum_{k=1}^M \exp(S_{ik})}, \qquad Y = A V$$

        Because each row of $A$ forms a valid probability distribution ($\sum_{j=1}^M A_{ij} = 1$ with $A_{ij} \ge 0$), each output vector $y_i \in \mathbb{R}^{d_v}$ is a strictly **convex combination** of all value vectors $v_1, \dots, v_M$.

        ### 2. The Variance Dilemma: Why Scale by $1/\sqrt{d_k}$?

        The scaling factor $\frac{1}{\sqrt{d_k}}$ is not an empirical heuristic; it is a mathematically required normalization to prevent softmax saturation.

        Consider a single query vector $q \in \mathbb{R}^{d_k}$ and key vector $k \in \mathbb{R}^{d_k}$. Assume their components $q_i$ and $k_i$ are independent, identically distributed random variables with zero mean and unit variance:

        $$\mathbb{E}[q_i] = \mathbb{E}[k_i] = 0, \qquad \operatorname{Var}(q_i) = \operatorname{Var}(k_i) = 1$$

        The unscaled dot product is the sum of $d_k$ independent random products:

        $$z = q^\top k = \sum_{i=1}^{d_k} q_i k_i$$

        Evaluating the expectation and variance of $z$:

        $$\mathbb{E}[z] = \sum_{i=1}^{d_k} \mathbb{E}[q_i] \mathbb{E}[k_i] = 0$$

        By the product rule of variances for independent zero-mean variables:

        $$\operatorname{Var}(q_i k_i) = \operatorname{Var}(q_i)\operatorname{Var}(k_i) + \operatorname{Var}(q_i)\mathbb{E}[k_i]^2 + \operatorname{Var}(k_i)\mathbb{E}[q_i]^2 = 1 \cdot 1 + 0 + 0 = 1$$

        Summing across all $d_k$ dimensions:

        $$\operatorname{Var}(z) = \sum_{i=1}^{d_k} \operatorname{Var}(q_i k_i) = d_k, \qquad \sigma(z) = \sqrt{d_k}$$

        #### Softmax Saturation and Vanishing Gradients:
        As projection dimensionality $d_k$ increases (e.g., $d_k = 64$ in Base Transformers, $d_k = 128$ in LLaMA-3), the standard deviation of raw dot products grows to $\sqrt{64} = 8$ or $\sqrt{128} \approx 11.3$. The dot product values $z$ routinely reach magnitudes in excess of $\pm 25$.

        Recall the Jacobian derivative of the softmax function with respect to its inputs:

        $$\frac{\partial A_i}{\partial S_j} = A_i (\delta_{ij} - A_j)$$

        When input magnitudes are large, the softmax output rapidly polarizes into a near-one-hot distribution ($A_{\max} \approx 1$, all other $A_j \approx 0$). In this regime:
        - For the winning index: $\frac{\partial A_i}{\partial S_i} = 1 \cdot (1 - 1) = 0$
        - For all losing indices: $\frac{\partial A_j}{\partial S_j} = 0 \cdot (1 - 0) = 0$

        The Jacobian vanishes completely, causing gradients to freeze and halting optimization. Dividing by $\sqrt{d_k}$ normalizes the variance:

        $$\operatorname{Var}\left(\frac{q^\top k}{\sqrt{d_k}}\right) = \frac{1}{d_k} \operatorname{Var}(q^\top k) = \frac{d_k}{d_k} = 1$$

        This maintains unit variance across all hidden dimensions, preserving active gradient propagation.

        ### 3. Backpropagation and Analytic Gradients

        During backpropagation, the loss gradient $\frac{\partial \mathcal{L}}{\partial Y} \in \mathbb{R}^{N \times d_v}$ flows backward through the attention layer:

        $$\frac{\partial \mathcal{L}}{\partial V} = A^\top \frac{\partial \mathcal{L}}{\partial Y}$$

        Let $\bar{A} = \frac{\partial \mathcal{L}}{\partial Y} V^\top \in \mathbb{R}^{N \times M}$. Using the vector-Jacobian product for softmax:

        $$\frac{\partial \mathcal{L}}{\partial S_{ij}} = A_{ij} \left( \bar{A}_{ij} - \sum_{k=1}^M \bar{A}_{ik} A_{ik} \right)$$

        In matrix notation, where $\mathbf{1}$ is an all-ones column vector:

        $$\frac{\partial \mathcal{L}}{\partial S} = A \odot \left( \bar{A} - ((\bar{A} \odot A)\mathbf{1})\mathbf{1}^\top \right)$$

        Finally, applying the chain rule to the scaled projections:

        $$\frac{\partial \mathcal{L}}{\partial Q} = \frac{1}{\sqrt{d_k}} \frac{\partial \mathcal{L}}{\partial S} K, \qquad \frac{\partial \mathcal{L}}{\partial K} = \frac{1}{\sqrt{d_k}} \left(\frac{\partial \mathcal{L}}{\partial S}\right)^\top Q$$
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Demonstrate linguistic self-attention on an illustrative resolved pronoun sentence
    sentence_tokens = ["The", "animal", "didn't", "cross", "the", "street", "because", "it", "was", "too", "tired"]
    seq_len = len(sentence_tokens)

    # Synthetic semantic embedding projections designed to simulate realistic coreference resolution
    np.random.seed(1337)
    d_k_demo = 32

    # Base embeddings
    emb_matrix = np.random.normal(0, 0.5, (seq_len, d_k_demo))
    # Make "animal" and "it" share strong semantic query-key alignment
    idx_animal = sentence_tokens.index("animal")
    idx_it = sentence_tokens.index("it")
    idx_street = sentence_tokens.index("street")
    idx_tired = sentence_tokens.index("tired")

    # Inject semantic correlation
    emb_matrix[idx_it] = 0.7 * emb_matrix[idx_animal] + 0.3 * emb_matrix[idx_tired] + 0.1 * np.random.randn(d_k_demo)
    emb_matrix[idx_tired] += 0.4 * emb_matrix[idx_animal]

    # Attention scores
    raw_dot_products = np.dot(emb_matrix, emb_matrix.T)
    scaled_scores = raw_dot_products / np.sqrt(d_k_demo)

    # Numerically stable softmax
    exp_scaled = np.exp(scaled_scores - np.max(scaled_scores, axis=-1, keepdims=True))
    attention_matrix = exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)

    # Softmax gradient simulation: Compare gradient magnitude as d_k scales from 4 to 512
    dim_grid = np.array([4, 8, 16, 32, 64, 128, 256, 512])
    n_trials = 200

    grad_norm_unscaled = []
    grad_norm_scaled = []

    for d in dim_grid:
        # Generate independent Gaussian queries and keys
        q_samples = np.random.normal(0, 1, (n_trials, d))
        k_samples = np.random.normal(0, 1, (n_trials, 10, d))  # 10 keys

        # Unscaled dot products
        unscaled_z = np.einsum("td,tkd->tk", q_samples, k_samples)
        # Scaled dot products
        scaled_z = unscaled_z / np.sqrt(d)

        # Softmax computation
        p_unscaled = np.exp(unscaled_z - np.max(unscaled_z, axis=1, keepdims=True))
        p_unscaled /= np.sum(p_unscaled, axis=1, keepdims=True)

        p_scaled = np.exp(scaled_z - np.max(scaled_z, axis=1, keepdims=True))
        p_scaled /= np.sum(p_scaled, axis=1, keepdims=True)

        # Softmax diagonal Jacobian norm: mean of p_i * (1 - p_i)
        grad_unscaled_mean = np.mean(p_unscaled * (1.0 - p_unscaled))
        grad_scaled_mean = np.mean(p_scaled * (1.0 - p_scaled))

        grad_norm_unscaled.append(grad_unscaled_mean)
        grad_norm_scaled.append(grad_scaled_mean)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Self-Attention Map A = Softmax(QK^T / sqrt(dk))</b>",
            "<b>Softmax Gradient Magnitude vs Key Dimension dk</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Attention Matrix Heatmap
    fig.add_trace(
        go.Heatmap(
            z=attention_matrix,
            x=sentence_tokens,
            y=sentence_tokens,
            colorscale="Blues",
            colorbar=dict(title="Attention Weight", x=0.42),
            hoverongaps=False,
        ),
        row=1,
        col=1,
    )

    # Panel 2: Vanishing gradient curves
    fig.add_trace(
        go.Scatter(
            x=dim_grid,
            y=grad_norm_scaled,
            mode="lines+markers",
            line=dict(color="#1D4ED8", width=2.5),
            marker=dict(size=8),
            name="Scaled Attention (1/sqrt(dk))",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=dim_grid,
            y=grad_norm_unscaled,
            mode="lines+markers",
            line=dict(color="#DC2626", width=2.5, dash="dash"),
            marker=dict(size=8),
            name="Unscaled Attention (Vanishing Gradient)",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Key Token", row=1, col=1)
    fig.update_yaxes(title_text="Query Token", autorange="reversed", row=1, col=1)
    fig.update_xaxes(title_text="Key Projection Dimension dk", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Mean Softmax Gradient E[p(1-p)]", range=[0, 0.12], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.72),
    )

    viz = mo.ui.plotly(fig)
    return (
        attention_matrix,
        dim_grid,
        fig,
        grad_norm_scaled,
        grad_norm_unscaled,
        sentence_tokens,
        viz,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below captures the linguistic and numerical dynamics of Scaled Dot-Product Attention:

                1. **Left Panel (Linguistic Self-Attention Heatmap)**: Querying the pronoun token `"it"` (row 7) reveals high attention mass concentrated on the antecedent `"animal"` and the condition `"tired"`, rather than the irrelevant syntactical token `"street"`. This exemplifies how attention dynamically routes contextual information across arbitrary sequence spans.
                2. **Right Panel (Softmax Gradient Decay Across Dimension $d_k$)**: In unscaled attention (red dashed curve), the softmax gradient decays toward zero as dimension $d_k$ grows from $4$ to $512$, proving empirical gradient vanishing. Scaled attention (blue solid curve) preserves stable, non-zero gradient magnitudes regardless of embedding dimension.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(dim_grid, grad_norm_scaled, grad_norm_unscaled, mo, np, pd):
    # Vectorized NumPy implementation of Scaled Dot-Product Attention with Analytic Backpropagation
    def scaled_dot_product_attention_np(Q, K, V, mask=None):
        """Pure NumPy implementation of Scaled Dot-Product Attention.

        Args:
            Q: Query tensor of shape (..., N, d_k)
            K: Key tensor of shape (..., M, d_k)
            V: Value tensor of shape (..., M, d_v)
            mask: Optional boolean mask of shape (..., N, M)
        """
        d_k = Q.shape[-1]
        scores = np.matmul(Q, np.swapaxes(K, -1, -2)) / np.sqrt(d_k)

        if mask is not None:
            scores = np.where(mask, scores, -1e9)

        # Numerically stable softmax
        exp_s = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        A = exp_s / np.sum(exp_s, axis=-1, keepdims=True)
        Y = np.matmul(A, V)
        return Y, A

    # Numerical verification of exact convex combination and row-sum properties
    np.random.seed(42)
    _N, _M, _dk, _dv = 5, 6, 16, 8
    _Q = np.random.randn(_N, _dk)
    _K = np.random.randn(_M, _dk)
    _V = np.random.randn(_M, _dv)

    _Y, _A = scaled_dot_product_attention_np(_Q, _K, _V)

    # Check row sums equal 1.0
    _row_sums = np.sum(_A, axis=-1)
    # Check output range bounded by convex hull of V
    _v_min_norm = np.min(np.linalg.norm(_V, axis=-1))
    _v_max_norm = np.max(np.linalg.norm(_V, axis=-1))
    _y_norms = np.linalg.norm(_Y, axis=-1)

    df_properties = pd.DataFrame(
        [
            {
                "Mathematical_Property": "Row Sum Normalization (sum_j A_ij = 1)",
                "Theoretical_Value": "1.000000",
                "Empirical_Min": f"{np.min(_row_sums):.6f}",
                "Empirical_Max": f"{np.max(_row_sums):.6f}",
                "Verification_Status": (
                    "Exact Match" if np.allclose(_row_sums, 1.0, atol=1e-6) else "Discrepancy"
                ),
            },
            {
                "Mathematical_Property": "Non-Negativity (A_ij >= 0)",
                "Theoretical_Value": ">= 0.0",
                "Empirical_Min": f"{np.min(_A):.6f}",
                "Empirical_Max": f"{np.max(_A):.6f}",
                "Verification_Status": "Strictly Non-Negative" if np.all(_A >= 0.0) else "Negative Entry Found",
            },
            {
                "Mathematical_Property": "Convex Hull Norm Boundedness",
                "Theoretical_Value": f"[{_v_min_norm:.2f}, {_v_max_norm:.2f}]",
                "Empirical_Min": f"{np.min(_y_norms):.2f}",
                "Empirical_Max": f"{np.max(_y_norms):.2f}",
                "Verification_Status": "Within Convex Hull" if np.max(_y_norms) <= _v_max_norm * 1.01 else "Exceeded",
            },
        ]
    )

    # Example 2: Softmax Vanishing Gradient Audit Table
    gradient_records = []
    for _idx, _d in enumerate(dim_grid):
        _ratio = grad_norm_scaled[_idx] / max(grad_norm_unscaled[_idx], 1e-12)
        gradient_records.append(
            {
                "Projection_Dimension_dk": int(_d),
                "Scaled_Grad_Norm": f"{grad_norm_scaled[_idx]:.5f}",
                "Unscaled_Grad_Norm": f"{grad_norm_unscaled[_idx]:.5f}",
                "Gradient_Ratio (Scaled / Unscaled)": f"{_ratio:.1f}x",
                "Regime": (
                    "Stable Gradient Flow" if grad_norm_unscaled[_idx] > 0.03 else "Severe Softmax Saturation"
                ),
            }
        )

    df_gradient_audit = pd.DataFrame(gradient_records)

    # Example 3: Contextual Representation Shift (Polysemous Word Disambiguation)
    # Target word: "bank" in two distinct sentences
    # Sentence 1: "deposit cash at river bank" vs Sentence 2: "deposit cash at investment bank"
    vocab = ["deposit", "cash", "at", "river", "investment", "bank"]
    v_dim = 8

    # Semantic prototypes
    np.random.seed(42)
    semantic_bases = {
        "deposit": np.array([2.0, 0.1, 0.0, 0.0, 1.5, 0.0, 0.0, 0.5]),
        "cash": np.array([2.5, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.2]),
        "at": np.array([0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1, 0.0]),
        "river": np.array([0.0, 3.0, 2.5, 2.0, 0.0, 0.0, 0.0, 0.0]),
        "investment": np.array([3.0, 0.0, 0.0, 0.0, 3.0, 1.5, 0.0, 1.0]),
        # Polysemous uncontextualized static embedding (mixture of finance and geography)
        "bank": np.array([1.5, 1.5, 1.0, 1.0, 1.5, 0.5, 0.0, 0.5]),
    }

    # Sentence A: river bank
    sent_A = ["deposit", "cash", "at", "river", "bank"]
    X_A = np.array([semantic_bases[w] for w in sent_A])
    Y_A, _ = scaled_dot_product_attention_np(X_A, X_A, X_A)
    bank_vec_A = Y_A[sent_A.index("bank")]

    # Sentence B: investment bank
    sent_B = ["deposit", "cash", "at", "investment", "bank"]
    X_B = np.array([semantic_bases[w] for w in sent_B])
    Y_B, _ = scaled_dot_product_attention_np(X_B, X_B, X_B)
    bank_vec_B = Y_B[sent_B.index("bank")]

    def cosine_sim(u, v):
        return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

    sim_static = cosine_sim(semantic_bases["bank"], semantic_bases["bank"])  # 1.0
    sim_contextual = cosine_sim(bank_vec_A, bank_vec_B)
    sim_bank_river = cosine_sim(bank_vec_A, semantic_bases["river"])
    sim_bank_invest = cosine_sim(bank_vec_B, semantic_bases["investment"])

    df_disambiguation = pd.DataFrame(
        [
            {
                "Comparison_Pair": "Static bank vs Static bank (Pre-Attention)",
                "Cosine_Similarity": f"{sim_static:.4f}",
                "Semantic_Interpretation": "Single polysemous static vector cannot differentiate senses",
            },
            {
                "Comparison_Pair": "Contextual bank (River) vs Contextual bank (Finance)",
                "Cosine_Similarity": f"{sim_contextual:.4f}",
                "Semantic_Interpretation": "Attention forces representations into distinct semantic regions",
            },
            {
                "Comparison_Pair": "Contextual bank (River) vs Static 'river'",
                "Cosine_Similarity": f"{sim_bank_river:.4f}",
                "Semantic_Interpretation": "Substantial alignment with geographic/hydrological context",
            },
            {
                "Comparison_Pair": "Contextual bank (Finance) vs Static 'investment'",
                "Cosine_Similarity": f"{sim_bank_invest:.4f}",
                "Semantic_Interpretation": "Strong alignment with financial enterprise context",
            },
        ]
    )

    table_properties = mo.ui.table(df_properties)
    table_gradient = mo.ui.table(df_gradient_audit)
    table_disambig = mo.ui.table(df_disambiguation)

    return (
        table_disambig,
        table_gradient,
        table_properties,
    )


@app.cell
def _(mo, table_disambig, table_gradient, table_properties):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Vectorized Attention and Algebraic Axiom Verification

                Confirming row-sum probability normalization and convex hull containment:
                """
            ),
            table_properties,
            mo.md(
                r"""
                ### Example 2: Vanishing Gradient Audit Across Projection Dimensionality $d_k$

                Evaluating the scaling ratio between scaled and unscaled softmax gradient norms:
                """
            ),
            table_gradient,
            mo.md(
                r"""
                ### Example 3: Contextual Representation Shift (Polysemous Disambiguation)

                Observing how self-attention transforms identical static token embeddings into contextually specialized vectors:
                """
            ),
            table_disambig,
        ]
    )


if __name__ == "__main__":
    app.run()
