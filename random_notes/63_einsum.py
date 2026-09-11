import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, time


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    # Einstein Summation Convention (`einsum`)

    [← 62 Grammar of Graphics](62_grammar_of_graphics.py) | [Index](../index.html) | [64 Pivoting →](64_pivoting.py)

    Introduced by Albert Einstein in 1916 for general relativity, the **Einstein summation convention** provides a compact, unified notation for linear algebra and multi-linear tensor algebra. In modern computational libraries (`numpy.einsum`, `torch.einsum`, `jax.numpy.einsum`), `einsum` allows engineers and researchers to express virtually any tensor contraction, transposition, diagonal extraction, or batched reduction in a single declarative string subscript.

    This notebook dissects the mathematical rules of index summation, formalizes free versus dummy indices, builds a comprehensive index taxonomy from inner products to multi-head transformer attention, audits contraction path optimization, and benchmarks execution efficiency against low-level BLAS routines.
    """)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Mathematical Formalism of Index Notation

    In classical index notation, if an index variable appears twice within a single multiplicative term, summation over that index's entire range is implied without an explicit $\sum$ symbol:

    $$
    A_i B_i \equiv \sum_{i=1}^n A_i B_i \quad \text{(Vector Dot Product)}
    $$

    $$
    A_{ij} B_{jk} \equiv \sum_{j=1}^m A_{ij} B_{jk} = C_{ik} \quad \text{(Matrix Multiplication)}
    $$

    ### The Two Classes of Indices

    1. **Free Indices**: Indices that appear in the output subscript (or appear exactly once on the left-hand side when the arrow `->` is omitted). They determine the rank and shape of the resulting output tensor.
    2. **Dummy (Summation) Indices**: Indices that appear two or more times in the input tensors but are omitted from the output subscript. These dimensions are contracted (summed over).

    ### Subscript Syntax Rules

    An `einsum` subscript string has the general form:

    $$
    \texttt{"<input\_subscripts> -> <output\_subscripts>"}
    $$

    - Comma `,` separates operand tensors.
    - Arrow `->` separates input indices from the desired output indices (explicit mode).
    - If `->` is omitted (implicit mode), indices appearing once are sorted alphabetically and become free indices, while repeated indices are summed over. **Explicit mode is strongly recommended in production** to avoid ambiguity.
    """)


@app.cell
def _(np):
    # Verify core linear algebraic operations with explicit einsum subscripts
    rng = np.random.default_rng(42)

    # 1D vectors
    u = np.array([1.0, 2.0, 3.0, 4.0])
    v = np.array([5.0, 6.0, 7.0, 8.0])

    # 2D matrices
    A = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # Shape (3, 2)
    B = np.array([[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]])  # Shape (2, 3)
    S = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])  # Shape (3, 3)

    # Contractions
    dot_np = float(np.dot(u, v))
    dot_ein = float(np.einsum("i,i->", u, v))

    outer_np = np.outer(u, v)
    outer_ein = np.einsum("i,j->ij", u, v)

    matmul_np = A @ B
    matmul_ein = np.einsum("ij,jk->ik", A, B)

    trace_np = float(np.trace(S))
    trace_ein = float(np.einsum("ii->", S))

    diag_np = np.diag(S)
    diag_ein = np.einsum("ii->i", S)

    hadamard_np = S * S
    hadamard_ein = np.einsum("ij,ij->ij", S, S)

    trans_np = A.T
    trans_ein = np.einsum("ij->ji", A)

    # Verification assertions
    assert np.isclose(dot_np, dot_ein)
    assert np.allclose(outer_np, outer_ein)
    assert np.allclose(matmul_np, matmul_ein)
    assert np.isclose(trace_np, trace_ein)
    assert np.allclose(diag_np, diag_ein)
    assert np.allclose(hadamard_np, hadamard_ein)
    assert np.allclose(trans_np, trans_ein)

    return (
        A,
        B,
        diag_ein,
        diag_np,
        dot_ein,
        dot_np,
        hadamard_ein,
        hadamard_np,
        matmul_ein,
        matmul_np,
        outer_ein,
        outer_np,
        rng,
        S,
        trace_ein,
        trace_np,
        trans_ein,
        trans_np,
        u,
        v,
    )


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Comprehensive Einsum Operation Taxonomy

    | Operation | Math Notation | `einsum` Subscript | Free Indices | Dummy Indices |
    | :--- | :--- | :--- | :--- | :--- |
    | **Vector Sum** | $s = \sum_i u_i$ | `"i->"` | $\emptyset$ | $i$ |
    | **Dot Product** | $s = \sum_i u_i v_i$ | `"i,i->"` | $\emptyset$ | $i$ |
    | **Outer Product** | $M_{ij} = u_i v_j$ | `"i,j->ij"` | $i, j$ | $\emptyset$ |
    | **Matrix Trace** | $\operatorname{tr}(A) = \sum_i A_{ii}$ | `"ii->"` | $\emptyset$ | $i$ |
    | **Diagonal Extraction** | $d_i = A_{ii}$ | `"ii->i"` | $i$ | $\emptyset$ |
    | **Matrix Transpose** | $A^T_{ji} = A_{ij}$ | `"ij->ji"` | $j, i$ | $\emptyset$ |
    | **Matrix-Vector Product** | $y_i = \sum_j A_{ij} x_j$ | `"ij,j->i"` | $i$ | $j$ |
    | **Matrix Multiplication** | $C_{ik} = \sum_j A_{ij} B_{jk}$ | `"ij,jk->ik"` | $i, k$ | $j$ |
    | **Hadamard Product** | $C_{ij} = A_{ij} B_{ij}$ | `"ij,ij->ij"` | $i, j$ | $\emptyset$ |
    | **Bilinear Form** | $s = \sum_i \sum_j x_i A_{ij} y_j$ | `"i,ij,j->"` | $\emptyset$ | $i, j$ |
    | **Batched Matmul** | $C_{bik} = \sum_j A_{bij} B_{bjk}$ | `"bij,bjk->bik"` | $b, i, k$ | $j$ |
    | **Batched Trace** | $t_b = \sum_i A_{bii}$ | `"bii->b"` | $b$ | $i$ |
    | **Self-Attention Scores** | $S_{bhij} = \sum_d Q_{bhid} K_{bhjd}$ | `"bhid,bhjd->bhij"` | $b, h, i, j$ | $d$ |
    | **Attention Context** | $O_{bhid} = \sum_j P_{bhij} V_{bhjd}$ | `"bhij,bhjd->bhid"` | $b, h, i, d$ | $j$ |
    """)


@app.cell(hide_code=True)
def _(go, mo):
    # Interactive visualization of einsum indices across tensor shapes
    _ops = [
        "Vector Dot", "Matrix Trace", "Matrix Diag", "Transpose",
        "Matrix-Vector", "Matmul", "Bilinear", "Batched Matmul",
        "Attention Scores", "Attention Context"
    ]
    _input_ranks = [2, 1, 1, 1, 2, 2, 3, 2, 2, 2]
    _output_ranks = [0, 0, 1, 2, 1, 2, 0, 3, 4, 4]
    _summed_dims = [1, 1, 0, 0, 1, 1, 2, 1, 1, 1]

    _fig = go.Figure()

    _fig.add_trace(
        go.Bar(
            name="Output Tensor Rank (Free Indices)",
            x=_ops,
            y=_output_ranks,
            marker_color="#3b82f6",
        )
    )

    _fig.add_trace(
        go.Bar(
            name="Contracted Axes (Dummy Indices)",
            x=_ops,
            y=_summed_dims,
            marker_color="#f59e0b",
        )
    )

    _fig.update_layout(
        title="Tensor Rank & Contraction Complexity by Einsum Operation",
        barmode="group",
        xaxis_title="Algebraic Operation",
        yaxis_title="Number of Indices / Axes",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=40, t=60, b=80),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    _md = mo.md(r"""
    ### Structural Analysis of Einsum Operations

    Notice the clean symmetry:
    - Dot products, traces, and bilinear forms reduce all dimensions to a scalar (Output Rank = 0).
    - Attention layers contract along a single inner dimension ($d$ or $j$) while preserving batch, head, and sequence indices ($b, h, i$).
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Deep Dive: Transformer Multi-Head Attention via `einsum`

    In Transformer architectures (Vaswani et al., 2017), the core self-attention computation operates over 4D tensors:
    - Queries $Q \in \mathbb{R}^{B \times H \times L_q \times D_k}$
    - Keys $K \in \mathbb{R}^{B \times H \times L_k \times D_k}$
    - Values $V \in \mathbb{R}^{B \times H \times L_k \times D_v}$

    where $B$ is batch size, $H$ is number of attention heads, $L$ is sequence length, and $D_k$ is head dimension.

    ### 1. Attention Score Computation

    The unnormalized attention logits matrix $S \in \mathbb{R}^{B \times H \times L_q \times L_k}$ contracts along the head embedding dimension $D_k$ (index $d$):

    $$
    S_{bhij} = \frac{1}{\sqrt{D_k}} \sum_{d=1}^{D_k} Q_{bhid} K_{bhjd}
    $$

    In `einsum`, this entire multi-headed batched operation is expressed as:
    ```python
    S = np.einsum("bhid,bhjd->bhij", Q, K) / np.sqrt(D_k)
    ```

    ### 2. Context Aggregation

    After computing attention weights $P = \operatorname{softmax}(S)$, the output context tensor $O \in \mathbb{R}^{B \times H \times L_q \times D_v}$ contracts along key sequence positions $L_k$ (index $j$):

    $$
    O_{bhid} = \sum_{j=1}^{L_k} P_{bhij} V_{bhjd}
    $$

    In `einsum`:
    ```python
    O = np.einsum("bhij,bhjd->bhid", P, V)
    ```
    """)


@app.cell
def _(np, rng):
    # Demonstrate Multi-Head Attention via einsum
    batch_size, num_heads, seq_len, head_dim = 2, 4, 8, 16

    Q_att = rng.standard_normal((batch_size, num_heads, seq_len, head_dim))
    K_att = rng.standard_normal((batch_size, num_heads, seq_len, head_dim))
    V_att = rng.standard_normal((batch_size, num_heads, seq_len, head_dim))

    # Attention scores via einsum
    scores_einsum = np.einsum("bhid,bhjd->bhij", Q_att, K_att) / np.sqrt(head_dim)

    # Conventional loop / reshape equivalent
    scores_matmul = np.matmul(Q_att, K_att.swapaxes(-1, -2)) / np.sqrt(head_dim)
    assert np.allclose(scores_einsum, scores_matmul)

    # Softmax along key sequence dimension (axis=-1)
    exp_scores = np.exp(scores_einsum - np.max(scores_einsum, axis=-1, keepdims=True))
    P_att = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    # Context representation via einsum
    context_einsum = np.einsum("bhij,bhjd->bhid", P_att, V_att)
    context_matmul = np.matmul(P_att, V_att)
    assert np.allclose(context_einsum, context_matmul)

    return (
        batch_size,
        context_einsum,
        context_matmul,
        exp_scores,
        head_dim,
        K_att,
        num_heads,
        P_att,
        Q_att,
        scores_einsum,
        scores_matmul,
        seq_len,
        V_att,
    )


@app.cell(hide_code=True)
def _(go, make_subplots, mo, P_att):
    # Visualize attention weight matrices for 4 heads of Batch 0
    _fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[f"Head {h+1}" for h in range(4)],
        horizontal_spacing=0.06,
    )

    for _h in range(4):
        _attn_map = P_att[0, _h, :, :]
        _fig.add_trace(
            go.Heatmap(
                z=_attn_map,
                colorscale="Viridis",
                zmin=0.0,
                zmax=float(_attn_map.max()),
                showscale=(_h == 3),
                colorbar=dict(title="Weight", len=0.8, x=1.02),
                hovertemplate="Query %{y} -> Key %{x}<br>Weight: %{z:.3f}<extra></extra>",
            ),
            row=1,
            col=_h + 1,
        )
        _fig.update_xaxes(title_text="Key Pos", row=1, col=_h + 1)

    _fig.update_yaxes(title_text="Query Pos", row=1, col=1)
    _fig.update_yaxes(autorange="reversed")

    _fig.update_layout(
        title="Multi-Head Attention Weights Computed via np.einsum('bhid,bhjd->bhij', Q, K)",
        template="plotly_white",
        height=360,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    _md = mo.md(r"""
    ### Attention Heatmap: Batch 0 across 4 Attention Heads

    Notice that `bhid,bhjd->bhij` naturally computes independent attention interaction matrices for each head without requiring explicit `for` loops or reshape gymnastics.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Contraction Path Optimization (`optimize=True`)

    When contracting networks of three or more tensors (such as tensor networks, graphical models, or bilinear transformations), the **order of pairwise contractions** fundamentally determines computational complexity.

    Consider the chain contraction:

    $$
    R_{il} = \sum_j \sum_k A_{ij} B_{jk} C_{kl}
    $$

    where $A \in \mathbb{R}^{10 \times 1000}$, $B \in \mathbb{R}^{1000 \times 10}$, and $C \in \mathbb{R}^{10 \times 1000}$.
    - Path 1: $(A B) C$
      - $(A B)$ costs $10 \times 1000 \times 10 = 10^5$ operations, yielding a $10 \times 10$ matrix.
      - $(A B) C$ costs $10 \times 10 \times 1000 = 10^5$ operations.
      - Total cost: **$2 \times 10^5$ FLOPs**.
    - Path 2: $A (B C)$
      - $(B C)$ costs $1000 \times 10 \times 1000 = 10^7$ operations, yielding a massive $1000 \times 1000$ matrix.
      - $A (B C)$ costs $10 \times 1000 \times 1000 = 10^7$ operations.
      - Total cost: **$2 \times 10^7$ FLOPs (100x slower!)**.

    NumPy's `np.einsum_path` analyzes operand shapes and uses dynamic programming or greedy search algorithms (powered by `opt_einsum`) to find the globally optimal contraction sequence.
    """)


@app.cell
def _(np, rng):
    # Benchmark chain contraction path
    dim_small = 10
    dim_large = 1000

    A_chain = rng.standard_normal((dim_small, dim_large))
    B_chain = rng.standard_normal((dim_large, dim_small))
    C_chain = rng.standard_normal((dim_small, dim_large))

    # Inspect einsum path
    path_info = np.einsum_path("ij,jk,kl->il", A_chain, B_chain, C_chain, optimize="optimal")

    # The string summary of path optimization
    path_summary = str(path_info[1])

    return A_chain, B_chain, C_chain, dim_large, dim_small, path_info, path_summary


@app.cell(hide_code=True)
def _(mo, path_summary):
    return mo.md(f"""
    ### Contraction Path Analysis from `np.einsum_path`

    ```text
    {path_summary}
    ```

    NumPy identifies that contracting $(A B)$ first reduces the intermediate matrix size to $10 \\times 10$, saving over $99\\%$ of memory traffic and arithmetic operations compared to the naive left-to-right evaluation.
    """)


@app.cell
def _(np, rng, time):
    # Micro-benchmark: np.einsum vs np.einsum(optimize=True) vs np.matmul (@)
    matrix_sizes = [64, 128, 256, 512]
    times_matmul = []
    times_einsum_naive = []
    times_einsum_opt = []

    for _n in matrix_sizes:
        _M1 = rng.standard_normal((_n, _n))
        _M2 = rng.standard_normal((_n, _n))

        # 1. BLAS @
        _t0 = time.perf_counter()
        for _ in range(5):
            _ = _M1 @ _M2
        _t_matmul = (time.perf_counter() - _t0) / 5.0
        times_matmul.append(_t_matmul * 1000.0)

        # 2. Einsum naive
        _t0 = time.perf_counter()
        for _ in range(5):
            _ = np.einsum("ij,jk->ik", _M1, _M2, optimize=False)
        _t_ein_naive = (time.perf_counter() - _t0) / 5.0
        times_einsum_naive.append(_t_ein_naive * 1000.0)

        # 3. Einsum optimized
        _t0 = time.perf_counter()
        for _ in range(5):
            _ = np.einsum("ij,jk->ik", _M1, _M2, optimize=True)
        _t_ein_opt = (time.perf_counter() - _t0) / 5.0
        times_einsum_opt.append(_t_ein_opt * 1000.0)

    return (
        matrix_sizes,
        times_einsum_naive,
        times_einsum_opt,
        times_matmul,
    )


@app.cell(hide_code=True)
def _(
    go,
    matrix_sizes,
    mo,
    times_einsum_naive,
    times_einsum_opt,
    times_matmul,
):
    _fig = go.Figure()

    _fig.add_trace(
        go.Scatter(
            x=matrix_sizes,
            y=times_matmul,
            mode="lines+markers",
            name="BLAS np.matmul (@)",
            line=dict(color="#10b981", width=2.5),
            marker=dict(size=8),
            hovertemplate="Size %{x}x%{x}<br>Latency: %{y:.3f} ms<extra></extra>",
        )
    )

    _fig.add_trace(
        go.Scatter(
            x=matrix_sizes,
            y=times_einsum_opt,
            mode="lines+markers",
            name="np.einsum(optimize=True)",
            line=dict(color="#3b82f6", width=2.5, dash="dash"),
            marker=dict(size=8),
            hovertemplate="Size %{x}x%{x}<br>Latency: %{y:.3f} ms<extra></extra>",
        )
    )

    _fig.add_trace(
        go.Scatter(
            x=matrix_sizes,
            y=times_einsum_naive,
            mode="lines+markers",
            name="np.einsum(optimize=False)",
            line=dict(color="#ef4444", width=2.5, dash="dot"),
            marker=dict(size=8),
            hovertemplate="Size %{x}x%{x}<br>Latency: %{y:.3f} ms<extra></extra>",
        )
    )

    _fig.update_layout(
        title="Execution Latency: Matrix Multiplication (NxN) across Runtimes",
        xaxis_title="Matrix Dimension N",
        yaxis_title="Execution Time (ms)",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    _md = mo.md(r"""
    ---

    ## Micro-Benchmark: Latency vs Native BLAS

    Modern `np.einsum` with `optimize=True` delegates standard matrix multiplications directly to hardware-tuned BLAS libraries (`dgemm` / OpenBLAS / MKL), matching native `@` performance while offering orders-of-magnitude greater expressiveness for arbitrary rank tensors!
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Key Takeaways and Best Practices

    1. **Explicit Arrows**: Always specify explicit output subscripts (e.g. `"ij,jk->ik"` instead of `"ij,jk"`). This eliminates silent index reordering bugs.
    2. **Avoid Intermediate Allocations**: For compound contractions like $s = x^T A y$, writing `np.einsum("i,ij,j->", x, A, y)` executes in a single streaming pass without allocating $A y \in \mathbb{R}^N$.
    3. **Always Set `optimize=True`**: For tensor contractions involving 3 or more operands, `optimize=True` enables dynamic programming contraction trees that prevent catastrophic $\mathcal{O}(N^k)$ algorithmic slowdowns.
    4. **Universal Portability**: Einsum notation translates seamlessly across NumPy, PyTorch, JAX, and TensorFlow with identical subscript strings.
    """)


if __name__ == "__main__":
    app.run()
