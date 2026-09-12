import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 04: Rank-One Matrices, Outer Products, and Low-Rank Decomposition

    &larr; Previous Note: [03 Hyperplanes](03_hyperplanes.py) | Next Note: [05 Orthogonality](05_orthogonality.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Rank-one matrices are the fundamental atomic building blocks of all linear algebra. Any complex linear transformation, dataset, or neural network weight tensor can be expressed as a linear combination of rank-one matrices.

    Understanding rank-one structures gives you foundational insight into:

    1. **Matrix Factorization**: Decomposing large, unwieldy data matrices into compact products of vectors.
    2. **Low-Rank Approximation**: Applying the Eckart-Young-Mirsky theorem to compress data, filter noise, and capture dominant latent patterns.
    3. **Parameter-Efficient Fine-Tuning (PEFT)**: Modern LLM adaptation techniques like LoRA (Low-Rank Adaptation) freeze billions of pre-trained parameters and train rank-one and low-rank factor updates.
    4. **Recommender Systems**: Factorizing sparse user-item rating grids into shared latent representation spaces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Rank of a Matrix

    The **rank** of a matrix $\mathbf{A}_{m \times n}$, denoted $\text{rank}(\mathbf{A})$, is the maximal number of linearly independent rows or columns in the matrix. Key properties include:
    * $\text{rank}(\mathbf{A}) \leq \min(m, n)$
    * Row rank always equals Column rank.

    ### Definition of a Rank-One Matrix

    A non-zero matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ has rank 1 if and only if it can be written as the **outer product** of two non-zero vectors $\mathbf{u} \in \mathbb{R}^m$ and $\mathbf{v} \in \mathbb{R}^n$:

    $$
    \mathbf{A} = \mathbf{u} \mathbf{v}^T = \begin{bmatrix} u_1 \\ u_2 \\ \vdots \\ u_m \end{bmatrix} \begin{bmatrix} v_1 & v_2 & \dots & v_n \end{bmatrix} = \begin{bmatrix} u_1 v_1 & u_1 v_2 & \dots & u_1 v_n \\ u_2 v_1 & u_2 v_2 & \dots & u_2 v_n \\ \vdots & \vdots & \ddots & \vdots \\ u_m v_1 & u_m v_2 & \dots & u_m v_n \end{bmatrix}
    $$

    Key structural properties of $\mathbf{u}\mathbf{v}^T$:
    * **Collinear Rows**: Every row of $\mathbf{A}$ is a scalar multiple of $\mathbf{v}^T$: $\text{Row}_i = u_i \mathbf{v}^T$.
    * **Collinear Columns**: Every column of $\mathbf{A}$ is a scalar multiple of $\mathbf{u}$: $\text{Col}_j = v_j \mathbf{u}$.
    * **Dimensionality**: The column space $\mathcal{C}(\mathbf{A}) = \text{span}(\mathbf{u})$ has dimension 1. By the Rank-Nullity Theorem, the null space has dimension $n - 1$.
    * **Eigenvalues and Trace**: The matrix $\mathbf{u}\mathbf{v}^T$ has at most one non-zero eigenvalue, which equals the inner product of the vectors: $\lambda = \mathbf{v}^T \mathbf{u} = \text{tr}(\mathbf{u}\mathbf{v}^T)$.

    ### Outer Product vs Inner Product

    For two vectors $\mathbf{u}, \mathbf{v} \in \mathbb{R}^n$:
    * **Inner Product (Scalar)**: $\mathbf{u}^T \mathbf{v} = \sum_{i=1}^n u_i v_i \in \mathbb{R}$ measures alignment, projection, and angle.
    * **Outer Product (Matrix)**: $\mathbf{u} \mathbf{v}^T \in \mathbb{R}^{n \times n}$ generates a rank-one directional mapping.

    ### Singular Value Decomposition as an Additive Sum of Rank-1 Matrices

    The Singular Value Decomposition (SVD) states that any real matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$ of rank $r \le \min(m, n)$ can be factored into orthogonal matrices $\mathbf{U}$, $\mathbf{V}$ and diagonal matrix $\mathbf{\Sigma}$:

    $$
    \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T
    $$

    where:
    * $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_r > 0$ are the singular values.
    * $\mathbf{u}_i \in \mathbb{R}^m$ are the orthonormal left singular vectors (columns of $\mathbf{U}$).
    * $\mathbf{v}_i \in \mathbb{R}^n$ are the orthonormal right singular vectors (columns of $\mathbf{V}$).
    * Each term $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ is an independent rank-one matrix weighted by $\sigma_i$.

    ### Eckart-Young-Mirsky Theorem

    The optimal rank-$k$ approximation ($k < r$) of $\mathbf{A}$ under both the Frobenius norm and spectral norm is obtained by retaining the top $k$ rank-one components:

    $$
    \mathbf{A}_k = \sum_{i=1}^k \sigma_i \mathbf{u}_i \mathbf{v}_i^T
    $$

    The approximation error is directly governed by the neglected singular values:

    $$
    \|\mathbf{A} - \mathbf{A}_k\|_F = \sqrt{\sum_{i=k+1}^r \sigma_i^2}
    $$

    ### Role in ML, AI, and Statistics

    **Low-Rank Adaptation (LoRA)**: In large language models, fine-tuning dense weight matrices $\mathbf{W}_0 \in \mathbb{R}^{d \times k}$ directly is computationally prohibitive. LoRA reparameterizes the update as $\Delta \mathbf{W} = \mathbf{B}\mathbf{A}$, where $\mathbf{B} \in \mathbb{R}^{d \times r}$ and $\mathbf{A} \in \mathbb{R}^{r \times k}$ with rank $r \ll \min(d, k)$. A rank-1 or rank-4 update adjusts weights with a tiny fraction of the memory footprint.

    **Principal Component Analysis (PCA)**: PCA finds the dominant rank-one projections of a centered empirical covariance matrix $\mathbf{S} = \frac{1}{n} \mathbf{X}^T \mathbf{X}$. The first principal component corresponds to the highest-energy rank-one approximation $\sigma_1 \mathbf{u}_1 \mathbf{v}_1^T$.

    **Recommender Systems and Matrix Completion**: Large interaction matrices (users $\times$ items) are extremely sparse. Assuming preferences depend on a small number of latent factors models the interaction matrix as a sum of low-rank outer products, enabling prediction of unobserved ratings.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    Below are two concrete implementations:
    1. **Rank-1 Outer Product & SVD Decomposition**: Constructing an outer product from scratch, inspecting properties, and verifying exact additive SVD reconstruction on a $3 \times 2$ matrix.
    2. **Low-Rank Image / Surface Reconstruction**: Synthesizing a 2D surface pattern and interactively visualizing its progressive rank-$k$ approximations via Plotly.

    ### Example 1: Rank-1 Mechanics and SVD
    """)
    return


@app.cell
def _(np):
    # Example 1: Numerical Mechanics of Rank-1 Matrices and Additive SVD Reconstruction
    # Define two arbitrary vectors for an outer product
    u_vec = np.array([2.0, -1.0, 3.0])  # 3 x 1
    v_vec = np.array([1.0, 4.0])        # 2 x 1

    # Outer product matrix A_rank1 = u @ v.T (shape: 3 x 2)
    A_rank1 = np.outer(u_vec, v_vec)
    rank_calculated = np.linalg.matrix_rank(A_rank1)

    # Singular Value Decomposition of a 3 x 2 matrix
    A_test = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    U, S, Vt = np.linalg.svd(A_test, full_matrices=False)

    # Construct individual rank-1 components: sigma_i * (u_i @ v_i.T)
    rank1_comp_1 = S[0] * np.outer(U[:, 0], Vt[0, :])
    rank1_comp_2 = S[1] * np.outer(U[:, 1], Vt[1, :])

    # Reconstruct original matrix by adding the rank-1 components
    A_reconstructed = rank1_comp_1 + rank1_comp_2
    reconstruction_error = float(np.linalg.norm(A_test - A_reconstructed, ord="fro"))

    {
        "rank_of_outer_product": int(rank_calculated),
        "singular_values": S.tolist(),
        "first_singular_value": float(S[0]),
        "second_singular_value": float(S[1]),
        "reconstruction_frobenius_error": reconstruction_error
    }
    return (
        A_rank1,
        A_reconstructed,
        A_test,
        S,
        U,
        Vt,
        rank1_comp_1,
        rank1_comp_2,
        reconstruction_error,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Interactive Low-Rank Surface Approximation
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Example 2: Interactive Low-Rank Surface Approximation with Plotly
    # Generate a synthetic 2D pattern (linear combinations of distinct spatial frequencies)
    x_axis = np.linspace(-3, 3, 40)
    y_axis = np.linspace(-3, 3, 40)
    X_grid, Y_grid = np.meshgrid(x_axis, y_axis)

    # Continuous 2D surface with distinct low-rank components
    pattern = (
        1.5 * np.outer(np.exp(-y_axis**2 / 2), np.exp(-x_axis**2 / 2))
        + 1.0 * np.outer(np.sin(1.5 * y_axis), np.cos(1.5 * x_axis))
        + 0.5 * np.outer(y_axis / 3, (x_axis / 3)**2)
    )

    # Perform full SVD
    U_p, S_p, Vt_p = np.linalg.svd(pattern, full_matrices=False)

    # Compute Rank-1, Rank-2, and Rank-3 approximations
    approx_rank_1 = S_p[0] * np.outer(U_p[:, 0], Vt_p[0, :])
    approx_rank_2 = approx_rank_1 + S_p[1] * np.outer(U_p[:, 1], Vt_p[1, :])
    approx_rank_3 = approx_rank_2 + S_p[2] * np.outer(U_p[:, 2], Vt_p[2, :])

    total_energy = np.sum(S_p**2)
    energy_r1 = 100.0 * (S_p[0]**2) / total_energy
    energy_r2 = 100.0 * (S_p[0]**2 + S_p[1]**2) / total_energy
    energy_r3 = 100.0 * np.sum(S_p[:3]**2) / total_energy

    # Render side-by-side comparison heatmaps
    fig = make_subplots(
        rows=1,
        cols=4,
        subplot_titles=[
            f"Rank 1 ({energy_r1:.1f}% energy)",
            f"Rank 2 ({energy_r2:.1f}% energy)",
            f"Rank 3 ({energy_r3:.1f}% energy)",
            "Original (Rank 40)"
        ],
        horizontal_spacing=0.04
    )

    fig.add_trace(go.Heatmap(z=approx_rank_1, colorscale="Tealgrn", showscale=False), row=1, col=1)
    fig.add_trace(go.Heatmap(z=approx_rank_2, colorscale="Tealgrn", showscale=False), row=1, col=2)
    fig.add_trace(go.Heatmap(z=approx_rank_3, colorscale="Tealgrn", showscale=False), row=1, col=3)
    fig.add_trace(go.Heatmap(z=pattern, colorscale="Tealgrn", showscale=False), row=1, col=4)

    fig.update_layout(
        title=dict(
            text="Low-Rank Matrix Approximations: Progressive Sum of Rank-1 Outer Products",
            font=dict(size=14)
        ),
        template="plotly_white",
        width=920,
        height=320,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    for i in range(1, 5):
        fig.update_xaxes(showticklabels=False, row=1, col=i)
        fig.update_yaxes(showticklabels=False, row=1, col=i)

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    * **Atomic Linear Units**: Every rank-one matrix is formed by an outer product $\mathbf{u}\mathbf{v}^T$, constraining all rows to lie along $\mathbf{v}^T$ and all columns to lie along $\mathbf{u}$.
    * **Additive SVD Representation**: Any matrix $\mathbf{A}$ of rank $r$ is an exact linear superposition of $r$ orthogonal rank-one components: $\mathbf{A} = \sum_{i=1}^r \sigma_i \mathbf{u}_i \mathbf{v}_i^T$.
    * **Optimal Compression**: Truncating the SVD to the top $k$ rank-one components produces the best rank-$k$ approximation under the Frobenius and spectral norms.
    * **Modern AI Applications**: Low-rank structures enable efficient large language model fine-tuning (LoRA), where rank-1 and low-rank factor updates adapt billions of frozen model weights.

    ---

    &larr; Previous Note: [03 Hyperplanes](03_hyperplanes.py) | Next Note: [05 Orthogonality](05_orthogonality.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
