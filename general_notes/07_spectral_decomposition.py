import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


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
    mo.md(r"""
    # Note 07: Spectral Decomposition and Symmetric Matrix Eigenspaces

    &larr; Previous Note: [06 Moore-Penrose Pseudoinverse](06_moore_penrose_inverse.py) | Next Note: [08 Matrix Calculus](08_matrix_calculus_short.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    The Spectral Theorem is one of the most celebrated results in linear algebra and mathematical statistics. It guarantees that any real symmetric matrix can be factored into mutually perpendicular (orthogonal) coordinate axes scaled by purely real eigenvalues.

    In modern data science and AI, symmetric matrices appear everywhere:
    1. **Empirical Covariance Matrices**: In Principal Component Analysis (PCA), eigendecomposition reveals the principal axes of maximum variance and decorrelates features.
    2. **Graph Laplacians in GNNs**: In Graph Neural Networks and spectral clustering, the eigenvectors of the Graph Laplacian define the graph Fourier transform modes (ChebNet, GCN).
    3. **Loss Function Hessians**: The spectrum of the Hessian $\mathbf{H} = \nabla^2 \mathcal{L}(\mathbf{w})$ governs optimization dynamics, gradient descent convergence rates ($\eta < 2/\lambda_{\max}$), and loss landscape sharpness.
    4. **Kernel Gram Matrices**: In Support Vector Machines and Gaussian Processes, positive semi-definite kernel matrices represent inner products in reproducing kernel Hilbert spaces (Mercer's Theorem).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### The Spectral Theorem for Real Symmetric Matrices

    Let $\mathbf{S} \in \mathbb{R}^{n \times n}$ be a real symmetric matrix, meaning $\mathbf{S} = \mathbf{S}^T$. The Spectral Theorem establishes three fundamental properties:
    1. All $n$ eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_n$ are real numbers ($\lambda_i \in \mathbb{R}$).
    2. Eigenvectors corresponding to distinct eigenvalues are mutually orthogonal.
    3. There exists an orthonormal basis of $\mathbb{R}^n$ composed entirely of eigenvectors of $\mathbf{S}$.

    Therefore, $\mathbf{S}$ can be diagonalized by an orthogonal matrix $\mathbf{Q} \in \mathbb{R}^{n \times n}$ ($\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$):

    $$
    \mathbf{S} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T
    $$

    where $\mathbf{\Lambda} = \text{diag}(\lambda_1, \lambda_2, \dots, \lambda_n)$.

    Writing $\mathbf{Q}$ in terms of its orthonormal column eigenvectors $[\mathbf{q}_1, \dots, \mathbf{q}_n]$, this expression is equivalent to an additive decomposition into rank-1 orthogonal projections:

    $$
    \mathbf{S} = \sum_{i=1}^n \lambda_i \mathbf{q}_i \mathbf{q}_i^T
    $$

    Each term $\mathbf{q}_i \mathbf{q}_i^T$ is a symmetric rank-1 projection matrix onto the 1D subspace spanned by $\mathbf{q}_i$.

    ---

    ### Positive Semi-Definite (PSD) Matrices

    A symmetric matrix $\mathbf{S} \in \mathbb{R}^{n \times n}$ is **positive semi-definite** ($\mathbf{S} \succeq 0$) if for every vector $\mathbf{x} \in \mathbb{R}^n$:

    $$
    \mathbf{x}^T \mathbf{S} \mathbf{x} \geq 0
    $$

    Substituting the spectral decomposition $\mathbf{S} = \mathbf{Q}\mathbf{\Lambda}\mathbf{Q}^T$ and letting $\mathbf{y} = \mathbf{Q}^T \mathbf{x}$:

    $$
    \mathbf{x}^T \mathbf{S} \mathbf{x} = \mathbf{y}^T \mathbf{\Lambda} \mathbf{y} = \sum_{i=1}^n \lambda_i y_i^2 \geq 0, \quad \forall \mathbf{y} \in \mathbb{R}^n
    $$

    This immediately proves that a symmetric matrix is positive semi-definite if and only if all its eigenvalues are non-negative:

    $$
    \mathbf{S} \succeq 0 \iff \lambda_i \geq 0, \quad \forall i \in \{1, \dots, n\}
    $$

    If the quadratic form is strictly positive ($\mathbf{x}^T \mathbf{S} \mathbf{x} > 0$ for all $\mathbf{x} \neq \mathbf{0}$), $\mathbf{S}$ is **positive definite** ($\mathbf{S} \succ 0$), meaning all eigenvalues are strictly positive ($\lambda_i > 0$).

    ---

    ### Universal Positive Semi-Definiteness of Gram Matrices

    For any rectangular real matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, the Gram matrices $\mathbf{A}^T \mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$ are unconditionally symmetric and positive semi-definite.

    #### Verification for $\mathbf{A}^T \mathbf{A}$

    Symmetry: $(\mathbf{A}^T \mathbf{A})^T = \mathbf{A}^T (\mathbf{A}^T)^T = \mathbf{A}^T \mathbf{A}$.
    Positive semi-definiteness: for any $\mathbf{x} \in \mathbb{R}^n$:

    $$
    \mathbf{x}^T (\mathbf{A}^T \mathbf{A}) \mathbf{x} = (\mathbf{A}\mathbf{x})^T (\mathbf{A}\mathbf{x}) = \|\mathbf{A}\mathbf{x}\|_2^2 \geq 0
    $$

    #### Verification for $\mathbf{A}\mathbf{A}^T$

    Symmetry: $(\mathbf{A} \mathbf{A}^T)^T = (\mathbf{A}^T)^T \mathbf{A}^T = \mathbf{A} \mathbf{A}^T$.
    Positive semi-definiteness: for any $\mathbf{y} \in \mathbb{R}^{m}$:

    $$
    \mathbf{y}^T (\mathbf{A} \mathbf{A}^T) \mathbf{y} = (\mathbf{A}^T \mathbf{y})^T (\mathbf{A}^T \mathbf{y}) = \|\mathbf{A}^T \mathbf{y}\|_2^2 \geq 0
    $$

    ---

    ### The Direct Bridge Between SVD and Spectral Decomposition

    Singular Value Decomposition (SVD) and Spectral Decomposition are intimately linked. Let the SVD of $\mathbf{A} \in \mathbb{R}^{m \times n}$ be:

    $$
    \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T
    $$

    where $\mathbf{U} \in \mathbb{R}^{m \times m}$ and $\mathbf{V} \in \mathbb{R}^{n \times n}$ are orthogonal, and $\mathbf{\Sigma} \in \mathbb{R}^{m \times n}$ contains singular values $\sigma_1 \geq \sigma_2 \geq \dots \geq 0$.

    Computing the Gram matrices via SVD yields:

    $$
    \mathbf{A}^T \mathbf{A} = (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)^T (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T) = \mathbf{V} \mathbf{\Sigma}^T \mathbf{\Sigma} \mathbf{V}^T = \mathbf{V} \mathbf{\Lambda}_{\text{right}} \mathbf{V}^T
    $$

    $$
    \mathbf{A} \mathbf{A}^T = (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T) (\mathbf{U} \mathbf{\Sigma} \mathbf{V}^T)^T = \mathbf{U} \mathbf{\Sigma} \mathbf{\Sigma}^T \mathbf{U}^T = \mathbf{U} \mathbf{\Lambda}_{\text{left}} \mathbf{U}^T
    $$

    This reveals three structural facts:
    * The right singular vectors $\mathbf{V}$ are the orthonormal eigenvectors of $\mathbf{A}^T \mathbf{A}$.
    * The left singular vectors $\mathbf{U}$ are the orthonormal eigenvectors of $\mathbf{A} \mathbf{A}^T$.
    * The singular values $\sigma_i$ of $\mathbf{A}$ are the square roots of the non-zero eigenvalues of both $\mathbf{A}^T \mathbf{A}$ and $\mathbf{A} \mathbf{A}^T$:

    $$
    \sigma_i = \sqrt{\lambda_i(\mathbf{A}^T \mathbf{A})} = \sqrt{\lambda_i(\mathbf{A}\mathbf{A}^T)}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    ### Example 1: Interactive Visualizations: Quadratic Form Geometry and Spectral Equivalence

    The interactive subplots below display the dual perspective of spectral decomposition:
    * **Left Panel**: Transformation of the unit circle under a 2D symmetric positive definite matrix $\mathbf{S}$. The eigenvectors $\mathbf{q}_1, \mathbf{q}_2$ define the principal axes of the resulting ellipse, scaled by $\lambda_1, \lambda_2$.
    * **Right Panel**: Verification that the non-zero eigenvalues of $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$ exactly equal the squared singular values $\sigma_i^2$ from SVD.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Left Panel: 2D Ellipsoid Geometry for Symmetric Matrix
    # Construct a 2D symmetric PSD matrix
    theta_deg = 35.0
    theta_rad = np.radians(theta_deg)
    cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
    q_mat = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    lambdas = np.array([2.4, 0.8])
    s_mat = q_mat @ np.diag(lambdas) @ q_mat.T

    # Unit circle points
    t = np.linspace(0, 2 * np.pi, 200)
    circle_pts = np.vstack([np.cos(t), np.sin(t)])

    # Transformed ellipse points: S @ circle
    ellipse_pts = s_mat @ circle_pts

    # Eigenvector axes scaled by eigenvalues
    q1 = q_mat[:, 0]
    q2 = q_mat[:, 1]
    axis1 = lambdas[0] * q1
    axis2 = lambdas[1] * q2

    # Right Panel: Gram matrix eigenvalue vs singular value comparison
    # Generate rectangular matrix A (4 x 3)
    rng = np.random.default_rng(101)
    a_rect = np.array([[2.0, 1.0, 0.5], [1.5, 3.0, 1.2], [0.8, 0.4, 2.5], [1.2, 2.1, 0.9]])
    _, s_vals, _ = np.linalg.svd(a_rect)
    eig_ata = np.sort(np.linalg.eigvalsh(a_rect.T @ a_rect))[::-1]
    eig_aat = np.sort(np.linalg.eigvalsh(a_rect @ a_rect.T))[::-1][:3]
    s_squared = s_vals**2

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Symmetric Matrix Geometry (λ₁={lambdas[0]:.1f}, λ₂={lambdas[1]:.1f})",
            "Gram Matrix Spectrum vs SVD Singular Values",
        ],
    )

    # Left: Unit circle
    fig.add_trace(
        go.Scatter(
            x=circle_pts[0],
            y=circle_pts[1],
            mode="lines",
            line=dict(color="#cbd5e1", dash="dash", width=1.5),
            name="Unit Circle ||x||=1",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Left: Transformed ellipse
    fig.add_trace(
        go.Scatter(
            x=ellipse_pts[0],
            y=ellipse_pts[1],
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name="Transformed Ellipse S x",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Left: Principal Axis 1
    fig.add_trace(
        go.Scatter(
            x=[0, axis1[0]],
            y=[0, axis1[1]],
            mode="lines+markers",
            line=dict(color="#dc2626", width=3.5),
            marker=dict(size=7, color="#dc2626"),
            name="Principal Axis λ₁ q₁",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Left: Principal Axis 2
    fig.add_trace(
        go.Scatter(
            x=[0, axis2[0]],
            y=[0, axis2[1]],
            mode="lines+markers",
            line=dict(color="#16a34a", width=3.5),
            marker=dict(size=7, color="#16a34a"),
            name="Principal Axis λ₂ q₂",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Right: Bar comparison
    categories = ["Component 1", "Component 2", "Component 3"]
    fig.add_trace(
        go.Bar(
            x=categories,
            y=eig_ata,
            name="λ(Aᵀ A)",
            marker_color="#2563eb",
            opacity=0.85,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=eig_aat,
            name="λ(A Aᵀ)",
            marker_color="#06b6d4",
            opacity=0.85,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=categories,
            y=s_squared,
            mode="markers",
            marker=dict(color="#dc2626", size=11, symbol="circle-open", line=dict(width=3)),
            name="σ² from SVD",
        ),
        row=1,
        col=2,
    )

    axis_config_left = dict(
        range=[-3.0, 3.0],
        zeroline=True,
        zerolinecolor="#cbd5e1",
        gridcolor="#f1f5f9",
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=axis_config_left,
        yaxis=axis_config_left,
        xaxis2=dict(gridcolor="#f1f5f9"),
        yaxis2=dict(title="Eigenvalue / σ²", gridcolor="#f1f5f9"),
        barmode="group",
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )

    return (
        a_rect,
        axis1,
        axis2,
        categories,
        circle_pts,
        eig_aat,
        eig_ata,
        ellipse_pts,
        fig,
        lambdas,
        q1,
        q2,
        q_mat,
        s_mat,
        s_squared,
        s_vals,
    )


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Numerical Verification of Spectral Theorem and Gram Equivalence

    Below, we generate a rectangular matrix $\mathbf{A} \in \mathbb{R}^{5 \times 3}$, construct its Gram matrices $\mathbf{A}^T\mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$, and verify:
    1. Reconstruction of the symmetric matrix from eigenvalues and eigenvectors: $\mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T = \mathbf{S}$
    2. Orthogonality of eigenvectors: $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$
    3. Non-negativity of all eigenvalues ($\lambda_i \geq 0$)
    4. Exact correspondence between Gram eigenvalues and SVD singular values: $|\lambda_i - \sigma_i^2| < 10^{-12}$
    """)
    return


@app.cell
def _(np):
    rng_ex1 = np.random.default_rng(42)

    # 5 x 3 rectangular matrix
    a_mat = rng_ex1.standard_normal((5, 3))

    # Gram matrices
    ata = a_mat.T @ a_mat
    aat = a_mat @ a_mat.T

    # Eigendecomposition of AtA
    eigvals_ata, q_ata = np.linalg.eigh(ata)
    # Sort descending
    sort_idx = np.argsort(eigvals_ata)[::-1]
    eigvals_ata = eigvals_ata[sort_idx]
    q_ata = q_ata[:, sort_idx]

    # Reconstruct AtA
    ata_reconstructed = q_ata @ np.diag(eigvals_ata) @ q_ata.T
    recon_error = float(np.max(np.abs(ata - ata_reconstructed)))

    # Orthogonality check of eigenvectors
    ortho_error = float(np.max(np.abs(q_ata.T @ q_ata - np.eye(3))))

    # SVD of A
    _, s_values, _ = np.linalg.svd(a_mat)
    svd_sq_err = float(np.max(np.abs(eigvals_ata - s_values**2)))

    # Minimum eigenvalue (confirming PSD property)
    min_eigval = float(np.min(eigvals_ata))

    spectral_summary = {
        "Property / Verification": [
            "Reconstruction error: max |Aᵀ A - Q Λ Qᵀ|",
            "Orthogonality error: max |Qᵀ Q - I|",
            "Discrepancy: max |λ(Aᵀ A) - σ²(A)|",
            "Minimum eigenvalue λ_min (must be >= 0)",
            "All eigenvalues of Aᵀ A",
            "All squared singular values σ² of A",
        ],
        "Value": [
            f"{recon_error:.2e}",
            f"{ortho_error:.2e}",
            f"{svd_sq_err:.2e}",
            f"{min_eigval:.6f}",
            ", ".join(f"{v:.4f}" for v in eigvals_ata),
            ", ".join(f"{v:.4f}" for v in (s_values**2)),
        ],
    }

    return (
        a_mat,
        aat,
        ata,
        ata_reconstructed,
        eigvals_ata,
        min_eigval,
        ortho_error,
        q_ata,
        recon_error,
        s_values,
        spectral_summary,
        svd_sq_err,
    )


@app.cell(hide_code=True)
def _(mo, pd, spectral_summary):
    df_spectral = pd.DataFrame(spectral_summary)
    mo.ui.table(df_spectral)
    return (df_spectral,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 3: Spectral Graph Partitioning via the Fiedler Vector

    In spectral graph theory and Graph Neural Networks, community detection is performed by computing the eigendecomposition of the **Graph Laplacian**:

    $$
    \mathbf{L} = \mathbf{D} - \mathbf{A}_{\text{adj}}
    $$

    where $\mathbf{D}$ is the diagonal degree matrix and $\mathbf{A}_{\text{adj}}$ is the adjacency matrix.

    Key mathematical properties of the Graph Laplacian:
    * $\mathbf{L}$ is symmetric and positive semi-definite ($\mathbf{L} \succeq 0$).
    * The smallest eigenvalue is always $\lambda_1 = 0$, with eigenvector $\mathbf{v}_1 = \frac{1}{\sqrt{n}} \mathbf{1}$.
    * The second smallest eigenvalue $\lambda_2$ is the **algebraic connectivity** (Fiedler value).
    * Its eigenvector $\mathbf{v}_2$ (the **Fiedler vector**) solves the relaxed normalized cut problem: sorting nodes by the sign of $\mathbf{v}_2(i)$ partitions the graph into optimal clusters.

    Below, we construct a 6-node barbell graph with two 3-node cliques joined by a single edge, compute $\mathbf{L}$, and verify exact community separation via the Fiedler vector.
    """)
    return


@app.cell
def _(np):
    # Construct 6-node graph with two clusters: {0, 1, 2} and {3, 4, 5} connected by edge (2, 3)
    adj = np.zeros((6, 6))

    # Cluster 1: nodes 0, 1, 2
    adj[0, 1] = adj[1, 0] = 1
    adj[0, 2] = adj[2, 0] = 1
    adj[1, 2] = adj[2, 1] = 1

    # Bridge edge
    adj[2, 3] = adj[3, 2] = 1

    # Cluster 2: nodes 3, 4, 5
    adj[3, 4] = adj[4, 3] = 1
    adj[3, 5] = adj[5, 3] = 1
    adj[4, 5] = adj[5, 4] = 1

    # Degree matrix
    degrees = np.sum(adj, axis=1)
    d_mat = np.diag(degrees)

    # Graph Laplacian
    laplacian = d_mat - adj

    # Spectral decomposition of symmetric Laplacian
    eigvals_l, eigvecs_l = np.linalg.eigh(laplacian)

    # Fiedler value and vector (index 1)
    fiedler_value = float(eigvals_l[1])
    fiedler_vector = eigvecs_l[:, 1]

    # Cluster assignment by sign of Fiedler vector
    cluster_assignment = ["Cluster 1" if val < 0 else "Cluster 2" for val in fiedler_vector]

    graph_summary = {
        "Node": [f"Node {i}" for i in range(6)],
        "Degree": [int(d) for d in degrees],
        "Fiedler Vector Value v₂(i)": [f"{v:+.4f}" for v in fiedler_vector],
        "Predicted Partition": cluster_assignment,
        "Ground Truth Community": [
            "Community A",
            "Community A",
            "Community A",
            "Community B",
            "Community B",
            "Community B",
        ],
    }

    return (
        adj,
        cluster_assignment,
        d_mat,
        degrees,
        eigvals_l,
        eigvecs_l,
        fiedler_value,
        fiedler_vector,
        graph_summary,
        laplacian,
    )


@app.cell(hide_code=True)
def _(mo, pd, graph_summary):
    df_graph = pd.DataFrame(graph_summary)
    mo.ui.table(df_graph)
    return (df_graph,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    * **Spectral Theorem**: Every real symmetric matrix $\mathbf{S}$ can be diagonalized by an orthogonal matrix $\mathbf{Q}$ and purely real eigenvalues $\mathbf{\Lambda}$. Geometrically, $\mathbf{S}$ acts by stretching vectors along mutually perpendicular principal axes.
    * **Positive Semi-Definiteness**: A symmetric matrix is PSD ($\mathbf{x}^T \mathbf{S} \mathbf{x} \ge 0$) if and only if all its eigenvalues are non-negative.
    * **Gram Matrices**: The matrices $\mathbf{A}^T \mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$ are unconditionally symmetric and PSD for any rectangular matrix $\mathbf{A}$.
    * **The SVD Bridge**: The eigenvectors of $\mathbf{A}^T \mathbf{A}$ and $\mathbf{A}\mathbf{A}^T$ exactly form the right ($\mathbf{V}$) and left ($\mathbf{U}$) singular vectors of $\mathbf{A}$, while their non-zero eigenvalues are the squared singular values $\sigma_i^2$.

    ---

    &larr; Previous Note: [06 Moore-Penrose Pseudoinverse](06_moore_penrose_inverse.py) | Next Note: [08 Matrix Calculus](08_matrix_calculus_short.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
