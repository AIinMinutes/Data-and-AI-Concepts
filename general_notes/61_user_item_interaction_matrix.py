import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    # Note 61: User-Item Interaction Matrix

    &larr; Previous Note: [60 VAE Anomaly Detection](60_vae_anomaly_detection.py) | [Index](../index.html) | Next Note: [62 Grammar of Graphics](../random_notes/62_grammar_of_graphics.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    The **user-item interaction matrix** is the foundational structure underlying collaborative filtering, recommender systems, and bipartite graph analytics. In real-world platforms (e-commerce, streaming services, social networks), users interact with subsets of a large catalog of items. These interactions can be **explicit** (numerical ratings, upvotes) or **implicit** (page views, clicks, purchases, dwell time).

    Understanding the algebraic and geometric structure of this matrix unlocks key analytical paradigms:
    - Bipartite graph representation and spectral graph theory
    - Dual user-user and item-item projection spaces via Gram matrices
    - Cosine and Jaccard similarity normalizations
    - Low-rank matrix factorization via Truncated Singular Value Decomposition (SVD) and the Eckart-Young-Mirsky theorem
    - Shared latent embedding spaces unifying users and items
    \"\"\")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Role in Machine Learning, AI, and Statistics

    The user-item matrix is the primary data structure for **Collaborative Filtering** and modern **Recommender Systems**:
    * **Matrix Factorization (ALS & SGD)**: To handle extreme sparsity, machine learning models decompose the matrix into lower-dimensional user and item embeddings. Algorithms like Alternating Least Squares (ALS) and Stochastic Gradient Descent (SGD) optimize over only the observed interactions.
    * **Two-Tower Neural Networks**: In deep learning architectures (like the YouTube recommendation system), user and item features are independently passed through neural networks (towers) to produce dense embeddings in a shared latent space, which are then scored using an inner product, effectively approximating the user-item matrix.
    * **Graph Neural Networks (GNNs)**: The bipartite graph representation allows GNNs (like LightGCN) to propagate embeddings iteratively between users and items, capturing higher-order network connectivity.

    ### Mathematical Formulation

    Let $\mathcal{U} = \{u_1, u_2, \dots, u_m\}$ denote a set of $m$ users and $\mathcal{V} = \{v_1, v_2, \dots, v_n\}$ denote a set of $n$ items. The interaction matrix is defined as:

    $$
    R \in \mathbb{R}^{m \times n}
    $$

    where entry $R_{ij}$ captures the interaction magnitude between user $u_i$ and item $v_j$. For binary implicit feedback:

    $$
    R_{ij} = \begin{cases} 1 & \text{if user } u_i \text{ interacted with item } v_j \\ 0 & \text{otherwise} \end{cases}
    $$

    ### Matrix Sparsity

    In practical applications, $m$ and $n$ range from $10^4$ to $10^9$. Most users engage with only a minuscule fraction of items. The matrix sparsity $S$ is quantified as:

    $$
    S = 1 - \frac{\|R\|_0}{m \cdot n} = 1 - \frac{|\{(i, j) : R_{ij} \neq 0\}|}{m \cdot n}
    $$

    In production systems, $S$ routinely exceeds $99\%$, necessitating sparse linear algebra (Compressed Sparse Row/Column formats) and specialized factorization algorithms.
    \"\"\")
    return


@app.cell
def _(np):
    # Synthetic user-item matrix with 8 users and 10 items
    # Categories: Items 0-2 (Sci-Fi / Tech), Items 3-5 (Action / Thriller), Items 6-9 (Romance / Comedy)
    user_names = [f"User {i+1}" for i in range(8)]
    item_names = [
        "Sci-Fi A", "Sci-Fi B", "Tech Doc",
        "Action A", "Action B", "Thriller",
        "Romance A", "Romance B", "Comedy A", "Comedy B"
    ]

    # Ground-truth binary interactions reflecting latent preference clusters
    R = np.array([
        [1, 1, 1, 0, 1, 0, 0, 0, 0, 0],  # User 1: Tech/Sci-Fi enthusiast
        [1, 0, 1, 1, 0, 0, 0, 0, 0, 0],  # User 2: Sci-Fi & Action
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0],  # User 3: Pure Sci-Fi
        [0, 0, 0, 1, 1, 1, 0, 0, 1, 0],  # User 4: Action / Thriller
        [0, 0, 0, 1, 1, 1, 0, 0, 0, 0],  # User 5: Pure Action
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],  # User 6: Romance / Comedy / Light Thriller
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 0],  # User 7: Romance / Comedy
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # User 8: Romance / Comedy
    ], dtype=float)

    n_users, n_items = R.shape
    nnz = np.count_nonzero(R)
    sparsity_ratio = 1.0 - (nnz / (n_users * n_items))
    return item_names, n_items, n_users, nnz, R, sparsity_ratio, user_names


@app.cell(hide_code=True)
def _(go, item_names, mo, n_items, n_users, nnz, R, sparsity_ratio, user_names):
    _fig = go.Figure(
        data=go.Heatmap(
            z=R,
            x=item_names,
            y=user_names,
            colorscale=[[0.0, "#0f172a"], [1.0, "#38bdf8"]],
            showscale=False,
            text=[[f"R[{i},{j}] = {int(R[i, j])}" for j in range(n_items)] for i in range(n_users)],
            hoverinfo="text",
        )
    )

    for _i in range(n_users):
        for _j in range(n_items):
            _val = int(R[_i, _j])
            _fig.add_annotation(
                x=item_names[_j],
                y=user_names[_i],
                text=str(_val),
                showarrow=False,
                font=dict(color="#ffffff" if _val == 1 else "#64748b", size=12),
            )

    _fig.update_layout(
        title=f"Binary User-Item Interaction Matrix R ({n_users} Users x {n_items} Items | Sparsity: {sparsity_ratio:.1%})",
        xaxis_title="Catalog Items",
        yaxis_title="Users",
        yaxis_autorange="reversed",
        template="plotly_white",
        height=450,
        margin=dict(l=80, r=40, t=60, b=80),
    )

    _md = mo.md(
        f"--- \n\n## [c] Interactive Visualizations\n\n"
        f"### Interaction Matrix Visualization\n\n"
        f"Total elements: **{n_users * n_items}** | "
        f"Non-zero interactions ($\\|R\\|_0$): **{nnz}** | "
        f"Sparsity: **{sparsity_ratio:.2%}**"
    )

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    ---

    ## Bipartite Graph Formulation & Adjacency Representation

    The user-item interaction matrix corresponds directly to a **bipartite graph** $G = (\mathcal{U} \cup \mathcal{V}, \mathcal{E})$, where:
    - User nodes $\mathcal{U}$ have no internal edges among themselves: $(u_i, u_k) \notin \mathcal{E}$.
    - Item nodes $\mathcal{V}$ have no internal edges among themselves: $(v_j, v_l) \notin \mathcal{E}$.
    - Every edge connects a user to an item: $(u_i, v_j) \in \mathcal{E} \iff R_{ij} \neq 0$.

    The complete $(m+n) \times (m+n)$ block adjacency matrix $B$ of the bipartite graph is structured as:

    $$
    B = \begin{pmatrix} 0_{m \times m} & R \\ R^T & 0_{n \times n} \end{pmatrix}
    $$

    The graph degree matrix is $D = \operatorname{diag}(d_1, \dots, d_{m+n})$, where user degrees equal row sums $\sum_j R_{ij}$ and item degrees equal column sums $\sum_i R_{ij}$.
    \"\"\")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    ---

    ## Dual Projections: User-User and Item-Item Geometries

    By multiplying $R$ with its transpose, we project the bipartite interaction structure onto homogeneous user-user and item-item similarity spaces.

    ### 1. User-User Co-occurrence and Similarity

    The user-user co-occurrence matrix $C_U \in \mathbb{R}^{m \times m}$ is:

    $$
    C_U = R R^T
    $$

    The entry $(C_U)_{ik} = \sum_{j=1}^n R_{ij} R_{kj}$ counts the exact number of shared items interacted with by both user $u_i$ and user $u_k$. The diagonal entry $(C_U)_{ii} = \|r_i\|_2^2$ is the total interaction count of user $u_i$.

    To normalize for varying user activity levels, we compute the **Cosine Similarity**:

    $$
    S_U^{\text{cos}}(i, k) = \frac{r_i \cdot r_k}{\|r_i\|_2 \|r_k\|_2} = \frac{(R R^T)_{ik}}{\sqrt{(R R^T)_{ii} (R R^T)_{kk}}}
    $$

    and the **Jaccard Similarity Coefficient** for binary sets:

    $$
    J_U(i, k) = \frac{|N(u_i) \cap N(u_k)|}{|N(u_i) \cup N(u_k)|} = \frac{(R R^T)_{ik}}{(R R^T)_{ii} + (R R^T)_{kk} - (R R^T)_{ik}}
    $$

    ### 2. Item-Item Co-occurrence and Similarity

    Symmetrically, the item-item co-occurrence matrix $C_V \in \mathbb{R}^{n \times n}$ is:

    $$
    C_V = R^T R
    $$

    The entry $(C_V)_{jl} = \sum_{i=1}^m R_{ij} R_{il}$ counts the number of users who interacted with both item $v_j$ and item $v_l$. The normalized item cosine similarity is:

    $$
    S_V^{\text{cos}}(j, l) = \frac{(R^T R)_{jl}}{\sqrt{(R^T R)_{jj} (R^T R)_{ll}}}
    $$
    \"\"\")
    return


@app.cell
def _(np, R):
    # Compute user-user co-occurrence and cosine similarity
    C_user = R @ R.T
    user_norms = np.linalg.norm(R, axis=1, keepdims=True)
    user_norms_safe = np.where(user_norms == 0, 1.0, user_norms)
    S_user = (R @ R.T) / (user_norms_safe @ user_norms_safe.T)

    # Compute item-item co-occurrence and cosine similarity
    C_item = R.T @ R
    item_norms = np.linalg.norm(R, axis=0, keepdims=True)
    item_norms_safe = np.where(item_norms == 0, 1.0, item_norms)
    S_item = (R.T @ R) / (item_norms_safe.T @ item_norms_safe)

    # Compute Jaccard similarity for users
    J_user = np.zeros_like(C_user)
    for _i in range(R.shape[0]):
        for _k in range(R.shape[0]):
            _intersection = C_user[_i, _k]
            _union = C_user[_i, _i] + C_user[_k, _k] - _intersection
            J_user[_i, _k] = _intersection / _union if _union > 0 else 0.0

    return C_item, C_user, item_norms, item_norms_safe, J_user, S_item, S_user, user_norms, user_norms_safe


@app.cell(hide_code=True)
def _(
    C_item,
    C_user,
    go,
    item_names,
    make_subplots,
    mo,
    S_item,
    S_user,
    user_names,
):
    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "User-User Cosine Similarity S_U = (R R')_norm",
            "Item-Item Cosine Similarity S_V = (R' R)_norm",
        ],
        horizontal_spacing=0.15,
    )

    _fig.add_trace(
        go.Heatmap(
            z=S_user,
            x=user_names,
            y=user_names,
            colorscale="Viridis",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Similarity", x=0.43, len=0.8),
            customdata=C_user,
            hovertemplate="User %{y} & User %{x}<br>Cosine: %{z:.3f}<br>Common Items: %{customdata}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Heatmap(
            z=S_item,
            x=item_names,
            y=item_names,
            colorscale="Plasma",
            zmin=0.0,
            zmax=1.0,
            colorbar=dict(title="Similarity", x=1.0, len=0.8),
            customdata=C_item,
            hovertemplate="Item %{y} & Item %{x}<br>Cosine: %{z:.3f}<br>Common Users: %{customdata}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _fig.update_layout(
        title="Dual Projections: User Neighborhoods vs Item Co-occurrence",
        template="plotly_white",
        height=480,
        margin=dict(l=60, r=60, t=70, b=80),
    )
    _fig.update_yaxes(autorange="reversed")

    _md = mo.md(r"""
    ### Dual Projections Heatmap

    Notice the clean block-diagonal structure:
    - Users 1 to 3 cluster tightly together around Sci-Fi/Tech items.
    - Users 4 and 5 form an Action cluster.
    - Users 6 to 8 form a Romance/Comedy cluster.
    - Item-item projections reveal high co-occurrence within content genres, enabling **item-to-item collaborative filtering** (e.g. Amazon's "Customers who bought this also bought...").
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    ---

    ## [d] Code Examples

    ### Low-Rank Matrix Factorization & SVD

    In recommender systems, user tastes and item attributes are driven by a small number $k \ll \min(m, n)$ of **unobserved latent factors** (e.g., genre preferences, tone, pace).

    ### Singular Value Decomposition (SVD)

    Any real interaction matrix $R \in \mathbb{R}^{m \times n}$ has a singular value decomposition:

    $$
    R = U \Sigma V^T = \sum_{r=1}^{\operatorname{rank}(R)} \sigma_r u_r v_r^T
    $$

    where:
    - $U \in \mathbb{R}^{m \times m}$ is an orthonormal matrix whose columns $u_r$ are the eigenvectors of $R R^T$.
    - $\Sigma \in \mathbb{R}^{m \times n}$ is a diagonal matrix containing singular values $\sigma_1 \ge \sigma_2 \ge \dots \ge \sigma_{\min(m,n)} \ge 0$.
    - $V \in \mathbb{R}^{n \times n}$ is an orthonormal matrix whose columns $v_r$ are the eigenvectors of $R^T R$.

    ### The Eckart-Young-Mirsky Theorem

    For any integer $k < \operatorname{rank}(R)$, the optimal rank-$k$ approximation $\hat{R}_k$ minimizing the Frobenius norm error:

    $$
    \hat{R}_k = \arg\min_{\operatorname{rank}(M) \le k} \|R - M\|_F
    $$

    is obtained by truncating the SVD at $k$ components:

    $$
    \hat{R}_k = U_k \Sigma_k V_k^T = P_k Q_k^T
    $$

    where:
    - User latent representation: $P_k = U_k \Sigma_k^{1/2} \in \mathbb{R}^{m \times k}$
    - Item latent representation: $Q_k = V_k \Sigma_k^{1/2} \in \mathbb{R}^{n \times k}$

    The minimum reconstruction error equals the sum of discarded squared singular values:

    $$
    \|R - \hat{R}_k\|_F^2 = \sum_{r=k+1}^{\min(m,n)} \sigma_r^2
    $$
    \"\"\")
    return


@app.cell
def _(np, R):
    # Perform full SVD
    U_svd, Sigma_vals, Vt_svd = np.linalg.svd(R, full_matrices=False)

    # Truncated approximations for ranks k = 1, 2, 3
    ranks = [1, 2, 3]
    R_hat_k = {}
    frob_errors = {}
    theoretical_errors = {}

    for _k in ranks:
        _R_rec = U_svd[:, :_k] @ np.diag(Sigma_vals[:_k]) @ Vt_svd[:_k, :]
        R_hat_k[_k] = _R_rec
        _err = float(np.linalg.norm(R - _R_rec, ord="fro") ** 2)
        frob_errors[_k] = _err
        theoretical_errors[_k] = float(np.sum(Sigma_vals[_k:] ** 2))

    # Shared 2D latent space embeddings (k = 2)
    sqrt_sigma_2 = np.sqrt(Sigma_vals[:2])
    P_2 = U_svd[:, :2] * sqrt_sigma_2  # Shape (m, 2)
    Q_2 = Vt_svd[:2, :].T * sqrt_sigma_2  # Shape (n, 2)

    return (
        frob_errors,
        P_2,
        Q_2,
        R_hat_k,
        ranks,
        Sigma_vals,
        sqrt_sigma_2,
        theoretical_errors,
        U_svd,
        Vt_svd,
    )


@app.cell(hide_code=True)
def _(frob_errors, mo, np, ranks, Sigma_vals, theoretical_errors):
    _rows = []
    _total_energy = float(np.sum(Sigma_vals**2))
    for _k in ranks:
        _err = frob_errors[_k]
        _theo = theoretical_errors[_k]
        _var_explained = 1.0 - (_err / _total_energy)
        _rows.append(
            f"| Rank {_k} | {Sigma_vals[_k-1]:.4f} | {_err:.4f} | {_theo:.4f} | {_var_explained:.2%} |"
        )
    _table_content = "\n".join(_rows)

    return mo.md(
        "### Eckart-Young-Mirsky Theorem Verification\n\n"
        "| Rank $k$ | Singular Value $\\sigma_k$ | Empirical $\\|R - \\hat{R}_k\\|_F^2$ | Theoretical $\\sum_{r=k+1} \\sigma_r^2$ | Variance Explained |\n"
        "| :--- | :--- | :--- | :--- | :--- |\n"
        f"{_table_content}\n\n"
        "Notice that the empirical Frobenius reconstruction error matches the theoretical sum of discarded singular values to machine precision."
    )


@app.cell(hide_code=True)
def _(go, item_names, make_subplots, mo, np, R, R_hat_k, user_names):
    # Plot original R vs Rank-2 reconstruction vs Residual
    _k = 2
    _R_rec = R_hat_k[_k]
    _residual = np.abs(R - _R_rec)

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "Original R (Binary 0/1)",
            f"Rank-{_k} Approximation R_hat (SVD)",
            "Absolute Residual |R - R_hat|",
        ],
        horizontal_spacing=0.08,
    )

    _fig.add_trace(
        go.Heatmap(
            z=R,
            x=item_names,
            y=user_names,
            colorscale="Blues",
            showscale=False,
            zmin=0,
            zmax=1,
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Heatmap(
            z=_R_rec,
            x=item_names,
            y=user_names,
            colorscale="RdBu_r",
            showscale=False,
            zmid=0.0,
            hovertemplate="User %{y}, Item %{x}<br>Predicted Affinity: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _fig.add_trace(
        go.Heatmap(
            z=_residual,
            x=item_names,
            y=user_names,
            colorscale="Reds",
            colorbar=dict(title="Residual", x=1.0, len=0.8),
            zmin=0,
            zmax=1,
            hovertemplate="User %{y}, Item %{x}<br>Residual Error: %{z:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        title=f"Matrix Factorization: Low-Rank Denoising and Recommendation Scoring (Rank k={_k})",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=60, t=60, b=60),
    )
    _fig.update_yaxes(autorange="reversed")

    _md = mo.md(r"""
    ### Low-Rank Reconstruction and Imputation

    In collaborative filtering, positive entries in $\hat{R}_k$ where $R_{ij} = 0$ represent **novel recommendations**! For instance, if User 2 has $R_{2, 4} = 0$ (has not seen Action B) but $\hat{R}_{2, 4} > 0.4$, the system flags Action B as a high-affinity candidate.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(go, item_names, mo, P_2, Q_2, user_names):
    # Plot Users and Items in the unified 2D Latent Factor Space
    _fig = go.Figure()

    # User coordinates
    _fig.add_trace(
        go.Scatter(
            x=P_2[:, 0],
            y=P_2[:, 1],
            mode="markers+text",
            marker=dict(size=14, color="#3b82f6", symbol="circle", line=dict(width=2, color="#1e40af")),
            text=user_names,
            textposition="top center",
            name="Users (P)",
            hovertemplate="<b>%{text}</b><br>Factor 1: %{x:.3f}<br>Factor 2: %{y:.3f}<extra></extra>",
        )
    )

    # Item coordinates
    _fig.add_trace(
        go.Scatter(
            x=Q_2[:, 0],
            y=Q_2[:, 1],
            mode="markers+text",
            marker=dict(size=14, color="#ef4444", symbol="diamond", line=dict(width=2, color="#991b1b")),
            text=item_names,
            textposition="bottom center",
            name="Items (Q)",
            hovertemplate="<b>%{text}</b><br>Factor 1: %{x:.3f}<br>Factor 2: %{y:.3f}<extra></extra>",
        )
    )

    # Add zero-reference dashed axes
    _fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8", opacity=0.6)
    _fig.add_vline(x=0, line_dash="dash", line_color="#94a3b8", opacity=0.6)

    _fig.update_layout(
        title="Joint Latent Embedding Space (Users and Items in R^2)",
        xaxis_title="Latent Factor 1 (Genre Spectrum: Sci-Fi / Tech vs Romance / Comedy)",
        yaxis_title="Latent Factor 2 (Action / Thriller Intensity)",
        template="plotly_white",
        height=520,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    _md = mo.md(r"""
    ---

    ## Unified Latent Embedding Geometry

    Because $\hat{R}_k = P_k Q_k^T$, the predicted affinity between user $u_i$ and item $v_j$ is the **inner product** of their latent factor vectors:

    $$
    \hat{R}_{ij} = \langle p_i, q_j \rangle = \|p_i\|_2 \|q_j\|_2 \cos \theta_{ij}
    $$

    When users and items are projected into the same shared coordinate system, users cluster directly alongside the items they prefer!
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    mo.md(r\"\"\"
    ---

    ## [e] Takeaway

    1. **Bipartite Duality**:
       - The interaction matrix $R$ encodes edges between disjoint sets $\mathcal{U}$ and $\mathcal{V}$.
       - $R R^T$ reveals the user-user projection; $R^T R$ reveals the item-item projection.
    2. **Handling Extreme Sparsity**:
       - Standard dense SVD has $\mathcal{O}(\min(m^2 n, m n^2))$ computational complexity and memory cost $\mathcal{O}(mn)$, which is infeasible for millions of users/items.
       - Modern recommenders use **Alternating Least Squares (ALS)** with implicit feedback weighting (Hu, Koren, Volinsky), **Stochastic Gradient Descent (SGD)** on observed pairs, or two-tower neural architectures (e.g. YouTube candidate generation).
    3. **The Cold-Start Problem**:
       - Pure interaction matrices suffer when new users or items with zero historical rows/columns join the system ($\|r_{\text{new}}\|_0 = 0$).
       - Hybrid recommenders bridge this gap by concatenating content features (metadata, embeddings) with interaction matrices.
    \"\"\")
    return


if __name__ == "__main__":
    app.run()
