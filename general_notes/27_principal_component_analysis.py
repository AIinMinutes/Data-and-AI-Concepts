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
    from sklearn.decomposition import PCA

    return PCA, go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 27: Principal Component Analysis, Maximum Variance Projections, and SVD Geometry

    &larr; Previous Note: [26 Hotelling T-Squared](26_hotelling.py) | Next Note: [28 Factor Analysis](28_factor_analysis.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In modern machine learning, high-dimensional datasets (embeddings, tabular feature collections, spectral signals, gene expression profiles) frequently contain hundreds or thousands of features. However, real-world data is rarely spread uniformly across all dimensions; it concentrates along an underlying low-dimensional manifold.

    **Principal Component Analysis (PCA)** is the foundational linear dimensionality reduction technique in statistical learning:
    1. **The Dual Formulations (Pearson & Hotelling)**:
       - **Maximum Variance (Hotelling 1933)**: Find orthogonal directions $\mathbf{w}_1, \dots, \mathbf{w}_k$ onto which the projection of the data retains maximum possible variance.
       - **Minimum Reconstruction Error (Pearson 1901)**: Find the $k$-dimensional linear subspace that minimizes the mean squared Euclidean distance between the original data points and their orthogonal projections.
       The Eckart-Young-Mirsky theorem establishes that both objectives lead to the exact same optimal subspace defined by the leading eigenvectors of the covariance matrix.
    2. **Orthogonal Decorrelation**: In the projected principal component coordinate system $\mathbf{Z} = \tilde{\mathbf{X}}\mathbf{W}$, all cross-covariances between distinct components are identically zero ($\operatorname{Cov}(Z_i, Z_j) = 0$ for $i \neq j$), completely eliminating multicollinearity for downstream linear and logistic regression models.
    3. **Data Compression and Noise Filtering**: In many physical systems, small eigenvalues represent isotropic background noise. Truncating the bottom components compresses data while actively filtering out unstructured high-frequency noise.
    4. **Bridge to Singular Value Decomposition (SVD)**: Computing PCA through thin SVD of the centered data matrix $\tilde{\mathbf{X}} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^\top$ avoids forming the $d \times d$ covariance matrix $\tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}$, providing numerical stability and optimal algorithmic performance ($\mathcal{O}(nd \min(n, d))$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Data Centering and the Sample Covariance Matrix

    Consider a dataset of $n$ observations across $d$ features represented as a matrix $\mathbf{X} \in \mathbb{R}^{n \times d}$. The empirical sample mean vector is:

    $$
    \bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i \in \mathbb{R}^d
    $$

    The **mean-centered data matrix** $\tilde{\mathbf{X}} \in \mathbb{R}^{n \times d}$ is obtained by subtracting the mean:

    $$
    \tilde{\mathbf{X}} = \mathbf{X} - \mathbf{1}_n \bar{\mathbf{x}}^\top
    $$

    The sample covariance matrix $\mathbf{S} \in \mathbb{R}^{d \times d}$ is:

    $$
    \mathbf{S} = \frac{1}{n - 1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}
    $$

    $\mathbf{S}$ is real, symmetric ($\mathbf{S} = \mathbf{S}^\top$), and positive semi-definite ($\mathbf{w}^\top \mathbf{S} \mathbf{w} \geq 0$ for all $\mathbf{w}$).

    ---

    ### 2. Derivation of the First Principal Component (Lagrangian Optimization)

    We seek a unit projection vector $\mathbf{w}_1 \in \mathbb{R}^d$ ($\|\mathbf{w}_1\|_2 = 1$) such that the variance of the projected scalar coordinates $z_{i1} = \tilde{\mathbf{x}}_i^\top \mathbf{w}_1$ is maximized:

    $$
    \operatorname{Var}(\tilde{\mathbf{X}}\mathbf{w}_1) = \frac{1}{n - 1} (\tilde{\mathbf{X}}\mathbf{w}_1)^\top (\tilde{\mathbf{X}}\mathbf{w}_1) = \mathbf{w}_1^\top \left(\frac{1}{n - 1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}}\right) \mathbf{w}_1 = \mathbf{w}_1^\top \mathbf{S} \mathbf{w}_1
    $$

    The constrained optimization problem is:

    $$
    \max_{\mathbf{w}_1} \mathbf{w}_1^\top \mathbf{S} \mathbf{w}_1 \quad \text{subject to} \quad \mathbf{w}_1^\top \mathbf{w}_1 = 1
    $$

    Formulating the Lagrangian with Lagrange multiplier $\lambda_1$:

    $$
    \mathcal{L}(\mathbf{w}_1, \lambda_1) = \mathbf{w}_1^\top \mathbf{S} \mathbf{w}_1 - \lambda_1 (\mathbf{w}_1^\top \mathbf{w}_1 - 1)
    $$

    Computing the gradient with respect to $\mathbf{w}_1$ and equating to zero:

    $$
    \nabla_{\mathbf{w}_1} \mathcal{L} = 2\mathbf{S}\mathbf{w}_1 - 2\lambda_1 \mathbf{w}_1 = \mathbf{0} \implies \mathbf{S}\mathbf{w}_1 = \lambda_1 \mathbf{w}_1
    $$

    This is the classical **matrix eigenvalue equation**! Multiplying both sides on the left by $\mathbf{w}_1^\top$:

    $$
    \mathbf{w}_1^\top \mathbf{S} \mathbf{w}_1 = \lambda_1 \mathbf{w}_1^\top \mathbf{w}_1 = \lambda_1
    $$

    Therefore, the maximum projected variance equals the eigenvalue $\lambda_1$. To maximize variance, $\mathbf{w}_1$ must be the eigenvector of $\mathbf{S}$ corresponding to its largest eigenvalue $\lambda_1 = \lambda_{\max}$.

    ---

    ### 3. Subsequent Orthogonal Principal Components

    For the $k$-th principal component ($k \leq d$), we maximize variance subject to unit length and mutual orthogonality to all previously extracted components:

    $$
    \max_{\mathbf{w}_k} \mathbf{w}_k^\top \mathbf{S} \mathbf{w}_k \quad \text{subject to} \quad \mathbf{w}_k^\top \mathbf{w}_k = 1 \quad \text{and} \quad \mathbf{w}_k^\top \mathbf{w}_j = 0 \ \forall j < k
    $$

    By the Spectral Theorem for symmetric matrices, $\mathbf{S}$ has an orthogonal eigendecomposition:

    $$
    \mathbf{S} = \mathbf{W} \boldsymbol{\Lambda} \mathbf{W}^\top
    $$

    where $\mathbf{W} = [\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_d]$ is an orthonormal matrix of eigenvectors ($\mathbf{W}^\top \mathbf{W} = \mathbf{I}_d$), and $\boldsymbol{\Lambda} = \operatorname{diag}(\lambda_1, \lambda_2, \dots, \lambda_d)$ with $\lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0$.

    ---

    ### 4. Principal Component Scores and Decorrelation

    The transformed coordinates $\mathbf{Z} \in \mathbb{R}^{n \times d}$ (called **principal component scores**) are:

    $$
    \mathbf{Z} = \tilde{\mathbf{X}} \mathbf{W}
    $$

    The sample covariance of the transformed scores is:

    $$
    \operatorname{Cov}(\mathbf{Z}) = \frac{1}{n - 1} \mathbf{Z}^\top \mathbf{Z} = \frac{1}{n - 1} \mathbf{W}^\top \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} \mathbf{W} = \mathbf{W}^\top \mathbf{S} \mathbf{W} = \boldsymbol{\Lambda}
    $$

    Because $\boldsymbol{\Lambda}$ is diagonal, all principal components are **mutually uncorrelated**:

    $$
    \operatorname{Cov}(Z_j, Z_k) = 0 \quad \forall j \neq k
    $$

    ---

    ### 5. Proportion of Variance Explained (PVE) and Scree Criterion

    The total sample variance is the trace of $\mathbf{S}$:

    $$
    \operatorname{tr}(\mathbf{S}) = \sum_{j=1}^d s_{jj} = \sum_{j=1}^d \lambda_j
    $$

    The Proportion of Variance Explained (PVE) by the $k$-th principal component is:

    $$
    \text{PVE}_k = \frac{\lambda_k}{\sum_{j=1}^d \lambda_j}
    $$

    The cumulative proportion of variance explained by the first $K$ components is:

    $$
    \text{Cumulative PVE}_K = \frac{\sum_{k=1}^K \lambda_k}{\sum_{j=1}^d \lambda_j}
    $$

    ---

    ### 6. SVD Duality and Reconstruction

    Let the Singular Value Decomposition of the centered matrix be:

    $$
    \tilde{\mathbf{X}} = \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top
    $$

    Then:

    $$
    \mathbf{S} = \frac{1}{n - 1} \tilde{\mathbf{X}}^\top \tilde{\mathbf{X}} = \frac{1}{n - 1} \mathbf{V} \boldsymbol{\Sigma} \mathbf{U}^\top \mathbf{U} \boldsymbol{\Sigma} \mathbf{V}^\top = \mathbf{V} \left(\frac{\boldsymbol{\Sigma}^2}{n - 1}\right) \mathbf{V}^\top
    $$

    Thus, the right singular vectors $\mathbf{V}$ are identically the principal component loading vectors ($\mathbf{W} = \mathbf{V}$), and the eigenvalues satisfy:

    $$
    \lambda_k = \frac{\sigma_k^2}{n - 1}
    $$

    The rank-$K$ low-rank reconstruction of the centered data is:

    $$
    \hat{\mathbf{X}}_K = \mathbf{Z}_K \mathbf{W}_K^\top = \tilde{\mathbf{X}} \mathbf{W}_K \mathbf{W}_K^\top
    $$
    """)
    return


@app.cell
def _(np, pd):
    # Simulation Data: Bivariate Elliptical Gaussian (n = 120) with Strong Correlation
    np.random.seed(20250101)
    _n = 120

    _mean_true = [3.0, 2.0]
    _cov_true = [[10.0, 8.5], [8.5, 12.0]]
    raw_samples = np.random.multivariate_normal(_mean_true, _cov_true, size=_n)

    # Standardize data
    x_standardized = (raw_samples - raw_samples.mean(axis=0)) / raw_samples.std(axis=0, ddof=1)
    df_pca_2d = pd.DataFrame(x_standardized, columns=["Feature_1", "Feature_2"])

    return df_pca_2d, x_standardized


@app.cell
def _(PCA, df_pca_2d, go, make_subplots, mo, np, x_standardized):
    # Interactive Visualizations Cell:
    # Subplot 1: Standardized Data Scatter + Principal Component Loading Vectors (scaled by 2 * sqrt(lambda))
    # Subplot 2: Projected PC Space (Z1 vs Z2: Decorrelated, Aligned with Axes)
    # Subplot 3: Scree Plot & Cumulative Explained Variance Ratio

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Original Space & Eigenvector Axes",
            "2. Decorrelated PC Score Space",
            "3. Scree Plot & Cumulative Variance",
        ),
        horizontal_spacing=0.09,
    )

    # Compute PCA
    _pca = PCA()
    _z_scores = _pca.fit_transform(x_standardized)
    _eigenvals = _pca.explained_variance_
    _eigenvecs = _pca.components_.T  # columns are eigenvectors
    _pve = _pca.explained_variance_ratio_

    # Subplot 1: Data + Eigenvectors
    _fig.add_trace(
        go.Scatter(
            x=df_pca_2d["Feature_1"],
            y=df_pca_2d["Feature_2"],
            mode="markers",
            marker=dict(size=6, color="#64748b", opacity=0.7),
            name="Centered Samples",
            hovertemplate="F1: %{x:.2f}<br>F2: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add PC1 vector (scaled by 2 * sqrt(lambda_1) for 2-sigma visualization)
    _scale_pc1 = 2.0 * np.sqrt(_eigenvals[0])
    _scale_pc2 = 2.0 * np.sqrt(_eigenvals[1])

    _v1 = _eigenvecs[:, 0] * _scale_pc1
    _v2 = _eigenvecs[:, 1] * _scale_pc2

    _fig.add_trace(
        go.Scatter(
            x=[0, _v1[0]],
            y=[0, _v1[1]],
            mode="lines+markers",
            line=dict(color="#06b6d4", width=3.5),
            marker=dict(size=8, symbol="arrow-bar-up"),
            name=f"PC1 Vector (PVE = {_pve[0]*100:.1f}%)",
        ),
        row=1,
        col=1,
    )

    _fig.add_trace(
        go.Scatter(
            x=[0, _v2[0]],
            y=[0, _v2[1]],
            mode="lines+markers",
            line=dict(color="#ec4899", width=3.5),
            marker=dict(size=8, symbol="arrow-bar-up"),
            name=f"PC2 Vector (PVE = {_pve[1]*100:.1f}%)",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Projected PC Space
    _fig.add_trace(
        go.Scatter(
            x=_z_scores[:, 0],
            y=_z_scores[:, 1],
            mode="markers",
            marker=dict(size=6, color="#10b981", opacity=0.75),
            name="PC Coordinates",
            hovertemplate="PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Scree Plot
    _components = ["PC 1", "PC 2"]
    _cum_pve = np.cumsum(_pve)

    _fig.add_trace(
        go.Bar(
            x=_components,
            y=_pve * 100.0,
            marker_color=["#06b6d4", "#ec4899"],
            name="Individual PVE (%)",
            text=[f"{v*100:.1f}%" for v in _pve],
            textposition="auto",
        ),
        row=1,
        col=3,
    )

    _fig.add_trace(
        go.Scatter(
            x=_components,
            y=_cum_pve * 100.0,
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2.5),
            marker=dict(size=8),
            name="Cumulative PVE (%)",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Principal Component Analysis: Maximum Variance Axes, Decorrelation, and PVE",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Standardized Feature 1", range=[-3.5, 3.5], row=1, col=1)
    _fig.update_yaxes(title_text="Standardized Feature 2", range=[-3.5, 3.5], row=1, col=1)

    _fig.update_xaxes(title_text="Principal Component 1 Score", range=[-3.5, 3.5], row=1, col=2)
    _fig.update_yaxes(title_text="Principal Component 2 Score", range=[-3.5, 3.5], row=1, col=2)

    _fig.update_xaxes(title_text="Principal Component", row=1, col=3)
    _fig.update_yaxes(title_text="Explained Variance Ratio (%)", range=[0, 110], row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Full Step-by-Step PCA from Scratch**: Pure NumPy implementation computing sample covariance, eigendecomposition, eigenvalue sorting, projection matrix, transformed scores, and low-rank reconstruction, verified against `sklearn.decomposition.PCA` to machine precision.
    2. **Multi-Feature Dimensionality Reduction & Reconstruction Diagnostics**: Generating a 5-dimensional correlated physical telemetry dataset, performing PCA, and evaluating the reconstruction error $\|\mathbf{X} - \hat{\mathbf{X}}_K\|_F$ as a function of retained components $K \in \{1, \dots, 5\}$.
    """)
    return


@app.cell
def _(PCA, mo, np, pd, x_standardized):
    # Example 1: Full Step-by-Step PCA from Scratch vs Scikit-Learn
    _n = x_standardized.shape[0]

    # 1. Manual Covariance Matrix
    _cov_manual = (x_standardized.T @ x_standardized) / (_n - 1)

    # 2. Eigendecomposition
    _eigenvals_raw, _eigenvecs_raw = np.linalg.eigh(_cov_manual)

    # 3. Sort eigenvalues descending
    _sort_idx = np.argsort(_eigenvals_raw)[::-1]
    _eigenvals_manual = _eigenvals_raw[_sort_idx]
    _eigenvecs_manual = _eigenvecs_raw[:, _sort_idx]

    # Adjust sign convention to match Scikit-Learn (ensure positive component on largest absolute loading)
    for _j in range(_eigenvecs_manual.shape[1]):
        _max_abs_idx = np.argmax(np.abs(_eigenvecs_manual[:, _j]))
        if _eigenvecs_manual[_max_abs_idx, _j] < 0:
            _eigenvecs_manual[:, _j] *= -1.0

    # 4. Transform scores
    _scores_manual = x_standardized @ _eigenvecs_manual

    # Scikit-Learn Reference
    _pca_sklearn = PCA().fit(x_standardized)
    _scores_sklearn = _pca_sklearn.transform(x_standardized)

    _summary_rows = [
        {
            "Quantity": "Eigenvalue 1 (lambda_1)",
            "Manual Scratch Value": f"{_eigenvals_manual[0]:.6f}",
            "Scikit-Learn Value": f"{_pca_sklearn.explained_variance_[0]:.6f}",
            "Discrepancy": f"{np.abs(_eigenvals_manual[0] - _pca_sklearn.explained_variance_[0]):.2e}",
        },
        {
            "Quantity": "Eigenvalue 2 (lambda_2)",
            "Manual Scratch Value": f"{_eigenvals_manual[1]:.6f}",
            "Scikit-Learn Value": f"{_pca_sklearn.explained_variance_[1]:.6f}",
            "Discrepancy": f"{np.abs(_eigenvals_manual[1] - _pca_sklearn.explained_variance_[1]):.2e}",
        },
        {
            "Quantity": "PC1 Variance Explained Ratio",
            "Manual Scratch Value": f"{_eigenvals_manual[0] / np.sum(_eigenvals_manual):.6f}",
            "Scikit-Learn Value": f"{_pca_sklearn.explained_variance_ratio_[0]:.6f}",
            "Discrepancy": "Exact Match",
        },
        {
            "Quantity": "Max Score Coordinate Discrepancy",
            "Manual Scratch Value": f"{np.max(np.abs(_scores_manual - _scores_sklearn)):.2e}",
            "Scikit-Learn Value": "Reference",
            "Discrepancy": "Machine Precision Match",
        },
    ]

    _df_scratch_comparison = pd.DataFrame(_summary_rows)

    return (
        mo.md("#### Step-by-Step PCA Scratch vs. Scikit-Learn Benchmark"),
        mo.ui.table(_df_scratch_comparison),
    )


@app.cell
def _(PCA, mo, np, pd):
    # Example 2: 5-Dimensional Telemetry Dataset Low-Rank Reconstruction
    np.random.seed(42)
    _n_samples = 150

    # Underlying 2-dimensional latent signal
    _latent_z1 = np.random.normal(0, 3.0, _n_samples)
    _latent_z2 = np.random.normal(0, 1.5, _n_samples)

    # 5 observed correlated telemetry channels
    _f1 = 1.0 * _latent_z1 + 0.2 * _latent_z2 + np.random.normal(0, 0.4, _n_samples)
    _f2 = 0.9 * _latent_z1 - 0.5 * _latent_z2 + np.random.normal(0, 0.4, _n_samples)
    _f3 = -1.2 * _latent_z1 + 0.8 * _latent_z2 + np.random.normal(0, 0.4, _n_samples)
    _f4 = 0.3 * _latent_z1 + 1.4 * _latent_z2 + np.random.normal(0, 0.4, _n_samples)
    _f5 = -0.5 * _latent_z1 - 1.1 * _latent_z2 + np.random.normal(0, 0.4, _n_samples)

    _X_5d = np.column_stack([_f1, _f2, _f3, _f4, _f5])
    _X_centered = _X_5d - np.mean(_X_5d, axis=0)

    _total_frobenius_norm = np.linalg.norm(_X_centered, ord="fro")

    _pca_5d = PCA().fit(_X_centered)
    _eval_records = []

    for _k in range(1, 6):
        _scores_k = _pca_5d.transform(_X_centered)[:, :_k]
        _components_k = _pca_5d.components_[:_k, :]
        _recon_k = _scores_k @ _components_k
        _residual_matrix = _X_centered - _recon_k
        _recon_error = np.linalg.norm(_residual_matrix, ord="fro")
        _pct_error = (_recon_error / _total_frobenius_norm) * 100.0
        _cum_var = np.sum(_pca_5d.explained_variance_ratio_[:_k]) * 100.0

        _eval_records.append({
            "Retained Components (K)": _k,
            "Cumulative Variance Explained": f"{_cum_var:.2f}%",
            "Frobenius Reconstruction Error": f"{_recon_error:.2f}",
            "Relative Error (%)": f"{_pct_error:.2f}%",
            "Subspace Recommendation": "Optimal Low-Rank Cutoff" if _k == 2 else ("Under-represented" if _k == 1 else "Diminishing Returns"),
        })

    _df_reconstruction = pd.DataFrame(_eval_records)

    return (
        mo.md("#### Low-Rank Reconstruction Diagnostics on 5D Telemetry Data"),
        mo.ui.table(_df_reconstruction),
    )


if __name__ == "__main__":
    app.run()
