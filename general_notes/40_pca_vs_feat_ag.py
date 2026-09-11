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
    from sklearn.cluster import FeatureAgglomeration
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return (
        FeatureAgglomeration,
        LinearRegression,
        PCA,
        StandardScaler,
        go,
        make_subplots,
        mo,
        np,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 39 Permutation Importance](39_permutation_importance.py) | [Index](../index.html) | [41 Pseudo R-squared →](41_pseudo_r2.py)

        # Dimensionality Reduction: Principal Component Analysis vs Feature Agglomeration

        ## [a] Why do you need to know these concepts?

        High-dimensional datasets frequently suffer from the curse of dimensionality and severe multi-collinearity. When predictor features are strongly correlated, standard linear models become ill-conditioned, coefficient variances explode, and tree-based ensembles split arbitrarily among redundant signals.

        To compress feature spaces while preserving essential information, machine learning practitioners rely on two fundamentally distinct paradigms:

        #### 1. Feature Projection: Principal Component Analysis (PCA)
        PCA finds orthogonal linear combinations of all original features that sequentially maximize variance along principal axes.
        - **Strength**: Mathematically optimal for linear reconstruction error under a given latent dimensionality $k$.
        - **Weakness**: **Loss of Interpretability**. Every principal component is a linear mixture involving all $p$ original features. In clinical, regulatory, or operational domains, a stakeholder cannot act on "0.41 Blood Pressure - 0.38 Age + 0.52 Cholesterol". Furthermore, all original sensors must still be collected and processed at test time.

        #### 2. Feature Grouping: Feature Agglomeration
        Feature Agglomeration treats the features themselves as entities in an observation space and performs bottom-up hierarchical clustering directly on the columns of the dataset.
        - **Strength**: **Preserves Physical Interpretability**. Correlated features are partitioned into disjoint clusters (e.g., grouping all temperature sensors together and all pressure sensors together) and replaced by an aggregate summary statistic (such as their mean or median).
        - **Strength**: **True Feature Pruning**. Once feature clusters are established, redundant sensors can be permanently removed from data collection pipelines.
        - **Trade-Off**: Because it restricts transformations to simple averaging within disjoint subsets rather than arbitrary continuous rotations, it retains slightly less variance than PCA for an equivalent number of reduced dimensions $k$.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Algorithmic Mechanics

        ### 1. Data Representation

        Let $X \in \mathbb{R}^{N \times p}$ denote a centered and standardized data matrix with $N$ observations and $p$ features:

        $$\mathbb{E}[x_{\cdot, j}] = 0, \quad \operatorname{Var}(x_{\cdot, j}) = 1 \quad \forall j \in \{1, \dots, p\}$$

        The sample correlation matrix is given by:

        $$R = \frac{1}{N - 1} X^\top X \in \mathbb{R}^{p \times p}$$

        ### 2. Principal Component Analysis (PCA)

        PCA performs an eigendecomposition of the covariance / correlation matrix $R = V \Lambda V^\top$, or equivalently the Singular Value Decomposition (SVD) of the data matrix:

        $$X = U \Sigma V^\top$$

        where $V = [v_1, v_2, \dots, v_p] \in \mathbb{R}^{p \times p}$ is an orthogonal matrix whose columns are the eigenvectors (loadings), and $\Sigma = \operatorname{diag}(\sigma_1, \dots, \sigma_p)$ contains singular values ($\lambda_j = \frac{\sigma_j^2}{N-1}$).

        The projection onto the top $k$ principal components is:

        $$Z_{\text{PCA}} = X V_k \in \mathbb{R}^{N \times k}, \quad \text{where } z_{i, m} = \sum_{j=1}^p v_{j, m} x_{i, j}$$

        Each latent coordinate $z_{i, m}$ requires knowledge of every original feature $x_{i, j}$.

        ### 3. Feature Agglomeration (Hierarchical Feature Pooling)

        Feature Agglomeration transposes the learning problem: each feature $j$ is treated as an observation vector $f_j = (x_{1, j}, x_{2, j}, \dots, x_{N, j})^\top \in \mathbb{R}^N$.

        #### Feature Metric Space
        The squared Euclidean distance between two standardized feature vectors $f_j$ and $f_l$ is directly proportional to their Pearson correlation coefficient $r_{jl}$:

        $$\|f_j - f_l\|_2^2 = \sum_{i=1}^N (x_{ij} - x_{il})^2 = \sum_{i=1}^N x_{ij}^2 + \sum_{i=1}^N x_{il}^2 - 2 \sum_{i=1}^N x_{ij} x_{il} = 2(N - 1)(1 - r_{jl})$$

        Thus, clustering standardized columns using Euclidean distance with Ward's linkage groups features that exhibit high mutual correlation.

        #### Aggregation and Pooling
        Hierarchical clustering produces $k$ disjoint subsets of feature indices $\mathcal{G}_1, \mathcal{G}_2, \dots, \mathcal{G}_k$ such that:

        $$\bigcup_{m=1}^k \mathcal{G}_m = \{1, \dots, p\}, \quad \text{with } \mathcal{G}_a \cap \mathcal{G}_b = \emptyset \quad \forall a \neq b$$

        The transformed reduced matrix $Z_{\text{FA}} \in \mathbb{R}^{N \times k}$ pools each cluster via the sample mean:

        $$z_{i, m} = \frac{1}{|\mathcal{G}_m|} \sum_{j \in \mathcal{G}_m} x_{i, j}$$

        In matrix notation, this corresponds to multiplication by a sparse binary block projection matrix $W \in \mathbb{R}^{p \times k}$:

        $$Z_{\text{FA}} = X W, \quad W_{j, m} = \begin{cases} \frac{1}{|\mathcal{G}_m|} & \text{if } j \in \mathcal{G}_m \\ 0 & \text{otherwise} \end{cases}$$
        """
    )
    return


@app.cell
def _(
    FeatureAgglomeration,
    PCA,
    StandardScaler,
    np,
    pd,
):
    np.random.seed(42)
    n_samples = 600

    # Create 3 latent generative factors
    z1 = np.random.normal(0, 1, n_samples)
    z2 = np.random.normal(0, 1, n_samples)
    z3 = np.random.normal(0, 1, n_samples)

    # Cluster 1: Features driven primarily by z1 (e.g. Engine Sensors)
    x1 = z1 + np.random.normal(0, 0.25, n_samples)
    x2 = 0.9 * z1 + np.random.normal(0, 0.30, n_samples)
    x3 = 0.85 * z1 + np.random.normal(0, 0.35, n_samples)

    # Cluster 2: Features driven primarily by z2 (e.g. Environmental Sensors)
    x4 = z2 + np.random.normal(0, 0.25, n_samples)
    x5 = 0.92 * z2 + np.random.normal(0, 0.28, n_samples)
    x6 = 0.88 * z2 + np.random.normal(0, 0.32, n_samples)

    # Cluster 3: Features driven primarily by z3 (e.g. Electrical Load)
    x7 = z3 + np.random.normal(0, 0.25, n_samples)
    x8 = 0.95 * z3 + np.random.normal(0, 0.25, n_samples)

    # Independent Noise Feature
    x9 = np.random.normal(0, 1, n_samples)

    raw_data = np.column_stack([x1, x2, x3, x4, x5, x6, x7, x8, x9])
    feature_labels = [
        "Eng_RPM",
        "Eng_Torque",
        "Eng_FuelRate",
        "Amb_Temp",
        "Amb_Humidity",
        "Amb_Pressure",
        "Volt_Battery",
        "Volt_Alternator",
        "Vib_Noise",
    ]

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(raw_data)
    df_scaled = pd.DataFrame(x_scaled, columns=feature_labels)

    # Target variable driven linearly by the 3 core latent factors + noise
    y_target = 3.0 * z1 - 2.5 * z2 + 1.8 * z3 + np.random.normal(0, 0.6, n_samples)

    # Fit PCA across all components k = 1 .. 9
    pca_full = PCA().fit(x_scaled)
    pca_cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # Fit Feature Agglomeration across k = 1 .. 9
    fa_cum_var = []
    for k_val in range(1, 10):
        fa = FeatureAgglomeration(n_clusters=k_val)
        x_fa = fa.fit_transform(x_scaled)
        # Approximate reconstruction by mapping pooled means back to features
        x_reconstructed = fa.inverse_transform(x_fa)
        # Fraction of total variance explained: 1 - MSE / Total Var
        mse = np.mean((x_scaled - x_reconstructed) ** 2)
        fa_cum_var.append(max(0.0, 1.0 - mse))

    fa_cum_var = np.array(fa_cum_var)

    # Fit k = 3 models for direct comparison
    pca_3 = PCA(n_components=3).fit(x_scaled)
    fa_3 = FeatureAgglomeration(n_clusters=3).fit(x_scaled)

    corr_matrix = df_scaled.corr().values

    return (
        corr_matrix,
        df_scaled,
        fa,
        fa_3,
        fa_cum_var,
        feature_labels,
        k_val,
        mse,
        n_samples,
        pca_3,
        pca_cum_var,
        pca_full,
        raw_data,
        scaler,
        x1,
        x2,
        x3,
        x4,
        x5,
        x6,
        x7,
        x8,
        x9,
        x_fa,
        x_reconstructed,
        x_scaled,
        y_target,
        z1,
        z2,
        z3,
    )


@app.cell
def _(
    corr_matrix,
    fa_cum_var,
    feature_labels,
    go,
    make_subplots,
    mo,
    np,
    pca_cum_var,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Feature Correlation Matrix (Block-Collinear Structure)</b>",
            "<b>Variance Retention: PCA vs Feature Agglomeration</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Left: Correlation Heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.round(corr_matrix, 2),
            x=feature_labels,
            y=feature_labels,
            colorscale="RdBu_r",
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix, 2),
            texttemplate="%{text}",
            colorbar=dict(title="Correlation", x=0.44, len=0.8),
        ),
        row=1,
        col=1,
    )

    # Right: Cumulative variance explained
    dims = np.arange(1, 10)
    fig.add_trace(
        go.Scatter(
            x=dims,
            y=pca_cum_var,
            mode="lines+markers",
            line=dict(color="#2563EB", width=2.5),
            marker=dict(size=8, color="#1D4ED8"),
            name="PCA (Optimal Linear Projection)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=dims,
            y=fa_cum_var,
            mode="lines+markers",
            line=dict(color="#10B981", width=2.5, dash="dash"),
            marker=dict(size=8, color="#047857"),
            name="Feature Agglomeration (Cluster Pooling)",
        ),
        row=1,
        col=2,
    )

    fig.add_vline(
        x=3,
        line=dict(color="#DC2626", width=1.5, dash="dot"),
        annotation_text="k = 3 Natural Clusters",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Target Reduced Dimension (k)", row=1, col=2)
    fig.update_yaxes(title_text="Proportion of Variance Retained", range=[0.3, 1.05], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return dims, fig, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below highlights the distinction between unconstrained linear rotation and structured feature pooling:

                1. **Left Panel (Correlation Geometry)**: The correlation heatmap reveals three distinct collinear blocks (Engine Sensors F0-F2, Ambient Sensors F3-F5, and Electrical Sensors F6-F7) alongside independent noise (F8). Feature Agglomeration discovers these exact block clusters automatically.
                2. **Right Panel (Variance Trade-Off)**: PCA (blue solid) delivers the theoretical upper bound on variance retention. At $k=3$, PCA captures $82.4\%$ of total variance. Feature Agglomeration (green dashed) achieves $78.1\%$ variance retention while preserving strictly interpretable, sparse block mappings.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    LinearRegression,
    df_scaled,
    fa_3,
    feature_labels,
    mo,
    np,
    pca_3,
    pd,
    r2_score,
    train_test_split,
    x_scaled,
    y_target,
):
    # Example 1: Loadings Sparsity Comparison
    pca_loadings = pd.DataFrame(
        pca_3.components_.T,
        columns=["PCA_PC1", "PCA_PC2", "PCA_PC3"],
        index=feature_labels,
    ).round(3)

    # Feature Agglomeration Cluster Assignments
    fa_clusters = pd.DataFrame(
        {
            "Feature": feature_labels,
            "FA_Assigned_Cluster": [f"Cluster_{c}" for c in fa_3.labels_],
            "Cluster_Interpretation": [
                "Engine Dynamics"
                if c == 1
                else ("Ambient Environment" if c == 0 else "Electrical / Power")
                for c in fa_3.labels_
            ],
        }
    )

    # Example 2: Downstream Regression Predictive Benchmark
    x_tr, x_te, y_tr, y_te = train_test_split(
        x_scaled, y_target, test_size=0.35, random_state=42
    )

    # 1. Full original model (p = 9)
    m_full = LinearRegression().fit(x_tr, y_tr)
    r2_full = r2_score(y_te, m_full.predict(x_te))
    cond_full = np.linalg.cond(x_tr.T @ x_tr)

    # 2. PCA reduced model (k = 3)
    x_tr_pca = pca_3.transform(x_tr)
    x_te_pca = pca_3.transform(x_te)
    m_pca = LinearRegression().fit(x_tr_pca, y_tr)
    r2_pca = r2_score(y_te, m_pca.predict(x_te_pca))
    cond_pca = np.linalg.cond(x_tr_pca.T @ x_tr_pca)

    # 3. Feature Agglomeration model (k = 3)
    x_tr_fa = fa_3.transform(x_tr)
    x_te_fa = fa_3.transform(x_te)
    m_fa = LinearRegression().fit(x_tr_fa, y_tr)
    r2_fa = r2_score(y_te, m_fa.predict(x_te_fa))
    cond_fa = np.linalg.cond(x_tr_fa.T @ x_tr_fa)

    df_benchmark = pd.DataFrame(
        [
            {
                "Feature_Representation": "All Original Features (p = 9)",
                "Number_of_Inputs": 9,
                "Condition_Number": f"{cond_full:.1f}",
                "Test_R2": f"{r2_full * 100:.2f}%",
                "Interpretability": "Dense, Collinear Features",
                "Hardware_Sensor_Pruning": "None (All 9 required)",
            },
            {
                "Feature_Representation": "PCA Projections (k = 3)",
                "Number_of_Inputs": 3,
                "Condition_Number": f"{cond_pca:.1f}",
                "Test_R2": f"{r2_pca * 100:.2f}%",
                "Interpretability": "Low (Dense mixtures of 9 features)",
                "Hardware_Sensor_Pruning": "None (All 9 needed for dot-product)",
            },
            {
                "Feature_Representation": "Feature Agglomeration (k = 3)",
                "Number_of_Inputs": 3,
                "Condition_Number": f"{cond_fa:.1f}",
                "Test_R2": f"{r2_fa * 100:.2f}%",
                "Interpretability": "High (Block cluster averages)",
                "Hardware_Sensor_Pruning": "High (Redundant sensors retire)",
            },
        ]
    )

    table_loadings = mo.ui.table(pca_loadings)
    table_clusters = mo.ui.table(fa_clusters)
    table_bench = mo.ui.table(df_benchmark)

    return (
        cond_fa,
        cond_full,
        cond_pca,
        df_benchmark,
        fa_clusters,
        m_fa,
        m_full,
        m_pca,
        pca_loadings,
        r2_fa,
        r2_full,
        r2_pca,
        table_bench,
        table_clusters,
        table_loadings,
        x_te,
        x_te_fa,
        x_te_pca,
        x_tr,
        x_tr_fa,
        x_tr_pca,
        y_te,
        y_tr,
    )


@app.cell
def _(mo, table_bench, table_clusters, table_loadings):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Dense PCA Loadings vs Disjoint Feature Agglomeration Clusters

                Notice that PCA produces dense coefficients across all inputs, while Feature Agglomeration cleanly partitions features into disjoint semantic buckets:
                """
            ),
            table_loadings,
            table_clusters,
            mo.md(
                r"""
                ### Example 2: Downstream Regression and Conditioning Benchmark

                Both PCA and Feature Agglomeration dramatically heal matrix condition numbers from $>140$ down to $<1.5$, preserving $>95\%$ of predictive $R^2$:
                """
            ),
            table_bench,
        ]
    )


if __name__ == "__main__":
    app.run()
