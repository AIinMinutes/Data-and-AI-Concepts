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
    from scipy import stats
    from sklearn.cross_decomposition import CCA

    return CCA, go, make_subplots, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 29: Canonical Correlation Analysis, Multimodal Alignment, and Generalized Eigendecompositions

    &larr; Previous Note: [28 Factor Analysis](28_factor_analysis.py) | Next Note: [30 Correspondence Analysis](30_correspondence_analysis.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In modern data science and deep learning, datasets often consist of multiple distinct views or modalities capturing the same underlying phenomenon (e.g. text descriptions paired with images in CLIP, audio signals paired with video frames, patient genomics paired with clinical outcomes).

    While standard multiple regression models a single scalar target from multiple predictors, and PCA finds directions of maximum variance within a single dataset, **Canonical Correlation Analysis (CCA)** solves the symmetric multiview alignment problem:
    1. **Maximizing Cross-Modal Linear Association**: Given two multivariate sets $\mathbf{X} \in \mathbb{R}^{n \times p}$ and $\mathbf{Y} \in \mathbb{R}^{n \times q}$, CCA seeks linear combinations $\mathbf{u}_1 = \mathbf{X}\mathbf{a}_1$ and $\mathbf{v}_1 = \mathbf{Y}\mathbf{b}_1$ that achieve maximum possible Pearson correlation $\rho_1 = \operatorname{corr}(\mathbf{u}_1, \mathbf{v}_1)$.
    2. **Invariance to Affine Coordinate Transformations**: Unlike PCA (which is sensitive to coordinate scaling and rotations), canonical correlations are invariant to any non-singular linear transformations of either $\mathbf{X}$ or $\mathbf{Y}$ ($\mathbf{X} \to \mathbf{X}\mathbf{M}_1, \mathbf{Y} \to \mathbf{Y}\mathbf{M}_2$).
    3. **Grand Unification of Classical Multivariate Statistics**: Harold Hotelling (1936) demonstrated that ordinary linear regression, multivariate regression, MANOVA, and Linear Discriminant Analysis (LDA) are all special cases of CCA.
    4. **Modern Self-Supervised and Contrastive Learning**: Algorithms like Deep CCA (DCCA), Barlow Twins, and cross-modal embedding projectors directly implement generalized canonical correlation objectives to align latent representations across neural network branches.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. The Optimization Problem

    Let $\mathbf{X} \in \mathbb{R}^{n \times p}$ and $\mathbf{Y} \in \mathbb{R}^{n \times q}$ be mean-centered observation matrices. The sample covariance blocks are:

    $$
    \boldsymbol{\Sigma}_{XX} = \frac{1}{n - 1} \mathbf{X}^\top \mathbf{X}, \quad \boldsymbol{\Sigma}_{YY} = \frac{1}{n - 1} \mathbf{Y}^\top \mathbf{Y}, \quad \boldsymbol{\Sigma}_{XY} = \frac{1}{n - 1} \mathbf{X}^\top \mathbf{Y}
    $$

    We seek projection vectors $\mathbf{a} \in \mathbb{R}^p$ and $\mathbf{b} \in \mathbb{R}^q$ to form canonical variates $u = \mathbf{X}\mathbf{a}$ and $v = \mathbf{Y}\mathbf{b}$ that maximize the correlation:

    $$
    \rho = \operatorname{corr}(u, v) = \frac{\mathbf{a}^\top \boldsymbol{\Sigma}_{XY} \mathbf{b}}{\sqrt{\mathbf{a}^\top \boldsymbol{\Sigma}_{XX} \mathbf{a}} \sqrt{\mathbf{b}^\top \boldsymbol{\Sigma}_{YY} \mathbf{b}}}
    $$

    Since correlation is scale-invariant, we fix the variances to unity:

    $$
    \max_{\mathbf{a}, \mathbf{b}} \mathbf{a}^\top \boldsymbol{\Sigma}_{XY} \mathbf{b} \quad \text{subject to} \quad \mathbf{a}^\top \boldsymbol{\Sigma}_{XX} \mathbf{a} = 1, \quad \mathbf{b}^\top \boldsymbol{\Sigma}_{YY} \mathbf{b} = 1
    $$

    ---

    ### 2. Lagrangian Formulation and the Generalized Eigenvalue Problem

    Formulating the Lagrangian with multipliers $\frac{\lambda_1}{2}$ and $\frac{\lambda_2}{2}$:

    $$
    \mathcal{L}(\mathbf{a}, \mathbf{b}, \lambda_1, \lambda_2) = \mathbf{a}^\top \boldsymbol{\Sigma}_{XY} \mathbf{b} - \frac{\lambda_1}{2}(\mathbf{a}^\top \boldsymbol{\Sigma}_{XX} \mathbf{a} - 1) - \frac{\lambda_2}{2}(\mathbf{b}^\top \boldsymbol{\Sigma}_{YY} \mathbf{b} - 1)
    $$

    Computing partial derivatives and setting them to zero:

    $$
    \nabla_{\mathbf{a}} \mathcal{L} = \boldsymbol{\Sigma}_{XY} \mathbf{b} - \lambda_1 \boldsymbol{\Sigma}_{XX} \mathbf{a} = \mathbf{0} \implies \boldsymbol{\Sigma}_{XY} \mathbf{b} = \lambda_1 \boldsymbol{\Sigma}_{XX} \mathbf{a}
    $$

    $$
    \nabla_{\mathbf{b}} \mathcal{L} = \boldsymbol{\Sigma}_{YX} \mathbf{a} - \lambda_2 \boldsymbol{\Sigma}_{YY} \mathbf{b} = \mathbf{0} \implies \boldsymbol{\Sigma}_{YX} \mathbf{a} = \lambda_2 \boldsymbol{\Sigma}_{YY} \mathbf{b}
    $$

    Multiplying the first equation by $\mathbf{a}^\top$ and the second by $\mathbf{b}^\top$:

    $$
    \lambda_1 = \mathbf{a}^\top \boldsymbol{\Sigma}_{XY} \mathbf{b} = \lambda_2 = \rho
    $$

    Solving for $\mathbf{b} = \frac{1}{\rho} \boldsymbol{\Sigma}_{YY}^{-1} \boldsymbol{\Sigma}_{YX} \mathbf{a}$ and substituting into the first equation yields the decoupled generalized eigenvalue problem for $\mathbf{a}$:

    $$
    \left(\boldsymbol{\Sigma}_{XX}^{-1} \boldsymbol{\Sigma}_{XY} \boldsymbol{\Sigma}_{YY}^{-1} \boldsymbol{\Sigma}_{YX}\right) \mathbf{a} = \rho^2 \mathbf{a}
    $$

    Symmetrically, solving for $\mathbf{b}$:

    $$
    \left(\boldsymbol{\Sigma}_{YY}^{-1} \boldsymbol{\Sigma}_{YX} \boldsymbol{\Sigma}_{XX}^{-1} \boldsymbol{\Sigma}_{XY}\right) \mathbf{b} = \rho^2 \mathbf{b}
    $$

    The squared canonical correlations $\rho_1^2 \geq \rho_2^2 \geq \dots \geq \rho_m^2$ ($m = \min(p, q)$) are the eigenvalues of these operator matrices.

    ---

    ### 3. SVD of the Normalized Coherence Matrix

    In practice, to ensure numerical stability and avoid explicit inversion of covariance matrices, CCA is solved via the Singular Value Decomposition of the **coherence matrix** $\mathbf{K}$:

    $$
    \mathbf{K} = \boldsymbol{\Sigma}_{XX}^{-1/2} \boldsymbol{\Sigma}_{XY} \boldsymbol{\Sigma}_{YY}^{-1/2}
    $$

    Computing the thin SVD:

    $$
    \mathbf{K} = \mathbf{U} \mathbf{D} \mathbf{V}^\top
    $$

    The canonical correlations are the diagonal entries $\rho_k = D_{kk}$, and the canonical weight vectors are:

    $$
    \mathbf{a}_k = \boldsymbol{\Sigma}_{XX}^{-1/2} \mathbf{u}_k, \quad \mathbf{b}_k = \boldsymbol{\Sigma}_{YY}^{-1/2} \mathbf{v}_k
    $$

    ---

    ### 4. Hypothesis Testing: Wilks' Lambda ($\Lambda^*$)

    To test the global null hypothesis that $\mathbf{X}$ and $\mathbf{Y}$ are completely independent ($H_0: \boldsymbol{\Sigma}_{XY} = \mathbf{0} \iff \rho_1 = \rho_2 = \dots = \rho_m = 0$), **Wilks' Lambda** computes the product of unexplained variances:

    $$
    \Lambda^* = \prod_{k=1}^m \left(1 - \rho_k^2\right)
    $$

    Bartlett's asymptotic Chi-Square approximation is:

    $$
    \chi^2 = -\left(n - 1 - \frac{p + q + 1}{2}\right) \ln \Lambda^* \sim \chi^2(p \cdot q)
    $$

    If $p\text{-value} < 0.05$, we reject $H_0$ and conclude that significant cross-modal linear association exists.
    """)
    return


@app.cell
def _(np, pd):
    # Simulation Data: Multimodal Biometric & Physical Performance Dataset
    # Set X (Physiological Variables): Weight, Waist, Pulse (p = 3)
    # Set Y (Athletic Performance): Pullups, Situps, Jumps (q = 3)
    np.random.seed(42)
    _n = 150

    # Common latent fitness factor + latent body scale factor
    _latent_fitness = np.random.normal(0.0, 1.0, _n)
    _latent_mass = np.random.normal(0.0, 1.0, _n)

    # Set X: Physical Attributes
    _weight = 75.0 + 10.0 * _latent_mass - 2.0 * _latent_fitness + np.random.normal(0.0, 2.0, _n)
    _waist = 85.0 + 8.0 * _latent_mass - 3.0 * _latent_fitness + np.random.normal(0.0, 1.5, _n)
    _pulse = 70.0 + 2.0 * _latent_mass - 6.0 * _latent_fitness + np.random.normal(0.0, 3.0, _n)

    # Set Y: Athletic Tests
    _pullups = 10.0 - 3.0 * _latent_mass + 5.0 * _latent_fitness + np.random.normal(0.0, 1.5, _n)
    _situps = 180.0 - 15.0 * _latent_mass + 25.0 * _latent_fitness + np.random.normal(0.0, 8.0, _n)
    _jumps = 50.0 - 5.0 * _latent_mass + 10.0 * _latent_fitness + np.random.normal(0.0, 3.0, _n)

    df_X = pd.DataFrame({"Weight": _weight, "Waist": _waist, "Pulse": _pulse})
    df_Y = pd.DataFrame({"Pullups": _pullups, "Situps": _situps, "Jumps": _jumps})

    return df_X, df_Y


@app.cell
def _(CCA, df_X, df_Y, go, make_subplots, mo, np):
    # Interactive Visualizations Cell:
    # Subplot 1: Scatter plot of 1st Canonical Variates (u1 vs v1) with correlation fit
    # Subplot 2: Canonical Correlation Spectrum (rho_1, rho_2, rho_3)
    # Subplot 3: Canonical Loadings Bar Chart for Set X and Set Y on Variate 1

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Alignment: Variate u1 vs. Variate v1",
            "2. Canonical Correlation Spectrum",
            "3. Feature Loadings on 1st Canonical Variate",
        ),
        horizontal_spacing=0.09,
    )

    # Fit Scikit-Learn CCA
    _cca = CCA(n_components=3, scale=True)
    _cca.fit(df_X, df_Y)
    _u, _v = _cca.transform(df_X, df_Y)

    # Compute exact canonical correlation coefficients
    _corrs = [np.corrcoef(_u[:, i], _v[:, i])[0, 1] for i in range(3)]

    # Subplot 1: Variates Scatter
    _fig.add_trace(
        go.Scatter(
            x=_u[:, 0],
            y=_v[:, 0],
            mode="markers",
            marker=dict(size=6, color="#6366f1", opacity=0.75),
            name="Aligned Pairs",
            hovertemplate="u1 (Physio): %{x:.2f}<br>v1 (Athletic): %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trend line
    _line_x = np.linspace(_u[:, 0].min(), _u[:, 0].max(), 50)
    _slope, _inter = np.polyfit(_u[:, 0], _v[:, 0], 1)
    _fig.add_trace(
        go.Scatter(
            x=_line_x,
            y=_slope * _line_x + _inter,
            mode="lines",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            name=f"Fit (rho_1 = {_corrs[0]:.3f})",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Spectrum Bar Chart
    _pair_names = ["Pair 1", "Pair 2", "Pair 3"]
    _fig.add_trace(
        go.Bar(
            x=_pair_names,
            y=_corrs,
            marker_color=["#10b981", "#3b82f6", "#94a3b8"],
            name="Canonical Correlation",
            text=[f"{c:.3f}" for c in _corrs],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Structural Loadings on Variate 1
    # Correlation between original variables and first canonical variates
    _loadings_x = [np.corrcoef(df_X[col], _u[:, 0])[0, 1] for col in df_X.columns]
    _loadings_y = [np.corrcoef(df_Y[col], _v[:, 0])[0, 1] for col in df_Y.columns]
    _feat_names = list(df_X.columns) + list(df_Y.columns)
    _all_loadings = _loadings_x + _loadings_y
    _colors = ["#3b82f6"] * 3 + ["#10b981"] * 3

    _fig.add_trace(
        go.Bar(
            x=_feat_names,
            y=_all_loadings,
            marker_color=_colors,
            name="Canonical Loadings",
            text=[f"{l:.2f}" for l in _all_loadings],
            textposition="auto",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Canonical Correlation Analysis: Cross-Modal Alignment & Structural Loadings",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Canonical Variate u1 (Physiology)", row=1, col=1)
    _fig.update_yaxes(title_text="Canonical Variate v1 (Athletics)", row=1, col=1)

    _fig.update_xaxes(title_text="Canonical Variate Pair", row=1, col=2)
    _fig.update_yaxes(title_text="Canonical Correlation (rho)", range=[0, 1.05], row=1, col=2)

    _fig.update_xaxes(title_text="Original Indicator Feature", tickangle=45, row=1, col=3)
    _fig.update_yaxes(title_text="Correlation with Variate 1", range=[-1.05, 1.05], row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Full Step-by-Step CCA from Scratch via SVD of the Coherence Matrix**: Computing covariance blocks $\boldsymbol{\Sigma}_{XX}, \boldsymbol{\Sigma}_{YY}, \boldsymbol{\Sigma}_{XY}$, matrix square roots via eigendecomposition, SVD of $\mathbf{K}$, and extracting canonical weights $\mathbf{a}_1, \mathbf{b}_1$ and correlation $\rho_1$, verified against Scikit-Learn.
    2. **Multivariate Independence Hypothesis Test (Wilks' Lambda)**: Computing Wilks' $\Lambda^*$, degrees of freedom $p \cdot q$, Bartlett's $\chi^2$-statistic, and two-tailed $p$-value to verify statistical significance of cross-modal coupling.
    """)
    return


@app.cell
def _(CCA, df_X, df_Y, mo, np, pd):
    # Example 1: Pure NumPy Step-by-Step CCA from Scratch
    _n = len(df_X)
    _X = df_X.to_numpy()
    _Y = df_Y.to_numpy()

    # Center variables
    _Xc = _X - np.mean(_X, axis=0)
    _Yc = _Y - np.mean(_Y, axis=0)

    # Covariance blocks
    _Sxx = (_Xc.T @ _Xc) / (_n - 1)
    _Syy = (_Yc.T @ _Yc) / (_n - 1)
    _Sxy = (_Xc.T @ _Yc) / (_n - 1)

    # Matrix square roots via symmetric eigendecomposition
    def _inv_sqrt_matrix(M):
        _evals, _evecs = np.linalg.eigh(M)
        _inv_sqrt_d = np.diag(1.0 / np.sqrt(np.maximum(_evals, 1e-12)))
        return _evecs @ _inv_sqrt_d @ _evecs.T

    _Sxx_inv_sqrt = _inv_sqrt_matrix(_Sxx)
    _Syy_inv_sqrt = _inv_sqrt_matrix(_Syy)

    # Normalized coherence matrix K
    _K = _Sxx_inv_sqrt @ _Sxy @ _Syy_inv_sqrt
    _U, _D, _Vt = np.linalg.svd(_K)

    # Canonical correlations from scratch
    _rho_scratch = _D

    # Canonical weights
    _A_weights = _Sxx_inv_sqrt @ _U
    _B_weights = _Syy_inv_sqrt @ _Vt.T

    # Scikit-Learn Reference
    _cca = CCA(n_components=3, scale=False)
    _cca.fit(_Xc, _Yc)
    _u_sk, _v_sk = _cca.transform(_Xc, _Yc)
    _rho_sklearn = [np.corrcoef(_u_sk[:, i], _v_sk[:, i])[0, 1] for i in range(3)]

    _comparison_rows = []
    for _i in range(len(_rho_scratch)):
        _comparison_rows.append({
            "Canonical Pair": f"Pair {_i + 1}",
            "Scratch SVD Correlation rho": f"{_rho_scratch[_i]:.6f}",
            "Scikit-Learn Correlation rho": f"{_rho_sklearn[_i]:.6f}",
            "Discrepancy": f"{np.abs(_rho_scratch[_i] - _rho_sklearn[_i]):.2e}",
            "Interpretation": "Primary Shared Alignment Axis" if _i == 0 else "Orthogonal Secondary Mode",
        })

    _df_cca_benchmark = pd.DataFrame(_comparison_rows)

    return (
        mo.md("#### Step-by-Step Coherence SVD vs. Scikit-Learn Benchmark"),
        mo.ui.table(_df_cca_benchmark),
    )


@app.cell
def _(df_X, df_Y, mo, np, pd, stats):
    # Example 2: Wilks' Lambda Multivariate Independence Test
    _n = len(df_X)
    _p = df_X.shape[1]
    _q = df_Y.shape[1]
    _m = min(_p, _q)

    # Compute canonical correlations
    _Xc = df_X.to_numpy() - np.mean(df_X.to_numpy(), axis=0)
    _Yc = df_Y.to_numpy() - np.mean(df_Y.to_numpy(), axis=0)

    _Sxx = (_Xc.T @ _Xc) / (_n - 1)
    _Syy = (_Yc.T @ _Yc) / (_n - 1)
    _Sxy = (_Xc.T @ _Yc) / (_n - 1)

    def _inv_sqrt(M):
        _evals, _evecs = np.linalg.eigh(M)
        return _evecs @ np.diag(1.0 / np.sqrt(np.maximum(_evals, 1e-12))) @ _evecs.T

    _K = _inv_sqrt(_Sxx) @ _Sxy @ _inv_sqrt(_Syy)
    _, _rhos, _ = np.linalg.svd(_K)

    # Wilks' Lambda
    _wilks_lambda = np.prod(1.0 - _rhos**2)

    # Bartlett's Chi-Square approximation
    _bartlett_factor = _n - 1.0 - (_p + _q + 1.0) / 2.0
    _chi2_stat = -_bartlett_factor * np.log(_wilks_lambda)
    _df_chi2 = _p * _q
    _p_value = 1.0 - stats.chi2.cdf(_chi2_stat, df=_df_chi2)

    _df_wilks = pd.DataFrame(
        [
            {"Statistical Metric": "Sample Size (n)", "Value": str(_n), "Description": "Observation count"},
            {"Statistical Metric": "Dimensions (p, q)", "Value": f"p = {_p}, q = {_q}", "Description": "Feature set cardinalities"},
            {"Statistical Metric": "Wilks' Lambda (Lambda*)", "Value": f"{_wilks_lambda:.6f}", "Description": "Product of unexplained variance ratios"},
            {"Statistical Metric": "Bartlett's Chi-Square Statistic", "Value": f"{_chi2_stat:.2f}", "Description": "Log-likelihood ratio test statistic"},
            {"Statistical Metric": "Degrees of Freedom (p * q)", "Value": str(_df_chi2), "Description": "Joint dimensionality df"},
            {"Statistical Metric": "Independence p-Value", "Value": f"{_p_value:.4e}", "Description": "Reject H0: strong cross-modal dependence"},
        ]
    )

    return (
        mo.md("#### Wilks' Lambda Test for Cross-Modal Independence"),
        mo.ui.table(_df_wilks),
    )


if __name__ == "__main__":
    app.run()
