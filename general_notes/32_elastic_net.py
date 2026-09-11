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
    from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split

    return (
        ElasticNet,
        Lasso,
        LinearRegression,
        Ridge,
        go,
        make_subplots,
        mean_squared_error,
        mo,
        np,
        pd,
        r2_score,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 32: Elastic Net Regularization, Grouping Effects, and Convex Penalty Geometry

    &larr; Previous Note: [31 Gaussian Mixture Models](31_gaussian_mixture_models.py) | Next Note: [33 Huber Loss](33_huber_loss.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Ordinary Least Squares (OLS) regression fails in high-dimensional or multicollinear regimes because the matrix $\mathbf{X}^\top \mathbf{X}$ becomes ill-conditioned or singular. While Ridge ($L_2$) and Lasso ($L_1$) provide classical regularized alternatives, each carries significant individual limitations:
    1. **The Limitations of Pure Lasso ($L_1$)**:
       - **Arbitrary Selection on Collinear Features**: When two or more features are strongly correlated ($r \approx 0.99$), Lasso arbitrarily selects one feature and zeroes out the others, leading to erratic feature attribution that changes with small data resamplings.
       - **Dimensionality Saturation**: If the number of features exceeds observations ($p > n$), Lasso can select at most $n$ non-zero predictors before saturating.
    2. **The Limitations of Pure Ridge ($L_2$)**:
       - Ridge shrinks correlated coefficients together smoothly, but because the $L_2$ penalty is strictly differentiable everywhere, it cannot set any coefficient to exactly zero, failing to produce sparse, interpretable models.
    3. **Zou & Hastie's Elastic Net Synthesis (2005)**:
       Elastic Net combines the $L_1$ and $L_2$ penalties into a convex compromise:

       $$
       \mathcal{L}(\boldsymbol{\beta}) = \frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \left[\rho \|\boldsymbol{\beta}\|_1 + \frac{1 - \rho}{2}\|\boldsymbol{\beta}\|_2^2\right]
       $$

       where $\rho \in [0, 1]$ is the `l1_ratio`.
    4. **The Grouping Property**: The strict convexity of the $L_2$ penalty ensures that strongly correlated features enter or leave the model together with similar coefficients, overcoming Lasso's arbitrary selection instability while maintaining exact sparsity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Mathematical Formulation

    Given standardized design matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ and response vector $\mathbf{y} \in \mathbb{R}^n$, the Elastic Net optimization problem is:

    $$
    \min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda_1 \|\boldsymbol{\beta}\|_1 + \frac{\lambda_2}{2} \|\boldsymbol{\beta}\|_2^2
    $$

    In Scikit-Learn parameterization, letting $\alpha = \lambda_1 + \lambda_2$ and $\rho = \frac{\lambda_1}{\lambda_1 + \lambda_2} \in [0, 1]$:

    $$
    \min_{\boldsymbol{\beta}} \frac{1}{2n} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha \rho \|\boldsymbol{\beta}\|_1 + \frac{\alpha(1 - \rho)}{2} \|\boldsymbol{\beta}\|_2^2
    $$

    - $\rho = 1$: Reduces identically to Lasso ($L_1$).
    - $\rho = 0$: Reduces identically to Ridge ($L_2$).
    - $0 < \rho < 1$: Elastic Net with balanced sparsity and grouping.

    ---

    ### 2. Coordinate Descent and the Soft-Thresholding Operator

    Elastic Net is optimized via cyclical coordinate descent. For each coordinate $j \in \{1, \dots, p\}$, keeping all other coefficients $\boldsymbol{\beta}_{\setminus j}$ fixed, we compute the partial residual:

    $$
    \mathbf{r}_{-j} = \mathbf{y} - \sum_{k \neq j} \mathbf{x}_k \beta_k
    $$

    The univariate subgradient optimality condition yields a closed-form two-stage update:

    $$
    \beta_j^{(t+1)} = \frac{\mathcal{S}\left(\frac{1}{n} \mathbf{x}_j^\top \mathbf{r}_{-j}, \, \alpha \rho\right)}{1 + \alpha (1 - \rho)}
    $$

    where $\mathcal{S}(z, \gamma)$ is the **Soft-Thresholding Operator**:

    $$
    \mathcal{S}(z, \gamma) = \operatorname{sign}(z) \max(0, \, |z| - \gamma)
    $$

    #### Two-Stage Mechanism:
    1. **Sparsity via Soft-Thresholding**: The numerator sets $\beta_j = 0$ if the correlation with the residual $|z| \leq \alpha \rho$.
    2. **Grouping via Ridge Shrinkage**: The denominator divides by $1 + \alpha(1 - \rho) > 1$, scaling down non-zero coefficients to prevent variance inflation under collinearity.

    ---

    ### 3. The Grouping Effect Theorem

    Zou & Hastie (2005) proved a fundamental theorem governing correlated features in Elastic Net:

    > **Theorem (Grouping Bound)**: Assume $\mathbf{x}_i$ and $\mathbf{x}_j$ are standardized ($\|\mathbf{x}_i\|_2^2 = \|\mathbf{x}_j\|_2^2 = n$), with sample correlation $r = \frac{1}{n} \mathbf{x}_i^\top \mathbf{x}_j$. If $\hat{\beta}_i \cdot \hat{\beta}_j > 0$, then:
    >
    > $$
    > |\hat{\beta}_i - \hat{\beta}_j| \leq \frac{\|\mathbf{y}\|_2}{\lambda_2} \sqrt{2(1 - r)}
    > $$

    As the correlation between features approaches unity ($r \to 1$), $\sqrt{2(1 - r)} \to 0$, forcing:

    $$
    \lim_{r \to 1} |\hat{\beta}_i - \hat{\beta}_j| = 0
    $$

    Under pure Lasso ($\lambda_2 = 0$), the bound denominator is zero, allowing $|\hat{\beta}_i - \hat{\beta}_j|$ to remain arbitrarily large. Elastic Net guarantees that collinear features receive virtually identical regression coefficients.

    ---

    ### 4. Constraint Geometry Comparison

    In the Lagrangian dual formulation, minimizing the squared loss subject to a norm constraint $\Omega(\boldsymbol{\beta}) \leq C$:

    | Penalty Type | Unit Ball Geometry $\Omega(\boldsymbol{\beta}) \leq 1$ | Singularities (Corners on Axes)? | Curvature along Edges? | Resulting Behavior |
    | :--- | :--- | :--- | :--- | :--- |
    | **Ridge ($L_2$)** | Circle / Sphere | No (smooth everywhere) | Strictly convex | Grouping, no feature selection |
    | **Lasso ($L_1$)** | Cross-Polytope / Diamond | Yes (sharp vertices on axes) | None (flat hyperplanes) | Sparse selection, erratic under collinearity |
    | **Elastic Net** | Curved Diamond | Yes (sharp vertices on axes) | Strictly convex edges | Both sparsity and robust grouping |
    """)
    return


@app.cell
def _(np, pd):
    # Data Generation: Controlled Multicollinearity Experiment
    # 500 samples, 6 features:
    # x1, x2: Strongly collinear pair (r ~ 0.95)
    # x3, x4: Moderately collinear pair (r ~ 0.80)
    # x5, x6: Pure noise variables (uncorrelated)
    np.random.seed(47)
    _n = 400

    _latent_1 = np.random.normal(0.0, 1.0, _n)
    _x1 = _latent_1 + np.random.normal(0.0, 0.20, _n)
    _x2 = _latent_1 + np.random.normal(0.0, 0.20, _n)

    _latent_2 = np.random.normal(0.0, 1.0, _n)
    _x3 = _latent_2 + np.random.normal(0.0, 0.40, _n)
    _x4 = _latent_2 + np.random.normal(0.0, 0.40, _n)

    _x5 = np.random.normal(0.0, 1.0, _n)
    _x6 = np.random.normal(0.0, 1.0, _n)

    # True model: both x1 and x2 have equal positive causal effect (beta1 = beta2 = 3.0)
    # x3 has moderate effect (beta3 = 2.0), x4, x5, x6 have zero effect
    _y = 3.0 * _x1 + 3.0 * _x2 + 2.0 * _x3 + np.random.normal(0.0, 1.5, _n)

    df_elastic = pd.DataFrame(
        {
            "x1_collinear_A": _x1,
            "x2_collinear_B": _x2,
            "x3_signal": _x3,
            "x4_correlated_noise": _x4,
            "x5_pure_noise": _x5,
            "x6_pure_noise": _x6,
            "y": _y,
        }
    )

    return (df_elastic,)


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    df_elastic,
    go,
    make_subplots,
    mo,
    np,
    train_test_split,
):
    # Interactive Visualizations Cell:
    # Subplot 1: 2D Geometry of Unit Balls (L1 vs L2 vs Elastic Net) with Tangent Loss Contours
    # Subplot 2: Grouping Effect: Lasso vs. Elastic Net Coefficient Trajectories as Alpha Increases
    # Subplot 3: Model Coefficient Profiles across OLS, Ridge, Lasso, and Elastic Net

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Constraint Balls: L1, L2, and Elastic Net",
            "2. Grouping Effect: Collinear x1 & x2 Paths",
            "3. Estimated Coefficients Across Regularizers",
        ),
        horizontal_spacing=0.09,
    )

    # Subplot 1: Geometric Unit Balls
    _theta = np.linspace(0, 2 * np.pi, 200)
    _cos_t = np.cos(_theta)
    _sin_t = np.sin(_theta)

    # L2 Circle
    _fig.add_trace(
        go.Scatter(
            x=_cos_t,
            y=_sin_t,
            mode="lines",
            line=dict(color="#3b82f6", width=2, dash="dot"),
            name="L2 Ball (Ridge)",
        ),
        row=1,
        col=1,
    )

    # L1 Diamond (|b1| + |b2| = 1)
    _l1_x = [1, 0, -1, 0, 1]
    _l1_y = [0, 1, 0, -1, 0]
    _fig.add_trace(
        go.Scatter(
            x=_l1_x,
            y=_l1_y,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="L1 Ball (Lasso)",
        ),
        row=1,
        col=1,
    )

    # Elastic Net Hybrid Ball: 0.5 * |b1| + 0.5 * |b2| + 0.25 * (b1^2 + b2^2) = C
    # Parametric trace:
    _en_pts_x = []
    _en_pts_y = []
    for _th in _theta:
        _dx, _dy = np.cos(_th), np.sin(_th)
        _l1_norm = abs(_dx) + abs(_dy)
        _l2_sq = _dx**2 + _dy**2
        # Solve r * (0.6 * l1 + 0.4 * r * l2_sq) = 1
        # 0.4 * r^2 + 0.6 * l1 * r - 1 = 0
        _a_q = 0.4 * _l2_sq
        _b_q = 0.6 * _l1_norm
        _r_sol = (-_b_q + np.sqrt(_b_q**2 + 4 * _a_q)) / (2 * _a_q)
        _en_pts_x.append(_r_sol * _dx)
        _en_pts_y.append(_r_sol * _dy)

    _fig.add_trace(
        go.Scatter(
            x=_en_pts_x,
            y=_en_pts_y,
            mode="lines",
            line=dict(color="#10b981", width=3),
            name="Elastic Net Ball",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Collinear Coefficient Paths
    _X_all = df_elastic.drop("y", axis=1)
    _y_all = df_elastic["y"]

    _alphas = np.logspace(-2, 1, 40)
    _lasso_b1 = []
    _lasso_b2 = []
    _enet_b1 = []
    _enet_b2 = []

    for _a in _alphas:
        _l_mod = Lasso(alpha=_a, random_state=42).fit(_X_all, _y_all)
        _lasso_b1.append(_l_mod.coef_[0])
        _lasso_b2.append(_l_mod.coef_[1])

        _e_mod = ElasticNet(alpha=_a, l1_ratio=0.5, random_state=42).fit(_X_all, _y_all)
        _enet_b1.append(_e_mod.coef_[0])
        _enet_b2.append(_e_mod.coef_[1])

    _fig.add_trace(
        go.Scatter(
            x=_alphas,
            y=_lasso_b1,
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Lasso x1 (Erratic Split)",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=_alphas,
            y=_lasso_b2,
            mode="lines",
            line=dict(color="#f87171", width=2, dash="dot"),
            name="Lasso x2 (Zeroed First)",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=_alphas,
            y=_enet_b1,
            mode="lines",
            line=dict(color="#10b981", width=2.5),
            name="ElasticNet x1 (Grouped)",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=_alphas,
            y=_enet_b2,
            mode="lines",
            line=dict(color="#047857", width=2.5),
            name="ElasticNet x2 (Grouped)",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Coefficient Profiles Across 4 Estimators
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(_X_all, _y_all, test_size=0.25, random_state=42)

    _ols = LinearRegression().fit(_X_tr, _y_tr)
    _ridge = Ridge(alpha=5.0).fit(_X_tr, _y_tr)
    _lasso = Lasso(alpha=0.3, random_state=42).fit(_X_tr, _y_tr)
    _enet = ElasticNet(alpha=0.3, l1_ratio=0.5, random_state=42).fit(_X_tr, _y_tr)

    _feat_labels = list(_X_all.columns)

    _fig.add_trace(
        go.Bar(
            x=_feat_labels,
            y=_ols.coef_,
            name="OLS (Unstable Weights)",
            marker_color="#94a3b8",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Bar(
            x=_feat_labels,
            y=_ridge.coef_,
            name="Ridge (No Sparsity)",
            marker_color="#3b82f6",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Bar(
            x=_feat_labels,
            y=_lasso.coef_,
            name="Lasso (Broken Grouping)",
            marker_color="#ef4444",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Bar(
            x=_feat_labels,
            y=_enet.coef_,
            name="ElasticNet (Sparse + Grouped)",
            marker_color="#10b981",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        barmode="group",
        title=dict(
            text="Elastic Net: Penalty Geometry, Grouping Dynamics, and Coefficient Selection",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Beta 1", range=[-1.4, 1.4], row=1, col=1)
    _fig.update_yaxes(title_text="Beta 2", range=[-1.4, 1.4], row=1, col=1)

    _fig.update_xaxes(title_text="Alpha Penalty Parameter", type="log", row=1, col=2)
    _fig.update_yaxes(title_text="Coefficient Value", row=1, col=2)

    _fig.update_xaxes(title_text="Predictor Feature", tickangle=35, row=1, col=3)
    _fig.update_yaxes(title_text="Estimated Weight (Beta)", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Cyclical Coordinate Descent Implementation of Elastic Net from Scratch**: Pure NumPy implementation applying the soft-thresholding operator and Ridge denominator update, verified against Scikit-Learn.
    2. **Multi-Model Generalization Benchmark on Collinear Data**: Comprehensive evaluation comparing OLS, Ridge, Lasso, and Elastic Net across Train/Test MSE, $R^2$, and Collinear Feature Balance ($|\hat{\beta}_1 - \hat{\beta}_2|$).
    """)
    return


@app.cell
def _(ElasticNet, df_elastic, mo, np, pd):
    # Example 1: Pure NumPy Coordinate Descent Implementation of Elastic Net
    _X = df_elastic.drop("y", axis=1).to_numpy()
    _y = df_elastic["y"].to_numpy()
    _n, _p = _X.shape

    # Standardize X and center y
    _X_mean = np.mean(_X, axis=0)
    _X_std = np.std(_X, axis=0, ddof=1)
    _X_norm = (_X - _X_mean) / _X_std
    _y_mean = np.mean(_y)
    _y_centered = _y - _y_mean

    _alpha = 0.2
    _l1_ratio = 0.6
    _lambda1 = _alpha * _l1_ratio
    _lambda2 = _alpha * (1.0 - _l1_ratio)

    # Soft thresholding function
    def _soft_threshold(z, gamma):
        return np.sign(z) * np.maximum(0.0, np.abs(z) - gamma)

    # Coordinate Descent Loop
    _beta_scratch = np.zeros(_p)
    _max_iter = 100
    for _iter in range(_max_iter):
        for _j in range(_p):
            # Compute partial residual: r_j = y - X_{-j} beta_{-j}
            _pred_others = _X_norm @ _beta_scratch - _X_norm[:, _j] * _beta_scratch[_j]
            _r_j = _y_centered - _pred_others

            # Correlation with column j
            _rho_j = np.dot(_X_norm[:, _j], _r_j) / _n

            # Elastic Net coordinate update
            _beta_scratch[_j] = _soft_threshold(_rho_j, _lambda1) / (1.0 + _lambda2)

    # Rescale scratch beta to original feature scale
    _beta_scratch_original = _beta_scratch / _X_std

    # Scikit-Learn Reference
    _enet_sk = ElasticNet(alpha=_alpha, l1_ratio=_l1_ratio, fit_intercept=True, random_state=42)
    _enet_sk.fit(_X, _y)

    _feature_names = list(df_elastic.drop("y", axis=1).columns)
    _comparison_rows = []
    for _i, _name in enumerate(_feature_names):
        _comparison_rows.append({
            "Feature": _name,
            "Scratch Coordinate Descent": f"{_beta_scratch_original[_i]:.4f}",
            "Scikit-Learn ElasticNet": f"{_enet_sk.coef_[_i]:.4f}",
            "Discrepancy": f"{np.abs(_beta_scratch_original[_i] - _enet_sk.coef_[_i]):.2e}",
        })

    _df_cd_comparison = pd.DataFrame(_comparison_rows)

    return (
        mo.md("#### Coordinate Descent Verification vs. Scikit-Learn"),
        mo.ui.table(_df_cd_comparison),
    )


@app.cell
def _(
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
    df_elastic,
    mean_squared_error,
    mo,
    np,
    pd,
    r2_score,
    train_test_split,
):
    # Example 2: Multi-Model Benchmark on Collinear Data
    _X = df_elastic.drop("y", axis=1)
    _y = df_elastic["y"]
    _X_tr, _X_te, _y_tr, _y_te = train_test_split(_X, _y, test_size=0.25, random_state=42)

    _models = {
        "OLS (Unregularized)": LinearRegression(),
        "Ridge (L2 = 1.0)": Ridge(alpha=1.0),
        "Lasso (L1 = 0.2)": Lasso(alpha=0.2, random_state=42),
        "Elastic Net (alpha=0.2, rho=0.5)": ElasticNet(alpha=0.2, l1_ratio=0.5, random_state=42),
    }

    _benchmark_records = []
    for _name, _mod in _models.items():
        _mod.fit(_X_tr, _y_tr)
        _pred_tr = _mod.predict(_X_tr)
        _pred_te = _mod.predict(_X_te)

        _b1 = _mod.coef_[0]
        _b2 = _mod.coef_[1]
        _collinear_diff = np.abs(_b1 - _b2)
        _zero_coefs = int(np.sum(np.abs(_mod.coef_) < 1e-4))

        _benchmark_records.append({
            "Model": _name,
            "Train R^2": f"{r2_score(_y_tr, _pred_tr):.4f}",
            "Test R^2": f"{r2_score(_y_te, _pred_te):.4f}",
            "Test MSE": f"{mean_squared_error(_y_te, _pred_te):.4f}",
            "Collinear Imbalance |b1 - b2|": f"{_collinear_diff:.3f}",
            "Zeroed Coefficients": f"{_zero_coefs} / 6",
            "Diagnosis": "Balanced & Sparse" if _name.startswith("Elastic") else ("Overfitting Collinearity" if "OLS" in _name else ("No Sparsity" if "Ridge" in _name else "Unstable Selection")),
        })

    _df_bench = pd.DataFrame(_benchmark_records)

    return (
        mo.md("#### Regularizer Benchmark under Collinear Features"),
        mo.ui.table(_df_bench),
    )


if __name__ == "__main__":
    app.run()
