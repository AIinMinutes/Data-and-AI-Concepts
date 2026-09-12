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
    # Note 09: Condition Number, Matrix Sensitivity, and Multicollinearity

    &larr; Previous Note: [08 Matrix Calculus](08_matrix_calculus_short.py) | Next Note: [10 Chebyshev Inequality](10_chebyshev_inequality.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Numerical computation is fundamentally vulnerable to error amplification. In theoretical mathematics, a matrix is either invertible or singular. In numerical computing and applied data science, however, matrices exist along a continuum of sensitivity measured by the **condition number** $\kappa(\mathbf{A})$.

    Understanding condition numbers and multicollinearity is essential across multiple core areas:
    1. **Numerical Stability of Linear Solvers and Inverses**: When solving $\mathbf{A}\mathbf{x} = \mathbf{b}$, machine round-off or small measurement noise in $\mathbf{b}$ is amplified by up to a factor of $\kappa(\mathbf{A})$ in the solution $\mathbf{x}$. If $\kappa(\mathbf{A}) \approx 10^k$, one can lose up to $k$ significant digits of floating-point precision.
    2. **Multicollinearity in Regression and Inference**: When predictor columns in a feature matrix $\mathbf{X}$ are nearly linearly dependent, the Gram matrix $\mathbf{X}^T \mathbf{X}$ becomes ill-conditioned. The parameter covariance matrix is $\text{Var}(\hat{\boldsymbol{\beta}}) = \sigma^2 (\mathbf{X}^T \mathbf{X})^{-1}$. High condition numbers cause parameter variances and standard errors to explode, flipping regression signs and destroying the reliability of hypothesis testing ($p$-values and confidence intervals).
    3. **Condition Squaring in Normal Equations**: The condition number of the Gram matrix is squared: $\kappa_2(\mathbf{X}^T \mathbf{X}) = (\kappa_2(\mathbf{X}))^2$. A feature matrix with a manageable condition number of $10^4$ produces a Gram matrix with $\kappa = 10^8$, consuming half of standard 64-bit double-precision digits. This explains why orthogonal factorizations (QR decomposition, SVD) are preferred over explicit matrix inversion.
    4. **Optimization Landscapes and Convergence Rates**: In first-order optimization (Gradient Descent), the convergence speed on a quadratic loss is bounded by $(\frac{\kappa - 1}{\kappa + 1})^2$, where $\kappa$ is the condition number of the Hessian $\mathbf{H} = \nabla^2 \mathcal{L}$. High condition numbers create steep, narrow canyons where gradients oscillate orthogonally to the descent direction.
    5. **Tikhonov / Ridge Regularization**: Adding an $L_2$ penalty $\lambda \|\mathbf{w}\|_2^2$ alters the spectrum to $\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}$, shrinking the condition number from $\frac{\sigma_1^2}{\sigma_p^2}$ to $\frac{\sigma_1^2 + \lambda}{\sigma_p^2 + \lambda}$ and restoring numerical stability.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Definition of the Condition Number

    For an invertible square matrix $\mathbf{A} \in \mathbb{R}^{n \times n}$ under an operator matrix norm $\|\cdot\|$, the **condition number** is defined as:

    $$
    \kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|
    $$

    Under the standard spectral norm (induced $L_2$ norm), $\|\mathbf{A}\|_2 = \sigma_{\max}(\mathbf{A})$ and $\|\mathbf{A}^{-1}\|_2 = 1 / \sigma_{\min}(\mathbf{A})$, where $\sigma_{\max}$ and $\sigma_{\min}$ are the largest and smallest singular values of $\mathbf{A}$:

    $$
    \kappa_2(\mathbf{A}) = \frac{\sigma_{\max}(\mathbf{A})}{\sigma_{\min}(\mathbf{A})}
    $$

    For a rectangular matrix $\mathbf{X} \in \mathbb{R}^{m \times n}$ with $m \geq n$ and full column rank, the 2-norm condition number is defined analogously via its non-zero extreme singular values:

    $$
    \kappa_2(\mathbf{X}) = \frac{\sigma_{\max}(\mathbf{X})}{\sigma_{\min}(\mathbf{X})}
    $$

    Since singular values are non-negative and $\sigma_{\max} \geq \sigma_{\min}$, we always have $\kappa_2(\mathbf{A}) \geq 1$. An orthogonal matrix has all singular values equal to $1$, yielding the optimal condition number $\kappa_2(\mathbf{Q}) = 1$.

    ---

    ### The Universal Perturbation Bound

    Consider the linear system $\mathbf{A}\mathbf{x} = \mathbf{b}$. Suppose the right-hand side vector is subject to perturbation $\Delta \mathbf{b}$, producing a perturbed solution $\mathbf{x} + \Delta \mathbf{x}$:

    $$
    \mathbf{A}(\mathbf{x} + \Delta\mathbf{x}) = \mathbf{b} + \Delta\mathbf{b} \implies \mathbf{A}\Delta\mathbf{x} = \Delta\mathbf{b} \implies \Delta\mathbf{x} = \mathbf{A}^{-1}\Delta\mathbf{b}
    $$

    Taking vector norms:

    $$
    \|\Delta\mathbf{x}\| \leq \|\mathbf{A}^{-1}\| \|\Delta\mathbf{b}\|
    $$

    Simultaneously, from $\mathbf{b} = \mathbf{A}\mathbf{x}$, submultiplicativity gives $\|\mathbf{b}\| \leq \|\mathbf{A}\| \|\mathbf{x}\|$, which is equivalent to:

    $$
    \frac{1}{\|\mathbf{x}\|} \leq \frac{\|\mathbf{A}\|}{\|\mathbf{b}\|}
    $$

    Multiplying these two inequalities yields the **relative perturbation bound**:

    $$
    \frac{\|\Delta\mathbf{x}\|}{\|\mathbf{x}\|} \leq \kappa(\mathbf{A}) \frac{\|\Delta\mathbf{b}\|}{\|\mathbf{b}\|}
    $$

    This inequality proves that $\kappa(\mathbf{A})$ acts as the maximum relative error amplification factor. If $\Delta \mathbf{b}$ aligns with the singular vector corresponding to $\sigma_{\min}$, the upper bound is achieved with exact equality.

    ---

    ### Condition Number Squaring in the Normal Equations

    In Ordinary Least Squares, the analytical parameter estimate is:

    $$
    \hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}
    $$

    Evaluating the condition number of the Gram matrix $\mathbf{X}^T \mathbf{X}$:

    $$
    \kappa_2(\mathbf{X}^T \mathbf{X}) = \frac{\lambda_{\max}(\mathbf{X}^T \mathbf{X})}{\lambda_{\min}(\mathbf{X}^T \mathbf{X})} = \frac{\sigma_{\max}^2(\mathbf{X})}{\sigma_{\min}^2(\mathbf{X})} = (\kappa_2(\mathbf{X}))^2
    $$

    This quadratic squaring is severe:
    * If $\mathbf{X}$ has condition number $10^3$, the Gram matrix has condition number $10^6$.
    * If $\mathbf{X}$ has condition number $10^8$, $\mathbf{X}^T \mathbf{X}$ has condition number $10^{16}$, which matches the limit of IEEE 754 double precision ($2^{-53} \approx 1.11 \times 10^{-16}$). In that scenario, inverting $\mathbf{X}^T \mathbf{X}$ results in total loss of numerical significance.

    ---

    ### Multicollinearity Diagnostics in Linear Models

    Multicollinearity arises when two or more feature columns in the design matrix $\mathbf{X}$ exhibit strong linear correlations.

    #### 1. Condition Index Diagnostic
    * $\kappa(\mathbf{X}) < 10$: Well-conditioned design; weak to no collinearity.
    * $10 \leq \kappa(\mathbf{X}) \leq 30$: Moderate collinearity; potential variance inflation.
    * $\kappa(\mathbf{X}) > 30$: Severe multicollinearity; parameter estimates and standard errors are unstable.

    #### 2. Variance Inflation Factor (VIF)
    For each predictor $x_j$, we regress $x_j$ against all other $p-1$ predictors in the dataset, obtaining the coefficient of determination $R_j^2$. The **Variance Inflation Factor** is:

    $$
    \text{VIF}_j = \frac{1}{1 - R_j^2}
    $$

    The variance of the estimated regression coefficient $\hat{\beta}_j$ can be decomposed into:

    $$
    \text{Var}(\hat{\beta}_j) = \frac{\sigma^2}{(n-1) s_j^2} \times \text{VIF}_j
    $$

    where $s_j^2 = \frac{1}{n-1} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2$ is the sample variance of feature $x_j$.
    * If $R_j^2 = 0.90$, $\text{VIF}_j = 10$ (the variance of $\hat{\beta}_j$ is inflated $10\times$, and standard error is inflated $\sqrt{10} \approx 3.16\times$).
    * If $R_j^2 = 0.99$, $\text{VIF}_j = 100$ (standard error inflated $10\times$).
    * Rule of thumb: $\text{VIF}_j > 10$ indicates critical multicollinearity requiring intervention.

    ---

    ### Stabilization via Ridge Regularization (Tikhonov)

    Ridge regression modifies the objective by adding an $L_2$ penalty:

    $$
    \mathcal{L}_{\text{ridge}}(\boldsymbol{\beta}) = \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \|\boldsymbol{\beta}\|_2^2
    $$

    The regularized closed-form estimator is:

    $$
    \hat{\boldsymbol{\beta}}_{\text{ridge}} = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
    $$

    Since the eigenvalues of $\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}$ are $\sigma_i^2 + \lambda$, the regularized condition number becomes:

    $$
    \kappa_2(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I}) = \frac{\sigma_{\max}^2 + \lambda}{\sigma_{\min}^2 + \lambda}
    $$

    Even for an ill-conditioned or rank-deficient matrix where $\sigma_{\min} \approx 0$, choosing $\lambda > 0$ guarantees that the denominator is at least $\lambda$, bounding the condition number and eliminating parameter variance explosion.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    ### Example 1: Interactive Visualizations: Geometric Sensitivity and Regularization Dynamics

    The interactive subplots below display the dual facets of conditioning:
    * **Left Panel**: Geometric demonstration of error magnification in an ill-conditioned linear system ($\kappa \approx 82$). Two nearly parallel lines intersect at $\mathbf{x}^* = [1.0, 1.0]^T$. An imperceptible $+2.4\%$ perturbation to $b_2$ causes the intersection point to fly across the coordinate plane to $\mathbf{x}_{\text{pert}} = [0.0, 2.0]^T$, yielding a displacement $\|\Delta\mathbf{x}\| = 1.414$ ($29\times$ amplification).
    * **Right Panel**: The stabilizing power of Ridge Regularization. As the regularization penalty $\lambda$ increases on a logarithmic scale, the condition number $\kappa_2(\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})$ collapses from over $900$ down below the critical threshold of $\kappa = 30$, restoring numerical stability.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Left Panel: Ill-conditioned 2x2 linear system
    # Line 1: x1 + x2 = 2.0 -> x2 = 2.0 - x1
    # Line 2: x1 + 1.05 x2 = 2.05 -> x2 = (2.05 - x1) / 1.05
    # Perturbed Line 2: x1 + 1.05 x2 = 2.10 (+0.05 perturbation, +2.4%)
    a_matrix = np.array([[1.0, 1.0], [1.0, 1.05]])
    b_vector = np.array([2.0, 2.05])
    delta_b_vec = np.array([0.0, 0.05])

    cond_a = float(np.linalg.cond(a_matrix))
    x_exact = np.linalg.solve(a_matrix, b_vector)
    x_perturbed = np.linalg.solve(a_matrix, b_vector + delta_b_vec)
    delta_x_vec = x_perturbed - x_exact
    norm_delta_x = float(np.linalg.norm(delta_x_vec))

    x_domain = np.linspace(-1.5, 3.5, 200)
    line1_y = 2.0 - x_domain
    line2_y = (2.05 - x_domain) / 1.05
    line2_perturbed_y = (2.10 - x_domain) / 1.05

    # Right Panel: Ridge Regularization Condition Number Curve
    # Typical collinear dataset with sigma_max^2 = 80.0, sigma_min^2 = 0.088 (cond = 909)
    sigma_max_squared = 80.0
    sigma_min_squared = 0.088
    lambda_grid = np.logspace(-4, 3, 200)
    cond_ridge_curve = (sigma_max_squared + lambda_grid) / (sigma_min_squared + lambda_grid)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Error Magnification in Ill-Conditioned System (κ={cond_a:.1f})",
            "Ridge Regularization: Condition Number vs λ",
        ],
    )

    # Left: Line 1
    fig.add_trace(
        go.Scatter(
            x=x_domain,
            y=line1_y,
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name="Line 1: x₁ + x₂ = 2.0",
        ),
        row=1,
        col=1,
    )

    # Left: Line 2 (Original)
    fig.add_trace(
        go.Scatter(
            x=x_domain,
            y=line2_y,
            mode="lines",
            line=dict(color="#dc2626", width=2.5),
            name="Line 2: x₁ + 1.05x₂ = 2.05",
        ),
        row=1,
        col=1,
    )

    # Left: Line 2 (Perturbed)
    fig.add_trace(
        go.Scatter(
            x=x_domain,
            y=line2_perturbed_y,
            mode="lines",
            line=dict(color="#dc2626", width=2.0, dash="dash"),
            name="Perturbed Line 2 (+2.4% Δb)",
        ),
        row=1,
        col=1,
    )

    # Left: Original Intersection Point
    fig.add_trace(
        go.Scatter(
            x=[x_exact[0]],
            y=[x_exact[1]],
            mode="markers",
            marker=dict(size=12, color="#16a34a", symbol="circle"),
            name="Original x* = [1.0, 1.0]",
        ),
        row=1,
        col=1,
    )

    # Left: Perturbed Intersection Point
    fig.add_trace(
        go.Scatter(
            x=[x_perturbed[0]],
            y=[x_perturbed[1]],
            mode="markers",
            marker=dict(size=12, color="#ea580c", symbol="diamond"),
            name=f"Perturbed x = [{x_perturbed[0]:.1f}, {x_perturbed[1]:.1f}]",
        ),
        row=1,
        col=1,
    )

    # Left: Error Displacement Vector Δx
    fig.add_trace(
        go.Scatter(
            x=[x_exact[0], x_perturbed[0]],
            y=[x_exact[1], x_perturbed[1]],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=3.5, dash="dot"),
            marker=dict(size=6, color="#7c3aed"),
            name=f"Amplified Error ||Δx||={norm_delta_x:.2f}",
        ),
        row=1,
        col=1,
    )

    # Right: Ridge Regularization Condition Number Curve
    fig.add_trace(
        go.Scatter(
            x=lambda_grid,
            y=cond_ridge_curve,
            mode="lines",
            line=dict(color="#2563eb", width=3.0),
            name="κ₂(XᵀX + λI)",
        ),
        row=1,
        col=2,
    )

    # Right: Severe Collinearity Threshold (κ = 30)
    fig.add_trace(
        go.Scatter(
            x=[1e-4, 1e3],
            y=[30.0, 30.0],
            mode="lines",
            line=dict(color="#dc2626", width=2.0, dash="dash"),
            name="Collinearity Threshold (κ=30)",
        ),
        row=1,
        col=2,
    )

    # Right: Unregularized Baseline (λ = 0)
    fig.add_trace(
        go.Scatter(
            x=[1e-4],
            y=[sigma_max_squared / sigma_min_squared],
            mode="markers",
            marker=dict(size=10, color="#dc2626", symbol="x"),
            name=f"Unregularized OLS (κ={sigma_max_squared / sigma_min_squared:.0f})",
        ),
        row=1,
        col=2,
    )

    axis_config_left = dict(
        title="x₁",
        range=[-1.0, 3.0],
        zeroline=True,
        zerolinecolor="#cbd5e1",
        gridcolor="#f1f5f9",
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=axis_config_left,
        yaxis=dict(
            title="x₂",
            range=[-0.5, 3.0],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            gridcolor="#f1f5f9",
        ),
        xaxis2=dict(type="log", title="Ridge Penalty λ (log scale)", gridcolor="#f1f5f9"),
        yaxis2=dict(type="log", title="Condition Number κ (log scale)", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        a_matrix,
        axis_config_left,
        b_vector,
        cond_a,
        cond_ridge_curve,
        delta_b_vec,
        delta_x_vec,
        fig,
        lambda_grid,
        line1_y,
        line2_perturbed_y,
        line2_y,
        norm_delta_x,
        sigma_max_squared,
        sigma_min_squared,
        x_domain,
        x_exact,
        x_perturbed,
    )


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Numerical Verification of the Relative Perturbation Bound

    In this example, we construct an ill-conditioned matrix $\mathbf{A} \in \mathbb{R}^{4 \times 4}$ with a pre-specified condition number $\kappa_2(\mathbf{A}) = 10,000$ using Singular Value Decomposition:

    $$
    \mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T, \quad \mathbf{\Sigma} = \text{diag}(10.0, 2.0, 0.5, 0.001)
    $$

    We apply five distinct perturbation vectors $\Delta\mathbf{b}$ of varying magnitudes and orientations, solve $\mathbf{A}(\mathbf{x} + \Delta\mathbf{x}) = \mathbf{b} + \Delta\mathbf{b}$, and verify that the empirical amplification factor never exceeds the theoretical condition number:

    $$
    \text{Amplification Ratio} = \frac{\|\Delta\mathbf{x}\| / \|\mathbf{x}\|}{\|\Delta\mathbf{b}\| / \|\mathbf{b}\|} \leq \kappa_2(\mathbf{A})
    $$
    """)
    return


@app.cell
def _(np):
    rng_p = np.random.default_rng(101)

    # Construct ill-conditioned 4x4 matrix with kappa = 10,000
    q_left, _ = np.linalg.qr(rng_p.standard_normal((4, 4)))
    q_right, _ = np.linalg.qr(rng_p.standard_normal((4, 4)))
    singular_values = np.array([10.0, 2.0, 0.5, 0.001])
    a_ill = q_left @ np.diag(singular_values) @ q_right.T
    kappa_exact = float(np.linalg.cond(a_ill))

    # True system Ax = b
    x_ground_truth = np.array([1.0, 1.0, 1.0, 1.0])
    b_exact = a_ill @ x_ground_truth
    norm_b = float(np.linalg.norm(b_exact))
    norm_x = float(np.linalg.norm(x_ground_truth))

    perturbation_scenarios = [
        ("Random Noise 1 (1e-4)", rng_p.standard_normal(4) * 1e-4),
        ("Random Noise 2 (1e-5)", rng_p.standard_normal(4) * 1e-5),
        ("Random Noise 3 (1e-3)", rng_p.standard_normal(4) * 1e-3),
        ("Aligned with Largest Mode (u₁)", q_left[:, 0] * 1e-4),
        ("Worst-Case Alignment (u₄)", q_left[:, 3] * 1e-4),
    ]

    bound_verification_records = []

    for name, delta_b in perturbation_scenarios:
        rel_delta_b = float(np.linalg.norm(delta_b) / norm_b)
        x_solved = np.linalg.solve(a_ill, b_exact + delta_b)
        delta_x = x_solved - x_ground_truth
        rel_delta_x = float(np.linalg.norm(delta_x) / norm_x)
        amplification = rel_delta_x / rel_delta_b if rel_delta_b > 0 else 0.0
        bound_satisfied = amplification <= (kappa_exact + 1e-7)

        bound_verification_records.append(
            {
                "Perturbation Type": name,
                "Relative ||Δb||/||b||": f"{rel_delta_b:.2e}",
                "Relative ||Δx||/||x||": f"{rel_delta_x:.2e}",
                "Amplification Factor": f"{amplification:.1f}",
                "Theoretical Bound κ": f"{kappa_exact:.1f}",
                "Bound Satisfied": str(bound_satisfied),
            }
        )

    return (
        a_ill,
        b_exact,
        bound_satisfied,
        bound_verification_records,
        delta_b,
        delta_x,
        kappa_exact,
        name,
        norm_b,
        norm_x,
        perturbation_scenarios,
        q_left,
        q_right,
        rel_delta_b,
        rel_delta_x,
        rng_p,
        singular_values,
        x_ground_truth,
        x_solved,
    )


@app.cell(hide_code=True)
def _(bound_verification_records, mo, pd):
    df_perturbation = pd.DataFrame(bound_verification_records)
    mo.ui.table(df_perturbation)
    return (df_perturbation,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 3: Multicollinearity Diagnostics, Variance Inflation, and Ridge Healing

    In this example, we generate a synthetic regression dataset ($N = 100$) with predictors:
    * $x_1 \sim \mathcal{N}(50, 15^2)$
    * $x_2 \sim \mathcal{N}(30, 10^2)$
    * $x_3 = 0.9 x_1 + 0.1 x_2 + \epsilon$ (an artificial linear duplicate)

    The true data-generating process does not depend on $x_3$:

    $$
    y = 5.0 + 2.5 x_1 - 1.8 x_2 + \mathcal{N}(0, 1)
    $$

    We compute the Variance Inflation Factors (VIF) and compare three models:
    1. **Full Collinear OLS ($x_1, x_2, x_3$)**: Suffers from variance explosion; standard errors blow up and estimated coefficients become unreliable.
    2. **Pruned OLS ($x_1, x_2$)**: Dropping the redundant collinear feature drops VIF to $\approx 1$, eliminates the variance explosion, and estimates true coefficients accurately.
    3. **Ridge Regression ($\lambda = 5.0$)**: Adds an $L_2$ penalty to control ill-conditioning, shrinking parameter variance without discarding features.
    """)
    return


@app.cell
def _(np):
    rng_reg = np.random.default_rng(42)
    n_pts = 100

    # Predictors
    x1_val = rng_reg.normal(50, 15, n_pts)
    x2_val = rng_reg.normal(30, 10, n_pts)
    # x3 is highly collinear with x1 and x2
    x3_val = 0.9 * x1_val + 0.1 * x2_val + rng_reg.normal(0, 0.05, n_pts)

    true_intercept = 5.0
    true_beta1 = 2.5
    true_beta2 = -1.8
    true_beta3 = 0.0
    y_response = true_intercept + true_beta1 * x1_val + true_beta2 * x2_val + rng_reg.normal(0, 1.0, n_pts)

    # Design matrices
    x_matrix_raw = np.column_stack([x1_val, x2_val, x3_val])

    # Standardize design matrix for condition number and VIF
    x_matrix_std = (x_matrix_raw - x_matrix_raw.mean(axis=0)) / x_matrix_raw.std(axis=0)
    cond_full_std = float(np.linalg.cond(x_matrix_std))
    cond_pruned_std = float(np.linalg.cond(x_matrix_std[:, :2]))

    # VIF computation via R^2
    def calc_vif_array(mat):
        vifs = []
        p_cols = mat.shape[1]
        for idx in range(p_cols):
            y_target = mat[:, idx]
            other_feats = np.delete(mat, idx, axis=1)
            design = np.column_stack([np.ones(len(y_target)), other_feats])
            beta_hat, _, _, _ = np.linalg.lstsq(design, y_target, rcond=None)
            pred = design @ beta_hat
            ss_tot = float(np.sum((y_target - np.mean(y_target)) ** 2))
            ss_res = float(np.sum((y_target - pred) ** 2))
            r_squared = 1.0 - (ss_res / ss_tot)
            vif_val = 1.0 / (1.0 - r_squared) if (1.0 - r_squared) > 1e-9 else 1e9
            vifs.append(vif_val)
        return vifs

    vif_full = calc_vif_array(x_matrix_raw)
    vif_pruned = calc_vif_array(x_matrix_raw[:, :2])

    # Model 1: Full OLS
    x_design_full = np.column_stack([np.ones(n_pts), x_matrix_raw])
    beta_full_ols, _, _, _ = np.linalg.lstsq(x_design_full, y_response, rcond=None)
    residuals_full = y_response - x_design_full @ beta_full_ols
    sigma_squared_full = float(np.sum(residuals_full**2) / (n_pts - 4))
    cov_matrix_full = sigma_squared_full * np.linalg.inv(x_design_full.T @ x_design_full)
    se_full_ols = np.sqrt(np.diag(cov_matrix_full))

    # Model 2: Pruned OLS (dropping x3)
    x_design_pruned = np.column_stack([np.ones(n_pts), x_matrix_raw[:, :2]])
    beta_pruned_ols, _, _, _ = np.linalg.lstsq(x_design_pruned, y_response, rcond=None)
    residuals_pruned = y_response - x_design_pruned @ beta_pruned_ols
    sigma_squared_pruned = float(np.sum(residuals_pruned**2) / (n_pts - 3))
    cov_matrix_pruned = sigma_squared_pruned * np.linalg.inv(x_design_pruned.T @ x_design_pruned)
    se_pruned_ols = np.sqrt(np.diag(cov_matrix_pruned))

    # Model 3: Ridge Regression (lambda = 5.0)
    lambda_param = 5.0
    y_centered = y_response - np.mean(y_response)
    ridge_gram = x_matrix_std.T @ x_matrix_std + lambda_param * np.eye(3)
    beta_ridge_std = np.linalg.solve(ridge_gram, x_matrix_std.T @ y_centered)
    beta_ridge_raw = beta_ridge_std / x_matrix_raw.std(axis=0)
    intercept_ridge = float(np.mean(y_response) - np.sum(beta_ridge_raw * x_matrix_raw.mean(axis=0)))
    cond_ridge = float(np.linalg.cond(ridge_gram))

    multicollinearity_comparison = {
        "Model": [
            "True Parameters",
            "1. Full OLS (with collinear x3)",
            "2. Pruned OLS (x3 removed)",
            "3. Ridge Regression (λ=5.0)",
        ],
        "Design κ (Standardized)": [
            "1.0",
            f"{cond_full_std:.1f}",
            f"{cond_pruned_std:.1f}",
            f"{cond_ridge:.1f} (effective)",
        ],
        "Max VIF": [
            "1.0",
            f"{max(vif_full):.1f}",
            f"{max(vif_pruned):.2f}",
            "Regularized",
        ],
        "β₁ Estimate (SE) [True: 2.50]": [
            "2.500 (-)",
            f"{beta_full_ols[1]:.3f} (SE: {se_full_ols[1]:.3f})",
            f"{beta_pruned_ols[1]:.3f} (SE: {se_pruned_ols[1]:.3f})",
            f"{beta_ridge_raw[0]:.3f} (Regularized)",
        ],
        "β₂ Estimate (SE) [True: -1.80]": [
            "-1.800 (-)",
            f"{beta_full_ols[2]:.3f} (SE: {se_full_ols[2]:.3f})",
            f"{beta_pruned_ols[2]:.3f} (SE: {se_pruned_ols[2]:.3f})",
            f"{beta_ridge_raw[1]:.3f} (Regularized)",
        ],
        "β₃ Estimate (SE) [True: 0.00]": [
            "0.000 (-)",
            f"{beta_full_ols[3]:.3f} (SE: {se_full_ols[3]:.3f})",
            "Dropped",
            f"{beta_ridge_raw[2]:.3f} (Regularized)",
        ],
    }

    return (
        beta_full_ols,
        beta_pruned_ols,
        beta_ridge_raw,
        beta_ridge_std,
        calc_vif_array,
        cond_full_std,
        cond_pruned_std,
        cond_ridge,
        cov_matrix_full,
        cov_matrix_pruned,
        intercept_ridge,
        lambda_param,
        multicollinearity_comparison,
        n_pts,
        residuals_full,
        residuals_pruned,
        ridge_gram,
        rng_reg,
        se_full_ols,
        se_pruned_ols,
        sigma_squared_full,
        sigma_squared_pruned,
        true_beta1,
        true_beta2,
        true_beta3,
        true_intercept,
        vif_full,
        vif_pruned,
        x1_val,
        x2_val,
        x3_val,
        x_design_full,
        x_design_pruned,
        x_matrix_raw,
        x_matrix_std,
        y_centered,
        y_response,
    )


@app.cell(hide_code=True)
def _(mo, multicollinearity_comparison, pd):
    df_comparison = pd.DataFrame(multicollinearity_comparison)
    mo.ui.table(df_comparison)
    return (df_comparison,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    * **Condition Number $\kappa$**: Measures the sensitivity of a linear system to perturbations. A high $\kappa$ means small input errors or round-off errors get massively amplified in the solution.
    * **Condition Squaring**: In OLS, solving the Normal Equations squares the condition number of the design matrix: $\kappa(\mathbf{X}^T\mathbf{X}) = \kappa(\mathbf{X})^2$. This is why SVD or QR factorization is preferred over explicit matrix inversion for collinear data.
    * **Multicollinearity & VIF**: High condition numbers ($\kappa > 30$) or High Variance Inflation Factors ($\text{VIF}_j > 10$) indicate severe multicollinearity, causing parameter variance to explode and making hypothesis tests unreliable.
    * **Ridge Regularization**: Adding an $L_2$ penalty $\lambda$ strictly bounds the minimum eigenvalue away from zero ($\sigma_{\min}^2 + \lambda$), safely artificially lowering the condition number and stabilizing parameter estimates at the cost of a little bias.

    ---

    &larr; Previous Note: [08 Matrix Calculus](08_matrix_calculus_short.py) | Next Note: [10 Chebyshev Inequality](10_chebyshev_inequality.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
