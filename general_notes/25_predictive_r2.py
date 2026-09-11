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
    from sklearn.linear_model import LinearRegression

    return LinearRegression, go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 25: Predictive R-Squared, PRESS Residuals, and Leverage Diagnostics

    &larr; Previous Note: [24 Adjusted R-Squared](24_adjusted_r_squared.py) | Next Note: [26 Hotelling T-Squared](26_hotelling.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Both ordinary $R^2$ and Adjusted $R^2$ evaluate goodness-of-fit strictly on in-sample training data. A model can achieve an Adjusted $R^2$ of $0.92$ yet fail catastrophically when deployed to predict new, unseen observations. This occurs when high-leverage training observations dictate the hyperplane slope or when polynomial basis expansions overfit local sample fluctuations.

    **Predictive $R^2$ ($R^2_{\text{pred}}$)** resolves this dilemma through Leave-One-Out Cross-Validation (LOOCV):
    1. **Zero-Cost Out-of-Sample Evaluation**: Standard LOOCV requires training $n$ separate models, which is computationally prohibitive for large datasets. In linear regression, the **Sherman-Morrison formula** enables the exact computation of all $n$ leave-one-out residuals in a single matrix operation without refitting the model even once.
    2. **The PRESS Statistic**: The Prediction Error Sum of Squares (PRESS) aggregates the squared leave-one-out prediction errors:

    $$
    \text{PRESS} = \sum_{i=1}^n \left(\frac{e_i}{1 - h_{ii}}\right)^2
    $$

    where $h_{ii}$ is the leverage of observation $i$. If a point has high leverage ($h_{ii} \to 1$), its residual is magnified by $\frac{1}{(1 - h_{ii})^2}$, heavily penalizing models that depend excessively on isolated, influential points.
    3. **Detecting Overfitting via the Generalization Gap**: Predictive $R^2$ is defined as $1 - \frac{\text{PRESS}}{\text{SS}_{\text{tot}}}$. While raw $R^2$ always increases with model complexity, Predictive $R^2$ reaches a maximum and drops precipitously (often becoming strongly negative), revealing precisely when additional features degrade generalization.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. The Hat (Projection) Matrix and Leverage

    In multiple linear regression with design matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$ (including intercept):

    $$
    \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \hat{\boldsymbol{\beta}} = (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y}
    $$

    The fitted values $\hat{\mathbf{y}}$ are obtained via the orthogonal projection (Hat) matrix $\mathbf{H}$:

    $$
    \hat{\mathbf{y}} = \mathbf{X}\hat{\boldsymbol{\beta}} = \mathbf{X}(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{X}^\top \mathbf{y} = \mathbf{H}\mathbf{y}
    $$

    The diagonal elements $h_{ii} = [\mathbf{H}]_{ii}$ are the **leverage values**:

    $$
    h_{ii} = \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i
    $$

    #### Fundamental Properties of Leverage:
    - Bounded in $[0, 1]$: $0 \leq h_{ii} \leq 1$.
    - Sum of leverages equals the number of parameters: $\operatorname{tr}(\mathbf{H}) = \sum_{i=1}^n h_{ii} = p$.
    - Average leverage: $\bar{h} = \frac{p}{n}$.
    - High leverage threshold: An observation is deemed high-leverage if $h_{ii} > \frac{2p}{n}$ (or $\frac{3p}{n}$).

    ---

    ### 2. The PRESS Shortcut Derivation (Sherman-Morrison)

    Let $(i)$ denote estimation with the $i$-th observation removed. The leave-one-out parameter estimate is:

    $$
    \hat{\boldsymbol{\beta}}_{(i)} = \left(\mathbf{X}_{(i)}^\top \mathbf{X}_{(i)}\right)^{-1} \mathbf{X}_{(i)}^\top \mathbf{y}_{(i)}
    $$

    Notice that removing row $\mathbf{x}_i$ is a symmetric rank-one downdate:

    $$
    \mathbf{X}_{(i)}^\top \mathbf{X}_{(i)} = \mathbf{X}^\top \mathbf{X} - \mathbf{x}_i \mathbf{x}_i^\top
    $$

    By the Sherman-Morrison rank-one inverse formula:

    $$
    \left(\mathbf{X}^\top \mathbf{X} - \mathbf{x}_i \mathbf{x}_i^\top\right)^{-1} = (\mathbf{X}^\top \mathbf{X})^{-1} + \frac{(\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{x}_i \mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1}}{1 - h_{ii}}
    $$

    Multiplying by $\mathbf{X}_{(i)}^\top \mathbf{y}_{(i)} = \mathbf{X}^\top \mathbf{y} - \mathbf{x}_i y_i$ and simplifying yields:

    $$
    \hat{\boldsymbol{\beta}}_{(i)} = \hat{\boldsymbol{\beta}} - \frac{(\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{x}_i e_i}{1 - h_{ii}}
    $$

    The predicted value for the held-out point is $\hat{y}_{(i)} = \mathbf{x}_i^\top \hat{\boldsymbol{\beta}}_{(i)}$:

    $$
    \hat{y}_{(i)} = \mathbf{x}_i^\top \hat{\boldsymbol{\beta}} - \frac{\mathbf{x}_i^\top (\mathbf{X}^\top \mathbf{X})^{-1}\mathbf{x}_i e_i}{1 - h_{ii}} = \hat{y}_i - \frac{h_{ii} e_i}{1 - h_{ii}}
    $$

    Subtracting this from the observed target $y_i$ yields the leave-one-out error $e_{(i)}$:

    $$
    e_{(i)} = y_i - \hat{y}_{(i)} = y_i - \hat{y}_i + \frac{h_{ii} e_i}{1 - h_{ii}} = e_i \left(1 + \frac{h_{ii}}{1 - h_{ii}}\right) = \frac{e_i}{1 - h_{ii}}
    $$

    This identity proves that **Leave-One-Out residuals can be computed directly from standard OLS residuals and leverage values with zero re-fitting**.

    ---

    ### 3. The PRESS Statistic and Predictive $R^2$

    The Prediction Error Sum of Squares (PRESS) is:

    $$
    \text{PRESS} = \sum_{i=1}^n e_{(i)}^2 = \sum_{i=1}^n \left(\frac{e_i}{1 - h_{ii}}\right)^2
    $$

    The **Predictive $R^2$** is defined as:

    $$
    R^2_{\text{pred}} = 1 - \frac{\text{PRESS}}{\text{SS}_{\text{tot}}}
    $$

    #### Comparison of the Three $R^2$ Metrics:
    $$
    R^2 \geq R^2_{\text{adj}} \geq R^2_{\text{pred}}
    $$
    - **Raw $R^2$**: Evaluates in-sample fitting accuracy.
    - **Adjusted $R^2$**: Penalizes degrees of freedom consumed by predictors.
    - **Predictive $R^2$**: Evaluates true out-of-sample leave-one-out generalization.
    """)
    return


@app.cell
def _(np, pd):
    # Simulation Data: Non-linear relationship y = 1.5 * x - 2.0 * x^2 + 0.8 * x^3 + noise
    # Designed to test polynomial models degree 1 through 6
    np.random.seed(47)
    _n = 35

    x_raw = np.sort(np.random.uniform(-1.8, 1.8, _n))
    _true_signal = 1.5 * x_raw - 2.0 * (x_raw**2) + 0.8 * (x_raw**3)
    _noise = np.random.normal(0.0, 0.8, _n)

    # Incur an intentional isolated high-leverage point at the extreme right
    y_raw = _true_signal + _noise
    y_raw[-1] += 2.5  # high-leverage perturbed observation

    df_poly = pd.DataFrame({"x": np.round(x_raw, 3), "y": np.round(y_raw, 3)})

    return df_poly, x_raw, y_raw


@app.cell
def _(df_poly, go, make_subplots, mo, np, x_raw, y_raw):
    # Interactive Visualizations Cell:
    # Subplot 1: Fitted Polynomial curves (Degree 1 Underfitting, Degree 3 Optimal, Degree 6 Overfitting)
    # Subplot 2: R^2 vs. Adjusted R^2 vs. Predictive R^2 across Polynomial Degrees 1 to 6
    # Subplot 3: Leverage (h_ii) vs. PRESS Residual Inflation Factor 1 / (1 - h_ii)

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Polynomial Fits: Underfit vs. Optimal vs. Overfit",
            "2. Generalization Gap: R^2 vs. Adj R^2 vs. Pred R^2",
            "3. Leverage & PRESS Residual Inflation",
        ),
        horizontal_spacing=0.09,
    )

    _n = len(x_raw)
    _ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)

    # Fit Polynomial Degrees 1 through 6
    _degrees = np.arange(1, 7)
    _r2_list = []
    _adj_r2_list = []
    _pred_r2_list = []

    _fits_to_plot = {}
    _x_dense = np.linspace(x_raw.min(), x_raw.max(), 100)

    for _d in _degrees:
        _X_mat = np.vander(x_raw, _d + 1)  # includes column of ones
        _beta, _, _, _ = np.linalg.lstsq(_X_mat, y_raw, rcond=None)
        _y_hat = _X_mat @ _beta
        _e = y_raw - _y_hat

        # Hat matrix
        _H = _X_mat @ np.linalg.solve(_X_mat.T @ _X_mat, _X_mat.T)
        _h = np.diag(_H)

        _ss_res = np.sum(_e**2)
        _press = np.sum((_e / (1.0 - _h)) ** 2)

        _p = _d + 1
        _r2 = 1.0 - (_ss_res / _ss_tot)
        _adj_r2 = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - _p)
        _pred_r2 = 1.0 - (_press / _ss_tot)

        _r2_list.append(_r2)
        _adj_r2_list.append(_adj_r2)
        _pred_r2_list.append(_pred_r2)

        if _d in [1, 3, 6]:
            _X_dense = np.vander(_x_dense, _d + 1)
            _fits_to_plot[_d] = _X_dense @ _beta

    # Subplot 1: Fits
    _fig.add_trace(
        go.Scatter(
            x=df_poly["x"],
            y=df_poly["y"],
            mode="markers",
            marker=dict(size=7, color="#334155"),
            name="Observed Samples",
        ),
        row=1,
        col=1,
    )

    _fit_colors = {1: "#ef4444", 3: "#10b981", 6: "#8b5cf6"}
    _fit_labels = {1: "Degree 1 (Underfit)", 3: "Degree 3 (Optimal)", 6: "Degree 6 (Overfit)"}
    for _d, _col in _fit_colors.items():
        _fig.add_trace(
            go.Scatter(
                x=_x_dense,
                y=_fits_to_plot[_d],
                mode="lines",
                line=dict(color=_col, width=2.5),
                name=_fit_labels[_d],
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Metric Trajectories
    _fig.add_trace(
        go.Scatter(
            x=_degrees,
            y=_r2_list,
            mode="lines+markers",
            line=dict(color="#3b82f6", width=2.5),
            name="Raw R^2 (In-sample)",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=_degrees,
            y=_adj_r2_list,
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            name="Adjusted R^2",
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Scatter(
            x=_degrees,
            y=_pred_r2_list,
            mode="lines+markers",
            line=dict(color="#ef4444", width=3, dash="dash"),
            name="Predictive R^2 (PRESS)",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Leverage Diagnostic for Degree 3
    _X_deg3 = np.vander(x_raw, 4)
    _H_deg3 = _X_deg3 @ np.linalg.solve(_X_deg3.T @ _X_deg3, _X_deg3.T)
    _h_vals = np.diag(_H_deg3)
    _inflation_factor = 1.0 / (1.0 - _h_vals)
    _high_lev_thresh = 2.0 * 4 / _n

    _fig.add_trace(
        go.Scatter(
            x=_h_vals,
            y=_inflation_factor,
            mode="markers",
            marker=dict(
                size=9,
                color=["#ef4444" if h > _high_lev_thresh else "#3b82f6" for h in _h_vals],
                line=dict(width=1, color="#1e293b"),
            ),
            name="Observations",
            hovertemplate="Leverage h_ii: %{x:.3f}<br>PRESS Multiplier: %{y:.2f}x<extra></extra>",
        ),
        row=1,
        col=3,
    )

    _fig.add_trace(
        go.Scatter(
            x=[_high_lev_thresh, _high_lev_thresh],
            y=[1.0, _inflation_factor.max() * 1.05],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=1.5),
            name="Threshold 2p/n",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Predictive R^2 & PRESS: Diagnosing Overfitting and High-Leverage Outliers",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="x", row=1, col=1)
    _fig.update_yaxes(title_text="y", row=1, col=1)

    _fig.update_xaxes(title_text="Polynomial Degree", tickvals=_degrees, row=1, col=2)
    _fig.update_yaxes(title_text="Metric Score", range=[-0.4, 1.05], row=1, col=2)

    _fig.update_xaxes(title_text="Leverage (h_ii)", row=1, col=3)
    _fig.update_yaxes(title_text="Error Inflation 1 / (1 - h_ii)", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two end-to-end production algorithmic workflows:
    1. **Exact Equivalence of the PRESS Shortcut vs. Brute-Force LOOCV**: We explicitly fit $n = 35$ separate linear regression models, dropping one point at a time, and verify that the brute-force LOOCV prediction error matches the Sherman-Morrison shortcut formula $\frac{e_i}{1 - h_{ii}}$ to machine precision ($10^{-12}$).
    2. **Comprehensive Model Order Selection Table**: Evaluating models of degree 1 through 6 across SSE, PRESS, $R^2, \bar{R}^2$, and $R^2_{\text{pred}}$, proving that Predictive $R^2$ prevents model overparameterization.
    """)
    return


@app.cell
def _(LinearRegression, mo, np, pd, x_raw, y_raw):
    # Example 1: Numerical Verification of the PRESS Shortcut vs. Brute-Force LOOCV
    _n = len(x_raw)
    _X = np.column_stack([np.ones(_n), x_raw, x_raw**2, x_raw**3])
    _p = _X.shape[1]

    # Standard full fit
    _beta = np.linalg.solve(_X.T @ _X, _X.T @ y_raw)
    _y_hat = _X @ _beta
    _residuals = y_raw - _y_hat

    # Hat matrix diagonal
    _H = _X @ np.linalg.solve(_X.T @ _X, _X.T)
    _h_diag = np.diag(_H)

    # 1. Shortcut PRESS residuals
    _press_shortcut = _residuals / (1.0 - _h_diag)
    _press_stat_shortcut = np.sum(_press_shortcut**2)

    # 2. Brute-force n-fold Leave-One-Out Cross-Validation
    _brute_force_errors = np.zeros(_n)
    for _i in range(_n):
        _mask = np.ones(_n, dtype=bool)
        _mask[_i] = False
        _X_train = _X[_mask]
        _y_train = y_raw[_mask]
        _X_test = _X[_i : _i + 1]

        _model = LinearRegression(fit_intercept=False).fit(_X_train, _y_train)
        _pred_loo = _model.predict(_X_test)[0]
        _brute_force_errors[_i] = y_raw[_i] - _pred_loo

    _press_stat_bruteforce = np.sum(_brute_force_errors**2)
    _max_discrepancy = np.max(np.abs(_press_shortcut - _brute_force_errors))

    _df_verification = pd.DataFrame(
        [
            {"Evaluation Method": "PRESS Shortcut: e_i / (1 - h_ii)", "PRESS Statistic": f"{_press_stat_shortcut:.6f}", "Compute Time / Complexity": "O(n p^2) [Single OLS Fit]"},
            {"Evaluation Method": "Brute-Force LOOCV: n re-fits", "PRESS Statistic": f"{_press_stat_bruteforce:.6f}", "Compute Time / Complexity": "O(n^2 p^2) [35 separate models]"},
            {"Evaluation Method": "Max Absolute Error Discrepancy", "PRESS Statistic": f"{_max_discrepancy:.2e}", "Compute Time / Complexity": "Exact to Machine Precision"},
        ]
    )

    return (
        mo.md("#### Mathematical Equivalence: PRESS Shortcut vs. Brute-Force LOOCV"),
        mo.ui.table(_df_verification),
    )


@app.cell
def _(mo, np, pd, x_raw, y_raw):
    # Example 2: Model Order Selection Diagnostic Table (Degrees 1 to 6)
    _n = len(x_raw)
    _ss_tot = np.sum((y_raw - np.mean(y_raw)) ** 2)

    _eval_rows = []
    for _deg in range(1, 7):
        _p = _deg + 1
        _X = np.vander(x_raw, _p)
        _beta = np.linalg.solve(_X.T @ _X, _X.T @ y_raw)
        _y_pred = _X @ _beta
        _e = y_raw - _y_pred

        _H = _X @ np.linalg.solve(_X.T @ _X, _X.T)
        _h = np.diag(_H)

        _sse = np.sum(_e**2)
        _press = np.sum((_e / (1.0 - _h)) ** 2)

        _r2 = 1.0 - (_sse / _ss_tot)
        _adj_r2 = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - _p)
        _pred_r2 = 1.0 - (_press / _ss_tot)

        _status = "Optimal Model" if _deg == 3 else ("Underfitting" if _deg < 3 else "Overfitting (Generalization Drops)")

        _eval_rows.append({
            "Polynomial Order": f"Degree {_deg} (p = {_p})",
            "SSE (In-sample)": f"{_sse:.2f}",
            "PRESS (Out-of-sample)": f"{_press:.2f}",
            "Raw R^2": f"{_r2:.4f}",
            "Adjusted R^2": f"{_adj_r2:.4f}",
            "Predictive R^2": f"{_pred_r2:.4f}",
            "Generalization Verdict": _status,
        })

    _df_order_selection = pd.DataFrame(_eval_rows)

    return (
        mo.md("#### Polynomial Complexity Evaluation: R^2 vs. Adjusted R^2 vs. Predictive R^2"),
        mo.ui.table(_df_order_selection),
    )


if __name__ == "__main__":
    app.run()
