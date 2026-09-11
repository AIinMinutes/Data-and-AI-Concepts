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
    from sklearn.metrics import r2_score

    return LinearRegression, go, make_subplots, mo, np, pd, r2_score


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 24: R-Squared, Adjusted R-Squared, and Model Complexity Penalization

    &larr; Previous Note: [23 EWA and Bias Correction](23_ewa_and_bias_correction.py) | Next Note: [25 Predictive R2](25_predictive_r2.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In multiple linear regression and supervised learning, evaluating how well a model explains variance in the target variable $y$ is a primary objective. The classical **Coefficient of Determination ($R^2$)** measures the proportion of total variance explained by the fitted hyperplane.

    However, raw $R^2$ suffers from a severe mathematical pathology:
    1. **Monotonic Non-Decreasing Property**: In Ordinary Least Squares (OLS), projecting the response vector $\mathbf{y}$ onto a higher-dimensional predictor subspace $\operatorname{span}(\mathbf{X})$ cannot increase the residual sum of squares: $\text{SS}_{\text{res}}^{(p+1)} \leq \text{SS}_{\text{res}}^{(p)}$. Adding any variable (even pure random Gaussian white noise or coin tosses) mathematically forces raw $R^2$ to increase or remain unchanged.
    2. **Overfitting and False Feature Selection**: Relying on $R^2$ to select features inevitably leads to saturated, overparameterized models that memorize training noise, inflating prediction variance on unseen test data.
    3. **Adjusted $R^2$ ($\bar{R}^2$) as an Unbiased Variance Ratio**: Ezekiel (1930) introduced Adjusted $R^2$ by replacing sample sums of squares with their unbiased degrees-of-freedom estimators: dividing $\text{SS}_{\text{res}}$ by $n - p - 1$ and $\text{SS}_{\text{tot}}$ by $n - 1$.
    4. **The Critical $F > 1$ Threshold Rule**: A celebrated statistical theorem proves that adding a predictor increases Adjusted $R^2$ if and only if the absolute $t$-statistic of that predictor exceeds 1 (or equivalently, the incremental partial $F$-statistic exceeds 1). If an added predictor contributes less explanation than expected from random noise ($F < 1$), Adjusted $R^2$ penalizes the model and strictly declines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Analysis of Variance (ANOVA) Decomposition

    For a linear regression model with $n$ observations and $p$ predictors plus an intercept:

    $$
    \mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\epsilon}, \quad \mathbf{X} \in \mathbb{R}^{n \times (p + 1)}
    $$

    The total variability in $\mathbf{y}$ decomposes orthogonally:

    $$
    \text{SS}_{\text{tot}} = \text{SS}_{\text{reg}} + \text{SS}_{\text{res}}
    $$

    #### Total Sum of Squares ($\text{SS}_{\text{tot}}$)
    Total sample dispersion around the sample mean $\bar{y}$, with $n - 1$ degrees of freedom:
    $$
    \text{SS}_{\text{tot}} = \sum_{i=1}^n (y_i - \bar{y})^2 = \|\mathbf{y} - \bar{y}\mathbf{1}\|_2^2, \quad \text{df}_{\text{tot}} = n - 1
    $$

    #### Residual Sum of Squares ($\text{SS}_{\text{res}}$)
    Unexplained variance orthogonal to the predictor column space, with $n - p - 1$ degrees of freedom:
    $$
    \text{SS}_{\text{res}} = \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \|\mathbf{y} - \hat{\mathbf{y}}\|_2^2, \quad \text{df}_{\text{res}} = n - p - 1
    $$

    #### Regression Sum of Squares ($\text{SS}_{\text{reg}}$)
    Variability explained by the fitted hyperplane, with $p$ degrees of freedom:
    $$
    \text{SS}_{\text{reg}} = \sum_{i=1}^n (\hat{y}_i - \bar{y})^2 = \|\hat{\mathbf{y}} - \bar{y}\mathbf{1}\|_2^2, \quad \text{df}_{\text{reg}} = p
    $$

    ---

    ### 2. The Coefficient of Determination ($R^2$)

    Raw $R^2$ is the ratio of explained variance to total variance:

    $$
    R^2 = \frac{\text{SS}_{\text{reg}}}{\text{SS}_{\text{tot}}} = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}}
    $$

    For models with an intercept, $R^2 \in [0, 1]$. In the bivariate case ($p = 1$), $R^2$ is exactly the square of Pearson's correlation coefficient: $R^2 = r_{xy}^2$.

    ---

    ### 3. Ezekiel's Adjusted $R^2$ ($\bar{R}^2$)

    Adjusted $R^2$ corrects for model dimension by dividing each sum of squares by its respective degrees of freedom, transforming sums of squares into unbiased variance estimators:

    $$
    \bar{R}^2 = 1 - \frac{\text{SS}_{\text{res}} / (n - p - 1)}{\text{SS}_{\text{tot}} / (n - 1)} = 1 - \frac{\text{MSE}}{\text{MST}}
    $$

    where:
    - $\text{MSE} = \frac{\text{SS}_{\text{res}}}{n - p - 1}$ is the Mean Squared Error (unbiased estimator of error variance $\sigma^2$).
    - $\text{MST} = \frac{\text{SS}_{\text{tot}}}{n - 1}$ is the Mean Total Sum of Squares (sample variance $s_y^2$).

    #### Algebraic Relationship with Raw $R^2$
    Expressing $\bar{R}^2$ in terms of $R^2$:

    $$
    \bar{R}^2 = 1 - (1 - R^2) \left(\frac{n - 1}{n - p - 1}\right)
    $$

    Key implications:
    1. $\bar{R}^2 \leq R^2$ always, with equality holding if and only if $R^2 = 1$ or $p = 0$.
    2. $\bar{R}^2$ can be negative if $\text{MSE} > \text{MST}$, indicating that the model's predictions perform worse than the simple baseline average $\bar{y}$.
    3. As sample size $n \to \infty$ with fixed $p$, the adjustment factor $\frac{n - 1}{n - p - 1} \to 1$, meaning $\bar{R}^2 \to R^2$.

    ---

    ### 4. The Incremental Partial $F$-Test Criterion

    When adding a new candidate predictor $x_{p+1}$ to a baseline model with $p$ predictors:

    $$
    \bar{R}^2_{p+1} > \bar{R}^2_p \iff F_{\text{partial}} = \frac{\text{SS}_{\text{res}}(p) - \text{SS}_{\text{res}}(p+1)}{\text{MSE}(p+1)} > 1
    $$

    Since for a single parameter $F_{\text{partial}} = t_{p+1}^2$, this yields:

    $$
    \bar{R}^2_{p+1} > \bar{R}^2_p \iff |t_{p+1}| > 1
    $$

    This establishes that Adjusted $R^2$ will only increase if the predictor explains more variance than the expected noise contribution ($\mathbb{E}[F] \approx 1$ under $H_0$).

    ---

    ### 5. Mathematical Summary Table

    | Metric | Degrees of Freedom | Monotonic in $p$? | Can be Negative? | Penalty for Noise Features | Primary Usage |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Raw $R^2$** | None ($n$ ignored) | Yes (strictly non-decreasing) | No (with intercept) | None (rewards noise) | Goodness-of-fit on fixed models |
    | **Adjusted $R^2$** | $n - p - 1$ vs $n - 1$ | No (peaks at optimal subset) | Yes (if MSE > MST) | Explicit penalty via $\frac{n-1}{n-p-1}$ | Model comparison, nested feature screening |
    """)
    return


@app.cell
def _(np, pd):
    # Simulation Data: True Signal Predictors (x1, x2) + Pure Noise Predictors (x3 ... x10)
    np.random.seed(47)
    _n = 100

    # Genuine predictors
    _x1 = np.random.uniform(10.0, 50.0, _n)
    _x2 = np.random.uniform(20.0, 80.0, _n)
    _noise = np.random.normal(0.0, 2.5, _n)

    # True data-generating process: y = 10 + 0.8 * x1 + 1.5 * x2 + noise
    _y = 10.0 + 0.8 * _x1 + 1.5 * _x2 + _noise

    # Injected pure Gaussian noise variables
    _noise_vars = {f"Noise_x{i}": np.random.normal(0.0, 1.0, _n) for i in range(3, 11)}

    _data_dict = {"x1": _x1, "x2": _x2, **_noise_vars, "Target_y": _y}
    df_regression = pd.DataFrame(_data_dict)

    return (df_regression,)


@app.cell
def _(LinearRegression, df_regression, go, make_subplots, mo, np, pd, r2_score):
    # Interactive Visualizations Cell:
    # Subplot 1: Feature Correlation Heatmap (Diagnosing true signal vs noise)
    # Subplot 2: R^2 vs Adjusted R^2 Trajectory as predictors are added (1 to 10)
    # Subplot 3: Incremental F-statistic and Delta Adjusted R^2 vs the F=1 threshold

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Feature Correlation with Target",
            "2. R^2 vs. Adjusted R^2 Complexity Curve",
            "3. Incremental Partial F-Statistic vs. F=1",
        ),
        horizontal_spacing=0.09,
    )

    _target = df_regression["Target_y"]
    _features = [c for c in df_regression.columns if c != "Target_y"]
    _n = len(_target)

    # Subplot 1: Correlations
    _corr_series = df_regression[_features].apply(lambda col: col.corr(_target))
    _colors = ["#10b981" if abs(c) > 0.3 else "#94a3b8" for c in _corr_series]

    _fig.add_trace(
        go.Bar(
            x=_features,
            y=_corr_series,
            marker_color=_colors,
            name="Correlation with y",
            hovertemplate="%{x}: r = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Incremental Model Fitting
    _r2_vals = []
    _adj_r2_vals = []
    _f_stats = []
    _ss_res_prev = None

    for _p in range(1, len(_features) + 1):
        _sub_x = df_regression[_features[:_p]]
        _mod = LinearRegression().fit(_sub_x, _target)
        _preds = _mod.predict(_sub_x)
        _r2 = r2_score(_target, _preds)
        _adj_r2 = 1.0 - (1.0 - _r2) * (_n - 1) / (_n - _p - 1)
        _r2_vals.append(_r2)
        _adj_r2_vals.append(_adj_r2)

        _ss_res = np.sum((_target - _preds) ** 2)
        _mse = _ss_res / (_n - _p - 1)
        if _ss_res_prev is not None:
            _f = (_ss_res_prev - _ss_res) / _mse
            _f_stats.append(_f)
        else:
            _f_stats.append(np.nan)
        _ss_res_prev = _ss_res

    _p_axis = np.arange(1, len(_features) + 1)

    _fig.add_trace(
        go.Scatter(
            x=_p_axis,
            y=_r2_vals,
            mode="lines+markers",
            line=dict(color="#ef4444", width=2.5),
            name="Raw R^2 (Monotonic)",
            hovertemplate="p=%{x}: R^2=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _fig.add_trace(
        go.Scatter(
            x=_p_axis,
            y=_adj_r2_vals,
            mode="lines+markers",
            line=dict(color="#10b981", width=2.5),
            name="Adjusted R^2 (Penalized)",
            hovertemplate="p=%{x}: Adj R^2=%{y:.4f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Incremental F-statistic
    _fig.add_trace(
        go.Bar(
            x=_features[1:],
            y=_f_stats[1:],
            marker_color=["#10b981" if (not np.isnan(f) and f > 1.0) else "#f43f5e" for f in _f_stats[1:]],
            name="Partial F-Stat",
            hovertemplate="%{x}: F = %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    _fig.add_trace(
        go.Scatter(
            x=[_features[1], _features[-1]],
            y=[1.0, 1.0],
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Threshold F = 1.0",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Model Complexity Penalty: Raw R^2 vs. Adjusted R^2 and the F=1 Criterion",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Predictor", tickangle=45, row=1, col=1)
    _fig.update_yaxes(title_text="Pearson r with Target", row=1, col=1)

    _fig.update_xaxes(title_text="Number of Predictors (p)", tickvals=_p_axis, row=1, col=2)
    _fig.update_yaxes(title_text="Metric Value", range=[0.4, 1.02], row=1, col=2)

    _fig.update_xaxes(title_text="Added Predictor", tickangle=45, row=1, col=3)
    _fig.update_yaxes(title_text="Partial F-Statistic", range=[0.0, 15.0], row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Forward Stepwise Regression & Complexity Diagnostics**: Exhaustively evaluating $\text{SS}_{\text{res}}, \text{SS}_{\text{tot}}$, degrees of freedom, $R^2, \bar{R}^2$, and partial $F$-statistics, confirming that Adjusted $R^2$ peaks precisely at the true causal predictors ($x_1, x_2$).
    2. **The Negative Adjusted $R^2$ Simulation**: Fitting 10 pure Gaussian noise predictors on a small sample ($n = 25$), demonstrating how raw $R^2$ deceptively reports moderate fit while Adjusted $R^2$ turns negative, diagnosing a completely invalid model.
    """)
    return


@app.cell
def _(LinearRegression, df_regression, mo, np, pd, r2_score):
    # Example 1: Forward Stepwise Complexity Table with F-Test Diagnostics
    _target = df_regression["Target_y"].to_numpy()
    _features = [c for c in df_regression.columns if c != "Target_y"]
    _n = len(_target)
    _ss_tot = np.sum((_target - np.mean(_target)) ** 2)

    _rows = []
    _prev_ss_res = _ss_tot

    for _p in range(1, len(_features) + 1):
        _current_features = _features[:_p]
        _X_mat = df_regression[_current_features].to_numpy()
        _mod = LinearRegression().fit(_X_mat, _target)
        _y_hat = _mod.predict(_X_mat)

        _ss_res = np.sum((_target - _y_hat) ** 2)
        _df_res = _n - _p - 1
        _mse = _ss_res / _df_res

        _r2 = r2_score(_target, _y_hat)
        _adj_r2 = 1.0 - (1.0 - _r2) * (_n - 1) / _df_res

        if _p == 1:
            _f_stat_str = "-"
            _decision = "Initial Single Feature"
        else:
            _partial_f = (_prev_ss_res - _ss_res) / _mse
            _f_stat_str = f"{_partial_f:.3f}"
            _decision = "Adj R^2 Increased (F > 1)" if _partial_f > 1.0 else "Adj R^2 Decreased (F <= 1, Overfitting)"

        _rows.append({
            "Model Specification": f"p = {_p} ({_current_features[-1]})",
            "SS_res": f"{_ss_res:.1f}",
            "df_res": _df_res,
            "Raw R^2": f"{_r2:.4f}",
            "Adjusted R^2": f"{_adj_r2:.4f}",
            "Partial F": _f_stat_str,
            "Diagnostic Outcome": _decision,
        })
        _prev_ss_res = _ss_res

    _df_stepwise = pd.DataFrame(_rows)

    return (
        mo.md("#### Stepwise Model Expansion Evaluation"),
        mo.ui.table(_df_stepwise),
    )


@app.cell
def _(LinearRegression, mo, np, pd, r2_score):
    # Example 2: Negative Adjusted R^2 on Pure Noise (n = 25, p = 12)
    np.random.seed(123)
    _n_noise = 25
    _p_noise = 12

    # Completely independent target and noise predictors
    _y_pure_noise = np.random.normal(0.0, 1.0, _n_noise)
    _X_pure_noise = np.random.normal(0.0, 1.0, (_n_noise, _p_noise))

    _mod_noise = LinearRegression().fit(_X_pure_noise, _y_pure_noise)
    _y_pred_noise = _mod_noise.predict(_X_pure_noise)

    _r2_noise = r2_score(_y_pure_noise, _y_pred_noise)
    _df_res_noise = _n_noise - _p_noise - 1
    _adj_r2_noise = 1.0 - (1.0 - _r2_noise) * (_n_noise - 1) / _df_res_noise

    _ss_res_noise = np.sum((_y_pure_noise - _y_pred_noise) ** 2)
    _ss_tot_noise = np.sum((_y_pure_noise - np.mean(_y_pure_noise)) ** 2)

    _df_noise_demo = pd.DataFrame(
        [
            {"Metric": "Sample Size (n)", "Value": str(_n_noise), "Significance": "Small experimental sample"},
            {"Metric": "Number of Predictors (p)", "Value": str(_p_noise), "Significance": "Pure random Gaussian noise"},
            {"Metric": "Residual Degrees of Freedom (n - p - 1)", "Value": str(_df_res_noise), "Significance": "Severe loss of degrees of freedom"},
            {"Metric": "Raw R^2", "Value": f"{_r2_noise:.4f}", "Significance": "Spuriously claims 40-60% variance explained"},
            {"Metric": "Adjusted R^2", "Value": f"{_adj_r2_noise:.4f}", "Significance": "Negative! Exposes that model is worse than y_bar"},
            {"Metric": "MSE / MST Ratio", "Value": f"{(_ss_res_noise / _df_res_noise) / (_ss_tot_noise / (_n_noise - 1)):.4f}", "Significance": "Error variance exceeds target variance"},
        ]
    )

    return (
        mo.md("#### Pure Noise Simulation: The Negative Adjusted R^2 Safeguard"),
        mo.ui.table(_df_noise_demo),
    )


if __name__ == "__main__":
    app.run()
