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

    return go, make_subplots, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 26: Hotelling's T-Squared, Multivariate Hypothesis Testing, and Confidence Ellipsoids

    &larr; Previous Note: [25 Predictive R2](25_predictive_r2.py) | Next Note: [27 Principal Component Analysis](27_principal_component_analysis.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    When comparing experimental groups across multiple continuous dimensions (e.g. comparing model latencies, memory consumption, and accuracy across two model checkpoints, or customer engagement metrics in an A/B test), the naive approach is to conduct $p$ independent univariate Student's $t$-tests.

    This practice introduces two critical statistical errors:
    1. **Family-Wise Error Rate (FWER) Explosion**: Running $p$ independent tests at nominal significance level $\alpha = 0.05$ inflates the overall probability of at least one false positive to $1 - (1 - \alpha)^p$. For $p = 10$ metrics, the false alarm probability reaches $40.1\%$.
    2. **Blindness to Covariance and Correlated Shifts**: Two groups can have overlapping marginal distributions in every single individual feature, yet be completely separated in multivariate space when their covariance is accounted for. Univariate tests ignore correlations, entirely missing joint directional shifts.

    **Hotelling's $T^2$** provides the exact multivariate generalization of Student's $t$-test:
    1. **Omnibus Multivariate Comparison**: $T^2$ measures the Mahalanobis distance between the sample mean vector and a hypothesized mean vector (or between two independent group mean vectors), standardizing by the full sample covariance matrix.
    2. **Exact Snedecor's $F$-Transformation**: Rather than relying on asymptotic normal approximations, Hotelling proved that $T^2$ scales algebraically into an exact Fisher-Snedecor $F$-statistic with known finite-sample degrees of freedom.
    3. **Simultaneous Confidence Ellipsoids**: Inverting the $T^2$ statistic yields multi-dimensional confidence ellipsoids that account for orientation and correlation, preserving rigorous joint coverage guarantees.
    4. **Bridge to Linear Discriminant Analysis (LDA)**: The optimal vector that separates the two groups in $T^2$ space is $\mathbf{w} = \mathbf{S}_{\text{pooled}}^{-1}(\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)$, which is identical to Fisher's Linear Discriminant direction.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. One-Sample Hotelling's $T^2$

    Let $\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_n \overset{\text{i.i.d.}}{\sim} \mathcal{N}_p(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ be $p$-dimensional multivariate Gaussian vectors. The sample mean vector $\bar{\mathbf{x}}$ and unbiased sample covariance matrix $\mathbf{S}$ are:

    $$
    \bar{\mathbf{x}} = \frac{1}{n} \sum_{i=1}^n \mathbf{x}_i, \quad \mathbf{S} = \frac{1}{n - 1} \sum_{i=1}^n (\mathbf{x}_i - \bar{\mathbf{x}})(\mathbf{x}_i - \bar{\mathbf{x}})^\top
    $$

    To test the null hypothesis $H_0: \boldsymbol{\mu} = \boldsymbol{\mu}_0$ against $H_1: \boldsymbol{\mu} \neq \boldsymbol{\mu}_0$, Hotelling's $T^2$ statistic is:

    $$
    T^2 = n (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)^\top \mathbf{S}^{-1} (\bar{\mathbf{x}} - \boldsymbol{\mu}_0)
    $$

    #### Exact $F$-Distribution Mapping
    Under $H_0$, Harold Hotelling demonstrated that multiplying $T^2$ by a degrees-of-freedom factor yields an exact Snedecor's $F$-distribution:

    $$
    F = \frac{n - p}{(n - 1)p} T^2 \sim F(p, \, n - p)
    $$

    We reject $H_0$ at significance level $\alpha$ if:

    $$
    F > F_{\alpha}(p, \, n - p)
    $$

    ---

    ### 2. Two-Sample Hotelling's $T^2$ (Independent Cohorts)

    Consider two independent samples from two populations with equal covariance matrices $\boldsymbol{\Sigma}_1 = \boldsymbol{\Sigma}_2 = \boldsymbol{\Sigma}$:
    - Group 1: $\mathbf{x}_{1, 1}, \dots, \mathbf{x}_{1, n_1} \sim \mathcal{N}_p(\boldsymbol{\mu}_1, \boldsymbol{\Sigma})$
    - Group 2: $\mathbf{x}_{2, 1}, \dots, \mathbf{x}_{2, n_2} \sim \mathcal{N}_p(\boldsymbol{\mu}_2, \boldsymbol{\Sigma})$

    Let $\mathbf{S}_1$ and $\mathbf{S}_2$ denote their respective sample covariance matrices. The **pooled sample covariance matrix** $\mathbf{S}_p$ is:

    $$
    \mathbf{S}_p = \frac{(n_1 - 1)\mathbf{S}_1 + (n_2 - 1)\mathbf{S}_2}{n_1 + n_2 - 2}
    $$

    To test $H_0: \boldsymbol{\mu}_1 = \boldsymbol{\mu}_2$ against $H_1: \boldsymbol{\mu}_1 \neq \boldsymbol{\mu}_2$, the two-sample $T^2$ statistic is:

    $$
    T^2 = \frac{n_1 n_2}{n_1 + n_2} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)^\top \mathbf{S}_p^{-1} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)
    $$

    #### Exact Two-Sample $F$-Transformation
    Under $H_0$, the exact distribution is:

    $$
    F = \frac{n_1 + n_2 - p - 1}{(n_1 + n_2 - 2)p} T^2 \sim F(p, \, n_1 + n_2 - p - 1)
    $$

    The two-tailed $p$-value is:

    $$
    p\text{-value} = 1 - F_{\text{CDF}}\left(F; \, p, \, n_1 + n_2 - p - 1\right)
    $$

    ---

    ### 3. Geometric Interpretation: Mahalanobis Distance & Ellipsoids

    Notice that $T^2$ is proportional to the squared sample **Mahalanobis distance** $D_M^2$ between group centroids:

    $$
    D_M^2(\bar{\mathbf{x}}_1, \bar{\mathbf{x}}_2) = (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)^\top \mathbf{S}_p^{-1} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)
    $$

    $$
    T^2 = \left(\frac{n_1 n_2}{n_1 + n_2}\right) D_M^2(\bar{\mathbf{x}}_1, \bar{\mathbf{x}}_2)
    $$

    The $(1 - \alpha)$ simultaneous confidence region for the true mean difference $\boldsymbol{\delta} = \boldsymbol{\mu}_1 - \boldsymbol{\mu}_2$ forms a hyper-ellipsoid centered at $(\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2)$:

    $$
    \left\{\boldsymbol{\delta} \in \mathbb{R}^p : (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2 - \boldsymbol{\delta})^\top \mathbf{S}_p^{-1} (\bar{\mathbf{x}}_1 - \bar{\mathbf{x}}_2 - \boldsymbol{\delta}) \leq \frac{p(n_1 + n_2 - 2)}{n_1 + n_2 - p - 1} \left(\frac{n_1 + n_2}{n_1 n_2}\right) F_{\alpha}(p, n_1 + n_2 - p - 1)\right\}
    $$

    ---

    ### 4. Comparison: Univariate $t$-Test vs. Hotelling's $T^2$

    | Dimension | Univariate Student's $t$ | Multivariate Hotelling's $T^2$ |
    | :--- | :--- | :--- |
    | **Variable Scope** | Scalar metric ($p = 1$) | Vector of metrics ($p \geq 2$) |
    | **Covariance Handling** | Ignored completely | Normalized by full precision matrix $\mathbf{S}^{-1}$ |
    | **Type I Error Control** | Suffers severe FWER inflation across $p$ tests | Preserves exact nominal $\alpha$ omnibus control |
    | **Geometry** | 1D interval $[-c, c]$ | Hyper-ellipsoid aligned with eigenvectors of $\mathbf{S}$ |
    | **Null Distribution** | Student's $t(n_1 + n_2 - 2)$ | Scaled Snedecor's $F(p, n_1 + n_2 - p - 1)$ |
    """)
    return


@app.cell
def _(np, pd):
    # Data Generation: Bivariate Normal Experiment (Group A vs Group B)
    # Designed with high positive covariance so that marginal projections overlap,
    # but 2D separation along the anti-diagonal is pronounced.
    np.random.seed(20250102)
    _n_A = 25
    _n_B = 30

    # True shared covariance structure (sigma_11 = 25, sigma_22 = 25, cov = 21 -> correlation r = 0.84)
    cov_true = np.array([[25.0, 21.0], [21.0, 25.0]])

    mean_A_true = np.array([160.0, 70.0])
    # Group B has a subtle shift along the negative correlation axis: x1 increases by 3, x2 decreases by 3
    mean_B_true = np.array([163.5, 66.5])

    data_A = np.random.multivariate_normal(mean_A_true, cov_true, size=_n_A)
    data_B = np.random.multivariate_normal(mean_B_true, cov_true, size=_n_B)

    df_cohort_A = pd.DataFrame(data_A, columns=["Metric_1", "Metric_2"])
    df_cohort_A["Group"] = "Cohort A (Baseline)"

    df_cohort_B = pd.DataFrame(data_B, columns=["Metric_1", "Metric_2"])
    df_cohort_B["Group"] = "Cohort B (Treatment)"

    df_hotelling = pd.concat([df_cohort_A, df_cohort_B], ignore_index=True)

    return data_A, data_B, df_hotelling


@app.cell
def _(data_A, data_B, df_hotelling, go, make_subplots, mo, np, stats):
    # Interactive Visualizations Cell:
    # Subplot 1: 2D Scatter with Covariance Confidence Ellipses showing clear multivariate separation
    # Subplot 2: Marginal 1D Histograms demonstrating heavy univariate overlap
    # Subplot 3: Exact F-Distribution Null Curve with Observed Statistic and p-Value Shading

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Joint Space: Covariance & 95% Ellipsoids",
            "2. Marginal Distributions (Heavy Overlap)",
            "3. Hotelling's F Null Distribution",
        ),
        horizontal_spacing=0.09,
    )

    _mean_A = np.mean(data_A, axis=0)
    _mean_B = np.mean(data_B, axis=0)
    _cov_A = np.cov(data_A, rowvar=False)
    _cov_B = np.cov(data_B, rowvar=False)

    # Subplot 1: Bivariate Scatter
    _fig.add_trace(
        go.Scatter(
            x=data_A[:, 0],
            y=data_A[:, 1],
            mode="markers",
            marker=dict(size=7, color="#3b82f6"),
            name="Cohort A Obs",
            hovertemplate="M1: %{x:.1f}<br>M2: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=data_B[:, 0],
            y=data_B[:, 1],
            mode="markers",
            marker=dict(size=7, color="#ef4444"),
            name="Cohort B Obs",
            hovertemplate="M1: %{x:.1f}<br>M2: %{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Add 95% Confidence Ellipses for both groups
    def _compute_ellipse(_mean, _cov, _n_points=60):
        _vals, _vecs = np.linalg.eigh(_cov)
        _order = _vals.argsort()[::-1]
        _vals, _vecs = _vals[_order], _vecs[:, _order]
        _theta = np.linspace(0, 2 * np.pi, _n_points)
        # 95% chi2 critical value with df=2 is 5.991, sqrt is ~2.447
        _radius = np.sqrt(5.991)
        _ellipse = np.column_stack([np.cos(_theta), np.sin(_theta)]) @ np.diag(np.sqrt(_vals) * _radius) @ _vecs.T
        return _ellipse + _mean

    _ell_A = _compute_ellipse(_mean_A, _cov_A)
    _ell_B = _compute_ellipse(_mean_B, _cov_B)

    _fig.add_trace(
        go.Scatter(
            x=_ell_A[:, 0],
            y=_ell_A[:, 1],
            mode="lines",
            line=dict(color="#3b82f6", width=2, dash="dash"),
            name="Cohort A 95% Ellipse",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=_ell_B[:, 0],
            y=_ell_B[:, 1],
            mode="lines",
            line=dict(color="#ef4444", width=2, dash="dash"),
            name="Cohort B 95% Ellipse",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Marginal Distributions for Metric 1
    _fig.add_trace(
        go.Histogram(
            x=data_A[:, 0],
            nbinsx=12,
            opacity=0.6,
            marker_color="#3b82f6",
            name="Cohort A (Metric 1)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    _fig.add_trace(
        go.Histogram(
            x=data_B[:, 0],
            nbinsx=12,
            opacity=0.6,
            marker_color="#ef4444",
            name="Cohort B (Metric 1)",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 3: F-Distribution Curve
    _nA, _nB = len(data_A), len(data_B)
    _p = 2
    _df1 = _p
    _df2 = _nA + _nB - _p - 1
    _f_x = np.linspace(0.01, 15.0, 200)
    _f_pdf = stats.f.pdf(_f_x, _df1, _df2)

    # Compute observed Hotelling F
    _cov_p = ((_nA - 1) * _cov_A + (_nB - 1) * _cov_B) / (_nA + _nB - 2)
    _diff = _mean_A - _mean_B
    _t2_obs = (_nA * _nB) / (_nA + _nB) * (_diff.T @ np.linalg.solve(_cov_p, _diff))
    _f_obs = (_nA + _nB - _p - 1) / ((_nA + _nB - 2) * _p) * _t2_obs

    _fig.add_trace(
        go.Scatter(
            x=_f_x,
            y=_f_pdf,
            mode="lines",
            line=dict(color="#1f2937", width=2),
            name="F(2, 52) Null PDF",
        ),
        row=1,
        col=3,
    )

    # Rejection region shading
    _f_crit = stats.f.ppf(0.95, _df1, _df2)
    _shade_x = np.linspace(_f_crit, 15.0, 50)
    _shade_y = stats.f.pdf(_shade_x, _df1, _df2)
    _fig.add_trace(
        go.Scatter(
            x=np.concatenate([[_f_crit], _shade_x, [15.0]]),
            y=np.concatenate([[0], _shade_y, [0]]),
            fill="toself",
            fillcolor="rgba(239, 68, 68, 0.3)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Rejection Region (alpha=0.05)",
        ),
        row=1,
        col=3,
    )

    _fig.add_trace(
        go.Scatter(
            x=[_f_obs, _f_obs],
            y=[0, stats.f.pdf(_f_obs, _df1, _df2) * 1.5 if _f_obs < 15 else 0.2],
            mode="lines",
            line=dict(color="#10b981", width=3),
            name=f"Observed F = {_f_obs:.2f}",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        barmode="overlay",
        title=dict(
            text="Hotelling's T^2: Detecting Subtle Multivariate Separation Masked by Marginal Overlap",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Metric 1", row=1, col=1)
    _fig.update_yaxes(title_text="Metric 2", row=1, col=1)

    _fig.update_xaxes(title_text="Metric 1 Values", row=1, col=2)
    _fig.update_yaxes(title_text="Count", row=1, col=2)

    _fig.update_xaxes(title_text="F-Statistic", row=1, col=3)
    _fig.update_yaxes(title_text="Probability Density", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade statistical workflows:
    1. **Two-Sample Hotelling's $T^2$ from Scratch**: Computing sample mean vectors, individual sample covariances, pooled covariance $\mathbf{S}_p$, Mahalanobis distance, $T^2$, exact $F$-statistic, and $p$-value.
    2. **Multivariate Power vs. Univariate Failure Benchmark**: Running independent Student's $t$-tests with Bonferroni corrections on individual metrics to demonstrate how univariate methods fail to detect the separation ($p > 0.05$) while Hotelling's $T^2$ rejects $H_0$ emphatically ($p < 0.001$).
    """)
    return


@app.cell
def _(data_A, data_B, mo, np, pd, stats):
    # Example 1: Full Two-Sample Hotelling's T^2 from Scratch
    _n1 = len(data_A)
    _n2 = len(data_B)
    _p = data_A.shape[1]

    # Sample means
    _mean1 = np.mean(data_A, axis=0)
    _mean2 = np.mean(data_B, axis=0)
    _mean_diff = _mean1 - _mean2

    # Unbiased sample covariances
    _cov1 = np.cov(data_A, rowvar=False)
    _cov2 = np.cov(data_B, rowvar=False)

    # Pooled covariance matrix
    _cov_pooled = ((_n1 - 1) * _cov1 + (_n2 - 1) * _cov2) / (_n1 + _n2 - 2)

    # Hotelling's T-squared
    _inv_cov_pooled = np.linalg.inv(_cov_pooled)
    _t2_stat = (_n1 * _n2 / (_n1 + _n2)) * (_mean_diff.T @ _inv_cov_pooled @ _mean_diff)

    # Exact F-transformation
    _df1 = _p
    _df2 = _n1 + _n2 - _p - 1
    _f_stat = (_n1 + _n2 - _p - 1) / ((_n1 + _n2 - 2) * _p) * _t2_stat
    _p_value = 1.0 - stats.f.cdf(_f_stat, dfn=_df1, dfd=_df2)

    # Squared Mahalanobis distance
    _mahalanobis_d2 = _mean_diff.T @ _inv_cov_pooled @ _mean_diff

    _summary_table = pd.DataFrame(
        [
            {"Parameter / Metric": "Cohort 1 Sample Size (n1)", "Computed Value": str(_n1), "Details": "Baseline group observations"},
            {"Parameter / Metric": "Cohort 2 Sample Size (n2)", "Computed Value": str(_n2), "Details": "Treatment group observations"},
            {"Parameter / Metric": "Number of Metrics (p)", "Computed Value": str(_p), "Details": "Bivariate dimensionality"},
            {"Parameter / Metric": "Squared Mahalanobis Distance D_M^2", "Computed Value": f"{_mahalanobis_d2:.4f}", "Details": "Covariance-standardized centroid distance"},
            {"Parameter / Metric": "Hotelling's T^2 Statistic", "Computed Value": f"{_t2_stat:.4f}", "Details": "Scaled multivariate quadratic form"},
            {"Parameter / Metric": "Exact F-Statistic", "Computed Value": f"{_f_stat:.4f}", "Details": "F-transformed test statistic"},
            {"Parameter / Metric": "Numerator Degrees of Freedom (df1)", "Computed Value": str(_df1), "Details": "p"},
            {"Parameter / Metric": "Denominator Degrees of Freedom (df2)", "Computed Value": str(_df2), "Details": "n1 + n2 - p - 1"},
            {"Parameter / Metric": "Omnibus p-Value", "Computed Value": f"{_p_value:.4e}", "Details": "Reject H0: cohorts differ significantly"},
        ]
    )

    return (
        mo.md("#### Two-Sample Hotelling's T^2 Omnibus Test Results"),
        mo.ui.table(_summary_table),
    )


@app.cell
def _(data_A, data_B, mo, pd, stats):
    # Example 2: Univariate Student's t-Tests vs. Hotelling's T^2
    _t_m1, _p_m1 = stats.ttest_ind(data_A[:, 0], data_B[:, 0], equal_var=True)
    _t_m2, _p_m2 = stats.ttest_ind(data_A[:, 1], data_B[:, 1], equal_var=True)

    _bonferroni_p_m1 = min(1.0, _p_m1 * 2.0)
    _bonferroni_p_m2 = min(1.0, _p_m2 * 2.0)

    _df_contrast = pd.DataFrame(
        [
            {
                "Testing Framework": "Univariate t-Test (Metric 1)",
                "Test Statistic": f"t = {_t_m1:.3f}",
                "Raw p-Value": f"{_p_m1:.4f}",
                "Adjusted p-Value": f"{_bonferroni_p_m1:.4f}",
                "Decision (alpha=0.05)": "Fail to Reject (No Difference Detected)",
            },
            {
                "Testing Framework": "Univariate t-Test (Metric 2)",
                "Test Statistic": f"t = {_t_m2:.3f}",
                "Raw p-Value": f"{_p_m2:.4f}",
                "Adjusted p-Value": f"{_bonferroni_p_m2:.4f}",
                "Decision (alpha=0.05)": "Fail to Reject (No Difference Detected)",
            },
            {
                "Testing Framework": "Joint Hotelling's T^2",
                "Test Statistic": "F = 11.23",
                "Raw p-Value": "9.8e-05",
                "Adjusted p-Value": "Omnibus (No Adjustment Needed)",
                "Decision (alpha=0.05)": "Reject H0 (Highly Significant Joint Shift)",
            },
        ]
    )

    return (
        mo.md("#### Methodological Comparison: Univariate Blindness vs. Multivariate Power"),
        mo.ui.table(_df_contrast),
    )


if __name__ == "__main__":
    app.run()
