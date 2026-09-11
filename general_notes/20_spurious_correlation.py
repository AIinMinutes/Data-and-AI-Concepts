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
    # Note 20: Spurious Correlation, Confounding, and Partial Correlation

    &larr; Previous Note: [19 Kendall Tau-b](19_kendalltaub.py) | Next Note: [21 Kruskal-Wallis](21_kruskal_wallis.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    The fundamental tenet of empirical science and machine learning is that **correlation does not imply causation**. Two variables $X$ and $Y$ may exhibit a statistically significant correlation ($r > 0.90, p < 10^{-15}$) despite having no direct causal link whatsoever.

    In machine learning, AI alignment, and causal inference, failing to distinguish between genuine association and spurious correlation leads to severe vulnerabilities:
    1. **Spurious Shortcuts in Deep Learning**: Computer vision models trained to detect pneumonia often learn to identify hospital ward markers or scanner metadata rather than lung pathology. Language models trained on web corpora associate demographic tokens with specific professions due to historical reporting biases. When deployed in new hospitals or environments, these models suffer catastrophic generalization failure.
    2. **Confounder Bias in Observational Data**: In observational studies (e.g. ad spend vs revenue, medical treatment vs patient recovery, hours studied vs exam scores), unobserved or unmodelled third variables (economic conditions, disease severity, student prior ability) confound both the predictor and the outcome.
    3. **Simpson's Paradox**: When a population consists of distinct subgroups, the aggregate correlation across the pooled sample can point in the exact opposite direction of the true relationship within every single subgroup. A drug can appear harmful in aggregate while being life-saving within every clinical severity stratum.
    4. **Partial Correlation as Orthogonal Deconfounding**: Partial correlation measures the linear association between two variables $X$ and $Y$ after algebraically removing the linear effect of one or more confounding variables $Z$. In linear models and Gaussian Graphical Models (GGMs), zero partial correlation implies conditional independence ($X \perp Y \mid Z$).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. The Common Cause Model (Fork DAG)

    Consider Reichenbach's Common Cause Principle. In a directed acyclic graph (DAG), when a latent or observed variable $Z$ exerts a direct causal influence on both $X$ and $Y$:

    $$
    X \leftarrow Z \rightarrow Y
    $$

    We can write the structural equations as:

    $$
    X = \alpha_X Z + \epsilon_X, \quad \epsilon_X \sim \mathcal{N}(0, \sigma_X^2)
    $$

    $$
    Y = \alpha_Y Z + \epsilon_Y, \quad \epsilon_Y \sim \mathcal{N}(0, \sigma_Y^2)
    $$

    where the disturbance terms $\epsilon_X$ and $\epsilon_Y$ are mutually independent of each other and of $Z$.

    The covariance between $X$ and $Y$ expands to:

    $$
    \operatorname{Cov}(X, Y) = \mathbb{E}[(X - \mu_X)(Y - \mu_Y)] = \alpha_X \alpha_Y \operatorname{Var}(Z) + \operatorname{Cov}(\epsilon_X, \epsilon_Y)
    $$

    Because $\operatorname{Cov}(\epsilon_X, \epsilon_Y) = 0$, the entire covariance is driven exclusively by the variance of $Z$:

    $$
    \operatorname{Cov}(X, Y) = \alpha_X \alpha_Y \operatorname{Var}(Z) \neq 0
    $$

    Thus, the sample correlation $r_{XY}$ can be arbitrarily close to $+1$ or $-1$, despite there being zero direct causal transmission from $X$ to $Y$.

    ---

    ### 2. First-Order Partial Correlation

    The partial correlation between $X$ and $Y$ controlling for a single confounder $Z$, denoted $r_{XY \cdot Z}$, is defined algebraically in terms of bivariate Pearson correlations:

    $$
    r_{XY \cdot Z} = \frac{r_{XY} - r_{XZ} r_{YZ}}{\sqrt{(1 - r_{XZ}^2)(1 - r_{YZ}^2)}}
    $$

    #### Orthogonal Residual Formulation (Frisch-Waugh-Lovell Theorem)
    Equivalently, partial correlation is the Pearson correlation between the residuals of $X$ and $Y$ after linearly regressing each onto $Z$:

    $$
    \hat{X} = \hat{\beta}_{XZ} Z, \quad \tilde{X} = X - \hat{X}
    $$

    $$
    \hat{Y} = \hat{\beta}_{YZ} Z, \quad \tilde{Y} = Y - \hat{Y}
    $$

    $$
    r_{XY \cdot Z} = \operatorname{corr}(\tilde{X}, \tilde{Y}) = \frac{\tilde{X}^\top \tilde{Y}}{\|\tilde{X}\|_2 \|\tilde{Y}\|_2}
    $$

    Geometrically, $\tilde{X}$ and $\tilde{Y}$ are the projections of vectors $X$ and $Y$ onto the orthogonal complement of the subspace spanned by $Z$.

    ---

    ### 3. Multivariate Partial Correlation via the Precision Matrix

    For a random vector $\mathbf{X} = [X_1, X_2, \dots, X_p]^\top \sim \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the partial correlation between any two variables $X_i$ and $X_j$ controlling for all remaining $p - 2$ variables $\mathbf{X}_{\setminus \{i, j\}}$ is computed from the **precision matrix** (inverse covariance matrix):

    $$
    \boldsymbol{\Theta} = \boldsymbol{\Sigma}^{-1}
    $$

    $$
    r_{ij \cdot \text{rest}} = -\frac{\theta_{ij}}{\sqrt{\theta_{ii} \theta_{jj}}}
    $$

    In Gaussian Graphical Models (GGMs), a zero entry $\theta_{ij} = 0$ corresponds to:

    $$
    r_{ij \cdot \text{rest}} = 0 \iff X_i \perp X_j \mid \mathbf{X}_{\setminus \{i, j\}}
    $$

    This establishes the direct edge structure in Markov Random Fields and sparse graphical Lasso algorithms.

    ---

    ### 4. Hypothesis Testing and Fisher's Z-Transformation

    Under the null hypothesis $H_0: \rho_{XY \cdot \mathbf{Z}} = 0$, the sample partial correlation test statistic follows a Student's $t$-distribution with $n - k - 2$ degrees of freedom, where $k$ is the number of conditioned variables ($k = 1$ for single variable $Z$):

    $$
    t = r_{XY \cdot Z} \sqrt{\frac{n - k - 2}{1 - r_{XY \cdot Z}^2}} \sim t(n - k - 2)
    $$

    Alternatively, Fisher's hyperbolic arctangent transformation yields an asymptotic standard normal variable:

    $$
    z = \operatorname{arctanh}(r_{XY \cdot Z}) = \frac{1}{2} \ln \left(\frac{1 + r_{XY \cdot Z}}{1 - r_{XY \cdot Z}}\right) \sim \mathcal{N}\left(0, \frac{1}{n - k - 3}\right)
    $$

    ---

    ### 5. Simpson's Paradox

    Simpson's paradox arises when a marginal trend across pooled data reverses when conditioning on a discrete stratifying confounder:

    $$
    \frac{\partial \mathbb{E}[Y \mid X]}{\partial X} > 0 \quad \text{while} \quad \frac{\partial \mathbb{E}[Y \mid X, Z = z]}{\partial X} < 0 \quad \forall z
    $$

    This occurs when the distribution of the confounder $Z$ differs substantially across values of $X$. Conditioning on $Z$ blocks the backdoor path $X \leftarrow Z \rightarrow Y$, isolating the true causal effect.
    """)
    return


@app.cell
def _(np, pd):
    # Data Generation Cell: Reproducible Confounding Experiment
    np.random.seed(42)
    _n = 50

    # Latent Confounder Z (e.g. Firm Scale / User Activity Index)
    z_latent = np.random.normal(loc=10.0, scale=2.5, size=_n)

    # Observed Predictor X and Outcome Y generated from common cause Z + independent noise
    x_obs = 10.0 * z_latent + np.random.normal(loc=0.0, scale=8.0, size=_n)
    y_obs = 20.0 * z_latent + np.random.normal(loc=1.0, scale=8.0, size=_n)

    df_confounded = pd.DataFrame({"X": np.round(x_obs, 2), "Y": np.round(y_obs, 2), "Z": np.round(z_latent, 2)})

    # Simpson's Paradox Dataset (3 Demographic Cohorts)
    np.random.seed(101)
    _n_group = 30
    _groups = ["Group A (Mild)", "Group B (Moderate)", "Group C (Severe)"]
    _simpson_records = []

    for _idx, _grp in enumerate(_groups):
        _dose_shift = _idx * 4.0
        _base_health = 80.0 - _idx * 25.0
        _dosage = np.random.uniform(2.0, 8.0, _n_group) + _dose_shift
        # True within-group causal effect is negative (-2.5 health drop per dose)
        _recovery = _base_health - 2.5 * _dosage + np.random.normal(0.0, 3.0, _n_group)
        for _d, _r in zip(_dosage, _recovery):
            _simpson_records.append({"Group": _grp, "Dosage": np.round(_d, 2), "RecoveryScore": np.round(_r, 2)})

    df_simpson = pd.DataFrame(_simpson_records)

    return df_confounded, df_simpson


@app.cell
def _(df_confounded, df_simpson, go, make_subplots, mo, np, stats):
    # Interactive Visualizations Cell:
    # Subplot 1: Raw Confounded Scatter (X vs Y color-coded by continuous Z)
    # Subplot 2: Deconfounded Residual Scatter (Residual X vs Residual Y)
    # Subplot 3: Simpson's Paradox (Aggregate positive vs Group-level negative trendlines)

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Raw Confounded Scatter (Color = Z)",
            "2. Deconfounded Residuals (Z Removed)",
            "3. Simpson's Paradox: Aggregate vs Groups",
        ),
        horizontal_spacing=0.08,
    )

    _x = df_confounded["X"].to_numpy()
    _y = df_confounded["Y"].to_numpy()
    _z = df_confounded["Z"].to_numpy()

    # Subplot 1: Confounded
    _fig.add_trace(
        go.Scatter(
            x=_x,
            y=_y,
            mode="markers",
            marker=dict(
                size=9,
                color=_z,
                colorscale="Viridis",
                colorbar=dict(title="Confounder Z", x=0.28, len=0.7),
                showscale=True,
                line=dict(width=1, color="#333333"),
            ),
            name="Confounded Obs",
            hovertemplate="X: %{x:.1f}<br>Y: %{y:.1f}<br>Z: %{marker.color:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    _slope_raw, _inter_raw, _, _, _ = stats.linregress(_x, _y)
    _line_x = np.linspace(_x.min(), _x.max(), 50)
    _fig.add_trace(
        go.Scatter(
            x=_line_x,
            y=_slope_raw * _line_x + _inter_raw,
            mode="lines",
            line=dict(color="#ef4444", width=2.5, dash="solid"),
            name="Raw Fit (Spurious)",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Residuals
    _slope_xz, _inter_xz, _, _, _ = stats.linregress(_z, _x)
    _slope_yz, _inter_yz, _, _, _ = stats.linregress(_z, _y)
    _res_x = _x - (_slope_xz * _z + _inter_xz)
    _res_y = _y - (_slope_yz * _z + _inter_yz)

    _fig.add_trace(
        go.Scatter(
            x=_res_x,
            y=_res_y,
            mode="markers",
            marker=dict(size=9, color="#6366f1", line=dict(width=1, color="#312e81")),
            name="Orthogonal Residuals",
            hovertemplate="Res X: %{x:.2f}<br>Res Y: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    _slope_res, _inter_res, _, _, _ = stats.linregress(_res_x, _res_y)
    _line_rx = np.linspace(_res_x.min(), _res_x.max(), 50)
    _fig.add_trace(
        go.Scatter(
            x=_line_rx,
            y=_slope_res * _line_rx + _inter_res,
            mode="lines",
            line=dict(color="#10b981", width=2.5, dash="dash"),
            name="Residual Fit (Near 0)",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Simpson's Paradox
    _group_colors = {"Group A (Mild)": "#3b82f6", "Group B (Moderate)": "#10b981", "Group C (Severe)": "#f59e0b"}
    for _grp, _col in _group_colors.items():
        _sub = df_simpson[df_simpson["Group"] == _grp]
        _fig.add_trace(
            go.Scatter(
                x=_sub["Dosage"],
                y=_sub["RecoveryScore"],
                mode="markers",
                marker=dict(size=8, color=_col),
                name=_grp,
                hovertemplate="Dosage: %{x:.1f}<br>Recovery: %{y:.1f}<extra></extra>",
            ),
            row=1,
            col=3,
        )
        _sl, _it, _, _, _ = stats.linregress(_sub["Dosage"], _sub["RecoveryScore"])
        _lx = np.linspace(_sub["Dosage"].min(), _sub["Dosage"].max(), 30)
        _fig.add_trace(
            go.Scatter(
                x=_lx,
                y=_sl * _lx + _it,
                mode="lines",
                line=dict(color=_col, width=2),
                showlegend=False,
            ),
            row=1,
            col=3,
        )

    # Aggregate fit for Simpson's Paradox
    _sl_agg, _it_agg, _, _, _ = stats.linregress(df_simpson["Dosage"], df_simpson["RecoveryScore"])
    _lx_agg = np.linspace(df_simpson["Dosage"].min(), df_simpson["Dosage"].max(), 50)
    _fig.add_trace(
        go.Scatter(
            x=_lx_agg,
            y=_sl_agg * _lx_agg + _it_agg,
            mode="lines",
            line=dict(color="#dc2626", width=3, dash="dash"),
            name="Aggregate Paradox Fit",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Spurious Association, Confounding Removal, and Simpson's Paradox",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=70, b=80),
    )

    _fig.update_xaxes(title_text="Observed X", row=1, col=1)
    _fig.update_yaxes(title_text="Observed Y", row=1, col=1)

    _fig.update_xaxes(title_text="Residual X (e_X|Z)", row=1, col=2)
    _fig.update_yaxes(title_text="Residual Y (e_Y|Z)", row=1, col=2)

    _fig.update_xaxes(title_text="Dosage Level", row=1, col=3)
    _fig.update_yaxes(title_text="Recovery Score", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    We demonstrate two complete, mathematically rigorous workflows:
    1. **Deconfounding via Three Equivalent Approaches**:
       - Analytical partial correlation formula
       - Frisch-Waugh-Lovell linear regression residual projection
       - Precision matrix inversion ($\boldsymbol{\Theta} = \boldsymbol{\Sigma}^{-1}$)
    2. **Hypothesis Testing and Confounder Diagnostic**:
       - Evaluating raw correlation vs. partial correlation
       - Degrees of freedom adjustment, $t$-statistics, and two-sided $p$-values
       - Simpson's paradox quantification across segmented cohorts
    """)
    return


@app.cell
def _(df_confounded, mo, np, pd, stats):
    # Example 1: Three Distinct Mathematical Formulations of Partial Correlation
    _x = df_confounded["X"].to_numpy()
    _y = df_confounded["Y"].to_numpy()
    _z = df_confounded["Z"].to_numpy()
    _n = len(_x)

    # 1. Bivariate Correlation Matrix
    _r_xy, _p_xy = stats.pearsonr(_x, _y)
    _r_xz, _ = stats.pearsonr(_x, _z)
    _r_yz, _ = stats.pearsonr(_y, _z)

    # Analytical Formula
    _r_part_formula = (_r_xy - _r_xz * _r_yz) / np.sqrt((1.0 - _r_xz**2) * (1.0 - _r_yz**2))

    # 2. Frisch-Waugh-Lovell Residual Regression
    _beta_xz, _alpha_xz, _, _, _ = stats.linregress(_z, _x)
    _beta_yz, _alpha_yz, _, _, _ = stats.linregress(_z, _y)
    _res_x = _x - (_beta_xz * _z + _alpha_xz)
    _res_y = _y - (_beta_yz * _z + _alpha_yz)
    _r_part_fwl, _ = stats.pearsonr(_res_x, _res_y)

    # 3. Precision Matrix Inversion
    _data_matrix = np.column_stack([_x, _y, _z])
    _cov_matrix = np.cov(_data_matrix, rowvar=False)
    _precision = np.linalg.inv(_cov_matrix)
    _r_part_precision = -_precision[0, 1] / np.sqrt(_precision[0, 0] * _precision[1, 1])

    # Hypothesis Testing (t-statistic and p-value for k=1 conditioned variable)
    _k = 1
    _df = _n - _k - 2
    _t_stat = _r_part_formula * np.sqrt(_df / (1.0 - _r_part_formula**2))
    _p_val = 2.0 * (1.0 - stats.t.cdf(np.abs(_t_stat), df=_df))

    _comparison_table = pd.DataFrame(
        [
            {"Method / Metric": "Raw Pearson Correlation r(X, Y)", "Estimated Value": f"{_r_xy:.4f}", "P-Value": f"{_p_xy:.2e}", "Interpretation": "Spurious high correlation caused by Z"},
            {"Method / Metric": "Analytical Formula r(XY.Z)", "Estimated Value": f"{_r_part_formula:.4f}", "P-Value": f"{_p_val:.4f}", "Interpretation": "True conditional association after removing Z"},
            {"Method / Metric": "Frisch-Waugh-Lovell Residuals", "Estimated Value": f"{_r_part_fwl:.4f}", "P-Value": f"{_p_val:.4f}", "Interpretation": "Pearson correlation of orthogonal residuals"},
            {"Method / Metric": "Precision Matrix (Theta_01)", "Estimated Value": f"{_r_part_precision:.4f}", "P-Value": f"{_p_val:.4f}", "Interpretation": "Normalized off-diagonal precision entry"},
            {"Method / Metric": "Degrees of Freedom (n - k - 2)", "Estimated Value": str(_df), "P-Value": "-", "Interpretation": "Adjusted for conditioning on 1 variable"},
            {"Method / Metric": "Student's t-Statistic", "Estimated Value": f"{_t_stat:.4f}", "P-Value": f"{_p_val:.4f}", "Interpretation": "Fail to reject H0: true association is ~ 0"},
        ]
    )

    return (
        mo.md("#### Methodological Verification: Raw vs. Deconfounded Association"),
        mo.ui.table(_comparison_table),
    )


@app.cell
def _(df_simpson, mo, pd, stats):
    # Example 2: Simpson's Paradox Quantification
    # Demonstrate aggregate slope vs subgroup slopes
    _groups = df_simpson["Group"].unique()
    _simpson_metrics = []

    # Aggregate metric
    _agg_slope, _, _agg_r, _agg_p, _ = stats.linregress(df_simpson["Dosage"], df_simpson["RecoveryScore"])
    _simpson_metrics.append({
        "Stratum": "Pooled Sample (Aggregate)",
        "Sample Size (n)": len(df_simpson),
        "Slope (Beta)": f"{_agg_slope:+.3f}",
        "Pearson r": f"{_agg_r:+.4f}",
        "p-Value": f"{_agg_p:.2e}",
        "Diagnosis": "Misleading Positive Trend (Confounded)",
    })

    # Within each subgroup
    for _grp in _groups:
        _sub = df_simpson[df_simpson["Group"] == _grp]
        _sl, _, _r, _p, _ = stats.linregress(_sub["Dosage"], _sub["RecoveryScore"])
        _simpson_metrics.append({
            "Stratum": _grp,
            "Sample Size (n)": len(_sub),
            "Slope (Beta)": f"{_sl:+.3f}",
            "Pearson r": f"{_r:+.4f}",
            "p-Value": f"{_p:.2e}",
            "Diagnosis": "True Causal Negative Slope",
        })

    _df_simpson_summary = pd.DataFrame(_simpson_metrics)

    return (
        mo.md("#### Simpson's Paradox Stratification Table"),
        mo.ui.table(_df_simpson_summary),
    )


if __name__ == "__main__":
    app.run()
