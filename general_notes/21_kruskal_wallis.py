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
    # Note 21: Kruskal-Wallis Test, Non-Parametric ANOVA, and Rank Sums

    &larr; Previous Note: [20 Spurious Correlation](20_spurious_correlation.py) | Next Note: [22 ACF and PACF](22_acf_and_pacf.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    When comparing numerical metrics across three or more experimental cohorts ($k \geq 3$), classical One-Way Analysis of Variance (ANOVA) is the traditional benchmark. However, ANOVA relies on three strict parametric assumptions:
    1. **Normality of Residuals**: Observations within each group must follow a Gaussian distribution.
    2. **Homoscedasticity**: All groups must share identical population variances ($\sigma_1^2 = \sigma_2^2 = \dots = \sigma_k^2$).
    3. **Independence**: Observations are independently sampled.

    In modern engineering, finance, and machine learning pipelines, raw metrics (such as server request latencies, web application load times, model inference latencies, transaction fraud amounts, and customer lifetime values) consistently violate normality. They exhibit severe positive skew, heavy tails (Laplace, Pareto, Cauchy), or multimodal clusters. In these conditions, ANOVA's $F$-test loses statistical power and yields unreliable $p$-values.

    **The Kruskal-Wallis $H$-test** is the non-parametric generalization of ANOVA:
    1. **Distribution-Free Rank Test**: By replacing raw continuous values with their ranks across the pooled dataset, Kruskal-Wallis eliminates sensitivity to extreme outliers and asymmetric heavy tails.
    2. **Hypothesis of Stochastic Dominance**: Rather than comparing population means $\mu_j$, Kruskal-Wallis tests whether the population distributions are identical against the alternative that at least one group tends to produce stochastically larger values than another.
    3. **Rigorous Diagnostic Workflow**: A principled data science workflow verifies homoscedasticity (Levene's test) and normality (Shapiro-Wilk test). If normality is rejected while homoscedasticity holds, Kruskal-Wallis is the appropriate test.
    4. **Post-Hoc Pairwise Localization**: If the omnibus Kruskal-Wallis test rejects the null hypothesis, post-hoc Dunn's rank sum tests with Bonferroni or FDR corrections determine precisely which group pairs exhibit statistically significant divergence.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Mathematical Formulation of the $H$-Statistic

    Consider $k$ independent groups with sample sizes $n_1, n_2, \dots, n_k$. The total number of pooled observations is:

    $$
    N = \sum_{j=1}^k n_j
    $$

    All $N$ observations across all groups are combined into a single sorted array and assigned ranks $r_{ij} \in \{1, 2, \dots, N\}$. For tied observations, each tied value receives the average (mid-rank) of the positions it spans.

    Let $R_j$ denote the sum of ranks for the $j$-th group, and $\bar{R}_j = \frac{R_j}{n_j}$ denote the mean rank of group $j$:

    $$
    R_j = \sum_{i=1}^{n_j} r_{ij}, \quad \bar{R}_j = \frac{R_j}{n_j}
    $$

    Under the null hypothesis $H_0$ that all $k$ groups originate from identical continuous distributions, the expected average rank across all observations is:

    $$
    \bar{R}_{\text{overall}} = \frac{N + 1}{2}
    $$

    The Kruskal-Wallis test statistic $H$ measures the weighted sum of squared deviations of group average ranks from the global expected mean rank:

    $$
    H = \frac{12}{N(N + 1)} \sum_{j=1}^k n_j \left(\bar{R}_j - \frac{N + 1}{2}\right)^2
    $$

    Expanding the quadratic term yields the standard computational form:

    $$
    H = \left[\frac{12}{N(N + 1)} \sum_{j=1}^k \frac{R_j^2}{n_j}\right] - 3(N + 1)
    $$

    ---

    ### 2. Tie Correction Factor

    When tied values exist in the sample, the variance of the ranks decreases. The test statistic is adjusted by dividing by a tie correction factor $C_{\text{tie}}$:

    $$
    C_{\text{tie}} = 1 - \frac{\sum_{m=1}^G (t_m^3 - t_m)}{N^3 - N}
    $$

    where $G$ is the number of distinct tie groups, and $t_m$ is the count of observations tied at the $m$-th value. The adjusted statistic is:

    $$
    H_{\text{adj}} = \frac{H}{C_{\text{tie}}}
    $$

    ---

    ### 3. Asymptotic Chi-Square Distribution and Inference

    Under $H_0$, as the individual group sample sizes satisfy $n_j \geq 5$, the sampling distribution of $H$ converges asymptotically to a Chi-Square distribution with $k - 1$ degrees of freedom:

    $$
    H \xrightarrow{d} \chi^2(k - 1)
    $$

    The two-tailed $p$-value is evaluated via the Chi-Square survival function:

    $$
    p = 1 - F_{\chi^2}(H; k - 1) = \int_H^\infty \frac{x^{\frac{k-1}{2} - 1} e^{-\frac{x}{2}}}{2^{\frac{k-1}{2}} \Gamma\left(\frac{k-1}{2}\right)} \, dx
    $$

    If $p < \alpha$ (typically $0.05$), we reject the null hypothesis and conclude that at least one group stochastically dominates another.

    ---

    ### 4. Post-Hoc Dunn's Test with Multiple Comparison Adjustments

    The Kruskal-Wallis test is an omnibus test: it establishes that a difference exists, but does not indicate which specific group pairs differ. To evaluate pairwise contrasts between group $i$ and group $j$, **Dunn's test** evaluates the standardized difference in mean ranks:

    $$
    z_{ij} = \frac{|\bar{R}_i - \bar{R}_j|}{\sigma_{ij}}
    $$

    where the standard error of the rank difference under $H_0$ is:

    $$
    \sigma_{ij} = \sqrt{\left(\frac{N(N + 1)}{12} - \frac{\sum (t_m^3 - t_m)}{12(N - 1)}\right) \left(\frac{1}{n_i} + \frac{1}{n_j}\right)}
    $$

    For $m = \binom{k}{2}$ simultaneous pairwise comparisons, the family-wise error rate is controlled using Bonferroni correction:

    $$
    p_{\text{adj}} = \min\left(1.0, \, m \cdot 2\left(1 - \Phi(|z_{ij}|)\right)\right)
    $$
    """)
    return


@app.cell
def _(np, pd):
    # Data Generation: Heavy-Tailed Laplace Distributed Cohorts (Server Latency Benchmark)
    # 3 Server Configurations: Standard (Loc=20ms), Optimized (Loc=21ms), Cache-Boosted (Loc=23.5ms)
    np.random.seed(47)
    _n = 100

    group1 = np.random.laplace(loc=20.0, scale=4.5, size=_n)
    group2 = np.random.laplace(loc=21.0, scale=4.5, size=_n)
    group3 = np.random.laplace(loc=23.5, scale=4.5, size=_n)

    df_kruskal = pd.DataFrame(
        {
            "Latency": np.concatenate([group1, group2, group3]),
            "Config": np.repeat(["Cluster A (20ms)", "Cluster B (21ms)", "Cluster C (23.5ms)"], _n),
        }
    )

    return df_kruskal, group1, group2, group3


@app.cell
def _(df_kruskal, go, group1, group2, group3, make_subplots, mo, np, stats):
    # Interactive Visualizations Cell:
    # Subplot 1: Violin & Box Plot showing non-Gaussian heavy tails
    # Subplot 2: Normal Q-Q Plots revealing systematic Laplace tail departure
    # Subplot 3: Pooled Rank Sum Distributions and Mean Rank Comparison

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Group Distributions (Violin + Quartiles)",
            "2. Normal Q-Q Diagnostics (Heavy Tails)",
            "3. Pooled Mean Ranks vs. H0 Expected",
        ),
        horizontal_spacing=0.09,
    )

    _colors = {"Cluster A (20ms)": "#3b82f6", "Cluster B (21ms)": "#10b981", "Cluster C (23.5ms)": "#8b5cf6"}

    # Subplot 1: Violins
    for _grp, _col in _colors.items():
        _sub = df_kruskal[df_kruskal["Config"] == _grp]["Latency"]
        _fig.add_trace(
            go.Violin(
                y=_sub,
                name=_grp,
                box_visible=True,
                meanline_visible=True,
                line_color=_col,
                fillcolor=_col,
                opacity=0.6,
                showlegend=False,
                points="outliers",
            ),
            row=1,
            col=1,
        )

    # Subplot 2: Q-Q Plot of Cluster A against Standard Normal
    _osm, _osr = stats.probplot(group1, dist="norm")
    _fig.add_trace(
        go.Scatter(
            x=_osm[0],
            y=_osm[1],
            mode="markers",
            marker=dict(size=5, color="#3b82f6"),
            name="Cluster A Quantiles",
            hovertemplate="Theoretical: %{x:.2f}<br>Sample: %{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    # Q-Q Reference Line
    _line_x = np.array([_osm[0].min(), _osm[0].max()])
    _line_y = _osr[1] + _osr[0] * _line_x
    _fig.add_trace(
        go.Scatter(
            x=_line_x,
            y=_line_y,
            mode="lines",
            line=dict(color="#ef4444", dash="dash", width=2),
            name="Gaussian Reference",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Pooled Mean Ranks
    _pooled_data = np.concatenate([group1, group2, group3])
    _pooled_ranks = stats.rankdata(_pooled_data)
    _n1, _n2, _n3 = len(group1), len(group2), len(group3)
    _mean_r1 = np.mean(_pooled_ranks[:_n1])
    _mean_r2 = np.mean(_pooled_ranks[_n1 : _n1 + _n2])
    _mean_r3 = np.mean(_pooled_ranks[_n1 + _n2 :])
    _expected_mean = (len(_pooled_data) + 1) / 2.0

    _fig.add_trace(
        go.Bar(
            x=["Cluster A", "Cluster B", "Cluster C"],
            y=[_mean_r1, _mean_r2, _mean_r3],
            marker_color=["#3b82f6", "#10b981", "#8b5cf6"],
            name="Mean Rank",
            text=[f"{_mean_r1:.1f}", f"{_mean_r2:.1f}", f"{_mean_r3:.1f}"],
            textposition="auto",
        ),
        row=1,
        col=3,
    )

    _fig.add_trace(
        go.Scatter(
            x=["Cluster A", "Cluster C"],
            y=[_expected_mean, _expected_mean],
            mode="lines",
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            name="H0 Expected Rank (150.5)",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Non-Parametric Group Diagnostics: Distributions, Normality, and Ranks",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=70, b=80),
    )

    _fig.update_yaxes(title_text="Latency (ms)", row=1, col=1)
    _fig.update_xaxes(title_text="Theoretical Normal Quantiles", row=1, col=2)
    _fig.update_yaxes(title_text="Sample Latency Quantiles", row=1, col=2)
    _fig.update_xaxes(title_text="Configuration Group", row=1, col=3)
    _fig.update_yaxes(title_text="Average Group Rank", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two end-to-end production statistical workflows:
    1. **Diagnostic Testing Pipeline**: Formally running Levene's test for homoscedasticity, Shapiro-Wilk tests for Gaussian distribution conformance, and deciding whether ANOVA or Kruskal-Wallis is statistically valid.
    2. **Kruskal-Wallis $H$-test from Scratch & Post-Hoc Dunn's Procedure**: Computing rank sums, tie corrections, asymptotic $\chi^2$ $p$-value against `scipy.stats.kruskal`, followed by Dunn's pairwise contrasts with Bonferroni multiple testing adjustment.
    """)
    return


@app.cell
def _(group1, group2, group3, mo, pd, stats):
    # Example 1: Formal Diagnostic Pipeline (Homoscedasticity & Normality)
    _lev_stat, _lev_p = stats.levene(group1, group2, group3, center="median")

    _shapiro_results = []
    _shapiro_results.append({
        "Diagnostic Test": "Levene's Homoscedasticity Test",
        "Target": "All 3 Groups (Joint)",
        "Test Statistic": f"W = {_lev_stat:.4f}",
        "p-Value": f"{_lev_p:.4f}",
        "Decision (alpha=0.05)": "Equal Variances Upheld" if _lev_p >= 0.05 else "Heteroscedasticity Detected",
    })

    for _idx, (_name, _grp) in enumerate([("Cluster A", group1), ("Cluster B", group2), ("Cluster C", group3)]):
        _w_stat, _p_val = stats.shapiro(_grp)
        _shapiro_results.append({
            "Diagnostic Test": "Shapiro-Wilk Normality Test",
            "Target": _name,
            "Test Statistic": f"W = {_w_stat:.4f}",
            "p-Value": f"{_p_val:.2e}",
            "Decision (alpha=0.05)": "Normal" if _p_val >= 0.05 else "Non-Gaussian (Reject Normality)",
        })

    _df_diagnostics = pd.DataFrame(_shapiro_results)

    return (
        mo.md("#### Assumption Verification: Homoscedasticity and Normality Screening"),
        mo.ui.table(_df_diagnostics),
    )


@app.cell
def _(group1, group2, group3, mo, np, pd, stats):
    # Example 2: Pure NumPy Kruskal-Wallis from Scratch + Post-Hoc Dunn's Test
    _groups = [group1, group2, group3]
    _k = len(_groups)
    _sizes = [len(_g) for _g in _groups]
    _N = sum(_sizes)

    # 1. Pool data and compute mid-ranks
    _pooled_vals = np.concatenate(_groups)
    _ranks = stats.rankdata(_pooled_vals)

    # 2. Group Rank Sums
    _offset = 0
    _rank_sums = []
    _mean_ranks = []
    for _n in _sizes:
        _r_g = _ranks[_offset : _offset + _n]
        _rank_sums.append(np.sum(_r_g))
        _mean_ranks.append(np.mean(_r_g))
        _offset += _n

    # 3. Unadjusted H statistic
    _sum_sq_term = sum((_R**2) / _n for _R, _n in zip(_rank_sums, _sizes))
    _H_raw = (12.0 / (_N * (_N + 1))) * _sum_sq_term - 3.0 * (_N + 1)

    # 4. Tie correction factor
    _, _counts = np.unique(_pooled_vals, return_counts=True)
    _tie_groups = _counts[_counts > 1]
    if len(_tie_groups) > 0:
        _c_tie = 1.0 - np.sum(_tie_groups**3 - _tie_groups) / (_N**3 - _N)
    else:
        _c_tie = 1.0

    _H_adj = _H_raw / _c_tie
    _df = _k - 1
    _p_scratch = 1.0 - stats.chi2.cdf(_H_adj, df=_df)

    # SciPy verification
    _scipy_h, _scipy_p = stats.kruskal(group1, group2, group3)

    # 5. Post-Hoc Dunn's Test (Pairwise Differences)
    _pairs = [("Cluster A", "Cluster B", 0, 1), ("Cluster A", "Cluster C", 0, 2), ("Cluster B", "Cluster C", 1, 2)]
    _num_comparisons = len(_pairs)
    _dunn_records = []

    for _name_a, _name_b, _i, _j in _pairs:
        _diff = np.abs(_mean_ranks[_i] - _mean_ranks[_j])
        _se = np.sqrt((_N * (_N + 1) / 12.0) * (1.0 / _sizes[_i] + 1.0 / _sizes[_j]))
        _z_pair = _diff / _se
        _p_unadj = 2.0 * (1.0 - stats.norm.cdf(_z_pair))
        _p_bonf = min(1.0, _p_unadj * _num_comparisons)
        _dunn_records.append({
            "Pairwise Contrast": f"{_name_a} vs {_name_b}",
            "Mean Rank Diff": f"{_diff:.2f}",
            "Standard Error": f"{_se:.2f}",
            "z-Score": f"{_z_pair:.3f}",
            "Unadjusted p-Value": f"{_p_unadj:.4f}",
            "Bonferroni Adjusted p-Value": f"{_p_bonf:.4f}",
            "Significant (alpha=0.05)": "Yes (p < 0.05)" if _p_bonf < 0.05 else "No (p >= 0.05)",
        })

    _df_kruskal_summary = pd.DataFrame(
        [
            {"Metric": "Omnibus H-Statistic (Scratch)", "Value": f"{_H_adj:.4f}", "Details": "Adjusted for tied mid-ranks"},
            {"Metric": "SciPy Reference H-Statistic", "Value": f"{_scipy_h:.4f}", "Details": "scipy.stats.kruskal()"},
            {"Metric": "Degrees of Freedom (k - 1)", "Value": str(_df), "Details": "k = 3 groups"},
            {"Metric": "Chi-Square Asymptotic p-Value", "Value": f"{_p_scratch:.4e}", "Details": "Reject H0 (medians differ)"},
        ]
    )

    _df_dunn = pd.DataFrame(_dunn_records)

    return (
        mo.md("#### Kruskal-Wallis Omnibus Test Results"),
        mo.ui.table(_df_kruskal_summary),
        mo.md("#### Post-Hoc Dunn's Test with Bonferroni Correction"),
        mo.ui.table(_df_dunn),
    )


if __name__ == "__main__":
    app.run()
