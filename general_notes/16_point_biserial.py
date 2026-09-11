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
    import scipy.stats as stats

    return go, make_subplots, mo, np, pd, stats


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 16: Point-Biserial Correlation and Continuous-Binary Association

    &larr; Previous Note: [15 Mutual Information](15_mutual_information.py) | Next Note: [17 Jensen Inequality](17_jensen_inequality.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Real-world datasets rarely consist exclusively of continuous numerical variables. In machine learning, medical diagnostics, A/B testing, and psychometrics, analysts frequently encounter mixed data where a continuous feature (e.g., patient blood pressure, annual income, model inference latency) must be tested for association with a binary variable (e.g., disease presence, subscription churn, experimental treatment vs control).

    The **Point-Biserial Correlation Coefficient ($r_{pb}$)** is the foundational metric for measuring association between a continuous variable and a binary indicator:
    1. **Direct Bridge to Pearson's Correlation**: Point-biserial correlation is not a different mathematical creature; it is mathematically identical to Pearson's product-moment correlation calculated between a continuous variable and a $0/1$ indicator dummy variable.
    2. **Direct Equivalence to Student's Two-Sample t-Test**: Testing whether $r_{pb} = 0$ is exactly equivalent to running an independent two-sample Student's t-test comparing group means ($H_0: \mu_1 = \mu_0$). The test statistic satisfies $t = r_{pb} \sqrt{\frac{n - 2}{1 - r_{pb}^2}}$.
    3. **Fast Feature Screening for Binary Classification**: Computing $r_{pb}$ across hundreds of continuous features provides a computationally efficient, scale-invariant filter for ranking features by linear discriminative power prior to training models (e.g., Logistic Regression, SVMs, or Gradient Boosted Trees).
    4. **Psychometric Item Discrimination**: In test design and Item Response Theory (IRT), the point-biserial correlation between an individual question score ($0 = \text{incorrect}, 1 = \text{correct}$) and overall test score measures whether a question effectively discriminates high-performing students from low-performing students.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Mathematical Formulation of Point-Biserial Correlation

    Let $Y \in \{0, 1\}$ be a naturally dichotomous binary variable, and let $X \in \mathbb{R}$ be a continuous random variable.
    Let $n_1$ and $n_0$ denote the sample counts of the two subgroups ($Y = 1$ and $Y = 0$), with total sample size $n = n_0 + n_1$.
    Let $M_1 = \bar{X}_1$ and $M_0 = \bar{X}_0$ denote the sample means of $X$ within each subgroup:

    $$
    M_1 = \frac{1}{n_1} \sum_{i: Y_i = 1} X_i, \quad M_0 = \frac{1}{n_0} \sum_{i: Y_i = 0} X_i
    $$

    The **point-biserial correlation coefficient** $r_{pb}$ is defined as:

    $$
    r_{pb} = \frac{M_1 - M_0}{s_X} \sqrt{\frac{n_1 n_0}{n(n - 1)}}
    $$

    where $s_X$ is the pooled sample standard deviation across all $n$ observations:

    $$
    s_X = \sqrt{\frac{1}{n - 1} \sum_{i=1}^n (X_i - \bar{X})^2}
    $$

    In terms of the population standard deviation $\sigma_X$ and subgroup proportion $p = n_1 / n$:

    $$
    r_{pb} = \frac{M_1 - M_0}{\sigma_X} \sqrt{p (1 - p)}
    $$

    ---

    ### Exact Equivalence to Pearson's Product-Moment Correlation

    If we encode the binary variable as a numeric vector $Y_i \in \{0, 1\}$ and evaluate the standard Pearson correlation formula:

    $$
    r_{XY} = \frac{\sum_{i=1}^n (X_i - \bar{X})(Y_i - \bar{Y})}{\sqrt{\sum_{i=1}^n (X_i - \bar{X})^2 \sum_{i=1}^n (Y_i - \bar{Y})^2}}
    $$

    Algebraic substitution of $\bar{Y} = p$ and $\sum (Y_i - \bar{Y})^2 = n p (1 - p)$ reveals that:

    $$
    r_{XY} \equiv r_{pb}
    $$

    Thus, point-biserial correlation is a specialized algebraic simplification of Pearson's $r$.

    ---

    ### Exact Correspondence with Student's Two-Sample t-Test

    Testing the null hypothesis of zero point-biserial correlation ($H_0: r_{pb} = 0$) is mathematically identical to testing for equality of group means ($H_0: \mu_1 = \mu_0$) in an independent two-sample t-test:

    $$
    t = \frac{r_{pb} \sqrt{n - 2}}{\sqrt{1 - r_{pb}^2}} \sim t(n - 2)
    $$

    Squaring both sides connects point-biserial correlation directly to the Analysis of Variance (ANOVA) F-statistic:

    $$
    F = t^2 = \frac{r_{pb}^2 (n - 2)}{1 - r_{pb}^2}
    $$

    Consequently, $r_{pb}^2$ represents the exact coefficient of determination ($R^2$), measuring the proportion of total variance in continuous variable $X$ accounted for by group membership $Y$.

    ---

    ### Sensitivity to Class Imbalance

    A critical consideration in machine learning is that $r_{pb}$ scales proportionally with $\sqrt{p(1 - p)}$.
    * When classes are balanced ($p = 0.5$), $\sqrt{p(1 - p)} = 0.5$ (maximum possible value).
    * When severe class imbalance exists (e.g., fraud detection where $p = 0.01$), $\sqrt{p(1 - p)} \approx 0.0995$.

    Even if the underlying standardized mean difference (Cohen's $d = \frac{M_1 - M_0}{\sigma_{\text{pooled}}}$) remains massive, severe class imbalance substantially compresses $r_{pb}$. Analysts must account for class proportions when using $r_{pb}$ for feature ranking.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Group Separation and Proportion Sensitivity

    The interactive subplots below display the mechanics of Point-Biserial Correlation:
    * **Left Panel**: Distribution of continuous variable $X$ stratified by binary indicator $Y \in \{0, 1\}$ ($N = 120$ samples). Box plots with jittered data points illustrate group mean separation ($\Delta M = M_1 - M_0 = 3.0$), yielding $r_{pb} \approx 0.53$.
    * **Right Panel**: Sensitivity of $r_{pb}$ to the subgroup proportion $p = n_1 / n$ across varying effect sizes (Cohen's $d \in \{0.5, 1.0, 1.5, 2.0\}$). The parabolic shape reveals how class imbalance dampens correlation even when true group separation remains unchanged.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    rng_vis = np.random.default_rng(42)
    n0_size = 60
    n1_size = 60

    # Continuous values for group 0 and group 1
    x0_data = rng_vis.normal(loc=10.0, scale=2.0, size=n0_size)
    x1_data = rng_vis.normal(loc=13.0, scale=2.0, size=n1_size)

    # Class proportion grid p in [0.01, 0.99]
    p_values_grid = np.linspace(0.01, 0.99, 120)
    cohen_d_values = [0.5, 1.0, 1.5, 2.0]
    palette_colors = ["#94a3b8", "#2563eb", "#16a34a", "#dc2626"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Continuous Feature Stratified by Binary Groups (r_pb ≈ 0.53)",
            "Effect of Class Proportion (p) on Point-Biserial r_pb",
        ],
    )

    # Left: Group 0 Box Plot
    fig.add_trace(
        go.Box(
            y=x0_data,
            name="Group Y = 0 (Control)",
            marker_color="#2563eb",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.6,
        ),
        row=1,
        col=1,
    )

    # Left: Group 1 Box Plot
    fig.add_trace(
        go.Box(
            y=x1_data,
            name="Group Y = 1 (Treatment)",
            marker_color="#ea580c",
            boxpoints="all",
            jitter=0.3,
            pointpos=-1.6,
        ),
        row=1,
        col=1,
    )

    # Right: Sensitivity curves for different Cohen's d
    for idx_d, d_effect in enumerate(cohen_d_values):
        # Formula: r = (d * sqrt(p(1-p))) / sqrt(1 + d^2 * p(1-p))
        pq_factor = p_values_grid * (1.0 - p_values_grid)
        r_pb_theoretical = (d_effect * np.sqrt(pq_factor)) / np.sqrt(1.0 + (d_effect**2) * pq_factor)

        fig.add_trace(
            go.Scatter(
                x=p_values_grid,
                y=r_pb_theoretical,
                mode="lines",
                line=dict(color=palette_colors[idx_d], width=2.5),
                name=f"Effect Size d = {d_effect:.1f}",
            ),
            row=1,
            col=2,
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(title="Continuous Variable (X)", gridcolor="#f1f5f9"),
        xaxis2=dict(
            title="Class Proportion p = n₁ / n",
            range=[0.0, 1.0],
            gridcolor="#f1f5f9",
        ),
        yaxis2=dict(
            title="Point-Biserial r_pb",
            range=[0.0, 0.85],
            gridcolor="#f1f5f9",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        cohen_d_values,
        d_effect,
        fig,
        idx_d,
        n0_size,
        n1_size,
        p_values_grid,
        palette_colors,
        pq_factor,
        r_pb_theoretical,
        rng_vis,
        x0_data,
        x1_data,
    )


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    ### Example 1: Mathematical Equivalence of Point-Biserial, Pearson, and Student's t-Test

    In this example, we generate an empirical dataset ($n = 100$) and compute:
    1. The Point-Biserial correlation $r_{pb}$ using `scipy.stats.pointbiserialr`.
    2. The Pearson correlation $r_{XY}$ using `scipy.stats.pearsonr`.
    3. The independent two-sample Student's t-test statistic using `scipy.stats.ttest_ind`.
    4. The transformed t-statistic derived analytically from correlation:

    $$t = \frac{r_{pb} \sqrt{n - 2}}{\sqrt{1 - r_{pb}^2}}$$

    We verify that all three formulations yield identical test statistics down to machine floating-point precision ($< 10^{-14}$).
    """)
    return


@app.cell
def _(np, stats):
    rng_ex1 = np.random.default_rng(101)
    n0_count = 60
    n1_count = 40
    n_total = n0_count + n1_count

    # Sample draws
    sample_group0 = rng_ex1.normal(10.0, 2.0, n0_count)
    sample_group1 = rng_ex1.normal(12.5, 2.0, n1_count)

    continuous_array = np.concatenate([sample_group0, sample_group1])
    binary_array = np.concatenate([np.zeros(n0_count), np.ones(n1_count)])

    # 1. Point-Biserial
    r_pb_val, p_pb_val = stats.pointbiserialr(binary_array, continuous_array)

    # 2. Pearson Correlation
    r_pearson_val, p_pearson_val = stats.pearsonr(binary_array, continuous_array)

    # 3. Two-Sample Student's t-test
    ttest_res = stats.ttest_ind(sample_group1, sample_group0, equal_var=True)
    t_stat_ind = float(ttest_res.statistic)
    p_val_ind = float(ttest_res.pvalue)

    # 4. Transformed t from r_pb
    t_from_r = float(r_pb_val * np.sqrt(n_total - 2) / np.sqrt(1.0 - r_pb_val**2))

    discrepancy_r = abs(r_pb_val - r_pearson_val)
    discrepancy_t = abs(t_stat_ind - t_from_r)

    equivalence_summary_records = [
        {
            "Statistical Test / Metric": "Point-Biserial Correlation r_pb",
            "Calculated Statistic": f"{r_pb_val:.6f}",
            "Calculated p-value": f"{p_pb_val:.2e}",
            "Equivalence Condition": "Reference Metric",
        },
        {
            "Statistical Test / Metric": "Pearson Product-Moment r_XY",
            "Calculated Statistic": f"{r_pearson_val:.6f}",
            "Calculated p-value": f"{p_pearson_val:.2e}",
            "Equivalence Condition": f"Exact Match (|Diff| = {discrepancy_r:.1e})",
        },
        {
            "Statistical Test / Metric": "Two-Sample Student's t-test",
            "Calculated Statistic": f"t = {t_stat_ind:.6f}",
            "Calculated p-value": f"{p_val_ind:.2e}",
            "Equivalence Condition": "Two-Group Hypothesis Test",
        },
        {
            "Statistical Test / Metric": "t Derived from r_pb Formula",
            "Calculated Statistic": f"t = {t_from_r:.6f}",
            "Calculated p-value": f"{p_pb_val:.2e}",
            "Equivalence Condition": f"Exact Match (|Diff| = {discrepancy_t:.1e})",
        },
    ]

    return (
        binary_array,
        continuous_array,
        discrepancy_r,
        discrepancy_t,
        equivalence_summary_records,
        n0_count,
        n1_count,
        n_total,
        p_pb_val,
        p_pearson_val,
        p_val_ind,
        r_pb_val,
        r_pearson_val,
        rng_ex1,
        sample_group0,
        sample_group1,
        t_from_r,
        t_stat_ind,
        ttest_res,
    )


@app.cell(hide_code=True)
def _(equivalence_summary_records, mo, pd):
    df_equiv = pd.DataFrame(equivalence_summary_records)
    mo.ui.table(df_equiv)
    return (df_equiv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Impact of Severe Class Imbalance on Feature Screening in Machine Learning

    When screening features for binary classification tasks (such as credit fraud or rare disease detection), class distributions are heavily imbalanced.

    Below, we simulate a constant true underlying group separation ($\mu_0 = 0.0, \mu_1 = 1.5, \sigma = 1.0$, corresponding to a fixed Cohen's $d = 1.5$) across five imbalance ratios ranging from $50:50$ to $99:1$.

    We compare the Point-Biserial correlation $r_{pb}$ against class-invariant metrics (Cohen's $d$ and Area Under the ROC Curve):
    """)
    return


@app.cell
def _(np, stats):
    rng_ex2 = np.random.default_rng(202)
    total_obs = 1000
    true_mu0 = 0.0
    true_mu1 = 1.5
    true_sd = 1.0

    imbalance_ratios = [
        ("Balanced (50:50)", 0.50),
        ("Moderate (70:30)", 0.30),
        ("High (90:10)", 0.10),
        ("Severe (95:5)", 0.05),
        ("Extreme (99:1)", 0.01),
    ]

    imbalance_records = []

    for name_scen, prop_1 in imbalance_ratios:
        n1_pts = int(total_obs * prop_1)
        n0_pts = total_obs - n1_pts

        pts_g0 = rng_ex2.normal(true_mu0, true_sd, size=n0_pts)
        pts_g1 = rng_ex2.normal(true_mu1, true_sd, size=n1_pts)

        all_x = np.concatenate([pts_g0, pts_g1])
        all_y = np.concatenate([np.zeros(n0_pts), np.ones(n1_pts)])

        # Calculate r_pb
        r_pb_emp, _ = stats.pointbiserialr(all_y, all_x)

        # Empirical Cohen's d
        s_pooled = np.sqrt(((n0_pts - 1) * np.var(pts_g0, ddof=1) + (n1_pts - 1) * np.var(pts_g1, ddof=1)) / (total_obs - 2))
        emp_cohen_d = (np.mean(pts_g1) - np.mean(pts_g0)) / s_pooled

        # Theoretical r_pb from d and p
        theo_r_pb = (1.5 * np.sqrt(prop_1 * (1.0 - prop_1))) / np.sqrt(1.0 + 1.5**2 * prop_1 * (1.0 - prop_1))

        imbalance_records.append(
            {
                "Imbalance Scenario": name_scen,
                "Positive Class %": f"{prop_1 * 100:.1f}%",
                "True Cohen's d": "1.500",
                "Empirical Cohen's d": f"{emp_cohen_d:.3f}",
                "Theoretical r_pb": f"{theo_r_pb:.3f}",
                "Empirical r_pb": f"{r_pb_emp:.3f}",
                "Correlation Degradation": f"{(1.0 - theo_r_pb / 0.600) * 100.0:.1f}% drop",
            }
        )

    return (
        all_x,
        all_y,
        emp_cohen_d,
        imbalance_ratios,
        imbalance_records,
        n0_pts,
        n1_pts,
        name_scen,
        prop_1,
        pts_g0,
        pts_g1,
        r_pb_emp,
        rng_ex2,
        s_pooled,
        theo_r_pb,
        total_obs,
        true_mu0,
        true_mu1,
        true_sd,
    )


@app.cell(hide_code=True)
def _(imbalance_records, mo, pd):
    df_imbalance = pd.DataFrame(imbalance_records)
    mo.ui.table(df_imbalance)
    return (df_imbalance,)


if __name__ == "__main__":
    app.run()
