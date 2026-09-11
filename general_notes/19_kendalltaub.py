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
    # Note 19: Kendall's Tau-b, Concordance, and Robust Rank Correlation

    &larr; Previous Note: [18 Cramer V](18_cramer_v.py) | Next Note: [20 Spurious Correlation](20_spurious_correlation.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In data science, experimental statistics, and modern machine learning evaluation, we frequently measure association between variables that are ordinal (e.g. Likert satisfaction scales, benchmark rankings, preference annotations, customer review stars) or continuous variables corrupted by severe non-Gaussian noise and extreme outliers.

    While Pearson's correlation coefficient $r$ measures linear association between continuous variables, it suffers from two major limitations:
    1. It assumes linearity and bivariate normality, breaking down when the underlying relationship is non-linear but strictly monotonic.
    2. It possesses an unbounded influence function: a single severe outlier can shift Pearson's $r$ from $+0.95$ to $-0.50$.

    Spearman's rank correlation $\rho$ mitigates non-linearity by converting raw values into ranks, but it does not account directly for tied observations and its sampling distribution converges slowly to normality.

    **Kendall's Tau-b ($\tau_b$)** solves these challenges through pairwise concordance:
    1. **Probabilistic Interpretation**: Kendall's $\tau$ is defined directly as the difference between the probability of concordance and discordance: $P(\text{concordance}) - P(\text{discordance})$. A value of $+0.60$ means that for any randomly selected pair of observations, the probability that they agree in ordering is 60 percentage points higher than the probability that they disagree.
    2. **Tie-Corrected Formulation ($\tau_b$)**: Unlike Kendall's raw $\tau_a$ (which underestimates association when ties exist), $\tau_b$ normalizes by the geometric mean of untied pairs in each variable, ensuring the coefficient reaches the canonical $[-1, 1]$ bounds even on heavily discretized ordinal scales.
    3. **Superior Asymptotic and U-Statistic Properties**: Because Kendall's $\tau$ is an unbiased U-statistic, its variance under the null hypothesis of independence depends purely on the sample size $n$ and tie counts, converging to a standard Gaussian distribution much faster than Spearman's $\rho$. This yields reliable hypothesis tests and confidence intervals even in small samples ($n < 30$).
    4. **LLM Evaluation, Preference Alignment, and Learning to Rank (LTR)**: In RLHF (Reinforcement Learning from Human Feedback), search engine ranking, and LLM-as-a-judge benchmarking, Kendall's $\tau_b$ is the standard metric for measuring inter-annotator agreement and ranking alignment between human annotators and language model judges.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Concordant, Discordant, and Tied Pairs

    Consider a bivariate sample of $n$ observations:

    $$
    \{(x_1, y_1), (x_2, y_2), \dots, (x_n, y_n)\}
    $$

    The total number of distinct pairs $(i, j)$ with $1 \leq i < j \leq n$ is:

    $$
    N_{\text{pairs}} = \binom{n}{2} = \frac{n(n - 1)}{2}
    $$

    For any pair of observations $(x_i, y_i)$ and $(x_j, y_j)$, we evaluate the sign of their coordinate differences:

    $$
    \Delta_{ij} = (x_i - x_j)(y_i - y_j)
    $$

    Every pair falls into exactly one of four mutual categories:

    #### Concordant Pairs ($C$)
    Both variables change in the same direction:
    $$
    \Delta_{ij} > 0 \iff (x_i > x_j \text{ and } y_i > y_j) \text{ or } (x_i < x_j \text{ and } y_i < y_j)
    $$

    #### Discordant Pairs ($D$)
    The variables change in opposite directions:
    $$
    \Delta_{ij} < 0 \iff (x_i > x_j \text{ and } y_i < y_j) \text{ or } (x_i < x_j \text{ and } y_i > y_j)
    $$

    #### Tied Pairs on $X$ Only ($T_X$)
    Observations share the same $x$ coordinate but differ in $y$:
    $$
    x_i = x_j \quad \text{and} \quad y_i \neq y_j
    $$

    #### Tied Pairs on $Y$ Only ($T_Y$)
    Observations share the same $y$ coordinate but differ in $x$:
    $$
    x_i \neq x_j \quad \text{and} \quad y_i = y_j
    $$

    #### Tied Pairs on Both ($T_{XY}$)
    Observations are identical in both dimensions:
    $$
    x_i = x_j \quad \text{and} \quad y_i = y_j
    $$

    The sum of all classifications satisfies:

    $$
    C + D + T_X + T_Y + T_{XY} = \binom{n}{2}
    $$

    ---

    ### 2. Kendall's Tau-a vs. Tau-b vs. Tau-c

    #### Kendall's Tau-a ($\tau_a$)
    When there are no ties in the data, Kendall's tau is simply:

    $$
    \tau_a = \frac{C - D}{\binom{n}{2}} = \frac{C - D}{\frac{1}{2} n (n - 1)}
    $$

    If ties are present, $\tau_a$ cannot reach $+1$ or $-1$ because ties reduce the maximum possible value of $C$ or $D$.

    #### Kendall's Tau-b ($\tau_b$)
    To account for tied observations, Maurice Kendall defined $\tau_b$ by adjusting the denominator with the geometric mean of pairs that are untied in $X$ and untied in $Y$:

    $$
    \tau_b = \frac{C - D}{\sqrt{(C + D + T_X)(C + D + T_Y)}}
    $$

    Equivalently, letting $t_k$ denote the size of the $k$-th group of tied values in $X$, and $u_m$ denote the size of the $m$-th group of tied values in $Y$:

    $$
    n_0 = \frac{n(n - 1)}{2}, \quad n_1 = \sum_{k} \frac{t_k(t_k - 1)}{2}, \quad n_2 = \sum_{m} \frac{u_m(u_m - 1)}{2}
    $$

    $$
    \tau_b = \frac{C - D}{\sqrt{(n_0 - n_1)(n_0 - n_2)}}
    $$

    Notice that $n_0 - n_1 = C + D + T_X$ and $n_0 - n_2 = C + D + T_Y$. When no ties exist, $n_1 = n_2 = 0$ and $\tau_b$ reduces identically to $\tau_a$.

    #### Stuart's Tau-c ($\tau_c$)
    When analyzing rectangular contingency tables where the number of rows $r$ and columns $c$ differ, $\tau_b$ cannot reach $\pm 1$. Stuart's $\tau_c$ adjusts for table dimensions:

    $$
    \tau_c = \frac{2 m (C - D)}{n^2 (m - 1)}, \quad \text{where } m = \min(r, c)
    $$

    ---

    ### 3. Hypothesis Testing and Asymptotic Variance

    Under the null hypothesis $H_0$ that $X$ and $Y$ are independent (no monotonic association), the expectation is $\mathbb{E}[\tau_b] = 0$.

    For samples without ties, the exact null variance is:

    $$
    \sigma_0^2 = \operatorname{Var}(\tau) = \frac{2(2n + 5)}{9n(n - 1)}
    $$

    When ties are present, the variance formula adjusts for the tie multiplicities $t_k$ and $u_m$:

    $$
    v_0 = \frac{n(n - 1)(2n + 5) - \sum t_k(t_k - 1)(2t_k + 5) - \sum u_m(u_m - 1)(2u_m + 5)}{18}
    $$

    $$
    z = \frac{C - D}{\sqrt{v_0}}
    $$

    The two-tailed $p$-value is then computed from the standard normal cumulative distribution function $\Phi$:

    $$
    p = 2 \cdot \left(1 - \Phi(|z|)\right)
    $$

    ---

    ### 4. Mathematical Comparison: Pearson vs. Spearman vs. Kendall

    | Metric | Formula Basis | Assumptions | Outlier Sensitivity | Tie Handling | Primary Domain |
    | :--- | :--- | :--- | :--- | :--- | :--- |
    | **Pearson $r$** | Covariance / $(\sigma_x \sigma_y)$ | Linear, Bivariate Normal, Continuous | Extremely High (unbounded influence) | Trivial (continuous) | Continuous linear models, OLS regression |
    | **Spearman $\rho$** | Pearson $r$ on Ranks | Monotonic relationship | Moderate (bounded by rank extremes) | Average ranks with correction | Continuous monotonic data, non-normal distributions |
    | **Kendall $\tau_b$** | Pairwise Concordance / Denominator | Monotonic, Ordinal scale | Minimal (pairwise swap count) | Explicit tie adjustment $\tau_b$ | Ordinal survey data, LLM alignment, small samples |
    """)
    return


@app.cell
def _(pd):
    # Raw Survey Observations: Job Satisfaction (X) vs Work-Life Balance (Y)
    _observations = [
        (5, 4),
        (3, 3),
        (4, 4),
        (2, 2),
        (1, 1),
        (4, 3),
        (5, 5),
        (3, 2),
        (2, 3),
        (4, 4),
        (5, 5),
        (3, 3),
        (1, 2),
        (2, 1),
        (4, 5),
        (3, 2),
        (5, 4),
        (1, 2),
        (2, 3),
        (3, 3),
    ]

    df_sample = pd.DataFrame(_observations, columns=["Job Satisfaction", "Work-Life Balance"])
    return (df_sample,)


@app.cell
def _(df_sample, go, make_subplots, mo, np, stats):
    # Interactive Visualizations Cell
    # Subplot 1: Jittered scatter plot of the survey observations (Job Satisfaction vs Work-Life Balance)
    # Subplot 2: Robustness breakdown curve: Pearson r vs Spearman rho vs Kendall tau-b under an injected extreme outlier

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "Ordinal Survey Observations (with Jitter & Concordance)",
            "Outlier Breakdown: Pearson vs. Spearman vs. Kendall",
        ),
        horizontal_spacing=0.14,
    )

    # Subplot 1: Jittered Scatter
    np.random.seed(42)
    _x_vals = df_sample["Job Satisfaction"].values
    _y_vals = df_sample["Work-Life Balance"].values
    _jitter_x = _x_vals + np.random.uniform(-0.12, 0.12, size=len(_x_vals))
    _jitter_y = _y_vals + np.random.uniform(-0.12, 0.12, size=len(_y_vals))

    _fig.add_trace(
        go.Scatter(
            x=_jitter_x,
            y=_jitter_y,
            mode="markers+text",
            marker=dict(
                size=12,
                color=_x_vals + _y_vals,
                colorscale="Purples",
                showscale=False,
                line=dict(width=1.5, color="#4a154b"),
            ),
            text=[f"({x}, {y})" for x, y in zip(_x_vals, _y_vals)],
            textposition="top center",
            textfont=dict(size=8, color="#555555"),
            name="Survey Pairs",
            hovertemplate="Job Satisfaction: %{x:.0f}<br>Work-Life Balance: %{y:.0f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Trend line for subplot 1
    _slope, _intercept, _, _, _ = stats.linregress(_x_vals, _y_vals)
    _line_x = np.array([1, 5])
    _line_y = _slope * _line_x + _intercept
    _fig.add_trace(
        go.Scatter(
            x=_line_x,
            y=_line_y,
            mode="lines",
            line=dict(color="#6366f1", width=2.5, dash="dash"),
            name="Linear Fit",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Outlier Stress Test
    # Simulate a clean bivariate normal sample (n=25) with true correlation ~ 0.85
    # Then displace one single data point (point 0) along y from +3 to -50
    np.random.seed(101)
    _base_x = np.linspace(1, 10, 25)
    _base_y = _base_x + np.random.normal(0, 1.0, 25)

    _outlier_displacements = np.linspace(10, -80, 40)
    _pearson_vals = []
    _spearman_vals = []
    _kendall_vals = []

    for _disp in _outlier_displacements:
        _curr_x = _base_x.copy()
        _curr_y = _base_y.copy()
        _curr_y[0] = _disp  # inject extreme outlier
        _p_val, _ = stats.pearsonr(_curr_x, _curr_y)
        _s_val, _ = stats.spearmanr(_curr_x, _curr_y)
        _k_val, _ = stats.kendalltau(_curr_x, _curr_y)
        _pearson_vals.append(_p_val)
        _spearman_vals.append(_s_val)
        _kendall_vals.append(_k_val)

    _fig.add_trace(
        go.Scatter(
            x=_outlier_displacements,
            y=_pearson_vals,
            mode="lines",
            line=dict(color="#ef4444", width=3),
            name="Pearson r",
        ),
        row=1,
        col=2,
    )

    _fig.add_trace(
        go.Scatter(
            x=_outlier_displacements,
            y=_spearman_vals,
            mode="lines",
            line=dict(color="#f59e0b", width=2.5, dash="dot"),
            name="Spearman rho",
        ),
        row=1,
        col=2,
    )

    _fig.add_trace(
        go.Scatter(
            x=_outlier_displacements,
            y=_kendall_vals,
            mode="lines",
            line=dict(color="#10b981", width=3),
            name="Kendall tau-b",
        ),
        row=1,
        col=2,
    )

    _fig.update_layout(
        template="plotly_white",
        height=500,
        title=dict(
            text="Kendall's Tau-b: Ordinal Agreement & Outlier Resistance Dynamics",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
        margin=dict(l=50, r=50, t=80, b=80),
    )

    _fig.update_xaxes(
        title_text="Job Satisfaction (1-5)",
        tickvals=[1, 2, 3, 4, 5],
        row=1,
        col=1,
    )
    _fig.update_yaxes(
        title_text="Work-Life Balance (1-5)",
        tickvals=[1, 2, 3, 4, 5],
        row=1,
        col=1,
    )

    _fig.update_xaxes(
        title_text="Outlier Y-Coordinate (Perturbation)",
        autorange="reversed",
        row=1,
        col=2,
    )
    _fig.update_yaxes(
        title_text="Correlation Coefficient Value",
        range=[-0.8, 1.05],
        row=1,
        col=2,
    )

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Step-by-step calculation from scratch**: An exhaustive counting of concordant pairs ($C$), discordant pairs ($D$), ties on $X$ ($T_X$), ties on $Y$ ($T_Y$), and joint ties ($T_{XY}$), manually computing $\tau_a$, $\tau_b$, asymptotic standard error, $z$-score, and $p$-value, directly verified against `scipy.stats.kendalltau`.
    2. **LLM-as-a-Judge Alignment Matrix**: 5 evaluators (Human Expert, GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, and Llama-3-70B) evaluate and rank 10 complex model responses. We construct the pairwise Kendall's $\tau_b$ correlation matrix and evaluate which automated evaluator exhibits the highest ranking fidelity with human ground truth.
    """)
    return


@app.cell
def _(df_sample, mo, np, pd, stats):
    # Example 1: Pure Python manual computation of C, D, T_X, T_Y, T_XY, and Tau-b
    _x = df_sample["Job Satisfaction"].to_numpy()
    _y = df_sample["Work-Life Balance"].to_numpy()
    _n = len(_x)

    _c = 0
    _d = 0
    _tx = 0
    _ty = 0
    _txy = 0

    for _i in range(_n):
        for _j in range(_i + 1, _n):
            _dx = _x[_i] - _x[_j]
            _dy = _y[_i] - _y[_j]
            _prod = _dx * _dy

            if _prod > 0:
                _c += 1
            elif _prod < 0:
                _d += 1
            elif _dx == 0 and _dy == 0:
                _txy += 1
            elif _dx == 0:
                _tx += 1
            elif _dy == 0:
                _ty += 1

    _total_pairs = _n * (_n - 1) // 2

    # Tau-a
    _tau_a = (_c - _d) / _total_pairs

    # Tau-b
    _denom_x = _c + _d + _tx
    _denom_y = _c + _d + _ty
    _tau_b_scratch = (_c - _d) / np.sqrt(_denom_x * _denom_y)

    # Scipy reference verification
    _scipy_tau, _scipy_p = stats.kendalltau(_x, _y)

    # Hypothesis test z-statistic using standard tie correction
    # Tie counts
    _, _counts_x = np.unique(_x, return_counts=True)
    _, _counts_y = np.unique(_y, return_counts=True)

    _t_term = np.sum(_counts_x * (_counts_x - 1) * (2 * _counts_x + 5))
    _u_term = np.sum(_counts_y * (_counts_y - 1) * (2 * _counts_y + 5))
    _v0 = (_n * (_n - 1) * (2 * _n + 5) - _t_term - _u_term) / 18.0
    _z_stat = (_c - _d) / np.sqrt(_v0)
    _p_val_scratch = 2.0 * (1.0 - stats.norm.cdf(np.abs(_z_stat)))

    _summary_table = pd.DataFrame(
        [
            {"Metric": "Sample Size (n)", "Computed Value": str(_n), "Formula / Description": "Number of bivariate observations"},
            {"Metric": "Total Unique Pairs", "Computed Value": str(_total_pairs), "Formula / Description": "n(n - 1) / 2"},
            {"Metric": "Concordant Pairs (C)", "Computed Value": str(_c), "Formula / Description": "(x_i - x_j)(y_i - y_j) > 0"},
            {"Metric": "Discordant Pairs (D)", "Computed Value": str(_d), "Formula / Description": "(x_i - x_j)(y_i - y_j) < 0"},
            {"Metric": "Tied on X only (T_X)", "Computed Value": str(_tx), "Formula / Description": "x_i == x_j and y_i != y_j"},
            {"Metric": "Tied on Y only (T_Y)", "Computed Value": str(_ty), "Formula / Description": "x_i != x_j and y_i == y_j"},
            {"Metric": "Tied on Both (T_XY)", "Computed Value": str(_txy), "Formula / Description": "x_i == x_j and y_i == y_j"},
            {"Metric": "Kendall's Tau-a", "Computed Value": f"{_tau_a:.4f}", "Formula / Description": "(C - D) / N_pairs (no tie adjustment)"},
            {"Metric": "Kendall's Tau-b (Scratch)", "Computed Value": f"{_tau_b_scratch:.4f}", "Formula / Description": "(C - D) / sqrt((C + D + T_X)(C + D + T_Y))"},
            {"Metric": "Kendall's Tau-b (SciPy)", "Computed Value": f"{_scipy_tau:.4f}", "Formula / Description": "scipy.stats.kendalltau(x, y)"},
            {"Metric": "Asymptotic z-Score", "Computed Value": f"{_z_stat:.4f}", "Formula / Description": "(C - D) / sqrt(Var_0)"},
            {"Metric": "Two-tailed p-value", "Computed Value": f"{_p_val_scratch:.6f}", "Formula / Description": "2 * (1 - Phi(|z|))"},
        ]
    )

    return (mo.ui.table(_summary_table),)


@app.cell
def _(mo, np, pd, stats):
    # Example 2: LLM Evaluation Agreement Matrix (Evaluating 5 Judges Across 10 Model Outputs)
    # Ranks assigned to 10 generated responses (1 = Best response, 10 = Worst response)
    _judges_data = {
        "Human Expert": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "GPT-4o": [1, 2, 4, 3, 5, 7, 6, 8, 10, 9],
        "Claude-3.5-Sonnet": [1, 2, 3, 5, 4, 6, 8, 7, 9, 10],
        "Gemini-1.5-Pro": [2, 1, 4, 3, 6, 5, 7, 9, 8, 10],
        "Llama-3-70B": [3, 1, 5, 2, 7, 4, 8, 6, 10, 9],
    }
    _df_judges = pd.DataFrame(_judges_data, index=[f"Prompt {i+1}" for i in range(10)])

    _judge_names = list(_judges_data.keys())
    _n_judges = len(_judge_names)
    _tau_matrix = np.zeros((_n_judges, _n_judges))
    _pval_matrix = np.zeros((_n_judges, _n_judges))

    for _i in range(_n_judges):
        for _j in range(_n_judges):
            _t, _p = stats.kendalltau(_df_judges[_judge_names[_i]], _df_judges[_judge_names[_j]])
            _tau_matrix[_i, _j] = _t
            _pval_matrix[_i, _j] = _p

    _df_tau_matrix = pd.DataFrame(_tau_matrix, index=_judge_names, columns=_judge_names).round(4)

    # Format human alignment leaderboard
    _alignment_scores = []
    for _judge in _judge_names[1:]:
        _t = _df_tau_matrix.loc["Human Expert", _judge]
        _alignment_scores.append({
            "Evaluator Model": _judge,
            "Kendall Tau-b with Human": f"{_t:.4f}",
            "Agreement Status": "High Alignment (> 0.80)" if _t >= 0.80 else "Moderate Alignment",
        })

    _df_leaderboard = pd.DataFrame(_alignment_scores).sort_values("Kendall Tau-b with Human", ascending=False)

    return (
        mo.md("#### Pairwise Kendall's Tau-b Correlation Matrix (Inter-Evaluator Agreement)"),
        mo.ui.table(_df_tau_matrix),
        mo.md("#### Human Ground-Truth Alignment Leaderboard"),
        mo.ui.table(_df_leaderboard),
    )


if __name__ == "__main__":
    app.run()
