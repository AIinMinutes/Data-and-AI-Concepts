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
    from scipy.stats import chi2_contingency

    return chi2_contingency, go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 18: Cramér's V, Chi-Square Independence Tests, and Nominal Association

    &larr; Previous Note: [17 Jensen Inequality](17_jensen_inequality.py) | Next Note: [19 Kendall Tau-b](19_kendalltaub.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In tabular data science, machine learning, and survey analytics, features are frequently nominal categorical variables with no inherent mathematical ordering (e.g. Job Role, Country of Origin, Operating System, Device Brand, Preferred Programming Language). For such variables, numerical correlation metrics like Pearson's $r$ or Spearman's $\rho$ cannot be computed.

    Understanding the **Pearson Chi-Square test of independence** and **Cramér's V** is essential:
    1. **Effect Size Beyond p-Values**: While the Chi-Square test produces a $p$-value indicating whether an association exists, the test statistic $\chi^2$ scales linearly with sample size $n$. In large datasets ($n \geq 100,000$), virtually any two variables achieve statistical significance ($p < 10^{-50}$), even when the practical relationship is utterly trivial. **Cramér's V** normalizes $\chi^2$ by sample size and table dimensions, yielding an interpretable effect size bounded in $[0, 1]$.
    2. **Categorical Feature Selection and Multicollinearity**: When preparing datasets for gradient boosting algorithms (CatBoost, LightGBM, XGBoost), pairwise Cramér's V matrices function as the categorical analogue of the Pearson correlation matrix, diagnosing redundant features and preventing multicollinearity among categorical variables.
    3. **Hypothesis Testing on Contingency Tables**: In clinical trials and marketing A/B tests, contingency tables compare conversion rates, adverse reactions, and user preferences across demographic segments.
    4. **Finite-Sample Bias Correction (Bergsma)**: Standard Cramér's V suffers from a positive upward bias in small samples. Bergsma's bias-corrected Cramér's $\tilde{V}$ subtracts the degrees of freedom expectation, preventing false alarms in sparse tables.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Contingency Tables and Expected Frequencies

    Consider two nominal categorical variables: $X$ with $r$ distinct categories and $Y$ with $c$ distinct categories. A cross-tabulation of $n$ observations produces an $r \times c$ **contingency table** with observed cell counts $O_{ij}$.

    The row totals $R_i$ and column totals $C_j$ are defined as:

    $$
    R_i = \sum_{j=1}^c O_{ij}, \quad C_j = \sum_{i=1}^r O_{ij}, \quad n = \sum_{i=1}^r \sum_{j=1}^c O_{ij}
    $$

    Under the null hypothesis $H_0$ of statistical independence ($P(X = i, Y = j) = P(X = i) P(Y = j)$), the expected frequency in cell $(i, j)$ is:

    $$
    E_{ij} = n \left(\frac{R_i}{n}\right) \left(\frac{C_j}{n}\right) = \frac{R_i C_j}{n}
    $$

    ---

    ### Pearson's Chi-Square Test Statistic ($\chi^2$)

    The Chi-Square test statistic measures the aggregated squared discrepancy between observed frequencies $O_{ij}$ and expected frequencies $E_{ij}$:

    $$
    \chi^2 = \sum_{i=1}^r \sum_{j=1}^c \frac{(O_{ij} - E_{ij})^2}{E_{ij}}
    $$

    Under $H_0$ and with adequate sample size per cell (Cochran's rule: $E_{ij} \geq 5$ in at least $80\%$ of cells), $\chi^2$ asymptotically follows a Chi-Square distribution with degrees of freedom:

    $$
    df = (r - 1)(c - 1)
    $$

    ---

    ### Definition of Cramér's V

    Because $\chi^2 \propto n$, the raw test statistic cannot be used as an association metric. Harald Cramér introduced **Cramér's V** to normalize the statistic:

    $$
    V = \sqrt{\frac{\chi^2}{n \cdot \min(r - 1, c - 1)}}
    $$

    where:
    * $\chi^2$ is Pearson's Chi-Square statistic.
    * $n$ is total sample count.
    * $r$ is number of rows (categories of $X$).
    * $c$ is number of columns (categories of $Y$).

    #### Properties and Boundedness
    * **Range**: $0 \leq V \leq 1$.
    * **Special Case ($2 \times 2$ table)**: For $r = c = 2$, $\min(r - 1, c - 1) = 1$, and Cramér's V reduces exactly to the absolute value of the **phi coefficient** $|\phi| = \sqrt{\chi^2 / n}$.
    * **$V = 0$**: Complete statistical independence ($O_{ij} = E_{ij}$ for all cells).
    * **$V = 1$**: Perfect deterministic association. Knowing category $X$ completely predicts $Y$ (or vice versa).

    #### Standard Effect Size Interpretation (Rea & Parker / Cohen)
    * $0.00 \leq V < 0.10$: Negligible association.
    * $0.10 \leq V < 0.20$: Weak association.
    * $0.20 \leq V < 0.40$: Moderate association.
    * $0.40 \leq V < 0.60$: Strong association.
    * $V \geq 0.60$: Very strong association.

    ---

    ### Bergsma's Bias-Corrected Cramér's V

    Standard Cramér's V tends to overestimate association when sample size $n$ is small or table dimensions $r, c$ are large. Bergsma (2013) proposed an unbiased correction:

    $$
    \tilde{\chi}^2 = \max\left(0, \chi^2 - \frac{(r - 1)(c - 1)}{n - 1}\right)
    $$

    $$
    \tilde{n} = n - \frac{(r - 1)(c - 1)}{n - 1}, \quad \tilde{r} = r - \frac{(r - 1)^2}{n - 1}, \quad \tilde{c} = c - \frac{(c - 1)^2}{n - 1}
    $$

    The **bias-corrected Cramér's V** is:

    $$
    \tilde{V} = \sqrt{\frac{\tilde{\chi}^2 / \tilde{n}}{\min(\tilde{r} - 1, \tilde{c} - 1)}}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Contingency Heatmaps and Categorical Association Matrix

    The interactive subplots below display the practical application of Cramér's V:
    * **Left Panel**: $3 \times 3$ contingency table of job roles versus preferred movie genres ($N = 225$ respondents). Cell annotations show observed counts. Data Scientists show strong preference for Action ($50$), ML Engineers for Drama ($50$), and GenAI Developers for Comedy ($50$), producing a strong association ($V \approx 0.54, p < 10^{-26}$).
    * **Right Panel**: Pairwise Categorical Association Matrix (Cramér's V) across four categorical features: Role, Preferred Genre, Primary Cloud, and Primary Language. This matrix allows data scientists to inspect categorical collinearity identically to numerical correlation matrices.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Left Panel: Role vs Genre Contingency Table
    roles_list = ["Data Scientist", "ML Engineer", "GenAI Dev"]
    genres_list = ["Action", "Comedy", "Drama"]

    contingency_counts = np.array([
        [50, 20, 5],
        [5, 20, 50],
        [5, 50, 20],
    ])

    # Right Panel: 4x4 Categorical Association Matrix (Cramer's V)
    cat_features = ["Job Role", "Movie Genre", "Primary Cloud", "Primary Lang"]
    cramers_matrix = np.array([
        [1.00, 0.54, 0.42, 0.61],
        [0.54, 1.00, 0.08, 0.12],
        [0.42, 0.08, 1.00, 0.48],
        [0.61, 0.12, 0.48, 1.00],
    ])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Contingency Table: Role vs Genre (V = 0.54)",
            "Categorical Association Matrix (Cramér's V)",
        ],
    )

    # Left: Contingency Heatmap
    fig.add_trace(
        go.Heatmap(
            z=contingency_counts,
            x=roles_list,
            y=genres_list,
            colorscale="Viridis",
            text=contingency_counts,
            texttemplate="%{text}",
            textfont=dict(size=14, color="white"),
            showscale=False,
            name="Contingency Counts",
        ),
        row=1,
        col=1,
    )

    # Right: Pairwise Cramer's V Matrix
    fig.add_trace(
        go.Heatmap(
            z=cramers_matrix,
            x=cat_features,
            y=cat_features,
            colorscale="Blues",
            zmin=0.0,
            zmax=1.0,
            text=cramers_matrix,
            texttemplate="%{text:.2f}",
            textfont=dict(size=14),
            showscale=True,
            colorbar=dict(title="Cramér's V", x=1.02),
            name="Cramér's V",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=60, t=60, b=40),
        xaxis=dict(gridcolor="#f1f5f9"),
        yaxis=dict(gridcolor="#f1f5f9"),
        xaxis2=dict(gridcolor="#f1f5f9"),
        yaxis2=dict(gridcolor="#f1f5f9"),
    )

    return (
        cat_features,
        contingency_counts,
        cramers_matrix,
        fig,
        genres_list,
        roles_list,
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

    ### Example 1: Full Chi-Square Test and Cramér's V Computation (Standard vs Bias-Corrected)

    Below, we analyze the $3 \times 3$ contingency table of tech roles vs movie preferences:
    1. Observed frequencies ($O_{ij}$) and expected frequencies under independence ($E_{ij}$).
    2. Chi-Square statistic ($\chi^2$), degrees of freedom, and $p$-value.
    3. Standard Cramér's $V$ versus Bergsma's bias-corrected $\tilde{V}$.
    """)
    return


@app.cell
def _(chi2_contingency, np):
    contingency_data = np.array([
        [50, 20, 5],
        [5, 20, 50],
        [5, 50, 20],
    ])

    # Chi-Square test
    chi2_val, p_value_val, dof_val, expected_arr = chi2_contingency(contingency_data)

    n_total_obs = int(np.sum(contingency_data))
    r_rows, c_cols = contingency_data.shape

    # Standard Cramer's V
    v_standard = float(np.sqrt(chi2_val / (n_total_obs * min(r_rows - 1, c_cols - 1))))

    # Bias-corrected Cramer's V (Bergsma 2013)
    chi2_adj = max(0.0, chi2_val - ((r_rows - 1.0) * (c_cols - 1.0)) / (n_total_obs - 1.0))
    n_adj = n_total_obs - ((r_rows - 1.0) * (c_cols - 1.0)) / (n_total_obs - 1.0)
    r_adj = r_rows - ((r_rows - 1.0) ** 2) / (n_total_obs - 1.0)
    c_adj = c_cols - ((c_cols - 1.0) ** 2) / (n_total_obs - 1.0)
    denom_dim = min(r_adj - 1.0, c_adj - 1.0)
    v_corrected = float(np.sqrt((chi2_adj / n_adj) / denom_dim)) if denom_dim > 0 else 0.0

    chi2_test_results = [
        {
            "Metric / Diagnostic": "Total Sample Size (n)",
            "Value": f"{n_total_obs}",
            "Interpretation": "Sum of all contingency cells",
        },
        {
            "Metric / Diagnostic": "Degrees of Freedom (r-1)(c-1)",
            "Value": f"{dof_val}",
            "Interpretation": "(3 - 1) x (3 - 1) = 4",
        },
        {
            "Metric / Diagnostic": "Pearson Chi-Square Statistic (χ²)",
            "Value": f"{chi2_val:.2f}",
            "Interpretation": "Sum of (O - E)² / E",
        },
        {
            "Metric / Diagnostic": "p-value",
            "Value": f"{p_value_val:.2e}",
            "Interpretation": "Statistically significant association (p < 0.001)",
        },
        {
            "Metric / Diagnostic": "Standard Cramér's V",
            "Value": f"{v_standard:.4f}",
            "Interpretation": "Strong nominal association (V > 0.50)",
        },
        {
            "Metric / Diagnostic": "Bias-Corrected Cramér's V",
            "Value": f"{v_corrected:.4f}",
            "Interpretation": "Bergsma correction for finite-sample stability",
        },
    ]

    return (
        c_adj,
        c_cols,
        chi2_adj,
        chi2_test_results,
        chi2_val,
        contingency_data,
        denom_dim,
        dof_val,
        expected_arr,
        n_adj,
        n_total_obs,
        p_value_val,
        r_adj,
        r_rows,
        v_corrected,
        v_standard,
    )


@app.cell(hide_code=True)
def _(chi2_test_results, mo, pd):
    df_chi2 = pd.DataFrame(chi2_test_results)
    mo.ui.table(df_chi2)
    return (df_chi2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Categorical Feature Collinearity Matrix for Machine Learning

    In tabular datasets with multiple categorical features, strong inter-feature dependencies create categorical multicollinearity, which degrades gradient boosting training efficiency and destabilizes feature importance scores.

    Below, we simulate $N = 1,000$ synthetic customer profiles with four categorical features:
    * **Industry**: Tech, Healthcare, Finance
    * **Operating System**: Linux, macOS, Windows (correlated with Industry)
    * **Subscription Plan**: Enterprise, Pro, Free (correlated with Industry)
    * **Random Survey Color**: Red, Green, Blue (pure independent noise)

    We construct the full $4 \times 4$ pairwise Cramér's V correlation matrix and flag redundant feature pairs ($V > 0.40$).
    """)
    return


@app.cell
def _(chi2_contingency, np, pd):
    rng_feat = np.random.default_rng(303)
    n_records = 1000

    # Industry
    industries = rng_feat.choice(["Tech", "Healthcare", "Finance"], size=n_records, p=[0.4, 0.3, 0.3])

    # OS depends on Industry
    os_choices = []
    for ind in industries:
        if ind == "Tech":
            os_choices.append(rng_feat.choice(["Linux", "macOS", "Windows"], p=[0.45, 0.45, 0.10]))
        elif ind == "Finance":
            os_choices.append(rng_feat.choice(["Linux", "macOS", "Windows"], p=[0.05, 0.15, 0.80]))
        else:
            os_choices.append(rng_feat.choice(["Linux", "macOS", "Windows"], p=[0.10, 0.30, 0.60]))

    # Subscription depends on Industry
    plans = []
    for ind in industries:
        if ind == "Finance":
            plans.append(rng_feat.choice(["Enterprise", "Pro", "Free"], p=[0.70, 0.25, 0.05]))
        elif ind == "Tech":
            plans.append(rng_feat.choice(["Enterprise", "Pro", "Free"], p=[0.40, 0.45, 0.15]))
        else:
            plans.append(rng_feat.choice(["Enterprise", "Pro", "Free"], p=[0.20, 0.40, 0.40]))

    # Random noise feature
    colors = rng_feat.choice(["Red", "Green", "Blue"], size=n_records)

    cat_df = pd.DataFrame({
        "Industry": industries,
        "OS": os_choices,
        "Plan": plans,
        "Color": colors,
    })

    feature_names = ["Industry", "OS", "Plan", "Color"]
    v_matrix_results = []

    for f1 in feature_names:
        row_dict = {"Feature": f1}
        for f2 in feature_names:
            if f1 == f2:
                row_dict[f2] = "1.000"
            else:
                c_table = pd.crosstab(cat_df[f1], cat_df[f2]).values
                chi2, _, _, _ = chi2_contingency(c_table)
                r_num, c_num = c_table.shape
                v_score = float(np.sqrt(chi2 / (n_records * min(r_num - 1, c_num - 1))))
                row_dict[f2] = f"{v_score:.3f}"
        v_matrix_results.append(row_dict)

    return (
        cat_df,
        colors,
        f1,
        f2,
        feature_names,
        ind,
        industries,
        n_records,
        os_choices,
        plans,
        rng_feat,
        row_dict,
        v_matrix_results,
    )


@app.cell(hide_code=True)
def _(mo, pd, v_matrix_results):
    df_cat_matrix = pd.DataFrame(v_matrix_results)
    mo.ui.table(df_cat_matrix)
    return (df_cat_matrix,)


if __name__ == "__main__":
    app.run()
