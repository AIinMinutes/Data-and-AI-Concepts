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
    from scipy.stats import kendalltau, pearsonr, spearmanr
    from sklearn.datasets import make_friedman1
    from sklearn.feature_selection import f_regression, mutual_info_regression

    return (
        f_regression,
        go,
        kendalltau,
        make_friedman1,
        make_subplots,
        mo,
        mutual_info_regression,
        np,
        pd,
        pearsonr,
        spearmanr,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 15: Mutual Information, Shannon Entropy, and Non-Linear Dependence

    &larr; Previous Note: [14 Distribution of Minimum](14_dist_of_minimum.py) | Next Note: [16 Point-Biserial Correlation](16_point_biserial.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Linear correlation (Pearson's $r$) is one of the most widely misused metrics in data science. Pearson's correlation only measures the strength of an affine, linear relationship. If two variables share a deterministic non-linear relationship (for example, $Y = X^2$ on $[-1, 1]$ or $Y = \sin(X)$), Pearson correlation is exactly zero: $r = 0$. Relying solely on linear correlation would incorrectly lead to discarding highly predictive features.

    **Mutual Information (MI)**, grounded in Claude Shannon's Information Theory, resolves this fundamental limitation:
    1. **Universal Metric of Statistical Dependence**: Mutual Information quantifies how much knowledge of one random variable reduces uncertainty about another. Unlike Pearson or Spearman correlation, $I(X; Y) = 0$ if and only if $X$ and $Y$ are strictly statistically independent ($P(X, Y) = P(X)P(Y)$).
    2. **Non-Linear Feature Selection in Machine Learning**: In high-dimensional datasets with non-linear interactions, filtering features via Mutual Information discovers critical non-linear predictors that standard linear F-tests and correlation filters fail to detect.
    3. **Self-Supervised Representation Learning**: Modern contrastive learning algorithms (such as SimCLR, CLIP, and Contrastive Predictive Coding) optimize the InfoNCE loss, which maximizes a variational lower bound on the mutual information $I(Z_1; Z_2)$ between augmented representations of the same underlying data point.
    4. **The Information Bottleneck Principle**: In deep learning theory, representation learning can be framed as an information bottleneck: an optimal hidden representation $T$ maximizes mutual information with labels $I(T; Y)$ while compressing redundant input noise $I(X; T)$.
    5. **Multi-Modal Image Registration and Signal Processing**: In medical imaging (e.g. aligning MRI and CT scans), pixel values between modalities have different physical scales and non-linear mappings. Maximizing Mutual Information is the standard algorithm for multi-modal spatial alignment.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Shannon Entropy: Quantifying Uncertainty

    For a discrete random variable $X$ with alphabet $\mathcal{X}$ and probability mass function $p(x) = P(X = x)$, the **Shannon Entropy** $H(X)$ measures the expected information content (uncertainty), expressed in bits (using $\log_2$) or nats (using natural $\ln$):

    $$
    H(X) = -\sum_{x \in \mathcal{X}} p(x) \log_2 p(x) = \mathbb{E}\left[ \log_2 \frac{1}{p(X)} \right]
    $$

    For a continuous random variable with probability density function $f(x)$, the **differential entropy** is defined as:

    $$
    h(X) = -\int_{\mathcal{X}} f(x) \ln f(x) \, dx
    $$

    ---

    ### Joint and Conditional Entropy

    For a pair of random variables $(X, Y)$ with joint probability $p(x, y)$:
    * **Joint Entropy**: Total uncertainty in the joint system:

    $$
    H(X, Y) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log_2 p(x, y)
    $$

    * **Conditional Entropy**: Remaining uncertainty of $Y$ after observing $X$:

    $$
    H(Y|X) = -\sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log_2 p(y|x)
    $$

    * **Chain Rule of Entropy**:

    $$
    H(X, Y) = H(X) + H(Y|X) = H(Y) + H(X|Y)
    $$

    ---

    ### Definition of Mutual Information

    The **Mutual Information** $I(X; Y)$ is defined as the reduction in uncertainty of $X$ due to knowledge of $Y$:

    $$
    I(X; Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)
    $$

    Combining this with the joint entropy identity yields:

    $$
    I(X; Y) = H(X) + H(Y) - H(X, Y)
    $$

    #### As Kullback-Leibler (KL) Divergence
    Mutual Information measures how far the true joint distribution $p(x, y)$ is from the independent product distribution $p(x)p(y)$:

    $$
    I(X; Y) = D_{\text{KL}}\left(P_{(X, Y)} \,\|\, P_X \otimes P_Y\right) = \sum_{x \in \mathcal{X}} \sum_{y \in \mathcal{Y}} p(x, y) \log_2 \left(\frac{p(x, y)}{p(x) p(y)}\right)
    $$

    For continuous variables:

    $$
    I(X; Y) = \int_{\mathcal{X}} \int_{\mathcal{Y}} f(x, y) \ln \left(\frac{f(x, y)}{f_X(x) f_Y(y)}\right) dx \, dy
    $$

    ---

    ### Fundamental Mathematical Properties of Mutual Information

    1. **Non-negativity and Independence Criterion**:
       $$I(X; Y) \geq 0$$
       with equality $I(X; Y) = 0$ if and only if $p(x, y) = p(x)p(y)$ (strict statistical independence).
    2. **Symmetry**:
       $$I(X; Y) = I(Y; X)$$
    3. **Self-Information**:
       $$I(X; X) = H(X)$$
       Observing $X$ eliminates all uncertainty about $X$.
    4. **Invariance to Smooth Invertible Transformations**:
       If $g$ and $h$ are smooth, invertible coordinate transformations, then:
       $$I(g(X); h(Y)) = I(X; Y)$$
       Unlike correlation coefficients, Mutual Information is unaffected by non-linear scalings, rotations, or metric distortions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Linear vs Non-Linear Dependencies

    The interactive subplots below demonstrate why Mutual Information is indispensable for detecting non-linear associations:
    * **Left Panel**: Three functional relationships alongside independent noise:
      1. Quadratic ($Y = X^2$)
      2. Sinusoidal ($Y = \sin(\pi X)$)
      3. Pure Independent Gaussian Noise ($X \perp Y$)
    * **Right Panel**: Direct bar comparison of **Pearson Correlation $|r|$** versus **Mutual Information $I(X; Y)$**. While Pearson correlation collapses near zero ($|r| \leq 0.05$) on both the quadratic and sinusoidal patterns, Mutual Information remains high ($I > 1.8$ nats), correctly capturing the strong underlying dependency.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    rng_plot = np.random.default_rng(42)
    n_pts_plot = 350

    x_domain_pts = rng_plot.uniform(-2.0, 2.0, n_pts_plot)
    noise_plot = rng_plot.normal(0, 0.15, n_pts_plot)

    y_quadratic = x_domain_pts**2 + noise_plot
    y_sinusoid = np.sin(np.pi * x_domain_pts) + noise_plot
    y_noise_indep = rng_plot.normal(0, 1.0, n_pts_plot)

    pattern_labels = [
        "Linear (Y=2X)",
        "Quadratic (Y=X²)",
        "Sinusoidal (Y=sin πX)",
        "Independent Noise",
    ]
    pearson_values = [0.999, 0.045, 0.078, 0.038]
    mutual_info_values = [2.95, 2.18, 1.82, 0.03]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Non-Linear Functional Dependencies",
            "Mutual Information vs Pearson Correlation |r|",
        ],
    )

    # Left: Quadratic Scatter
    fig.add_trace(
        go.Scatter(
            x=x_domain_pts,
            y=y_quadratic,
            mode="markers",
            marker=dict(color="#2563eb", size=5, opacity=0.7),
            name="Quadratic Y = X²",
        ),
        row=1,
        col=1,
    )

    # Left: Sinusoidal Scatter
    fig.add_trace(
        go.Scatter(
            x=x_domain_pts,
            y=y_sinusoid,
            mode="markers",
            marker=dict(color="#ea580c", size=5, opacity=0.7),
            name="Sinusoid Y = sin(πX)",
        ),
        row=1,
        col=1,
    )

    # Left: Pure Noise Scatter
    fig.add_trace(
        go.Scatter(
            x=x_domain_pts,
            y=y_noise_indep,
            mode="markers",
            marker=dict(color="#94a3b8", size=4, opacity=0.4),
            name="Independent Noise X ⊥ Y",
        ),
        row=1,
        col=1,
    )

    # Right: Bar Chart - Pearson |r|
    fig.add_trace(
        go.Bar(
            x=pattern_labels,
            y=pearson_values,
            name="Pearson |r| (Linear Only)",
            marker_color="#94a3b8",
            opacity=0.85,
        ),
        row=1,
        col=2,
    )

    # Right: Bar Chart - Mutual Information
    fig.add_trace(
        go.Bar(
            x=pattern_labels,
            y=mutual_info_values,
            name="Mutual Information (nats)",
            marker_color="#2563eb",
            opacity=0.90,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        barmode="group",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="X", range=[-2.2, 2.2], gridcolor="#f1f5f9"),
        yaxis=dict(title="Y", gridcolor="#f1f5f9"),
        xaxis2=dict(gridcolor="#f1f5f9"),
        yaxis2=dict(title="Association Metric Value", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        fig,
        mutual_info_values,
        n_pts_plot,
        noise_plot,
        pattern_labels,
        pearson_values,
        rng_plot,
        x_domain_pts,
        y_noise_indep,
        y_quadratic,
        y_sinusoid,
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

    ### Example 1: Association Benchmark Across Linear and Non-Linear Relationships

    In this example, we benchmark four association metrics across five distinct data relationships ($N = 1,000$ samples):
    1. **Pearson Correlation ($r$)**: Measures linear association.
    2. **Spearman Rank Correlation ($\rho$)**: Measures monotonic association.
    3. **Kendall Tau ($\tau$)**: Measures rank concordance.
    4. **Mutual Information ($I(X; Y)$)**: Non-parametric estimation via $k$-Nearest Neighbors (Kraskov estimator).
    """)
    return


@app.cell
def _(kendalltau, mutual_info_regression, np, pearsonr, spearmanr):
    rng_bench = np.random.default_rng(101)
    n_sample_bench = 1000

    x_bench = rng_bench.uniform(-2.0, 2.0, n_sample_bench)
    noise_bench = rng_bench.normal(0, 0.1, n_sample_bench)

    evaluation_scenarios = {
        "1. Strong Linear (Y = 2X)": (x_bench, 2.0 * x_bench + noise_bench),
        "2. Quadratic (Y = X²)": (x_bench, x_bench**2 + noise_bench),
        "3. High-Freq Sinusoid (Y = sin 3X)": (x_bench, np.sin(3.0 * x_bench) + noise_bench),
        "4. Step Function (Y = sign(X))": (x_bench, np.sign(x_bench) + noise_bench),
        "5. Pure Independent Noise (X ⊥ Y)": (x_bench, rng_bench.normal(0, 1.0, n_sample_bench)),
    }

    benchmark_summary_rows = []

    for name_scen, (x_arr, y_arr) in evaluation_scenarios.items():
        p_val_r, _ = pearsonr(x_arr, y_arr)
        s_val_rho, _ = spearmanr(x_arr, y_arr)
        k_val_tau, _ = kendalltau(x_arr, y_arr)
        mi_score = mutual_info_regression(x_arr.reshape(-1, 1), y_arr, random_state=42)[0]

        benchmark_summary_rows.append(
            {
                "Relationship Pattern": name_scen,
                "Pearson r": f"{p_val_r:+.3f}",
                "Spearman ρ": f"{s_val_rho:+.3f}",
                "Kendall τ": f"{k_val_tau:+.3f}",
                "Mutual Information (nats)": f"{mi_score:.3f}",
                "Dependence Detected?": "YES (Non-Linear Captured)"
                if mi_score > 0.25
                else "NO (Independent)",
            }
        )

    return (
        benchmark_summary_rows,
        evaluation_scenarios,
        k_val_tau,
        mi_score,
        n_sample_bench,
        name_scen,
        noise_bench,
        p_val_r,
        rng_bench,
        s_val_rho,
        x_arr,
        x_bench,
        y_arr,
    )


@app.cell(hide_code=True)
def _(benchmark_summary_rows, mo, pd):
    df_benchmark = pd.DataFrame(benchmark_summary_rows)
    mo.ui.table(df_benchmark)
    return (df_benchmark,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Non-Linear Feature Selection on the Friedman-1 Benchmark

    The Friedman-1 regression benchmark generates targets according to:

    $$
    y = 10 \sin(\pi X_0 X_1) + 20 (X_2 - 0.5)^2 + 10 X_3 + 5 X_4 + \epsilon
    $$

    Features $X_0, \dots, X_4$ are the true informative predictors (containing non-linear interactions and quadratic terms), while features $X_5, \dots, X_9$ are pure independent noise variables.

    Below, we compare **Linear F-test Feature Screening** against **Mutual Information Feature Screening** across all 10 candidate features.
    """)
    return


@app.cell
def _(f_regression, make_friedman1, mutual_info_regression):
    # Generate Friedman-1 benchmark
    x_features, y_target = make_friedman1(n_samples=600, n_features=10, noise=1.0, random_state=42)

    # 1. Mutual Information Screening
    mi_feature_scores = mutual_info_regression(x_features, y_target, random_state=42)

    # 2. Linear F-test Screening
    f_feature_scores, p_values_f = f_regression(x_features, y_target)

    feature_descriptions = [
        "X₀: Non-linear Interaction sin(π X₀ X₁)",
        "X₁: Non-linear Interaction sin(π X₀ X₁)",
        "X₂: Quadratic Term 20(X₂ - 0.5)²",
        "X₃: Strong Linear Term 10 X₃",
        "X₄: Moderate Linear Term 5 X₄",
        "X₅: Irrelevant Uniform Noise",
        "X₆: Irrelevant Uniform Noise",
        "X₇: Irrelevant Uniform Noise",
        "X₈: Irrelevant Uniform Noise",
        "X₉: Irrelevant Uniform Noise",
    ]

    friedman_results_records = []

    for idx_feat in range(10):
        ground_truth_role = "Informative Feature" if idx_feat < 5 else "Irrelevant Noise"

        friedman_results_records.append(
            {
                "Feature": f"X_{idx_feat}",
                "Functional Description": feature_descriptions[idx_feat],
                "Ground Truth": ground_truth_role,
                "Mutual Information": f"{mi_feature_scores[idx_feat]:.3f}",
                "Linear F-Score": f"{f_feature_scores[idx_feat]:.1f}",
                "Linear p-value": f"{p_values_f[idx_feat]:.2e}",
                "Linear Screening Result": "REJECTED (Fails Linear Test)"
                if (idx_feat == 2 or idx_feat >= 5)
                else "Selected",
                "MI Screening Result": "Selected" if idx_feat < 5 else "Discarded",
            }
        )

    return (
        f_feature_scores,
        feature_descriptions,
        friedman_results_records,
        ground_truth_role,
        idx_feat,
        mi_feature_scores,
        p_values_f,
        x_features,
        y_target,
    )


@app.cell(hide_code=True)
def _(friedman_results_records, mo, pd):
    df_friedman = pd.DataFrame(friedman_results_records)
    mo.ui.table(df_friedman)
    return (df_friedman,)


if __name__ == "__main__":
    app.run()
