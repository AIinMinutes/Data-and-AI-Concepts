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
    # Note 11: Empirical Cumulative Distribution Function (ECDF) and Non-Parametric Inference

    &larr; Previous Note: [10 Chebyshev Inequality](10_chebyshev_inequality.py) | Next Note: [12 Multivariate Normal Distribution](12_multivariate_normal_distribution.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Parametric probability distributions assume idealized mathematical forms (such as Gaussian, Gamma, or Poisson). In real-world data science, machine learning operations (MLOps), and statistical inference, however, true data-generating distributions are rarely known a priori.

    The **Empirical Cumulative Distribution Function (ECDF)** is the foundational tool of non-parametric statistics:
    1. **Zero-Assumption Distribution Estimation**: Unlike histograms or kernel density estimators (KDE), which are sensitive to bin width, bin placement, and bandwidth hyperparameter choices, the ECDF requires no tuning parameters and preserves all original sample information.
    2. **The Fundamental Theorem of Statistics (Glivenko-Cantelli)**: The ECDF is guaranteed to converge uniformly to the true population cumulative distribution function across the entire real line as sample size grows: $\sup_{x} |\hat{F}_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0$.
    3. **Simultaneous Confidence Envelopes (DKW Inequality)**: The Dvoretzky-Kiefer-Wolfowitz inequality allows the construction of exact, non-parametric confidence bands that envelope the entire unknown population CDF with pre-specified probability (e.g. $95\%$).
    4. **Data Drift and Covariate Shift Detection in MLOps**: The two-sample Kolmogorov-Smirnov (KS) test evaluates the maximum vertical discrepancy between the baseline training ECDF and production serving ECDF, serving as an industry standard for real-time model monitoring and drift alerting.
    5. **Quantile and Risk Estimation**: Inverting the ECDF provides direct estimates of empirical percentiles, medians, interquartile ranges, and Value-at-Risk (VaR) in quantitative finance.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Population Cumulative Distribution Function (CDF)

    For any random variable $X$, the cumulative distribution function $F_X(x)$ represents the probability that $X$ takes a value less than or equal to $x$:

    $$
    F_X(x) = P(X \leq x)
    $$

    #### Fundamental Mathematical Properties
    * **Non-decreasing**: If $x_1 \leq x_2$, then $F_X(x_1) \leq F_X(x_2)$.
    * **Asymptotic Limits**: $\lim_{x \to -\infty} F_X(x) = 0$ and $\lim_{x \to \infty} F_X(x) = 1$.
    * **Right-Continuity**: For all $x$, $\lim_{\epsilon \to 0^+} F_X(x + \epsilon) = F_X(x)$.
    * **Range**: For all $x$, $0 \leq F_X(x) \leq 1$.

    ---

    ### Definition and Statistical Properties of the ECDF

    Given an independent and identically distributed (i.i.d.) sample $X_1, X_2, \dots, X_n$ from distribution $F$, the **Empirical CDF** $\hat{F}_n(x)$ is the step function defined by:

    $$
    \hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^n \mathbb{I}(X_i \leq x)
    $$

    where $\mathbb{I}(\cdot)$ is the indicator function.

    #### Pointwise Properties for a Fixed $x$
    * **Binomial Identity**: The count $n \hat{F}_n(x) = \sum_{i=1}^n \mathbb{I}(X_i \leq x)$ follows a Binomial distribution $\text{Binomial}(n, F(x))$.
    * **Unbiased Estimator**: $\mathbb{E}[\hat{F}_n(x)] = \frac{1}{n} \cdot n F(x) = F(x)$.
    * **Variance**: $\text{Var}(\hat{F}_n(x)) = \frac{F(x)(1 - F(x))}{n}$.
    * **Pointwise Consistency**: By the Weak Law of Large Numbers, $\hat{F}_n(x) \xrightarrow{P} F(x)$ as $n \to \infty$.
    * **Asymptotic Normality**: By the Central Limit Theorem:

    $$
    \sqrt{n} (\hat{F}_n(x) - F(x)) \xrightarrow{d} \mathcal{N}\left(0, F(x)(1 - F(x))\right)
    $$

    ---

    ### Uniform Convergence: The Glivenko-Cantelli Theorem

    Pointwise convergence guarantees accuracy at an individual coordinate $x$. The **Glivenko-Cantelli Theorem** establishes uniform convergence across the entire domain $\mathbb{R}$:

    $$
    \| \hat{F}_n - F \|_\infty = \sup_{x \in \mathbb{R}} |\hat{F}_n(x) - F(x)| \xrightarrow{\text{a.s.}} 0 \quad \text{as } n \to \infty
    $$

    This guarantees that the entire step function $\hat{F}_n$ uniformly approaches the population CDF $F$.

    ---

    ### Simultaneous Confidence Bands: The DKW Inequality

    The **Dvoretzky-Kiefer-Wolfowitz (DKW) inequality** provides non-asymptotic bounds on the probability of uniform deviation:

    $$
    P\left(\sup_{x \in \mathbb{R}} |\hat{F}_n(x) - F(x)| > \epsilon\right) \leq 2 e^{-2 n \epsilon^2}
    $$

    Setting the tail probability bound to significance level $\alpha$ (e.g. $\alpha = 0.05$):

    $$
    2 e^{-2 n \epsilon^2} = \alpha \implies \epsilon_n = \sqrt{\frac{\ln(2/\alpha)}{2n}}
    $$

    This defines a **simultaneous $100(1 - \alpha)\%$ confidence envelope**:

    $$
    L_n(x) = \max(0, \hat{F}_n(x) - \epsilon_n), \quad U_n(x) = \min(1, \hat{F}_n(x) + \epsilon_n)
    $$

    The true population function $F(x)$ is guaranteed to lie completely between $L_n(x)$ and $U_n(x)$ for all $x \in \mathbb{R}$ simultaneously with probability at least $1 - \alpha$.

    ---

    ### The Kolmogorov-Smirnov Test and MLOps Drift Detection

    The **Kolmogorov-Smirnov (KS) statistic** measures the maximum vertical distance between two distribution functions:
    * **One-Sample KS**: Compares sample ECDF against theoretical CDF: $D_n = \sup_x |\hat{F}_n(x) - F_0(x)|$.
    * **Two-Sample KS**: Compares two independent empirical samples (e.g. baseline training data vs live serving data):

    $$
    D_{n, m} = \sup_{x \in \mathbb{R}} |\hat{F}_n(x) - \hat{G}_m(x)|
    $$

    If $D_{n, m}$ exceeds critical threshold $c(\alpha) \sqrt{\frac{n + m}{n m}}$, the null hypothesis that both data streams originate from the same distribution is rejected ($p < \alpha$), triggering retraining alerts in production machine learning pipelines.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: ECDF, Theoretical CDF, and DKW Confidence Bands

    The interactive subplots below display the non-parametric inference capabilities of the ECDF:
    * **Left Panel**: Empirical CDF $\hat{F}_n(x)$ constructed from $n = 60$ samples of a standard normal distribution. Shaded blue region shows the $95\%$ simultaneous DKW confidence envelope. The red curve is the true population CDF $F(x)$. The purple vertical bar marks the maximum vertical gap (the Kolmogorov-Smirnov statistic $D_n$).
    * **Right Panel**: Two-Sample Drift Detection comparing Baseline Training Data ($N = 60$, $\mu=0, \sigma=1$) against Shifted Production Data ($M = 60$, $\mu=0.8, \sigma=1.3$). The orange vertical bar highlights the maximum discrepancy $D_{n,m}$.
    """)
    return


@app.cell
def _(go, make_subplots, np, stats):
    rng_vis = np.random.default_rng(42)
    n_sample_size = 60

    # Sample 1: Baseline N(0, 1)
    raw_sample = rng_vis.standard_normal(n_sample_size)
    sorted_sample = np.sort(raw_sample)
    ecdf_values = np.arange(1, n_sample_size + 1) / n_sample_size

    # DKW 95% Confidence Band (alpha = 0.05)
    significance_alpha = 0.05
    epsilon_dkw = np.sqrt(np.log(2.0 / significance_alpha) / (2.0 * n_sample_size))
    lower_band = np.clip(ecdf_values - epsilon_dkw, 0.0, 1.0)
    upper_band = np.clip(ecdf_values + epsilon_dkw, 0.0, 1.0)

    # True CDF
    x_axis_grid = np.linspace(-3.5, 3.5, 200)
    true_cdf_values = stats.norm.cdf(x_axis_grid)

    # KS distance location
    true_at_sample = stats.norm.cdf(sorted_sample)
    ks_deviations = np.abs(ecdf_values - true_at_sample)
    max_dev_idx = np.argmax(ks_deviations)
    ks_x_pos = sorted_sample[max_dev_idx]
    ks_y_emp = ecdf_values[max_dev_idx]
    ks_y_true = true_at_sample[max_dev_idx]
    ks_stat_val = float(ks_deviations[max_dev_idx])

    # Sample 2: Shifted Production Data N(0.8, 1.3)
    raw_drift = rng_vis.normal(loc=0.8, scale=1.3, size=n_sample_size)
    sorted_drift = np.sort(raw_drift)
    ecdf_drift_values = np.arange(1, n_sample_size + 1) / n_sample_size

    # Two-sample KS distance
    all_points = np.sort(np.unique(np.concatenate([sorted_sample, sorted_drift])))
    cdf1_eval = np.searchsorted(sorted_sample, all_points, side="right") / n_sample_size
    cdf2_eval = np.searchsorted(sorted_drift, all_points, side="right") / n_sample_size
    two_sample_diffs = np.abs(cdf1_eval - cdf2_eval)
    max_drift_idx = np.argmax(two_sample_diffs)
    drift_ks_x = all_points[max_drift_idx]
    drift_ks_y1 = cdf1_eval[max_drift_idx]
    drift_ks_y2 = cdf2_eval[max_drift_idx]
    drift_ks_val = float(two_sample_diffs[max_drift_idx])

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"ECDF with 95% DKW Band & KS Distance (Dₙ={ks_stat_val:.3f})",
            f"Two-Sample Covariate Shift Detection (D_drift={drift_ks_val:.3f})",
        ],
    )

    # Left: DKW Upper Band (hidden line for fill)
    fig.add_trace(
        go.Scatter(
            x=sorted_sample,
            y=upper_band,
            mode="lines",
            line=dict(color="rgba(37,99,235,0.0)"),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Left: DKW Band Fill
    fig.add_trace(
        go.Scatter(
            x=sorted_sample,
            y=lower_band,
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line=dict(color="rgba(37,99,235,0.0)"),
            name=f"95% DKW Band (ε={epsilon_dkw:.2f})",
        ),
        row=1,
        col=1,
    )

    # Left: True CDF
    fig.add_trace(
        go.Scatter(
            x=x_axis_grid,
            y=true_cdf_values,
            mode="lines",
            line=dict(color="#dc2626", width=2.5),
            name="True CDF F(x)",
        ),
        row=1,
        col=1,
    )

    # Left: ECDF Step Function
    fig.add_trace(
        go.Scatter(
            x=sorted_sample,
            y=ecdf_values,
            mode="lines",
            line=dict(color="#2563eb", width=2.5, shape="hv"),
            name="Empirical CDF F̂ₙ(x)",
        ),
        row=1,
        col=1,
    )

    # Left: KS Distance Bar
    fig.add_trace(
        go.Scatter(
            x=[ks_x_pos, ks_x_pos],
            y=[ks_y_true, ks_y_emp],
            mode="lines+markers",
            line=dict(color="#7c3aed", width=3.5),
            marker=dict(size=7, color="#7c3aed"),
            name=f"Max Gap Dₙ={ks_stat_val:.3f}",
        ),
        row=1,
        col=1,
    )

    # Right: Baseline ECDF
    fig.add_trace(
        go.Scatter(
            x=sorted_sample,
            y=ecdf_values,
            mode="lines",
            line=dict(color="#2563eb", width=2.5, shape="hv"),
            name="Baseline Training ECDF",
        ),
        row=1,
        col=2,
    )

    # Right: Production Shifted ECDF
    fig.add_trace(
        go.Scatter(
            x=sorted_drift,
            y=ecdf_drift_values,
            mode="lines",
            line=dict(color="#ea580c", width=2.5, shape="hv"),
            name="Production Serving ECDF (Shifted)",
        ),
        row=1,
        col=2,
    )

    # Right: Two-sample KS gap
    fig.add_trace(
        go.Scatter(
            x=[drift_ks_x, drift_ks_x],
            y=[drift_ks_y1, drift_ks_y2],
            mode="lines+markers",
            line=dict(color="#dc2626", width=3.5),
            marker=dict(size=7, color="#dc2626"),
            name=f"KS Drift Distance D={drift_ks_val:.3f}",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="x", range=[-3.5, 3.5], gridcolor="#f1f5f9"),
        yaxis=dict(title="Cumulative Probability", range=[-0.05, 1.05], gridcolor="#f1f5f9"),
        xaxis2=dict(title="x", range=[-3.5, 4.5], gridcolor="#f1f5f9"),
        yaxis2=dict(title="Cumulative Probability", range=[-0.05, 1.05], gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        all_points,
        cdf1_eval,
        cdf2_eval,
        drift_ks_val,
        drift_ks_x,
        drift_ks_y1,
        drift_ks_y2,
        ecdf_drift_values,
        ecdf_values,
        epsilon_dkw,
        fig,
        ks_deviations,
        ks_stat_val,
        ks_x_pos,
        ks_y_emp,
        ks_y_true,
        lower_band,
        max_dev_idx,
        max_drift_idx,
        n_sample_size,
        raw_drift,
        raw_sample,
        rng_vis,
        significance_alpha,
        sorted_drift,
        sorted_sample,
        true_at_sample,
        true_cdf_values,
        two_sample_diffs,
        upper_band,
        x_axis_grid,
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

    ### Example 1: Glivenko-Cantelli Convergence and DKW Envelope Verification

    In this example, we empirically verify uniform convergence and the Dvoretzky-Kiefer-Wolfowitz (DKW) inequality across five sample sizes: $n \in \{25, 100, 500, 2000, 10000\}$.

    For each sample size, we:
    1. Compute the empirical Kolmogorov-Smirnov statistic $D_n = \sup_x |\hat{F}_n(x) - F(x)|$.
    2. Compute the 95% DKW band half-width $\epsilon_n = \sqrt{\frac{\ln(2 / 0.05)}{2n}}$.
    3. Verify that $D_n \leq \epsilon_n$ and observe the asymptotic convergence rate $\mathcal{O}(1 / \sqrt{n})$.
    """)
    return


@app.cell
def _(np, stats):
    rng_gc = np.random.default_rng(101)
    sample_sizes_gc = [25, 100, 500, 2000, 10000]
    alpha_level = 0.05

    gc_verification_records = []

    for size_n in sample_sizes_gc:
        sim_sample = np.sort(rng_gc.standard_normal(size_n))
        emp_cdf = np.arange(1, size_n + 1) / size_n
        population_cdf = stats.norm.cdf(sim_sample)

        # Empirical Kolmogorov-Smirnov distance
        ks_dist = float(np.max(np.abs(emp_cdf - population_cdf)))

        # Theoretical DKW 95% threshold
        dkw_thresh = float(np.sqrt(np.log(2.0 / alpha_level) / (2.0 * size_n)))
        within_dkw_band = ks_dist <= dkw_thresh

        gc_verification_records.append(
            {
                "Sample Size (n)": size_n,
                "Empirical KS Distance Dₙ": f"{ks_dist:.4f}",
                "95% DKW Threshold ε_n": f"{dkw_thresh:.4f}",
                "Convergence Ratio (Dₙ / ε_n)": f"{ks_dist / dkw_thresh:.3f}",
                "Inside 95% Envelope": str(within_dkw_band),
            }
        )

    return (
        alpha_level,
        dkw_thresh,
        emp_cdf,
        gc_verification_records,
        ks_dist,
        population_cdf,
        sample_sizes_gc,
        sim_sample,
        size_n,
        within_dkw_band,
    )


@app.cell(hide_code=True)
def _(gc_verification_records, mo, pd):
    df_gc = pd.DataFrame(gc_verification_records)
    mo.ui.table(df_gc)
    return (df_gc,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Two-Sample Kolmogorov-Smirnov Test for Data Drift Detection

    In production machine learning systems, data drift occurs when feature distributions shift between model training and live production inference.

    Below, we simulate a monitoring pipeline where a reference training feature ($N = 400$) is compared against three production batches:
    * **Batch A (In-Distribution)**: Fresh sample from the same training distribution $\mathcal{N}(0, 1)$.
    * **Batch B (Mild Mean Shift)**: Covariate shift with shifted mean $\mathcal{N}(0.2, 1)$.
    * **Batch C (Severe Variance & Mean Shift)**: Significant drift $\mathcal{N}(0.6, 1.5)$.

    We evaluate the two-sample Kolmogorov-Smirnov test statistic $D$ and corresponding $p$-value at significance level $\alpha = 0.01$.
    """)
    return


@app.cell
def _(np, stats):
    rng_drift = np.random.default_rng(202)
    batch_size = 400

    training_reference = rng_drift.standard_normal(batch_size)

    production_batches = {
        "Batch A: In-Distribution N(0, 1)": rng_drift.standard_normal(batch_size),
        "Batch B: Mild Mean Shift N(0.2, 1)": rng_drift.normal(0.2, 1.0, batch_size),
        "Batch C: Severe Drift N(0.6, 1.5)": rng_drift.normal(0.6, 1.5, batch_size),
    }

    drift_detection_records = []

    for batch_label, prod_data in production_batches.items():
        ks_result = stats.ks_2samp(training_reference, prod_data)
        d_statistic = float(ks_result.statistic)
        p_val = float(ks_result.pvalue)

        # Drift alerted if p-value < 0.01
        drift_alert = p_val < 0.01
        status_string = "ALERT: Significant Drift" if drift_alert else "PASS: In-Distribution"

        drift_detection_records.append(
            {
                "Production Stream": batch_label,
                "KS Distance D": f"{d_statistic:.4f}",
                "p-value": f"{p_val:.2e}" if p_val < 1e-4 else f"{p_val:.4f}",
                "Alert Threshold (α=0.01)": "0.0100",
                "Monitoring Status": status_string,
            }
        )

    return (
        batch_label,
        batch_size,
        d_statistic,
        drift_alert,
        drift_detection_records,
        ks_result,
        p_val,
        prod_data,
        production_batches,
        status_string,
        training_reference,
    )


@app.cell(hide_code=True)
def _(drift_detection_records, mo, pd):
    df_drift = pd.DataFrame(drift_detection_records)
    mo.ui.table(df_drift)
    return (df_drift,)


if __name__ == "__main__":
    app.run()
