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

    return go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 13: Unbiased vs Consistent Estimators and the Bias-Variance-Consistency Trade-Off

    &larr; Previous Note: [12 Multivariate Normal Distribution](12_multivariate_normal_distribution.py) | Next Note: [14 Distribution of Minimum](14_dist_of_minimum.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    A fundamental goal in statistical inference and machine learning is estimating unknown population parameters $\theta$ (such as weights, variance, or probabilities) from a finite dataset of observations. But how do we define what makes an estimator $\hat{\theta}$ mathematically "good"?

    Two foundational properties govern estimator quality:
    * **Unbiasedness** is a **finite-sample property**: Across repeated realizations of a fixed sample size $n$, the expected value of the estimator exactly matches the true parameter ($\mathbb{E}[\hat{\theta}_n] = \theta$). It does not systematically overshoot or undershoot.
    * **Consistency** is an **asymptotic property**: As the sample size grows toward infinity ($n \to \infty$), the probability that the estimate differs from the true parameter by any margin $\epsilon > 0$ collapses to zero ($\hat{\theta}_n \xrightarrow{P} \theta$).

    Why this distinction is vital in machine learning and data science:
    1. **The Modern ML Reality: Biased but Consistent**: In contemporary machine learning, almost all effective estimators are deliberately biased to reduce variance. Examples include Ridge regression ($\hat{\boldsymbol{\beta}}_{\text{ridge}}$), weight decay, Lasso, and Maximum Likelihood variance ($\hat{\sigma}^2_{\text{MLE}}$). While biased for small $n$, they are consistent as $n \to \infty$ and dramatically outperform unbiased estimators in Mean Squared Error (MSE).
    2. **The Danger of Inconsistent Unbiased Estimators**: An estimator can be perfectly unbiased for every sample size $n$, yet completely useless because its variance never shrinks (e.g., using only the first sample observation $X_1$ to estimate the population mean). No matter how much data is collected, it never gets closer to the truth.
    3. **Mean Squared Error Decomposition**: $\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$. A sufficient condition for consistency is that both the squared bias and the variance shrink to zero as $n \to \infty$.
    4. **Bessel's Correction**: The Maximum Likelihood estimator for variance divides by $n$, introducing a small bias of $-\sigma^2/n$ that disappears asymptotically. In contrast, dividing by $n-1$ achieves exact finite-sample unbiasedness.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Formal Definition: Unbiased Estimator

    Let $X_1, \dots, X_n$ be an i.i.d. sample from a distribution parameterized by $\theta$. An estimator $\hat{\theta}_n = g(X_1, \dots, X_n)$ is **unbiased** if its mathematical expectation equals $\theta$:

    $$
    \mathbb{E}[\hat{\theta}_n] = \theta \quad \text{for all } n \geq 1
    $$

    The **bias** of an estimator is defined as:

    $$
    \text{Bias}(\hat{\theta}_n) = \mathbb{E}[\hat{\theta}_n] - \theta
    $$

    An estimator is unbiased if and only if $\text{Bias}(\hat{\theta}_n) = 0$.

    ---

    ### Formal Definition: Consistent Estimator

    An estimator $\hat{\theta}_n$ is **consistent** for $\theta$ if it converges in probability to $\theta$ as sample size $n \to \infty$:

    $$
    \lim_{n \to \infty} P(|\hat{\theta}_n - \theta| < \epsilon) = 1 \quad \text{for every } \epsilon > 0
    $$

    This is written compactly as $\hat{\theta}_n \xrightarrow{P} \theta$.

    #### Mean Squared Error and Consistency Criterion
    The Mean Squared Error (MSE) of an estimator decomposes into:

    $$
    \text{MSE}(\hat{\theta}_n) = \mathbb{E}[(\hat{\theta}_n - \theta)^2] = \text{Bias}^2(\hat{\theta}_n) + \text{Var}(\hat{\theta}_n)
    $$

    By Chebyshev's inequality (Note 10):

    $$
    P(|\hat{\theta}_n - \theta| \geq \epsilon) \leq \frac{\mathbb{E}[(\hat{\theta}_n - \theta)^2]}{\epsilon^2} = \frac{\text{MSE}(\hat{\theta}_n)}{\epsilon^2}
    $$

    Therefore, if $\lim_{n \to \infty} \text{MSE}(\hat{\theta}_n) = 0$, the estimator is guaranteed to be consistent. This holds whenever both the bias and variance vanish asymptotically:

    $$
    \lim_{n \to \infty} \text{Bias}(\hat{\theta}_n) = 0 \quad \text{and} \quad \lim_{n \to \infty} \text{Var}(\hat{\theta}_n) = 0 \implies \hat{\theta}_n \xrightarrow{P} \theta
    $$

    ---

    ### The Four Canonical Estimator Archetypes

    To build intuition, consider estimating the population mean $\mu$ from an i.i.d. sample $X_1, \dots, X_n \sim \mathcal{N}(\mu, \sigma^2)$:

    #### 1. Unbiased and Consistent: The Sample Mean
    $$
    T_1 = \bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i
    $$
    * Expectation: $\mathbb{E}[T_1] = \mu \implies \text{Bias} = 0$ (Unbiased).
    * Variance: $\text{Var}(T_1) = \frac{\sigma^2}{n} \to 0$ as $n \to \infty$ (Consistent).

    #### 2. Biased but Consistent: Shrinkage Estimator
    $$
    T_2 = \frac{n}{n + 1} \bar{X}_n
    $$
    * Expectation: $\mathbb{E}[T_2] = \frac{n}{n + 1} \mu \neq \mu \implies \text{Bias} = -\frac{\mu}{n + 1}$ (Biased for every finite $n$).
    * Asymptotics: $\lim_{n \to \infty} \text{Bias}(T_2) = 0$ and $\lim_{n \to \infty} \text{Var}(T_2) = \lim_{n \to \infty} \left(\frac{n}{n+1}\right)^2 \frac{\sigma^2}{n} = 0$ (Consistent).

    #### 3. Unbiased but Inconsistent: Single-Sample Estimator
    $$
    T_3 = X_1
    $$
    * Expectation: $\mathbb{E}[T_3] = \mathbb{E}[X_1] = \mu \implies \text{Bias} = 0$ (Unbiased).
    * Variance: $\text{Var}(T_3) = \sigma^2 \neq 0$ (Does not shrink with $n$). Because variance remains constant, $T_3$ never converges to $\mu$ (Inconsistent).

    #### 4. Biased and Inconsistent: Shifted Single-Sample Estimator
    $$
    T_4 = X_1 + 1.0
    $$
    * Expectation: $\mathbb{E}[T_4] = \mu + 1.0 \implies \text{Bias} = 1.0 \neq 0$ (Biased).
    * Variance: $\text{Var}(T_4) = \sigma^2 \neq 0$ (Inconsistent).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Finite-Sample Sampling Distributions vs Asymptotic Trajectories

    The interactive subplots below display the dual perspectives of estimation theory:
    * **Left Panel**: Sampling distributions at a fixed small sample size ($n = 5$) across $1,000$ Monte Carlo trials. Notice that $T_1$ (Sample Mean) and $T_3$ (Single Sample $X_1$) are both centered exactly at the true mean $\mu = 5.0$ (Unbiased), but $T_3$ has wide spread. In contrast, $T_2$ (Shrinkage) is shifted to the left at $\frac{5}{6}\mu \approx 4.17$ (Biased).
    * **Right Panel**: Real-time convergence trajectories as sample size $n$ increases from $1$ to $300$. Notice how both $T_1(n)$ and $T_2(n)$ rapidly converge to the true parameter line $\mu = 5.0$ (Consistent), while $T_3$ continues to scatter with constant dispersion (Inconsistent).
    """)
    return


@app.cell
def _(go, make_subplots, np):
    rng_sim = np.random.default_rng(42)
    true_param_mu = 5.0
    true_param_sigma = 1.2
    n_monte_carlo = 1000

    # 1. Left Panel: Fixed small sample size n = 5
    n_fixed = 5
    batch_samples_n5 = rng_sim.normal(true_param_mu, true_param_sigma, size=(n_monte_carlo, n_fixed))
    t1_fixed = np.mean(batch_samples_n5, axis=1)
    t2_fixed = (n_fixed / (n_fixed + 1.0)) * t1_fixed
    t3_fixed = batch_samples_n5[:, 0]

    # 2. Right Panel: Dynamic sample size expansion n = 1 to 300
    n_grid_sizes = np.arange(1, 301)
    stream_observations = rng_sim.normal(true_param_mu, true_param_sigma, size=300)
    t1_trajectory = np.cumsum(stream_observations) / n_grid_sizes
    t2_trajectory = (n_grid_sizes / (n_grid_sizes + 1.0)) * t1_trajectory
    t3_independent_draws = rng_sim.normal(true_param_mu, true_param_sigma, size=300)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Sampling Distributions at Fixed n={n_fixed} (Unbiasedness Check)",
            "Convergence Trajectories as n → 300 (Consistency Check)",
        ],
    )

    # Left: T1 (Sample Mean)
    fig.add_trace(
        go.Histogram(
            x=t1_fixed,
            opacity=0.65,
            marker_color="#2563eb",
            name="T₁: Sample Mean (Unbiased)",
            nbinsx=35,
        ),
        row=1,
        col=1,
    )

    # Left: T2 (Shrinkage)
    fig.add_trace(
        go.Histogram(
            x=t2_fixed,
            opacity=0.65,
            marker_color="#ea580c",
            name="T₂: Shrinkage Mean (Biased)",
            nbinsx=35,
        ),
        row=1,
        col=1,
    )

    # Left: T3 (Single observation X1)
    fig.add_trace(
        go.Histogram(
            x=t3_fixed,
            opacity=0.40,
            marker_color="#9333ea",
            name="T₃: Single Observation X₁ (Unbiased)",
            nbinsx=35,
        ),
        row=1,
        col=1,
    )

    # Left: True mean line
    fig.add_trace(
        go.Scatter(
            x=[true_param_mu, true_param_mu],
            y=[0, 160],
            mode="lines",
            line=dict(color="#dc2626", width=2.5, dash="dash"),
            name="True Parameter μ = 5.0",
        ),
        row=1,
        col=1,
    )

    # Right: T1 Trajectory
    fig.add_trace(
        go.Scatter(
            x=n_grid_sizes,
            y=t1_trajectory,
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name="T₁(n) Path (Consistent)",
        ),
        row=1,
        col=2,
    )

    # Right: T2 Trajectory
    fig.add_trace(
        go.Scatter(
            x=n_grid_sizes,
            y=t2_trajectory,
            mode="lines",
            line=dict(color="#ea580c", width=2.5),
            name="T₂(n) Path (Consistent)",
        ),
        row=1,
        col=2,
    )

    # Right: T3 Trajectory
    fig.add_trace(
        go.Scatter(
            x=n_grid_sizes,
            y=t3_independent_draws,
            mode="markers",
            marker=dict(size=4, color="#9333ea", opacity=0.45),
            name="T₃(n) Draws (Inconsistent)",
        ),
        row=1,
        col=2,
    )

    # Right: True mean line
    fig.add_trace(
        go.Scatter(
            x=[1, 300],
            y=[true_param_mu, true_param_mu],
            mode="lines",
            line=dict(color="#dc2626", width=2.5, dash="dash"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        barmode="overlay",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="Estimate Value", range=[1.0, 9.0], gridcolor="#f1f5f9"),
        yaxis=dict(title="Frequency", gridcolor="#f1f5f9"),
        xaxis2=dict(title="Sample Size (n)", range=[1, 300], gridcolor="#f1f5f9"),
        yaxis2=dict(title="Estimate Value", range=[1.0, 9.0], gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        batch_samples_n5,
        fig,
        n_fixed,
        n_grid_sizes,
        n_monte_carlo,
        rng_sim,
        stream_observations,
        t1_fixed,
        t1_trajectory,
        t2_fixed,
        t2_trajectory,
        t3_fixed,
        t3_independent_draws,
        true_param_mu,
        true_param_sigma,
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

    ### Example 1: Monte Carlo Verification of Bias, Variance, and MSE Across the 4 Archetypes

    In this example, we evaluate all four archetype estimators across sample sizes $n \in \{5, 20, 100, 500, 2000\}$ using $M = 2,000$ Monte Carlo simulations per sample size.

    For each estimator, we calculate:
    1. **Empirical Bias**: $\hat{\mathbb{E}}[\hat{\theta}] - \theta$
    2. **Empirical Variance**: $\widehat{\text{Var}}(\hat{\theta})$
    3. **Mean Squared Error**: $\widehat{\text{MSE}}(\hat{\theta}) = \text{Bias}^2 + \text{Var}$
    4. **Consistency Verification**: Whether MSE strictly converges to $0$ as $n \to \infty$.
    """)
    return


@app.cell
def _(np):
    rng_ex1 = np.random.default_rng(101)
    true_mu_val = 5.0
    true_sigma_val = 1.5
    num_trials = 2000
    sample_sizes_grid = [5, 20, 100, 500, 2000]

    archetype_results_list = []

    for sample_n in sample_sizes_grid:
        batch_draws = rng_ex1.normal(true_mu_val, true_sigma_val, size=(num_trials, sample_n))

        t1_vals = np.mean(batch_draws, axis=1)
        t2_vals = (sample_n / (sample_n + 1.0)) * t1_vals
        t3_vals = batch_draws[:, 0]
        t4_vals = batch_draws[:, 0] + 1.0

        estimator_configs = [
            ("T₁: Sample Mean X̄", t1_vals, "Unbiased", "Consistent"),
            ("T₂: Shrinkage Mean [n/(n+1)]X̄", t2_vals, "Biased", "Consistent"),
            ("T₃: First Observation X₁", t3_vals, "Unbiased", "Inconsistent"),
            ("T₄: Shifted Observation X₁ + 1", t4_vals, "Biased", "Inconsistent"),
        ]

        for label, values, unb_class, cons_class in estimator_configs:
            emp_bias = float(np.mean(values) - true_mu_val)
            emp_var = float(np.var(values))
            emp_mse = float(np.mean((values - true_mu_val) ** 2))

            archetype_results_list.append(
                {
                    "Estimator": label,
                    "Sample Size (n)": sample_n,
                    "Empirical Bias": f"{emp_bias:+.4f}",
                    "Empirical Variance": f"{emp_var:.4f}",
                    "Empirical MSE": f"{emp_mse:.4f}",
                    "Theoretical Class": f"{unb_class}, {cons_class}",
                    "MSE Shrinks to 0?": "Yes" if cons_class == "Consistent" else "No",
                }
            )

    return (
        archetype_results_list,
        batch_draws,
        cons_class,
        emp_bias,
        emp_mse,
        emp_var,
        estimator_configs,
        label,
        num_trials,
        rng_ex1,
        sample_n,
        sample_sizes_grid,
        t1_vals,
        t2_vals,
        t3_vals,
        t4_vals,
        true_mu_val,
        true_sigma_val,
        unb_class,
        values,
    )


@app.cell(hide_code=True)
def _(archetype_results_list, mo, pd):
    df_archetypes = pd.DataFrame(archetype_results_list)
    mo.ui.table(df_archetypes)
    return (df_archetypes,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Bessel's Correction and Covariance Estimation in Machine Learning

    When estimating population variance $\sigma^2$ from sample $X_1, \dots, X_n$, two estimators compete:
    1. **Maximum Likelihood Estimator ($\hat{\sigma}^2_{\text{MLE}}$)**: Divides by $n$. Biased for finite $n$ with expected value $\frac{n-1}{n}\sigma^2$, but asymptotically consistent.
    2. **Unbiased Sample Variance ($S^2$)**: Divides by $n - 1$ (Bessel's correction). Exactly unbiased for all $n \geq 2$.

    Below, we simulate true variance $\sigma^2 = 4.0$ across small and large sample sizes $n \in \{2, 5, 10, 50, 200\}$ over $M = 5,000$ trials, measuring the shrinkage of MLE bias toward zero.
    """)
    return


@app.cell
def _(np):
    rng_bessel = np.random.default_rng(202)
    true_variance = 4.0
    true_std = np.sqrt(true_variance)
    bessel_sim_trials = 5000
    bessel_sample_sizes = [2, 5, 10, 50, 200]

    bessel_comparison_records = []

    for n_count in bessel_sample_sizes:
        trials_matrix = rng_bessel.normal(0.0, true_std, size=(bessel_sim_trials, n_count))
        trials_means = np.mean(trials_matrix, axis=1, keepdims=True)
        residuals_sq = np.sum((trials_matrix - trials_means) ** 2, axis=1)

        # MLE variance (divides by n)
        var_mle_trials = residuals_sq / n_count
        # Unbiased variance (divides by n - 1)
        var_unbiased_trials = residuals_sq / (n_count - 1)

        mean_mle = float(np.mean(var_mle_trials))
        bias_mle = mean_mle - true_variance
        mean_unbiased = float(np.mean(var_unbiased_trials))
        bias_unbiased = mean_unbiased - true_variance

        pct_bias_mle = (bias_mle / true_variance) * 100.0

        bessel_comparison_records.append(
            {
                "Sample Size (n)": n_count,
                "True σ²": f"{true_variance:.2f}",
                "MLE E[σ̂²] (Divides by n)": f"{mean_mle:.4f}",
                "MLE Bias": f"{bias_mle:+.4f} ({pct_bias_mle:+.1f}%)",
                "Unbiased E[S²] (Divides by n-1)": f"{mean_unbiased:.4f}",
                "Unbiased Bias": f"{bias_unbiased:+.4f}",
                "Consistency Observed": "Bias → 0 as n increases",
            }
        )

    return (
        bessel_comparison_records,
        bessel_sample_sizes,
        bessel_sim_trials,
        bias_mle,
        bias_unbiased,
        mean_mle,
        mean_unbiased,
        n_count,
        pct_bias_mle,
        residuals_sq,
        rng_bessel,
        trials_matrix,
        trials_means,
        true_std,
        true_variance,
        var_mle_trials,
        var_unbiased_trials,
    )


@app.cell(hide_code=True)
def _(bessel_comparison_records, mo, pd):
    df_bessel = pd.DataFrame(bessel_comparison_records)
    mo.ui.table(df_bessel)
    return (df_bessel,)


if __name__ == "__main__":
    app.run()
