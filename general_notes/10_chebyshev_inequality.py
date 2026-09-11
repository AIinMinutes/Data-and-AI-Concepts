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
    # Note 10: Chebyshev Inequality, Tail Bounds, and Concentration of Measure

    &larr; Previous Note: [09 Condition Number](09_condition_number.py) | Next Note: [11 Empirical CDF](11_ecdf.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Most statistical methods rely on distributional assumptions, such as Gaussianity. In real-world data science, machine learning, and streaming pipelines, however, true underlying probability distributions are frequently non-Gaussian, asymmetric, multi-modal, or heavy-tailed.

    **Chebyshev's Inequality** provides a universal mathematical guarantee that holds across every probability distribution with finite variance.

    Key applications across modern AI and statistics:
    1. **Distribution-Free Statistical Guarantees**: Chebyshev's inequality guarantees that no matter how strange, skewed, or uncharacterized a distribution is, the probability of falling beyond $k$ standard deviations from the mean cannot exceed $1/k^2$.
    2. **Foundation of Learning Theory (PAC Learning)**: Generalization bounds in statistical learning theory bound the difference between empirical risk and expected risk using concentration inequalities. Chebyshev is the gateway inequality connecting first-moment bounds (Markov) to exponential concentration (Chernoff, Hoeffding, McDiarmid).
    3. **The Weak Law of Large Numbers (WLLN)**: Chebyshev's inequality provides the most direct proof that sample averages converge in probability to population means ($\bar{X}_n \xrightarrow{P} \mu$), establishing that empirical estimations stabilize as sample sizes grow.
    4. **Robust Outlier Detection and Quality Control**: Unlike Gaussian rules (e.g. the 3-sigma rule asserting $99.7\%$ coverage), which fail under heavy tails, Chebyshev gives an unconditional upper bound on outlier probabilities ($P(|X - \mu| \geq 3\sigma) \leq 11.1\%$) valid for any production metric.
    5. **Randomized and Streaming Algorithms**: In big-data sketching algorithms (such as the Count-Min Sketch or the Median-of-Means estimator), Chebyshev bounds the failure probability of sub-linear memory approximations.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Markov's Inequality: The First-Moment Foundation

    Markov's inequality applies to any non-negative random variable $Y \geq 0$ with finite expectation $\mathbb{E}[Y] < \infty$. For any constant $a > 0$:

    $$
    P(Y \geq a) \leq \frac{\mathbb{E}[Y]}{a}
    $$

    #### Derivation
    Let $\mathbb{I}_{Y \geq a}$ denote the indicator random variable that equals $1$ if $Y \geq a$ and $0$ otherwise. Since $Y \geq 0$:

    $$
    a \cdot \mathbb{I}_{Y \geq a} \leq Y
    $$

    Taking mathematical expectations on both sides:

    $$
    \mathbb{E}[a \cdot \mathbb{I}_{Y \geq a}] \leq \mathbb{E}[Y] \implies a P(Y \geq a) \leq \mathbb{E}[Y] \implies P(Y \geq a) \leq \frac{\mathbb{E}[Y]}{a}
    $$

    ---

    ### Chebyshev's Inequality: The Second-Moment Extension

    Let $X$ be any random variable with finite mean $\mu = \mathbb{E}[X]$ and finite variance $\sigma^2 = \text{Var}(X) < \infty$.
    We apply Markov's inequality to the non-negative random variable $Y = (X - \mu)^2$ with threshold $a = (k\sigma)^2$ for $k > 0$:

    $$
    P(|X - \mu| \geq k\sigma) = P\left((X - \mu)^2 \geq k^2 \sigma^2\right) \leq \frac{\mathbb{E}[(X - \mu)^2]}{k^2 \sigma^2} = \frac{\sigma^2}{k^2 \sigma^2} = \frac{1}{k^2}
    $$

    #### Version 1 (Tail Probability Upper Bound)
    The probability that a random variable deviates from its mean by $k$ or more standard deviations is bounded from above by $1/k^2$:

    $$
    P(|X - \mu| \geq k\sigma) \leq \frac{1}{k^2}
    $$

    #### Version 2 (Concentration Lower Bound)
    Taking the complementary probability, the fraction of values lying strictly within $k$ standard deviations is bounded from below by $1 - 1/k^2$:

    $$
    P(|X - \mu| < k\sigma) \geq 1 - \frac{1}{k^2}
    $$

    #### Practical Numerical Thresholds
    * $k = 2$: $P(|X - \mu| < 2\sigma) \geq 1 - \frac{1}{4} = 75.00\%$
    * $k = 3$: $P(|X - \mu| < 3\sigma) \geq 1 - \frac{1}{9} \approx 88.89\%$
    * $k = 4$: $P(|X - \mu| < 4\sigma) \geq 1 - \frac{1}{16} = 93.75\%$
    * $k = 5$: $P(|X - \mu| < 5\sigma) \geq 1 - \frac{1}{25} = 96.00\%$

    ---

    ### Proof of the Weak Law of Large Numbers (WLLN)

    Let $X_1, X_2, \dots, X_n$ be independent, identically distributed (i.i.d.) random variables with mean $\mu$ and variance $\sigma^2$.
    The sample mean is $\bar{X}_n = \frac{1}{n} \sum_{i=1}^n X_i$.
    The expectation and variance of the sample mean are:

    $$
    \mathbb{E}[\bar{X}_n] = \mu, \quad \text{Var}(\bar{X}_n) = \frac{\sigma^2}{n}
    $$

    Applying Chebyshev's inequality to $\bar{X}_n$ with arbitrary error tolerance $\epsilon > 0$:

    $$
    P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\text{Var}(\bar{X}_n)}{\epsilon^2} = \frac{\sigma^2}{n \epsilon^2}
    $$

    Taking the limit as the sample size $n \to \infty$:

    $$
    \lim_{n \to \infty} P(|\bar{X}_n - \mu| \geq \epsilon) \leq \lim_{n \to \infty} \frac{\sigma^2}{n \epsilon^2} = 0
    $$

    This establishes **convergence in probability** ($\bar{X}_n \xrightarrow{P} \mu$), proving the fundamental theorem of empirical science.

    ---

    ### Tightness and Comparison Across Probability Distributions

    Chebyshev's bound is sharp: it cannot be improved without making additional assumptions beyond finite variance. A discrete random variable taking values $\mu - k\sigma$ and $\mu + k\sigma$ each with probability $\frac{1}{2k^2}$, and $\mu$ with probability $1 - \frac{1}{k^2}$, satisfies the bound with exact equality.

    When specific distributional families are known, exact tail probabilities are substantially smaller:
    * **Normal Distribution**: $P(|X - \mu| \geq 2\sigma) \approx 4.55\%$ (Chebyshev upper bound: $25\%$).
    * **Uniform Distribution**: $P(|X - \mu| \geq 2\sigma) = 0\%$ (since maximum deviation is $\sqrt{3}\sigma \approx 1.732\sigma$).
    * **Exponential Distribution**: $P(|X - \mu| \geq 2\sigma) \approx 4.98\%$.
    * **Student-t ($df = 4$)**: $P(|X - \mu| \geq 2\sigma) \approx 7.92\%$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Universal Bounds Across Diverse Distributions

    The interactive subplots below display Chebyshev's distribution-free guarantee:
    * **Left Panel**: Tail probability $P(|X - \mu| \geq k\sigma)$ as a function of the deviation factor $k$. The red curve represents the universal Chebyshev envelope $1/k^2$. Regardless of distribution (Gaussian, Exponential, Uniform, Student-t), all tail probabilities remain strictly beneath this theoretical ceiling.
    * **Right Panel**: Probability density of an asymmetric Exponential distribution ($\lambda = 1.0$), with shaded regions marking the guaranteed inner mass ($P \geq 75\%$) and bounded tail mass ($P \leq 25\%$) at $k = 2$.
    """)
    return


@app.cell
def _(go, make_subplots, np, stats):
    # Left Panel: Tail Probability vs k comparison
    k_values = np.linspace(1.1, 4.5, 120)
    chebyshev_bound_curve = 1.0 / (k_values**2)

    # Theoretical tails for standardized distributions (mean=0, std=1)
    # 1. Normal N(0, 1)
    normal_tail_curve = 2.0 * (1.0 - stats.norm.cdf(k_values))

    # 2. Standardized Exponential(1): mean=1, std=1.
    # Deviation |X - 1| >= k => X >= 1 + k (since X >= 0, for k > 1, 1 - k < 0 is impossible)
    exp_tail_curve = np.exp(-(1.0 + k_values))

    # 3. Student-t with df=4 (heavy tails, std = sqrt(4/(4-2)) = sqrt(2))
    t_tail_curve = 2.0 * (1.0 - stats.t.cdf(k_values * np.sqrt(2.0), df=4))

    # 4. Standardized Uniform[-sqrt(3), sqrt(3)]
    uniform_tail_curve = np.maximum(0.0, 1.0 - k_values / np.sqrt(3.0))

    # Right Panel: Exponential distribution with k = 2 coverage
    x_grid = np.linspace(0.0, 6.0, 300)
    exp_pdf = np.exp(-x_grid)
    mu_exp = 1.0
    sigma_exp = 1.0
    k_demo = 2.0
    lower_cutoff = max(0.0, mu_exp - k_demo * sigma_exp)
    upper_cutoff = mu_exp + k_demo * sigma_exp

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Tail Probability vs Deviation Factor (k)",
            "Coverage on Skewed Exponential Distribution (k = 2)",
        ],
    )

    # Left: Chebyshev Upper Bound
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=chebyshev_bound_curve,
            mode="lines",
            line=dict(color="#dc2626", width=3.5),
            name="Chebyshev Envelope (1/k²)",
        ),
        row=1,
        col=1,
    )

    # Left: Normal Distribution
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=normal_tail_curve,
            mode="lines",
            line=dict(color="#2563eb", width=2.0),
            name="Normal N(0, 1)",
        ),
        row=1,
        col=1,
    )

    # Left: Student-t Distribution
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=t_tail_curve,
            mode="lines",
            line=dict(color="#ea580c", width=2.0),
            name="Student-t (df=4, heavy-tailed)",
        ),
        row=1,
        col=1,
    )

    # Left: Exponential Distribution
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=exp_tail_curve,
            mode="lines",
            line=dict(color="#16a34a", width=2.0),
            name="Exponential(1) (skewed)",
        ),
        row=1,
        col=1,
    )

    # Left: Uniform Distribution
    fig.add_trace(
        go.Scatter(
            x=k_values,
            y=uniform_tail_curve,
            mode="lines",
            line=dict(color="#9333ea", width=2.0, dash="dot"),
            name="Uniform[-√3, √3]",
        ),
        row=1,
        col=1,
    )

    # Right: Full PDF
    fig.add_trace(
        go.Scatter(
            x=x_grid,
            y=exp_pdf,
            mode="lines",
            line=dict(color="#2563eb", width=2.5),
            name="Exponential PDF f(x)",
            showlegend=True,
        ),
        row=1,
        col=2,
    )

    # Right: Shaded central region [0, 3] (within ±2σ)
    mask_inside = (x_grid >= lower_cutoff) & (x_grid <= upper_cutoff)
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([[lower_cutoff], x_grid[mask_inside], [upper_cutoff]]),
            y=np.concatenate([[0.0], exp_pdf[mask_inside], [0.0]]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.20)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Inside ±2σ (Actual: 95.0%, Bound: ≥75%)",
        ),
        row=1,
        col=2,
    )

    # Right: Shaded tail region [3, 6] (outside ±2σ)
    mask_tail = x_grid >= upper_cutoff
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([[upper_cutoff], x_grid[mask_tail], [6.0]]),
            y=np.concatenate([[0.0], exp_pdf[mask_tail], [0.0]]),
            fill="toself",
            fillcolor="rgba(220, 38, 38, 0.30)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Tail > μ+2σ (Actual: 5.0%, Bound: ≤25%)",
        ),
        row=1,
        col=2,
    )

    # Right: Vertical line at upper cutoff
    fig.add_trace(
        go.Scatter(
            x=[upper_cutoff, upper_cutoff],
            y=[0.0, np.exp(-upper_cutoff) + 0.15],
            mode="lines+text",
            line=dict(color="#dc2626", width=2.0, dash="dash"),
            text=["", "μ + 2σ = 3.0"],
            textposition="top center",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="Deviation Factor k", gridcolor="#f1f5f9"),
        yaxis=dict(title="Tail Probability P(|X - μ| ≥ kσ)", gridcolor="#f1f5f9"),
        xaxis2=dict(title="x", gridcolor="#f1f5f9"),
        yaxis2=dict(title="Probability Density", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        chebyshev_bound_curve,
        exp_pdf,
        exp_tail_curve,
        fig,
        k_demo,
        k_values,
        lower_cutoff,
        mask_inside,
        mask_tail,
        mu_exp,
        normal_tail_curve,
        sigma_exp,
        t_tail_curve,
        uniform_tail_curve,
        upper_cutoff,
        x_grid,
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

    ### Example 1: Empirical Verification Across Five Diverse Distributions

    In this example, we generate $N = 200,000$ synthetic samples from five fundamentally distinct distributions:
    1. **Gaussian**: Standard bell-shaped symmetric distribution.
    2. **Exponential**: Heavily right-skewed non-negative distribution.
    3. **Uniform**: Bounded distribution with finite support.
    4. **Bimodal Gaussian Mixture**: Mixture of two separated Gaussian components.
    5. **Student-t ($df = 5$)**: Heavy-tailed polynomial decay distribution.

    For each distribution, we compute the empirical tail probability $P(|X - \mu| \geq k\sigma)$ at $k \in \{2, 3, 4\}$ and verify that it strictly satisfies Chebyshev's upper bound ($1/k^2$).
    """)
    return


@app.cell
def _(np):
    rng_sim = np.random.default_rng(42)
    n_samples_sim = 200000

    test_distributions = {
        "Gaussian N(0, 1)": rng_sim.standard_normal(n_samples_sim),
        "Exponential(λ=1)": rng_sim.exponential(scale=1.0, size=n_samples_sim),
        "Uniform[-2, 2]": rng_sim.uniform(-2.0, 2.0, size=n_samples_sim),
        "Bimodal Mixture": np.concatenate(
            [
                rng_sim.normal(-2.5, 0.8, n_samples_sim // 2),
                rng_sim.normal(2.5, 0.8, n_samples_sim // 2),
            ]
        ),
        "Student-t (df=5)": rng_sim.standard_t(df=5, size=n_samples_sim),
    }

    empirical_bound_records = []

    for dist_name, sample_data in test_distributions.items():
        sample_mean = float(np.mean(sample_data))
        sample_std = float(np.std(sample_data))

        for k_factor in [2, 3, 4]:
            deviations = np.abs(sample_data - sample_mean)
            empirical_tail_prob = float(np.mean(deviations >= k_factor * sample_std))
            theoretical_upper_bound = 1.0 / (k_factor**2)
            bound_holds = empirical_tail_prob <= theoretical_upper_bound

            empirical_bound_records.append(
                {
                    "Distribution": dist_name,
                    "k (Std Devs)": k_factor,
                    "Empirical Tail P": f"{empirical_tail_prob:.4f}",
                    "Chebyshev Bound (1/k²)": f"{theoretical_upper_bound:.4f}",
                    "Empirical Inner P": f"{1.0 - empirical_tail_prob:.4f}",
                    "Chebyshev Lower (1 - 1/k²)": f"{1.0 - theoretical_upper_bound:.4f}",
                    "Bound Satisfied": str(bound_holds),
                }
            )

    return (
        bound_holds,
        deviations,
        dist_name,
        empirical_bound_records,
        empirical_tail_prob,
        k_factor,
        n_samples_sim,
        rng_sim,
        sample_data,
        sample_mean,
        sample_std,
        test_distributions,
        theoretical_upper_bound,
    )


@app.cell(hide_code=True)
def _(empirical_bound_records, mo, pd):
    df_chebyshev = pd.DataFrame(empirical_bound_records)
    mo.ui.table(df_chebyshev)
    return (df_chebyshev,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Weak Law of Large Numbers Convergence and Sample Complexity

    By Chebyshev's inequality, the probability that the sample mean $\bar{X}_n$ deviates from the true expectation $\mu$ by more than tolerance $\epsilon$ shrinks as $\mathcal{O}(1/n)$:

    $$
    P(|\bar{X}_n - \mu| \geq \epsilon) \leq \frac{\sigma^2}{n \epsilon^2}
    $$

    Below, we draw from a population with true mean $\mu = 5.0$ and standard deviation $\sigma = 2.0$. We run $M = 1,000$ independent Monte Carlo experiments across increasing sample sizes $n \in \{50, 100, 250, 500, 1000, 2500\}$ with error tolerance $\epsilon = 0.08$.

    We compare the empirical fraction of trials violating tolerance against the theoretical Chebyshev ceiling.
    """)
    return


@app.cell
def _(np):
    rng_wlln = np.random.default_rng(101)
    num_monte_carlo = 1000
    epsilon_tol = 0.08
    pop_mean = 5.0
    pop_std = 2.0
    sample_sizes_list = [50, 100, 250, 500, 1000, 2500]

    wlln_summary_records = []

    for n_size in sample_sizes_list:
        # Draw matrix of shape (num_monte_carlo, n_size)
        mc_samples = rng_wlln.normal(loc=pop_mean, scale=pop_std, size=(num_monte_carlo, n_size))
        sample_means = np.mean(mc_samples, axis=1)

        # Proportion of trials where |X_bar - mu| >= epsilon
        empirical_violations = float(np.mean(np.abs(sample_means - pop_mean) >= epsilon_tol))
        theoretical_bound_wlln = min(1.0, float((pop_std**2) / (n_size * (epsilon_tol**2))))

        wlln_summary_records.append(
            {
                "Sample Size (n)": n_size,
                "Error Tolerance (ε)": epsilon_tol,
                "Empirical P(|X̄ - μ| ≥ ε)": f"{empirical_violations:.4f}",
                "Chebyshev Upper Bound (σ²/nε²)": f"{theoretical_bound_wlln:.4f}",
                "Guaranteed Convergence": str(empirical_violations <= theoretical_bound_wlln),
            }
        )

    return (
        empirical_violations,
        epsilon_tol,
        mc_samples,
        n_size,
        num_monte_carlo,
        pop_mean,
        pop_std,
        rng_wlln,
        sample_means,
        sample_sizes_list,
        theoretical_bound_wlln,
        wlln_summary_records,
    )


@app.cell(hide_code=True)
def _(mo, pd, wlln_summary_records):
    df_wlln = pd.DataFrame(wlln_summary_records)
    mo.ui.table(df_wlln)
    return (df_wlln,)


if __name__ == "__main__":
    app.run()
