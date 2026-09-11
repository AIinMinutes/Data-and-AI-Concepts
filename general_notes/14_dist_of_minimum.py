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
    # Note 14: Distribution of the Minimum, Order Statistics, and Extreme Values

    &larr; Previous Note: [13 Unbiased vs Consistent](13_unbiased_vs_consistent.py) | Next Note: [15 Mutual Information](15_mutual_information.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Standard introductory statistics focuses on sample averages and sums (governed by the Central Limit Theorem). In engineering, finance, machine learning, and computer systems, however, critical outcomes are frequently dictated not by averages, but by the **minimum** or **maximum** across a sample:

    1. **Reliability and Series System Architectures**: In distributed systems, microservice DAG pipelines, and hardware component arrays, a series system functions only as long as its weakest link survives. The system failure time is the minimum of component failure times: $T_{\text{system}} = \min(T_1, \dots, T_n)$.
    2. **Extreme Value Theory (EVT) and Tail Risk**: In quantitative risk management, financial engineering, and flood prediction, the focus is on extreme tail events (such as maximum portfolio drawdown or minimum liquidity reserves). The Fisher-Tippett-Gnedenko theorem shows that sample extremes converge to generalized extreme value (GEV) distributions.
    3. **First-Hitting Times in Optimization**: In stochastic gradient descent, simulated annealing, and hyperparameter sweeps across $n$ parallel worker threads, the time until the first worker reaches a target loss is governed by the distribution of the minimum.
    4. **Nearest Neighbor Distances in Vector Databases**: In $k$-NN classifiers, recommendation systems, and vector databases (HNSW, ScaNN, FAISS), the distance from a query vector $\mathbf{q}$ to the nearest database item is $R_{(1)} = \min_{i} \|\mathbf{q} - \mathbf{x}_i\|_2$. Understanding the distribution of the minimum explains the geometric curse of dimensionality in high-dimensional vector search.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Definition of Order Statistics

    Let $X_1, X_2, \dots, X_n$ be independent and identically distributed (i.i.d.) continuous random variables with common cumulative distribution function $F_X(x)$ and probability density function $f_X(x)$.

    Sorting the sample in ascending order yields the **order statistics**:

    $$
    X_{(1)} \leq X_{(2)} \leq \dots \leq X_{(n)}
    $$

    where $X_{(1)} = \min(X_1, \dots, X_n)$ is the sample minimum, and $X_{(n)} = \max(X_1, \dots, X_n)$ is the sample maximum.

    ---

    ### Derivation of the Cumulative Distribution Function (CDF) of the Minimum

    To find the distribution of $X_{(1)} = \min(X_1, \dots, X_n)$, we analyze the survival probability:

    $$
    F_{X_{(1)}}(x) = P(X_{(1)} \leq x) = 1 - P(X_{(1)} > x)
    $$

    The minimum of a set of values is strictly greater than $x$ if and only if **every individual sample** $X_i$ is strictly greater than $x$:

    $$
    P(X_{(1)} > x) = P(X_1 > x, X_2 > x, \dots, X_n > x)
    $$

    Because the random variables are independent and identically distributed:

    $$
    P(X_1 > x, \dots, X_n > x) = \prod_{i=1}^n P(X_i > x) = \left(1 - F_X(x)\right)^n
    $$

    Substituting this back yields the exact **CDF of the minimum**:

    $$
    F_{X_{(1)}}(x) = 1 - \left(1 - F_X(x)\right)^n
    $$

    ---

    ### Derivation of the Probability Density Function (PDF) of the Minimum

    Differentiating the CDF with respect to $x$ using the chain rule:

    $$
    f_{X_{(1)}}(x) = \frac{d}{dx} F_{X_{(1)}}(x) = \frac{d}{dx} \left[ 1 - (1 - F_X(x))^n \right]
    $$

    $$
    f_{X_{(1)}}(x) = n \left(1 - F_X(x)\right)^{n-1} f_X(x)
    $$

    #### Intuition Behind the Formula
    To have the minimum fall in an infinitesimal interval $[x, x + dx]$:
    * One of the $n$ items must land in $[x, x + dx]$ ($n \cdot f_X(x) \, dx$ ways).
    * All remaining $n - 1$ items must be strictly greater than $x$ (probability $(1 - F_X(x))^{n-1}$).

    ---

    ### Special Case 1: Exponential Distribution (Memoryless Series Systems)

    Let $X_i \sim \text{Exp}(\lambda)$ with CDF $F_X(x) = 1 - e^{-\lambda x}$ and PDF $f_X(x) = \lambda e^{-\lambda x}$ for $x \geq 0$.

    Computing the survival function:

    $$
    1 - F_X(x) = e^{-\lambda x}
    $$

    Substituting into the minimum CDF formula:

    $$
    F_{X_{(1)}}(x) = 1 - \left(e^{-\lambda x}\right)^n = 1 - e^{-n \lambda x}
    $$

    Differentiating to obtain the density:

    $$
    f_{X_{(1)}}(x) = n \lambda e^{-n \lambda x}
    $$

    #### Remarkable Mathematical Result
    The minimum of $n$ independent Exponential variables with rate $\lambda$ is itself an Exponential variable with scaled rate parameter $n\lambda$:

    $$
    X_{(1)} \sim \text{Exp}(n\lambda)
    $$

    The expected value scales inversely with $n$:

    $$
    \mathbb{E}[X_{(1)}] = \frac{1}{n\lambda} = \frac{1}{n} \mathbb{E}[X]
    $$

    If a server has Mean Time to Failure (MTTF) of $100$ hours, a cluster of $n = 10$ servers running in series has an MTTF of only $\frac{100}{10} = 10$ hours.

    ---

    ### Special Case 2: Uniform Distribution $\text{Uniform}(0, 1)$

    Let $X_i \sim \text{Uniform}(0, 1)$ where $F_X(x) = x$ and $f_X(x) = 1$ for $x \in [0, 1]$.

    $$
    F_{X_{(1)}}(x) = 1 - (1 - x)^n
    $$

    $$
    f_{X_{(1)}}(x) = n (1 - x)^{n-1}
    $$

    This corresponds to a $\text{Beta}(1, n)$ distribution. The expected minimum is:

    $$
    \mathbb{E}[X_{(1)}] = \int_0^1 x \cdot n (1 - x)^{n-1} \, dx = \frac{1}{n + 1}
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Density Compression and Empirical Verification

    The interactive subplots below demonstrate how the distribution of the minimum behaves:
    * **Left Panel**: Theoretical PDF compression for $X_{(1)} \sim \text{Exp}(n\lambda)$ with base rate $\lambda = 2.0$ across increasing sample sizes $n \in \{1, 2, 5, 10, 25\}$. As $n$ increases, probability mass is compressed toward the origin, and the initial density $f_{(1)}(0) = n\lambda$ scales linearly.
    * **Right Panel**: Empirical validation of $50,000$ simulated minimums ($n = 5$, $\lambda = 2.0$) compared against the exact analytical distribution $\text{Exp}(10.0)$, demonstrating exact correspondence between finite-sample simulation and theoretical derivation.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    lambda_param = 2.0
    x_axis_range = np.linspace(0.001, 1.6, 250)

    # Sample sizes to compare
    batch_sizes = [1, 2, 5, 10, 25]
    color_palette = ["#2563eb", "#06b6d4", "#16a34a", "#ea580c", "#dc2626"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "PDF Compression of the Minimum (λ = 2.0)",
            "Empirical Histogram vs Analytical PDF (n = 5, Rate = 10.0)",
        ],
    )

    # Left: Theoretical Curves for varying n
    for idx, n_size in enumerate(batch_sizes):
        effective_rate = n_size * lambda_param
        pdf_curve = effective_rate * np.exp(-effective_rate * x_axis_range)
        fig.add_trace(
            go.Scatter(
                x=x_axis_range,
                y=pdf_curve,
                mode="lines",
                line=dict(color=color_palette[idx], width=2.5),
                name=f"n={n_size} (Rate={effective_rate:.0f}, E={1.0 / effective_rate:.2f})",
            ),
            row=1,
            col=1,
        )

    # Right: Monte Carlo Simulation for n = 5
    rng_mc = np.random.default_rng(42)
    n_fixed_demo = 5
    n_draws = 50000
    raw_exponential_matrix = rng_mc.exponential(scale=1.0 / lambda_param, size=(n_draws, n_fixed_demo))
    simulated_minimums = np.min(raw_exponential_matrix, axis=1)

    effective_rate_demo = n_fixed_demo * lambda_param
    analytical_pdf_demo = effective_rate_demo * np.exp(-effective_rate_demo * x_axis_range)

    fig.add_trace(
        go.Histogram(
            x=simulated_minimums,
            histnorm="probability density",
            nbinsx=65,
            marker_color="#2563eb",
            opacity=0.60,
            name="Simulated Minimum (50k Trials)",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=x_axis_range,
            y=analytical_pdf_demo,
            mode="lines",
            line=dict(color="#dc2626", width=3.0),
            name=f"Analytical Exp(Rate={effective_rate_demo:.0f})",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="Value (x)", range=[0.0, 1.6], gridcolor="#f1f5f9"),
        yaxis=dict(title="Probability Density", range=[0, 52], gridcolor="#f1f5f9"),
        xaxis2=dict(title="Value (x)", range=[0.0, 0.8], gridcolor="#f1f5f9"),
        yaxis2=dict(title="Density", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        analytical_pdf_demo,
        batch_sizes,
        color_palette,
        effective_rate,
        effective_rate_demo,
        fig,
        idx,
        lambda_param,
        n_draws,
        n_fixed_demo,
        n_size,
        pdf_curve,
        raw_exponential_matrix,
        rng_mc,
        simulated_minimums,
        x_axis_range,
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

    ### Example 1: Theoretical vs Empirical Verification of Order Statistics

    In this example, we draw $M = 50,000$ independent samples across batch sizes $n \in \{1, 2, 5, 10, 25, 50\}$ for both Exponential ($\lambda = 2.0$) and Uniform ($[0, 1]$) random variables.

    For each batch size $n$, we evaluate:
    1. **Exponential Minimum**: Empirical Mean vs Theoretical $\mathbb{E}[X_{(1)}] = \frac{1}{n\lambda}$
    2. **Uniform Minimum**: Empirical Mean vs Theoretical $\mathbb{E}[X_{(1)}] = \frac{1}{n + 1}$
    3. **Relative Percentage Error**: Demonstrating convergence within $< 1\%$ across all sample sizes.
    """)
    return


@app.cell
def _(np):
    rng_ex1 = np.random.default_rng(101)
    lambda_val = 2.0
    num_experiments = 50000
    sample_sizes_tested = [1, 2, 5, 10, 25, 50]

    order_stats_records = []

    for n_count in sample_sizes_tested:
        # Exponential draws
        exp_matrix = rng_ex1.exponential(scale=1.0 / lambda_val, size=(num_experiments, n_count))
        exp_mins = np.min(exp_matrix, axis=1)
        mean_exp_empirical = float(np.mean(exp_mins))
        mean_exp_theoretical = 1.0 / (n_count * lambda_val)
        exp_rel_err = abs(mean_exp_empirical - mean_exp_theoretical) / mean_exp_theoretical * 100.0

        # Uniform draws
        unif_matrix = rng_ex1.uniform(0.0, 1.0, size=(num_experiments, n_count))
        unif_mins = np.min(unif_matrix, axis=1)
        mean_unif_empirical = float(np.mean(unif_mins))
        mean_unif_theoretical = 1.0 / (n_count + 1.0)
        unif_rel_err = abs(mean_unif_empirical - mean_unif_theoretical) / mean_unif_theoretical * 100.0

        order_stats_records.append(
            {
                "Sample Size (n)": n_count,
                "Empirical E[Exp Min]": f"{mean_exp_empirical:.4f}",
                "Theoretical E[1/nλ]": f"{mean_exp_theoretical:.4f}",
                "Exp Relative Error": f"{exp_rel_err:.2f}%",
                "Empirical E[Unif Min]": f"{mean_unif_empirical:.4f}",
                "Theoretical E[1/(n+1)]": f"{mean_unif_theoretical:.4f}",
                "Unif Relative Error": f"{unif_rel_err:.2f}%",
            }
        )

    return (
        exp_matrix,
        exp_mins,
        exp_rel_err,
        lambda_val,
        mean_exp_empirical,
        mean_exp_theoretical,
        mean_unif_empirical,
        mean_unif_theoretical,
        n_count,
        num_experiments,
        order_stats_records,
        rng_ex1,
        sample_sizes_tested,
        unif_matrix,
        unif_mins,
        unif_rel_err,
    )


@app.cell(hide_code=True)
def _(mo, order_stats_records, pd):
    df_order = pd.DataFrame(order_stats_records)
    mo.ui.table(df_order)
    return (df_order,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Distributed Cloud System Reliability (Series vs Parallel Architectures)

    In distributed systems, microservices can be wired in two canonical configurations:
    * **Series Configuration (All Must Succeed)**: System fails at $T_{\text{series}} = \min(T_1, \dots, T_n)$.
    * **Parallel Quorum Configuration (At Least One Succeeds)**: System fails at $T_{\text{parallel}} = \max(T_1, \dots, T_n)$.

    Below, we simulate a cluster of $n = 6$ nodes where each component's lifetime follows an Exponential distribution with base MTTF $= 100.0$ hours ($\lambda = 0.01$).

    We compare the Mean Time to Failure (MTTF) and the probability of survival beyond $t = 20.0$ hours for both architectures.
    """)
    return


@app.cell
def _(np):
    rng_sys = np.random.default_rng(202)
    mttf_component = 100.0
    lambda_comp = 1.0 / mttf_component
    cluster_nodes = 6
    sim_runs = 50000
    target_time_hours = 20.0

    lifetimes_matrix = rng_sys.exponential(scale=mttf_component, size=(sim_runs, cluster_nodes))

    # Series: Minimum
    series_lifetimes = np.min(lifetimes_matrix, axis=1)
    mttf_series_emp = float(np.mean(series_lifetimes))
    mttf_series_theo = mttf_component / cluster_nodes
    survival_series_emp = float(np.mean(series_lifetimes > target_time_hours)) * 100.0
    survival_series_theo = np.exp(-cluster_nodes * lambda_comp * target_time_hours) * 100.0

    # Parallel: Maximum
    parallel_lifetimes = np.max(lifetimes_matrix, axis=1)
    mttf_parallel_emp = float(np.mean(parallel_lifetimes))
    # Theoretical harmonic sum for parallel MTTF: (1/lambda) * sum(1/i for i in 1..n)
    mttf_parallel_theo = mttf_component * np.sum(1.0 / np.arange(1, cluster_nodes + 1))
    survival_parallel_emp = float(np.mean(parallel_lifetimes > target_time_hours)) * 100.0
    survival_parallel_theo = (1.0 - (1.0 - np.exp(-lambda_comp * target_time_hours)) ** cluster_nodes) * 100.0

    reliability_summary_records = [
        {
            "Architecture": f"Single Component (n=1)",
            "Governing Statistic": "T",
            "Empirical MTTF (hrs)": f"{float(np.mean(lifetimes_matrix[:, 0])):.2f}",
            "Theoretical MTTF (hrs)": f"{mttf_component:.2f}",
            "Survival P(T > 20h) Emp": f"{float(np.mean(lifetimes_matrix[:, 0] > target_time_hours)) * 100.0:.2f}%",
            "Survival P(T > 20h) Theo": f"{np.exp(-lambda_comp * target_time_hours) * 100.0:.2f}%",
        },
        {
            "Architecture": f"Series Cluster (n={cluster_nodes})",
            "Governing Statistic": "T_(1) = min(T₁..Tₙ)",
            "Empirical MTTF (hrs)": f"{mttf_series_emp:.2f}",
            "Theoretical MTTF (hrs)": f"{mttf_series_theo:.2f}",
            "Survival P(T > 20h) Emp": f"{survival_series_emp:.2f}%",
            "Survival P(T > 20h) Theo": f"{survival_series_theo:.2f}%",
        },
        {
            "Architecture": f"Parallel Redundancy (n={cluster_nodes})",
            "Governing Statistic": "T_(n) = max(T₁..Tₙ)",
            "Empirical MTTF (hrs)": f"{mttf_parallel_emp:.2f}",
            "Theoretical MTTF (hrs)": f"{mttf_parallel_theo:.2f}",
            "Survival P(T > 20h) Emp": f"{survival_parallel_emp:.2f}%",
            "Survival P(T > 20h) Theo": f"{survival_parallel_theo:.2f}%",
        },
    ]

    return (
        cluster_nodes,
        lambda_comp,
        lifetimes_matrix,
        mttf_component,
        mttf_parallel_emp,
        mttf_parallel_theo,
        mttf_series_emp,
        mttf_series_theo,
        parallel_lifetimes,
        reliability_summary_records,
        rng_sys,
        series_lifetimes,
        sim_runs,
        survival_parallel_emp,
        survival_parallel_theo,
        survival_series_emp,
        survival_series_theo,
        target_time_hours,
    )


@app.cell(hide_code=True)
def _(mo, pd, reliability_summary_records):
    df_reliability = pd.DataFrame(reliability_summary_records)
    mo.ui.table(df_reliability)
    return (df_reliability,)


if __name__ == "__main__":
    app.run()
