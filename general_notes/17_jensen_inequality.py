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
    # Note 17: Jensen's Inequality, Convexity, and Information Bounds

    &larr; Previous Note: [16 Point-Biserial Correlation](16_point_biserial.py) | Next Note: [18 Cramer V](18_cramer_v.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Jensen's inequality is arguably the most ubiquitous mathematical inequality in modern statistical learning, information theory, and generative artificial intelligence. It connects the geometric definition of function convexity directly to mathematical expectations.

    Key reasons why Jensen's inequality is essential:
    1. **The Engine of Variational Autoencoders (VAEs) and Diffusion Models**: In deep generative models with continuous latent variables $\mathbf{z}$, the true marginal log-likelihood $\ln p(\mathbf{x}) = \ln \int p(\mathbf{x}, \mathbf{z}) d\mathbf{z}$ is computationally intractable. By applying Jensen's inequality to the concave logarithm function, we derive the **Evidence Lower Bound (ELBO)**:

    $$\ln p(\mathbf{x}) \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \ln \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right] = \text{ELBO}$$

    VAEs and latent diffusion models train by maximizing this Jensen lower bound.
    2. **Expectation-Maximization (EM) Algorithm**: In Gaussian Mixture Models (GMMs) and Hidden Markov Models (HMMs), the E-step constructs a tight Jensen lower bound around the incomplete data log-likelihood, and the M-step maximizes it.
    3. **Information Theory and Non-Negativity of Relative Entropy**: Gibbs' inequality ($D_{\text{KL}}(P \| Q) \geq 0$), which establishes that cross-entropy is always greater than or equal to true entropy, is proven in two steps via Jensen's inequality on $f(t) = -\ln(t)$.
    4. **The Arithmetic Mean - Geometric Mean (AM-GM) Inequality**: The fundamental inequality $\frac{1}{n} \sum x_i \geq (\prod x_i)^{1/n}$ is a direct corollary of Jensen's inequality applied to the concave logarithmic function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Definition of Convex and Concave Functions

    A real-valued function $f: \mathcal{C} \to \mathbb{R}$ defined on a convex set $\mathcal{C} \subseteq \mathbb{R}^d$ is **convex** if for all $\mathbf{x}_1, \mathbf{x}_2 \in \mathcal{C}$ and all $\alpha \in [0, 1]$:

    $$
    f(\alpha \mathbf{x}_1 + (1 - \alpha) \mathbf{x}_2) \leq \alpha f(\mathbf{x}_1) + (1 - \alpha) f(\mathbf{x}_2)
    $$

    #### Geometric Meaning
    The secant line segment (chord) connecting any two points $(\mathbf{x}_1, f(\mathbf{x}_1))$ and $(\mathbf{x}_2, f(\mathbf{x}_2))$ on the graph lies entirely above or on the function curve.

    * **Second-Order Condition**: If $f$ is twice differentiable, $f$ is convex if and only if its Hessian is positive semi-definite everywhere: $\nabla^2 f(\mathbf{x}) \succeq 0$.
    * **Concave Function**: A function $g$ is concave if $-g$ is convex, meaning the secant line segment lies below or on the graph. Examples include $\ln(x)$ and $\sqrt{x}$.

    ---

    ### Statement of Jensen's Inequality

    #### Convex Function Case
    Let $f$ be a convex function, and let $X$ be a random variable with finite expectation $\mathbb{E}[X] < \infty$. Then:

    $$
    f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]
    $$

    The function evaluated at the mean is less than or equal to the expected value of the function.

    #### Concave Function Case
    Let $g$ be a concave function (such as $\ln(x)$). Reversing the inequality yields:

    $$
    g(\mathbb{E}[X]) \geq \mathbb{E}[g(X)]
    $$

    ---

    ### The Jensen Gap and Second-Order Taylor Approximation

    The difference between the two sides of Jensen's inequality is the **Jensen gap**:

    $$
    \Delta_J = \mathbb{E}[f(X)] - f(\mathbb{E}[X]) \geq 0
    $$

    Performing a second-order Taylor expansion of $f(X)$ around the mean $\mu = \mathbb{E}[X]$:

    $$
    f(X) \approx f(\mu) + f'(\mu)(X - \mu) + \frac{1}{2} f''(\mu)(X - \mu)^2
    $$

    Taking mathematical expectations on both sides:

    $$
    \mathbb{E}[f(X)] \approx f(\mathbb{E}[X]) + 0 + \frac{1}{2} f''(\mathbb{E}[X]) \text{Var}(X)
    $$

    Therefore, the Jensen gap is approximately:

    $$
    \Delta_J \approx \frac{1}{2} f''(\mathbb{E}[X]) \text{Var}(X)
    $$

    This fundamental formula reveals that the magnitude of the Jensen gap is governed by two factors:
    1. The curvature of the function ($f''(\mu)$).
    2. The variance of the random variable ($\text{Var}(X)$).
    If either $f$ is linear ($f'' = 0$) or $X$ is deterministic ($\text{Var}(X) = 0$), the gap vanishes with exact equality.

    ---

    ### Derivation of the Evidence Lower Bound (ELBO) in Variational Inference

    In generative modeling, let $\mathbf{x}$ be observed data and $\mathbf{z}$ be latent variables. We want to maximize the marginal log-likelihood:

    $$
    \ln p(\mathbf{x}) = \ln \int p(\mathbf{x}, \mathbf{z}) \, d\mathbf{z}
    $$

    Introducing an arbitrary proposal distribution $q(\mathbf{z}|\mathbf{x})$ over the latent space:

    $$
    \ln p(\mathbf{x}) = \ln \int q(\mathbf{z}|\mathbf{x}) \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \, d\mathbf{z} = \ln \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right]
    $$

    Since the natural logarithm function $\ln(\cdot)$ is strictly concave, applying Jensen's inequality yields:

    $$
    \ln \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right] \geq \mathbb{E}_{q(\mathbf{z}|\mathbf{x})}\left[ \ln \frac{p(\mathbf{x}, \mathbf{z})}{q(\mathbf{z}|\mathbf{x})} \right] = \text{ELBO}
    $$

    The difference between the marginal log-likelihood and the ELBO is the Kullback-Leibler divergence:

    $$
    \ln p(\mathbf{x}) - \text{ELBO} = D_{\text{KL}}\left( q(\mathbf{z}|\mathbf{x}) \,\|\, p(\mathbf{z}|\mathbf{x}) \right) \geq 0
    $$

    Maximizing the ELBO directly minimizes the divergence between the variational posterior $q(\mathbf{z}|\mathbf{x})$ and the true posterior $p(\mathbf{z}|\mathbf{x})$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Geometric Secant Chords and the Jensen Gap

    The interactive subplots below display the geometric foundation of Jensen's inequality:
    * **Left Panel**: Convex parabola $f(x) = x^2$. A discrete random variable takes values $x_1 = 1.0$ ($40\%$ probability) and $x_2 = 5.0$ ($60\%$ probability). The secant chord connects the points on the curve. The mean $\mathbb{E}[X] = 3.4$ evaluates to $f(\mathbb{E}[X]) = 11.56$ on the curve, while $\mathbb{E}[f(X)] = 15.40$ sits on the secant chord. The vertical purple bar marks the Jensen gap $\Delta = 3.84$.
    * **Right Panel**: Concave logarithm $g(x) = \ln(x)$. Connecting $x_1 = 0.6$ and $x_2 = 4.5$ with equal weights ($p = 0.5$) shows the secant chord lying entirely beneath the curve: $\mathbb{E}[\ln X] = 0.50 \leq \ln(\mathbb{E}[X]) = 0.94$, demonstrating the classical AM-GM inequality.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Left Panel: Convex Function f(x) = x^2
    x_grid_left = np.linspace(0.4, 5.6, 200)
    f_convex = x_grid_left**2

    x1_val, x2_val = 1.0, 5.0
    p1_val, p2_val = 0.4, 0.6
    e_x_convex = p1_val * x1_val + p2_val * x2_val  # 3.4
    f_e_x = e_x_convex**2  # 11.56
    e_f_x = p1_val * (x1_val**2) + p2_val * (x2_val**2)  # 15.40
    jensen_gap_convex = e_f_x - f_e_x  # 3.84

    # Right Panel: Concave Function g(x) = ln(x)
    x_grid_right = np.linspace(0.3, 5.5, 200)
    g_concave = np.log(x_grid_right)

    x1_log, x2_log = 0.6, 4.5
    e_x_log = 0.5 * x1_log + 0.5 * x2_log  # 2.55
    g_e_x = float(np.log(e_x_log))  # 0.936
    e_g_x = float(0.5 * np.log(x1_log) + 0.5 * np.log(x2_log))  # 0.497
    jensen_gap_log = g_e_x - e_g_x  # 0.439

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Convex f(x) = x²: f(E[X]) ≤ E[f(X)] (Gap = {jensen_gap_convex:.2f})",
            f"Concave g(x) = ln(x): E[ln X] ≤ ln(E[X]) (Gap = {jensen_gap_log:.2f})",
        ],
    )

    # Left: Curve
    fig.add_trace(
        go.Scatter(
            x=x_grid_left,
            y=f_convex,
            mode="lines",
            line=dict(color="#2563eb", width=3.0),
            name="Convex Curve f(x) = x²",
        ),
        row=1,
        col=1,
    )

    # Left: Secant Chord
    fig.add_trace(
        go.Scatter(
            x=[x1_val, x2_val],
            y=[x1_val**2, x2_val**2],
            mode="lines+markers",
            line=dict(color="#ea580c", width=2.5, dash="dash"),
            marker=dict(size=9, color="#ea580c"),
            name="Secant Chord (Expectation Line)",
        ),
        row=1,
        col=1,
    )

    # Left: Point on curve f(E[X])
    fig.add_trace(
        go.Scatter(
            x=[e_x_convex],
            y=[f_e_x],
            mode="markers",
            marker=dict(size=12, color="#16a34a", symbol="circle"),
            name=f"f(E[X]) = {f_e_x:.2f}",
        ),
        row=1,
        col=1,
    )

    # Left: Point on chord E[f(X)]
    fig.add_trace(
        go.Scatter(
            x=[e_x_convex],
            y=[e_f_x],
            mode="markers",
            marker=dict(size=12, color="#dc2626", symbol="diamond"),
            name=f"E[f(X)] = {e_f_x:.2f}",
        ),
        row=1,
        col=1,
    )

    # Left: Jensen Gap Bar
    fig.add_trace(
        go.Scatter(
            x=[e_x_convex, e_x_convex],
            y=[f_e_x, e_f_x],
            mode="lines",
            line=dict(color="#7c3aed", width=3.5),
            name=f"Jensen Gap Δ = {jensen_gap_convex:.2f}",
        ),
        row=1,
        col=1,
    )

    # Right: Concave Log Curve
    fig.add_trace(
        go.Scatter(
            x=x_grid_right,
            y=g_concave,
            mode="lines",
            line=dict(color="#2563eb", width=3.0),
            name="Concave Curve g(x) = ln(x)",
        ),
        row=1,
        col=2,
    )

    # Right: Secant Chord
    fig.add_trace(
        go.Scatter(
            x=[x1_log, x2_log],
            y=[np.log(x1_log), np.log(x2_log)],
            mode="lines+markers",
            line=dict(color="#ea580c", width=2.5, dash="dash"),
            marker=dict(size=9, color="#ea580c"),
            name="Secant Chord E[ln X]",
        ),
        row=1,
        col=2,
    )

    # Right: Point on curve ln(E[X])
    fig.add_trace(
        go.Scatter(
            x=[e_x_log],
            y=[g_e_x],
            mode="markers",
            marker=dict(size=12, color="#16a34a", symbol="circle"),
            name=f"ln(E[X]) = {g_e_x:.2f}",
        ),
        row=1,
        col=2,
    )

    # Right: Point on chord E[ln X]
    fig.add_trace(
        go.Scatter(
            x=[e_x_log],
            y=[e_g_x],
            mode="markers",
            marker=dict(size=12, color="#dc2626", symbol="diamond"),
            name=f"E[ln X] = {e_g_x:.2f}",
        ),
        row=1,
        col=2,
    )

    # Right: AM-GM Gap Bar
    fig.add_trace(
        go.Scatter(
            x=[e_x_log, e_x_log],
            y=[e_g_x, g_e_x],
            mode="lines",
            line=dict(color="#7c3aed", width=3.5),
            name=f"AM-GM Log Gap = {jensen_gap_log:.2f}",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(title="x", range=[0.0, 5.8], gridcolor="#f1f5f9"),
        yaxis=dict(title="f(x) = x²", range=[0, 30], gridcolor="#f1f5f9"),
        xaxis2=dict(title="x", range=[0.0, 5.8], gridcolor="#f1f5f9"),
        yaxis2=dict(title="g(x) = ln(x)", range=[-1.2, 2.0], gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.26, xanchor="center", x=0.5),
    )

    return (
        e_f_x,
        e_g_x,
        e_x_convex,
        e_x_log,
        f_convex,
        f_e_x,
        fig,
        g_concave,
        g_e_x,
        jensen_gap_convex,
        jensen_gap_log,
        p1_val,
        p2_val,
        x1_log,
        x1_val,
        x2_log,
        x2_val,
        x_grid_left,
        x_grid_right,
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

    ### Example 1: Numerical Verification of Jensen's Inequality and Second-Order Taylor Approximation

    In this example, we draw $N = 100,000$ samples from an Exponential distribution $X \sim \text{Exp}(\lambda = 1.5)$ ($\mu = \frac{1}{1.5} \approx 0.667, \sigma^2 = \frac{1}{1.5^2} \approx 0.444$).

    We evaluate:
    1. $f_1(x) = x^2$ (convex)
    2. $f_2(x) = e^{0.8 x}$ (convex)
    3. $g_1(x) = -\ln(x)$ (convex)
    4. $g_2(x) = \sqrt{x}$ (concave)

    For each function, we verify the inequality and compare the empirical Jensen gap with the second-order Taylor approximation $\Delta_J \approx \frac{1}{2} f''(\mu) \sigma^2$.
    """)
    return


@app.cell
def _(np):
    rng_ex1 = np.random.default_rng(101)
    lambda_param_ex = 1.5
    n_sample_ex = 100000

    x_samples_ex = rng_ex1.exponential(scale=1.0 / lambda_param_ex, size=n_sample_ex)
    mu_emp = float(np.mean(x_samples_ex))
    var_emp = float(np.var(x_samples_ex))

    test_functions = [
        ("Quadratic f(x) = x²", lambda t: t**2, lambda t: 2.0, "Convex", "f(E[X]) ≤ E[f(X)]"),
        ("Exponential f(x) = e^(0.8x)", lambda t: np.exp(0.8 * t), lambda t: 0.64 * np.exp(0.8 * t), "Convex", "f(E[X]) ≤ E[f(X)]"),
        ("Negative Log f(x) = -ln(x)", lambda t: -np.log(t), lambda t: 1.0 / (t**2), "Convex", "f(E[X]) ≤ E[f(X)]"),
        ("Square Root g(x) = √x", lambda t: np.sqrt(t), lambda t: -0.25 * (t**(-1.5)), "Concave", "E[g(X)] ≤ g(E[X])"),
    ]

    jensen_verification_records = []

    for name_func, fn, d2_fn, curvature_type, ineq_rule in test_functions:
        f_of_mean = float(fn(mu_emp))
        mean_of_f = float(np.mean(fn(x_samples_ex)))
        empirical_gap = abs(mean_of_f - f_of_mean)
        approx_taylor_gap = abs(0.5 * d2_fn(mu_emp) * var_emp)

        if curvature_type == "Convex":
            satisfied = f_of_mean <= (mean_of_f + 1e-7)
        else:
            satisfied = mean_of_f <= (f_of_mean + 1e-7)

        jensen_verification_records.append(
            {
                "Function": name_func,
                "Type": curvature_type,
                "f(E[X])": f"{f_of_mean:.4f}",
                "E[f(X)]": f"{mean_of_f:.4f}",
                "Empirical Gap": f"{empirical_gap:.4f}",
                "Taylor Approx (0.5 f'' σ²)": f"{approx_taylor_gap:.4f}",
                "Inequality Satisfied": str(satisfied),
            }
        )

    return (
        approx_taylor_gap,
        curvature_type,
        d2_fn,
        empirical_gap,
        f_of_mean,
        fn,
        ineq_rule,
        jensen_verification_records,
        lambda_param_ex,
        mean_of_f,
        mu_emp,
        n_sample_ex,
        name_func,
        rng_ex1,
        satisfied,
        test_functions,
        var_emp,
        x_samples_ex,
    )


@app.cell(hide_code=True)
def _(jensen_verification_records, mo, pd):
    df_jensen = pd.DataFrame(jensen_verification_records)
    mo.ui.table(df_jensen)
    return (df_jensen,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Variational Autoencoder (VAE) ELBO Decomposition

    In a latent variable model, observed data $x$ is generated via latent code $z \sim \mathcal{N}(0, 1)$ with emission probability $p(x|z) = \mathcal{N}(z, \sigma_x^2 = 1.0)$.

    For an observation $x = 2.0$:
    1. The true marginal likelihood is computed via exact Gaussian convolution:

    $$p(x) = \mathcal{N}(0, \sigma_z^2 + \sigma_x^2 = 2.0) \implies \ln p(x) \approx -1.8663$$

    2. We test three variational candidate posteriors $q(z) = \mathcal{N}(\mu_q, \sigma_q^2)$:
       * **Poor Proposal**: $\mu_q = -1.0, \sigma_q = 0.5$
       * **Moderate Proposal**: $\mu_q = 0.5, \sigma_q = 0.8$
       * **Optimal Posterior**: $\mu_q = 1.0, \sigma_q = 1 / \sqrt{2} \approx 0.707$

    We verify that $\text{ELBO}(q) \leq \ln p(x)$ holds universally, with the Jensen gap exactly matching $D_{\text{KL}}(q \| p(z|x))$.
    """)
    return


@app.cell
def _(np):
    x_obs = 2.0
    sigma_prior = 1.0
    sigma_likelihood = 1.0

    # True marginal likelihood: X ~ N(0, 1 + 1 = 2)
    var_marginal = sigma_prior**2 + sigma_likelihood**2
    p_x_exact = (1.0 / np.sqrt(2.0 * np.pi * var_marginal)) * np.exp(-0.5 * (x_obs**2) / var_marginal)
    log_p_exact = float(np.log(p_x_exact))

    # True posterior p(z|x): N(x/2 = 1.0, 1/2 = 0.5)
    mu_true_post = x_obs / 2.0
    var_true_post = 0.5
    sd_true_post = np.sqrt(var_true_post)

    variational_proposals = [
        ("Candidate A (Poorly Fitted)", -1.0, 0.50),
        ("Candidate B (Moderate Fit)", 0.50, 0.80),
        ("Candidate C (Exact Posterior)", mu_true_post, sd_true_post),
    ]

    elbo_decomposition_records = []
    rng_vae = np.random.default_rng(202)
    num_mc_latent = 200000

    for label_q, mu_q, sd_q in variational_proposals:
        z_samples = rng_vae.normal(mu_q, sd_q, size=num_mc_latent)

        # log p(z)
        log_pz = -0.5 * np.log(2.0 * np.pi) - 0.5 * (z_samples**2)
        # log p(x|z)
        log_px_given_z = -0.5 * np.log(2.0 * np.pi) - 0.5 * ((x_obs - z_samples) ** 2)
        # log q(z)
        log_qz = -0.5 * np.log(2.0 * np.pi * (sd_q**2)) - 0.5 * ((z_samples - mu_q) ** 2) / (sd_q**2)

        # ELBO = E_q [ log p(x, z) - log q(z) ]
        elbo_estimate = float(np.mean(log_pz + log_px_given_z - log_qz))

        # KL divergence between Gaussian proposals: KL(N(mu_q, sd_q^2) || N(mu_post, var_post))
        kl_div = float(
            np.log(sd_true_post / sd_q)
            + (sd_q**2 + (mu_q - mu_true_post) ** 2) / (2.0 * var_true_post)
            - 0.5
        )

        elbo_decomposition_records.append(
            {
                "Variational Family q(z)": label_q,
                "Parameters (μ_q, σ_q)": f"μ={mu_q:.2f}, σ={sd_q:.3f}",
                "True log p(x)": f"{log_p_exact:.4f}",
                "ELBO Lower Bound": f"{elbo_estimate:.4f}",
                "Jensen Gap (KL Divergence)": f"{kl_div:.4f}",
                "Lower Bound Holds?": str(elbo_estimate <= log_p_exact + 1e-4),
            }
        )

    return (
        elbo_decomposition_records,
        elbo_estimate,
        kl_div,
        label_q,
        log_p_exact,
        log_px_given_z,
        log_pz,
        log_qz,
        mu_q,
        mu_true_post,
        num_mc_latent,
        p_x_exact,
        rng_vae,
        sd_q,
        sd_true_post,
        sigma_likelihood,
        sigma_prior,
        var_marginal,
        var_true_post,
        variational_proposals,
        x_obs,
        z_samples,
    )


@app.cell(hide_code=True)
def _(elbo_decomposition_records, mo, pd):
    df_elbo = pd.DataFrame(elbo_decomposition_records)
    mo.ui.table(df_elbo)
    return (df_elbo,)


if __name__ == "__main__":
    app.run()
