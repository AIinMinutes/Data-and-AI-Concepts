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
    # Note 12: Multivariate Normal Distribution and Covariance Geometry

    &larr; Previous Note: [11 Empirical CDF](11_ecdf.py) | Next Note: [13 Unbiased vs Consistent](13_unbiased_vs_consistent.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    The Multivariate Normal (Gaussian) distribution is the cornerstone of modern probabilistic modeling, machine learning, and statistical signal processing. Whenever real-valued random vectors arise from multiple aggregated micro-effects, the Multivariate Central Limit Theorem dictates that their joint distribution approaches a Gaussian.

    Core applications across AI, statistics, and data science:
    1. **The Maximum Entropy Principle**: Among all continuous multivariate distributions with a specified mean $\boldsymbol{\mu}$ and covariance matrix $\boldsymbol{\Sigma}$, the Multivariate Normal uniquely maximizes differential entropy. It is the most conservative and honest probability model when only first and second moments are known.
    2. **Generative Modeling and VAEs**: Variational Autoencoders (VAEs) and Diffusion Models rely on isotropic Gaussian latent priors $p(\mathbf{z}) = \mathcal{N}(\mathbf{0}, \mathbf{I})$ and Gaussian posterior approximations $q_\phi(\mathbf{z}|\mathbf{x}) = \mathcal{N}(\boldsymbol{\mu}_\phi(\mathbf{x}), \text{diag}(\boldsymbol{\sigma}_\phi^2(\mathbf{x})))$, utilizing the reparameterization trick $\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}$.
    3. **Clustering and Classification**: Gaussian Mixture Models (GMMs) model complex multi-modal density landscapes via Expectation-Maximization (EM). Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA) construct Bayes-optimal decision boundaries using Gaussian assumptions.
    4. **Kalman Filtering and State Estimation**: Tracking algorithms in robotics, autonomous driving, and aerospace represent dynamic uncertainty as propagating Gaussian belief states through linear-Gaussian state space updates.
    5. **Mahalanobis Distance and Outlier Detection**: In high dimensions with correlated features, standard Euclidean distance $\|\mathbf{x} - \boldsymbol{\mu}\|_2$ is misleading. The covariance-weighted Mahalanobis distance accounts for variance and correlation, following an exact Chi-Square distribution for principled anomaly detection.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Mathematical Definition of the Density Function

    A random vector $\mathbf{X} \in \mathbb{R}^d$ follows a Multivariate Normal distribution $\mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ if its probability density function (PDF) is given by:

    $$
    f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} \det(\boldsymbol{\Sigma})^{1/2}} \exp\left(-\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})\right)
    $$

    where:
    * $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}] \in \mathbb{R}^d$ is the population mean vector.
    * $\boldsymbol{\Sigma} = \mathbb{E}[(\mathbf{X} - \boldsymbol{\mu})(\mathbf{X} - \boldsymbol{\mu})^T] \in \mathbb{R}^{d \times d}$ is the symmetric positive definite covariance matrix ($\boldsymbol{\Sigma} = \boldsymbol{\Sigma}^T$, $\mathbf{v}^T \boldsymbol{\Sigma} \mathbf{v} > 0$ for all $\mathbf{v} \neq \mathbf{0}$).
    * $\boldsymbol{\Sigma}^{-1}$ is the **precision matrix** (or concentration matrix), whose off-diagonal entries directly encode conditional independence structures in Gaussian Graphical Models.

    ---

    ### The Mahalanobis Distance and Chi-Square Level Sets

    The exponent of the Gaussian density defines the **squared Mahalanobis distance**:

    $$
    D_M^2(\mathbf{x}) = (\mathbf{x} - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu})
    $$

    Level sets of constant probability density $f(\mathbf{x}) = c$ are hyper-ellipsoids defined by $D_M^2(\mathbf{x}) = \text{constant}$.

    #### Distribution of the Mahalanobis Distance
    If $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the squared Mahalanobis distance follows an exact Chi-Square distribution with $d$ degrees of freedom:

    $$
    D_M^2(\mathbf{X}) \sim \chi^2(d)
    $$

    In two dimensions ($d = 2$), the cumulative distribution function of $\chi^2(2)$ is exponential: $P(D_M^2 \leq c) = 1 - e^{-c/2}$.
    * $c = 1.0$ ($1\sigma$ contour): $1 - e^{-0.5} \approx 39.3\%$ coverage.
    * $c = 4.0$ ($2\sigma$ contour): $1 - e^{-2.0} \approx 86.5\%$ coverage.
    * $c = 5.991$ ($95\%$ contour): $\chi^2_{0.05, 2} \approx 5.991$.
    * $c = 9.0$ ($3\sigma$ contour): $1 - e^{-4.5} \approx 98.9\%$ coverage.

    ---

    ### Covariance Geometry and Mahalanobis Whitening

    By the Spectral Theorem (Note 07), the real symmetric positive definite matrix $\boldsymbol{\Sigma}$ has orthogonal eigendecomposition:

    $$
    \boldsymbol{\Sigma} = \mathbf{Q} \mathbf{\Lambda} \mathbf{Q}^T
    $$

    where $\mathbf{Q} = [\mathbf{q}_1, \dots, \mathbf{q}_d]$ contains orthonormal column eigenvectors, and $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_d)$ contains strictly positive eigenvalues $\lambda_i > 0$.

    * **Eigenvectors $\mathbf{q}_i$**: Point along the principal axes of rotation of the probability density ellipsoids.
    * **Eigenvalues $\lambda_i$**: Measure the variance along the $i$-th principal axis. The half-axis lengths of the $k$-sigma ellipsoid are given by $k \sqrt{\lambda_i}$.

    #### Mahalanobis Whitening Transformation
    Given correlated data $\mathbf{X} \sim \mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, we define the whitening matrix $\mathbf{W} = \mathbf{\Sigma}^{-1/2} = \mathbf{Q} \mathbf{\Lambda}^{-1/2} \mathbf{Q}^T$. Transforming the data:

    $$
    \mathbf{Z} = \mathbf{\Sigma}^{-1/2} (\mathbf{X} - \boldsymbol{\mu})
    $$

    Evaluating the covariance of $\mathbf{Z}$:

    $$
    \text{Cov}(\mathbf{Z}) = \mathbf{\Sigma}^{-1/2} \boldsymbol{\Sigma} \mathbf{\Sigma}^{-1/2} = \mathbf{\Sigma}^{-1/2} \mathbf{\Sigma}^{1/2} \mathbf{\Sigma}^{1/2} \mathbf{\Sigma}^{-1/2} = \mathbf{I}_d
    $$

    The whitened random vector $\mathbf{Z}$ follows a standard spherical isotropic Gaussian $\mathcal{N}_d(\mathbf{0}, \mathbf{I}_d)$, completely removing all feature correlations and equalizing variances.

    ---

    ### Maximum Likelihood Estimation (MLE)

    Given an i.i.d. sample $\mathbf{x}_1, \dots, \mathbf{x}_N$ from $\mathcal{N}_d(\boldsymbol{\mu}, \boldsymbol{\Sigma})$, the log-likelihood function is:

    $$
    \ln \mathcal{L}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) = -\frac{N d}{2} \ln(2\pi) - \frac{N}{2} \ln \det(\boldsymbol{\Sigma}) - \frac{1}{2} \sum_{i=1}^N (\mathbf{x}_i - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{x}_i - \boldsymbol{\mu})
    $$

    Using matrix calculus (Note 08), maximizing with respect to $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$ yields the closed-form MLE estimators:

    $$
    \hat{\boldsymbol{\mu}}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i
    $$

    $$
    \hat{\boldsymbol{\Sigma}}_{\text{MLE}} = \frac{1}{N} \sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T
    $$

    While $\hat{\boldsymbol{\mu}}_{\text{MLE}}$ is unconditionally unbiased ($\mathbb{E}[\hat{\boldsymbol{\mu}}] = \boldsymbol{\mu}$), the covariance MLE is slightly biased: $\mathbb{E}[\hat{\boldsymbol{\Sigma}}_{\text{MLE}}] = \frac{N-1}{N} \boldsymbol{\Sigma}$. The unbiased sample covariance divides by $N - 1$:

    $$
    \mathbf{S} = \frac{1}{N - 1} \sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^T
    $$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Covariance Geometry, Density Contours, and 3D Surface

    The interactive subplots below display the structural geometry of the Bivariate Normal distribution:
    * **Left Panel**: 2D level contours of probability density alongside $N = 500$ drawn samples. The orthogonal principal axes $\mathbf{q}_1, \mathbf{q}_2$ (scaled by $2\sqrt{\lambda_i}$) radiate from the mean vector $\boldsymbol{\mu} = [2.0, 3.0]^T$, demonstrating that the eigenvectors dictate the orientation and stretching of the covariance ellipse.
    * **Right Panel**: 3D probability density surface $f(x_1, x_2)$, illustrating the mode at $(\boldsymbol{\mu}, f(\boldsymbol{\mu}))$ and the exponential drop-off dictated by the Mahalanobis distance.
    """)
    return


@app.cell
def _(go, make_subplots, np, stats):
    rng_mvn = np.random.default_rng(42)

    # Population parameters
    mu_true = np.array([2.0, 3.0])
    sigma_true = np.array([[3.0, 1.8], [1.8, 2.0]])

    # Spectral decomposition of Sigma
    eigvals, eigvecs = np.linalg.eigh(sigma_true)
    # Sort descending
    sort_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_indices]
    eigvecs = eigvecs[:, sort_indices]

    # Sample points
    n_pts_vis = 500
    sample_points = rng_mvn.multivariate_normal(mu_true, sigma_true, size=n_pts_vis)

    # 2D Grid
    x1_axis = np.linspace(-3.0, 7.0, 80)
    x2_axis = np.linspace(-2.0, 8.0, 80)
    x1_grid, x2_grid = np.meshgrid(x1_axis, x2_axis)
    coords_stacked = np.dstack((x1_grid, x2_grid))

    mvn_distribution = stats.multivariate_normal(mu_true, sigma_true)
    density_grid = mvn_distribution.pdf(coords_stacked)

    # Principal axes vectors (scaled by 2 * sqrt(lambda))
    axis1_vec = eigvecs[:, 0] * 2.0 * np.sqrt(eigvals[0])
    axis2_vec = eigvecs[:, 1] * 2.0 * np.sqrt(eigvals[1])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "xy"}, {"type": "scene"}]],
        subplot_titles=[
            "2D Density Contours & Covariance Principal Axes",
            "3D Bivariate Gaussian Density Surface",
        ],
    )

    # Left: Contours
    fig.add_trace(
        go.Contour(
            z=density_grid,
            x=x1_axis,
            y=x2_axis,
            colorscale="Viridis",
            opacity=0.75,
            showscale=False,
            contours=dict(showlines=True),
            name="Density Contours",
        ),
        row=1,
        col=1,
    )

    # Left: Sample Scatter
    fig.add_trace(
        go.Scatter(
            x=sample_points[:, 0],
            y=sample_points[:, 1],
            mode="markers",
            marker=dict(size=4, color="#64748b", opacity=0.45),
            name="Sampled Points (N=500)",
        ),
        row=1,
        col=1,
    )

    # Left: Mean marker
    fig.add_trace(
        go.Scatter(
            x=[mu_true[0]],
            y=[mu_true[1]],
            mode="markers",
            marker=dict(size=12, color="#dc2626", symbol="cross"),
            name="Mean μ = [2, 3]",
        ),
        row=1,
        col=1,
    )

    # Left: Principal Axis 1 (Largest Variance)
    fig.add_trace(
        go.Scatter(
            x=[mu_true[0], mu_true[0] + axis1_vec[0]],
            y=[mu_true[1], mu_true[1] + axis1_vec[1]],
            mode="lines+markers",
            line=dict(color="#dc2626", width=3.5),
            marker=dict(size=7, color="#dc2626"),
            name=f"Principal Axis 1 (2√λ₁={2.0 * np.sqrt(eigvals[0]):.2f})",
        ),
        row=1,
        col=1,
    )

    # Left: Principal Axis 2 (Smallest Variance)
    fig.add_trace(
        go.Scatter(
            x=[mu_true[0], mu_true[0] + axis2_vec[0]],
            y=[mu_true[1], mu_true[1] + axis2_vec[1]],
            mode="lines+markers",
            line=dict(color="#16a34a", width=3.5),
            marker=dict(size=7, color="#16a34a"),
            name=f"Principal Axis 2 (2√λ₂={2.0 * np.sqrt(eigvals[1]):.2f})",
        ),
        row=1,
        col=1,
    )

    # Right: 3D Surface
    fig.add_trace(
        go.Surface(
            z=density_grid,
            x=x1_axis,
            y=x2_axis,
            colorscale="Viridis",
            opacity=0.85,
            showscale=False,
            name="PDF Surface",
        ),
        row=1,
        col=2,
    )

    # Right: Peak mode marker
    fig.add_trace(
        go.Scatter3d(
            x=[mu_true[0]],
            y=[mu_true[1]],
            z=[float(mvn_distribution.pdf(mu_true))],
            mode="markers",
            marker=dict(size=8, color="#dc2626", symbol="diamond"),
            name="Peak Mode f(μ)",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=30, r=30, t=50, b=30),
        xaxis=dict(
            title="X₁",
            range=[-3.0, 7.0],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            gridcolor="#f1f5f9",
        ),
        yaxis=dict(
            title="X₂",
            range=[-2.0, 8.0],
            zeroline=True,
            zerolinecolor="#cbd5e1",
            gridcolor="#f1f5f9",
        ),
        scene=dict(
            xaxis=dict(title="X₁", gridcolor="#f1f5f9"),
            yaxis=dict(title="X₂", gridcolor="#f1f5f9"),
            zaxis=dict(title="Density f(X)", gridcolor="#f1f5f9"),
            camera=dict(eye=dict(x=-1.5, y=-1.5, z=1.3)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )

    return (
        axis1_vec,
        axis2_vec,
        coords_stacked,
        density_grid,
        eigvals,
        eigvecs,
        fig,
        mu_true,
        mvn_distribution,
        n_pts_vis,
        rng_mvn,
        sample_points,
        sigma_true,
        sort_indices,
        x1_axis,
        x1_grid,
        x2_axis,
        x2_grid,
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

    ### Example 1: Maximum Likelihood Estimation and Mahalanobis Whitening

    In this example, we generate $N = 1,000$ points from a correlated bivariate normal distribution $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$. We:
    1. Estimate $\hat{\boldsymbol{\mu}}_{\text{MLE}}$ and unbiased sample covariance $\mathbf{S}$.
    2. Apply the Mahalanobis whitening transformation:

    $$\mathbf{Z} = \boldsymbol{\Sigma}^{-1/2} (\mathbf{X} - \boldsymbol{\mu})$$

    3. Verify that the transformed covariance matrix equals the identity matrix $\mathbf{I}_2$.
    """)
    return


@app.cell
def _(np):
    rng_ex1 = np.random.default_rng(101)
    sample_size_mle = 1000

    mu_pop = np.array([2.5, -1.5])
    sigma_pop = np.array([[4.0, 2.2], [2.2, 3.0]])

    # Generate samples
    raw_data = rng_ex1.multivariate_normal(mu_pop, sigma_pop, size=sample_size_mle)

    # MLE estimates
    mu_hat_mle = np.mean(raw_data, axis=0)
    centered_data = raw_data - mu_hat_mle
    sigma_hat_mle = (centered_data.T @ centered_data) / sample_size_mle
    sigma_unbiased_s = (centered_data.T @ centered_data) / (sample_size_mle - 1)

    # Mahalanobis Whitening
    eigenvalues_pop, eigenvectors_pop = np.linalg.eigh(sigma_pop)
    whitening_op = eigenvectors_pop @ np.diag(1.0 / np.sqrt(eigenvalues_pop)) @ eigenvectors_pop.T
    whitened_samples = (raw_data - mu_pop) @ whitening_op
    cov_whitened_samples = np.cov(whitened_samples, rowvar=False)

    max_whitening_error = float(np.max(np.abs(cov_whitened_samples - np.eye(2))))

    mle_whitening_summary = {
        "Metric / Parameter": [
            "Mean Vector μ",
            "Variance Var(X₁)",
            "Variance Var(X₂)",
            "Covariance Cov(X₁, X₂)",
            "Correlation Corr(X₁, X₂)",
            "Whitened Covariance Max |Cov(Z) - I|",
        ],
        "True Population Value": [
            f"[{mu_pop[0]:.2f}, {mu_pop[1]:.2f}]",
            f"{sigma_pop[0, 0]:.4f}",
            f"{sigma_pop[1, 1]:.4f}",
            f"{sigma_pop[0, 1]:.4f}",
            f"{sigma_pop[0, 1] / np.sqrt(sigma_pop[0, 0] * sigma_pop[1, 1]):.4f}",
            "0.0000 (Exact Identity)",
        ],
        "Sample Estimate": [
            f"[{mu_hat_mle[0]:.2f}, {mu_hat_mle[1]:.2f}]",
            f"{sigma_unbiased_s[0, 0]:.4f}",
            f"{sigma_unbiased_s[1, 1]:.4f}",
            f"{sigma_unbiased_s[0, 1]:.4f}",
            f"{sigma_unbiased_s[0, 1] / np.sqrt(sigma_unbiased_s[0, 0] * sigma_unbiased_s[1, 1]):.4f}",
            f"{max_whitening_error:.4f} (Identity restored)",
        ],
    }

    return (
        centered_data,
        cov_whitened_samples,
        eigenvalues_pop,
        eigenvectors_pop,
        max_whitening_error,
        mle_whitening_summary,
        mu_hat_mle,
        mu_pop,
        raw_data,
        sample_size_mle,
        sigma_hat_mle,
        sigma_pop,
        sigma_unbiased_s,
        whitened_samples,
        whitening_op,
    )


@app.cell(hide_code=True)
def _(mle_whitening_summary, mo, pd):
    df_mle = pd.DataFrame(mle_whitening_summary)
    mo.ui.table(df_mle)
    return (df_mle,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Mahalanobis Distance vs Euclidean Distance for Anomaly Detection

    In correlated multivariate spaces, Euclidean distance $\|\mathbf{x} - \boldsymbol{\mu}\|_2$ produces severe false alarms and missed anomalies because it treats all directions as equally probable.

    Below, we evaluate four probe points against $\mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with $\boldsymbol{\mu} = [2.0, 3.0]^T$ and $\boldsymbol{\Sigma} = \begin{bmatrix} 3.0 & 1.8 \\ 1.8 & 2.0 \end{bmatrix}$:
    * **Probe A**: A point lying along the principal correlation axis.
    * **Probe B**: A point lying strictly orthogonal to the correlation axis with a smaller Euclidean distance.
    * **Probe C**: A severe outlier far off the axis.
    * **Probe D**: A point near the center.

    Using the Chi-Square distribution $D_M^2 \sim \chi^2(2)$, we compute the exact $p$-value for each point and flag anomalies at $\alpha = 0.01$ ($D_M^2 > 9.21$).
    """)
    return


@app.cell
def _(np, stats):
    mu_ref = np.array([2.0, 3.0])
    sigma_ref = np.array([[3.0, 1.8], [1.8, 2.0]])
    sigma_inv_ref = np.linalg.inv(sigma_ref)

    probe_points_dict = {
        "Probe A (Along Correlation Axis)": np.array([5.5, 5.8]),
        "Probe B (Orthogonal to Correlation)": np.array([0.0, 5.0]),
        "Probe C (Severe Outlier)": np.array([-1.5, 7.5]),
        "Probe D (Near Mean Mode)": np.array([2.2, 3.1]),
    }

    anomaly_evaluation_records = []

    for label, probe_coords in probe_points_dict.items():
        diff_vec = probe_coords - mu_ref
        euclidean_dist = float(np.linalg.norm(diff_vec))
        mahalanobis_dist_sq = float(diff_vec.T @ sigma_inv_ref @ diff_vec)
        mahalanobis_dist = float(np.sqrt(mahalanobis_dist_sq))

        # p-value under Chi^2(df=2)
        chi2_pval = float(1.0 - stats.chi2.cdf(mahalanobis_dist_sq, df=2))
        is_anomaly = chi2_pval < 0.01

        anomaly_evaluation_records.append(
            {
                "Candidate Point": label,
                "Coordinates [x₁, x₂]": f"[{probe_coords[0]:.1f}, {probe_coords[1]:.1f}]",
                "Euclidean Distance": f"{euclidean_dist:.2f}",
                "Mahalanobis Distance D_M": f"{mahalanobis_dist:.2f}",
                "Chi-Square p-value": f"{chi2_pval:.4f}",
                "Detection (α=0.01)": "ANOMALY DETECTED" if is_anomaly else "NORMAL IN-BOUNDS",
            }
        )

    return (
        anomaly_evaluation_records,
        chi2_pval,
        diff_vec,
        euclidean_dist,
        is_anomaly,
        label,
        mahalanobis_dist,
        mahalanobis_dist_sq,
        mu_ref,
        probe_coords,
        probe_points_dict,
        sigma_inv_ref,
        sigma_ref,
    )


@app.cell(hide_code=True)
def _(anomaly_evaluation_records, mo, pd):
    df_anomaly = pd.DataFrame(anomaly_evaluation_records)
    mo.ui.table(df_anomaly)
    return (df_anomaly,)


if __name__ == "__main__":
    app.run()
