import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import FormatStrFormatter
    from scipy.stats import multivariate_normal

    plt.rc("axes", titlesize=24, labelsize=20, labelpad=5)
    plt.rc("xtick", labelsize=18)
    plt.rc("ytick", labelsize=18)
    plt.rc("legend", fontsize=12)

    # Generate synthetic data from a bivariate normal distribution
    np.random.seed(47)
    plt.style.use("dark_background")
    mu_true = np.array([2, 3])  # True mean vector
    sigma_true = np.array([[1, 0.6], [0.6, 2]])  # True covariance matrix
    n_samples = 1000
    data = np.random.multivariate_normal(mu_true, sigma_true, size=n_samples)

    # MLE estimate of mean and covariance
    mu_hat = np.mean(data, axis=0)
    sigma_hat = (data - mu_hat).T @ (data - mu_hat) / data.shape[0]

    # Create meshgrid for contour plotting
    x = np.linspace(mu_hat[0] - 3 * np.sqrt(sigma_hat[0, 0]), mu_hat[0] + 3 * np.sqrt(sigma_hat[0, 0]), 100)
    y = np.linspace(mu_hat[1] - 3 * np.sqrt(sigma_hat[1, 1]), mu_hat[1] + 3 * np.sqrt(sigma_hat[1, 1]), 100)
    X, Y = np.meshgrid(x, y)

    # Calculate the PDF for true and estimated bivariate normal distributions
    rv_true = multivariate_normal(mu_true, sigma_true)
    rv_est = multivariate_normal(mu_hat, sigma_hat)
    Z_true = rv_true.pdf(np.dstack((X, Y)))
    Z_est = rv_est.pdf(np.dstack((X, Y)))

    # Plot 2D contour plots
    plt.figure(figsize=(12, 6), dpi=300)

    # True distribution
    plt.subplot(1, 2, 1)
    cp_true = plt.contour(X, Y, Z_true, 5, cmap="inferno", linewidths=5)
    plt.clabel(cp_true, inline=True, fontsize=8)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.4, label="Data Points", color="red", s=10)
    plt.scatter(mu_true[0], mu_true[1], color="blue", label="True Mean", marker="+", s=120)
    plt.title("True Bivariate \n Normal Distribution")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    # Estimated distribution
    plt.subplot(1, 2, 2)
    cp_est = plt.contour(X, Y, Z_est, 5, cmap="viridis", linewidths=5)
    plt.clabel(cp_est, inline=True, fontsize=8)
    plt.scatter(data[:, 0], data[:, 1], alpha=0.4, label="Data Points", color="red", s=10)
    plt.scatter(mu_hat[0], mu_hat[1], color="green", label="Estimated Mean", marker="x", s=120)
    plt.title("Estimated Bivariate \n Normal Distribution")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Plot 3D surface plots
    fig = plt.figure(figsize=(15, 6), dpi=300)

    # True distribution 3D plot
    ax = fig.add_subplot(121, projection="3d")
    ax.view_init(elev=30, azim=100)
    ax.plot_surface(X, Y, Z_true, cmap="inferno", alpha=0.9)
    ax.scatter(mu_true[0], mu_true[1], np.max(Z_true), color="blue", label="True Mean", marker="+", s=150)
    ax.set_title("True Bivariate \n Normal Distribution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability Density", labelpad=15)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.legend()

    # Estimated distribution 3D plot
    ax = fig.add_subplot(122, projection="3d")
    ax.view_init(elev=30, azim=100)
    ax.plot_surface(X, Y, Z_est, cmap="coolwarm", alpha=0.9)
    ax.scatter(mu_hat[0], mu_hat[1], np.max(Z_est), color="green", label="Estimated Mean", marker="x", s=150)
    ax.set_title("Estimated Bivariate \n Normal Distribution")
    ax.set_xlabel("X1")
    ax.set_ylabel("X2")
    ax.set_zlabel("Probability Density", labelpad=15)
    ax.zaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    ax.zaxis.label.set_position((1.0, 0.5, 0))
    ax.legend()

    plt.subplots_adjust(wspace=0, hspace=0.5)
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Multivariate Normal Distribution** probability distribution that generalizes the univariate normal distribution to multiple dimensions. A random vector $ \mathbf{X} \in \mathbb{R}^d $ is said to follow a multivariate normal distribution if its probability density function (PDF) is:
    $$
    f(\mathbf{x}) = \frac{1}{(2\pi)^{d/2} |\boldsymbol{\Sigma}|^{1/2}} \exp\left( -\frac{1}{2} (\mathbf{x} - \boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1} (\mathbf{x} - \boldsymbol{\mu}) \right),
    $$
    where:
    - $ \boldsymbol{\mu} \in \mathbb{R}^d $ is the mean vector,
    - $ \boldsymbol{\Sigma} \in \mathbb{R}^{d \times d} $ is the covariance matrix (symmetric and positive semi-definite).

    ### Multivariate Central Limit Theorem
    The multivariate central limit theorem states that the sum (or average) of independent random vectors with finite mean and covariance converges in distribution to a multivariate normal distribution as the sample size $ N \to \infty $. Specifically, for $ \{\mathbf{X}_i\}_{i=1}^N $ i.i.d. random vectors with mean $ \boldsymbol{\mu} $ and covariance $ \boldsymbol{\Sigma} $:
    $$
    \sqrt{N} \left( \bar{\mathbf{X}} - \boldsymbol{\mu} \right) \xrightarrow{d} \mathcal{N}(\mathbf{0}, \boldsymbol{\Sigma}),
    $$
    where $ \bar{\mathbf{X}} = \frac{1}{N} \sum_{i=1}^N \mathbf{X}_i $.

    ### MLE Estimate of Mean Vector
    The MLE for the mean vector $ \boldsymbol{\mu} $ is given by:
    $$
    \hat{\boldsymbol{\mu}} = \frac{1}{N} \sum_{i=1}^N \mathbf{x}_i,
    $$
    which is the sample mean.

    ### MLE Estimate of Covariance Matrix
    The MLE for the covariance matrix $ \boldsymbol{\Sigma} $ is given by:
    $$
    \hat{\boldsymbol{\Sigma}} = \frac{1}{N} \sum_{i=1}^N (\mathbf{x}_i - \hat{\boldsymbol{\mu}})(\mathbf{x}_i - \hat{\boldsymbol{\mu}})^\top,
    $$
    which is the sample covariance matrix.
    """)
    return


if __name__ == "__main__":
    app.run()
