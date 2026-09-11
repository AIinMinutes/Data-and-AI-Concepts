import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    **Hotelling's T² as a Multivariate Analogue of the t-Test**

    Hotelling's T² generalizes the univariate t-test to multivariate cases, considering covariance between variables.

    **Univariate t-Test**
    The univariate t-statistic:

    $$
    t = \frac{\bar{x} - \mu_0}{s / \sqrt{n}}
    $$

    **Multivariate Hotelling’s T²**
    The Hotelling’s T² statistic for testing $H_0: \mu = \mu_0$:

    $$
    T^2 = n (\bar{x} - \mu_0)^T \mathbf{S}^{-1} (\bar{x} - \mu_0)
    $$

    Where:
    - $\bar{x}$ is the sample mean vector,
    - $\mu_0$ is the hypothesized population mean vector,
    - $\mathbf{S}$ is the sample covariance matrix,
    - $n$ is the sample size.

    ---

    **One-Sample Test**

    **Test Statistic**
    For a one-sample test:

    $$
    T^2 = n (\bar{x} - \mu_0)^T \mathbf{S}^{-1} (\bar{x} - \mu_0)
    $$

    **F-Transformation**
    $$
    F = \frac{(n - 1) p}{(n - p) T^2}
    $$

    Where $F$ follows an $F(p, n - p)$-distribution.

    **Decision Rule**
    Reject $H_0$ if:

    $$
    T^2 > F_{\alpha, p, n-p}
    $$

    ---

    **Two-Sample Test**

    **Test Statistic**
    For two independent samples:

    $$
    T^2 = \frac{n_1 n_2}{n_1 + n_2} (\bar{x}_1 - \bar{x}_2)^T \mathbf{S}_p^{-1} (\bar{x}_1 - \bar{x}_2)
    $$

    Where:
    - $\mathbf{S}_p = \frac{(n_1 - 1)\mathbf{S}_1 + (n_2 - 1)\mathbf{S}_2}{n_1 + n_2 - 2}$ is the pooled covariance matrix.

    **F-Transformation**
    $$
    F = \frac{(n_1 + n_2 - p - 1)}{p(n_1 + n_2 - 2)} T^2
    $$

    Where $F$ follows an $F(p, n_1 + n_2 - p - 2)$-distribution.

    **Decision Rule**
    Reject $H_0$ if:

    $$
    F > F_{\alpha, p, n_1 + n_2 - p - 2}
    $$

    ---

    **Assumptions**

    **Multivariate Normality**
    For each sample:

    $$
    X_i \sim N(\mu, \Sigma)
    $$

    Where $X_i \in \mathbb{R}^p$, $\mu \in \mathbb{R}^p$, and $\Sigma \in \mathbb{R}^{p \times p}$.

    **Homogeneity of Covariance (*Population Level*)**
    For two-sample test:

    $$
    \Sigma_1 = \Sigma_2
    $$

    This is the assumption at the *population level*, meaning the covariance structures of the two populations are identical.

    **Independence**
    Samples from group 1 and group 2 (or observations within a single sample) must be *independent*.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import f

    np.random.seed(20250102)
    plt.style.use('dark_background')

    # Equal covariance structure
    cov_A = np.array([[20, 15], [15, 20]])
    cov_B = np.array([[20, 15], [15, 20]])

    mean_A = np.array([162, 75])
    mean_B = np.array([180, 85])

    data_A = np.random.multivariate_normal(
        mean_A, cov_A, size=20)
    data_B = np.random.multivariate_normal(
        mean_B, cov_B, size=25)

    # Sample means
    mean_samp_A = np.mean(data_A, axis=0)
    mean_samp_B = np.mean(data_B, axis=0)

    n_A = data_A.shape[0]
    n_B = data_B.shape[0]

    # Sample covariance matrices
    cov_samp_A = (
        (data_A - mean_samp_A).T @ 
        (data_A - mean_samp_A) / 
        (n_A - 1)
    )
    cov_samp_B = (
        (data_B - mean_samp_B).T @ 
        (data_B - mean_samp_B) / 
        (n_B - 1)
    )

    # Pooled covariance matrix
    cov_pooled = (
        (n_A - 1) * cov_samp_A + 
        (n_B - 1) * cov_samp_B) / (n_A + n_B - 2)


    # T-squared statistic
    mean_diff = mean_samp_A - mean_samp_B
    T_sq = (
        (n_A * n_B) / (n_A + n_B) * 
        mean_diff.T @ np.linalg.inv(cov_pooled) @ 
        mean_diff
    )

    # F-statistic and p-value
    p = mean_samp_A.shape[0]
    df1 = p
    df2 = n_A + n_B - p - 2
    F_stat = T_sq / df2

    p_val = 1 - f.cdf(F_stat, df1, df2)

    # Results
    print(f"Hotelling's T-squared statistic: {T_sq}")
    print(f"F-statistic: {F_stat}")
    print(f"p-value: {p_val}")

    # Hypothesis decision
    if p_val < 0.05:
        print("Reject the null hypothesis: Significant difference between the mean vector groups.")
    else:
        print("Fail to reject the null hypothesis: No significant difference between the groups.")
    return


if __name__ == "__main__":
    app.run()
