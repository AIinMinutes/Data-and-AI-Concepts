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
    # Autocorrelation and Partial Autocorrelation

    A regularly sampled **univariate** series $\{X_t\}$ is one scalar observed over time for a single unit. The autocorrelation function (ACF) and partial autocorrelation function (PACF) describe how that series depends on its own past, and they are the classical tools for identifying ARMA structure (Box and Jenkins).

    Assume throughout that $\{X_t\}$ is weakly stationary: constant mean, constant variance, and $\mathrm{Cov}(X_t, X_{t-k})$ depending only on the lag $k$.

    ## Autocorrelation function (ACF)

    The population ACF at lag $k$ is the correlation of the series with itself $k$ steps earlier:

    $$
    \rho_k = \frac{\mathrm{Cov}(X_t, X_{t-k})}{\mathrm{Var}(X_t)}, \qquad k = 0, 1, 2, \ldots
    $$

    with $\rho_0 = 1$. The sample ACF from $n$ observations is

    $$
    r_k = \frac{\sum_{t=k+1}^{n}(X_t - \bar{X})(X_{t-k} - \bar{X})}{\sum_{t=1}^{n}(X_t - \bar{X})^2}.
    $$

    Under white noise, $r_k$ is approximately $\mathcal{N}(0, 1/n)$, so the usual 95% bands are $\pm 1.96 / \sqrt{n}$.

    ## Partial autocorrelation function (PACF)

    The PACF at lag $k$, written $\phi_{kk}$, is the correlation between $X_t$ and $X_{t-k}$ **after removing** the linear effect of the intermediate lags $X_{t-1},\ldots,X_{t-k+1}$. Equivalently, it is the last coefficient in the AR($k$) projection

    $$
    X_t = \phi_{k1} X_{t-1} + \cdots + \phi_{kk} X_{t-k} + \varepsilon_t.
    $$

    ## Identification

    | Process | ACF | PACF |
    |---------|-----|------|
    | AR($p$), $\|\phi\|<1$ | Tails off (geometric decay) | Cuts off after lag $p$ |
    | MA($q$) | Cuts off after lag $q$ | Tails off |
    | ARMA($p,q$) | Tails off | Tails off |

    Closed forms used below:

    - AR(1), $X_t = \phi X_{t-1} + \varepsilon_t$: $\rho_k = \phi^k$, and $\phi_{11} = \phi$, $\phi_{kk} = 0$ for $k \ge 2$.
    - MA(1), $X_t = \varepsilon_t + \theta \varepsilon_{t-1}$: $\rho_1 = \theta / (1+\theta^2)$, $\rho_k = 0$ for $k \ge 2$.

    The experiment generates one AR(1) and one MA(1) series ($n=400$, $\phi=\theta=0.6$) and plots the series, ACF, and PACF for each.
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

    np.random.seed(47)
    plt.rcParams.update({
        "figure.dpi": 120,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


    def simulate_ar1(n, phi, sigma=1.0, burn_in=200):
        """X_t = phi X_{t-1} + eps_t. Discard burn-in so the series is near stationarity."""
        eps = np.random.normal(0, sigma, n + burn_in)
        x = np.zeros(n + burn_in)
        for t in range(1, n + burn_in):
            x[t] = phi * x[t - 1] + eps[t]
        return x[burn_in:]


    def simulate_ma1(n, theta, sigma=1.0):
        """X_t = eps_t + theta eps_{t-1}."""
        eps = np.random.normal(0, sigma, n + 1)
        return eps[1:] + theta * eps[:-1]


    n = 400
    phi = 0.6
    theta = 0.6
    ar = simulate_ar1(n, phi)
    ma = simulate_ma1(n, theta)

    rho1_ma = theta / (1 + theta**2)
    print(f"AR(1): theoretical PACF at lag 1 = {phi:.2f}; ACF(k) = {phi}^k")
    print(f"MA(1): theoretical ACF at lag 1 = {rho1_ma:.3f}; ACF(k) = 0 for k >= 2")
    return ar, ma, plot_acf, plot_pacf, plt


@app.cell
def _(ar, ma, plot_acf, plot_pacf, plt):
    def plot_series_acf_pacf(series, name, n_lags=20):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
        axes[0].plot(series, color="#1f4e79", lw=0.9)
        axes[0].set_title(f"{name} series")
        axes[0].set_xlabel("t")
        axes[0].set_ylabel("$X_t$")

        plot_acf(series, lags=n_lags, ax=axes[1], alpha=0.05, zero=False)
        axes[1].set_title(f"ACF of {name}")
        axes[1].set_xlabel("lag")

        plot_pacf(series, lags=n_lags, ax=axes[2], alpha=0.05, zero=False, method="ywm")
        axes[2].set_title(f"PACF of {name}")
        axes[2].set_xlabel("lag")

        fig.tight_layout()
        return fig


    plot_series_acf_pacf(ar, "AR(1), $\\phi = 0.6$")
    plot_series_acf_pacf(ma, "MA(1), $\\theta = 0.6$")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Read the figures against the table:

    - **AR(1).** The ACF decays geometrically ($\rho_k = 0.6^k$). The PACF is large at lag 1 and statistically zero afterwards — a cutoff at $p = 1$.
    - **MA(1).** The ACF is nonzero only at lag 1 (theory: $\rho_1 \approx 0.441$) and then sits inside the bands. The PACF decays rather than cutting off.

    That pairing — which plot cuts off, which tails off — is the identification rule. Both plots are required; showing only the ACF of MA and only the PACF of AR hides the complementary pattern.

    These series are regularly spaced. If observation times were irregular, $r_k$ is no longer well-defined at integer lags without first choosing a clock (binning, interpolation, or a continuous-time model).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
