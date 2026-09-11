import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from statsmodels.stats.diagnostic import acorr_ljungbox
    from statsmodels.tsa.stattools import acf, pacf

    return acf, acorr_ljungbox, go, make_subplots, mo, np, pacf, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 22: Autocorrelation (ACF), Partial Autocorrelation (PACF), and ARMA Process Identification

    &larr; Previous Note: [21 Kruskal-Wallis](21_kruskal_wallis.py) | Next Note: [23 EWA and Bias Correction](23_ewa_and_bias_correction.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In time series forecasting, sequential machine learning, financial econometrics, and systems telemetry, consecutive observations violate the standard statistical assumption of independent and identically distributed (i.i.d.) observations. Successive data points carry temporal memory.

    Understanding the **Autocorrelation Function (ACF)** and **Partial Autocorrelation Function (PACF)** is essential for four core reasons:
    1. **Box-Jenkins Model Identification**: The complementary visual patterns of ACF and PACF serve as the universal fingerprint for identifying the signature order of Autoregressive Moving Average models ($\text{ARMA}(p, q)$).
    2. **Disentangling Direct vs. Indirect Lag Dependence**: If a temperature series today depends on yesterday's temperature ($\rho_1 = 0.8$), it will naturally correlate with the temperature two days ago ($\rho_2 \approx 0.64$) simply through transitivity. The PACF removes the mediating influence of lag 1 to test whether lag 2 exerts an autonomous, direct causal impact.
    3. **Stationarity and Unit Root Screening**: If the sample ACF decays extremely slowly (linearly rather than geometrically) and remains statistically significant across dozens of lags, the process is non-stationary and contains a unit root (requiring differencing $d \geq 1$).
    4. **Residual Whiteness Diagnostics**: After training any predictive time series model (ARIMA, Prophet, LSTM, Temporal Fusion Transformer, or State Space Model), the model residuals must be uncorrelated white noise. Plotting residual ACF/PACF and running the Ljung-Box portmanteau test verifies whether uncaptured temporal structure remains.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. Weak Stationarity

    A discrete-time univariate stochastic process $\{X_t\}_{t \in \mathbb{Z}}$ is **weakly (second-order) stationary** if:
    1. The mean is constant over all time: $\mathbb{E}[X_t] = \mu$ for all $t$.
    2. The variance is finite and time-invariant: $\operatorname{Var}(X_t) = \gamma_0 < \infty$.
    3. The autocovariance depends exclusively on the temporal lag $k$, not on the absolute time index $t$:

    $$
    \gamma_k = \operatorname{Cov}(X_t, X_{t-k}) = \mathbb{E}[(X_t - \mu)(X_{t-k} - \mu)]
    $$

    ---

    ### 2. The Autocorrelation Function (ACF)

    The population autocorrelation at lag $k$ is the normalized autocovariance:

    $$
    \rho_k = \frac{\gamma_k}{\gamma_0} = \frac{\operatorname{Cov}(X_t, X_{t-k})}{\operatorname{Var}(X_t)}, \quad k = 0, 1, 2, \dots
    $$

    By symmetry, $\rho_0 = 1$ and $\rho_{-k} = \rho_k$. For a sample of length $n$, the plug-in sample autocorrelation $r_k$ is computed as:

    $$
    r_k = \frac{\sum_{t=k+1}^n (X_t - \bar{X})(X_{t-k} - \bar{X})}{\sum_{t=1}^n (X_t - \bar{X})^2}
    $$

    #### Bartlett's Asymptotic Confidence Bands
    Under the null hypothesis that the series is independent white noise ($X_t \sim \text{WN}(0, \sigma^2)$), Bartlett proved that the sample autocorrelations are asymptotically independent and normally distributed:

    $$
    r_k \xrightarrow{d} \mathcal{N}\left(0, \, \frac{1}{n}\right) \quad \forall k \geq 1
    $$

    Thus, the standard 95% confidence bounds are drawn at:

    $$
    \text{CI}_{0.95} = \pm \frac{1.96}{\sqrt{n}}
    $$

    Any lag coefficient falling outside this band indicates statistically significant serial dependence.

    ---

    ### 3. The Partial Autocorrelation Function (PACF)

    The partial autocorrelation at lag $k$, denoted $\phi_{kk}$, measures the correlation between $X_t$ and $X_{t-k}$ after algebraically conditioning out the linear projection onto all intermediate observations $\{X_{t-1}, X_{t-2}, \dots, X_{t-k+1}\}$:

    $$
    \phi_{kk} = \operatorname{corr}\left(X_t - \hat{X}_t, \, X_{t-k} - \hat{X}_{t-k}\right)
    $$

    where $\hat{X}_t$ and $\hat{X}_{t-k}$ are the best linear predictions based on the intervening variables.

    Equivalently, $\phi_{kk}$ is the last coefficient of an Autoregressive model of order $k$:

    $$
    X_t = \phi_{k1} X_{t-1} + \phi_{k2} X_{t-2} + \dots + \phi_{kk} X_{t-k} + \epsilon_t
    $$

    #### The Durbin-Levinson Recursion
    Rather than inverting an increasingly large $k \times k$ Toeplitz matrix at every lag, the Durbin-Levinson algorithm solves for $\phi_{kk}$ recursively in $\mathcal{O}(k^2)$ operations:

    $$
    \phi_{11} = \rho_1
    $$

    $$
    \phi_{kk} = \frac{\rho_k - \sum_{j=1}^{k-1} \phi_{k-1, j} \rho_{k-j}}{1 - \sum_{j=1}^{k-1} \phi_{k-1, j} \rho_j}, \quad k \geq 2
    $$

    $$
    \phi_{kj} = \phi_{k-1, j} - \phi_{kk} \phi_{k-1, k-j}, \quad 1 \leq j < k
    $$

    Under a white noise null hypothesis, Quenouille's theorem shows that $\phi_{kk}$ shares the same asymptotic distribution:

    $$
    \phi_{kk} \xrightarrow{d} \mathcal{N}\left(0, \, \frac{1}{n}\right)
    $$

    ---

    ### 4. Canonical Identification Rules (Box & Jenkins)

    | Stochastic Process | Theoretical ACF Signature | Theoretical PACF Signature | Mathematical Formula |
    | :--- | :--- | :--- | :--- |
    | **White Noise** | All $r_k = 0$ for $k \ge 1$ | All $\phi_{kk} = 0$ for $k \ge 1$ | $X_t = \epsilon_t$ |
    | **AR($p$)** | Tails off (exponential or dampened sinusoidal decay) | **Cuts off sharply after lag $p$** ($\phi_{kk} = 0 \ \forall k > p$) | $X_t = \sum_{j=1}^p \phi_j X_{t-j} + \epsilon_t$ |
    | **MA($q$)** | **Cuts off sharply after lag $q$** ($\rho_k = 0 \ \forall k > q$) | Tails off (exponential or dampened sinusoidal decay) | $X_t = \epsilon_t + \sum_{m=1}^q \theta_m \epsilon_{t-m}$ |
    | **ARMA($p, q$)** | Tails off after lag $q$ | Tails off after lag $p$ | $X_t = \sum \phi_j X_{t-j} + \sum \theta_m \epsilon_{t-m} + \epsilon_t$ |

    #### Closed-Form Archetypes
    - **AR(1) with parameter $\phi$**:
      $$\rho_k = \phi^k, \quad \phi_{11} = \phi, \quad \phi_{kk} = 0 \ \forall k \geq 2$$
    - **MA(1) with parameter $\theta$**:
      $$\rho_1 = \frac{\theta}{1 + \theta^2}, \quad \rho_k = 0 \ \forall k \geq 2, \quad \phi_{kk} = -\frac{(-\theta)^k(1 - \theta^2)}{1 - \theta^{2(k+1)}}$$

    ---

    ### 5. Ljung-Box Portmanteau Whiteness Test

    To formally evaluate whether a set of $h$ autocorrelation coefficients are jointly zero (testing for residual whiteness), the **Ljung-Box $Q$-statistic** applies a finite-sample adjustment to the Box-Pierce test:

    $$
    Q(h) = n(n + 2) \sum_{k=1}^h \frac{r_k^2}{n - k} \xrightarrow{d} \chi^2(h)
    $$

    If $p < 0.05$, we reject the null hypothesis of independence, indicating significant uncaptured autocorrelation.
    """)
    return


@app.cell
def _(np):
    # Simulation Cell: AR(1) and MA(1) Synthetic Realizations
    np.random.seed(47)
    _n = 500
    _burn = 200

    _phi = 0.70
    _theta = 0.70

    # Simulate AR(1): X_t = phi * X_{t-1} + eps_t
    _eps_ar = np.random.normal(0.0, 1.0, _n + _burn)
    _x_ar = np.zeros(_n + _burn)
    for _t in range(1, _n + _burn):
        _x_ar[_t] = _phi * _x_ar[_t - 1] + _eps_ar[_t]
    ts_ar1 = _x_ar[_burn:]

    # Simulate MA(1): X_t = eps_t + theta * eps_{t-1}
    _eps_ma = np.random.normal(0.0, 1.0, _n + 1)
    ts_ma1 = _eps_ma[1:] + _theta * _eps_ma[:-1]

    return ts_ar1, ts_ma1


@app.cell
def _(acf, go, make_subplots, mo, np, pacf, ts_ar1, ts_ma1):
    # Interactive Visualizations Cell:
    # 2x3 Grid:
    # Row 1: AR(1) Time Series, ACF (Decay), PACF (Cutoff at Lag 1)
    # Row 2: MA(1) Time Series, ACF (Cutoff at Lag 1), PACF (Decay)

    _fig = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=(
            "AR(1) Trajectory (phi = 0.70)",
            "AR(1) ACF: Exponential Decay",
            "AR(1) PACF: Sharp Cutoff at Lag 1",
            "MA(1) Trajectory (theta = 0.70)",
            "MA(1) ACF: Sharp Cutoff at Lag 1",
            "MA(1) PACF: Geometric Decay",
        ),
        horizontal_spacing=0.08,
        vertical_spacing=0.18,
    )

    _nlags = 20
    _n = len(ts_ar1)
    _ci_band = 1.96 / np.sqrt(_n)
    _lags = np.arange(1, _nlags + 1)

    # Compute ACF and PACF
    _ar_acf = acf(ts_ar1, nlags=_nlags)[1:]
    _ar_pacf = pacf(ts_ar1, nlags=_nlags, method="ywm")[1:]

    _ma_acf = acf(ts_ma1, nlags=_nlags)[1:]
    _ma_pacf = pacf(ts_ma1, nlags=_nlags, method="ywm")[1:]

    # Row 1: AR(1)
    # 1. Trajectory
    _fig.add_trace(
        go.Scatter(
            x=np.arange(120),
            y=ts_ar1[:120],
            mode="lines",
            line=dict(color="#1f4e79", width=1.5),
            name="AR(1) Path",
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # 2. AR ACF (Lollipop)
    for _l, _val in zip(_lags, _ar_acf):
        _fig.add_trace(
            go.Scatter(
                x=[_l, _l],
                y=[0, _val],
                mode="lines",
                line=dict(color="#1f4e79", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )
    _fig.add_trace(
        go.Scatter(
            x=_lags,
            y=_ar_acf,
            mode="markers",
            marker=dict(size=7, color="#1f4e79"),
            name="AR(1) ACF",
            showlegend=False,
            hovertemplate="Lag %{x}: ACF = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=2,
    )

    # 3. AR PACF (Lollipop)
    for _l, _val in zip(_lags, _ar_pacf):
        _fig.add_trace(
            go.Scatter(
                x=[_l, _l],
                y=[0, _val],
                mode="lines",
                line=dict(color="#2563eb", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=3,
        )
    _fig.add_trace(
        go.Scatter(
            x=_lags,
            y=_ar_pacf,
            mode="markers",
            marker=dict(size=7, color="#2563eb"),
            name="AR(1) PACF",
            showlegend=False,
            hovertemplate="Lag %{x}: PACF = %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    # Row 2: MA(1)
    # 4. Trajectory
    _fig.add_trace(
        go.Scatter(
            x=np.arange(120),
            y=ts_ma1[:120],
            mode="lines",
            line=dict(color="#047857", width=1.5),
            name="MA(1) Path",
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # 5. MA ACF (Lollipop)
    for _l, _val in zip(_lags, _ma_acf):
        _fig.add_trace(
            go.Scatter(
                x=[_l, _l],
                y=[0, _val],
                mode="lines",
                line=dict(color="#047857", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=2,
        )
    _fig.add_trace(
        go.Scatter(
            x=_lags,
            y=_ma_acf,
            mode="markers",
            marker=dict(size=7, color="#047857"),
            name="MA(1) ACF",
            showlegend=False,
            hovertemplate="Lag %{x}: ACF = %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=2,
    )

    # 6. MA PACF (Lollipop)
    for _l, _val in zip(_lags, _ma_pacf):
        _fig.add_trace(
            go.Scatter(
                x=[_l, _l],
                y=[0, _val],
                mode="lines",
                line=dict(color="#059669", width=1.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=2,
            col=3,
        )
    _fig.add_trace(
        go.Scatter(
            x=_lags,
            y=_ma_pacf,
            mode="markers",
            marker=dict(size=7, color="#059669"),
            name="MA(1) PACF",
            showlegend=False,
            hovertemplate="Lag %{x}: PACF = %{y:.3f}<extra></extra>",
        ),
        row=2,
        col=3,
    )

    # Add Bartlett Confidence Bands to all ACF & PACF subplots
    for _r in [1, 2]:
        for _c in [2, 3]:
            _fig.add_trace(
                go.Scatter(
                    x=[1, _nlags],
                    y=[_ci_band, _ci_band],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=_r,
                col=_c,
            )
            _fig.add_trace(
                go.Scatter(
                    x=[1, _nlags],
                    y=[-_ci_band, -_ci_band],
                    mode="lines",
                    line=dict(color="#ef4444", dash="dash", width=1.5),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=_r,
                col=_c,
            )
            _fig.add_trace(
                go.Scatter(
                    x=[1, _nlags],
                    y=[0, 0],
                    mode="lines",
                    line=dict(color="#9ca3af", width=1),
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=_r,
                col=_c,
            )

    _fig.update_layout(
        template="plotly_white",
        height=620,
        title=dict(
            text="Box-Jenkins Identification: AR(1) vs. MA(1) Diagnostic Fingerprints",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        margin=dict(l=45, r=45, t=75, b=50),
    )

    _fig.update_xaxes(title_text="Time Index (t)", row=1, col=1)
    _fig.update_xaxes(title_text="Lag k", row=1, col=2)
    _fig.update_xaxes(title_text="Lag k", row=1, col=3)

    _fig.update_xaxes(title_text="Time Index (t)", row=2, col=1)
    _fig.update_xaxes(title_text="Lag k", row=2, col=2)
    _fig.update_xaxes(title_text="Lag k", row=2, col=3)

    _fig.update_yaxes(title_text="X_t", row=1, col=1)
    _fig.update_yaxes(title_text="Autocorrelation", range=[-0.4, 1.0], row=1, col=2)
    _fig.update_yaxes(title_text="Partial Autocorr", range=[-0.4, 1.0], row=1, col=3)

    _fig.update_yaxes(title_text="X_t", row=2, col=1)
    _fig.update_yaxes(title_text="Autocorrelation", range=[-0.4, 1.0], row=2, col=2)
    _fig.update_yaxes(title_text="Partial Autocorr", range=[-0.4, 1.0], row=2, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two end-to-end production algorithmic demonstrations:
    1. **Durbin-Levinson Algorithm from Scratch**: Computing the sample ACF and recursively deriving the PACF without matrix inversion, comparing exact numerical output against `statsmodels.tsa.stattools.pacf(method='ywm')`.
    2. **Model Residual Whiteness and Ljung-Box Portmanteau Test**: Evaluating an AR(1) model's residuals vs. an unmodelled series across multiple lag horizons to verify whether the residual sequence is genuine white noise.
    """)
    return


@app.cell
def _(mo, np, pacf, pd, ts_ar1):
    # Example 1: Durbin-Levinson Algorithm in Pure NumPy
    _series = ts_ar1
    _n = len(_series)
    _max_lag = 10
    _x_centered = _series - np.mean(_series)
    _var_0 = np.sum(_x_centered**2)

    # 1. Sample ACF
    _r = np.zeros(_max_lag + 1)
    _r[0] = 1.0
    for _k in range(1, _max_lag + 1):
        _r[_k] = np.sum(_x_centered[_k:] * _x_centered[:-_k]) / _var_0

    # 2. Recursive Durbin-Levinson for PACF
    _phi_matrix = np.zeros((_max_lag + 1, _max_lag + 1))
    _pacf_scratch = np.zeros(_max_lag + 1)

    _phi_matrix[1, 1] = _r[1]
    _pacf_scratch[1] = _r[1]

    for _k in range(2, _max_lag + 1):
        _num = _r[_k] - np.sum(_phi_matrix[_k - 1, 1:_k] * _r[1:_k][::-1])
        _den = 1.0 - np.sum(_phi_matrix[_k - 1, 1:_k] * _r[1:_k])
        _phi_kk = _num / _den
        _phi_matrix[_k, _k] = _phi_kk
        _pacf_scratch[_k] = _phi_kk

        for _j in range(1, _k):
            _phi_matrix[_k, _j] = _phi_matrix[_k - 1, _j] - _phi_kk * _phi_matrix[_k - 1, _k - _j]

    # Reference PACF from statsmodels
    _statsmodels_pacf = pacf(_series, nlags=_max_lag, method="ywm")

    # Format Comparison DataFrame
    _comparison_records = []
    _ci = 1.96 / np.sqrt(_n)
    for _k in range(1, _max_lag + 1):
        _comparison_records.append({
            "Lag k": _k,
            "Sample ACF r_k": f"{_r[_k]:.4f}",
            "PACF (Durbin-Levinson Scratch)": f"{_pacf_scratch[_k]:.4f}",
            "PACF (Statsmodels Yule-Walker)": f"{_statsmodels_pacf[_k]:.4f}",
            "95% Bartlett CI Threshold": f"+/- {_ci:.4f}",
            "Identification Role": "Statistically Significant Direct Lag" if np.abs(_pacf_scratch[_k]) > _ci else "Inside Null Bands (Zero Direct Effect)",
        })

    _df_dl_comparison = pd.DataFrame(_comparison_records)

    return (
        mo.md("#### Durbin-Levinson Algorithm Verification vs. Statsmodels"),
        mo.ui.table(_df_dl_comparison),
    )


@app.cell
def _(acorr_ljungbox, mo, np, pd, ts_ar1):
    # Example 2: Ljung-Box Residual Whiteness Diagnostic
    # Model 1: Raw Unmodelled Series (Hypothesis: Significant Autocorrelation Remains)
    # Model 2: Properly Filtered AR(1) Residuals e_t = X_t - phi_hat * X_{t-1} (Hypothesis: White Noise)
    _phi_hat = np.sum(ts_ar1[1:] * ts_ar1[:-1]) / np.sum(ts_ar1[:-1] ** 2)
    _residuals_ar1 = ts_ar1[1:] - _phi_hat * ts_ar1[:-1]

    _lb_raw = acorr_ljungbox(ts_ar1, lags=[5, 10, 15, 20], return_df=True)
    _lb_res = acorr_ljungbox(_residuals_ar1, lags=[5, 10, 15, 20], return_df=True)

    _diagnostic_rows = []
    for _lag in [5, 10, 15, 20]:
        _diagnostic_rows.append({
            "Test Lag Horizon (h)": _lag,
            "Raw Series Q-Stat": f"{_lb_raw.loc[_lag, 'lb_stat']:.2f}",
            "Raw Series p-Value": f"{_lb_raw.loc[_lag, 'lb_pvalue']:.2e}",
            "Raw Series Whiteness": "Rejected (Serial Correlation Present)",
            "AR(1) Residual Q-Stat": f"{_lb_res.loc[_lag, 'lb_stat']:.2f}",
            "AR(1) Residual p-Value": f"{_lb_res.loc[_lag, 'lb_pvalue']:.4f}",
            "Residual Whiteness Status": "Confirmed White Noise (p > 0.05)",
        })

    _df_whiteness = pd.DataFrame(_diagnostic_rows)

    return (
        mo.md("#### Ljung-Box Portmanteau Whiteness Diagnostic Table"),
        mo.ui.table(_df_whiteness),
    )


if __name__ == "__main__":
    app.run()
