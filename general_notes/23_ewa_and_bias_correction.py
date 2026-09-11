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
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    return ExponentialSmoothing, go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 23: Exponentially Weighted Averages, Bias Correction, and the Adam Optimizer Engine

    &larr; Previous Note: [22 ACF and PACF](22_acf_and_pacf.py) | Next Note: [24 Adjusted R-Squared](24_adjusted_r_squared.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    In machine learning, deep neural network optimization, streaming data pipelines, and quantitative forecasting, tracking running estimates of means and variances is ubiquitous. While an ordinary moving average over window size $W$ requires buffering $W$ historical observations in memory, the **Exponentially Weighted Average (EWA)** (also known as Exponential Moving Average, EMA) tracks central tendency using a single scalar state variable in $\mathcal{O}(1)$ space and time.

    However, practical implementations face key challenges that make mathematical mastery essential:
    1. **The Cold-Start Zero-Initialization Bias**: When initializing an EWA at $v_0 = 0$, early iterations ($t = 1, 2, \dots$) suffer from severe downward bias because the recursive formula multiplies past states by the decay factor $\beta$. If $\beta = 0.98$, the first update only incorporates $2\%$ of the first observation ($v_1 = 0.02 \theta_1$), dragging early values toward zero.
    2. **Universal Bias Correction**: Dividing by the scaling factor $1 - \beta^t$ removes this cold-start artifact entirely, producing an unbiased estimator that automatically transitions to standard EWA as $t \to \infty$ ($\beta^t \to 0$).
    3. **Core Engine of Modern Optimizers (Adam, AdamW, RMSProp)**: The Adam optimizer relies fundamentally on bias-corrected first moments ($\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$) and second moments ($\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$). Without bias correction, initial gradient steps are severely suppressed, impeding pre-training convergence in Large Language Models and Vision Transformers.
    4. **Holt-Winters Multi-Component Forecasting**: In demand planning and latency forecasting, extending single exponential smoothing to Double (Holt's linear trend) and Triple (Holt-Winters seasonality) provides lightweight, highly accurate operational forecasts.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. The Recursive EWA Formulation

    Given a sequential stream of observations $\theta_1, \theta_2, \dots, \theta_t$, the exponentially weighted average $v_t$ with momentum parameter $\beta \in [0, 1)$ is defined by the recurrence:

    $$
    v_t = \beta v_{t-1} + (1 - \beta) \theta_t, \quad v_0 = 0
    $$

    Unrolling the recursion backwards to time $t = 0$:

    $$
    \begin{aligned}
    v_1 &= (1 - \beta) \theta_1 \\
    v_2 &= \beta v_1 + (1 - \beta) \theta_2 = \beta (1 - \beta) \theta_1 + (1 - \beta) \theta_2 \\
    v_t &= (1 - \beta) \sum_{i=1}^t \beta^{t - i} \theta_i
    \end{aligned}
    $$

    ---

    ### 2. Derivation of the Cold-Start Bias

    Suppose the observations $\theta_i$ are drawn from a distribution with constant mean $\mathbb{E}[\theta_i] = \mu$. Taking expectations of both sides:

    $$
    \mathbb{E}[v_t] = (1 - \beta) \sum_{i=1}^t \beta^{t - i} \mathbb{E}[\theta_i] = \mu (1 - \beta) \sum_{i=1}^t \beta^{t - i}
    $$

    The finite geometric series sum is:

    $$
    \sum_{i=1}^t \beta^{t - i} = \sum_{k=0}^{t-1} \beta^k = \frac{1 - \beta^t}{1 - \beta}
    $$

    Substituting this back into the expectation:

    $$
    \mathbb{E}[v_t] = \mu (1 - \beta) \left[\frac{1 - \beta^t}{1 - \beta}\right] = \mu (1 - \beta^t)
    $$

    Because $\beta \in (0, 1)$, the factor $(1 - \beta^t) < 1$. Thus:

    $$
    \mathbb{E}[v_t] \neq \mu \quad \text{for finite } t
    $$

    For example, when $\beta = 0.98$ and $t = 10$:

    $$
    1 - \beta^{10} = 1 - 0.98^{10} \approx 1 - 0.817 = 0.183
    $$

    The raw estimate $v_{10}$ captures only $18.3\%$ of the true expectation.

    ---

    ### 3. The Exact Bias Correction Formula

    To construct an estimator $\hat{v}_t$ whose expectation equals $\mu$ for all $t \geq 1$, we divide by the accumulated geometric weight:

    $$
    \hat{v}_t = \frac{v_t}{1 - \beta^t}
    $$

    Taking expectations confirms unbiasedness:

    $$
    \mathbb{E}[\hat{v}_t] = \frac{\mathbb{E}[v_t]}{1 - \beta^t} = \frac{\mu (1 - \beta^t)}{1 - \beta^t} = \mu \quad \forall t \geq 1
    $$

    As $t \to \infty$, $\beta^t \to 0$, so $1 - \beta^t \to 1$, meaning $\hat{v}_t \to v_t$. The correction automatically decays as historical data accumulates.

    ---

    ### 4. Effective Window Size

    Since $\lim_{\epsilon \to 0} (1 - \epsilon)^{1/\epsilon} = \frac{1}{e} \approx 0.368$, after $k = \frac{1}{1 - \beta}$ steps, the weight assigned to an observation decays to approximately $\frac{1}{e}$ of its initial contribution.

    Therefore, an EWA with parameter $\beta$ effectively aggregates over the preceding:

    $$
    W_{\text{eff}} \approx \frac{1}{1 - \beta} \quad \text{observations}
    $$

    - $\beta = 0.90 \implies W_{\text{eff}} \approx 10$ steps
    - $\beta = 0.98 \implies W_{\text{eff}} \approx 50$ steps
    - $\beta = 0.999 \implies W_{\text{eff}} \approx 1000$ steps (standard in Adam $\beta_2$)

    ---

    ### 5. Application in Deep Learning: The Adam Optimizer

    The Adam (Adaptive Moment Estimation) optimizer maintains exponentially decaying averages of past gradients ($m_t$, first moment) and squared gradients ($v_t$, second uncentered moment):

    $$
    g_t = \nabla_\theta \mathcal{L}_t(\theta_t)
    $$

    $$
    m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t, \quad \hat{m}_t = \frac{m_t}{1 - \beta_1^t}
    $$

    $$
    v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
    $$

    $$
    \theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
    $$

    Without bias correction, because $\beta_2 = 0.999$ initially, $v_1 = 0.001 g_1^2$, which would make $\sqrt{v_1}$ tiny and cause the initial parameter update step size to explode. Bias correction prevents this instability.

    ---

    ### 6. Holt-Winters Exponential Smoothing Hierarchy

    | Model | Components Captured | Recurrence Equations | Forecast $\hat{Y}_{t+h}$ |
    | :--- | :--- | :--- | :--- |
    | **Single (SES)** | Level ($L_t$) | $L_t = \alpha Y_t + (1 - \alpha) L_{t-1}$ | $\hat{Y}_{t+h} = L_t$ |
    | **Double (DES / Holt)** | Level + Trend ($T_t$) | $L_t = \alpha Y_t + (1 - \alpha)(L_{t-1} + T_{t-1})$<br>$T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$ | $\hat{Y}_{t+h} = L_t + h T_t$ |
    | **Triple (TES / Holt-Winters)** | Level + Trend + Seasonality ($S_t$) | $L_t = \alpha (Y_t - S_{t-m}) + (1 - \alpha)(L_{t-1} + T_{t-1})$<br>$T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}$<br>$S_t = \gamma (Y_t - L_t) + (1 - \gamma) S_{t-m}$ | $\hat{Y}_{t+h} = L_t + h T_t + S_{t-m+h}$ |
    """)
    return


@app.cell
def _(np, pd):
    # Simulation Data: Cold-start step function + Seasonal time series
    np.random.seed(47)
    _n = 80

    # Constant signal with additive Gaussian noise to demonstrate cold-start bias
    _true_mean = 50.0
    signal_coldstart = _true_mean + np.random.normal(0.0, 3.0, _n)

    # Seasonal series for Holt-Winters (Daily data with weekly period m=7, linear trend)
    _n_seasonal = 45
    _periods = 7
    _t = np.arange(_n_seasonal)
    _trend = 0.6 * _t + 20.0
    _season = 5.0 * np.sin(2 * np.pi * _t / _periods)
    _noise = np.random.normal(0.0, 1.2, _n_seasonal)
    _series_seasonal = _trend + _season + _noise

    dates_seasonal = pd.date_range("2025-01-01", periods=_n_seasonal, freq="D")
    df_seasonal = pd.DataFrame({"Date": dates_seasonal, "Value": np.round(_series_seasonal, 2)})

    return df_seasonal, signal_coldstart


@app.cell
def _(ExponentialSmoothing, df_seasonal, go, make_subplots, mo, np, pd, signal_coldstart):
    # Interactive Visualizations Cell:
    # Subplot 1: Cold-Start Bias: Raw EWA vs. Bias-Corrected EWA (beta=0.98)
    # Subplot 2: Exponential Memory Decay Kernels across beta values
    # Subplot 3: Holt-Winters (SES, DES, TES) Decomposition & Forecast

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Cold-Start Bias: Raw vs. Corrected (beta=0.98)",
            "2. Effective Memory Window Weight Kernels",
            "3. Holt-Winters Hierarchy & Seasonal Forecast",
        ),
        horizontal_spacing=0.08,
    )

    # Subplot 1: Cold Start EWA
    _beta = 0.98
    _n_steps = len(signal_coldstart)
    _raw_ewa = np.zeros(_n_steps)
    _corrected_ewa = np.zeros(_n_steps)

    _running_v = 0.0
    for _i, _obs in enumerate(signal_coldstart):
        _running_v = _beta * _running_v + (1.0 - _beta) * _obs
        _raw_ewa[_i] = _running_v
        _corrected_ewa[_i] = _running_v / (1.0 - _beta ** (_i + 1))

    _fig.add_trace(
        go.Scatter(
            x=np.arange(1, _n_steps + 1),
            y=signal_coldstart,
            mode="markers",
            marker=dict(size=4, color="#9ca3af"),
            name="Noisy Observations (True Mean=50)",
            hovertemplate="t=%{x}: Obs=%{y:.1f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=np.arange(1, _n_steps + 1),
            y=_raw_ewa,
            mode="lines",
            line=dict(color="#ef4444", width=2.5),
            name="Raw EWA (v_t, Zero-Biased)",
            hovertemplate="t=%{x}: Raw=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    _fig.add_trace(
        go.Scatter(
            x=np.arange(1, _n_steps + 1),
            y=_corrected_ewa,
            mode="lines",
            line=dict(color="#10b981", width=2.5),
            name="Bias-Corrected EWA (v_hat_t)",
            hovertemplate="t=%{x}: Corrected=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Subplot 2: Exponential Memory Kernels
    _lags = np.arange(0, 40)
    for _b, _col in [(0.80, "#3b82f6"), (0.90, "#f59e0b"), (0.98, "#8b5cf6")]:
        _weights = (1.0 - _b) * (_b**_lags)
        _fig.add_trace(
            go.Scatter(
                x=_lags,
                y=_weights,
                mode="lines+markers",
                marker=dict(size=5),
                line=dict(color=_col, width=2),
                name=f"beta = {_b:.2f} (W_eff ~ {int(1 / (1 - _b))})",
            ),
            row=1,
            col=2,
        )

    # Subplot 3: Holt-Winters Fits & Forecast
    _val_series = df_seasonal["Value"]
    _ses_mod = ExponentialSmoothing(_val_series, trend=None, seasonal=None).fit(smoothing_level=0.3)
    _des_mod = ExponentialSmoothing(_val_series, trend="add", seasonal=None).fit()
    _tes_mod = ExponentialSmoothing(_val_series, trend="add", seasonal="add", seasonal_periods=7).fit()

    _h = 7
    _future_dates = pd.date_range(df_seasonal["Date"].iloc[-1] + pd.Timedelta(days=1), periods=_h, freq="D")
    _tes_forecast = _tes_mod.forecast(_h)

    _fig.add_trace(
        go.Scatter(
            x=df_seasonal["Date"],
            y=_val_series,
            mode="markers+lines",
            marker=dict(size=4),
            line=dict(color="#6b7280", width=1),
            name="Seasonal Observed",
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Scatter(
            x=df_seasonal["Date"],
            y=_ses_mod.fittedvalues,
            mode="lines",
            line=dict(color="#f97316", width=1.5, dash="dot"),
            name="Single (SES)",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Scatter(
            x=df_seasonal["Date"],
            y=_des_mod.fittedvalues,
            mode="lines",
            line=dict(color="#3b82f6", width=1.5, dash="dash"),
            name="Double (DES / Holt)",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Scatter(
            x=df_seasonal["Date"],
            y=_tes_mod.fittedvalues,
            mode="lines",
            line=dict(color="#10b981", width=2),
            name="Triple (TES / Holt-Winters)",
        ),
        row=1,
        col=3,
    )
    _fig.add_trace(
        go.Scatter(
            x=_future_dates,
            y=_tes_forecast,
            mode="lines+markers",
            marker=dict(size=6, symbol="star"),
            line=dict(color="#047857", width=2.5, dash="dash"),
            name="TES 7-Day Forecast",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=480,
        title=dict(
            text="Exponential Moving Averages: Cold-Start Bias Correction, Decay Kernels, and Holt-Winters",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text="Iteration Step (t)", row=1, col=1)
    _fig.update_yaxes(title_text="Value", row=1, col=1)

    _fig.update_xaxes(title_text="Past Lag (k)", row=1, col=2)
    _fig.update_yaxes(title_text="Weight Contribution", row=1, col=2)

    _fig.update_xaxes(title_text="Date", row=1, col=3)
    _fig.update_yaxes(title_text="Observed / Fitted Value", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two end-to-end production algorithmic workflows:
    1. **Bias-Corrected EWA vs. Raw EWA Step-by-Step Diagnostic**: Step-by-step numerical verification of how the factor $1 - \beta^t$ eliminates the cold-start deficit across early iterations ($t = 1, \dots, 20$).
    2. **Adam Optimizer Engine Simulation**: A vectorized simulation comparing naive SGD, Adam without bias correction, and Adam with full bias correction on a noisy gradient optimization task.
    """)
    return


@app.cell
def _(mo, np, pd, signal_coldstart):
    # Example 1: Pure NumPy Step-by-Step Cold-Start Verification
    _beta = 0.98
    _steps = 15
    _true_mean = 50.0

    _records = []
    _v_raw = 0.0

    for _t in range(1, _steps + 1):
        _obs = signal_coldstart[_t - 1]
        _v_raw = _beta * _v_raw + (1.0 - _beta) * _obs
        _correction_factor = 1.0 - _beta**_t
        _v_hat = _v_raw / _correction_factor
        _pct_recovered = _correction_factor * 100.0

        _records.append({
            "Step t": _t,
            "Observation theta_t": f"{_obs:.2f}",
            "Raw EWA (v_t)": f"{_v_raw:.2f}",
            "Weight Sum (1 - beta^t)": f"{_correction_factor:.4f}",
            "Bias-Corrected EWA": f"{_v_hat:.2f}",
            "Unbiased Recovery": f"{_pct_recovered:.1f}%",
            "Error vs Mean (50.0)": f"{np.abs(_v_hat - _true_mean):.2f}",
        })

    _df_step_eval = pd.DataFrame(_records)

    return (
        mo.md("#### Cold-Start Bias Correction Step-by-Step Evaluation (beta = 0.98)"),
        mo.ui.table(_df_step_eval),
    )


@app.cell
def _(mo, np, pd):
    # Example 2: Adam Optimizer Simulation (Raw vs. Bias-Corrected Moments)
    # Objective: Minimize 0.5 * a * theta^2 with noisy gradients g_t = a * theta + N(0, sigma^2)
    _a = 2.0
    _lr = 0.1
    _beta1 = 0.90
    _beta2 = 0.999
    _eps = 1e-8
    _iterations = 40

    np.random.seed(42)
    _theta_sgd = 10.0
    _theta_uncorrected = 10.0
    _theta_adam = 10.0

    _m_unc = 0.0
    _v_unc = 0.0

    _m_cor = 0.0
    _v_cor = 0.0

    _history = []

    for _t in range(1, _iterations + 1):
        _noise = np.random.normal(0.0, 1.0)
        _grad_sgd = _a * _theta_sgd + _noise
        _grad_unc = _a * _theta_uncorrected + _noise
        _grad_cor = _a * _theta_adam + _noise

        # 1. Pure SGD
        _theta_sgd -= _lr * _grad_sgd

        # 2. Adam without Bias Correction
        _m_unc = _beta1 * _m_unc + (1.0 - _beta1) * _grad_unc
        _v_unc = _beta2 * _v_unc + (1.0 - _beta2) * (_grad_unc**2)
        _theta_uncorrected -= _lr * (_m_unc / (np.sqrt(_v_unc) + _eps))

        # 3. Adam with Full Bias Correction
        _m_cor = _beta1 * _m_cor + (1.0 - _beta1) * _grad_cor
        _v_cor = _beta2 * _v_cor + (1.0 - _beta2) * (_grad_cor**2)
        _m_hat = _m_cor / (1.0 - _beta1**_t)
        _v_hat = _v_cor / (1.0 - _beta2**_t)
        _theta_adam -= _lr * (_m_hat / (np.sqrt(_v_hat) + _eps))

        if _t in [1, 2, 5, 10, 20, 30, 40]:
            _history.append({
                "Iteration t": _t,
                "SGD Parameter": f"{_theta_sgd:.4f}",
                "Adam (Uncorrected)": f"{_theta_uncorrected:.4f}",
                "Adam (Bias-Corrected)": f"{_theta_adam:.4f}",
                "Loss Ratio (Corrected vs Uncorrected)": f"{(_theta_adam**2) / (_theta_uncorrected**2 + 1e-9):.4f}",
            })

    _df_adam_eval = pd.DataFrame(_history)

    return (
        mo.md("#### Convergence Comparison: SGD vs. Adam (Uncorrected vs. Bias-Corrected)"),
        mo.ui.table(_df_adam_eval),
    )


if __name__ == "__main__":
    app.run()
