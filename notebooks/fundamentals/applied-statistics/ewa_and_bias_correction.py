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
    ### Exponentially Weighted Average (EWA)
    - **EWA** smooths timeseries considering both past and recent data.
    - **Alpha ($\alpha$)**: Smoothing parameter that controls the weight on recent data. If high, more weight to recent data, and if low then more weight to past data.
    - **Formula**:
    - $V_t = \alpha \cdot X_t + (1 - \alpha) \cdot V_{t-1}$
    - $V_t$: Smoothed value at time $t$
    - $X_t$: Actual value at time $t$
    - $V_{t-1}$: Previous smoothed value
    - **Alpha ($\alpha$)**: Ranges from 0 (no smoothing) to 1 (no past data used).

    ### Key Terms:
    - **$L_t$ (Level)**: Smoothed value of the series at time $t$.
    - **$T_t$ (Trend)**: Slope or growth/decline at time $t$.
    - **$S_t$ (Seasonality)**: Seasonal component at time $t$.
    - **$Y_t$ (Observed value)**: Actual value at time $t$.
    - **$\alpha$**: Level smoothing constant (0 to 1).
    - **$\beta$**: Trend smoothing constant (0 to 1).
    - **$\gamma$**: Seasonality smoothing constant (0 to 1).
    - **$m$**: Seasonal period length.
    - **$h$**: Forecast horizon.
    - **$\hat{Y}_{t+h}$**: Forecasted value at time $t+h$.

    ### Exponential Smoothing Methods for Forecasting
    1. **Single Exponential Smoothing (SES)**
    - **Assumptions**: No trend or seasonality.
    - **Formula**:
    $$
    L_t = \alpha Y_t + (1 - \alpha) L_{t-1}
    $$
    - **Forecast**:
    $$
    \hat{Y}_{t+h} = L_t
    $$
    2. **Double Exponential Smoothing (DES)**
    - **Assumptions**: Linear trend, no seasonality.
    - **Formula**:
    $$
    L_t = \alpha Y_t + (1 - \alpha)(L_{t-1} + T_{t-1}),
    $$
    $$
    T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1}
    $$
    - **Forecast**:
    $$
    \hat{Y}_{t+h} = L_t + h T_t
    $$
    3. **Triple Exponential Smoothing (Holt-Winters)**
    - **Assumptions**: Linear trend and seasonality.
    - **Formula (additive seasonality)**:
    $$
    L_t = \alpha (Y_t - S_{t-m}) + (1 - \alpha)(L_{t-1} + T_{t-1}),
    $$
    $$
    T_t = \beta (L_t - L_{t-1}) + (1 - \beta) T_{t-1},
    $$
    $$
    S_t = \gamma (Y_t - L_t) + (1 - \gamma) S_{t-m}
    $$
    - **Forecast**:
    $$
    \hat{Y}_{t+h} = L_t + h T_t + S_{t-m+h}
    $$
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    np.random.seed(47); plt.style.use('dark_background')

    def calculate_ewa(series, alpha):
        """
        Calculate the Exponentially Weighted Average (EWA) of 
        a time series.
        """
        ewa = np.zeros_like(series)
        ewa[0] = series[0]

        for t in range(1, len(series)): 
            ewa[t] = alpha * series[t] + (1 - alpha) * ewa[t - 1]
        return ewa

    return ExponentialSmoothing, calculate_ewa, np, pd, plt


@app.cell
def _(calculate_ewa, np, plt):
    # Simulation
    n = 100; t = np.arange(n); mean = 0; std = 1; alpha = 0.2
    timeseries = np.random.normal(mean, std, n)
    ewa_series = calculate_ewa(timeseries, alpha)

    plt.figure(figsize=(6, 3), dpi=200)
    plt.plot(range(1, n + 1), timeseries, label="Timeseries", 
             color='skyblue', alpha=0.7)
    plt.plot(range(1, n + 1), ewa_series, label=f'EWA (alpha={alpha})', 
             color='orange', linewidth=2)
    plt.title("EWA of a Stationary Series", fontsize=16)
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("Value", fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.2)
    plt.show()
    return


@app.cell
def _(np, pd):
    # Generate data (1 month with daily data, weekly seasonality)
    n_1 = 30
    periods = 7  # weekly seasonality
    trend_slope = 0.5
    seasonality_amplitude = 3
    date_range = pd.date_range(start='2024-11-01', periods=n_1, freq='D')
    trend = trend_slope * np.arange(n_1)
    seasonality = seasonality_amplitude * np.sin(2 * np.pi * np.arange(n_1) / periods)
    noise = np.random.normal(0, 1, size=n_1)
    data = trend + seasonality + noise
    df = pd.DataFrame({'date': date_range, 'value': data})
    df.sample(5)
    return df, n_1, periods, seasonality_amplitude, trend_slope


@app.cell
def _(
    ExponentialSmoothing,
    df,
    n_1,
    np,
    pd,
    periods,
    plt,
    seasonality_amplitude,
    trend_slope,
):
    # Fit Single Exponential Smoothing (SES)
    ses_fit = ExponentialSmoothing(df['value']).fit()
    des_fit = ExponentialSmoothing(df['value'], trend='add', seasonal=None).fit(smoothing_level=0.2, smoothing_trend=0.2)
    # Fit Double Exponential Smoothing (DES)
    tes_fit = ExponentialSmoothing(df['value'], trend='add', seasonal='add', seasonal_periods=periods).fit()
    ses_fitted, des_fitted, tes_fitted = (ses_fit.fittedvalues, des_fit.fittedvalues, tes_fit.fittedvalues)
    ses_forecast, des_forecast, tes_forecast = (ses_fit.forecast(steps=1), des_fit.forecast(steps=1), tes_fit.forecast(steps=1))  # just for demo
    forecast_date = df['date'].iloc[-1] + pd.Timedelta(days=1)
    # Fit Triple Exponential Smoothing (TES)
    expected_next_value = trend_slope * n_1 + seasonality_amplitude * np.sin(2 * np.pi * n_1 / periods)
    forecast_df = pd.DataFrame({'date': [forecast_date], 'SES': ses_forecast, 'DES': des_forecast, 'TES': tes_forecast, 'Expected value': expected_next_value})
    plt.figure(figsize=(8, 6), dpi=300)
    plt.plot(df['date'], df['value'], label='Simulated Time Series', color='white', alpha=0.7, linestyle='--', linewidth=3)
    plt.plot(df['date'], ses_fitted, label='SES Fitted Values', color='red', linewidth=3)
    # Get fitted values from each model
    plt.plot(df['date'], des_fitted, label='DES Fitted Values', color='yellow', linewidth=3)
    plt.plot(df['date'], tes_fitted, label='TES Fitted Values', color='green', linewidth=3)
    plt.plot(forecast_df['date'], forecast_df['SES'], 'x', color='red', label='SES Forecast', markersize=8)
    plt.plot(forecast_df['date'], forecast_df['DES'], '+', color='yellow', label='DES Forecast', markersize=8)
    plt.plot(forecast_df['date'], forecast_df['TES'], 'o', color='green', label='TES Forecast', markersize=8)
    plt.plot(forecast_df['date'], forecast_df['Expected value'], '*', color='white', markersize=12, label='Expected Value')
    # Forecast the next value after the last point
    plt.title('Comparison of SES, DES, and TES Forecasting Methods', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Value', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.1)
    plt.tight_layout()
    # Define forecast date
    plt.xticks(rotation=45)
    # Calculate the expected value for the next timestamp (no noise)
    # Create forecast dataframe with only the forecasted values
    # Plotting the generated time series and the fitted values from each model
    # Plotting the forecast points as markers directly on the lines (+ for prediction)
    # Marking the expected value (without noise) for the next timestamp
    # Customize the plot
    plt.show()  # Adding expected value for next timestamp
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Conclusion:
    As you can see TES method fits the timeseries the best as expected because TES is the only method that can capture seasonality out of the three methods.
    It forecasts the closet to the expected value on Dec 1st, 2024
    """)
    return


if __name__ == "__main__":
    app.run()
