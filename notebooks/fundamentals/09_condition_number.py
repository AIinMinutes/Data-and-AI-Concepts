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
    ### Condition Number ($\kappa(\mathbf{X})$)
    - $\kappa(\mathbf{X}) = \frac{\sigma_{\text{max}}}{\sigma_{\text{min}}}$
    - $\sigma_{\text{max}}$: Largest singular value of $\mathbf{X}$
    - $\sigma_{\text{min}}$: Smallest singular value of $\mathbf{X}$
    - **Interpretation**:
    - **High $\kappa(\mathbf{X})$ (>30)**: Severe multicollinearity, near-singular matrix, unstable regression model.
    - **Low $\kappa(\mathbf{X})$ (~1)**: Well-conditioned matrix, less multicollinearity, stable model.
    - **Use**: Measures how sensitive the solution is to small changes in input features.

    The **Variance Inflation Factor (VIF)** of a feature measures how much the variance of the estimated regression coefficient for that feature is inflated due to multicollinearity with other features.
    **Formula for the $i$-th feature**
    $$
    VIF_i = \frac{1}{1 - R_i^2}
    $$
    Where $R_i^2$ is coefficient of determination when the $i$-th feature is regressed on all other features in the model.
    **Interpretation**:
    - $VIF_i > 10$: Indicates high multicollinearity
    - $VIF_i \leq 10$: Indicates that the feature is not significantly collinear with others
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    np.random.seed(47)
    plt.style.use('dark_background')
    n = 50
    x1 = np.random.randint(10, 100, n)
    x2 = np.random.randint(10, 100, n)
    x3 = (0.8 * x1 + 0.2 * x2 + np.random.normal(0, 10)).astype('int')
    _y = 10 + 0.4 * x1 + 0.6 * x2 + np.random.randn()
    data = pd.DataFrame(dict(x1=x1, x2=x2, x3=x3, y=_y))
    # True relationship for target variable y
    data.sample()
    return data, np, pd, plt, sm, sns, variance_inflation_factor


@app.cell
def _(data, plt, sns):
    plt.figure(figsize=(3, 5), dpi=300)
    pp = sns.pairplot(
        data, kind='kde', 
        height=2, corner=True, 
        diag_kws={'color': 'brown'}, 
        plot_kws={'color': 'magenta'}
    )
    plt.suptitle(
        "Relationships among Predictors and Response", 
        y=1.02, fontsize=16
    )
    plt.show()
    return


@app.cell
def _(data, np):
    X = data.drop('y', axis=1)
    _y = data['y']
    CN = np.linalg.cond(X)
    print(f'The condition number of X is : {CN:.2f}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since the condition number ($\kappa(\mathbf{X})$) of the matrix is greater than 30, it indicates **severe multicollinearity**.
    """)
    return


@app.cell
def _(pd, variance_inflation_factor):
    def calc_vif(X):
        vif = pd.DataFrame(); vif["vars"] = X.columns
        vif["VIF"] = [
            variance_inflation_factor(X.values, i) 
            for i in range(X.shape[1])
        ]
        return vif

    return (calc_vif,)


@app.cell
def _(calc_vif, data):
    X_1 = data.drop(['y'], axis=1)
    _vif_result = calc_vif(X_1)
    _vif_result
    return


@app.cell
def _(calc_vif, data):
    X_2 = data.drop(['y', 'x3'], axis=1)
    _vif_result = calc_vif(X_2)
    _vif_result
    return (X_2,)


@app.cell
def _(X_2, data, sm):
    # Now VIFs for both x1 and x2 is < 10; fit OLSE
    X_3 = X_2.assign(constant=1)
    model = sm.OLS(data['y'], X_3).fit()
    model.summary()
    return


if __name__ == "__main__":
    app.run()
