# /// script
# dependencies = ["pingouin"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # packages added via marimo's package management: pingouin==0.5.5 !pip install -q pingouin==0.5.5

    import numpy as np
    import pandas as pd
    import pingouin as pg
    import matplotlib.pyplot as plt

    import itertools
    from utils import validate_plot_config, plot_regression_and_heatmap_plots # Plotting utilities

    # Set the random seed for reproducibility
    np.random.seed(42)

    # Create an underlying variable Z that influences both X and Y
    Z = np.random.normal(loc=10, scale=2, size=30)
    # Generate X and Y based on Z with added noise
    X = 10 * Z + np.random.normal(loc=0, scale=10, size=30)
    Y = 20 * Z + np.random.normal(loc=1, scale=10, size=30)

    # Combine into a DataFrame
    data = pd.DataFrame(dict(X=X, Y=Y, Z=Z))
    data = data.apply(lambda x: round(x, 1))
    data.head(5)
    return (
        data,
        itertools,
        pg,
        plot_regression_and_heatmap_plots,
        plt,
        validate_plot_config,
    )


@app.cell
def _(
    data,
    itertools,
    plot_regression_and_heatmap_plots,
    plt,
    validate_plot_config,
):
    plt.style.use('dark_background')
    pairs = list(itertools.combinations(data.columns, 2))
    save_path = 'reg_heatmap_plots.png'
    validate_plot_config(pairs, data, save_path)
    fig = plot_regression_and_heatmap_plots(pairs, data, save_path)
    return


@app.cell
def _(data, pg):
    # Pearson's Correlation Coefficient of X and Y
    pearson_corr = data[['X', 'Y']].corr().iloc[0, 1]
    print(f"""
    Pearson's Correlation Coefficient between X and Y: {pearson_corr:.2f}
    """)
    # Partial correlation coefficient between X and Y, controlling for Z
    partial_corr_coefficient = pg.partial_corr(data=data, x='X', 
                                               y='Y', covar='Z', 
                                               method='pearson')
    print(f"""Partial Correlation Coefficient b/w X and Y, controllling for Z: 
    {partial_corr_coefficient["r"].values[0]:.3f}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Conclusion

    The Pearson's correlation coefficient between `X` and `Y` is relatively high, suggesting a strong relationship.
    <br> <br>
    However, after calculating the partial correlation between `X` and `Y` while controlling for `Z`, we see that the correlation significantly weakens. <br> <br>  This indicates that `X` and `Y` are spuriously correlated due to their mutual dependence on `Z`.
    """)
    return


if __name__ == "__main__":
    app.run()
