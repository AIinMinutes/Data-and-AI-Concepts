import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import pandas as pd
    import seaborn as sns
    from matplotlib import pyplot as plt
    from scipy.stats import kendalltau  # For calculating Kendall's tau-b

    return kendalltau, pd, plt, sns


@app.cell
def _():
    # Observations
    s = [
        (5, 4),
        (3, 3),
        (4, 4),
        (2, 2),
        (1, 1),
        (4, 3),
        (5, 5),
        (3, 2),
        (2, 3),
        (4, 4),
        (5, 5),
        (3, 3),
        (1, 2),
        (2, 1),
        (4, 5),
        (3, 2),
        (5, 4),
        (1, 2),
        (2, 3),
        (3, 3),
    ]
    return (s,)


@app.cell
def _(pd, s):
    # Creating pandas dataframe from the observations
    df = pd.DataFrame(s, columns=["Job Satisfaction", "Work-Life Balance"])
    df.head(5)
    return (df,)


@app.cell
def _(df, plt, sns):
    plt.style.use("dark_background")
    plt.figure(figsize=(4, 3), dpi=300)
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], color="violet")
    plt.show()
    return


@app.cell
def _(df, kendalltau):
    tau, _ = kendalltau(df["Job Satisfaction"], df["Work-Life Balance"])
    print(f"Kendall's correlation coefficient value is {tau:.2f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Since Kendall's correlation coefficient value is positive and greater than 0.6, the given variables in the given
    scenario are **positively** associated and are in **good** agreement.
    """)
    return


if __name__ == "__main__":
    app.run()
