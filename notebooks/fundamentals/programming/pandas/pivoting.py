import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import numpy as np
    import seaborn as sns
    import pandas as pd

    from matplotlib import pyplot as plt

    np.random.seed(47)
    plt.style.use('dark_background')
    return plt, sns


@app.cell
def _(sns):
    fmri_dataset = sns.load_dataset('fmri')
    fmri_dataset.sample(10, random_state=47)
    return (fmri_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Pivot Table Explanation

    A pivot table is a data transformation tool that reshapes a dataset by aggregating values based on specific row and column identifiers. It allows for easy comparison and summarization of data.

    - **`index`**: Defines the rows of the resulting table.
    - **`columns`**: Defines the columns of the resulting table.
    - **`values`**: Specifies the data to be aggregated.
    - **`aggfunc`**: Defines the aggregation function, such as `'mean'`, `'sum'`, etc. By default, it is `'mean'`.

    For the given example:

    - **`index='timepoint'`**: The rows represent unique timepoints in the fMRI scan.
    - **`columns='event'`**: Separate columns are created for each event type (`stim` and `cue`).
    - **`values='signal'`**: The `signal` values are aggregated.
    - **`aggfunc='mean'`**: The average signal for each combination of `timepoint` and `event` is computed.

    This structure allows for an easy comparison of the signal trends over time for each event type.
    """)
    return


@app.cell
def _(fmri_dataset):
    pivoted_fmri = fmri_dataset.pivot_table(
        index='timepoint', 
        columns='event', 
        values='signal',
        aggfunc='mean'
    )

    pivoted_fmri.sample(10)
    return (pivoted_fmri,)


@app.cell
def _(pivoted_fmri, plt):
    pivoted_fmri.plot(figsize=(8, 8), color=['c', 'm'], 
                      linewidth=3
    )
    plt.title('Average Signal by Timepoint and Event')
    plt.ylabel('Signal')
    plt.xlabel('Timepoint')
    plt.xticks([*range(0, 21, 2)])
    plt.legend(title='Event')
    plt.show()
    return


if __name__ == "__main__":
    app.run()
