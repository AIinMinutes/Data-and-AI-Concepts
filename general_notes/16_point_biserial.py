import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    import warnings

    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from scipy.stats import pointbiserialr

    # Set global font sizes using rc
    plt.rc("axes", titlesize=20, labelsize=18)  # Title and label sizes
    plt.rc("xtick", labelsize=16)  # X-axis tick label size
    plt.rc("ytick", labelsize=16)  # Y-axis tick label size

    warnings.filterwarnings("ignore")
    np.random.seed(47)
    plt.style.use("dark_background")

    n = 100  # Number of samples
    continuous_var = np.random.normal(loc=0, scale=1, size=n)

    # Generate a binary categorical variable (0 or 1)
    binary_var = np.random.choice([0, 1], size=n)
    binary_var_correlated = np.where(continuous_var > 0, 1, 0)

    # Compute the point-biserial correlation for both cases
    r_pb, _ = pointbiserialr(binary_var, continuous_var)
    r_pb2, _ = pointbiserialr(binary_var_correlated, continuous_var)

    # Create a figure for the violin plots
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.violinplot(x=binary_var, y=continuous_var, palette="Set2")
    plt.title(f"Case-1: Uncorrelated\nPoint-Biserial r = {r_pb:.2f}")
    plt.xlabel("Binary Variable")
    plt.ylabel("Continuous Variable")
    plt.subplot(1, 2, 2)
    sns.violinplot(x=binary_var_correlated, y=continuous_var, palette="Set2")
    plt.title(f"Case-2: Correlated\nPoint-Biserial r = {r_pb2:.2f}")
    plt.xlabel("Binary Variable")
    plt.ylabel("Continuous Variable")
    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Point-Biserial Correlation Coefficient

    The **point-biserial correlation coefficient** measures the association between a binary variable and a continuous variable.

    #### Formula

    $$
    r_{pb} = \frac{M_1 - M_0}{s} \cdot \sqrt{\frac{n_1 n_0}{n (n-1)}}
    $$

    Where:
    - $M_1$ and $M_0$ are the means of the continuous variable for each group.
    - $s$ is the standard deviation of the continuous variable.
    - $n_1$ and $n_0$ are the group sizes, and $n$ is the total sample size.

    #### Interpretation
    - $r_{pb} = 0$: No correlation.
    - $r_{pb} > 0$: Positive correlation.
    - $r_{pb} < 0$: Negative correlation.

    #### Assumptions
    - One variable is continuous, and the other is binary.
    - The continuous variable is approximately normally distributed within each group.
    """)
    return


if __name__ == "__main__":
    app.run()
