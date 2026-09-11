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
    import matplotlib.pyplot as plt

    plt.style.use('dark_background')
    np.random.seed(47)
    return np, plt


@app.cell
def _(np):
    # Generate data
    mean = 0
    std = 2
    sample_size = 500
    sample = np.random.normal(loc=mean, scale=std, size=sample_size)
    return sample, sample_size


@app.cell
def _(np, sample, sample_size):
    # Define support range (empirical)
    support = (min(sample) - 0.5, max(sample) + 0.5)

    # Number of points to calculate the relative frequency for
    num_points = 20
    points = np.linspace(support[0], support[1], num_points)

    # Calculate relative frequencies
    cumulative_reqlative_frequencies = []
    for point in points:
        cumulative_reqlative_frequencies.append(
            (sample <= point).sum() / sample_size
        )
    return cumulative_reqlative_frequencies, points


@app.cell
def _(cumulative_reqlative_frequencies, plt, points):
    # Plot the cumulative relative frequencies vs points
    plt.figure(figsize=(4, 3), dpi=300)
    plt.step(points, cumulative_reqlative_frequencies, 
             where='mid', color='magenta')
    plt.xlabel('Values')
    plt.ylabel('Cumulative Relative Frequency')
    plt.title('ECDF')
    plt.grid(True, alpha=0.5, linestyle='--')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    <h1 style="font-weight: bold; background: linear-gradient(to right, teal, black); -webkit-background-clip: text; color: transparent;"> Cumulative Distribution Function (CDF)
    </h1>

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> CDF Definition
    </h2>

    The CDF $F_X(x)$ of a random variable $X$ gives the probability that $X$ is less than or equal to $x$:
    $$
    F_X(x) = P(X \leq x)
    $$

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> CDF for Discrete Random Variables
    </h2>

    For a discrete random variable $X$ with probability mass function (PMF) $p_X(x)$, the CDF is the cumulative sum of the probabilities up to $x$:
    $$
    F_X(x) = \sum_{x_i \leq x} p_X(x_i)
    $$
    Where $p_X(x_i)$ is the probability mass at $x_i$. The CDF increases in steps at each observed value of $X$.

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> CDF for Continuous Random Variables
    </h2>

    For a continuous random variable $X$ with probability density function (PDF) $f_X(x)$, the CDF is the integral of the PDF from $-\infty$ to $x$:
    $$
    F_X(x) = \int_{-\infty}^{x} f_X(t) \, dt
    $$
    Where $f_X(x)$ is the PDF of $X$, and the CDF is a smooth, continuous function.

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> Properties of CDF
    </h2>
    - **Non-decreasing**: The CDF is always non-decreasing, i.e., $F_X(x) \leq F_X(y)$ for $x \leq y$.
    - **Limits**:
      - $$\lim_{x \to -\infty} F_X(x) = 0$$: As $x \to -\infty$, the probability that $X$ is less than or equal to $x$ approaches 0.
      - $$\lim_{x \to \infty} F_X(x) = 1$$: As $x \to \infty$, the probability that $X$ is less than or equal to $x$ approaches 1.
    - **Right-continuity**: The CDF is right-continuous, i.e., $\lim_{\epsilon \to 0^+} F_X(x + \epsilon) = F_X(x)$, ensuring no jumps when approaching from the right.
    - **Range**: The CDF always lies within the range $[0, 1]$, as it represents a cumulative probability.

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> Empirical CDF (ECDF)
    </h2>

    Given a sample $X_1, X_2, \dots, X_n$, the empirical CDF $\hat{F}_n(x)$ is defined as the proportion of data points less than or equal to $x$:
    $$
    \hat{F}_n(x) = \frac{1}{n} \sum_{i=1}^{n} \mathbb{I}(X_i \leq x)
    $$
    Where $\mathbb{I}(X_i \leq x)$ is the indicator function, which equals 1 if $X_i \leq x$, and 0 otherwise. The ECDF is a step function that increases by $\frac{1}{n}$ at each observed value.

    <h2 style="font-weight: bold; background: linear-gradient(to right, magenta, cyan); -webkit-background-clip: text; color: transparent;"> Uses of Empirical CDF
    </h2>

    - **Non-parametric estimation**: The ECDF provides an estimate of the CDF without assuming any specific distribution for the data, making it useful in non-parametric statistics.
    - **Goodness-of-fit tests**: The ECDF is used in tests like the Kolmogorov-Smirnov test to compare a sample’s distribution with a theoretical distribution.
    - **Quantile estimation**: The ECDF can be used to estimate empirical quantiles, such as the median or percentiles, of the sample data.
    - **Anomaly Detection**: It is also used for point anomaly detection.
    """)
    return


if __name__ == "__main__":
    app.run()
