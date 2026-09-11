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
    ### Jensen's Inequality
    - **Statement**:
     For a convex function $f$ and a random variable $X$:
     $$
    f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)].
    $$
    - **Convex Function**:
     A function $f(x)$ is convex if:
     $$
    f(\alpha x_1 + (1-\alpha)x_2) \leq \alpha f(x_1) + (1-\alpha)f(x_2), \quad \forall \alpha \in [0, 1].
    $$
    - **Interpretation**:
     The value of $f$ at the mean $\mathbb{E}[X]$ is always less than or equal to the average value of $f(X)$.
    - **Key Terms**:
        - $f(\mathbb{E}[X])$: Function applied to the expected value of $X$.
        - $\mathbb{E}[f(X)]$: Expected value of the function applied to $X$.
    - **Applications**:
        - **Optimization**: Guarantees for algorithms working with convex objectives.
        - **Variational Inference**: Jensen's inequality is the foundation for deriving the **evidence lower bound (ELBO)** in variational inference.
    """)
    return


@app.cell
def _():
    import sympy as sp
    from IPython.display import Math, display

    x, lambda_ = sp.symbols("x lambda", positive=True)
    # Define symbols
    f_x = x**2
    pdf_x = lambda_ * sp.exp(-lambda_ * x)
    # Define the function and the pdf
    _E_X = sp.integrate(x * pdf_x, (x, 0, sp.oo))
    _f_E_X = f_x.subs(x, _E_X)
    _E_f_X = sp.integrate(f_x * pdf_x, (x, 0, sp.oo))
    # Calculate E[X], f(E[X]), and E[f(X)]
    _E_X = sp.simplify(_E_X)  # E[X]
    _f_E_X = sp.simplify(_f_E_X)  # f(E[X])
    _E_f_X = sp.simplify(_E_f_X)  # E[f(X)]
    display(Math(f"E[X] = {sp.latex(_E_X)}"))
    # Simplify the results
    display(Math(f"E[f(X)] = {sp.latex(_E_f_X)}"))
    display(Math(f"f(E[X]) = {sp.latex(_f_E_X)}"))
    jensen_inequality = sp.simplify(_f_E_X <= _E_f_X)
    # Display the results using Math for proper LaTeX rendering in Jupyter
    # Check if Jensen's inequality holds
    print("Does Jensen's Inequality hold?:", jensen_inequality)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import expon

    # Define a convex function f(x) = x^2
    def f(x):
        return x**2

    np.random.seed(47)
    # Generate random samples for the random variable X using scipy.stats
    plt.style.use("dark_background")  # For reproducibility
    scale_param = 1
    X = expon.rvs(scale=scale_param, size=5000)
    # Set scale parameter for the exponential distribution (1/λ)
    _E_X = np.mean(X)  # Lambda = 1
    _f_E_X = f(_E_X)  # Exponential distribution samples
    _E_f_X = np.mean(f(X))
    # Compute values
    x_vals = np.linspace(0, np.max(X), 100)  # E[X]
    y_vals = f(x_vals)  # f(E[X])
    plt.figure(figsize=(6, 5), dpi=300)  # E[f(X)]
    plt.plot(x_vals, y_vals, label="f(x) = x²", color="orange", linewidth=2)
    # Plot the convex function f(x) = x^2
    plt.scatter(_E_X, _f_E_X, color="cyan", label="f(E[X])", zorder=5, marker="x")
    plt.scatter(X, f(X), color="gray", alpha=0.3, label="Samples (X, f(X))", zorder=1, marker="*")
    plt.axhline(y=_E_f_X, color="magenta", linestyle="--", label="E[f(X)]")
    plt.text(_E_X + 0.05, _f_E_X, f"f(E[X]) = {_f_E_X:.3f}", color="cyan", fontsize=10, ha="left", va="top")
    plt.text(
        _E_X, _E_f_X, f"E[f(X)] = {_E_f_X:.3f}", color="magenta", fontsize=10, ha="right", va="bottom"
    )  # Function in orange
    plt.title("Jensen's Inequality", fontsize=14)
    # Plot the points for Jensen's inequality
    plt.xlabel("X", fontsize=12)  # f(E[X]) in green
    plt.ylabel("f(X)", fontsize=12)  # Samples in gray
    plt.xticks([i for i in range(11)])  # E[f(X)] in blue
    plt.legend(fontsize=10)
    # Annotate the plot
    plt.grid(alpha=0.3, linestyle="--")  # f(E[X]) annotation in green
    # Add labels and legend
    plt.show()  # E[f(X)] annotation in blue
    return


if __name__ == "__main__":
    app.run()
