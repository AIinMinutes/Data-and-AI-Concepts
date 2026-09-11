import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Chapter 3: Norms and Metrics: Measuring Size and Distance

    ---

    ## [a] Why Do You Need to Know This?

    Machine learning is fundamentally about **minimizing distances** between predictions and targets, **constraining the size** of model parameters, and **measuring separation** between data points. Every one of these operations relies on a norm or a metric:

    - **Loss functions**: MSE computes the squared $L_2$ norm of the residual vector $\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$. MAE uses the $L_1$ norm.
    - **Regularization**: Ridge uses the $L_2$ norm of weights ($\|\mathbf{w}\|_2^2$). Lasso uses $L_1$ ($\|\mathbf{w}\|_1$). The choice of norm determines whether weights are shrunk uniformly or driven to exact zero (sparsity).
    - **Nearest-neighbor algorithms**: KNN, DBSCAN, and retrieval systems all require a distance metric to define "closeness."
    - **Gradient clipping**: Training deep networks often clips $\|\nabla\|_2$ to prevent exploding gradients.
    - **Batch normalization**: Normalizes activations by their $L_2$ statistics across the batch.

    **Prerequisites**: Chapter 1 (vectors), Chapter 2 (inner products, from which norms are derived).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] The Concept, the Math, and Its Role in ML / AI / Stats

    ### Norm: Measuring the "Size" of a Vector

    A **norm** is a function $\|\cdot\| : V \to \mathbb{R}_{\geq 0}$ satisfying three axioms:

    | Axiom | Formula | Intuition |
    | :--- | :--- | :--- |
    | **Non-negativity** | $\|\mathbf{v}\| \geq 0$; $\|\mathbf{v}\| = 0 \iff \mathbf{v} = \mathbf{0}$ | Only the zero vector has zero size |
    | **Absolute scalability** | $\|\alpha \mathbf{v}\| = |\alpha| \|\mathbf{v}\|$ | Scaling a vector scales its length |
    | **Triangle inequality** | $\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$ | The "shortcut" is never longer than going around |

    ### The $L_p$ Norm Family

    For $\mathbf{x} \in \mathbb{R}^n$ and $p \geq 1$:
    $$
    \|\mathbf{x}\|_p = \left(\sum_{i=1}^n |x_i|^p\right)^{1/p}
    $$

    | Norm | $p$ | Unit "Circle" Shape | ML Usage |
    | :--- | :---: | :--- | :--- |
    | $L_1$ (Manhattan) | 1 | Diamond | Lasso ($L_1$ penalty), sparsity, MAE loss |
    | $L_2$ (Euclidean) | 2 | Circle | Ridge ($L_2$ penalty), MSE loss, cosine similarity |
    | $L_\infty$ (Chebyshev) | $\infty$ | Square | Adversarial robustness (max perturbation) |
    | $L_p$ (general) | $p$ | Interpolation between diamond and square | Elastic Net blends $L_1$ and $L_2$ |

    ---

    ### Metric: Measuring "Distance" Between Two Points

    A **metric** $d : X \times X \to \mathbb{R}_{\geq 0}$ satisfies:

    | Axiom | Formula |
    | :--- | :--- |
    | **Non-negativity** | $d(\mathbf{x}, \mathbf{y}) \geq 0$ |
    | **Identity of indiscernibles** | $d(\mathbf{x}, \mathbf{y}) = 0 \iff \mathbf{x} = \mathbf{y}$ |
    | **Symmetry** | $d(\mathbf{x}, \mathbf{y}) = d(\mathbf{y}, \mathbf{x})$ |
    | **Triangle inequality** | $d(\mathbf{x}, \mathbf{z}) \leq d(\mathbf{x}, \mathbf{y}) + d(\mathbf{y}, \mathbf{z})$ |

    Every norm induces a metric via $d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|$, but not every metric comes from a norm (e.g., edit distance on strings, Jaccard distance on sets).

    ### The Minkowski Inequality: $L_2 \leq L_1$

    For any two points, the Euclidean distance is always less than or equal to the Manhattan distance:
    $$
    \|\mathbf{x} - \mathbf{y}\|_2 \leq \|\mathbf{x} - \mathbf{y}\|_1
    $$

    This follows from the Cauchy-Schwarz inequality and has practical consequences: a nearest-neighbor search with $L_1$ is more conservative (larger distances) than with $L_2$.

    ---

    ### Role in ML / AI / Stats

    | Context | Norm/Metric Used | Why |
    | :--- | :--- | :--- |
    | **MSE Loss** | $\|\mathbf{y} - \hat{\mathbf{y}}\|_2^2$ | Penalizes large errors quadratically |
    | **MAE Loss** | $\|\mathbf{y} - \hat{\mathbf{y}}\|_1$ | Robust to outliers (linear penalty) |
    | **Ridge Regression** | $\lambda\|\mathbf{w}\|_2^2$ | Shrinks all weights uniformly |
    | **Lasso Regression** | $\lambda\|\mathbf{w}\|_1$ | Drives irrelevant weights to exact zero |
    | **Gradient Clipping** | $\text{clip}(\nabla, \|\nabla\|_2 \leq c)$ | Prevents exploding gradients in RNNs/Transformers |
    | **KNN / DBSCAN** | $\|\mathbf{x}_i - \mathbf{x}_j\|_p$ | Defines neighborhoods for classification/clustering |
    | **Adversarial Robustness** | $\|\boldsymbol{\delta}\|_\infty \leq \epsilon$ | Bounds worst-case pixel perturbation |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    ### Example 1: Visualizing Unit Norm "Circles" for Different $p$-Norms

    The unit norm set $\{\mathbf{x} : \|\mathbf{x}\|_p = 1\}$ reveals the geometry each norm imposes:
    """)
    return


@app.cell
def _():
    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    return np, plt


@app.cell
def _(np, plt):
    def unit_circle(p, num_points=1000):
        """Compute the unit circle boundary for L_p norm."""
        if p == np.inf:
            x = np.array([1, 1, -1, -1, 1])
            y = np.array([1, -1, -1, 1, 1])
            return x, y
        x = np.linspace(-1, 1, num_points)
        y_pos = np.maximum(0, 1 - np.abs(x)**p) ** (1/p)
        y_neg = -y_pos
        return np.concatenate([x, x[::-1]]), np.concatenate([y_pos, y_neg])

    fig, ax = plt.subplots(figsize=(7, 7), dpi=150)

    norms = [
        (0.5, "#9b59b6", "$p = 0.5$ (quasi-norm)"),
        (1,   "#e74c3c", "$L_1$: Diamond (Lasso)"),
        (2,   "#3498db", "$L_2$: Circle (Ridge)"),
        (4,   "#e67e22", "$L_4$: Rounded square"),
        (np.inf, "#2ecc71", r"$L_\infty$: Square (Adversarial)"),
    ]

    for p, color, label in norms:
        x, y = unit_circle(p)
        ax.plot(x, y, color=color, linewidth=2, label=label)

    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_aspect("equal")
    ax.set_title("Unit Norm Sets: How Different $p$-Norms Define 'Distance = 1'", fontsize=12, fontweight="bold")
    ax.set_xlabel("$x_1$", fontsize=11)
    ax.set_ylabel("$x_2$", fontsize=11)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig
    return ax, fig, norms, unit_circle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Euclidean vs Manhattan Distance: The Minkowski Inequality in Action

    For random 2D vector pairs, Euclidean distance is always $\leq$ Manhattan distance:
    """)
    return


@app.cell
def _(np, plt):
    rng = np.random.default_rng(47)
    points1 = rng.integers(-3, 4, size=(200, 2))
    points2 = rng.integers(-3, 4, size=(200, 2))

    d_euclidean = np.sqrt(np.sum((points2 - points1)**2, axis=1))
    d_manhattan = np.sum(np.abs(points2 - points1), axis=1)

    fig2, ax2 = plt.subplots(figsize=(6, 6), dpi=150)
    ax2.scatter(d_euclidean, d_manhattan, color="#1B7A7A", alpha=0.6, edgecolors="k", s=40, label="Vector pairs")
    max_val = max(d_manhattan.max(), d_euclidean.max()) + 0.5
    ax2.plot([0, max_val], [0, max_val], color="#D65A31", linestyle="--", linewidth=2, label=r"$d_{L_2} = d_{L_1}$ (equality line)")
    ax2.set_xlabel("Euclidean Distance ($L_2$)", fontsize=11)
    ax2.set_ylabel("Manhattan Distance ($L_1$)", fontsize=11)
    ax2.set_title(r"Minkowski Inequality: $\|x\|_2 \leq \|x\|_1$ (all points above the line)", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    fig2
    return ax2, d_euclidean, d_manhattan, fig2, max_val, points1, points2, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    - A **norm** measures the size of a vector; a **metric** measures the distance between two vectors. Every norm induces a metric, but not vice versa.
    - The $L_p$ norm family ($p = 1, 2, \infty$) controls the **geometry of your model**: $L_1$ encourages sparsity (Lasso), $L_2$ encourages small uniform weights (Ridge), and $L_\infty$ bounds worst-case deviations (adversarial robustness).
    - **Euclidean distance is always ≤ Manhattan distance** (Minkowski inequality), which affects nearest-neighbor search behavior and distance-based clustering.
    - In practice, the **choice of norm is a modeling decision**: it determines how errors are penalized (loss), how parameters are constrained (regularization), and what "nearby" means (retrieval).
    """)
    return


if __name__ == "__main__":
    app.run()
