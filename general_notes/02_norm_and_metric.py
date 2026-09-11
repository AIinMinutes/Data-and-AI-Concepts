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
    # Note 02: Norms and Metrics: Measuring Size and Distance

    &larr; Previous Note: [01 Inner Products](01_inner_product.py) | Next Note: [03 Hyperplanes](03_hyperplanes.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

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

    ## [b] Concept explanation with their role in ML/AI/Stats?

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
    import plotly.graph_objects as go

    return go, np


@app.cell
def _(go, np):
    def unit_circle(p, num_points=1000):
        """Compute the unit circle boundary for L_p norm."""
        if p == np.inf:
            x = np.array([1, 1, -1, -1, 1])
            y = np.array([1, -1, -1, 1, 1])
            return x, y
        x = np.linspace(-1, 1, num_points)
        y_pos = np.maximum(0, 1 - np.abs(x) ** p) ** (1 / p)
        y_neg = -y_pos
        return np.concatenate([x, x[::-1]]), np.concatenate([y_pos, y_neg])

    fig = go.Figure()

    norms = [
        (0.5, "#9b59b6", "p = 0.5 (quasi-norm)"),
        (1, "#e74c3c", "L₁: Diamond (Lasso)"),
        (2, "#3498db", "L₂: Circle (Ridge)"),
        (4, "#e67e22", "L₄: Rounded square"),
        (np.inf, "#2ecc71", "L_∞: Square (Adversarial)"),
    ]

    for p, color, label in norms:
        x, y = unit_circle(p)
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=label, line=dict(color=color, width=2.5)))

    fig.update_layout(
        title=dict(text="Unit Norm Sets: How Different p-Norms Define Distance = 1", font=dict(size=14)),
        xaxis=dict(title="x₁", range=[-1.6, 1.6], zeroline=True, gridcolor="#e5e5e5", scaleanchor="y", scaleratio=1),
        yaxis=dict(title="x₂", range=[-1.6, 1.6], zeroline=True, gridcolor="#e5e5e5"),
        template="plotly_white",
        legend=dict(x=0.68, y=0.98),
        width=600,
        height=600,
    )

    fig
    return fig, norms, unit_circle


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Euclidean vs Manhattan Distance: The Minkowski Inequality in Action

    For random 2D vector pairs, Euclidean distance is always $\leq$ Manhattan distance:
    """)
    return


@app.cell
def _(go, np):
    rng = np.random.default_rng(47)
    points1 = rng.integers(-3, 4, size=(200, 2))
    points2 = rng.integers(-3, 4, size=(200, 2))

    d_euclidean = np.sqrt(np.sum((points2 - points1) ** 2, axis=1))
    d_manhattan = np.sum(np.abs(points2 - points1), axis=1)

    max_val = float(max(d_manhattan.max(), d_euclidean.max()) + 0.5)

    fig2 = go.Figure()
    fig2.add_trace(
        go.Scatter(
            x=d_euclidean,
            y=d_manhattan,
            mode="markers",
            name="Vector pairs",
            marker=dict(color="#1B7A7A", size=7, opacity=0.7, line=dict(color="black", width=0.5)),
        )
    )
    fig2.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            name="d_L₂ = d_L₁ (equality line)",
            line=dict(color="#D65A31", dash="dash", width=2),
        )
    )
    fig2.update_layout(
        title=dict(
            text="Minkowski Inequality: ||x||₂ ≤ ||x||₁ (all points on or above equality line)", font=dict(size=13)
        ),
        xaxis=dict(title="Euclidean Distance (L₂)", zeroline=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Manhattan Distance (L₁)", zeroline=True, gridcolor="#e5e5e5"),
        template="plotly_white",
        width=650,
        height=550,
    )

    fig2
    return d_euclidean, d_manhattan, fig2, max_val, points1, points2, rng


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    - A **norm** measures the size of a vector; a **metric** measures the distance between two vectors. Every norm induces a metric, but not vice versa.
    - The $L_p$ norm family ($p = 1, 2, \infty$) controls the **geometry of your model**: $L_1$ encourages sparsity (Lasso), $L_2$ encourages small uniform weights (Ridge), and $L_\infty$ bounds worst-case deviations (adversarial robustness).
    - **Euclidean distance is always ≤ Manhattan distance** (Minkowski inequality), which affects nearest-neighbor search behavior and distance-based clustering.
    - In practice, the **choice of norm is a modeling decision**: it determines how errors are penalized (loss), how parameters are constrained (regularization), and what "nearby" means (retrieval).

    ---

    &larr; Previous Note: [01 Inner Products](01_inner_product.py) | Next Note: [03 Hyperplanes](03_hyperplanes.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
