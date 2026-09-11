import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import plotly.graph_objects as go

    return go, mo, np


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 03: Hyperplanes, Half-Spaces, and Separation Distance

    &larr; Previous Note: [02 Norm and Metric](02_norm_and_metric.py) | Next Note: [04 Rank-One Matrices](04_rank_one_matrices.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Hyperplanes are the fundamental geometric building blocks of linear decision boundaries in machine learning, statistics, and optimization. Whenever a model splits data into discrete categories using a linear weighting of features, it constructs a hyperplane.

    Mastering the geometry of hyperplanes allows you to:

    1. Understand how linear classifiers (Perceptrons, Logistic Regression, Support Vector Machines) partition an $n$-dimensional feature space into decision regions.
    2. Compute exact orthogonal distances from observations to decision boundaries to evaluate prediction certainty and margins.
    3. Formalize the concept of maximum-margin classification, directly motivating the primal and dual formulations of Support Vector Machines.
    4. Quantify adversarial vulnerability by calculating the minimal perturbation needed to push a data point across a decision threshold into an opposing class.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Definition of a Hyperplane

    An affine hyperplane $H$ in $\mathbb{R}^n$ is an $(n-1)$-dimensional subspace shifted by an intercept. It is defined algebraically as the set of all points $\mathbf{x} \in \mathbb{R}^n$ satisfying:

    $$
    H = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{w}^T \mathbf{x} + b = 0\}
    $$

    where:
    * $\mathbf{w} = [w_1, w_2, \dots, w_n]^T \in \mathbb{R}^n \setminus \{\mathbf{0}\}$ is the **normal vector**, orthogonal to every directional vector lying within the hyperplane.
    * $b \in \mathbb{R}$ is the **bias** or **offset**, determining the position of the hyperplane relative to the origin.
    * When $b = 0$, the hyperplane passes directly through the origin and forms an $(n-1)$-dimensional linear subspace.

    ### Half-Spaces

    A hyperplane splits $\mathbb{R}^n$ into two closed, convex half-spaces:

    $$
    H^+ = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{w}^T \mathbf{x} + b \ge 0\}
    $$

    $$
    H^- = \{\mathbf{x} \in \mathbb{R}^n : \mathbf{w}^T \mathbf{x} + b \le 0\}
    $$

    In binary classification, the decision rule assigns class labels according to the sign of the linear score: $\hat{y} = \text{sgn}(\mathbf{w}^T \mathbf{x} + b)$.

    ### Orthogonal Distance from a Point to a Hyperplane

    Let $\mathbf{x}_0 \in \mathbb{R}^n$ be an arbitrary observation, and let $\mathbf{x}_p \in H$ be its orthogonal projection onto $H$. The displacement vector $\mathbf{x}_0 - \mathbf{x}_p$ is collinear with the unit normal vector $\hat{\mathbf{w}} = \frac{\mathbf{w}}{\|\mathbf{w}\|_2}$:

    $$
    \mathbf{x}_0 - \mathbf{x}_p = d_{\text{signed}} \cdot \frac{\mathbf{w}}{\|\mathbf{w}\|_2}
    $$

    Taking the inner product of both sides with $\mathbf{w}$ and substituting $\mathbf{w}^T \mathbf{x}_p = -b$ gives:

    $$
    \mathbf{w}^T \mathbf{x}_0 - \mathbf{w}^T \mathbf{x}_p = d_{\text{signed}} \cdot \frac{\mathbf{w}^T \mathbf{w}}{\|\mathbf{w}\|_2} = d_{\text{signed}} \|\mathbf{w}\|_2
    $$

    $$
    \mathbf{w}^T \mathbf{x}_0 + b = d_{\text{signed}} \|\mathbf{w}\|_2
    $$

    Solving for the absolute geometric Euclidean distance yields:

    $$
    d(\mathbf{x}_0, H) = \frac{|\mathbf{w}^T \mathbf{x}_0 + b|}{\|\mathbf{w}\|_2}
    $$

    ### Distance Between Two Parallel Hyperplanes

    Consider two parallel hyperplanes sharing the same normal vector $\mathbf{w}$:

    $$
    P_1: \mathbf{w}^T \mathbf{x} + b_1 = 0 \quad \text{and} \quad P_2: \mathbf{w}^T \mathbf{x} + b_2 = 0
    $$

    Pick any point $\mathbf{x}_1 \in P_1$, so $\mathbf{w}^T \mathbf{x}_1 = -b_1$. The perpendicular distance from $\mathbf{x}_1$ to $P_2$ is:

    $$
    D = d(\mathbf{x}_1, P_2) = \frac{|\mathbf{w}^T \mathbf{x}_1 + b_2|}{\|\mathbf{w}\|_2} = \frac{|-b_1 + b_2|}{\|\mathbf{w}\|_2} = \frac{|b_2 - b_1|}{\|\mathbf{w}\|_2}
    $$

    This separation distance depends purely on the difference in offsets normalized by the Euclidean norm of the weight vector.

    ### Role in ML, AI, and Statistics

    **Support Vector Machine Margins**: The hard-margin SVM defines two bounding canonical hyperplanes supporting the margin: $\mathbf{w}^T \mathbf{x} + b = 1$ and $\mathbf{w}^T \mathbf{x} + b = -1$. Applying the parallel hyperplane distance formula yields:

    $$
    \text{Margin} = \frac{|1 - (-1)|}{\|\mathbf{w}\|_2} = \frac{2}{\|\mathbf{w}\|_2}
    $$

    Maximizing this separation margin is equivalent to minimizing $\frac{1}{2}\|\mathbf{w}\|_2^2$, establishing $L_2$ regularization as a geometric imperative.

    **Logistic Regression Level Sets**: The log-odds $\ln \frac{p}{1-p} = \mathbf{w}^T \mathbf{x} + b$ defines a continuum of parallel level sets. The decision boundary $p = 0.5$ is the central hyperplane, while parallel planes $p = 0.9$ or $p = 0.99$ correspond to confidence contours whose distance from the boundary scales with $\frac{1}{\|\mathbf{w}\|_2}$.

    **Adversarial Perturbations**: For a linear decision rule, the minimal $L_2$ perturbation $\boldsymbol{\delta}$ required to flip the prediction of an input $\mathbf{x}_0$ is the vector directly pointing toward the boundary: $\boldsymbol{\delta}^* = - \frac{\mathbf{w}^T \mathbf{x}_0 + b}{\|\mathbf{w}\|_2^2} \mathbf{w}$, having length exactly equal to the point-to-plane distance $d(\mathbf{x}_0, H)$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code examples

    Below are two practical examples:
    1. **Numerical Verification**: Computing perpendicular distances from arbitrary points to a hyperplane and calculating the separation distance between parallel planes.
    2. **Interactive 3D Visualization**: Visualizing parallel hyperplanes, their shared normal vector, and the orthogonal separation distance with Plotly.
    """)
    return


@app.cell
def _(np):
    # Example 1: Analytical and numerical distance computation in R^3
    # Hyperplane 1: 2x + 3y + 4z = 0   => w = [2, 3, 4], b1 = 0
    # Hyperplane 2: 2x + 3y + 4z - 10 = 0 => w = [2, 3, 4], b2 = -10
    w = np.array([2.0, 3.0, 4.0])
    b1 = 0.0
    b2 = -10.0

    w_norm = np.linalg.norm(w)
    theoretical_distance = abs(b2 - b1) / w_norm

    # Test point on Plane 1: x1 = [0, 0, 0] since 2(0) + 3(0) + 4(0) + 0 = 0
    x_on_p1 = np.array([0.0, 0.0, 0.0])

    # Calculate distance from point x_on_p1 to Plane 2
    point_distance = abs(np.dot(w, x_on_p1) + b2) / w_norm

    # Orthogonal projection of x_on_p1 onto Plane 2
    x_proj_on_p2 = x_on_p1 - ((np.dot(w, x_on_p1) + b2) / (w_norm**2)) * w

    # Verify projected point lies exactly on Plane 2: w . x_proj + b2 == 0
    plane_2_residual = float(np.dot(w, x_proj_on_p2) + b2)
    euclidean_displacement = float(np.linalg.norm(x_proj_on_p2 - x_on_p1))

    {
        "normal_norm_sqrt29": float(w_norm),
        "theoretical_distance": float(theoretical_distance),
        "point_distance": float(point_distance),
        "projected_point_on_p2": x_proj_on_p2.tolist(),
        "plane_2_residual": plane_2_residual,
        "euclidean_displacement": euclidean_displacement
    }
    return b1, b2, w, w_norm, x_on_p1, x_proj_on_p2


@app.cell
def _(b1, b2, go, np, w, x_on_p1, x_proj_on_p2):
    # Example 2: Interactive 3D Plotly Visualization of Parallel Hyperplanes
    # Solving for z: w[0]*x + w[1]*y + w[2]*z + b = 0 => z = -(w[0]*x + w[1]*y + b) / w[2]
    x_grid = np.linspace(-4, 4, 30)
    y_grid = np.linspace(-4, 4, 30)
    X, Y = np.meshgrid(x_grid, y_grid)

    Z1 = -(w[0] * X + w[1] * Y + b1) / w[2]
    Z2 = -(w[0] * X + w[1] * Y + b2) / w[2]

    fig = go.Figure()

    # Plane 1: 2x + 3y + 4z = 0
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z1,
        colorscale=[[0, "rgba(31, 119, 180, 0.5)"], [1, "rgba(31, 119, 180, 0.5)"]],
        showscale=False,
        name="Plane 1: 2x + 3y + 4z = 0"
    ))

    # Plane 2: 2x + 3y + 4z - 10 = 0
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z2,
        colorscale=[[0, "rgba(214, 39, 40, 0.5)"], [1, "rgba(214, 39, 40, 0.5)"]],
        showscale=False,
        name="Plane 2: 2x + 3y + 4z = 10"
    ))

    # Normal vector connecting the origin to normal direction
    unit_w = w / np.linalg.norm(w)
    fig.add_trace(go.Scatter3d(
        x=[x_on_p1[0], x_proj_on_p2[0]],
        y=[x_on_p1[1], x_proj_on_p2[1]],
        z=[x_on_p1[2], x_proj_on_p2[2]],
        mode="lines+markers",
        name="Separation Segment (D ≈ 1.86)",
        line=dict(color="#2ca02c", width=8),
        marker=dict(size=5, color=["#1f77b4", "#d62728"])
    ))

    fig.update_layout(
        title=dict(
            text="Parallel Hyperplanes in R³ and Orthogonal Separation Distance",
            font=dict(size=15)
        ),
        scene=dict(
            xaxis=dict(title="X₁", gridcolor="#e5e5e5"),
            yaxis=dict(title="X₂", gridcolor="#e5e5e5"),
            zaxis=dict(title="X₃", gridcolor="#e5e5e5"),
            camera=dict(
                eye=dict(x=1.6, y=1.6, z=1.2)
            )
        ),
        template="plotly_white",
        width=780,
        height=580,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    * **Affine Structure**: A hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ splits feature space into two half-spaces. The vector $\mathbf{w}$ is strictly perpendicular to every displacement within the plane.
    * **Distance Formula**: The geometric distance from any point $\mathbf{x}_0$ to a hyperplane is $\frac{|\mathbf{w}^T \mathbf{x}_0 + b|}{\|\mathbf{w}\|_2}$. Dividing by the norm ensures metric invariance under arbitrary scalar rescalings of $\mathbf{w}$ and $b$.
    * **Parallel Separation**: The distance between parallel hyperplanes $\mathbf{w}^T \mathbf{x} + b_1 = 0$ and $\mathbf{w}^T \mathbf{x} + b_2 = 0$ is $D = \frac{|b_1 - b_2|}{\|\mathbf{w}\|_2}$.
    * **SVM Margin Maximization**: In support vector machines, setting canonical hyperplanes at $\pm 1$ produces a margin of $\frac{2}{\|\mathbf{w}\|_2}$, proving why maximizing geometric separation is equivalent to minimizing the $L_2$ weight norm $\|\mathbf{w}\|_2$.

    ---

    &larr; Previous Note: [02 Norm and Metric](02_norm_and_metric.py) | Next Note: [04 Rank-One Matrices](04_rank_one_matrices.py) &rarr;
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
