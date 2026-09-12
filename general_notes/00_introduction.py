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
    mo.Html(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=Fira+Code&display=swap');

        /* Core Typography */
        html, body, .marimo {
            font-family: 'Inter', sans-serif !important;
            background-color: #ffffff !important;
            color: #1a1a1a !important;
            line-height: 1.75 !important;
        }

        /* Headings */
        h1, h2, h3, h4 {
            font-weight: 600 !important;
            color: #000000 !important;
            letter-spacing: -0.02em !important;
        }
        h1 { font-size: 2.25rem !important; margin-bottom: 1.5rem !important; }
        h2 { font-size: 1.5rem !important; margin-top: 3rem !important; }

        /* Code blocks */
        pre, code {
            font-family: 'Fira Code', monospace !important;
            font-size: 0.9em !important;
        }

        /* Whitespace and margins for readability */
        .prose {
            max-width: 65ch !important;
            margin: 0 auto !important;
        }
        </style>
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 00: Systems of Linear Equations and Vector Foundations

    Next Note: [01 Inner Products](01_inner_product.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Linear equations form the computational bedrock of data science, machine learning, and statistical modeling. Real-world phenomena frequently involve multiple interdependent variables constrained by simultaneous observations.

    Understanding how to express these relationships with vectors and matrices allows you to:

    1. Formulate complex multidimensional problems in a compact mathematical language.
    2. Determine whether a unique solution exists, whether multiple solutions exist, or whether the system is inconsistent.
    3. Compute unknown parameters efficiently using standard linear algebra routines.
    4. Gain geometric intuition for how data points, hyperplanes, and transformations behave in vector spaces.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Vector Representation of Quantities

    Consider a simple 2D coordinate system tracking two distinct quantities:
    * $x$: Number of apples
    * $y$: Number of bananas

    Using vector notation, we represent individual units along coordinate axes:

    $$
    \mathbf{v}_{\text{apple}} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}, \quad
    \mathbf{v}_{\text{banana}} = \begin{bmatrix} 0 \\ 1 \end{bmatrix}, \quad
    \mathbf{v}_{\text{combo}} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}
    $$

    When a vendor packages goods in fixed proportions, say two apples for every one banana, the combo follows a directional ray through the origin:

    $$
    \mathbf{v}_{\text{combo}} = \alpha \begin{bmatrix} 2 \\ 1 \end{bmatrix}, \quad \alpha \in \mathbb{R}
    $$

    ### Pricing and Matrix Representation

    While quantities count discrete items, prices provide a scalar weighting of value. Let:
    * $x_1$: Price per apple
    * $x_2$: Price per banana

    For a bundle containing $a_1$ apples and $b_1$ bananas, the total cost $c_1$ is a linear combination:

    $$
    c_1 = a_1 x_1 + b_1 x_2
    $$

    When we observe two distinct purchase scenarios:
    1. $a_1$ apples and $b_1$ bananas sold for total cost $c_1$
    2. $a_2$ apples and $b_2$ bananas sold for total cost $c_2$

    We can stack these relationships into a single matrix equation:

    $$
    \begin{bmatrix} x_1 & x_2 \end{bmatrix}
    \begin{bmatrix} a_1 & a_2 \\ b_1 & b_2 \end{bmatrix}
    = \begin{bmatrix} c_1 & c_2 \end{bmatrix}
    $$

    In row-vector notation:

    $$
    \mathbf{x}_{\text{row}} \mathbf{A} = \mathbf{c}_{\text{row}}
    $$

    Taking the transpose yields the conventional column-vector formulation:

    $$
    \mathbf{A}^T \mathbf{x} = \mathbf{c}^T
    $$

    ### Solvability and Determinants

    A square linear system has a unique solution if and only if the coefficient matrix is invertible, which requires a non-zero determinant:

    $$
    \det(\mathbf{A}) \neq 0
    $$

    If $\det(\mathbf{A}) = 0$, the rows (or columns) are linearly dependent. In that case:
    * The system has infinitely many solutions if the target vector lies within the column span.
    * The system has no solution if the target vector lies outside the column span.

    ### Role in ML, AI, and Statistics

    * **Parameter Estimation**: In supervised learning, linear models define predictions as linear combinations of feature vectors. Solving for optimal weights involves solving or approximating a system of linear equations.
    * **Coordinate Transformations**: Every linear neural network layer performs a matrix-vector product, transforming representations from one coordinate basis to another.
    * **Solvability and Multicollinearity**: In regression analysis, highly correlated predictors lead to near-singular matrices with determinants close to zero, causing numerical instability. Understanding linear independence is essential to diagnosing and resolving multicollinearity.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    ### Example 1: Solving a 2x2 System (Unique Solution)

    A vendor observes two purchase scenarios: (40 apples, 6 bananas) totaling 100 currency units, and (20 apples, 8 bananas) totaling 80 currency units. We solve for the unknown per-unit prices $x_1$ (apple) and $x_2$ (banana) using the matrix equation $\mathbf{A}^T \mathbf{x} = \mathbf{c}$, verify invertibility via the determinant, and visualize the constraint lines whose intersection is the unique solution.
    """)
    return


@app.cell
def _(A, c, np):
    # Ensure matrix A is invertible by checking its determinant
    det_A = np.linalg.det(A)
    if det_A == 0:
        raise ValueError("Matrix A is singular and cannot be inverted")

    # Solve the system of linear equations: x_row @ A = c_row <=> A.T @ x = c
    # Using np.linalg.solve on the transposed system is numerically preferred
    x_solution = np.linalg.solve(A.T, c)

    # Verification via matrix inversion
    x_inv = c @ np.linalg.inv(A)

    {"determinant": float(det_A), "solution_solve": x_solution.tolist(), "solution_inv": x_inv.tolist()}
    return (x_solution,)


@app.cell
def _(np):
    # Coefficient matrix A (quantity matrix)
    # Row 1: scenario 1 (40 apples, 20 bananas)
    # Row 2: scenario 2 (6 apples, 8 bananas)
    A = np.array([[40.0, 20.0], [6.0, 8.0]])

    # Target vector c (total costs for each scenario)
    c = np.array([100.0, 80.0])
    return A, c


@app.cell
def _(A, c, go, np, x_solution):
    # Compute lines for visualization
    # Equation 1: A[0, 0]*x1 + A[1, 0]*x2 = c[0] => 40*x1 + 6*x2 = 100
    # Equation 2: A[0, 1]*x1 + A[1, 1]*x2 = c[1] => 20*x1 + 8*x2 = 80
    x1_range = np.linspace(0, 4, 200)
    x2_eq1 = (c[0] - A[0, 0] * x1_range) / A[1, 0]
    x2_eq2 = (c[1] - A[0, 1] * x1_range) / A[1, 1]

    fig = go.Figure()

    # Scenario 1 constraint line
    fig.add_trace(
        go.Scatter(
            x=x1_range,
            y=x2_eq1,
            mode="lines",
            name="Scenario 1: 40x₁ + 6x₂ = 100",
            line=dict(color="#1f77b4", width=2.5),
        )
    )

    # Scenario 2 constraint line
    fig.add_trace(
        go.Scatter(
            x=x1_range,
            y=x2_eq2,
            mode="lines",
            name="Scenario 2: 20x₁ + 8x₂ = 80",
            line=dict(color="#ff7f0e", width=2.5),
        )
    )

    # Solution point (intersection)
    fig.add_trace(
        go.Scatter(
            x=[x_solution[0]],
            y=[x_solution[1]],
            mode="markers+text",
            name=f"Solution: ({x_solution[0]:.2f}, {x_solution[1]:.2f})",
            text=[f"  Solution ({x_solution[0]:.2f}, {x_solution[1]:.2f})"],
            textposition="top right",
            marker=dict(color="#d62728", size=11, symbol="circle"),
        )
    )

    fig.update_layout(
        title=dict(text="Systems of Linear Equations: Constraint Lines and Unique Solution", font=dict(size=14)),
        xaxis=dict(title="Price per Apple (x₁)", range=[0, 3.5], zeroline=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Price per Banana (x₂)", range=[0, 12], zeroline=True, gridcolor="#e5e5e5"),
        template="plotly_white",
        legend=dict(x=0.55, y=0.98),
        width=720,
        height=480,
    )

    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Overdetermined System (Least-Squares Approximation)

    In practice, we often have more observations (equations) than unknowns. A vendor records **three** purchase scenarios but there are still only two unknown prices. The system $\mathbf{A}^T \mathbf{x} = \mathbf{c}$ is now overdetermined and generally has no exact solution.

    The **least-squares solution** minimizes the squared residual $\|\mathbf{A}^T \mathbf{x} - \mathbf{c}\|_2^2$. This is the foundation of Ordinary Least Squares (OLS) regression:

    $$
    \hat{\mathbf{x}} = (\mathbf{A} \mathbf{A}^T)^{-1} \mathbf{A} \, \mathbf{c}
    $$

    NumPy's `np.linalg.lstsq` computes this via the numerically stable SVD decomposition rather than forming the normal equations explicitly.
    """)
    return


@app.cell
def _(np):
    # Three purchase scenarios (overdetermined: 3 equations, 2 unknowns)
    # Scenario 1: 40 apples, 6 bananas = 100
    # Scenario 2: 20 apples, 8 bananas = 80
    # Scenario 3: 10 apples, 3 bananas = 49 (slightly noisy observation)
    A_over = np.array([[40.0, 20.0, 10.0], [6.0, 8.0, 3.0]])
    c_over = np.array([100.0, 80.0, 49.0])

    # Least-squares solution via SVD
    x_lstsq, residuals, rank, sv = np.linalg.lstsq(A_over.T, c_over, rcond=None)

    # Normal equations solution for comparison
    x_normal = np.linalg.solve(A_over @ A_over.T, A_over @ c_over)

    {
        "lstsq_solution": x_lstsq.tolist(),
        "normal_eq_solution": x_normal.tolist(),
        "residual_norm_squared": float(np.sum((A_over.T @ x_lstsq - c_over) ** 2)),
        "matrix_rank": int(rank),
        "match": bool(np.allclose(x_lstsq, x_normal)),
    }
    return


@app.cell
def _(A_over, c_over, go, np):
    # Least-squares solution for plotting
    _x_ls, _, _, _ = np.linalg.lstsq(A_over.T, c_over, rcond=None)

    x1_grid = np.linspace(0, 5, 200)

    fig_over = go.Figure()

    labels = [
        f"Scenario 1: {A_over[0,0]:.0f}x\u2081 + {A_over[1,0]:.0f}x\u2082 = {c_over[0]:.0f}",
        f"Scenario 2: {A_over[0,1]:.0f}x\u2081 + {A_over[1,1]:.0f}x\u2082 = {c_over[1]:.0f}",
        f"Scenario 3: {A_over[0,2]:.0f}x\u2081 + {A_over[1,2]:.0f}x\u2082 = {c_over[2]:.0f}",
    ]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    for k in range(3):
        x2_line = (c_over[k] - A_over[0, k] * x1_grid) / A_over[1, k]
        fig_over.add_trace(
            go.Scatter(
                x=x1_grid, y=x2_line, mode="lines",
                name=labels[k], line=dict(color=colors[k], width=2.5),
            )
        )

    fig_over.add_trace(
        go.Scatter(
            x=[_x_ls[0]], y=[_x_ls[1]], mode="markers+text",
            name=f"Least-Squares Solution ({_x_ls[0]:.2f}, {_x_ls[1]:.2f})",
            text=[f"  LS Solution ({_x_ls[0]:.2f}, {_x_ls[1]:.2f})"],
            textposition="top right",
            marker=dict(color="#d62728", size=11, symbol="diamond"),
        )
    )

    fig_over.update_layout(
        title=dict(
            text="Overdetermined System: Three Constraints, Least-Squares Compromise",
            font=dict(size=14),
        ),
        xaxis=dict(title="Price per Apple (x\u2081)", range=[0, 5], zeroline=True, gridcolor="#e5e5e5"),
        yaxis=dict(title="Price per Banana (x\u2082)", range=[0, 18], zeroline=True, gridcolor="#e5e5e5"),
        template="plotly_white",
        legend=dict(x=0.45, y=0.98),
        width=720,
        height=480,
    )

    fig_over
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    * **Geometric Meaning**: Each linear equation represents a hyperplane of constraints. Solving the system corresponds to finding the intersection point satisfying all constraints simultaneously.
    * **Matrix Formulation**: Grouping quantities and observations into matrices turns a set of simultaneous equations into a compact matrix multiplication problem.
    * **Invertibility and Determinants**: A unique solution exists if and only if the matrix determinant is non-zero, indicating that the equations are linearly independent.
    * **Overdetermined Systems**: When there are more equations than unknowns, the system is generally inconsistent. The least-squares solution minimizes the total squared error across all constraints, forming the mathematical foundation of linear regression.
    * **Computational Practice**: While matrix inversion ($\mathbf{A}^{-1}$) is theoretically convenient, numerical solvers like `np.linalg.solve` and `np.linalg.lstsq` should always be preferred in practice for speed and stability.

    ---

    Next Note: [01 Inner Products](01_inner_product.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
