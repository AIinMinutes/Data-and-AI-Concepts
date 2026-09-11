import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 06: Moore-Penrose Pseudoinverse, Least-Squares, and Minimum-Norm Solutions

    &larr; Previous Note: [05 Orthogonality](05_orthogonality.py) | Next Note: [07 Spectral Decomposition](07_spectral_decomposition.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    When solving a linear system $\mathbf{A}\mathbf{x} = \mathbf{b}$, a square, full-rank matrix yields a unique solution via the standard inverse $\mathbf{x} = \mathbf{A}^{-1}\mathbf{b}$. In practical machine learning, however, matrices are rarely square and invertible. Data matrices are typically tall (overdetermined, with more observations than features) or wide (underdetermined, with more parameters than observations).

    The **Moore-Penrose pseudoinverse** $\mathbf{A}^+$ generalizes matrix inversion to any rectangular or rank-deficient matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$. It guarantees a uniquely defined, optimal solution:
    1. For overdetermined systems, it provides the **least-squares solution** minimizing residual error $\|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2$.
    2. For underdetermined systems, it isolates the unique **minimum-norm solution** minimizing $\|\mathbf{x}\|_2$ among all exact interpolators.
    3. For overparameterized deep models ($p \gg n$), gradient descent implicitly converges to this pseudoinverse solution, providing the foundation for understanding double descent.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### The Four Penrose Conditions

    For any matrix $\mathbf{A} \in \mathbb{R}^{m \times n}$, there exists a unique matrix $\mathbf{A}^+ \in \mathbb{R}^{n \times m}$ satisfying the four Moore-Penrose conditions:

    $$
    \mathbf{A} \mathbf{A}^+ \mathbf{A} = \mathbf{A}
    $$

    $$
    \mathbf{A}^+ \mathbf{A} \mathbf{A}^+ = \mathbf{A}^+
    $$

    $$
    (\mathbf{A} \mathbf{A}^+)^T = \mathbf{A} \mathbf{A}^+
    $$

    $$
    (\mathbf{A}^+ \mathbf{A})^T = \mathbf{A}^+ \mathbf{A}
    $$

    Conditions 3 and 4 have fundamental geometric interpretations:
    * $\mathbf{P}_{\text{col}(\mathbf{A})} = \mathbf{A}\mathbf{A}^+$ is the symmetric, orthogonal projection operator onto the column space of $\mathbf{A}$.
    * $\mathbf{P}_{\text{row}(\mathbf{A})} = \mathbf{A}^+\mathbf{A}$ is the symmetric, orthogonal projection operator onto the row space of $\mathbf{A}$.

    ---

    ### Closed-Form Expressions for Full-Rank Rectangular Cases

    When $\mathbf{A}$ has full rank, the pseudoinverse takes explicit algebraic forms:

    #### Full Column Rank ($m > n$, Overdetermined)

    The Gram matrix $\mathbf{A}^T \mathbf{A} \in \mathbb{R}^{n \times n}$ is invertible. The pseudoinverse is the **left inverse**:

    $$
    \mathbf{A}^+ = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T
    $$

    Notice that $\mathbf{A}^+ \mathbf{A} = (\mathbf{A}^T \mathbf{A})^{-1} \mathbf{A}^T \mathbf{A} = \mathbf{I}_n$. The solution $\hat{\mathbf{x}} = \mathbf{A}^+ \mathbf{b}$ minimizes the sum of squared residuals:

    $$
    \min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{A}\mathbf{x} - \mathbf{b}\|_2^2
    $$

    #### Full Row Rank ($m < n$, Underdetermined)

    The matrix $\mathbf{A} \mathbf{A}^T \in \mathbb{R}^{m \times m}$ is invertible. The pseudoinverse is the **right inverse**:

    $$
    \mathbf{A}^+ = \mathbf{A}^T (\mathbf{A} \mathbf{A}^T)^{-1}
    $$

    Notice that $\mathbf{A} \mathbf{A}^+ = \mathbf{A} \mathbf{A}^T (\mathbf{A} \mathbf{A}^T)^{-1} = \mathbf{I}_m$. The system $\mathbf{A}\mathbf{x} = \mathbf{b}$ has infinitely many solutions, and $\hat{\mathbf{x}} = \mathbf{A}^+ \mathbf{b}$ selects the unique solution with the smallest Euclidean norm:

    $$
    \min_{\mathbf{x} \in \mathbb{R}^n} \|\mathbf{x}\|_2 \quad \text{subject to} \quad \mathbf{A}\mathbf{x} = \mathbf{b}
    $$

    ---

    ### General Construction via Singular Value Decomposition (SVD)

    For arbitrary rank $r \leq \min(m, n)$, the pseudoinverse is computed universally via the Singular Value Decomposition $\mathbf{A} = \mathbf{U} \mathbf{\Sigma} \mathbf{V}^T$:

    $$
    \mathbf{A}^+ = \mathbf{V} \mathbf{\Sigma}^+ \mathbf{U}^T
    $$

    where $\mathbf{\Sigma}^+ \in \mathbb{R}^{n \times m}$ is obtained by reciprocating non-zero singular values and transposing:

    $$
    \sigma_i^+ = \begin{cases} \frac{1}{\sigma_i} & \text{if } \sigma_i > 0 \\ 0 & \text{if } \sigma_i = 0 \end{cases}
    $$

    ---

    ### Implicit Bias in Overparameterized Models and Double Descent

    In modern deep learning architectures, the number of parameters $p$ frequently exceeds the number of training samples $n$ ($p \gg n$). When training an overparameterized linear model on a regression task using gradient descent initialized at zero:

    $$
    \mathbf{w}_{t+1} = \mathbf{w}_t - \eta \mathbf{X}^T (\mathbf{X}\mathbf{w}_t - \mathbf{y}), \quad \mathbf{w}_0 = \mathbf{0}
    $$

    Because all gradient updates lie in the span of the data rows $\text{row}(\mathbf{X})$, gradient descent converges to the unique minimum $L_2$-norm interpolating solution:

    $$
    \mathbf{w}^* = \mathbf{X}^+ \mathbf{y}
    $$

    This implicit regularization explains why modern overparameterized models can interpolate training data without catastrophic overfitting, a phenomenon linked to benign overfitting and double descent.

    ---

    ### Ridge Regularization Limit (Tikhonov Regularization)

    To address ill-conditioned or rank-deficient systems, ridge regression introduces an $L_2$ penalty:

    $$
    \mathbf{w}_\lambda = (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T \mathbf{y}
    $$

    As the regularization parameter vanishes ($\lambda \to 0^+$), the ridge solution converges continuously to the Moore-Penrose pseudoinverse solution:

    $$
    \lim_{\lambda \to 0^+} (\mathbf{X}^T \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^T = \mathbf{X}^+
    $$

    This establishes that unregularized pseudoinverse interpolation is the structural limit of $L_2$ regularization.

    ---

    ### Linear Probing of Frozen Representations

    When evaluating foundation models (such as CLIP, BERT, or DINO), researchers freeze the pre-trained feature extractor and train a linear classification or regression head. The optimal weights $\mathbf{W}^*$ can be found in closed form via the pseudoinverse:

    $$
    \mathbf{W}^* = \mathbf{Z}^+ \mathbf{Y}
    $$

    where $\mathbf{Z} \in \mathbb{R}^{N \times d}$ contains the extracted embeddings and $\mathbf{Y} \in \mathbb{R}^{N \times C}$ represents the target labels.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Interactive Visualizations: Overdetermined vs Underdetermined Systems

    The two figures below illustrate the geometric duality of the pseudoinverse:
    1. **Overdetermined System (3D)**: Target vector $\mathbf{b}$ projected orthogonally onto the 2D column space $\text{col}(\mathbf{A})$. The residual $\mathbf{r} = \mathbf{b} - \mathbf{A}\mathbf{x}^+$ is strictly perpendicular to the plane.
    2. **Underdetermined System (2D)**: The affine line of valid solutions $\mathbf{A}\mathbf{x} = \mathbf{b}$. The pseudoinverse selects $\mathbf{x}^+ = \mathbf{A}^+ \mathbf{b}$, which is the unique point on the line closest to the origin.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Panel 1: Overdetermined System (3 equations, 2 variables)
    # A has 2 columns in R^3, b is outside the column space
    a1 = np.array([1.5, 0.2, 0.5])
    a2 = np.array([0.3, 1.4, 0.4])
    a_over = np.column_stack([a1, a2])
    b_over = np.array([1.8, 1.6, 2.2])

    # Least squares projection: x_hat = pinv(A) @ b
    x_hat = np.linalg.pinv(a_over) @ b_over
    b_proj = a_over @ x_hat
    residual = b_over - b_proj

    # Generate grid for the column space plane
    u_grid = np.linspace(-0.2, 1.5, 15)
    v_grid = np.linspace(-0.2, 1.5, 15)
    u_mesh, v_mesh = np.meshgrid(u_grid, v_grid)
    plane_x = a1[0] * u_mesh + a2[0] * v_mesh
    plane_y = a1[1] * u_mesh + a2[1] * v_mesh
    plane_z = a1[2] * u_mesh + a2[2] * v_mesh

    # 3D Figure for Overdetermined System
    fig_3d = go.Figure()

    # Column space surface
    fig_3d.add_trace(
        go.Surface(
            x=plane_x,
            y=plane_y,
            z=plane_z,
            opacity=0.45,
            colorscale=[[0, "#93c5fd"], [1, "#3b82f6"]],
            showscale=False,
            name="col(A) Plane",
        )
    )

    # Basis vector a1
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0, a1[0]],
            y=[0, a1[1]],
            z=[0, a1[2]],
            mode="lines+markers",
            line=dict(color="#2563eb", width=6),
            marker=dict(size=4),
            name="Basis a1",
        )
    )

    # Basis vector a2
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0, a2[0]],
            y=[0, a2[1]],
            z=[0, a2[2]],
            mode="lines+markers",
            line=dict(color="#06b6d4", width=6),
            marker=dict(size=4),
            name="Basis a2",
        )
    )

    # Target vector b
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0, b_over[0]],
            y=[0, b_over[1]],
            z=[0, b_over[2]],
            mode="lines+markers",
            line=dict(color="#dc2626", width=7),
            marker=dict(size=5, color="#dc2626"),
            name="Target b (Inconsistent)",
        )
    )

    # Projected vector b_hat = A x_hat
    fig_3d.add_trace(
        go.Scatter3d(
            x=[0, b_proj[0]],
            y=[0, b_proj[1]],
            z=[0, b_proj[2]],
            mode="lines+markers",
            line=dict(color="#16a34a", width=7),
            marker=dict(size=5, color="#16a34a"),
            name="Projection b_hat = A x^+",
        )
    )

    # Orthogonal residual vector connecting b_proj to b_over
    fig_3d.add_trace(
        go.Scatter3d(
            x=[b_proj[0], b_over[0]],
            y=[b_proj[1], b_over[1]],
            z=[b_proj[2], b_over[2]],
            mode="lines+markers",
            line=dict(color="#d97706", width=5, dash="dash"),
            marker=dict(size=4, color="#d97706"),
            name="Residual r = b - A x^+",
        )
    )

    fig_3d.update_layout(
        template="plotly_white",
        title="Overdetermined System: Least-Squares Orthogonal Projection onto col(A)",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            camera=dict(eye=dict(x=1.6, y=-1.5, z=1.2)),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
    )

    # Panel 2: Underdetermined System (1 equation, 2 unknowns: 2*x1 + 3*x2 = 6)
    a_under = np.array([[2.0, 3.0]])
    b_val = 6.0
    x_min_norm = np.linalg.pinv(a_under) @ np.array([b_val])

    x1_vals = np.linspace(-1.0, 4.0, 200)
    x2_vals = (b_val - a_under[0, 0] * x1_vals) / a_under[0, 1]

    # Sample alternate feasible points
    sample_x1 = np.array([0.0, 1.5, 3.0])
    sample_x2 = (b_val - a_under[0, 0] * sample_x1) / a_under[0, 1]

    fig_2d = go.Figure()

    # Solution line
    fig_2d.add_trace(
        go.Scatter(
            x=x1_vals,
            y=x2_vals,
            mode="lines",
            line=dict(color="#2563eb", width=3),
            name="Affine Solution Set: 2x_1 + 3x_2 = 6",
        )
    )

    # Origin
    fig_2d.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(color="#64748b", size=9),
            name="Origin (0, 0)",
        )
    )

    # Orthogonal segment from origin to minimum-norm solution
    fig_2d.add_trace(
        go.Scatter(
            x=[0, x_min_norm[0]],
            y=[0, x_min_norm[1]],
            mode="lines+markers",
            line=dict(color="#16a34a", width=4),
            marker=dict(color="#16a34a", size=9),
            name=f"Min-Norm Solution x^+: [{x_min_norm[0]:.2f}, {x_min_norm[1]:.2f}]",
        )
    )

    # Alternate feasible points
    fig_2d.add_trace(
        go.Scatter(
            x=sample_x1,
            y=sample_x2,
            mode="markers",
            marker=dict(color="#d97706", size=8, symbol="diamond"),
            name="Alternative Feasible Solutions",
        )
    )

    # Dashed segments from origin to alternative solutions
    for sx, sy in zip(sample_x1, sample_x2):
        fig_2d.add_trace(
            go.Scatter(
                x=[0, sx],
                y=[0, sy],
                mode="lines",
                line=dict(color="#cbd5e1", dash="dot", width=1.5),
                showlegend=False,
            )
        )

    norm_min = float(np.linalg.norm(x_min_norm))
    fig_2d.update_layout(
        template="plotly_white",
        title=f"Underdetermined System: Minimum-Norm Solution (||x^+||_2 = {norm_min:.3f})",
        height=450,
        margin=dict(l=40, r=40, t=50, b=40),
        xaxis=dict(range=[-1.0, 4.0], zeroline=True, zerolinecolor="#cbd5e1", gridcolor="#f1f5f9"),
        yaxis=dict(range=[-1.0, 3.0], zeroline=True, zerolinecolor="#cbd5e1", gridcolor="#f1f5f9"),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    return (
        a1,
        a2,
        a_over,
        a_under,
        b_over,
        b_proj,
        b_val,
        fig_2d,
        fig_3d,
        norm_min,
        residual,
        x_hat,
        x_min_norm,
    )


@app.cell
def _(fig_3d, mo):
    mo.ui.plotly(fig_3d)
    return


@app.cell
def _(fig_2d, mo):
    mo.ui.plotly(fig_2d)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    ### Example 1: Numerical Verification of the Four Penrose Conditions

    Below, we construct a rank-deficient rectangular matrix $\mathbf{A} \in \mathbb{R}^{4 \times 3}$ with rank 2. We construct its pseudoinverse using both SVD and NumPy's `np.linalg.pinv`, verifying:
    1. $\mathbf{A}\mathbf{A}^+\mathbf{A} = \mathbf{A}$
    2. $\mathbf{A}^+\mathbf{A}\mathbf{A}^+ = \mathbf{A}^+$
    3. $(\mathbf{A}\mathbf{A}^+)^T = \mathbf{A}\mathbf{A}^+$
    4. $(\mathbf{A}^+\mathbf{A})^T = \mathbf{A}^+\mathbf{A}$
    5. Agreement between SVD-derived pseudoinverse and `np.linalg.pinv`
    """)
    return


@app.cell
def _(np):
    # Seed for reproducibility
    rng = np.random.default_rng(42)

    # Construct rank-deficient 4x3 matrix with rank 2
    basis_cols = rng.standard_normal((4, 2))
    third_col = 0.5 * basis_cols[:, 0] + 0.8 * basis_cols[:, 1]
    a_mat = np.column_stack([basis_cols, third_col])

    # Method 1: SVD construction of pseudoinverse
    u, s, vt = np.linalg.svd(a_mat, full_matrices=False)
    tol = 1e-12
    s_inv = np.array([1.0 / val if val > tol else 0.0 for val in s])
    pinv_svd = vt.T @ np.diag(s_inv) @ u.T

    # Method 2: NumPy pinv
    pinv_numpy = np.linalg.pinv(a_mat)

    # 1. Condition 1: A A+ A == A
    c1_err = float(np.max(np.abs(a_mat @ pinv_svd @ a_mat - a_mat)))

    # 2. Condition 2: A+ A A+ == A+
    c2_err = float(np.max(np.abs(pinv_svd @ a_mat @ pinv_svd - pinv_svd)))

    # 3. Condition 3: (A A+)^T == A A+
    aa_plus = a_mat @ pinv_svd
    c3_err = float(np.max(np.abs(aa_plus.T - aa_plus)))

    # 4. Condition 4: (A+ A)^T == A+ A
    a_plus_a = pinv_svd @ a_mat
    c4_err = float(np.max(np.abs(a_plus_a.T - a_plus_a)))

    # 5. SVD vs NumPy discrepancy
    svd_vs_numpy_err = float(np.max(np.abs(pinv_svd - pinv_numpy)))

    penrose_results = {
        "Penrose Condition / Verification": [
            "1. A A+ A == A (max error)",
            "2. A+ A A+ == A+ (max error)",
            "3. (A A+)^T == A A+ (symmetry error)",
            "4. (A+ A)^T == A+ A (symmetry error)",
            "Manual SVD vs np.linalg.pinv discrepancy",
            "Matrix rank",
            "Matrix shape (m, n)",
        ],
        "Value": [
            f"{c1_err:.2e}",
            f"{c2_err:.2e}",
            f"{c3_err:.2e}",
            f"{c4_err:.2e}",
            f"{svd_vs_numpy_err:.2e}",
            str(np.linalg.matrix_rank(a_mat)),
            f"{a_mat.shape[0]} x {a_mat.shape[1]}",
        ],
    }

    return (
        a_mat,
        a_plus_a,
        aa_plus,
        basis_cols,
        c1_err,
        c2_err,
        c3_err,
        c4_err,
        penrose_results,
        pinv_numpy,
        pinv_svd,
        s,
        s_inv,
        svd_vs_numpy_err,
        third_col,
        u,
        vt,
    )


@app.cell(hide_code=True)
def _(mo, pd, penrose_results):
    df_penrose = pd.DataFrame(penrose_results)
    mo.ui.table(df_penrose)
    return (df_penrose,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: Minimum-Norm Verification for Underdetermined Systems

    Consider an underdetermined system with $m = 3$ equations and $n = 10$ parameters. Infinitely many solutions $\mathbf{w}$ satisfy $\mathbf{X}\mathbf{w} = \mathbf{y}$.

    Any feasible solution can be parameterized as:

    $$
    \mathbf{w} = \mathbf{w}_{\text{pinv}} + \mathbf{n}
    $$

    where $\mathbf{n} \in \text{null}(\mathbf{X})$ is a vector in the null space. Because $\mathbf{w}_{\text{pinv}} \in \text{row}(\mathbf{X})$ and $\text{row}(\mathbf{X}) \perp \text{null}(\mathbf{X})$, the Pythagorean theorem gives:

    $$
    \|\mathbf{w}\|_2^2 = \|\mathbf{w}_{\text{pinv}}\|_2^2 + \|\mathbf{n}\|_2^2 \geq \|\mathbf{w}_{\text{pinv}}\|_2^2
    $$

    Below, we sample 1,000 random feasible solutions and numerically verify that $\mathbf{w}_{\text{pinv}}$ achieves strictly minimal $L_2$ norm.
    """)
    return


@app.cell
def _(np):
    rng_sample = np.random.default_rng(123)

    # 3 samples, 10 features (underdetermined)
    x_data = rng_sample.standard_normal((3, 10))
    y_data = rng_sample.standard_normal(3)

    # Pseudoinverse minimum-norm solution
    w_pinv = np.linalg.pinv(x_data) @ y_data
    norm_pinv = float(np.linalg.norm(w_pinv))

    # Compute null space basis using SVD
    _, _, vt_null = np.linalg.svd(x_data)
    # The last 7 right singular vectors span the null space
    null_basis = vt_null[3:, :].T  # shape: (10, 7)

    # Sample 1,000 random vectors in the null space
    num_samples = 1000
    random_coeffs = rng_sample.standard_normal((7, num_samples))
    null_perturbations = null_basis @ random_coeffs  # shape: (10, 1000)

    # Construct feasible solutions: w = w_pinv + null_vector
    feasible_solutions = w_pinv[:, np.newaxis] + null_perturbations

    # Verify all samples satisfy X w = y
    residuals = np.max(np.abs(x_data @ feasible_solutions - y_data[:, np.newaxis]))

    # Compute norms of all alternative feasible solutions
    sample_norms = np.linalg.norm(feasible_solutions, axis=0)
    min_sampled_norm = float(np.min(sample_norms))
    avg_sampled_norm = float(np.mean(sample_norms))
    max_sampled_norm = float(np.max(sample_norms))

    # All samples must have norm greater than or equal to w_pinv
    all_greater = bool(np.all(sample_norms >= norm_pinv - 1e-12))

    underdetermined_summary = {
        "Metric": [
            "Dataset dimension (m x n)",
            "Null space dimension (n - m)",
            "Number of sampled feasible solutions",
            "Maximum interpolation error max |X w - y|",
            "Minimum-norm solution ||w_pinv||_2",
            "Lowest norm among sampled solutions",
            "Average norm among sampled solutions",
            "Highest norm among sampled solutions",
            "Is ||w_pinv||_2 <= ||w_sample||_2 for all 1,000 samples?",
        ],
        "Value": [
            f"{x_data.shape[0]} x {x_data.shape[1]}",
            str(null_basis.shape[1]),
            str(num_samples),
            f"{residuals:.2e}",
            f"{norm_pinv:.6f}",
            f"{min_sampled_norm:.6f}",
            f"{avg_sampled_norm:.6f}",
            f"{max_sampled_norm:.6f}",
            str(all_greater),
        ],
    }

    return (
        all_greater,
        avg_sampled_norm,
        feasible_solutions,
        max_sampled_norm,
        min_sampled_norm,
        norm_pinv,
        null_basis,
        null_perturbations,
        num_samples,
        random_coeffs,
        residuals,
        sample_norms,
        underdetermined_summary,
        vt_null,
        w_pinv,
        x_data,
        y_data,
    )


@app.cell(hide_code=True)
def _(mo, pd, underdetermined_summary):
    df_under = pd.DataFrame(underdetermined_summary)
    mo.ui.table(df_under)
    return (df_under,)


if __name__ == "__main__":
    app.run()
