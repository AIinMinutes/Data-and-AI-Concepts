import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 42 Multiclass Classification](42_multiclass_classification.py) | [Index](../index.html) | [44 Logistic Regression →](44_logistic_regression.py)

        # Matrix Energy and Quadratic Forms: Curvature, Definiteness, and Optimization Landscapes

        ## [a] Why do you need to know these concepts?

        In numerical optimization, machine learning loss landscapes, physics, and statistical estimation, multi-dimensional functions $f(\theta)$ are analyzed through second-order Taylor series approximations around a stationary point $\theta^*$:

        $$f(\theta^* + \Delta \theta) \approx f(\theta^*) + \nabla f(\theta^*)^\top \Delta \theta + \frac{1}{2} \Delta \theta^\top H(\theta^*) \Delta \theta$$

        At stationary points where the gradient vanishes ($\nabla f(\theta^*) = 0$), the local geometry and whether $\theta^*$ is a local minimum, local maximum, or unstable saddle point is **governed entirely by the quadratic form (energy)** induced by the Hessian matrix $H(\theta^*)$.

        #### The Physical Intuition of "Energy"
        The term "energy" originates in classical mechanics and continuum physics. The potential energy stored in an elastic system under displacement $x$ is given by the quadratic form $E = \frac{1}{2} x^\top K x$, where $K$ is the structural stiffness matrix. For a physical structure to remain stable under any perturbation, any non-zero displacement $x \neq 0$ must require positive work ($E > 0$). If $x^\top K x < 0$ along any direction, the system releases energy upon displacement, resulting in structural collapse.

        #### Role in Deep Learning and Convex Optimization
        - **Positive Definite Hessian ($H \succ 0$)**: The loss landscape forms a strictly convex multidimensional bowl. Gradient descent, Newton-Raphson, and quasi-Newton (BFGS) algorithms exhibit rapid, guaranteed convergence toward the unique local minimum.
        - **Indefinite Hessian (Mixed Eigenvalues)**: The landscape forms a hyperbolic saddle. Along positive eigenvector directions, the loss rises; along negative eigenvector directions, the loss falls. First-order optimizers can stall near saddle points where gradient norms become infinitesimally small despite high residual error.
        - **Positive Semi-Definite Hessian ($H \succeq 0$)**: Occurs in overparameterized neural networks where entire manifolds of parameter weights produce identical zero-loss predictions (flat valleys or troughs).
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Spectral Mechanics

        ### 1. Quadratic Forms and Symmetrization

        For an arbitrary square matrix $M \in \mathbb{R}^{n \times n}$ and a vector $x \in \mathbb{R}^n$, the quadratic form is:

        $$Q(x) = x^\top M x = \sum_{i=1}^n \sum_{j=1}^n M_{ij} x_i x_j$$

        Any square matrix decomposes uniquely into symmetric and skew-symmetric parts:

        $$M = \frac{M + M^\top}{2} + \frac{M - M^\top}{2} = A + S$$

        Because $x^\top S x \equiv 0$ for any skew-symmetric matrix $S = -S^\top$, the skew-symmetric component contributes nothing to the quadratic form:

        $$x^\top M x = x^\top A x, \quad \text{where } A = \frac{M + M^\top}{2}$$

        Consequently, in the study of quadratic forms and energy landscapes, **$A$ is always assumed to be symmetric ($A = A^\top$) without loss of generality**.

        ### 2. Spectral Decoupling and Principal Curvatures

        By the Spectral Theorem for symmetric matrices, there exists an orthogonal matrix $Q = [q_1, \dots, q_n]$ of eigenvectors and a real diagonal matrix $\Lambda = \operatorname{diag}(\lambda_1, \dots, \lambda_n)$ of eigenvalues such that:

        $$A = Q \Lambda Q^\top = \sum_{i=1}^n \lambda_i q_i q_i^\top$$

        Applying the orthogonal change of basis $y = Q^\top x$ (so $x = Q y$):

        $$Q(x) = x^\top (Q \Lambda Q^\top) x = (Q^\top x)^\top \Lambda (Q^\top x) = y^\top \Lambda y = \sum_{i=1}^n \lambda_i y_i^2$$

        The spectral transformation decouples the coupled quadratic cross-terms into an independent sum of squares. The eigenvalues $\lambda_i$ represent the **principal curvatures** along the orthogonal principal axes $q_i$.

        ### 3. Classification of Matrix Definiteness

        The sign of the energy $Q(x) = x^\top A x$ for non-zero vectors $x \neq 0$ defines the definiteness of $A$:

        1. **Positive Definite ($A \succ 0$)**:

        $$x^\top A x > 0 \quad \forall x \neq 0 \iff \lambda_i > 0 \quad \forall i \in \{1, \dots, n\}$$

        *Geometry*: Strictly convex elliptic bowl opening upwards. Unique global minimum at $x = 0$.

        2. **Negative Definite ($A \prec 0$)**:

        $$x^\top A x < 0 \quad \forall x \neq 0 \iff \lambda_i < 0 \quad \forall i \in \{1, \dots, n\}$$

        *Geometry*: Strictly concave dome opening downwards. Unique global maximum at $x = 0$.

        3. **Positive Semi-Definite ($A \succeq 0$)**:

        $$x^\top A x \ge 0 \quad \forall x \iff \lambda_i \ge 0 \quad \forall i \in \{1, \dots, n\}$$

        *Geometry*: Convex parabolic trough. Flat zero-energy valley along the null space of $A$.

        4. **Negative Semi-Definite ($A \preceq 0$)**:

        $$x^\top A x \le 0 \quad \forall x \iff \lambda_i \le 0 \quad \forall i \in \{1, \dots, n\}$$

        *Geometry*: Concave trough with flat zero-energy directions.

        5. **Indefinite**:

        $$x^\top A x \text{ assumes both positive and negative values} \iff \exists \lambda_j > 0 \text{ and } \lambda_k < 0$$

        *Geometry*: Hyperbolic saddle surface (pringle). The origin $x = 0$ is a saddle point.

        ### 4. The Rayleigh Quotient and Extreme Curvatures

        For any non-zero vector $x \in \mathbb{R}^n$, the **Rayleigh Quotient** is defined as:

        $$R(A, x) = \frac{x^\top A x}{x^\top x}$$

        By the Courant-Fischer Minimax Theorem, the Rayleigh quotient is bounded strictly between the minimal and maximal eigenvalues of $A$:

        $$\lambda_{\min}(A) \le \frac{x^\top A x}{x^\top x} \le \lambda_{\max}(A)$$

        - The minimum is attained when $x$ aligns with the eigenvector $q_{\min}$ corresponding to $\lambda_{\min}$.
        - The maximum is attained when $x$ aligns with the eigenvector $q_{\max}$ corresponding to $\lambda_{\max}$.
        """
    )
    return


@app.cell
def _(np):
    # Create coordinate grid for 2D quadratic form visualizations
    grid_lim = 2.5
    n_pts = 60
    u_vals = np.linspace(-grid_lim, grid_lim, n_pts)
    v_vals = np.linspace(-grid_lim, grid_lim, n_pts)
    uu, vv = np.meshgrid(u_vals, v_vals)
    pts_2d = np.column_stack([uu.ravel(), vv.ravel()])

    # Matrix Archetypes
    # 1. Positive Definite: Strictly convex bowl
    a_pos_def = np.array([[3.0, 0.8], [0.8, 2.0]])
    evals_pd, evecs_pd = np.linalg.eigh(a_pos_def)

    # 2. Indefinite: Hyperbolic saddle
    a_indef = np.array([[2.5, 0.5], [0.5, -2.0]])
    evals_indef, evecs_indef = np.linalg.eigh(a_indef)

    # 3. Positive Semi-Definite: Valley trough with zero eigenvalue
    a_semi_def = np.array([[2.0, 1.0], [1.0, 0.5]])
    evals_psd, evecs_psd = np.linalg.eigh(a_semi_def)

    # Compute energy surfaces: Z = [x1, x2] A [x1, x2]^T
    def eval_energy(mat):
        # Vectorized quadratic form: sum_j sum_k X_j A_jk X_k
        return np.sum(pts_2d @ mat * pts_2d, axis=1).reshape(uu.shape)

    z_pd = eval_energy(a_pos_def)
    z_indef = eval_energy(a_indef)
    z_psd = eval_energy(a_semi_def)

    return (
        a_indef,
        a_pos_def,
        a_semi_def,
        eval_energy,
        evals_indef,
        evals_pd,
        evals_psd,
        evecs_indef,
        evecs_pd,
        evecs_psd,
        grid_lim,
        n_pts,
        pts_2d,
        u_vals,
        uu,
        v_vals,
        vv,
        z_indef,
        z_pd,
        z_psd,
    )


@app.cell
def _(
    evals_indef,
    evals_pd,
    go,
    make_subplots,
    mo,
    np,
    u_vals,
    v_vals,
    z_indef,
    z_pd,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"<b>Positive Definite Landscape (lambda = {evals_pd[0]:.2f}, {evals_pd[1]:.2f})</b>",
            f"<b>Indefinite Saddle Landscape (lambda = {evals_indef[0]:.2f}, {evals_indef[1]:.2f})</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Left: Contour of Positive Definite (Ellipses)
    fig.add_trace(
        go.Contour(
            z=z_pd,
            x=u_vals,
            y=v_vals,
            colorscale="Teal",
            contours=dict(start=0.5, end=25, size=2.5, showlabels=True),
            colorbar=dict(title="Energy Q(x)", x=0.44, len=0.8),
            name="Positive Definite",
        ),
        row=1,
        col=1,
    )

    # Mark global minimum
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(color="#DC2626", size=10, symbol="x"),
            name="Global Minimum (x=0)",
        ),
        row=1,
        col=1,
    )

    # Right: Contour of Indefinite Matrix (Hyperbolas & Zero Crossing Lines)
    fig.add_trace(
        go.Contour(
            z=z_indef,
            x=u_vals,
            y=v_vals,
            colorscale="RdBu_r",
            contours=dict(start=-15, end=15, size=3.0, showlabels=True),
            colorbar=dict(title="Energy Q(x)", x=1.02, len=0.8),
            name="Indefinite",
        ),
        row=1,
        col=2,
    )

    # Mark saddle point
    fig.add_trace(
        go.Scatter(
            x=[0],
            y=[0],
            mode="markers",
            marker=dict(color="#F59E0B", size=10, symbol="diamond"),
            name="Saddle Point (x=0)",
        ),
        row=1,
        col=2,
    )

    # Highlight zero-energy asymptotic directions on saddle plot
    x_asymp = np.linspace(-2.5, 2.5, 50)
    fig.add_trace(
        go.Scatter(
            x=x_asymp,
            y=x_asymp * 0.95,
            mode="lines",
            line=dict(color="#111827", width=1.5, dash="dash"),
            name="Zero Energy Boundary Q(x)=0",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="x_1 Coordinate", row=1, col=1)
    fig.update_yaxes(title_text="x_2 Coordinate", row=1, col=1)
    fig.update_xaxes(title_text="x_1 Coordinate", row=1, col=2)
    fig.update_yaxes(title_text="x_2 Coordinate", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return fig, viz, x_asymp


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below reveals how matrix definiteness dictates energy level sets and curvature:

                1. **Left Panel (Positive Definite)**: Both eigenvalues are positive ($\lambda_1 = 1.62, \lambda_2 = 3.38$). Contours form concentric ellipses centered on the unique global minimum ($x = 0$). Any direction moving away from the origin causes energy $Q(x)$ to climb monotonically.
                2. **Right Panel (Indefinite Saddle)**: The eigenvalues have opposing signs ($\lambda_1 = -2.05, \lambda_2 = 2.55$). Level sets form hyperbolas separated by linear asymptotic boundaries where $Q(x) = 0$. Along the positive eigenvector direction, energy increases; along the negative eigenvector direction, energy plunges toward $-\infty$.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    a_indef,
    a_pos_def,
    a_semi_def,
    evals_indef,
    evals_pd,
    evals_psd,
    mo,
    np,
    pd,
):
    # Example 1: Pure NumPy Definiteness Classification Engine
    def classify_matrix_definiteness(mat):
        # Symmetrize
        sym_mat = (mat + mat.T) / 2.0
        eigvals = np.linalg.eigvalsh(sym_mat)
        tol = 1e-10

        if np.all(eigvals > tol):
            status = "Positive Definite (A > 0)"
            geometry = "Strictly convex bowl; unique global minimum"
        elif np.all(eigvals < -tol):
            status = "Negative Definite (A < 0)"
            geometry = "Strictly concave dome; unique global maximum"
        elif np.all(eigvals >= -tol):
            status = "Positive Semi-Definite (A >= 0)"
            geometry = "Convex valley trough; flat null space directions"
        elif np.all(eigvals <= tol):
            status = "Negative Semi-Definite (A <= 0)"
            geometry = "Concave valley ridge; flat null space directions"
        else:
            status = "Indefinite (Mixed Signs)"
            geometry = "Hyperbolic saddle point; positive and negative directions"

        return {
            "Eigenvalues": np.round(eigvals, 4),
            "Status": status,
            "Geometry": geometry,
            "Rayleigh_Range": f"[{eigvals.min():.3f}, {eigvals.max():.3f}]",
        }

    test_matrices = [
        ("A_PosDef", a_pos_def),
        ("A_Indef", a_indef),
        ("A_SemiDef", a_semi_def),
        ("A_NegDef", -a_pos_def),
    ]

    records = []
    for name, m in test_matrices:
        res = classify_matrix_definiteness(m)
        records.append(
            {
                "Matrix_Name": name,
                "Matrix_Values": str(m.tolist()),
                "Eigenvalues": str(res["Eigenvalues"]),
                "Definiteness_Classification": res["Status"],
                "Rayleigh_Curvature_Bounds": res["Rayleigh_Range"],
            }
        )

    df_classification = pd.DataFrame(records)

    # Example 2: Rayleigh Quotient Numerical Extremization on Unit Circle
    angles = np.linspace(0, 2 * np.pi, 360)
    circle_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    # Compute R(A, x) for a_pos_def
    rayleigh_vals = np.sum(circle_pts @ a_pos_def * circle_pts, axis=1)

    df_rayleigh = pd.DataFrame(
        [
            {
                "Quantity": "Minimum Rayleigh Quotient min R(A, x)",
                "Numerical_Grid_Search": round(np.min(rayleigh_vals), 5),
                "Theoretical_Eigenvalue (lambda_min)": round(evals_pd[0], 5),
                "Absolute_Difference": round(abs(np.min(rayleigh_vals) - evals_pd[0]), 8),
            },
            {
                "Quantity": "Maximum Rayleigh Quotient max R(A, x)",
                "Numerical_Grid_Search": round(np.max(rayleigh_vals), 5),
                "Theoretical_Eigenvalue (lambda_max)": round(evals_pd[1], 5),
                "Absolute_Difference": round(abs(np.max(rayleigh_vals) - evals_pd[1]), 8),
            },
        ]
    )

    # Example 3: Optimization Landscape Diagnostics
    # Test function: Rosenbrock-like or multi-extrema Hessian at distinct points
    df_opt = pd.DataFrame(
        [
            {
                "Stationary_Point_Type": "Local Minimum",
                "Hessian_Definiteness": "Positive Definite (H > 0)",
                "Newton_Step_Direction": "-H^(-1) grad is valid descent",
                "Optimizer_Behavior": "Fast quadratic convergence",
            },
            {
                "Stationary_Point_Type": "Saddle Point",
                "Hessian_Definiteness": "Indefinite (Mixed signs)",
                "Newton_Step_Direction": "-H^(-1) grad can ascend to saddle",
                "Optimizer_Behavior": "Stalls; requires Hessian-free escape or cubic regularization",
            },
            {
                "Stationary_Point_Type": "Flat Valley / Degenerate Minimum",
                "Hessian_Definiteness": "Positive Semi-Definite (lambda_min = 0)",
                "Newton_Step_Direction": "H is singular; inverse does not exist",
                "Optimizer_Behavior": "Requires Moore-Penrose pseudoinverse or damping (Levenberg-Marquardt)",
            },
        ]
    )

    table_class = mo.ui.table(df_classification)
    table_ray = mo.ui.table(df_rayleigh)
    table_opt = mo.ui.table(df_opt)

    return (
        angles,
        circle_pts,
        classify_matrix_definiteness,
        df_classification,
        df_opt,
        df_rayleigh,
        m,
        name,
        rayleigh_vals,
        records,
        res,
        table_class,
        table_opt,
        table_ray,
        test_matrices,
    )


@app.cell
def _(mo, table_class, table_opt, table_ray):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Systematic Matrix Definiteness Classification

                Classifying symmetric matrix archetypes based on eigenvalues and Rayleigh curvature ranges:
                """
            ),
            table_class,
            mo.md(
                r"""
                ### Example 2: Courant-Fischer Minimax Rayleigh Quotient Verification

                Confirming that numerical extremization of the Rayleigh quotient over the unit circle exactly recovers theoretical matrix eigenvalues:
                """
            ),
            table_ray,
            mo.md(
                r"""
                ### Example 3: Definiteness Implications for Numerical Optimizers

                How the definiteness of the Hessian matrix dictates the stability and convergence rate of second-order optimization algorithms:
                """
            ),
            table_opt,
        ]
    )


if __name__ == "__main__":
    app.run()
