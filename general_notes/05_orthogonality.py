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
    # Note 05: Orthogonal Matrices, Isometries, and Rotary Embeddings

    &larr; Previous Note: [04 Rank-One Matrices](04_rank_one_matrices.py) | Next Note: [06 Moore-Penrose Pseudoinverse](06_moore_penrose_inverse.py) &rarr;

    ---

    An orthogonal matrix represents a linear transformation that preserves the geometry of Euclidean space. When an orthogonal operator acts on a vector space, it strictly preserves lengths, distances, and angles between vectors. These distance-preserving transformations, known as **isometries**, form the structural foundation of stable numerical solvers, coordinate frame rotations, and rotary position embeddings (RoPE) in modern deep learning.

    ---

    ## 1. Mathematical Foundations

    ### Definition

    A square matrix $\mathbf{Q} \in \mathbb{R}^{n \times n}$ is orthogonal if its transpose equals its inverse:

    $$
    \mathbf{Q}^T \mathbf{Q} = \mathbf{Q}\mathbf{Q}^T = \mathbf{I}_n
    $$

    This identity directly implies:

    $$
    \mathbf{Q}^{-1} = \mathbf{Q}^T
    $$

    Writing $\mathbf{Q}$ in terms of its column vectors $[\mathbf{q}_1, \mathbf{q}_2, \dots, \mathbf{q}_n]$, the condition $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}_n$ states that the columns form an orthonormal basis of $\mathbb{R}^n$:

    $$
    \mathbf{q}_i^T \mathbf{q}_j = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}
    $$

    Similarly, $\mathbf{Q}\mathbf{Q}^T = \mathbf{I}_n$ establishes that the rows of $\mathbf{Q}$ also form an orthonormal basis.

    ---

    ### The Isometry Property: Preservation of Lengths and Angles

    Let $\mathbf{x}, \mathbf{y} \in \mathbb{R}^n$ be arbitrary vectors, and let $\tilde{\mathbf{x}} = \mathbf{Q}\mathbf{x}$ and $\tilde{\mathbf{y}} = \mathbf{Q}\mathbf{y}$ denote their transformed counterparts.

    The inner product between the transformed vectors satisfies:

    $$
    \langle \mathbf{Q}\mathbf{x}, \mathbf{Q}\mathbf{y} \rangle = (\mathbf{Q}\mathbf{x})^T (\mathbf{Q}\mathbf{y}) = \mathbf{x}^T (\mathbf{Q}^T \mathbf{Q}) \mathbf{y} = \mathbf{x}^T \mathbf{I}_n \mathbf{y} = \mathbf{x}^T \mathbf{y} = \langle \mathbf{x}, \mathbf{y} \rangle
    $$

    Setting $\mathbf{y} = \mathbf{x}$ demonstrates the preservation of Euclidean norm ($L_2$ length):

    $$
    \|\mathbf{Q}\mathbf{x}\|_2^2 = \langle \mathbf{Q}\mathbf{x}, \mathbf{Q}\mathbf{x} \rangle = \|\mathbf{x}\|_2^2 \implies \|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2
    $$

    Because both lengths and dot products remain invariant, the angle $\theta$ between any pair of vectors is strictly preserved:

    $$
    \cos(\theta) = \frac{\langle \mathbf{Q}\mathbf{x}, \mathbf{Q}\mathbf{y} \rangle}{\|\mathbf{Q}\mathbf{x}\|_2 \|\mathbf{Q}\mathbf{y}\|_2} = \frac{\langle \mathbf{x}, \mathbf{y} \rangle}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}
    $$

    Linear operators that preserve Euclidean distances are called **isometries**.

    ---

    ### Determinant Dichotomy: Rotations vs Reflections

    Taking the determinant of both sides of $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}_n$:

    $$
    \det(\mathbf{Q}^T \mathbf{Q}) = \det(\mathbf{Q}^T) \det(\mathbf{Q}) = (\det(\mathbf{Q}))^2 = \det(\mathbf{I}_n) = 1
    $$

    Therefore, every orthogonal matrix satisfies:

    $$
    \det(\mathbf{Q}) \in \{+1, -1\}
    $$

    This separates orthogonal transformations into two distinct geometric classes:

    1. **Rotations ($\det(\mathbf{Q}) = +1$):**
       Belong to the Special Orthogonal Group $\text{SO}(n)$. These transformations preserve orientation and chirality. In two dimensions, a counter-clockwise rotation by angle $\theta$ is:

    $$
    \mathbf{R}_\theta = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}, \quad \det(\mathbf{R}_\theta) = \cos^2\theta + \sin^2\theta = +1
    $$

    2. **Reflections ($\det(\mathbf{Q}) = -1$):**
       Transformations that reverse spatial orientation (mirror reflection). A reflection across a line oriented at angle $\phi$ relative to the positive x-axis is:

    $$
    \mathbf{H}_\phi = \begin{bmatrix} \cos(2\phi) & \sin(2\phi) \\ \sin(2\phi) & -\cos(2\phi) \end{bmatrix}, \quad \det(\mathbf{H}_\phi) = -\cos^2(2\phi) - \sin^2(2\phi) = -1
    $$

    ---

    ## 2. Applications in Modern AI and Machine Learning

    ### Rotary Position Embedding (RoPE) in Large Language Models

    State-of-the-art transformer architectures (such as Llama 3, Mistral, and Gemma) replace additive absolute positional embeddings with Rotary Position Embeddings (RoPE).

    RoPE pairs consecutive channels in query and key vectors and applies a 2D orthogonal rotation matrix $\mathbf{R}_m$ corresponding to token position $m$:

    $$
    \tilde{\mathbf{q}}_m = \mathbf{R}_m \mathbf{q}_m, \quad \tilde{\mathbf{k}}_n = \mathbf{R}_n \mathbf{k}_n
    $$

    When computing the self-attention dot product between query token $m$ and key token $n$:

    $$
    \langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = (\mathbf{R}_m \mathbf{q}_m)^T (\mathbf{R}_n \mathbf{k}_n) = \mathbf{q}_m^T \mathbf{R}_m^T \mathbf{R}_n \mathbf{k}_n
    $$

    Because rotation matrices compose orthogonally ($\mathbf{R}_m^T = \mathbf{R}_{-m}$ and $\mathbf{R}_{-m}\mathbf{R}_n = \mathbf{R}_{n - m}$):

    $$
    \langle \tilde{\mathbf{q}}_m, \tilde{\mathbf{k}}_n \rangle = \mathbf{q}_m^T \mathbf{R}_{n - m} \mathbf{k}_n
    $$

    The inner product depends solely on the relative displacement $(n - m)$, while the $L_2$ norm of every individual query and key vector is preserved.

    ---

    ### Orthogonal Weight Initialization for Deep Networks

    In deep neural networks and recurrent models, vanishing or exploding gradients occur when signal magnitudes change exponentially across layers. Initializing weight matrices $\mathbf{W}$ as random orthogonal matrices:

    $$
    \mathbf{W}^T \mathbf{W} = \mathbf{I}
    $$

    ensures that every singular value equals 1:

    $$
    \sigma_i(\mathbf{W}) = 1, \quad \forall i
    $$

    This guarantees that the Euclidean norm of forward activations and backward gradient vectors is preserved layer by layer at initialization (Saxe et al., 2013).

    ---

    ### Numerical Stability and QR Factorization

    In ordinary least squares, solving normal equations directly via $(\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$ squares the condition number:

    $$
    \kappa(\mathbf{X}^T \mathbf{X}) = (\kappa(\mathbf{X}))^2
    $$

    By using QR factorization, $\mathbf{X} = \mathbf{Q}\mathbf{R}$ with orthogonal $\mathbf{Q}$ and upper triangular $\mathbf{R}$, the least-squares solution simplifies to back-substitution:

    $$
    \mathbf{R}\mathbf{w} = \mathbf{Q}^T \mathbf{y}
    $$

    Because $\kappa(\mathbf{Q}) = 1$, the condition number is not squared, avoiding severe loss of numerical precision in floating-point arithmetic.
    """)
    return


@app.cell
def _(go, make_subplots, np):
    # Angles for 2D transformation comparison
    rot_angle_deg = 60.0
    rot_angle_rad = np.radians(rot_angle_deg)
    cos_r = np.cos(rot_angle_rad)
    sin_r = np.sin(rot_angle_rad)
    q_rot = np.array([[cos_r, -sin_r], [sin_r, cos_r]])

    ref_angle_deg = 30.0
    ref_angle_rad = np.radians(ref_angle_deg)
    cos_2f = np.cos(2 * ref_angle_rad)
    sin_2f = np.sin(2 * ref_angle_rad)
    q_ref = np.array([[cos_2f, sin_2f], [sin_2f, -cos_2f]])

    # Unit circle points
    t = np.linspace(0, 2 * np.pi, 200)
    circle_pts = np.vstack([np.cos(t), np.sin(t)])

    # Sample basis and test vectors
    e1 = np.array([1.0, 0.0])
    e2 = np.array([0.0, 1.0])
    v_test = np.array([1.2, 0.7])

    rot_circle = q_rot @ circle_pts
    rot_e1 = q_rot @ e1
    rot_e2 = q_rot @ e2
    rot_v = q_rot @ v_test

    ref_circle = q_ref @ circle_pts
    ref_e1 = q_ref @ e1
    ref_e2 = q_ref @ e2
    ref_v = q_ref @ v_test

    det_rot = np.linalg.det(q_rot)
    det_ref = np.linalg.det(q_ref)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            f"Rotation ({rot_angle_deg:.0f}°), det(Q) = {det_rot:+.1f}",
            f"Reflection (across {ref_angle_deg:.0f}° mirror line), det(Q) = {det_ref:+.1f}",
        ],
    )

    # Left: Rotation
    fig.add_trace(
        go.Scatter(
            x=circle_pts[0],
            y=circle_pts[1],
            mode="lines",
            line=dict(color="#cbd5e1", dash="dash", width=1.5),
            name="Unit Circle",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=rot_circle[0],
            y=rot_circle[1],
            mode="lines",
            line=dict(color="#3b82f6", width=2),
            name="Rotated Circle",
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, e1[0]],
            y=[0, e1[1]],
            mode="lines+markers",
            line=dict(color="#94a3b8", width=2),
            marker=dict(size=6),
            name="Original e1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, rot_e1[0]],
            y=[0, rot_e1[1]],
            mode="lines+markers",
            line=dict(color="#2563eb", width=3),
            marker=dict(size=8),
            name="Rotated e1",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, rot_e2[0]],
            y=[0, rot_e2[1]],
            mode="lines+markers",
            line=dict(color="#06b6d4", width=3),
            marker=dict(size=8),
            name="Rotated e2",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, v_test[0]],
            y=[0, v_test[1]],
            mode="lines+markers",
            line=dict(color="#d97706", dash="dot", width=2),
            marker=dict(size=6),
            name="Original v",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, rot_v[0]],
            y=[0, rot_v[1]],
            mode="lines+markers",
            line=dict(color="#ea580c", width=3),
            marker=dict(size=8),
            name="Rotated v",
        ),
        row=1,
        col=1,
    )

    # Right: Reflection
    line_x = np.array([-1.8, 1.8])
    line_y = line_x * np.tan(ref_angle_rad)
    fig.add_trace(
        go.Scatter(
            x=line_x,
            y=line_y,
            mode="lines",
            line=dict(color="#94a3b8", dash="dashdot", width=1.5),
            name=f"Mirror Line ({ref_angle_deg:.0f}°)",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=circle_pts[0],
            y=circle_pts[1],
            mode="lines",
            line=dict(color="#cbd5e1", dash="dash", width=1.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=ref_circle[0],
            y=ref_circle[1],
            mode="lines",
            line=dict(color="#ec4899", width=2),
            name="Reflected Circle",
            showlegend=True,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, e1[0]],
            y=[0, e1[1]],
            mode="lines+markers",
            line=dict(color="#94a3b8", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, ref_e1[0]],
            y=[0, ref_e1[1]],
            mode="lines+markers",
            line=dict(color="#be185d", width=3),
            marker=dict(size=8),
            name="Reflected e1",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, ref_e2[0]],
            y=[0, ref_e2[1]],
            mode="lines+markers",
            line=dict(color="#8b5cf6", width=3),
            marker=dict(size=8),
            name="Reflected e2",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, v_test[0]],
            y=[0, v_test[1]],
            mode="lines+markers",
            line=dict(color="#d97706", dash="dot", width=2),
            marker=dict(size=6),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=[0, ref_v[0]],
            y=[0, ref_v[1]],
            mode="lines+markers",
            line=dict(color="#9333ea", width=3),
            marker=dict(size=8),
            name="Reflected v",
        ),
        row=1,
        col=2,
    )

    axis_config = dict(
        range=[-1.8, 1.8],
        zeroline=True,
        zerolinecolor="#e2e8f0",
        gridcolor="#f1f5f9",
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=axis_config,
        yaxis=axis_config,
        xaxis2=axis_config,
        yaxis2=axis_config,
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
    )

    return (fig,)


@app.cell
def _(fig, mo):
    mo.ui.plotly(fig)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## 3. Code Verification and Applications

    ### Example 1: Numerical Verification of Orthogonality and Isometry Properties

    Below, we generate a random $4 \times 4$ orthogonal matrix using QR decomposition and verify:
    1. $\mathbf{Q}^T \mathbf{Q} = \mathbf{I}$
    2. Length preservation: $\|\mathbf{Q}\mathbf{x}\|_2 = \|\mathbf{x}\|_2$
    3. Inner product preservation: $\langle \mathbf{Q}\mathbf{x}, \mathbf{Q}\mathbf{y}\rangle = \langle \mathbf{x}, \mathbf{y}\rangle$
    4. Eigenvalue magnitudes: $|\lambda_i| = 1$
    """)
    return


@app.cell
def _(np):
    # Seed for reproducibility
    rng = np.random.default_rng(42)

    # Generate a random 4x4 matrix and obtain an orthogonal matrix via QR
    a_mat = rng.standard_normal((4, 4))
    q_mat, _ = np.linalg.qr(a_mat)

    # 1. Identity check
    qt_q = q_mat.T @ q_mat
    identity_error = float(np.max(np.abs(qt_q - np.eye(4))))

    # 2. Vector length preservation
    x_vec = rng.standard_normal(4)
    qx_vec = q_mat @ x_vec
    norm_x = float(np.linalg.norm(x_vec))
    norm_qx = float(np.linalg.norm(qx_vec))
    norm_diff = abs(norm_x - norm_qx)

    # 3. Inner product preservation
    y_vec = rng.standard_normal(4)
    qy_vec = q_mat @ y_vec
    dot_xy = float(np.dot(x_vec, y_vec))
    dot_q = float(np.dot(qx_vec, qy_vec))
    dot_diff = abs(dot_xy - dot_q)

    # 4. Eigenvalue magnitudes on the unit circle
    eigenvalues = np.linalg.eigvals(q_mat)
    eigen_magnitudes = np.abs(eigenvalues)

    results_table = {
        "Property": [
            "Identity check: max |Q^T Q - I|",
            "Original norm ||x||_2",
            "Transformed norm ||Q x||_2",
            "Norm difference",
            "Original dot product <x, y>",
            "Transformed dot product <Qx, Qy>",
            "Dot product difference",
            "Determinant det(Q)",
            "All eigenvalue magnitudes |lambda|",
        ],
        "Value": [
            f"{identity_error:.2e}",
            f"{norm_x:.6f}",
            f"{norm_qx:.6f}",
            f"{norm_diff:.2e}",
            f"{dot_xy:.6f}",
            f"{dot_q:.6f}",
            f"{dot_diff:.2e}",
            f"{np.linalg.det(q_mat):+.6f}",
            ", ".join(f"{m:.4f}" for m in eigen_magnitudes),
        ],
    }

    return (
        dot_diff,
        dot_q,
        dot_xy,
        eigen_magnitudes,
        eigenvalues,
        identity_error,
        norm_diff,
        norm_qx,
        norm_x,
        q_mat,
        results_table,
        x_vec,
        y_vec,
    )


@app.cell(hide_code=True)
def _(mo, pd, results_table):
    df_results = pd.DataFrame(results_table)
    mo.ui.table(df_results)
    return (df_results,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Example 2: 2D Rotary Position Embedding (RoPE) Invariance

    We implement a minimal 2D RoPE rotation block. We demonstrate that when query $\mathbf{q}$ is at position $m$ and key $\mathbf{k}$ is at position $n$, the attention inner product $\langle \mathbf{R}_m \mathbf{q}, \mathbf{R}_n \mathbf{k} \rangle$ depends solely on the relative offset $(n - m)$, regardless of absolute positions.
    """)
    return


@app.cell
def _(np):
    def get_rope_rotation(pos: int, base_theta: float = 0.05) -> np.ndarray:
        angle = pos * base_theta
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        return np.array([[cos_a, -sin_a], [sin_a, cos_a]])

    # Query and key vectors
    q_vec = np.array([1.5, -0.8])
    k_vec = np.array([-0.4, 2.1])

    # Case A: Positions m=15, n=10 (relative distance = -5)
    r_m1 = get_rope_rotation(pos=15)
    r_n1 = get_rope_rotation(pos=10)
    q_rot1 = r_m1 @ q_vec
    k_rot1 = r_n1 @ k_vec
    dot_case_a = float(np.dot(q_rot1, k_rot1))

    # Case B: Positions m=42, n=37 (same relative distance = -5)
    r_m2 = get_rope_rotation(pos=42)
    r_n2 = get_rope_rotation(pos=37)
    q_rot2 = r_m2 @ q_vec
    k_rot2 = r_n2 @ k_vec
    dot_case_b = float(np.dot(q_rot2, k_rot2))

    # Direct relative formula: q^T R(n - m) k
    r_rel = get_rope_rotation(pos=(10 - 15))
    dot_relative_formula = float(q_vec @ r_rel @ k_vec)

    rope_summary = {
        "Configuration": [
            "Case A (m=15, n=10, diff=-5)",
            "Case B (m=42, n=37, diff=-5)",
            "Relative formulation q^T R(-5) k",
            "Absolute discrepancy |Case A - Case B|",
            "Relative formula discrepancy |Case A - Formula|",
        ],
        "Attention Dot Product": [
            f"{dot_case_a:.8f}",
            f"{dot_case_b:.8f}",
            f"{dot_relative_formula:.8f}",
            f"{abs(dot_case_a - dot_case_b):.2e}",
            f"{abs(dot_case_a - dot_relative_formula):.2e}",
        ],
    }

    return (
        dot_case_a,
        dot_case_b,
        dot_relative_formula,
        get_rope_rotation,
        k_rot1,
        k_rot2,
        k_vec,
        q_rot1,
        q_rot2,
        q_vec,
        r_m1,
        r_m2,
        r_n1,
        r_n2,
        r_rel,
        rope_summary,
    )


@app.cell(hide_code=True)
def _(mo, pd, rope_summary):
    df_rope = pd.DataFrame(rope_summary)
    mo.ui.table(df_rope)
    return (df_rope,)


if __name__ == "__main__":
    app.run()
