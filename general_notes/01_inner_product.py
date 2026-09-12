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
    # Note 01: Inner Products: Measuring Similarity in Vector Spaces

    &larr; Previous Note: [00 Systems of Linear Equations](00_introduction.py) | Next Note: [02 Norms and Metrics](02_norm_and_metric.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Every time a machine learning model computes a **similarity score**, a **distance**, or a **projection**, it is evaluating an inner product. This single operation is the computational atom behind:

    - **Cosine similarity** in recommendation engines and search (how similar are two document embeddings?)
    - **Attention scores** in Transformers ($\mathbf{q}^T \mathbf{k}$: how relevant is token $j$ to token $i$?)
    - **Kernel methods** in SVMs (computing decision boundaries in implicitly high-dimensional spaces without ever constructing those spaces)
    - **Confusion matrices** as dot products (TP, FP, FN, TN computed via binary vector inner products)
    - **Projection** of data onto principal components (PCA), regression hyperplanes, and orthogonal bases

    Without inner products, there is no notion of angle, length, orthogonality, or similarity, and therefore no geometry to learn from.

    **Prerequisites**: Chapter 1 (vectors, linear combinations, matrix-vector multiplication).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### Definition: What Is an Inner Product?

    An **inner product** on a vector space $V$ is a function $\langle \cdot, \cdot \rangle : V \times V \to F$ (where $F = \mathbb{R}$ or $\mathbb{C}$) satisfying:

    | Property | Formula | Intuition |
    | :--- | :--- | :--- |
    | **Linearity** | $\langle au + bv, w \rangle = a\langle u, w \rangle + b\langle v, w \rangle$ | Distributes over addition and scaling |
    | **Conjugate symmetry** | $\langle u, v \rangle = \overline{\langle v, u \rangle}$ | Order matters only over $\mathbb{C}$ |
    | **Positive definiteness** | $\langle v, v \rangle \geq 0$, with equality iff $v = \mathbf{0}$ | Non-zero vectors have positive "length" |

    In $\mathbb{R}^n$, the standard inner product is the familiar **dot product**:

    $$
    \langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i = \mathbf{u}^T \mathbf{v}
    $$

    ### Beyond Euclidean Space

    The inner product generalizes beyond real coordinate spaces:

    - **Complex Inner Product ($\mathbb{C}^n$)**: To maintain positive definiteness, one vector is complex conjugated:
      $$
      \langle \mathbf{u}, \mathbf{v} \rangle = \sum_{i=1}^n \overline{u}_i v_i = \mathbf{u}^H \mathbf{v}
      $$
    - **Function Spaces ($L^2$)**: For square-integrable functions over $[a,b]$, the inner product is a continuous sum (integral):
      $$
      \langle f, g \rangle = \int_a^b f(x) \overline{g(x)} \, dx
      $$
      This generalization is the foundation of Fourier series and functional analysis.

    ### What the Inner Product Unlocks

    From this single operation, we derive the entire geometric toolkit:

    | Derived Concept | Formula | Where It Appears |
    | :--- | :--- | :--- |
    | **Length (Norm)** | $\|\mathbf{v}\| = \sqrt{\langle \mathbf{v}, \mathbf{v} \rangle}$ | Weight magnitudes, gradient norms |
    | **Angle / Cosine Similarity** | $\cos\theta = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | Semantic similarity, attention scores |
    | **Orthogonality** | $\langle \mathbf{u}, \mathbf{v} \rangle = 0$ | PCA axes, decorrelated features |
    | **Projection** | $\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \mathbf{u}$ | Gram-Schmidt, least-squares, QR |

    ---

    ### Role in Statistics: The Confusion Matrix as a Dot Product

    For binary classification with true labels $\mathbf{y} \in \{0,1\}^n$ and predictions $\hat{\mathbf{y}} \in \{0,1\}^n$, every cell of the confusion matrix is an inner product:

    $$
    \text{TP} = \mathbf{y} \cdot \hat{\mathbf{y}}, \quad
    \text{FN} = \mathbf{y} \cdot (\mathbf{1} - \hat{\mathbf{y}}), \quad
    \text{FP} = (\mathbf{1} - \mathbf{y}) \cdot \hat{\mathbf{y}}, \quad
    \text{TN} = (\mathbf{1} - \mathbf{y}) \cdot (\mathbf{1} - \hat{\mathbf{y}})
    $$

    Once TP is known, the other three counts follow from the class totals, so only **one full dot product** is required.

    ---

    ### Role in Machine Learning: The Kernel Trick & SVMs

    In the **primal form** of the soft-margin Support Vector Machine, we minimize weights $\mathbf{w}$ and slack variables $\xi$ subject to margin constraints:

    $$
    \min_{\mathbf{w}, b, \xi} \frac{1}{2} \|\mathbf{w}\|^2 + C \sum_{i=1}^n \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \geq 1 - \xi_i
    $$

    By introducing Lagrange multipliers $\alpha$, we convert this into the **dual form**, where the optimization depends entirely on pairwise inner products of the data points, eliminating $\mathbf{w}$ entirely:

    $$
    \max_{\alpha} \sum_{i=1}^n \alpha_i - \frac{1}{2} \sum_{i,j} \alpha_i \alpha_j y_i y_j \langle \mathbf{x}_i, \mathbf{x}_j \rangle
    \quad \text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_i \alpha_i y_i = 0
    $$

    **The Kernel Trick**: Replace $\langle \mathbf{x}_i, \mathbf{x}_j \rangle$ with a kernel function $K(\mathbf{x}_i, \mathbf{x}_j) = \langle \phi(\mathbf{x}_i), \phi(\mathbf{x}_j) \rangle$ that implicitly computes inner products in a higher-dimensional feature space, without ever constructing $\phi(\mathbf{x})$ explicitly:

    | Kernel | $K(\mathbf{x}, \mathbf{y})$ | Effect |
    | :--- | :--- | :--- |
    | Linear | $\mathbf{x} \cdot \mathbf{y}$ | No transformation |
    | Polynomial (degree $d$) | $(\mathbf{x} \cdot \mathbf{y} + 1)^d$ | Captures interactions up to order $d$ |
    | RBF (Gaussian) | $\exp(-\gamma \|\mathbf{x} - \mathbf{y}\|^2)$ | Infinite-dimensional feature space |

    ---

    ### Role in Deep Learning: Attention Scores

    The scaled dot-product attention in Transformers is fundamentally a **batch of inner products** between query and key vectors:

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
    $$

    Each element of $QK^T$ is $\langle \mathbf{q}_i, \mathbf{k}_j \rangle$, the inner product that measures how much token $i$ should attend to token $j$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [c] Code Examples

    ### Example 1: Confusion Matrix via the Dot Product
    """)
    return


@app.cell
def _():
    import numpy as np

    y_true = np.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 0] * 10000)
    y_pred = np.array([1, 1, 1, 0, 0, 1, 1, 0, 1, 1] * 10000)

    def confusion_matrix_dot(t, p):
        """Full dot-product approach: 4 inner products."""
        TP = t @ p
        FP = (1 - t) @ p
        FN = t @ (1 - p)
        TN = (1 - t) @ (1 - p)
        return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

    def confusion_matrix_fast(t, p):
        """Optimized: 1 dot product + scalar arithmetic."""
        TP = t @ p
        P = np.sum(t)  # actual positives
        PP = np.sum(p)  # predicted positives
        N = len(t) - P  # actual negatives
        FP = PP - TP
        FN = P - TP
        TN = N - FP
        return {"TP": TP, "FP": FP, "FN": FN, "TN": TN}

    cm_full = confusion_matrix_dot(y_true, y_pred)
    cm_fast = confusion_matrix_fast(y_true, y_pred)

    assert cm_full == cm_fast, "Results must match"
    cm_full
    return cm_fast, cm_full, confusion_matrix_dot, confusion_matrix_fast, np, y_pred, y_true


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 2: Kernel Trick: Polynomial Kernel vs Explicit Feature Mapping

    For $\mathbf{x}_1 = (1, 2)$ and $\mathbf{x}_2 = (3, 4)$ with degree-2 polynomial kernel $K(\mathbf{x}, \mathbf{y}) = (\mathbf{x} \cdot \mathbf{y} + 1)^2$:

    **Explicit mapping**: $\phi(\mathbf{x}) = (1, \sqrt{2}x_1, \sqrt{2}x_2, x_1^2, \sqrt{2}\,x_1 x_2, x_2^2)$
    """)
    return


@app.cell
def _(np):
    # Kernel trick: compute in input space
    x1_kern = np.array([1, 2])
    x2_kern = np.array([3, 4])
    kernel_result = (x1_kern @ x2_kern + 1) ** 2

    # Explicit feature mapping: compute in transformed space
    def phi(x):
        return np.array([1.0, np.sqrt(2) * x[0], np.sqrt(2) * x[1], x[0] ** 2, np.sqrt(2) * x[0] * x[1], x[1] ** 2])

    explicit_result = phi(x1_kern) @ phi(x2_kern)

    assert np.isclose(kernel_result, explicit_result), "Kernel trick must equal explicit mapping"
    {
        "kernel_result": kernel_result,
        "explicit_result": explicit_result,
        "match": np.isclose(kernel_result, explicit_result),
    }
    return explicit_result, kernel_result, phi, x1_kern, x2_kern


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Example 3: SVM Decision Boundaries with Different Kernels

    The inner product (or its kernel generalization) determines the shape of the decision boundary:
    - **Linear kernel**: straight line
    - **Polynomial kernel**: curved boundary
    - **RBF kernel**: highly flexible, locally adaptive boundary
    """)
    return


@app.cell
def _(np):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.datasets import make_moons
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC

    X, y = make_moons(n_samples=300, noise=0.2, random_state=47)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47)

    kernels = ["linear", "poly", "rbf"]
    C_values = [0.1, 1.0, 10.0]

    fig = make_subplots(
        rows=len(C_values),
        cols=len(kernels),
        subplot_titles=[f"{k.capitalize()} (C={c})" for c in C_values for k in kernels],
        vertical_spacing=0.08,
        horizontal_spacing=0.06,
    )

    h = 0.05
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx = np.arange(x_min, x_max, h)
    yy = np.arange(y_min, y_max, h)
    grid_x, grid_y = np.meshgrid(xx, yy)

    for i, C in enumerate(C_values):
        for j, kernel in enumerate(kernels):
            model = SVC(kernel=kernel, degree=2, C=C)
            model.fit(X_train, y_train)
            acc = accuracy_score(y_test, model.predict(X_test))
            Z = model.predict(np.c_[grid_x.ravel(), grid_y.ravel()]).reshape(grid_x.shape)

            fig.add_trace(
                go.Contour(
                    x=xx,
                    y=yy,
                    z=Z,
                    showscale=False,
                    opacity=0.3,
                    colorscale=[[0, "#3498db"], [1, "#e74c3c"]],
                    hoverinfo="skip",
                ),
                row=i + 1,
                col=j + 1,
            )

            show_legend = i == 0 and j == 0
            fig.add_trace(
                go.Scatter(
                    x=X_test[y_test == 0, 0],
                    y=X_test[y_test == 0, 1],
                    mode="markers",
                    marker=dict(color="#2980b9", size=6, line=dict(width=0.5, color="black")),
                    name="Class 0",
                    showlegend=show_legend,
                ),
                row=i + 1,
                col=j + 1,
            )
            fig.add_trace(
                go.Scatter(
                    x=X_test[y_test == 1, 0],
                    y=X_test[y_test == 1, 1],
                    mode="markers",
                    marker=dict(color="#c0392b", size=6, line=dict(width=0.5, color="black")),
                    name="Class 1",
                    showlegend=show_legend,
                ),
                row=i + 1,
                col=j + 1,
            )

    fig.update_layout(
        title=dict(text="SVM Decision Boundaries Across Kernels and Margins", font=dict(size=14)),
        template="plotly_white",
        width=880,
        height=750,
    )

    fig
    return (
        C_values,
        SVC,
        X,
        X_test,
        X_train,
        accuracy_score,
        fig,
        grid_x,
        grid_y,
        h,
        kernels,
        make_moons,
        make_subplots,
        train_test_split,
        x_max,
        x_min,
        xx,
        y,
        y_max,
        y_min,
        y_test,
        y_train,
        yy,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Takeaway

    - The **inner product** $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$ is the single operation from which length, angle, orthogonality, and projection are derived.
    - **In Statistics**: The confusion matrix for binary classification can be computed as dot products between binary label vectors, where one dot product plus scalar arithmetic suffices.
    - **In Machine Learning**: SVMs depend entirely on pairwise inner products. The **kernel trick** replaces these with kernel functions to learn non-linear decision boundaries without explicit high-dimensional feature construction.
    - **In Deep Learning**: Every attention score in a Transformer is an inner product $\langle \mathbf{q}_i, \mathbf{k}_j \rangle$, making the inner product the computational heartbeat of modern language models.
    - **Practical rule**: If two vectors point in similar directions, their inner product is large and positive. If orthogonal, it is zero. If opposing, it is negative. This geometric intuition drives similarity search, attention, and classification.

    ---

    &larr; Previous Note: [00 Systems of Linear Equations](00_introduction.py) | Next Note: [02 Norms and Metrics](02_norm_and_metric.py) &rarr;
    """)
    return


if __name__ == "__main__":
    app.run()
