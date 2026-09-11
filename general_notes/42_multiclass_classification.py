import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import torch
    from plotly.subplots import make_subplots

    return go, make_subplots, mo, np, pd, torch


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 41 Pseudo R-squared](41_pseudo_r2.py) | [Index](../index.html) | [43 Energy Statistics →](43_energy.py)

        # Matrix Calculus of Multiclass Classification: Softmax, Cross-Entropy, and Vector-Jacobian Products

        ## [a] Why do you need to know these concepts?

        Every modern classification neural network—from Vision Transformers (ViTs) and ResNets to Large Language Models (LLMs) predicting vocabulary distributions over hundreds of thousands of tokens—terminates with a **linear projection layer followed by Softmax and Cross-Entropy Loss**.

        #### The Elegant Cancellation: $\nabla_z L = p - y$
        When computing the gradient of Cross-Entropy loss through Softmax, a mathematical cancellation occurs:
        - The gradient of the scalar loss with respect to probabilities is non-linear and divided by probabilities: $\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$.
        - The Jacobian matrix of the Softmax function is dense with quadratic terms: $J_{i, j} = p_i(\delta_{ij} - p_j)$.
        - When multiplied via the chain rule, the $p_i$ terms in the denominator cancel out completely, yielding the simple result:

        $$\nabla_z L = \frac{\partial L}{\partial z} = p - y$$

        The gradient with respect to the raw unnormalized logits is simply the **prediction error vector** (the predicted probability vector minus the one-hot target vector).

        #### Vector-Jacobian Products (VJPs) in Backpropagation
        In reverse-mode automatic differentiation (backpropagation), forming and materializing the full $n \times n$ Jacobian matrix $\frac{\partial p}{\partial z}$ would consume $O(n^2)$ memory and compute. For an LLM vocabulary size of $n = 128,000$, an explicit Jacobian would require over $65$ gigabytes of memory for a single token. By computing the Vector-Jacobian Product (VJP) analytically, automatic differentiation evaluates the backward pass in $O(n)$ time and $O(n)$ memory.

        #### Numerical Stability and the LogSumExp Trick
        Directly evaluating $e^{z_i}$ leads to floating-point overflow for large logits (e.g., $e^{100} \approx 2.68 \times 10^{43}$) and underflow for negative logits. In practice, production implementations combine Softmax and Cross-Entropy into a fused, numerically stable kernel using the LogSumExp identity:

        $$\ln \sum_{j=1}^n e^{z_j} = M + \ln \sum_{j=1}^n e^{z_j - M}, \quad \text{where } M = \max_{j} z_j$$
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Matrix Derivations

        ### 1. Model Formulation

        Let $x \in \mathbb{R}^m$ be an input feature vector, $W \in \mathbb{R}^{n \times m}$ the weight parameter matrix, and $b \in \mathbb{R}^n$ the bias vector, where $n$ is the number of mutually exclusive classes.

        The linear pre-activation (logits) vector is:

        $$z = W x + b \in \mathbb{R}^n$$

        The Softmax function normalizes logits into a valid probability distribution $p \in \Delta^{n-1}$:

        $$p_i = \operatorname{softmax}(z)_i = \frac{e^{z_i}}{\sum_{k=1}^n e^{z_k}}, \quad \text{for } i \in \{1, \dots, n\}$$

        Let $y \in \{0, 1\}^n$ be the one-hot encoded ground truth target vector, where $y_c = 1$ for the true class $c$ and $y_j = 0$ for all $j \neq c$. The Multiclass Cross-Entropy Loss is defined as:

        $$L = -\sum_{i=1}^n y_i \ln(p_i) = -y^\top \ln(p) = -\ln(p_c)$$

        ### 2. Jacobian Matrix of the Softmax Function

        The Softmax mapping $\operatorname{softmax}: \mathbb{R}^n \to \mathbb{R}^n$ produces an $n \times n$ Jacobian matrix $J \in \mathbb{R}^{n \times n}$, where $J_{i, j} = \frac{\partial p_i}{\partial z_j}$.

        We evaluate two cases using the quotient rule:

        **Case 1: Diagonal Elements ($i = j$)**
        $$\frac{\partial p_i}{\partial z_i} = \frac{\frac{\partial}{\partial z_i}(e^{z_i}) \cdot \sum_{k} e^{z_k} - e^{z_i} \cdot \frac{\partial}{\partial z_i}(\sum_{k} e^{z_k})}{\left(\sum_{k} e^{z_k}\right)^2} = \frac{e^{z_i}}{\sum e^{z_k}} - \left(\frac{e^{z_i}}{\sum e^{z_k}}\right)^2 = p_i - p_i^2 = p_i(1 - p_i)$$

        **Case 2: Off-Diagonal Elements ($i \neq j$)**
        $$\frac{\partial p_i}{\partial z_j} = \frac{0 \cdot \sum_{k} e^{z_k} - e^{z_i} \cdot e^{z_j}}{\left(\sum_{k} e^{z_k}\right)^2} = -\frac{e^{z_i}}{\sum e^{z_k}} \frac{e^{z_j}}{\sum e^{z_k}} = -p_i p_j$$

        Using the Kronecker delta $\delta_{ij}$ (where $\delta_{ij} = 1$ if $i=j$, else $0$):

        $$\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)$$

        In compact matrix form:

        $$J_{\text{softmax}} = \frac{\partial p}{\partial z} = \operatorname{diag}(p) - p p^\top$$

        ### 3. Derivation of the Logit Gradient $\frac{\partial L}{\partial z}$

        The gradient of Cross-Entropy loss with respect to probability $p_i$ is:

        $$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$$

        Applying the multivariate chain rule:

        $$\frac{\partial L}{\partial z_j} = \sum_{i=1}^n \frac{\partial L}{\partial p_i} \frac{\partial p_i}{\partial z_j} = \sum_{i=1}^n \left(-\frac{y_i}{p_i}\right) \left[ p_i(\delta_{ij} - p_j) \right] = -\sum_{i=1}^n y_i (\delta_{ij} - p_j)$$

        Distributing the sum:

        $$\frac{\partial L}{\partial z_j} = -\sum_{i=1}^n y_i \delta_{ij} + p_j \sum_{i=1}^n y_i$$

        Because $y$ is a one-hot distribution, $\sum_{i=1}^n y_i = 1$, and the sifting property gives $\sum_{i=1}^n y_i \delta_{ij} = y_j$. Therefore:

        $$\frac{\partial L}{\partial z_j} = -y_j + p_j = p_j - y_j$$

        In full vector notation:

        $$\nabla_z L = \frac{\partial L}{\partial z} = p - y$$

        ### 4. Gradient with Respect to Parameters $W$ and $b$

        Since $z = W x + b$, the component-wise derivative is $\frac{\partial z_j}{\partial W_{j, k}} = x_k$. Applying the chain rule:

        $$\frac{\partial L}{\partial W_{j, k}} = \frac{\partial L}{\partial z_j} \frac{\partial z_j}{\partial W_{j, k}} = (p_j - y_j) x_k$$

        Expressing this as an outer product in matrix calculus:

        $$\frac{\partial L}{\partial W} = (\nabla_z L) x^\top = (p - y) x^\top \in \mathbb{R}^{n \times m}$$

        $$\frac{\partial L}{\partial b} = \nabla_z L = p - y \in \mathbb{R}^n$$
        """
    )
    return


@app.cell
def _(np, torch):
    # Fix seed for reproducible gradient audit
    torch.manual_seed(42)
    np.random.seed(42)

    # 3-class classification with 2 input features
    m_features = 2
    n_classes = 3

    # Sample input x and weight matrix W
    x_input = torch.tensor([[2.5], [-1.2]], dtype=torch.float64)
    w_matrix = torch.tensor(
        [[0.8, -0.5], [-0.3, 1.2], [0.4, 0.6]],
        dtype=torch.float64,
        requires_grad=True,
    )
    b_bias = torch.tensor([[0.1], [-0.2], [0.3]], dtype=torch.float64, requires_grad=True)

    # Target class: Class 1 (0-indexed: [0, 1, 0]^T)
    target_idx = 1
    y_onehot = torch.zeros(n_classes, 1, dtype=torch.float64)
    y_onehot[target_idx] = 1.0

    # Forward pass
    z_logits = w_matrix @ x_input + b_bias
    z_logits.retain_grad()

    p_probs = torch.softmax(z_logits, dim=0)
    p_probs.retain_grad()

    loss_val = -torch.sum(y_onehot * torch.log(p_probs))
    loss_val.backward(retain_graph=True)

    # Analytical computations
    p_np = p_probs.detach().numpy().flatten()
    y_np = y_onehot.detach().numpy().flatten()
    x_np = x_input.detach().numpy()

    # 1. Softmax Jacobian
    jacobian_analytical = np.diag(p_np) - np.outer(p_np, p_np)

    # 2. Gradient w.r.t logits: p - y
    grad_z_analytical = (p_np - y_np).reshape(-1, 1)

    # 3. Gradient w.r.t weights: (p - y) x^T
    grad_w_analytical = grad_z_analytical @ x_np.T

    return (
        b_bias,
        grad_w_analytical,
        grad_z_analytical,
        jacobian_analytical,
        loss_val,
        m_features,
        n_classes,
        p_np,
        p_probs,
        target_idx,
        w_matrix,
        x_input,
        x_np,
        y_np,
        y_onehot,
        z_logits,
    )


@app.cell
def _(
    go,
    grad_w_analytical,
    jacobian_analytical,
    make_subplots,
    mo,
    np,
    p_np,
    y_np,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Softmax Jacobian Matrix: diag(p) - p p^T</b>",
            "<b>Weight Gradient Outer Product: (p - y) x^T</b>",
        ],
        horizontal_spacing=0.14,
    )

    class_names = ["Class 0", "Class 1", "Class 2"]
    feat_names = ["Feat 0", "Feat 1"]

    # Left: Softmax Jacobian Heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.round(jacobian_analytical, 3),
            x=class_names,
            y=class_names,
            colorscale="Blues",
            text=np.round(jacobian_analytical, 3),
            texttemplate="%{text}",
            colorbar=dict(title="dp_i / dz_j", x=0.44, len=0.8),
        ),
        row=1,
        col=1,
    )

    # Right: Gradient w.r.t W Heatmap
    fig.add_trace(
        go.Heatmap(
            z=np.round(grad_w_analytical, 3),
            x=feat_names,
            y=class_names,
            colorscale="RdBu_r",
            text=np.round(grad_w_analytical, 3),
            texttemplate="%{text}",
            colorbar=dict(title="dL / dW", x=1.02, len=0.8),
        ),
        row=1,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=40, t=70, b=40),
    )

    viz = mo.ui.plotly(fig)
    return class_names, feat_names, fig, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates the matrix structures powering multiclass gradient propagation:

                1. **Left Panel (Softmax Jacobian Matrix)**: The diagonal elements $p_i(1 - p_i)$ are strictly positive, representing the variance of individual class probabilities. The off-diagonal entries $-p_i p_j$ are negative, reflecting probability conservation: increasing logit $z_j$ inevitably depresses probability $p_i$.
                2. **Right Panel (Weight Gradient Outer Product)**: The resulting parameter gradient $\frac{\partial L}{\partial W} = (p - y) x^\top$. For the target class (Class 1), $p_1 - 1 < 0$ produces opposite-sign gradients that pull weights toward the input vector direction. For incorrect classes, $p_i - 0 > 0$ pushes weights away.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    b_bias,
    grad_w_analytical,
    grad_z_analytical,
    jacobian_analytical,
    loss_val,
    mo,
    np,
    p_np,
    pd,
    torch,
    w_matrix,
    y_np,
    z_logits,
):
    # Example 1: Numerical Validation: Analytical Formulas vs PyTorch Autograd
    autograd_grad_w = w_matrix.grad.detach().numpy()
    autograd_grad_z = z_logits.grad.detach().numpy().flatten()
    autograd_grad_b = b_bias.grad.detach().numpy().flatten()

    df_verification = pd.DataFrame(
        [
            {
                "Quantity": "Logit Gradient: dL / dz",
                "Analytical_Formula": "p - y",
                "Analytical_Values": str(grad_z_analytical.flatten().round(5)),
                "PyTorch_Autograd": str(autograd_grad_z.round(5)),
                "Max_Absolute_Diff": float(
                    np.max(np.abs(grad_z_analytical.flatten() - autograd_grad_z))
                ),
            },
            {
                "Quantity": "Weight Gradient: dL / dW",
                "Analytical_Formula": "(p - y) x^T",
                "Analytical_Values": str(grad_w_analytical.flatten().round(5)),
                "PyTorch_Autograd": str(autograd_grad_w.flatten().round(5)),
                "Max_Absolute_Diff": float(np.max(np.abs(grad_w_analytical - autograd_grad_w))),
            },
            {
                "Quantity": "Bias Gradient: dL / db",
                "Analytical_Formula": "p - y",
                "Analytical_Values": str(grad_z_analytical.flatten().round(5)),
                "PyTorch_Autograd": str(autograd_grad_b.round(5)),
                "Max_Absolute_Diff": float(
                    np.max(np.abs(grad_z_analytical.flatten() - autograd_grad_b))
                ),
            },
        ]
    )

    # Example 2: PyTorch Autograd Functional Jacobian vs Analytical Jacobian
    def softmax_wrapper(z):
        return torch.softmax(z, dim=0)

    autograd_jacobian = (
        torch.autograd.functional.jacobian(softmax_wrapper, z_logits).squeeze().detach().numpy()
    )

    df_jacobian = pd.DataFrame(
        {
            "Jacobian_Element": [f"J[{i},{j}]" for i in range(3) for j in range(3)],
            "Analytical_Value": jacobian_analytical.flatten().round(6),
            "Autograd_Functional": autograd_jacobian.flatten().round(6),
            "Difference": np.abs(jacobian_analytical.flatten() - autograd_jacobian.flatten()).round(
                9
            ),
        }
    )

    # Example 3: Numerical Stability Benchmark: Naive Softmax vs LogSumExp Trick
    extreme_logits = np.array([1000.0, 1002.0, 995.0])

    # Naive Softmax: Overflow occurs in exp(1000)
    with np.errstate(over="ignore", invalid="ignore"):
        exp_naive = np.exp(extreme_logits)
        p_naive = exp_naive / np.sum(exp_naive)
        loss_naive = -np.log(p_naive[1])

    # Numerically Stable Softmax: Subtract max(z)
    max_z = np.max(extreme_logits)
    exp_stable = np.exp(extreme_logits - max_z)
    p_stable = exp_stable / np.sum(exp_stable)
    loss_stable = -(extreme_logits[1] - max_z) + np.log(np.sum(exp_stable))

    df_stability = pd.DataFrame(
        [
            {
                "Implementation": "Naive Softmax (exp(z) / sum(exp(z)))",
                "Max_Logit": 1002.0,
                "Probabilities": str(p_naive),
                "Calculated_Loss": str(loss_naive),
                "Numerical_Status": "Failed (NaN / Inf Overflow)",
            },
            {
                "Implementation": "LogSumExp Stable (z - max(z))",
                "Max_Logit": 1002.0,
                "Probabilities": str(np.round(p_stable, 4)),
                "Calculated_Loss": f"{loss_stable:.4f}",
                "Numerical_Status": "Exact, Mathematically Stable",
            },
        ]
    )

    table_verif = mo.ui.table(df_verification)
    table_jac = mo.ui.table(df_jacobian)
    table_stab = mo.ui.table(df_stability)

    return (
        autograd_grad_b,
        autograd_grad_w,
        autograd_grad_z,
        autograd_jacobian,
        df_jacobian,
        df_stability,
        df_verification,
        exp_naive,
        exp_stable,
        extreme_logits,
        loss_naive,
        loss_stable,
        max_z,
        p_naive,
        p_stable,
        softmax_wrapper,
        table_jac,
        table_stab,
        table_verif,
    )


@app.cell
def _(mo, table_jac, table_stab, table_verif):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Analytical Matrix Calculus vs PyTorch Autograd

                Verifying that our derived analytical formulas for $\nabla_z L$, $\nabla_W L$, and $\nabla_b L$ match PyTorch backward execution down to floating-point precision:
                """
            ),
            table_verif,
            mo.md(
                r"""
                ### Example 2: Softmax Jacobian Matrix Exact Equivalence

                Comparing our analytical $\operatorname{diag}(p) - p p^\top$ formulation against `torch.autograd.functional.jacobian`:
                """
            ),
            table_jac,
            mo.md(
                r"""
                ### Example 3: The LogSumExp Numerical Stability Audit

                Demonstrating why production machine learning frameworks rely on the LogSumExp trick to prevent catastrophic float overflow:
                """
            ),
            table_stab,
        ]
    )


if __name__ == "__main__":
    app.run()
