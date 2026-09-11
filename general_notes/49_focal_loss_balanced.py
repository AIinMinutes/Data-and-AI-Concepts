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
    from scipy.special import expit
    from sklearn.datasets import make_classification
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    return (
        DataLoader,
        TensorDataset,
        average_precision_score,
        balanced_accuracy_score,
        expit,
        f1_score,
        go,
        make_classification,
        make_subplots,
        mo,
        nn,
        np,
        optim,
        pd,
        precision_score,
        recall_score,
        torch,
        train_test_split,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 48 Temperature Scaled Softmax](48_temperature_scaled_softmax.py) | [Index](../index.html) | [50 Scaled Dot-Product Attention →](50_scaled_dot_product_attention.py)

        # 49. Class-Balanced Focal Loss: Dynamically Down-Weighting Easy Negatives in Imbalanced Regimes

        ### Executive Summary

        In dense object detection, rare disease diagnostics, and financial fraud surveillance, class distributions are characterized by extreme skew, where the background (negative) class outnumbers the foreground (positive) class by ratios of $100:1$ to $100{,}000:1$. While individual easy negative examples each incur a tiny cross-entropy penalty, their sheer volume generates a cumulative gradient signal that drowns out the sparse gradient contributions of hard positive instances during stochastic gradient descent.

        **Focal Loss** (Lin et al., 2017) overcomes this structural failure mode by introducing a dynamic modulating factor $(1 - p_t)^\gamma$ into the standard binary cross-entropy objective. This factor automatically suppresses the loss and backpropagated gradients for well-classified examples ($p_t \gg 0.5$) while preserving gradients for misclassified, ambiguous, or hard boundary samples. Combined with an $\alpha$-balancing scalar, Focal Loss re-aligns gradient dynamics toward the true minority decision boundary without requiring artificial sampling tricks or heuristic bootstrapping.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Gradient Mechanics

        ### 1. The Pathology of Standard Binary Cross-Entropy

        Consider binary classification with target $y \in \{0, 1\}$ and predicted probability $p = \sigma(z) \in (0, 1)$, where $z \in \mathbb{R}$ is the unnormalized logit and $\sigma(z) = (1 + e^{-z})^{-1}$. For notational convenience, define the ground-truth probability $p_t$:

        $$p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

        The conventional Binary Cross-Entropy (BCE) loss is expressed concisely as:

        $$\text{CE}(p, y) = \text{CE}(p_t) = -\ln(p_t)$$

        In extreme imbalance regimes (e.g., $99\%$ negative background instances), the vast majority of negative samples are easily distinguishable ($p_t \ge 0.99$). Although the loss for a single easy sample is modest ($-\ln(0.99) \approx 0.01005$), summing over $N_{\text{neg}} = 10^5$ examples yields:

        $$\sum_{i=1}^{N_{\text{neg}}} \text{CE}(p_{t, i}) \approx 10^5 \times 0.01005 = 1{,}005$$

        Meanwhile, 100 rare positive examples ($y=1$), even if completely misclassified with $p_t = 0.01$, yield only:

        $$\sum_{j=1}^{N_{\text{pos}}} \text{CE}(p_{t, j}) \approx 100 \times (-\ln(0.01)) = 100 \times 4.605 = 460.5$$

        The easy negative examples contribute over $68\%$ of the total loss and dominate the optimization surface, driving network weights toward degenerate solutions that predict the majority class.

        ### 2. The Focal Loss Formulation

        To neutralize easy example dominance, Lin et al. (2017) introduce a power-law modulating term $(1 - p_t)^\gamma$ parameterized by the focusing factor $\gamma \ge 0$, combined with an $\alpha_t$ class-weighting factor:

        $$\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$$

        where the class balance scalar $\alpha_t \in [0, 1]$ is defined as:

        $$\alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

        #### Behavior Across Regimes:
        - When an instance is misclassified and $p_t \to 0$, the modulating factor $(1 - p_t)^\gamma \to 1$, leaving the loss virtually unaltered compared to standard cross-entropy.
        - As an instance becomes well-classified ($p_t \to 1$), the modulating factor $(1 - p_t)^\gamma \to 0$, exponentially attenuating the loss.
        - When $\gamma = 0$ and $\alpha = 0.5$, Focal Loss reduces identically to standard Binary Cross-Entropy.

        ### 3. Quantitative Impact of the Focusing Parameter $\gamma$

        Let us evaluate the loss scaling factor $(1 - p_t)^\gamma$ for an easy example with $p_t = 0.99$:

        - For $\gamma = 0$: $(1 - 0.99)^0 = 1.0$ (no down-weighting)
        - For $\gamma = 1$: $(1 - 0.99)^1 = 0.01$ ($100\times$ suppression)
        - For $\gamma = 2$: $(1 - 0.99)^2 = 0.0001$ ($10{,}000\times$ suppression)
        - For $\gamma = 5$: $(1 - 0.99)^5 = 10^{-10}$ (complete suppression)

        Under $\gamma = 2$, the $100{,}000$ easy background negatives that previously accumulated $1{,}005$ loss units now contribute a total loss of only:

        $$10^5 \times 0.0001 \times 0.01005 \approx 0.1005$$

        The hard positives now dominate the loss ($460.5 \text{ vs } 0.10$), completely redirecting the model's capacity toward learning the minority class.

        ### 4. Gradient Derivation with Respect to Logits

        To understand how Focal Loss impacts backpropagation, we compute the analytical gradient with respect to the input logit $z$. Let $y^* \in \{-1, +1\}$ such that $y^* = 2y - 1$. Then $p_t = \sigma(y^* z) = (1 + e^{-y^* z})^{-1}$. The derivative of $p_t$ with respect to $z$ is:

        $$\frac{\partial p_t}{\partial z} = y^* p_t (1 - p_t)$$

        Differentiating $\text{FL}(p_t) = -\alpha_t (1 - p_t)^\gamma \ln(p_t)$ with respect to $z$ using the product rule:

        $$\frac{\partial \text{FL}}{\partial z} = \frac{\partial \text{FL}}{\partial p_t} \cdot \frac{\partial p_t}{\partial z}$$

        $$\frac{\partial \text{FL}}{\partial p_t} = -\alpha_t \left[ -\gamma (1 - p_t)^{\gamma - 1} \ln(p_t) + \frac{(1 - p_t)^\gamma}{p_t} \right]$$

        Multiplying by $\frac{\partial p_t}{\partial z} = y^* p_t (1 - p_t)$:

        $$\frac{\partial \text{FL}}{\partial z} = y^* \alpha_t (1 - p_t)^\gamma \left[ \gamma p_t \ln(p_t) + p_t - 1 \right]$$

        For $\gamma = 0$ (Standard BCE with $\alpha_t = 1$), this simplifies directly to the classical error residual:

        $$\left.\frac{\partial \text{FL}}{\partial z}\right|_{\gamma=0, \alpha_t=1} = y^* (p_t - 1) = p - y$$

        For $\gamma > 0$, as $p_t \to 1$ (well-classified easy samples), the factor $(1 - p_t)^\gamma \to 0$ aggressively forces the gradient to zero. The network receives negligible parameter updates from confident predictions, dedicating its entire gradient budget to unresolved boundary cases.
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Generate probability grid for visualization
    pt_grid = np.linspace(0.001, 0.999, 500)
    gamma_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    gamma_colors = {
        0.0: "#1D4ED8",  # Standard CE: Blue
        0.5: "#0D9488",  # Teal
        1.0: "#F59E0B",  # Amber
        2.0: "#DC2626",  # Red (RetinaNet standard)
        5.0: "#7C3AED",  # Purple
    }

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Focal Loss Curves vs True Probability pt</b>",
            "<b>Absolute Gradient Magnitude |dFL/dz| vs pt</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Focal Loss curves
    for g in gamma_values:
        fl_curve = -((1.0 - pt_grid) ** g) * np.log(pt_grid)
        fig.add_trace(
            go.Scatter(
                x=pt_grid,
                y=fl_curve,
                mode="lines",
                line=dict(color=gamma_colors[g], width=2.5),
                name=f"gamma = {g}" + (" (CE)" if g == 0.0 else ""),
            ),
            row=1,
            col=1,
        )

    # Panel 2: Gradient magnitude |dFL/dz| assuming y=1 (pt = p)
    for g in gamma_values:
        # Gradient formula: (1 - pt)^gamma * |gamma * pt * ln(pt) + pt - 1|
        grad_mag = ((1.0 - pt_grid) ** g) * np.abs(g * pt_grid * np.log(pt_grid) + pt_grid - 1.0)
        fig.add_trace(
            go.Scatter(
                x=pt_grid,
                y=grad_mag,
                mode="lines",
                line=dict(color=gamma_colors[g], width=2.5),
                name=f"grad (gamma = {g})",
                showlegend=False,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Probability of Ground-Truth Class pt", row=1, col=1)
    fig.update_yaxes(title_text="Focal Loss FL(pt)", range=[0, 5.0], row=1, col=1)
    fig.update_xaxes(title_text="Probability of Ground-Truth Class pt", row=1, col=2)
    fig.update_yaxes(title_text="Logit Gradient Magnitude |dFL/dz|", range=[0, 1.05], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return fig, gamma_colors, gamma_values, pt_grid, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates the core mechanisms of Focal Loss:

                1. **Left Panel (Loss Attenuation Curves)**: At $\gamma = 0$ (blue curve, standard cross-entropy), an example with $p_t = 0.6$ still produces substantial loss ($\approx 0.51$). Under $\gamma = 2$ (red curve), the loss for the same example drops to $(1 - 0.6)^2 \times 0.51 \approx 0.082$. For easy examples ($p_t > 0.8$), the loss under $\gamma \ge 2$ collapses completely to zero.
                2. **Right Panel (Gradient Magnitude)**: Standard cross-entropy ($\gamma = 0$) exhibits a strictly linear logit error gradient magnitude $|p - 1|$. In stark contrast, when $\gamma = 2$ or $5$, the gradient signal for easy examples ($p_t \to 1$) plunges to zero exponentially fast. This effectively turns off backpropagation for settled samples and focuses gradient descent exclusively on difficult boundary examples.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    DataLoader,
    TensorDataset,
    average_precision_score,
    balanced_accuracy_score,
    expit,
    f1_score,
    make_classification,
    mo,
    nn,
    np,
    optim,
    pd,
    precision_score,
    recall_score,
    torch,
    train_test_split,
):
    # Vectorized NumPy Focal Loss implementation
    def numpy_focal_loss(y_true, logits, gamma=2.0, alpha=0.25):
        probs = expit(logits)
        pt = np.where(y_true == 1, probs, 1.0 - probs)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        loss = -alpha_t * ((1.0 - pt) ** gamma) * np.log(np.maximum(pt, 1e-12))
        return np.mean(loss)

    def numpy_focal_grad(y_true, logits, gamma=2.0, alpha=0.25):
        probs = expit(logits)
        pt = np.where(y_true == 1, probs, 1.0 - probs)
        alpha_t = np.where(y_true == 1, alpha, 1.0 - alpha)
        # y_star = +1 for y=1, -1 for y=0
        y_star = np.where(y_true == 1, 1.0, -1.0)
        grad = y_star * alpha_t * ((1.0 - pt) ** gamma) * (gamma * pt * np.log(np.maximum(pt, 1e-12)) + pt - 1.0)
        return grad

    # Numerical gradient check using finite differences
    _eps = 1e-6
    _test_y = np.array([1, 0, 1, 0])
    _test_z = np.array([1.5, -2.0, -0.8, 3.2])
    _analytical_grads = numpy_focal_grad(_test_y, _test_z, gamma=2.0, alpha=0.25)
    _numerical_grads = []
    for _i in range(len(_test_z)):
        _z_plus = _test_z.copy()
        _z_plus[_i] += _eps
        _z_minus = _test_z.copy()
        _z_minus[_i] -= _eps
        # Scale by N because focal_loss computes mean
        _loss_plus = numpy_focal_loss(_test_y, _z_plus, gamma=2.0, alpha=0.25) * len(_test_z)
        _loss_minus = numpy_focal_loss(_test_y, _z_minus, gamma=2.0, alpha=0.25) * len(_test_z)
        _num_g = (_loss_plus - _loss_minus) / (2.0 * _eps)
        _numerical_grads.append(_num_g)

    _numerical_grads = np.array(_numerical_grads)
    _grad_errors = np.abs(_analytical_grads - _numerical_grads)

    df_grad_check = pd.DataFrame(
        {
            "True_Label": _test_y,
            "Input_Logit": _test_z,
            "Analytical_Grad": np.round(_analytical_grads, 6),
            "Numerical_Grad": np.round(_numerical_grads, 6),
            "Abs_Difference": [f"{err:.2e}" for err in _grad_errors],
            "Status": ["Verified Match" if err < 1e-4 else "Discrepancy" for err in _grad_errors],
        }
    )

    # Synthetic highly imbalanced classification experiment (95% Negative, 5% Positive)
    np.random.seed(42)
    torch.manual_seed(42)

    X_raw, y_raw = make_classification(
        n_samples=2500,
        n_features=16,
        n_informative=10,
        n_classes=2,
        weights=[0.95, 0.05],
        flip_y=0.01,
        random_state=42,
    )

    X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
        X_raw, y_raw, test_size=0.3, stratify=y_raw, random_state=42
    )

    # PyTorch Tensors and Loader
    _X_tr_t = torch.tensor(X_train_raw, dtype=torch.float32)
    _y_tr_t = torch.tensor(y_train_raw, dtype=torch.float32).unsqueeze(1)
    _X_te_t = torch.tensor(X_test_raw, dtype=torch.float32)

    _ds = TensorDataset(_X_tr_t, _y_tr_t)
    _loader = DataLoader(_ds, batch_size=64, shuffle=True)

    # PyTorch Model
    class SimpleClassifier(nn.Module):
        def __init__(self, in_features=16):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_features, 32),
                nn.ReLU(),
                nn.Linear(32, 16),
                nn.ReLU(),
                nn.Linear(16, 1),
            )

        def forward(self, x):
            return self.net(x)

    # PyTorch Focal Loss Module
    class PyTorchFocalLoss(nn.Module):
        def __init__(self, gamma=2.0, alpha=0.25):
            super().__init__()
            self.gamma = gamma
            self.alpha = alpha

        def forward(self, logits, targets):
            probs = torch.sigmoid(logits)
            pt = torch.where(targets == 1.0, probs, 1.0 - probs)
            alpha_t = torch.where(targets == 1.0, self.alpha, 1.0 - self.alpha)
            loss = -alpha_t * ((1.0 - pt) ** self.gamma) * torch.log(torch.clamp(pt, min=1e-12))
            return torch.mean(loss)

    # Training routine
    def train_classifier(criterion, epochs=35, lr=0.01):
        torch.manual_seed(42)
        model = SimpleClassifier()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model.train()
        for _ep in range(epochs):
            for bx, by in _loader:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
        model.eval()
        return model

    # 1. Standard BCE Model
    model_bce = train_classifier(nn.BCEWithLogitsLoss())
    # 2. Focal Loss Model (gamma=2, alpha=0.25)
    model_fl = train_classifier(PyTorchFocalLoss(gamma=2.0, alpha=0.25))

    with torch.no_grad():
        test_logits_bce = model_bce(_X_te_t).squeeze().numpy()
        test_probs_bce = expit(test_logits_bce)
        test_preds_bce = (test_probs_bce >= 0.5).astype(int)

        test_logits_fl = model_fl(_X_te_t).squeeze().numpy()
        test_probs_fl = expit(test_logits_fl)
        test_preds_fl = (test_probs_fl >= 0.5).astype(int)

    def evaluate_predictions(y_true, y_pred, y_prob):
        return {
            "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
            "Precision_Minority": precision_score(y_true, y_pred, zero_division=0),
            "Recall_Minority": recall_score(y_true, y_pred, zero_division=0),
            "F1_Minority": f1_score(y_true, y_pred, zero_division=0),
            "PR_AUC_AP": average_precision_score(y_true, y_prob),
        }

    bce_metrics = evaluate_predictions(y_test_raw, test_preds_bce, test_probs_bce)
    fl_metrics = evaluate_predictions(y_test_raw, test_preds_fl, test_probs_fl)

    df_imbalance_eval = pd.DataFrame(
        [
            {
                "Objective_Function": "Binary Cross-Entropy (BCE)",
                "Hyperparameters": "gamma = 0, alpha = 0.5",
                "Balanced_Accuracy": f"{bce_metrics['Balanced_Accuracy'] * 100:.2f}%",
                "Precision (Class 1)": f"{bce_metrics['Precision_Minority'] * 100:.2f}%",
                "Recall (Class 1)": f"{bce_metrics['Recall_Minority'] * 100:.2f}%",
                "F1-Score (Class 1)": f"{bce_metrics['F1_Minority'] * 100:.2f}%",
                "PR-AUC (Avg Precision)": f"{bce_metrics['PR_AUC_AP'] * 100:.2f}%",
            },
            {
                "Objective_Function": "Class-Balanced Focal Loss",
                "Hyperparameters": "gamma = 2.0, alpha = 0.25",
                "Balanced_Accuracy": f"{fl_metrics['Balanced_Accuracy'] * 100:.2f}%",
                "Precision (Class 1)": f"{fl_metrics['Precision_Minority'] * 100:.2f}%",
                "Recall (Class 1)": f"{fl_metrics['Recall_Minority'] * 100:.2f}%",
                "F1-Score (Class 1)": f"{fl_metrics['F1_Minority'] * 100:.2f}%",
                "PR-AUC (Avg Precision)": f"{fl_metrics['PR_AUC_AP'] * 100:.2f}%",
            },
        ]
    )

    # Example 3: Gradient Breakdown Analysis (Easy Negatives vs Hard Positives)
    # Compute cumulative gradient contribution on the entire training set
    with torch.no_grad():
        tr_logits = model_bce(_X_tr_t).squeeze().numpy()
        tr_probs = expit(tr_logits)
        tr_pt = np.where(y_train_raw == 1, tr_probs, 1.0 - tr_probs)

        # Gradients under BCE (gamma=0, alpha=1.0)
        grad_bce = np.abs(tr_probs - y_train_raw)

        # Gradients under Focal Loss (gamma=2, alpha=0.25)
        grad_fl = np.abs(numpy_focal_grad(y_train_raw, tr_logits, gamma=2.0, alpha=0.25))

    is_easy_neg = (y_train_raw == 0) & (tr_pt >= 0.8)
    is_hard_or_pos = ~is_easy_neg

    bce_easy_neg_grad_share = np.sum(grad_bce[is_easy_neg]) / np.sum(grad_bce) * 100.0
    fl_easy_neg_grad_share = np.sum(grad_fl[is_easy_neg]) / np.sum(grad_fl) * 100.0

    bce_target_grad_share = np.sum(grad_bce[is_hard_or_pos]) / np.sum(grad_bce) * 100.0
    fl_target_grad_share = np.sum(grad_fl[is_hard_or_pos]) / np.sum(grad_fl) * 100.0

    df_grad_shares = pd.DataFrame(
        [
            {
                "Loss_Function": "Standard Binary Cross-Entropy",
                "Easy_Negatives_Count": f"{int(np.sum(is_easy_neg))} ({np.mean(is_easy_neg) * 100:.1f}%)",
                "Easy_Negatives_Gradient_Share": f"{bce_easy_neg_grad_share:.2f}%",
                "Hard_&_Minority_Gradient_Share": f"{bce_target_grad_share:.2f}%",
                "Optimization_Regime": "Negatives Dominate Optimization Gradient",
            },
            {
                "Loss_Function": "Class-Balanced Focal Loss (gamma=2)",
                "Easy_Negatives_Count": f"{int(np.sum(is_easy_neg))} ({np.mean(is_easy_neg) * 100:.1f}%)",
                "Easy_Negatives_Gradient_Share": f"{fl_easy_neg_grad_share:.2f}%",
                "Hard_&_Minority_Gradient_Share": f"{fl_target_grad_share:.2f}%",
                "Optimization_Regime": "Minority / Hard Cases Drive Updates",
            },
        ]
    )

    table_grad_check = mo.ui.table(df_grad_check)
    table_eval = mo.ui.table(df_imbalance_eval)
    table_shares = mo.ui.table(df_grad_shares)

    return (
        table_eval,
        table_grad_check,
        table_shares,
    )


@app.cell
def _(mo, table_eval, table_grad_check, table_shares):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Vectorized Focal Loss and Exact Gradient Verification

                Verifying our analytical derivative $\frac{\partial \text{FL}}{\partial z}$ against two-sided finite difference numerical approximations:
                """
            ),
            table_grad_check,
            mo.md(
                r"""
                ### Example 2: PyTorch Benchmark on 95:5 Skewed Class Distribution

                Comparing holdout test performance of a neural network trained with standard Cross-Entropy vs Focal Loss ($\gamma = 2.0, \alpha = 0.25$):
                """
            ),
            table_eval,
            mo.md(
                r"""
                ### Example 3: Empirical Gradient Budget Allocation (Easy Negatives vs Hard Samples)

                Auditing how Focal Loss drastically re-allocates the backpropagated gradient norm away from easy background negatives and toward hard minority samples:
                """
            ),
            table_shares,
        ]
    )


if __name__ == "__main__":
    app.run()
