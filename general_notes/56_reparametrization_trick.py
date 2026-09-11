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
    import torch
    import torch.nn as nn

    return go, make_subplots, mo, nn, np, pd, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 55 Perplexity](55_perplexity.py) | [Index](../index.html) | [57 Autoencoder →](57_autoencoder.py)

        # 56. The Reparameterization Trick: Backpropagating Through Stochastic Nodes in Latent Variable Models

        ### Executive Summary

        In deep generative modeling—specifically Variational Autoencoders (VAEs), diffusion models, and variational reinforcement learning—the network architecture requires sampling a latent variable $z$ from a learned conditional distribution $q_\phi(z \mid x)$. However, standard Monte Carlo sampling is an inherently non-differentiable operation: because sample generation cannot provide an analytical derivative with respect to distribution parameters ($\phi$), standard backpropagation halts at the stochastic node, preventing the decoder's reconstruction loss from updating the encoder.

        The **Reparameterization Trick** (Kingma & Welling, 2013; Rezende et al., 2014) overcomes this fundamental optimization barrier. By decoupling the randomness into an external, parameter-free noise variable $\epsilon \sim \mathcal{N}(0, I)$ and expressing the latent code as a deterministic, differentiable transformation $z = g_\phi(\epsilon, x) = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$, gradients flow seamlessly through the network via the multivariable chain rule with dramatically lower estimator variance than score-function (REINFORCE) methods.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Gradient Mechanics

        ### 1. The Stochastic Gradient Barrier in Latent Variable Models

        In Variational Autoencoders, the objective is to maximize the Evidence Lower Bound (ELBO):

        $$\mathcal{L}_{\text{ELBO}}(\theta, \phi; x) = \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ \ln p_\theta(x \mid z) \right] - D_{\text{KL}}\left( q_\phi(z \mid x) \parallel p(z) \right)$$

        To train the recognition model (encoder) parameterized by $\phi$, we must evaluate the gradient of an expectation with respect to the distribution's own parameters:

        $$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z \mid x)} \left[ f(z) \right]$$

        where $f(z) = \ln p_\theta(x \mid z)$.

        Because the probability density function $q_\phi(z \mid x)$ depends directly on $\phi$, the gradient operator **cannot** simply be swapped with the expectation:

        $$\nabla_\phi \mathbb{E}_{z \sim q_\phi(z \mid x)} [f(z)] \neq \mathbb{E}_{z \sim q_\phi(z \mid x)} [\nabla_\phi f(z)]$$

        Direct sampling $z \sim q_\phi(z \mid x)$ creates a non-differentiable bottleneck where $\frac{\partial z}{\partial \phi}$ is undefined, severing gradient flow between the decoder and the encoder.

        ---

        ### 2. Score Function Estimator vs Pathwise Derivative

        Historically, two primary mathematical frameworks have addressed this challenge:

        #### A. The Score Function Estimator (REINFORCE / Likelihood Ratio)
        Using the identity $\nabla_\phi q_\phi(z) = q_\phi(z) \nabla_\phi \ln q_\phi(z)$:

        $$\nabla_\phi \mathbb{E}_{q_\phi}[f(z)] = \nabla_\phi \int q_\phi(z) f(z) \, dz = \int \nabla_\phi q_\phi(z) f(z) \, dz = \int q_\phi(z) \left[ f(z) \nabla_\phi \ln q_\phi(z) \right] \, dz = \mathbb{E}_{q_\phi} \left[ f(z) \nabla_\phi \ln q_\phi(z) \right]$$

        - **Advantage**: Applies broadly to discrete and non-differentiable distributions.
        - **Critical Flaw**: Catastrophic Monte Carlo variance. The estimator requires complex control variates (baselines) and millions of samples to yield a stable optimization signal.

        #### B. The Reparameterization Trick (Pathwise Derivative)
        Suppose we can express the random variable $z \sim q_\phi(z \mid x)$ as a deterministic, differentiable function $g_\phi(\epsilon, x)$ of an auxiliary noise variable $\epsilon$ drawn from a fixed distribution $p(\epsilon)$ that contains **no parameters $\phi$**:

        $$z = g_\phi(\epsilon, x), \qquad \epsilon \sim p(\epsilon)$$

        Under this change of variables, the expectation is reformulated over the parameter-free distribution $p(\epsilon)$:

        $$\mathbb{E}_{z \sim q_\phi(z \mid x)} [f(z)] = \mathbb{E}_{\epsilon \sim p(\epsilon)} \left[ f\left( g_\phi(\epsilon, x) \right) \right]$$

        Because $p(\epsilon)$ has no dependence on $\phi$, Leibniz's integral rule permits moving the gradient operator directly inside the expectation:

        $$\nabla_\phi \mathbb{E}_{z \sim q_\phi}[f(z)] = \nabla_\phi \int p(\epsilon) f(g_\phi(\epsilon, x)) \, d\epsilon = \int p(\epsilon) \nabla_\phi f(g_\phi(\epsilon, x)) \, d\epsilon = \mathbb{E}_{\epsilon \sim p(\epsilon)} \left[ \nabla_\phi f(g_\phi(\epsilon, x)) \right]$$

        Applying the multivariable chain rule:

        $$\nabla_\phi f(g_\phi(\epsilon, x)) = \left. \nabla_z f(z) \right|_{z=g_\phi(\epsilon, x)} \cdot \nabla_\phi g_\phi(\epsilon, x)$$

        The empirical Monte Carlo estimator using a single sample $\epsilon \sim p(\epsilon)$ is:

        $$\widehat{\nabla}_\phi f \approx \nabla_z f(z) \cdot \nabla_\phi g_\phi(\epsilon, x)$$

        ---

        ### 3. Gaussian Latent Space Implementation

        For a multivariate Gaussian with diagonal covariance $q_\phi(z \mid x) = \mathcal{N}(\mu, \operatorname{diag}(\sigma^2))$:

        $$z = g(\mu, \sigma, \epsilon) = \mu + \sigma \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I)$$

        The partial derivatives are straightforward:

        $$\frac{\partial z}{\partial \mu} = I, \qquad \frac{\partial z}{\partial \sigma} = \operatorname{diag}(\epsilon)$$

        To enforce strictly positive standard deviations without constrained optimization, neural networks parameterize the log-variance $s = \ln(\sigma^2)$:

        $$\sigma = \exp\left( \frac{1}{2} s \right) \implies z = \mu + \exp\left( \frac{1}{2} s \right) \odot \epsilon$$

        The gradient with respect to log-variance $s$ is:

        $$\frac{\partial z}{\partial s} = \frac{1}{2} \exp\left( \frac{1}{2} s \right) \odot \epsilon = \frac{1}{2} \sigma \odot \epsilon$$

        Both $\mu$ and $s$ receive smooth, low-variance backpropagated gradients directly scaled by the decoder's loss surface.
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Panel 1: Simulation Comparing Monte Carlo Gradient Estimator Variance
    # Objective: Minimize E_{z ~ N(mu, sigma^2)} [ (z - 3.0)^2 ]
    # True optimum: mu = 3.0, sigma -> 0
    # Analytic gradient wrt mu: E[ 2(z - 3) ] = 2(mu - 3)
    np.random.seed(42)
    n_steps = 150
    mu_init = 0.5
    sigma_val = 1.8

    # True gradient
    true_grad = 2.0 * (mu_init - 3.0)  # -5.0

    # Simulate single-sample gradient estimates across 150 trials
    pathwise_grads = []
    reinforce_grads = []

    for _ in range(n_steps):
        eps = np.random.normal(0, 1)
        z = mu_init + sigma_val * eps

        # Pathwise derivative: d/dmu [ (z - 3)^2 ] = 2(z - 3) * (dz/dmu) = 2(z - 3) * 1
        g_pathwise = 2.0 * (z - 3.0)
        pathwise_grads.append(g_pathwise)

        # REINFORCE: (z - 3)^2 * d/dmu [ ln q(z; mu, sigma) ]
        # ln q(z) = -0.5 * ((z - mu) / sigma)^2 - ln(sigma*sqrt(2pi))
        # d/dmu ln q(z) = (z - mu) / (sigma^2) = eps / sigma
        cost = (z - 3.0) ** 2
        score_mu = (z - mu_init) / (sigma_val**2)
        g_reinforce = cost * score_mu
        reinforce_grads.append(g_reinforce)

    pathwise_grads = np.array(pathwise_grads)
    reinforce_grads = np.array(reinforce_grads)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Single-Sample Gradient Variance: Pathwise vs REINFORCE</b>",
            "<b>Cumulative Moving Average Convergence to True Gradient (-5.0)</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Scatter of raw single-sample gradient estimates
    trial_idx = np.arange(1, n_steps + 1)
    fig.add_trace(
        go.Scatter(
            x=trial_idx,
            y=reinforce_grads,
            mode="lines",
            line=dict(color="#DC2626", width=1.5),
            name=f"REINFORCE (Var={np.var(reinforce_grads):.1f})",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=trial_idx,
            y=pathwise_grads,
            mode="lines",
            line=dict(color="#1D4ED8", width=2.5),
            name=f"Reparameterization (Var={np.var(pathwise_grads):.1f})",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=true_grad,
        line=dict(color="#111827", width=1.5, dash="dash"),
        annotation_text=f"True Grad = {true_grad:.1f}",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Panel 2: Running Cumulative Mean
    cum_pathwise = np.cumsum(pathwise_grads) / trial_idx
    cum_reinforce = np.cumsum(reinforce_grads) / trial_idx

    fig.add_trace(
        go.Scatter(
            x=trial_idx,
            y=cum_reinforce,
            mode="lines",
            line=dict(color="#DC2626", width=2.0),
            name="REINFORCE Cumulative Mean",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=trial_idx,
            y=cum_pathwise,
            mode="lines",
            line=dict(color="#1D4ED8", width=2.5),
            name="Reparameterization Cumulative Mean",
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_hline(
        y=true_grad,
        line=dict(color="#111827", width=1.5, dash="dash"),
        annotation_text="True Gradient",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Monte Carlo Sample Index", row=1, col=1)
    fig.update_yaxes(title_text="Estimated Gradient dCost/d_mu", range=[-35, 35], row=1, col=1)
    fig.update_xaxes(title_text="Number of Averaged Samples", row=1, col=2)
    fig.update_yaxes(title_text="Cumulative Mean Gradient", range=[-12, 5], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        cost,
        cum_pathwise,
        cum_reinforce,
        eps,
        fig,
        g_pathwise,
        g_reinforce,
        mu_init,
        n_steps,
        pathwise_grads,
        reinforce_grads,
        score_mu,
        sigma_val,
        trial_idx,
        true_grad,
        viz,
        z,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below benchmarks estimator variance between the Pathwise Reparameterization Trick and the REINFORCE Score Function:

                1. **Left Panel (Raw Gradient Variance)**: Single-sample gradient estimates under REINFORCE (red curve) oscillate wildly between $-30$ and $+30$ (variance $\approx 120+$). In stark contrast, the Reparameterization Trick (blue curve) maintains a tightly bounded trajectory (variance $\approx 13$), providing a reliable directional signal at every training step.
                2. **Right Panel (Cumulative Mean Convergence)**: The Reparameterization estimator locks onto the true analytical gradient ($-5.0$) within fewer than $15$ samples, whereas the REINFORCE estimator exhibits prolonged instability and erratic wandering.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(mo, nn, np, pd, torch):
    # Vectorized NumPy Reparameterization with Analytical Finite Difference Check
    def numpy_reparameterize(mu, logvar, eps=None):
        std = np.exp(0.5 * logvar)
        if eps is None:
            eps = np.random.normal(0, 1, size=mu.shape)
        z = mu + std * eps
        return z, eps

    # Target loss: f(z) = (z - 2.5)^2 + 0.5 * z^3
    def target_cost(z):
        return (z - 2.5) ** 2 + 0.5 * (z**3)

    def target_cost_grad_z(z):
        return 2.0 * (z - 2.5) + 1.5 * (z**2)

    # Gradient check using fixed epsilon
    np.random.seed(1337)
    _test_mu = np.array([1.2, -0.8, 2.5])
    _test_logvar = np.array([0.4, -1.2, 0.0])
    _fixed_eps = np.array([0.65, -1.10, 0.35])

    _z, _ = numpy_reparameterize(_test_mu, _test_logvar, eps=_fixed_eps)
    _dz = target_cost_grad_z(_z)

    # Analytical gradients
    _std = np.exp(0.5 * _test_logvar)
    _grad_mu_analytic = _dz * 1.0
    _grad_logvar_analytic = _dz * (0.5 * _std * _fixed_eps)

    # Finite difference numerical gradients
    _delta = 1e-6
    _num_grad_mu = []
    _num_grad_logvar = []

    for _i in range(len(_test_mu)):
        # Mu perturbation
        _m_plus = _test_mu.copy()
        _m_plus[_i] += _delta
        _z_plus, _ = numpy_reparameterize(_m_plus, _test_logvar, eps=_fixed_eps)

        _m_minus = _test_mu.copy()
        _m_minus[_i] -= _delta
        _z_minus, _ = numpy_reparameterize(_m_minus, _test_logvar, eps=_fixed_eps)

        _num_grad_mu.append((target_cost(_z_plus[_i]) - target_cost(_z_minus[_i])) / (2 * _delta))

        # Logvar perturbation
        _lv_plus = _test_logvar.copy()
        _lv_plus[_i] += _delta
        _z_lv_plus, _ = numpy_reparameterize(_test_mu, _lv_plus, eps=_fixed_eps)

        _lv_minus = _test_logvar.copy()
        _lv_minus[_i] -= _delta
        _z_lv_minus, _ = numpy_reparameterize(_test_mu, _lv_minus, eps=_fixed_eps)

        _num_grad_logvar.append((target_cost(_z_lv_plus[_i]) - target_cost(_z_lv_minus[_i])) / (2 * _delta))

    _num_grad_mu = np.array(_num_grad_mu)
    _num_grad_logvar = np.array(_num_grad_logvar)

    _err_mu = np.abs(_grad_mu_analytic - _num_grad_mu)
    _err_logvar = np.abs(_grad_logvar_analytic - _num_grad_logvar)

    df_grad_check = pd.DataFrame(
        [
            {
                "Parameter": f"mu[{idx}]",
                "Analytical_Grad": f"{_grad_mu_analytic[idx]:.6f}",
                "Numerical_Grad": f"{_num_grad_mu[idx]:.6f}",
                "Absolute_Difference": f"{_err_mu[idx]:.2e}",
                "Verification_Status": "Exact Gradient Verified" if _err_mu[idx] < 1e-4 else "Discrepancy",
            }
            for idx in range(3)
        ]
        + [
            {
                "Parameter": f"logvar[{idx}]",
                "Analytical_Grad": f"{_grad_logvar_analytic[idx]:.6f}",
                "Numerical_Grad": f"{_num_grad_logvar[idx]:.6f}",
                "Absolute_Difference": f"{_err_logvar[idx]:.2e}",
                "Verification_Status": "Exact Gradient Verified" if _err_logvar[idx] < 1e-4 else "Discrepancy",
            }
            for idx in range(3)
        ]
    )

    # Example 2: PyTorch Production VAE Latent Module
    class DifferentiableVAELatent(nn.Module):
        def __init__(self, in_features=16, latent_dim=4):
            super().__init__()
            self.fc_mu = nn.Linear(in_features, latent_dim)
            self.fc_logvar = nn.Linear(in_features, latent_dim)

        def reparameterize(self, mu, logvar):
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std

        def forward(self, x):
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            z = self.reparameterize(mu, logvar)
            return z, mu, logvar

    torch.manual_seed(42)
    latent_module = DifferentiableVAELatent(in_features=16, latent_dim=4)
    dummy_encoder_output = torch.randn(8, 16, requires_grad=True)

    z_sample, mu_out, logvar_out = latent_module(dummy_encoder_output)
    # Simulate a decoder loss depending on z
    dummy_loss = (z_sample**2).sum()
    dummy_loss.backward()

    enc_grad_norm = dummy_encoder_output.grad.norm().item()
    mu_weight_grad = latent_module.fc_mu.weight.grad.norm().item()
    logvar_weight_grad = latent_module.fc_logvar.weight.grad.norm().item()

    df_pytorch_audit = pd.DataFrame(
        [
            {
                "Component": "Latent Sample Tensor z",
                "Tensor_Shape": f"{tuple(z_sample.shape)}",
                "Gradient_State": "Active in Computational Graph",
            },
            {
                "Component": "Input Encoder Activation Gradient",
                "Tensor_Shape": f"{tuple(dummy_encoder_output.shape)}",
                "Gradient_State": f"Flowing Smoothly (Norm: {enc_grad_norm:.4f})",
            },
            {
                "Component": "Encoder Mu Linear Layer Weights",
                "Tensor_Shape": f"{tuple(latent_module.fc_mu.weight.shape)}",
                "Gradient_State": f"Updated (Grad Norm: {mu_weight_grad:.4f})",
            },
            {
                "Component": "Encoder Logvar Linear Layer Weights",
                "Tensor_Shape": f"{tuple(latent_module.fc_logvar.weight.shape)}",
                "Gradient_State": f"Updated (Grad Norm: {logvar_weight_grad:.4f})",
            },
        ]
    )

    table_grad = mo.ui.table(df_grad_check)
    table_pt = mo.ui.table(df_pytorch_audit)

    return (
        DifferentiableVAELatent,
        df_grad_check,
        df_pytorch_audit,
        dummy_encoder_output,
        dummy_loss,
        enc_grad_norm,
        latent_module,
        logvar_out,
        logvar_weight_grad,
        mu_out,
        mu_weight_grad,
        numpy_reparameterize,
        table_grad,
        table_pt,
        target_cost,
        target_cost_grad_z,
        z_sample,
    )


@app.cell
def _(mo, table_grad, table_pt):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Analytical vs Finite Difference Gradient Verification

                Validating exact analytical pathwise derivatives $\frac{\partial z}{\partial \mu} = 1$ and $\frac{\partial z}{\partial \ln \sigma^2} = \frac{1}{2} \sigma \epsilon$ down to floating-point precision:
                """
            ),
            table_grad,
            mo.md(
                r"""
                ### Example 2: PyTorch Production VAE Latent Module

                Confirming end-to-end backpropagation through the stochastic reparameterization layer into upstream encoder parameters:
                """
            ),
            table_pt,
        ]
    )


if __name__ == "__main__":
    app.run()
