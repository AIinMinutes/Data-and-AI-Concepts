import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - **Reparameterization Trick**: A technique to enable backpropagation through stochastic processes by making them differentiable.
    - **Problem**: Direct sampling from distributions like $z \sim \mathcal{N}(\mu, \sigma^2)$ is non-differentiable, blocking gradient computation.
    - **Without reparameterization**: Sampling $z$ is non-differentiable:
     $$
    z = \text{Sample from } \mathcal{N}(\mu, \sigma^2)
    $$
     No gradients can be computed for $\mu$ or $\sigma$.
    - **Solution**: Represent $z$ as $z = \mu + \sigma \epsilon$, where $\epsilon \sim \mathcal{N}(0, 1)$.
    - **Example**: If $\mu = 2$, $\sigma = 3$, and $\epsilon = -1.5$:
     $$
    z = 2 + 3(-1.5) = -2.5
    $$
    - **Why It Works**: The transformation $z = \mu + \sigma \epsilon$ is differentiable, allowing gradient flow.
    - Gradients:
     $$
    \frac{\partial z}{\partial \mu} = 1, \quad \frac{\partial z}{\partial \sigma} = \epsilon
    $$
    - **Without Reparameterization**: The sampling operation is non-differentiable, making gradient computation impossible.
    - **Applications**:
        - **VAEs**: Enables training of both encoder and decoder networks.
        - **Generative Models**: Used in models with latent variables (e.g., GANs, RL).
        - **Benefit**: Makes non-differentiable operations differentiable, enabling gradient-based optimization.
    """)
    return


@app.cell
def _():
    import torch

    torch.manual_seed(47)

    # Parameters of the Gaussian distribution
    mu = torch.tensor(2., requires_grad=True)  # Mean (learnable)
    sigma = torch.tensor(5., requires_grad=True)  # std (learnable)

    # Reparametrization trick: z = mu + sigma * epsilon
    # where epsilon ~ N(0,1)
    epsilon = torch.randn_like(mu)  # Sample epsilon from N(0, 1)
    z = mu + sigma * epsilon  # Apply the reparametrization trick

    loss = z  # Will be involved in chain rule 
    loss.backward()  

    # Output the results
    print(f"Sample from N(0, 1): {epsilon.item()}")
    print(f"Sampled z: {z.item()}")
    print(f"Loss: {loss.item()}")
    print(f"Gradient wrt mu: {mu.grad.item()}")
    print(f"Gradient wrt sigma: {sigma.grad.item()}")
    return


if __name__ == "__main__":
    app.run()
