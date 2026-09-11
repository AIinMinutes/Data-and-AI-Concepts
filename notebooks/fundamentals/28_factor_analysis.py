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
    ### 1. **Orthogonal Factor Model**

    $$
    X = \mu + \Lambda F + \epsilon
    $$

    where:
    - $X \in \mathbb{R}^{p}$ is the vector of observed variables.
    - $\mu \in \mathbb{R}^{p}$ is the mean of the observed variables.
    - $\Lambda \in \mathbb{R}^{p \times k}$ is the loading matrix.
    - $F \in \mathbb{R}^{k}$ is the vector of latent factors.
    - $\epsilon \in \mathbb{R}^{p}$ is the vector of errors.

    ---

    ### 2. **Assumptions on Latent Factors**

    - **Uncorrelated Factors**:

    $$
    \text{Cov}(F) = I_k
    $$

    - **Independence from Errors**:

    $$
    \text{Cov}(F, \epsilon) = 0
    $$

    ---

    ### 3. **Assumption on Error**

    - **No Correlation Among Errors**:

    $$
    \text{Cov}(\epsilon) = \Psi, \quad \Psi \in \mathbb{R}^{p \times p}, \quad \text{diagonal}
    $$

    - **Independence from Latent Factors**:

    $$
    \text{Cov}(F, \epsilon) = 0
    $$

    ---

    ### 4. **Covariance Expressed in Terms of Latent Factors and Error**

    We start with:

    $$
    \Sigma = \text{Cov}(X) = E[(X - \mu)(X - \mu)']
    $$

    Substitute $X = \mu + \Lambda F + \epsilon$:

    $$
    \Sigma = E[(\Lambda F + \epsilon)(\Lambda F + \epsilon)']
    $$

    Expanding the product:

    $$
    \Sigma = E[\Lambda F F' \Lambda' + \Lambda F \epsilon' + \epsilon F' \Lambda' + \epsilon \epsilon']
    $$

    $$
    \Sigma = \Lambda \Lambda' + \Psi
    $$

    ---

    ### 5. **Why the Loading Matrix Need Not Be Unique**

    #### Proof:

    **Original Model**:

    $$
    X = \mu + \Lambda F + \epsilon
    $$

    **Covariance of $X$**:

    $$
    \text{Cov}(X) = \Sigma = \Lambda \Lambda^T + \Psi
    $$

    **Orthogonal Rotation**: Let $Q \in \mathbb{R}^{k \times k}$ be an orthogonal matrix, $Q^T Q = I_k$. Apply rotation:

    $$
    \Lambda' = \Lambda Q
    $$

    Substituting $\Lambda' = \Lambda Q$:

    $$
    \text{Cov}(X) = (\Lambda Q)(Q^T \Lambda^T) + \Psi
    $$

    Since $Q^T Q = I_k$:

    $$
    \text{Cov}(X) = \Lambda \Lambda^T + \Psi
    $$

    ---

    ### 6. **Rotation Schemes for Improving Interpretation**

    #### a) **Varimax Rotation**:
    Maximizes the variance of squared loadings for each factor:

    $$
    \sum_{j=1}^k \left( \sum_{i=1}^p \lambda_{ij}^2 \right)^2 \quad \text{maximize over } Q
    $$

    #### b) **Quartimax Rotation**:
    Maximizes the variance of squared loadings for each observed variable:

    $$
    \sum_{i=1}^p \left( \sum_{j=1}^k \lambda_{ij}^2 \right)^2 \quad \text{maximize over } Q
    $$

    #### c) **Equamax Rotation**:
    Balance between varimax and quartimax:

    $$
    \sum_{i=1}^p \left( \sum_{j=1}^k \lambda_{ij}^2 \right)^2 + \sum_{j=1}^k \left( \sum_{i=1}^p \lambda_{ij}^2 \right)^2 \quad \text{maximize over } Q
    $$
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.datasets import make_regression
    from statsmodels.multivariate.factor import Factor

    plt.style.use('dark_background')

    X, _ = make_regression(
        n_samples=100, n_features=5, 
        noise=0.1, effective_rank=2, 
        random_state=47
    )

    fa = Factor(X, n_factor=2) 
    fa_results = fa.fit()

    X_fa = np.dot(X, fa_results.loadings)

    # Plot the results to visualize
    plt.figure(figsize=(6, 6), dpi=300)
    plt.scatter(X_fa[:, 0], X_fa[:, 1], c='blue', edgecolors='k')
    plt.xlabel('Latent Factor 1')
    plt.ylabel('Latent Factor 2')
    plt.title('Factor Analysis - 2 Components')
    plt.savefig('factor_analysis.png', dpi=300)
    plt.show()
    return


if __name__ == "__main__":
    app.run()
