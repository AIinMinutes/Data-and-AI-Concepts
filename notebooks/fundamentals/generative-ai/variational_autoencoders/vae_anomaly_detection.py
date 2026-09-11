# /// script
# dependencies = ["pyod"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # packages added via marimo's package management: pyod==2.0.2 !pip3 install -q pyod==2.0.2
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    from pyod.models.vae import VAE
    from pyod.models.auto_encoder import AutoEncoder
    from pyod.utils.data import (
        generate_data, evaluate_print
    )
    from sklearn.metrics import (
        balanced_accuracy_score, f1_score
    )

    plt.style.use('dark_background')

    # Generate synthetic data
    contamination = 0.1
    n_train = 1000
    n_test = 100
    n_features = 2

    X_train, X_test, y_train, y_test = generate_data(
        n_train=n_train, n_test=n_test, 
        n_features=n_features,
        contamination=contamination, random_state=1
    )

    # Train the VAE model
    clf_name_vae = 'VAE'
    vae_clf = VAE(epoch_num=30, 
                  contamination=contamination, 
                  beta=1.0)
    vae_clf.fit(X_train)
    return (
        AutoEncoder,
        X_test,
        X_train,
        balanced_accuracy_score,
        contamination,
        f1_score,
        plt,
        vae_clf,
        y_test,
        y_train,
    )


@app.cell
def _(
    AutoEncoder,
    X_test,
    X_train,
    balanced_accuracy_score,
    contamination,
    f1_score,
    vae_clf,
):
    # Train the AE model
    clf_name_ae = 'AE'
    ae_clf = AutoEncoder(epoch_num=30, contamination=contamination)
    ae_clf.fit(X_train)
    y_test_pred_vae = vae_clf.predict(X_test)
    y_test_pred_ae = ae_clf.predict(X_test)
    # Predictions and scores for VAE

    def compute_metrics(y_true, y_pred):
    # Predictions and scores for AE
        balanced_acc = balanced_accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
    # Compute metrics function
        return (balanced_acc, f1)

    return ae_clf, compute_metrics, y_test_pred_ae, y_test_pred_vae


@app.cell
def _(
    X_test,
    X_train,
    ae_clf,
    compute_metrics,
    plt,
    vae_clf,
    y_test,
    y_test_pred_ae,
    y_test_pred_vae,
    y_train,
):
    def visualize_detailed_results(X, y_true, y_pred, model_name, dataset_name, ax):
        # Compute metrics
        balanced_acc, f1 = compute_metrics(y_true, y_pred)
    
        # Plot points with different categories
        ax.scatter(X[(y_true == 1) & (y_pred == 1), 0], X[(y_true == 1) & (y_pred == 1), 1], 
                   c='red', marker='x', label='True Positive (Anomaly)')
        ax.scatter(X[(y_true == 0) & (y_pred == 0), 0], X[(y_true == 0) & (y_pred == 0), 1], 
                   c='green', marker='+', label='True Negative (Non-Anomaly)')
        ax.scatter(X[(y_true == 0) & (y_pred == 1), 0], X[(y_true == 0) & (y_pred == 1), 1], 
                   c='orange', marker='*', label='False Positive (Non-Anomaly)')
        ax.scatter(X[(y_true == 1) & (y_pred == 0), 0], X[(y_true == 1) & (y_pred == 0), 1], 
                   c='blue', marker='^', label='False Negative (Anomaly)')
    
        # Title with metrics
        ax.set_title(f"{model_name} - {dataset_name}\nBalanced Acc: {balanced_acc:.2f}, F1: {f1:.2f}")
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.legend(loc='upper left')

    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(9, 9), dpi=300)

    # Visualize results for VAE on test data
    visualize_detailed_results(X_test, y_test, y_test_pred_vae, "VAE", "Test Data", axes[0, 0])

    # Visualize results for AE on test data
    visualize_detailed_results(X_test, y_test, y_test_pred_ae, "AE", "Test Data", axes[0, 1])

    # Visualize results for VAE on training data
    visualize_detailed_results(X_train, y_train, vae_clf.labels_, "VAE", "Training Data", axes[1, 0])

    # Visualize results for AE on training data
    visualize_detailed_results(X_train, y_train, ae_clf.labels_, "AE", "Training Data", axes[1, 1])

    plt.tight_layout()
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    $ \text{ELBO} = \| x - f(z) \|^2 + \frac{\beta}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_{q,i}^2 - 1 - \log \sigma_i^2 \right) $

    Where:
    - First term: **Reconstruction Loss** (MSE)
    - Second term: **$ \beta $ scaled KL Divergence** (regularizes the posterior)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    | **Aspect**                   | **Autoencoder (AE)**                                      | **Variational Autoencoder (VAE)**                          |
    |------------------------------|-----------------------------------------------------------|------------------------------------------------------------|
    | **Latent Space**              | Deterministic representation (fixed point for each input) | Probabilistic representation (distribution over latent variables) |
    | **Latent Space Structure**    | No specific regularization; can be scattered and unstructured | Regularized to match a predefined prior (typically Gaussian), resulting in a smooth, continuous space |
    | **Objective**                 | Minimize reconstruction error (e.g., MSE)                 | Minimize both reconstruction error and KL divergence to enforce latent space structure |
    | **Encoder Output**            | Direct mapping to a single point in latent space          | Outputs parameters of a distribution (mean and variance) for each latent variable |
    | **Generative Capability**     | Limited generative ability; may not generalize well for new data | Strong generative capability due to regularized latent space |
    | **Latent Variable Interpolation** | Less smooth interpolation between latent variables        | Smooth interpolation due to the continuous nature of the latent space |
    | **KL Divergence**             | Not used in the loss function                             | KL divergence term in the loss function regularizes the latent space |
    | **Reconstruction**            | Reconstructs the input deterministically                  | Reconstructs the input probabilistically, sampling from the learned latent distribution |
    | **Use Cases**                 | Mainly used for dimensionality reduction and reconstruction tasks | Used for generative modeling, data generation, and anomaly detection |
    | **Regularization**            | None                                                     | Explicit regularization to ensure the latent space follows a known distribution (e.g., Gaussian) |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Derivation of the Loss Function used in Variational AutoEncoder
    ### 1. Bayes' Theorem:
    $p(z|x) = \frac{p(x|z)p(z)}{p(x)}$

    Where:
    - $ p(z|x) $ = Posterior
    - $ p(x|z) $ = Likelihood
    - $ p(z) $ = Prior
    - $ p(x) = \int p(x|z)p(z) \, dz $ = Marginal Likelihood (Evidence)

    ### 2. Log of Marginal Likelihood:
    Taking the log:
    $\log p(x) = \log \int p(x, z) \, dz$

    ### 3. Variational Approximation ($q(z|x)$):
    Introduce a simpler distribution $q(z|x)$ to approximate the posterior $p(z|x)$.

    ### 4. Jensen's Inequality (Lower Bound):
    $\log p(x) = \log \int \frac{p(x, z)}{q(z|x)} q(z|x) \, dz \geq \mathbb{E}_{q(z|x)} [ \log \frac{p(x, z)}{q(z|x)} ]$

    The ELBO (Evidence Lower Bound) becomes:
    $\text{ELBO} = \mathbb{E}_{q(z|x)} [ \log p(x, z) - \log q(z|x) ]$

    ### 5. Expand the Joint Distribution:
    $p(x, z) = p(x|z)p(z)$

    So the ELBO becomes:
    $\text{ELBO} = \mathbb{E}_{q(z|x)} [ \log p(x|z) + \log p(z) - \log q(z|x) ]$

    ### 6. ELBO Formula:
    $\text{ELBO} = \mathbb{E}_{q(z|x)} [ \log p(x|z) ] - D_{KL}(q(z|x) \| p(z))$

    Where:
    - $\mathbb{E}_{q(z|x)} [ \log p(x|z) ]$ = Reconstruction Error
    - $ D_{KL}(q(z|x) \| p(z)) $ = KL Divergence

    ### 7. Simplification with Gaussian Assumptions:
    - **Prior**: $ p(z) = \mathcal{N}(z; 0, I) $
    - **Variational Posterior**: $ q(z|x) = \mathcal{N}(z; \mu_q(x), \Sigma_q(x)) $

    ### 8. Reconstruction Error (Gaussian Likelihood):
    $\mathbb{E}_{q(z|x)} [ \log p(x|z) ] \approx \| x - f(z) \|^2$

    Where $ f(z) $ is the decoder network (reconstruction).

    ### 9. KL Divergence:
    $ D_{KL}(q(z|x) \| p(z)) = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_{q,i}^2 - 1 - \log \sigma_i^2 \right) $

    Where $ \mu_q(x), \sigma_q(x) $ are parameters of the variational posterior, and $ \sigma_i^2 $ are diagonal variances of the approximate posterior.

    ### 10. Final Simplified ELBO Loss with $ \beta $:
    $ \text{ELBO} = \| x - f(z) \|^2 + \frac{\beta}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_{q,i}^2 - 1 - \log \sigma_i^2 \right) $

    Where:
    - First term: **Reconstruction Loss** (MSE)
    - Second term: **$ \beta $ scaled KL Divergence** (regularizes the posterior)
    """)
    return


if __name__ == "__main__":
    app.run()
