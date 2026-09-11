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
    from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

    return (
        average_precision_score,
        go,
        make_subplots,
        mo,
        np,
        pd,
        precision_recall_fscore_support,
        roc_auc_score,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 57 Autoencoder Latent Space](57_autoencoder_latent_space.py) | [Index](../index.html) | [59 VAE on MNIST →](59_vae_mnist.py)

        # 58. PCA for Anomaly Detection: Subspace Reconstruction Error and SPE / Q-Residual Analysis

        ### Executive Summary

        In industrial telemetry, network security, and financial fraud surveillance, multivariate observations routinely encompass dozens or hundreds of correlated sensor streams. Under nominal operating conditions, physical laws and structural couplings constrain observations to reside along a low-dimensional linear subspace $\mathcal{S}_k$.

        **Principal Component Analysis (PCA) for Anomaly Detection** leverages this geometric constraint by decomposing ambient observation space into two orthogonal subspaces: the **Principal Subspace** ($\mathcal{S}_k$), capturing normal shared variance, and the **Residual Subspace** ($\mathcal{S}_k^\perp$), capturing unmodeled noise and anomalies. By projecting data into $\mathcal{S}_k$ and evaluating the **Squared Prediction Error (SPE / $Q$-statistic)**, the algorithm detects structural faults that violate normal correlation patterns, while the **Hotelling's $T^2$ statistic** pinpoints extreme in-plane deviations.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations of Subspace Anomaly Detection

        ### 1. Orthogonal Subspace Decomposition

        Let $X \in \mathbb{R}^{N \times p}$ represent a standardized dataset of $N$ observations across $p$ features, with sample mean $\mu = \mathbf{0}$ and unit standard deviation. The sample covariance (correlation) matrix $R \in \mathbb{R}^{p \times p}$ is:

        $$R = \frac{1}{N - 1} X^\top X$$

        Applying spectral decomposition:

        $$R = V \Lambda V^\top = \begin{bmatrix} V_k & V_\perp \end{bmatrix} \begin{bmatrix} \Lambda_k & 0 \\ 0 & \Lambda_\perp \end{bmatrix} \begin{bmatrix} V_k^\top \\ V_\perp^\top \end{bmatrix}$$

        where:
        - $V_k \in \mathbb{R}^{p \times k}$ contains the top $k$ eigenvectors corresponding to dominant eigenvalues $\lambda_1 \ge \dots \ge \lambda_k$.
        - $V_\perp \in \mathbb{R}^{p \times (p - k)}$ contains the remaining $p - k$ eigenvectors capturing residual noise.
        - $\Lambda_k = \operatorname{diag}(\lambda_1, \dots, \lambda_k)$ represents the variance along each principal axis.

        This decomposes the ambient space $\mathbb{R}^p$ into mutually orthogonal subspaces:

        $$\mathbb{R}^p = \mathcal{S}_k \oplus \mathcal{S}_k^\perp$$

        ---

        ### 2. Projection and Reconstruction Operators

        For any new observation $x \in \mathbb{R}^p$:

        1. **Latent Code**: $z = x V_k \in \mathbb{R}^k$
        2. **Orthogonal Projection onto $\mathcal{S}_k$**: $P_k = V_k V_k^\top \in \mathbb{R}^{p \times p}$
        3. **Reconstructed Observation**:

        $$\hat{x} = z V_k^\top = x V_k V_k^\top = x P_k \in \mathbb{R}^p$$

        4. **Residual Error Vector**:

        $$e = x - \hat{x} = x (I - P_k) = x V_\perp V_\perp^\top \in \mathcal{S}_k^\perp$$

        If an observation conforms to the nominal correlation structure, its energy is almost entirely contained within $\mathcal{S}_k$, yielding $\|e\|_2 \approx 0$. If an observation violates the correlation structure (e.g., an abnormal sensor reading uncoupled from its correlated neighbors), it projects heavily into $\mathcal{S}_k^\perp$, triggering a large residual $\|e\|_2$.

        ---

        ### 3. Complementary Anomaly Statistics: SPE ($Q$) vs Hotelling's $T^2$

        Comprehensive fault detection employs two complementary metrics:

        #### A. Squared Prediction Error (SPE / $Q$-Statistic)
        The $Q$-statistic measures the squared Euclidean distance from the observation to the principal hyperplane:

        $$Q(x) = \| e \|_2^2 = \| x - \hat{x} \|_2^2 = x (I - P_k) x^\top = \sum_{j=1}^p (x_j - \hat{x}_j)^2$$

        - **Diagnostic Role**: Detects **out-of-model structural anomalies**—events where the underlying physical process changes or correlations break down.

        #### B. Hotelling's $T^2$ Statistic
        The $T^2$ statistic measures the normalized Mahalanobis distance **within** the principal subspace:

        $$T^2(x) = z \Lambda_k^{-1} z^\top = \sum_{j=1}^k \frac{z_j^2}{\lambda_j}$$

        - **Diagnostic Role**: Detects **in-model leverage anomalies**—events that obey the normal correlation structure but exhibit extreme magnitudes (e.g., high-throughput surges).

        ---

        ### 4. Statistical Control Limits and Fault Localization

        To automate alert generation without manual threshold tuning, Jackson & Mudholkar (1979) derived the theoretical upper control limit $Q_\alpha$ at significance level $\alpha$:

        $$\theta_i = \sum_{j=k+1}^p \lambda_j^i, \qquad h_0 = 1 - \frac{2 \theta_1 \theta_3}{3 \theta_2^2}$$

        $$Q_\alpha = \theta_1 \left[ 1 - \frac{\theta_2 h_0 (1 - h_0)}{\theta_1^2} + \frac{z_\alpha \sqrt{2 \theta_2 h_0^2}}{\theta_1} \right]^{\frac{1}{h_0}}$$

        Once an anomaly is flagged ($Q(x) > Q_\alpha$), the fault is localized to individual variables by decomposing the scalar $Q$ into feature contributions:

        $$\operatorname{Contribution}(j) = e_j^2 = (x_j - \hat{x}_j)^2$$

        The feature with the largest contribution indicates the root-cause faulty sensor.
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Simulate a 3D physical system (e.g. 3 correlated temperature sensors)
    np.random.seed(42)
    n_samples = 300

    # True correlation: sensors lie near a 2D plane in 3D space
    # x1 = t1, x2 = 0.8 * t1 + t2, x3 = 0.6 * t1 - 0.5 * t2
    t1 = np.random.normal(0, 2.5, n_samples)
    t2 = np.random.normal(0, 1.2, n_samples)

    x1 = t1 + np.random.normal(0, 0.15, n_samples)
    x2 = 0.85 * t1 + 0.9 * t2 + np.random.normal(0, 0.15, n_samples)
    x3 = 0.65 * t1 - 0.7 * t2 + np.random.normal(0, 0.15, n_samples)

    X_nom = np.column_stack([x1, x2, x3])

    # Inject distinct anomaly classes:
    # 1. Structural Anomaly (Sensor 3 fails and outputs +7.0, breaking correlation)
    anom_struct = np.array([[2.0, 1.8, 7.5], [-2.2, -1.9, -6.8], [0.5, 0.4, 5.2]])
    # 2. In-Subspace Leverage Outlier (Sensors spike along valid correlation plane)
    anom_leverage = np.array([[8.0, 7.2, 5.4], [-7.5, -6.8, -5.0]])

    X_all = np.vstack([X_nom, anom_struct, anom_leverage])
    true_labels = np.array([0] * n_samples + [1] * len(anom_struct) + [2] * len(anom_leverage))

    # Standardize data
    mu_vec = np.mean(X_all, axis=0)
    std_vec = np.std(X_all, axis=0)
    X_std = (X_all - mu_vec) / std_vec

    # Fit PCA on nominal data
    X_nom_std = (X_nom - mu_vec) / std_vec
    cov_mat = np.cov(X_nom_std, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov_mat)
    sort_idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sort_idx]
    eigvecs = eigvecs[:, sort_idx]

    # Retain k = 2 components (explaining >98% of variance)
    k_comps = 2
    V_k = eigvecs[:, :k_comps]
    Lambda_k = eigvals[:k_comps]

    # Compute Q and T^2 for all samples
    Z = np.dot(X_std, V_k)  # (N, 2)
    X_rec = np.dot(Z, V_k.T)  # (N, 3)
    residuals = X_std - X_rec

    Q_scores = np.sum(residuals**2, axis=1)
    T2_scores = np.sum((Z**2) / Lambda_k, axis=1)

    # Thresholds (97.5th percentile on nominal set)
    Q_thresh = float(np.percentile(Q_scores[:n_samples], 97.5))
    T2_thresh = float(np.percentile(T2_scores[:n_samples], 97.5))

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Reconstruction Error Residuals across Observations</b>",
            "<b>Four-Quadrant Anomaly Map: SPE (Q) vs Hotelling's T^2</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Bar chart of Q-scores
    nom_idx = np.where(true_labels == 0)[0]
    struct_idx = np.where(true_labels == 1)[0]
    lev_idx = np.where(true_labels == 2)[0]

    fig.add_trace(
        go.Scatter(
            x=nom_idx,
            y=Q_scores[nom_idx],
            mode="markers",
            marker=dict(color="#1D4ED8", size=5, opacity=0.6),
            name="Nominal Operating Data",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=struct_idx,
            y=Q_scores[struct_idx],
            mode="markers",
            marker=dict(color="#DC2626", size=10, symbol="diamond"),
            name="Structural Anomaly (Broken Correlation)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=lev_idx,
            y=Q_scores[lev_idx],
            mode="markers",
            marker=dict(color="#F59E0B", size=10, symbol="triangle-up"),
            name="In-Plane Leverage Outlier",
        ),
        row=1,
        col=1,
    )

    fig.add_hline(
        y=Q_thresh,
        line=dict(color="#DC2626", dash="dash", width=1.5),
        annotation_text=f"Q-Limit ({Q_thresh:.2f})",
        annotation_position="top left",
        row=1,
        col=1,
    )

    # Panel 2: Q vs T^2 Four-Quadrant Map
    fig.add_trace(
        go.Scatter(
            x=T2_scores[nom_idx],
            y=Q_scores[nom_idx],
            mode="markers",
            marker=dict(color="#1D4ED8", size=6, opacity=0.5),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=T2_scores[struct_idx],
            y=Q_scores[struct_idx],
            mode="markers",
            marker=dict(color="#DC2626", size=11, symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=T2_scores[lev_idx],
            y=Q_scores[lev_idx],
            mode="markers",
            marker=dict(color="#F59E0B", size=11, symbol="triangle-up"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.add_hline(y=Q_thresh, line=dict(color="#DC2626", dash="dash", width=1.5), row=1, col=2)
    fig.add_vline(
        x=T2_thresh,
        line=dict(color="#F59E0B", dash="dash", width=1.5),
        annotation_text=f"T^2 Limit ({T2_thresh:.1f})",
        annotation_position="top left",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Sample Index", row=1, col=1)
    fig.update_yaxes(title_text="Squared Prediction Error Q", range=[0, max(Q_scores) * 1.1], row=1, col=1)
    fig.update_xaxes(title_text="Hotelling's T^2 (In-Subspace Distance)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Squared Prediction Error Q (Out-of-Subspace)", type="log", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        Lambda_k,
        Q_scores,
        Q_thresh,
        T2_scores,
        T2_thresh,
        V_k,
        X_all,
        X_nom,
        X_nom_std,
        X_rec,
        X_std,
        Z,
        anom_leverage,
        anom_struct,
        cov_mat,
        eigvals,
        eigvecs,
        fig,
        k_comps,
        lev_idx,
        mu_vec,
        n_samples,
        nom_idx,
        residuals,
        sort_idx,
        std_vec,
        struct_idx,
        t1,
        t2,
        true_labels,
        viz,
        x1,
        x2,
        x3,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below captures subspace fault detection and anomaly categorization:

                1. **Left Panel (SPE / $Q$-Residual Trajectory)**: Nominal operating data (blue) remains tightly clustered near zero error. The structural anomalies (red diamonds) produce massive $Q$-statistic spikes far exceeding the statistical control limit ($Q_\alpha \approx 0.48$), whereas in-plane leverage outliers (amber triangles) incur zero excess reconstruction error because they conform to the principal plane.
                2. **Right Panel ($T^2$ vs $Q$ Four-Quadrant Map)**:
                   - **Lower-Left Quadrant**: Nominal operating data (Low $T^2$, Low $Q$).
                   - **Upper-Left Quadrant**: Structural anomalies (Low $T^2$, Extreme $Q$) caused by broken inter-variable physics.
                   - **Lower-Right Quadrant**: In-plane leverage outliers (Extreme $T^2$, Low $Q$) caused by extreme system throughput along valid operating trajectories.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    Lambda_k,
    Q_scores,
    Q_thresh,
    T2_scores,
    T2_thresh,
    V_k,
    X_std,
    average_precision_score,
    mo,
    np,
    pd,
    precision_recall_fscore_support,
    residuals,
    roc_auc_score,
    true_labels,
):
    # Example 1: Pure NumPy Vectorized PCA Anomaly Engine Implementation
    def pca_anomaly_detector_np(X_train, X_test, n_components=2):
        mu = np.mean(X_train, axis=0)
        std = np.std(X_train, axis=0)
        X_tr_norm = (X_train - mu) / std
        X_te_norm = (X_test - mu) / std

        # SVD
        _, S, Vt = np.linalg.svd(X_tr_norm, full_matrices=False)
        eigenvalues = (S**2) / (len(X_train) - 1)
        V_sub = Vt[:n_components].T
        Lambdas = eigenvalues[:n_components]

        # Project and reconstruct
        Z_te = np.dot(X_te_norm, V_sub)
        X_rec_te = np.dot(Z_te, V_sub.T)
        E_te = X_te_norm - X_rec_te

        Q_stats = np.sum(E_te**2, axis=1)
        T2_stats = np.sum((Z_te**2) / Lambdas, axis=1)

        return Q_stats, T2_stats, E_te

    # Example 2: Benchmark Metrics on Structural Anomalies
    binary_ground_truth = (true_labels == 1).astype(int)
    pred_anomaly = (Q_scores > Q_thresh).astype(int)

    prec, rec, f1, _ = precision_recall_fscore_support(
        binary_ground_truth, pred_anomaly, average="binary", zero_division=0
    )
    roc_auc = roc_auc_score(binary_ground_truth, Q_scores)
    pr_auc = average_precision_score(binary_ground_truth, Q_scores)

    df_benchmark = pd.DataFrame(
        [
            {
                "Anomaly_Detector": "PCA Squared Prediction Error (SPE / Q)",
                "Detection_Threshold": f"Q >= {Q_thresh:.4f} (97.5th percentile)",
                "Precision": f"{prec * 100:.2f}%",
                "Recall": f"{rec * 100:.2f}%",
                "F1-Score": f"{f1 * 100:.2f}%",
                "ROC_AUC": f"{roc_auc * 100:.2f}%",
                "PR_AUC": f"{pr_auc * 100:.2f}%",
            }
        ]
    )

    # Example 3: Root-Cause Fault Localization via Residual Decomposition
    # Examine the first structural anomaly (Sensor 3 was corrupted)
    target_anom_idx = np.where(true_labels == 1)[0][0]
    anom_res = residuals[target_anom_idx]
    feature_contributions = anom_res**2
    pct_contributions = (feature_contributions / np.sum(feature_contributions)) * 100.0

    df_localization = pd.DataFrame(
        [
            {
                "Sensor_Channel": f"Sensor {ch + 1}",
                "Standardized_Residual_e_j": f"{anom_res[ch]:.4f}",
                "Squared_Contribution_e_j^2": f"{feature_contributions[ch]:.4f}",
                "Percent_of_Total_Q": f"{pct_contributions[ch]:.1f}%",
                "Fault_Diagnosis": (
                    "Root-Cause Sensor Fault (Isolated)" if pct_contributions[ch] > 80.0 else "Coupled Nominal Sensor"
                ),
            }
            for ch in range(3)
        ]
    )

    # Example 4: Four-Quadrant Category Distribution
    is_high_q = Q_scores > Q_thresh
    is_high_t2 = T2_scores > T2_thresh

    quad_nominal = np.sum(~is_high_q & ~is_high_t2)
    quad_structural = np.sum(is_high_q & ~is_high_t2)
    quad_leverage = np.sum(~is_high_q & is_high_t2)
    quad_severe = np.sum(is_high_q & is_high_t2)

    df_quadrants = pd.DataFrame(
        [
            {
                "Operating_Quadrant": "Quadrant I: Low Q & Low T^2",
                "Diagnostic_Interpretation": "Nominal In-Control Operation",
                "Sample_Count": int(quad_nominal),
                "Percent_of_Dataset": f"{quad_nominal / len(Q_scores) * 100:.1f}%",
            },
            {
                "Operating_Quadrant": "Quadrant II: High Q & Low T^2",
                "Diagnostic_Interpretation": "Broken Correlation / Sensor Fault",
                "Sample_Count": int(quad_structural),
                "Percent_of_Dataset": f"{quad_structural / len(Q_scores) * 100:.1f}%",
            },
            {
                "Operating_Quadrant": "Quadrant III: Low Q & High T^2",
                "Diagnostic_Interpretation": "In-Model Leverage Outlier / Surge",
                "Sample_Count": int(quad_leverage),
                "Percent_of_Dataset": f"{quad_leverage / len(Q_scores) * 100:.1f}%",
            },
            {
                "Operating_Quadrant": "Quadrant IV: High Q & High T^2",
                "Diagnostic_Interpretation": "Catastrophic Joint Failure",
                "Sample_Count": int(quad_severe),
                "Percent_of_Dataset": f"{quad_severe / len(Q_scores) * 100:.1f}%",
            },
        ]
    )

    table_bench = mo.ui.table(df_benchmark)
    table_diag = mo.ui.table(df_localization)
    table_quad = mo.ui.table(df_quadrants)

    return (
        anom_res,
        binary_ground_truth,
        df_benchmark,
        df_localization,
        df_quadrants,
        f1,
        feature_contributions,
        is_high_q,
        is_high_t2,
        pca_anomaly_detector_np,
        pct_contributions,
        pr_auc,
        prec,
        pred_anomaly,
        quad_leverage,
        quad_nominal,
        quad_severe,
        quad_structural,
        rec,
        roc_auc,
        table_bench,
        table_diag,
        table_quad,
        target_anom_idx,
    )


@app.cell
def _(mo, table_bench, table_diag, table_quad):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Subspace Anomaly Detection Benchmark

                Evaluating classification performance on structural correlation anomalies:
                """
            ),
            table_bench,
            mo.md(
                r"""
                ### Example 2: Sensor Fault Localization via Residual Breakdown

                Decomposing the scalar $Q = \sum_{j=1}^p e_j^2$ to identify which individual channel generated the anomaly:
                """
            ),
            table_diag,
            mo.md(
                r"""
                ### Example 3: Operating Regime Distribution Across Four Quadrants

                Categorizing system observations into nominal, structural, leverage, and catastrophic failure regimes:
                """
            ),
            table_quad,
        ]
    )


if __name__ == "__main__":
    app.run()
