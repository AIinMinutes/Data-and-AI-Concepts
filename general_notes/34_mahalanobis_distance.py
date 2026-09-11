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
    from scipy.spatial.distance import mahalanobis
    from scipy.stats import chi2, multivariate_normal
    from sklearn.covariance import EmpiricalCovariance, MinCovDet
    from sklearn.metrics import balanced_accuracy_score

    return (
        EmpiricalCovariance,
        MinCovDet,
        balanced_accuracy_score,
        chi2,
        go,
        mahalanobis,
        make_subplots,
        mo,
        multivariate_normal,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 33 Huber Loss](33_huber_loss.py) | [Index](../index.html) | [35 Gini Impurity vs Entropy →](35_gini_impurity_vs_entropy.py)

        # Mahalanobis Distance, Covariance Metric Tensors, and Robust Outlier Detection

        ## [a] Why do you need to know these concepts?

        In multivariable data science, machine learning, and anomaly detection, Euclidean distance often fails when features have disparate scales or non-zero correlations.

        #### The Pitfalls of Isotropic Distance
        Standard Euclidean distance implicitly assumes that every coordinate axis possesses equal variance and that all pairs of features are mutually independent (an identity covariance matrix). In real-world data, coordinates are correlated and stretched along dominant principal directions. Two points located at identical Euclidean distances from the dataset centroid can have radically disparate likelihoods. A point shifted along the minor axis of variation represents a severe, statistically improbable deviation, whereas the same shift along the major axis represents routine random fluctuation.

        #### The Metric Tensor and Statistical Distance
        Mahalanobis distance resolves this flaw by measuring statistical distance scaled by the covariance structure. It transforms coordinates into a standardized space where distances correspond directly to standard deviations from the centroid along any arbitrary direction.

        #### The Masking and Swamping Effects
        Calculating Mahalanobis distance requires an estimate of the mean vector and covariance matrix. When a dataset contains severe outliers, conventional sample estimators suffer from catastrophic breakdown. Standard sample mean and covariance have a breakdown point of 0%. Even a small fraction of contaminated observations can artificially inflate the sample covariance, stretching the estimated ellipsoids toward the outliers. This phenomenon, known as the masking effect, causes genuine anomalies to receive deceptively small Mahalanobis distances.

        To overcome this vulnerability, robust statistics employs high-breakdown estimators such as the Minimum Covariance Determinant (FastMCD). FastMCD isolates the uncontaminated core of the observations, yielding an uncorrupted metric tensor that reliably flags multivariate anomalies.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Geometric Mechanics

        ### 1. Definition of Mahalanobis Distance

        Let $x \in \mathbb{R}^p$ be an observation vector, $\mu \in \mathbb{R}^p$ the population centroid, and $\Sigma \in \mathbb{R}^{p \times p}$ a symmetric positive-definite covariance matrix. The Mahalanobis distance $d_M(x, \mu; \Sigma)$ is defined as:

        $$d_M(x, \mu; \Sigma) = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)}$$

        When $\Sigma = \sigma^2 I_p$, the metric simplifies to standardized Euclidean distance:

        $$d_M(x, \mu; \sigma^2 I_p) = \frac{1}{\sigma} \sqrt{(x - \mu)^\top (x - \mu)} = \frac{\|x - \mu\|_2}{\sigma}$$

        ### 2. The Metric Tensor and Level Sets

        In Riemannian geometry and linear algebra, the inverse covariance matrix $M = \Sigma^{-1}$ acts as a Riemannian metric tensor on $\mathbb{R}^p$. The inner product induced by $\Sigma^{-1}$ is:

        $$\langle u, v \rangle_{\Sigma^{-1}} = u^\top \Sigma^{-1} v$$

        The level sets of constant squared Mahalanobis distance $d_M^2(x, \mu) = c^2$ define $(p-1)$-dimensional ellipsoids centered at $\mu$:

        $$\mathcal{E}_c = \left\{ x \in \mathbb{R}^p : (x - \mu)^\top \Sigma^{-1} (x - \mu) = c^2 \right\}$$

        Via the spectral decomposition $\Sigma = Q \Lambda Q^\top$, where $Q = [q_1, \dots, q_p]$ is the orthogonal matrix of eigenvectors and $\Lambda = \text{diag}(\lambda_1, \dots, \lambda_p)$ contains positive eigenvalues:

        $$\Sigma^{-1} = Q \Lambda^{-1} Q^\top = \sum_{j=1}^p \frac{1}{\lambda_j} q_j q_j^\top$$

        The principal semi-axes of the ellipsoid $\mathcal{E}_c$ align exactly with the eigenvectors $q_j$, with semi-axis lengths proportional to $c \sqrt{\lambda_j}$.

        ### 3. Equivalence to Whitening (Sphering)

        Mahalanobis distance can be computed as standard Euclidean distance following a linear whitening transformation. Define the whitening matrix $W = \Lambda^{-1/2} Q^\top = \Sigma^{-1/2}$. Let:

        $$z = W (x - \mu) = \Lambda^{-1/2} Q^\top (x - \mu)$$

        The transformed random vector $z$ has zero mean and an identity covariance matrix:

        $$\operatorname{Cov}(z) = W \Sigma W^\top = (\Sigma^{-1/2}) \Sigma (\Sigma^{-1/2}) = I_p$$

        Computing the Euclidean norm of $z$ yields:

        $$\|z\|_2 = \sqrt{z^\top z} = \sqrt{(x - \mu)^\top W^\top W (x - \mu)} = \sqrt{(x - \mu)^\top \Sigma^{-1} (x - \mu)} = d_M(x, \mu)$$

        Thus, Mahalanobis distance is the Euclidean distance measured after rotating into principal axes and scaling each axis to unit variance.

        ### 4. Chi-Square Distribution of Squared Distance

        If the random vector $X$ follows a multivariate normal distribution $X \sim \mathcal{N}_p(\mu, \Sigma)$, the whitened vector $Z = \Sigma^{-1/2}(X - \mu)$ follows standard multivariate normal distribution $Z \sim \mathcal{N}_p(0, I_p)$. The squared Mahalanobis distance represents the sum of $p$ independent standard squared normals:

        $$d_M^2(X, \mu) = \sum_{j=1}^p Z_j^2 \sim \chi^2_p$$

        This distribution provides an exact statistical hypothesis test for anomaly detection. For a significance level $\alpha \in (0, 1)$, the critical decision boundary is the $(1 - \alpha)$-quantile of the chi-square distribution with $p$ degrees of freedom:

        $$\text{Decision Rule: Outlier if } d_M^2(x, \mu) > \chi^2_p(1 - \alpha)$$

        ### 5. Minimum Covariance Determinant (FastMCD)

        Standard sample mean $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$ and sample covariance $S = \frac{1}{n-1} \sum_{i=1}^n (x_i - \bar{x})(x_i - \bar{x})^\top$ are vulnerable to leverage points and contamination.

        The Minimum Covariance Determinant (MCD) estimator, pioneered by Rousseeuw, achieves a 50% breakdown point. For a sample of size $n$, MCD seeks an optimal subset $H \subset \{1, \dots, n\}$ of size $h$ (typically $h \approx \frac{n + p + 1}{2}$) that minimizes the generalized variance:

        $$\hat{H} = \arg\min_{H \subset \{1, \dots, n\}, |H|=h} \det\left( \frac{1}{h-1} \sum_{i \in H} (x_i - \bar{x}_H)(x_i - \bar{x}_H)^\top \right)$$

        The FastMCD algorithm iteratively alternates between computing covariance on subset $H_k$ and updating $H_{k+1}$ to the $h$ observations with the smallest Mahalanobis distances under the current covariance. This C-step (concentration step) guarantees monotonic decrease of the determinant until convergence.
        """
    )
    return


@app.cell
def _(np):
    # Fix random seed for reproducible synthetic experiments
    np.random.seed(42)

    # Ground truth parameters for 2D Gaussian
    true_mean = np.array([2.0, 2.0])
    true_cov = np.array([[2.5, 1.8], [1.8, 2.0]])

    # Generate uncontaminated Gaussian core
    n_clean = 450
    clean_points = np.random.multivariate_normal(true_mean, true_cov, size=n_clean)

    # Injected anomalies: extreme orthogonal leverage points and cluster anomalies
    outlier_specs = [
        (-2.5, 5.0),
        (-3.0, 4.0),
        (-1.5, 4.5),
        (6.0, -1.0),
        (5.5, -2.0),
        (6.5, -0.5),
        (7.0, 7.5),
        (8.0, 7.0),
        (-3.5, -1.0),
        (-2.0, -2.5),
    ]
    anomaly_points = np.array(outlier_specs)
    n_anomalies = len(anomaly_points)

    # Combined dataset
    all_features = np.vstack([clean_points, anomaly_points])
    ground_truth_labels = np.zeros(len(all_features), dtype=int)
    ground_truth_labels[n_clean:] = 1

    return (
        all_features,
        anomaly_points,
        clean_points,
        ground_truth_labels,
        n_anomalies,
        n_clean,
        true_cov,
        true_mean,
    )


@app.cell
def _(
    EmpiricalCovariance,
    MinCovDet,
    all_features,
    chi2,
    clean_points,
    ground_truth_labels,
    np,
):
    # 1. Fit Standard Empirical Covariance (vulnerable to masking)
    emp_cov_model = EmpiricalCovariance().fit(all_features)
    emp_mean = emp_cov_model.location_
    emp_cov = emp_cov_model.covariance_
    emp_d2 = emp_cov_model.mahalanobis(all_features)

    # 2. Fit Robust Minimum Covariance Determinant (FastMCD)
    mcd_model = MinCovDet(random_state=42).fit(all_features)
    mcd_mean = mcd_model.location_
    mcd_cov = mcd_model.covariance_
    mcd_d2 = mcd_model.mahalanobis(all_features)

    # Chi-square critical threshold for p = 2 dimensions at alpha = 0.01 (99% confidence)
    dof = 2
    critical_d2 = chi2.ppf(0.99, df=dof)
    critical_d = np.sqrt(critical_d2)

    # Predictions based on chi-square threshold
    emp_pred_anomalies = (emp_d2 > critical_d2).astype(int)
    mcd_pred_anomalies = (mcd_d2 > critical_d2).astype(int)

    # Ground truth clean distances for theoretical comparison
    inv_mcd_cov = np.linalg.pinv(mcd_cov)
    clean_diff = clean_points - mcd_mean
    clean_d2 = np.sum(clean_diff @ inv_mcd_cov * clean_diff, axis=1)

    return (
        clean_d2,
        clean_diff,
        critical_d,
        critical_d2,
        dof,
        emp_cov,
        emp_cov_model,
        emp_d2,
        emp_mean,
        emp_pred_anomalies,
        inv_mcd_cov,
        mcd_cov,
        mcd_d2,
        mcd_mean,
        mcd_model,
        mcd_pred_anomalies,
    )


@app.cell
def _(
    all_features,
    chi2,
    clean_d2,
    critical_d,
    emp_cov,
    emp_mean,
    go,
    ground_truth_labels,
    make_subplots,
    mcd_cov,
    mcd_d2,
    mcd_mean,
    mcd_pred_anomalies,
    mo,
    np,
    true_cov,
    true_mean,
):
    # Helper to generate 2D ellipse coordinates for a given center, covariance, and distance level
    def get_ellipse_coords(center, cov_mat, distance_radius, n_pts=120):
        angles = np.linspace(0, 2 * np.pi, n_pts)
        circle_pts = np.vstack([np.cos(angles), np.sin(angles)])
        # Cholesky factor L such that L L^T = Cov
        eigenvalues, eigenvectors = np.linalg.eigh(cov_mat)
        # Scale by sqrt(eigenvalue)
        trans_matrix = eigenvectors @ np.diag(np.sqrt(np.maximum(eigenvalues, 1e-9)))
        ellipse_pts = center[:, None] + distance_radius * (trans_matrix @ circle_pts)
        return ellipse_pts[0], ellipse_pts[1]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Covariance Ellipsoids: Standard vs Robust FastMCD</b>",
            "<b>Empirical d_M^2 vs Theoretical Chi-Square (df=2) Distribution</b>",
        ],
        horizontal_spacing=0.12,
    )

    # Left Subplot: Scatter of clean vs anomaly points
    clean_idx = ground_truth_labels == 0
    anomaly_idx = ground_truth_labels == 1

    fig.add_trace(
        go.Scatter(
            x=all_features[clean_idx, 0],
            y=all_features[clean_idx, 1],
            mode="markers",
            marker=dict(color="#2563EB", size=6, opacity=0.6),
            name="Normal Points (Clean)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=all_features[anomaly_idx, 0],
            y=all_features[anomaly_idx, 1],
            mode="markers",
            marker=dict(color="#DC2626", size=10, symbol="x", line=dict(width=2)),
            name="Ground Truth Anomalies",
        ),
        row=1,
        col=1,
    )

    # Add 99% Chi-Square boundary ellipse for Ground Truth
    gt_ex, gt_ey = get_ellipse_coords(true_mean, true_cov, critical_d)
    fig.add_trace(
        go.Scatter(
            x=gt_ex,
            y=gt_ey,
            mode="lines",
            line=dict(color="#10B981", width=2, dash="dash"),
            name="True 99% Ellipse",
        ),
        row=1,
        col=1,
    )

    # Add 99% Chi-Square boundary ellipse for Standard Empirical Covariance (Masked)
    emp_ex, emp_ey = get_ellipse_coords(emp_mean, emp_cov, critical_d)
    fig.add_trace(
        go.Scatter(
            x=emp_ex,
            y=emp_ey,
            mode="lines",
            line=dict(color="#F59E0B", width=2.5, dash="dot"),
            name="Empirical Covariance 99% (Masked)",
        ),
        row=1,
        col=1,
    )

    # Add 99% Chi-Square boundary ellipse for FastMCD (Robust)
    mcd_ex, mcd_ey = get_ellipse_coords(mcd_mean, mcd_cov, critical_d)
    fig.add_trace(
        go.Scatter(
            x=mcd_ex,
            y=mcd_ey,
            mode="lines",
            line=dict(color="#8B5CF6", width=2.5),
            name="FastMCD Robust 99% Boundary",
        ),
        row=1,
        col=1,
    )

    # Right Subplot: Empirical Chi-Square histogram vs Theoretical Chi-Square Density
    x_theory = np.linspace(0.01, 25.0, 300)
    pdf_theory = chi2.pdf(x_theory, df=2)

    fig.add_trace(
        go.Histogram(
            x=clean_d2,
            histnorm="probability density",
            nbinsx=35,
            marker_color="#60A5FA",
            opacity=0.6,
            name="Clean Samples d_M^2",
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=x_theory,
            y=pdf_theory,
            mode="lines",
            line=dict(color="#1E3A8A", width=2.5),
            name="Theoretical Chi-Square(df=2) PDF",
        ),
        row=1,
        col=2,
    )

    # Mark the 99% critical threshold line
    fig.add_vline(
        x=critical_d**2,
        line=dict(color="#DC2626", width=2, dash="dash"),
        annotation_text=f"99% Cutoff ({critical_d**2:.2f})",
        annotation_position="top right",
        row=1,
        col=2,
    )

    # Plot anomaly distances on right subplot as scatter ticks
    fig.add_trace(
        go.Scatter(
            x=mcd_d2[anomaly_idx],
            y=[0.02] * len(all_features[anomaly_idx]),
            mode="markers",
            marker=dict(color="#DC2626", symbol="line-ns", size=16, line=dict(width=2)),
            name="Anomalies d_M^2 Scores",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Feature 1 (X1)", row=1, col=1)
    fig.update_yaxes(title_text="Feature 2 (X2)", row=1, col=1)
    fig.update_xaxes(title_text="Squared Mahalanobis Distance (d_M^2)", range=[0, 30], row=1, col=2)
    fig.update_yaxes(title_text="Density", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=50, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        anomaly_idx,
        clean_idx,
        fig,
        get_ellipse_coords,
        pdf_theory,
        viz,
        x_theory,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual panel below illustrates the geometry of Mahalanobis level sets and the statistical power of robust covariance estimation:

                1. **Left (Geometric Ellipsoids)**: The empirical covariance ellipse (yellow dotted) is pulled outward and rotated by the leverage outliers, masking multiple anomalies. In contrast, the FastMCD robust ellipse (purple solid) accurately traces the true data-generating core (green dashed), correctly isolating the anomalies outside the 99% boundary.
                2. **Right (Theoretical Distribution Calibration)**: The empirical distribution of squared Mahalanobis distances for clean points matches the theoretical $\chi^2_2$ density curve ($e^{-x/2}/2$). The injected anomalies appear far past the 99% cutoff threshold ($\chi^2_{2, 0.99} \approx 9.21$).
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    all_features,
    balanced_accuracy_score,
    clean_points,
    critical_d2,
    emp_d2,
    emp_pred_anomalies,
    ground_truth_labels,
    mahalanobis,
    mcd_cov,
    mcd_d2,
    mcd_mean,
    mcd_pred_anomalies,
    mo,
    np,
    pd,
):
    # Example 1: Pure NumPy Vectorized Mahalanobis Distance vs Scipy Implementation
    def vectorized_mahalanobis(x_mat, mean_vec, cov_mat):
        centered = x_mat - mean_vec
        cov_inv = np.linalg.pinv(cov_mat)
        sq_dist = np.sum(centered @ cov_inv * centered, axis=1)
        return np.sqrt(np.maximum(sq_dist, 0.0))

    test_subset = all_features[:5]
    numpy_distances = vectorized_mahalanobis(test_subset, mcd_mean, mcd_cov)
    cov_inv_test = np.linalg.pinv(mcd_cov)
    scipy_distances = np.array([mahalanobis(pt, mcd_mean, cov_inv_test) for pt in test_subset])

    df_verification = pd.DataFrame(
        {
            "Point_Index": np.arange(5),
            "Feature_1": test_subset[:, 0].round(4),
            "Feature_2": test_subset[:, 1].round(4),
            "NumPy_Vectorized_dM": numpy_distances.round(5),
            "SciPy_Spatial_dM": scipy_distances.round(5),
            "Max_Absolute_Diff": np.abs(numpy_distances - scipy_distances).round(8),
        }
    )

    # Example 2: Outlier Detection Benchmark: Standard Empirical vs FastMCD Robust Estimator
    emp_bal_acc = balanced_accuracy_score(ground_truth_labels, emp_pred_anomalies)
    mcd_bal_acc = balanced_accuracy_score(ground_truth_labels, mcd_pred_anomalies)

    # Detection statistics
    n_true_anomalies = np.sum(ground_truth_labels == 1)
    emp_detected = np.sum((emp_pred_anomalies == 1) & (ground_truth_labels == 1))
    mcd_detected = np.sum((mcd_pred_anomalies == 1) & (ground_truth_labels == 1))

    df_benchmark = pd.DataFrame(
        [
            {
                "Estimator": "Empirical Covariance (Masked)",
                "Breakdown_Point": "0.0%",
                "Chi2_Cutoff": f"{critical_d2:.2f}",
                "True_Anomalies_Detected": f"{emp_detected} / {n_true_anomalies}",
                "Balanced_Accuracy": f"{emp_bal_acc * 100:.2f}%",
                "Robust_to_Masking": "No (Covariance inflated by outliers)",
            },
            {
                "Estimator": "FastMCD (Minimum Covariance Determinant)",
                "Breakdown_Point": "~50.0%",
                "Chi2_Cutoff": f"{critical_d2:.2f}",
                "True_Anomalies_Detected": f"{mcd_detected} / {n_true_anomalies}",
                "Balanced_Accuracy": f"{mcd_bal_acc * 100:.2f}%",
                "Robust_to_Masking": "Yes (Isolates uncontaminated sub-sample)",
            },
        ]
    )

    # Example 3: Comparison of Euclidean Distance vs Mahalanobis Distance for Key Boundary Samples
    euclidean_dist = np.linalg.norm(all_features - mcd_mean, axis=1)
    selected_indices = [0, 15, 100, -1, -5, -8]
    selected_samples = all_features[selected_indices]

    df_comparison = pd.DataFrame(
        {
            "Sample_Type": [
                "Core Point A",
                "Core Point B",
                "Core Point C",
                "Orthogonal Outlier 1",
                "Orthogonal Outlier 2",
                "Cluster Outlier 3",
            ],
            "Feature_1": selected_samples[:, 0].round(3),
            "Feature_2": selected_samples[:, 1].round(3),
            "Euclidean_Dist": euclidean_dist[selected_indices].round(3),
            "Empirical_dM": np.sqrt(emp_d2[selected_indices]).round(3),
            "FastMCD_dM": np.sqrt(mcd_d2[selected_indices]).round(3),
            "True_Label": ["Normal", "Normal", "Normal", "Anomaly", "Anomaly", "Anomaly"],
            "FastMCD_Decision": [
                "Anomaly" if mcd_d2[idx] > critical_d2 else "Normal" for idx in selected_indices
            ],
        }
    )

    table_verif = mo.ui.table(df_verification)
    table_bench = mo.ui.table(df_benchmark)
    table_comp = mo.ui.table(df_comparison)

    return (
        cov_inv_test,
        df_benchmark,
        df_comparison,
        df_verification,
        emp_bal_acc,
        emp_detected,
        euclidean_dist,
        mcd_bal_acc,
        mcd_detected,
        n_true_anomalies,
        numpy_distances,
        scipy_distances,
        selected_indices,
        selected_samples,
        table_bench,
        table_comp,
        table_verif,
        test_subset,
        vectorized_mahalanobis,
    )


@app.cell
def _(mo, table_bench, table_comp, table_verif):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Vectorized NumPy Implementation vs SciPy Spatial

                Verification that the matrix form $\sqrt{(X - \mu)^\top \Sigma^{-1} (X - \mu)}$ matches standard implementations up to machine precision:
                """
            ),
            table_verif,
            mo.md(
                r"""
                ### Example 2: Outlier Detection Benchmark (Empirical vs FastMCD)

                Quantitative comparison demonstrating how empirical covariance suffers from masking while FastMCD achieves superior detection power:
                """
            ),
            table_bench,
            mo.md(
                r"""
                ### Example 3: Euclidean vs Mahalanobis Distance Across Challenging Scenarios

                Notice how points with similar Euclidean distance have drastically different Mahalanobis distances depending on their alignment with the covariance metric tensor:
                """
            ),
            table_comp,
        ]
    )


if __name__ == "__main__":
    app.run()
