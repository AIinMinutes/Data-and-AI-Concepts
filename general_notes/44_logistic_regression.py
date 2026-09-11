import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    import statsmodels.api as sm
    from plotly.subplots import make_subplots
    from sklearn.datasets import load_breast_cancer
    from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
    from sklearn.model_selection import train_test_split

    return (
        accuracy_score,
        confusion_matrix,
        go,
        load_breast_cancer,
        make_subplots,
        mo,
        np,
        pd,
        roc_auc_score,
        sm,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 43 Matrix Energy and Definiteness](43_energy.py) | [Index](../index.html) | [45 Shapley Values →](45_shapley.py)

        # Logistic Regression: Log-Odds, Sigmoid Link, Odds Ratios, and Maximum Likelihood Estimation

        ## [a] Why do you need to know these concepts?

        Logistic regression is the foundational parametric model for binary classification, probabilistic risk scoring, clinical epidemiological trials, credit underwriting, and causal propensity score matching.

        #### Why Ordinary Least Squares (OLS) Fails for Binary Outcomes
        When modeling a binary response $y_i \in \{0, 1\}$, fitting a standard linear regression model $y = X \beta + \epsilon$ introduces structural violations of classical statistical theory:
        1. **Nonsensical Unbounded Probabilities**: Linear functions produce predictions outside the valid unit interval $[0, 1]$. A model predicting a probability of $-0.35$ or $1.42$ violates the Kolmogorov axioms of probability.
        2. **Severe Heteroscedasticity**: The binary outcome variance depends directly on the mean:

        $$\operatorname{Var}(y_i | x_i) = p_i(1 - p_i)$$

        Because error variances vary across observations, OLS standard errors are inconsistent and hypothesis tests ($t$-tests and $F$-tests) become invalid.
        3. **Discrete Non-Gaussian Residuals**: Residuals take only two discrete values ($1 - \hat{y}_i$ or $-\hat{y}_i$), violating the normality assumption required for finite-sample inference.

        #### The Logit Link Transformation
        Logistic regression resolves these flaws by modeling the **log-odds** (logit) of the positive class as an unconstrained linear function of predictors:

        $$\eta = \ln\left( \frac{p}{1 - p} \right) = X \beta$$

        Inverting this transformation maps the unconstrained real line $(-\infty, \infty)$ back onto $(0, 1)$ via the **Sigmoid activation function**.

        #### Intuitive Interpretation via Odds Ratios
        Unlike neural networks or black-box tree ensembles, logistic regression provides transparent, interpretable coefficients. Exponentiating a feature's coefficient yields its **Odds Ratio** ($\text{OR} = e^{\beta_j}$), indicating the multiplicative factor by which the odds of the outcome change for each one-unit increase in that predictor.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Likelihood Mechanics

        ### 1. Odds, Log-Odds, and the Sigmoid Link

        Let $p = P(Y = 1 | X)$ denote the conditional probability of the event of interest.

        #### The Odds
        The odds represent the ratio of the probability of success to the probability of failure:

        $$\operatorname{Odds} = \frac{p}{1 - p} \in (0, \infty)$$

        - When $p = 0.5$, $\operatorname{Odds} = 1$ (even chance).
        - When $p = 0.8$, $\operatorname{Odds} = 4$ (success is 4 times as likely as failure).
        - When $p = 0.2$, $\operatorname{Odds} = 0.25$ (failure is 4 times as likely as success).

        #### The Logit (Log-Odds) Link
        The natural logarithm of the odds maps the positive semi-infinite interval $(0, \infty)$ onto $(-\infty, \infty)$:

        $$\operatorname{logit}(p) = \ln\left( \frac{p}{1 - p} \right) = \beta_0 + \beta_1 x_1 + \dots + \beta_k x_k = x^\top \beta$$

        #### The Sigmoid Function
        Solving for $p$ yields the standard logistic sigmoid function $\sigma(\eta)$:

        $$p = \sigma(x^\top \beta) = \frac{1}{1 + e^{-x^\top \beta}} = \frac{e^{x^\top \beta}}{1 + e^{x^\top \beta}}$$

        The first derivative of the sigmoid has the convenient algebraic property:

        $$\frac{d\sigma}{d\eta} = \sigma(\eta)(1 - \sigma(\eta)) = p(1 - p)$$

        ### 2. The Odds Ratio (OR)

        Consider the effect of increasing a single predictor $x_j$ by one unit ($x_j \to x_j + 1$) while holding all other features constant:

        $$\operatorname{Odds}(x_j + 1) = \exp\left( \beta_0 + \dots + \beta_j(x_j + 1) + \dots \right) = \exp(\beta_j) \cdot \operatorname{Odds}(x_j)$$

        The Odds Ratio for predictor $j$ is:

        $$\text{OR}_j = \frac{\operatorname{Odds}(x_j + 1)}{\operatorname{Odds}(x_j)} = e^{\beta_j}$$

        - If $\beta_j > 0 \implies \text{OR}_j > 1$: A one-unit increase in $x_j$ increases the odds by $(e^{\beta_j} - 1) \times 100\%$.
        - If $\beta_j < 0 \implies \text{OR}_j < 1$: A one-unit increase in $x_j$ decreases the odds by $(1 - e^{\beta_j}) \times 100\%$.
        - If $\beta_j = 0 \implies \text{OR}_j = 1$: Predictor $x_j$ has no association with the outcome.

        The $(1 - \alpha)$ confidence interval for the Odds Ratio is computed by exponentiating the coefficient bounds:

        $$\text{CI}_{1-\alpha}(\text{OR}_j) = \left[ \exp\left(\hat{\beta}_j - z_{1-\alpha/2} \cdot \operatorname{SE}(\hat{\beta}_j)\right), \ \exp\left(\hat{\beta}_j + z_{1-\alpha/2} \cdot \operatorname{SE}(\hat{\beta}_j)\right) \right]$$

        ### 3. Maximum Likelihood Estimation (MLE) and IRLS

        Assuming independent Bernoulli trials $y_i \sim \operatorname{Bernoulli}(p_i)$, the joint likelihood function is:

        $$L(\beta) = \prod_{i=1}^N p_i^{y_i} (1 - p_i)^{1 - y_i}$$

        The log-likelihood is:

        $$\ln L(\beta) = \sum_{i=1}^N \left[ y_i \ln p_i + (1 - y_i) \ln (1 - p_i) \right] = \sum_{i=1}^N \left[ y_i (x_i^\top \beta) - \ln\left(1 + e^{x_i^\top \beta}\right) \right]$$

        The gradient (score vector) is:

        $$\nabla_\beta \ln L = \sum_{i=1}^N (y_i - p_i) x_i = X^\top (y - p)$$

        The negative Hessian (Fisher Information matrix) is:

        $$-H = -\nabla_\beta^2 \ln L = \sum_{i=1}^N p_i (1 - p_i) x_i x_i^\top = X^\top W X$$

        where $W = \operatorname{diag}(p_1(1 - p_1), \dots, p_N(1 - p_N))$. Since $0 < p_i < 1$, $W$ is strictly positive definite, proving that $\ln L(\beta)$ is **strictly concave** with a unique global maximum.

        Parameters are estimated iteratively via Newton-Raphson / Iteratively Reweighted Least Squares (IRLS):

        $$\beta^{(t+1)} = \beta^{(t)} + (X^\top W_t X)^{-1} X^\top (y - p_t)$$
        """
    )
    return


@app.cell
def _(
    load_breast_cancer,
    np,
    pd,
    sm,
    train_test_split,
):
    # Load standardized clinical breast cancer dataset
    cancer = load_breast_cancer()
    feature_subset = ["mean radius", "mean texture", "mean smoothness", "mean compactness"]
    feat_indices = [list(cancer.feature_names).index(f) for f in feature_subset]

    x_raw = cancer.data[:, feat_indices]
    # In cancer dataset: 0 = Malignant, 1 = Benign. Keep 1 as Benign.
    y_raw = cancer.target

    # Standardize predictors for numerical stability and coefficient interpretability
    x_mean = np.mean(x_raw, axis=0)
    x_std = np.std(x_raw, axis=0)
    x_std_data = (x_raw - x_mean) / x_std

    clean_feature_names = [
        "Radius_Standardized",
        "Texture_Standardized",
        "Smoothness_Standardized",
        "Compactness_Standardized",
    ]

    df_features = pd.DataFrame(x_std_data, columns=clean_feature_names)

    # Train-test split
    x_train_raw, x_test_raw, y_train, y_test = train_test_split(
        df_features.values, y_raw, test_size=0.3, random_state=42, stratify=y_raw
    )

    # Add intercept column
    x_train_const = sm.add_constant(x_train_raw)
    x_test_const = sm.add_constant(x_test_raw)

    # Fit statsmodels Logit
    logit_model = sm.Logit(y_train, x_train_const)
    logit_res = logit_model.fit(disp=False)

    # Predictions
    test_pred_prob = logit_res.predict(x_test_const)
    test_pred_class = (test_pred_prob >= 0.5).astype(int)

    # Univariate slice for Sigmoid curve visualization: Radius feature only
    x_radius = x_train_raw[:, 0]
    radius_grid = np.linspace(x_radius.min() - 0.5, x_radius.max() + 0.5, 200)

    # Univariate logistic fit
    uni_model = sm.Logit(y_train, sm.add_constant(x_radius)).fit(disp=False)
    uni_probs = uni_model.predict(sm.add_constant(radius_grid))

    # OLS linear regression for comparison
    ols_model = sm.OLS(y_train, sm.add_constant(x_radius)).fit()
    ols_preds = ols_model.predict(sm.add_constant(radius_grid))

    return (
        cancer,
        clean_feature_names,
        df_features,
        feat_indices,
        feature_subset,
        logit_model,
        logit_res,
        ols_model,
        ols_preds,
        radius_grid,
        test_pred_class,
        test_pred_prob,
        uni_model,
        uni_probs,
        x_mean,
        x_radius,
        x_raw,
        x_std,
        x_std_data,
        x_test_const,
        x_test_raw,
        x_train_const,
        x_train_raw,
        y_raw,
        y_test,
        y_train,
    )


@app.cell
def _(
    clean_feature_names,
    go,
    logit_res,
    make_subplots,
    mo,
    np,
    ols_preds,
    radius_grid,
    uni_probs,
    x_radius,
    y_train,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Sigmoid Probability Curve vs Linear Regression (OLS)</b>",
            "<b>Odds Ratios with 95% Confidence Intervals (Forest Plot)</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Sigmoid vs Linear
    # Jittered scatter of true training points
    y_jitter = y_train + np.random.normal(0, 0.025, len(y_train))
    fig.add_trace(
        go.Scatter(
            x=x_radius,
            y=y_jitter,
            mode="markers",
            marker=dict(color="#64748B", size=5, opacity=0.45),
            name="Observed Data (Jittered)",
        ),
        row=1,
        col=1,
    )

    # Sigmoid Curve
    fig.add_trace(
        go.Scatter(
            x=radius_grid,
            y=uni_probs,
            mode="lines",
            line=dict(color="#2563EB", width=2.5),
            name="Logistic Sigmoid Probability",
        ),
        row=1,
        col=1,
    )

    # Linear Regression Line (showing unconstrained overshoot)
    fig.add_trace(
        go.Scatter(
            x=radius_grid,
            y=ols_preds,
            mode="lines",
            line=dict(color="#DC2626", width=2, dash="dash"),
            name="OLS Linear Regression",
        ),
        row=1,
        col=1,
    )

    # Decision threshold line at p = 0.5
    fig.add_hline(
        y=0.5,
        line=dict(color="#10B981", width=1.5, dash="dot"),
        annotation_text="Decision Threshold (p = 0.5)",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Panel 2: Forest Plot of Odds Ratios
    coef_vals = logit_res.params[1:]
    se_vals = logit_res.bse[1:]
    odds_ratios = np.exp(coef_vals)
    or_lower = np.exp(coef_vals - 1.96 * se_vals)
    or_upper = np.exp(coef_vals + 1.96 * se_vals)

    fig.add_trace(
        go.Scatter(
            x=odds_ratios,
            y=clean_feature_names,
            mode="markers",
            marker=dict(size=10, color="#8B5CF6"),
            error_x=dict(
                type="data",
                symmetric=False,
                array=or_upper - odds_ratios,
                arrayminus=odds_ratios - or_lower,
                visible=True,
                color="#8B5CF6",
                thickness=2,
                width=6,
            ),
            name="Odds Ratio (95% CI)",
        ),
        row=1,
        col=2,
    )

    # Reference line at OR = 1.0 (no effect)
    fig.add_vline(
        x=1.0,
        line=dict(color="#DC2626", width=1.5, dash="dash"),
        annotation_text="OR = 1.0 (No Effect)",
        annotation_position="top left",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Standardized Mean Radius", row=1, col=1)
    fig.update_yaxes(title_text="Predicted Probability P(Y=1)", range=[-0.15, 1.15], row=1, col=1)
    fig.update_xaxes(title_text="Odds Ratio (Log Scale)", type="log", row=1, col=2)
    fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        coef_vals,
        fig,
        odds_ratios,
        or_lower,
        or_upper,
        se_vals,
        viz,
        y_jitter,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below contrasts probability modeling and odds ratio interpretation:

                1. **Left Panel (Sigmoid vs OLS)**: The Logistic Sigmoid (blue solid) smoothly transitions between asymptotes at $0$ and $1$. In contrast, OLS linear regression (red dashed) violates probability bounds, predicting negative probabilities for large radius values ($P < 0$) and exceeding $1.0$ for small radius values.
                2. **Right Panel (Odds Ratio Forest Plot)**: Forest plot showing estimated Odds Ratios on a logarithmic scale with 95% confidence intervals. Features with intervals strictly below $1.0$ (such as standardized Radius and Texture) significantly decrease the odds of benign tumor status.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    accuracy_score,
    clean_feature_names,
    confusion_matrix,
    logit_res,
    mo,
    np,
    pd,
    roc_auc_score,
    test_pred_class,
    test_pred_prob,
    x_train_const,
    y_test,
    y_train,
):
    # Example 1: Pure NumPy Newton-Raphson / IRLS Solver from scratch
    def newton_raphson_logistic(x_mat, y_vec, max_iter=25, tol=1e-8):
        n, p_dim = x_mat.shape
        beta = np.zeros(p_dim)

        for _ in range(max_iter):
            # Predicted probabilities: sigma(X beta)
            eta = x_mat @ beta
            p = 1.0 / (1.0 + np.exp(-np.clip(eta, -30, 30)))

            # Gradient: X^T (y - p)
            grad = x_mat.T @ (y_vec - p)

            # Weight diagonal matrix W = diag(p * (1 - p))
            w_diag = p * (1.0 - p)
            # Hessian: -X^T W X
            hess = -x_mat.T @ (w_diag[:, None] * x_mat)

            # Newton step: beta_new = beta - H^(-1) grad
            delta = np.linalg.solve(-hess, grad)
            beta = beta + delta

            if np.linalg.norm(delta) < tol:
                break

        # Compute covariance matrix inv(X^T W X)
        cov_beta = np.linalg.inv(-hess)
        se = np.sqrt(np.diag(cov_beta))
        return beta, se

    beta_scratch, se_scratch = newton_raphson_logistic(x_train_const, y_train)

    df_verification = pd.DataFrame(
        {
            "Parameter": ["Intercept"] + clean_feature_names,
            "From_Scratch_Beta": np.round(beta_scratch, 5),
            "Statsmodels_Beta": np.round(logit_res.params, 5),
            "From_Scratch_SE": np.round(se_scratch, 5),
            "Statsmodels_SE": np.round(logit_res.bse, 5),
            "Max_Diff": np.abs(beta_scratch - logit_res.params).round(8),
        }
    )

    # Example 2: Comprehensive Odds Ratio and Hypothesis Testing Table
    or_values = np.exp(logit_res.params)
    ci_mat = np.asarray(logit_res.conf_int())
    or_ci_low = np.exp(ci_mat[:, 0])
    or_ci_high = np.exp(ci_mat[:, 1])

    df_or_summary = pd.DataFrame(
        {
            "Variable": ["Intercept"] + clean_feature_names,
            "Coefficient_Beta": logit_res.params.round(4),
            "Std_Error": logit_res.bse.round(4),
            "Wald_z_stat": logit_res.tvalues.round(3),
            "p_value": [
                "< 0.001" if p < 0.001 else f"{p:.4f}" for p in logit_res.pvalues
            ],
            "Odds_Ratio": np.round(or_values, 3),
            "95%_CI_OR": [f"[{l:.3f}, {h:.3f}]" for l, h in zip(or_ci_low, or_ci_high)],
        }
    )

    # Example 3: Model Evaluation Diagnostics on Holdout Test Partition
    test_acc = accuracy_score(y_test, test_pred_class)
    test_auc = roc_auc_score(y_test, test_pred_prob)
    cm = confusion_matrix(y_test, test_pred_class)

    df_diagnostics = pd.DataFrame(
        [
            {
                "Evaluation_Metric": "Classification Accuracy",
                "Test_Set_Value": f"{test_acc * 100:.2f}%",
                "Threshold": "p >= 0.5",
            },
            {
                "Evaluation_Metric": "Area Under ROC Curve (AUC)",
                "Test_Set_Value": f"{test_auc:.4f}",
                "Threshold": "Rank-order across all cutoffs",
            },
            {
                "Evaluation_Metric": "True Negatives (Malignant correctly flagged)",
                "Test_Set_Value": str(cm[0, 0]),
                "Threshold": "Class 0",
            },
            {
                "Evaluation_Metric": "True Positives (Benign correctly flagged)",
                "Test_Set_Value": str(cm[1, 1]),
                "Threshold": "Class 1",
            },
        ]
    )

    table_verif = mo.ui.table(df_verification)
    table_or = mo.ui.table(df_or_summary)
    table_diag = mo.ui.table(df_diagnostics)

    return (
        beta_scratch,
        cm,
        df_diagnostics,
        df_or_summary,
        df_verification,
        newton_raphson_logistic,
        or_ci_high,
        or_ci_low,
        or_values,
        se_scratch,
        table_diag,
        table_or,
        table_verif,
        test_acc,
        test_auc,
    )


@app.cell
def _(mo, table_diag, table_or, table_verif):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Newton-Raphson / IRLS Verification

                Validating our vectorized iteratively reweighted least squares solver against `statsmodels.api.Logit` down to machine precision:
                """
            ),
            table_verif,
            mo.md(
                r"""
                ### Example 2: Odds Ratio and Hypothesis Testing Summary

                Comprehensive table displaying log-odds coefficients, Wald statistics, $p$-values, and exponentiated Odds Ratios:
                """
            ),
            table_or,
            mo.md(
                r"""
                ### Example 3: Holdout Performance Diagnostics

                Evaluating out-of-sample accuracy, ROC AUC, and confusion matrix counts on the holdout test partition:
                """
            ),
            table_diag,
        ]
    )


if __name__ == "__main__":
    app.run()
