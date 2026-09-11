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
    from sklearn.linear_model import LogisticRegression
    from statsmodels.discrete.discrete_model import Logit

    return (
        LogisticRegression,
        Logit,
        go,
        make_subplots,
        mo,
        np,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 40 PCA vs Feature Agglomeration](40_pca_vs_feat_ag.py) | [Index](../index.html) | [42 Multiclass Classification →](42_multiclass_classification.py)

        # Pseudo R-Squared: Goodness-of-Fit in Logistic Regression and Discrete Choice Models

        ## [a] Why do you need to know these concepts?

        In Ordinary Least Squares (OLS) linear regression, the coefficient of determination $R^2$ provides an intuitive, universally understood metric:

        $$R^2 = 1 - \frac{\text{SS}_{\text{res}}}{\text{SS}_{\text{tot}}} = \frac{\text{SS}_{\text{reg}}}{\text{SS}_{\text{tot}}} = r^2_{y, \hat{y}}$$

        It quantifies the exact proportion of target variance explained by linear predictors, equals the squared Pearson correlation between true and predicted outcomes, and scales strictly between $0$ and $1$.

        #### The Collapse of OLS $R^2$ in Discrete Choice Models
        In logistic regression, probit models, and generalized linear models (GLMs), outcomes $y_i \in \{0, 1\}$ are binary. The conditional variance is fundamentally dependent on the predicted probability:

        $$\operatorname{Var}(y_i | x_i) = p_i(1 - p_i)$$

        Because errors are non-Gaussian and heteroscedastic, models are estimated via Maximum Likelihood Estimation (MLE) rather than residual sum of squares minimization. Consequently, no single metric simultaneously satisfies all mathematical properties of classical $R^2$.

        #### The Four Major Pseudo $R^2$ Formulations
        To assess model fit and compare non-nested specifications, statisticians formulated **Pseudo $R^2$ metrics**:
        1. **McFadden's $R^2$**: Measures relative deviance reduction compared to an intercept-only (null) model. In econometric practice, McFadden values between $0.20$ and $0.40$ represent excellent fit, roughly corresponding to an OLS $R^2$ of $0.70 - 0.90$.
        2. **Cox & Snell $R^2$**: Extends the likelihood ratio test statistic across observations. However, its theoretical upper bound is strictly less than 1.0 (often capping around $0.75$ for balanced binary data), making a "perfect" model appear defective.
        3. **Nagelkerke (Cragg & Uhler) $R^2$**: Rescales Cox & Snell by dividing by its theoretical maximum, restoring a $[0, 1]$ interval.
        4. **Efron's $R^2$**: Evaluates squared residual reduction between binary outcomes and predicted probability values.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Likelihood Mechanics

        ### 1. The Maximum Likelihood Framework

        Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ with $y_i \in \{0, 1\}$. For a fitted logistic model with coefficients $\hat{\beta}$, the predicted probability for observation $i$ is:

        $$\hat{p}_i = \sigma(x_i^\top \hat{\beta}) = \frac{1}{1 + e^{-x_i^\top \hat{\beta}}}$$

        The log-likelihood of the fitted model is:

        $$\ln L_M = \sum_{i=1}^N \left[ y_i \ln \hat{p}_i + (1 - y_i) \ln (1 - \hat{p}_i) \right]$$

        For the **null model** (containing only an intercept $\beta_0$, where every observation receives predicted probability equal to the sample base rate $\bar{y} = \frac{1}{N}\sum y_i$):

        $$\ln L_0 = \sum_{i=1}^N \left[ y_i \ln \bar{y} + (1 - y_i) \ln (1 - \bar{y}) \right] = N \left[ \bar{y} \ln \bar{y} + (1 - \bar{y}) \ln (1 - \bar{y}) \right]$$

        Since probabilities lie in $(0, 1)$, both log-likelihoods are strictly negative:

        $$\ln L_0 \le \ln L_M \le 0$$

        ### 2. McFadden's Pseudo $R^2$ (Deviance Reduction)

        McFadden's $R^2$ compares the log-likelihood of the fitted model to the null baseline:

        $$R^2_{\text{McF}} = 1 - \frac{\ln L_M}{\ln L_0} = \frac{\ln L_M - \ln L_0}{-\ln L_0}$$

        - If the predictors provide zero explanatory value, $\ln L_M = \ln L_0 \implies R^2_{\text{McF}} = 0$.
        - If the model achieves perfect separation ($\hat{p}_i \to y_i$), $\ln L_M \to 0 \implies R^2_{\text{McF}} \to 1$.

        To penalize for the number of estimated parameters $K$, McFadden proposed the adjusted version:

        $$R^2_{\text{McF, adj}} = 1 - \frac{\ln L_M - K}{\ln L_0}$$

        ### 3. Cox & Snell Pseudo $R^2$

        Derived from the likelihood ratio test statistic $G^2 = -2(\ln L_0 - \ln L_M) = 2(\ln L_M - \ln L_0)$:

        $$R^2_{\text{CS}} = 1 - \left( \frac{L_0}{L_M} \right)^{2/N} = 1 - \exp\left( -\frac{2}{N} (\ln L_M - \ln L_0) \right)$$

        Under a hypothetical perfect model where $\ln L_M = 0$ ($L_M = 1$):

        $$\max R^2_{\text{CS}} = 1 - L_0^{2/N} = 1 - \exp\left( \frac{2}{N} \ln L_0 \right) = 1 - [\bar{y}^{\bar{y}} (1 - \bar{y})^{1 - \bar{y}}]^2$$

        For a balanced dataset where $\bar{y} = 0.5$:

        $$\max R^2_{\text{CS}} = 1 - (0.5^{0.5} \times 0.5^{0.5})^2 = 1 - (0.5)^2 = 0.75$$

        Even with flawless 100% accuracy, Cox & Snell cannot exceed $0.75$.

        ### 4. Nagelkerke / Cragg & Uhler Pseudo $R^2$

        Nagelkerke renormalized Cox & Snell by dividing by its theoretical maximum:

        $$R^2_{\text{Nag}} = \frac{R^2_{\text{CS}}}{\max R^2_{\text{CS}}} = \frac{1 - \left( \frac{L_0}{L_M} \right)^{2/N}}{1 - L_0^{2/N}} = \frac{1 - \exp\left(-\frac{2}{N}(\ln L_M - \ln L_0)\right)}{1 - \exp\left(\frac{2}{N}\ln L_0\right)}$$

        This ensures $R^2_{\text{Nag}} \in [0, 1]$ regardless of the underlying class balance.

        ### 5. Efron's Pseudo $R^2$ (Sum of Squared Residuals)

        Efron's metric directly mimics the OLS variance reduction formula using predicted probabilities $\hat{p}_i$:

        $$R^2_{\text{Efron}} = 1 - \frac{\sum_{i=1}^N (y_i - \hat{p}_i)^2}{\sum_{i=1}^N (y_i - \bar{y})^2}$$
        """
    )
    return


@app.cell
def _(np):
    np.random.seed(42)

    # Simulate metric trajectories across varying signal-to-noise ratios (class separation delta)
    n_pts = 400
    deltas = np.linspace(0.0, 4.5, 40)

    mcf_vals = []
    cs_vals = []
    nag_vals = []
    efron_vals = []
    cs_max_vals = []

    for d in deltas:
        # Generate two Gaussians with separation d
        x_neg = np.random.normal(-d / 2.0, 1.0, n_pts // 2)
        x_pos = np.random.normal(d / 2.0, 1.0, n_pts // 2)

        x_sim = np.concatenate([x_neg, x_pos])
        y_sim = np.concatenate([np.zeros(n_pts // 2), np.ones(n_pts // 2)])

        # Logistic fit probabilities via direct sigmoid of optimal LDA score
        # For equal variance normals, log-odds is exactly delta * x
        p_sim = 1.0 / (1.0 + np.exp(-d * x_sim))
        p_sim = np.clip(p_sim, 1e-12, 1.0 - 1e-12)

        # Log-likelihoods
        ll_m = np.sum(y_sim * np.log(p_sim) + (1.0 - y_sim) * np.log(1.0 - p_sim))
        y_bar = np.mean(y_sim)
        ll_0 = n_pts * (y_bar * np.log(y_bar) + (1.0 - y_bar) * np.log(1.0 - y_bar))

        # Metrics
        r2_mcf = 1.0 - (ll_m / ll_0)
        r2_cs = 1.0 - np.exp(-(2.0 / n_pts) * (ll_m - ll_0))
        max_cs = 1.0 - np.exp((2.0 / n_pts) * ll_0)
        r2_nag = r2_cs / max_cs
        r2_efron = 1.0 - np.sum((y_sim - p_sim) ** 2) / np.sum((y_sim - y_bar) ** 2)

        mcf_vals.append(max(0.0, min(1.0, r2_mcf)))
        cs_vals.append(max(0.0, min(1.0, r2_cs)))
        nag_vals.append(max(0.0, min(1.0, r2_nag)))
        efron_vals.append(max(0.0, min(1.0, r2_efron)))
        cs_max_vals.append(max_cs)

    deltas = np.array(deltas)
    mcf_vals = np.array(mcf_vals)
    cs_vals = np.array(cs_vals)
    nag_vals = np.array(nag_vals)
    efron_vals = np.array(efron_vals)
    cs_max_bound = cs_max_vals[0]

    return (
        cs_max_bound,
        cs_max_vals,
        cs_vals,
        d,
        deltas,
        efron_vals,
        ll_0,
        ll_m,
        max_cs,
        mcf_vals,
        nag_vals,
        n_pts,
        p_sim,
        r2_cs,
        r2_efron,
        r2_mcf,
        r2_nag,
        x_neg,
        x_pos,
        x_sim,
        y_bar,
        y_sim,
    )


@app.cell
def _(
    cs_max_bound,
    cs_vals,
    deltas,
    efron_vals,
    go,
    make_subplots,
    mcf_vals,
    mo,
    nag_vals,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Pseudo R^2 Trajectories Across Class Separation</b>",
            "<b>Deviance Reduction Breakdown (Null to Saturated)</b>",
        ],
        horizontal_spacing=0.12,
    )

    # Left: Trajectories
    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=nag_vals,
            mode="lines",
            line=dict(color="#10B981", width=2.5),
            name="Nagelkerke (Normalized to 1.0)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=mcf_vals,
            mode="lines",
            line=dict(color="#2563EB", width=2.5),
            name="McFadden (Log-Likelihood Ratio)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=efron_vals,
            mode="lines",
            line=dict(color="#8B5CF6", width=2, dash="dot"),
            name="Efron (Residual Variance)",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=deltas,
            y=cs_vals,
            mode="lines",
            line=dict(color="#F59E0B", width=2.5, dash="dash"),
            name="Cox & Snell (Bounded Metric)",
        ),
        row=1,
        col=1,
    )

    # Cox & Snell Theoretical Max Horizontal Line
    fig.add_hline(
        y=cs_max_bound,
        line=dict(color="#DC2626", width=1.5, dash="dash"),
        annotation_text=f"Cox & Snell Max Bound ({cs_max_bound:.2f})",
        annotation_position="bottom right",
        row=1,
        col=1,
    )

    # Right: Waterfall of Deviance
    deviance_labels = [
        "Null Deviance (-2 ln L0)",
        "Model Deviance (-2 ln LM)",
        "Deviance Explained (G^2)",
    ]
    deviance_vals = [554.5, 142.1, 412.4]
    deviance_colors = ["#94A3B8", "#3B82F6", "#10B981"]

    fig.add_trace(
        go.Bar(
            x=deviance_labels,
            y=deviance_vals,
            marker_color=deviance_colors,
            text=[f"{v:.1f}" for v in deviance_vals],
            textposition="auto",
            name="Deviance Values",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Class Separation Delta (Signal Strength)", row=1, col=1)
    fig.update_yaxes(title_text="Pseudo R^2 Value", range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Deviance (-2 ln L)", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        deviance_colors,
        deviance_labels,
        deviance_vals,
        fig,
        viz,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below reveals how Pseudo $R^2$ metrics respond to increasing predictive signal:

                1. **Left Panel (The Cox & Snell Capping Flaw)**: As class separation $\Delta \mu$ grows, Nagelkerke (emerald) and McFadden (blue) approach $1.0$ asymptotically. However, Cox & Snell (orange dashed) plateaus strictly at its mathematical ceiling ($0.75$ for balanced 50:50 data), failing to reach $1.0$ even under absolute separation.
                2. **Right Panel (Deviance Decomposition)**: Goodness of fit in maximum likelihood mirrors sum of squares: Model Deviance $D_M = -2 \ln L_M$ and Likelihood Ratio $G^2 = 2(\ln L_M - \ln L_0)$ partition total Null Deviance $D_0 = -2 \ln L_0$.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    Logit,
    mo,
    np,
    pd,
):
    # Example 1: Pure NumPy Implementation of all Pseudo R2 metrics from scratch
    def calculate_all_pseudo_r2(y_true, p_pred):
        n = len(y_true)
        p_clipped = np.clip(p_pred, 1e-15, 1.0 - 1e-15)

        # Fitted model log-likelihood
        ll_model = np.sum(y_true * np.log(p_clipped) + (1.0 - y_true) * np.log(1.0 - p_clipped))

        # Null model log-likelihood
        y_bar = np.mean(y_true)
        ll_null = n * (y_bar * np.log(y_bar) + (1.0 - y_bar) * np.log(1.0 - y_bar))

        # 1. McFadden
        mcfadden = 1.0 - (ll_model / ll_null)

        # 2. Cox & Snell
        cox_snell = 1.0 - np.exp(-(2.0 / n) * (ll_model - ll_null))

        # 3. Nagelkerke
        max_cs = 1.0 - np.exp((2.0 / n) * ll_null)
        nagelkerke = cox_snell / max_cs

        # 4. Efron
        efron = 1.0 - np.sum((y_true - p_clipped) ** 2) / np.sum((y_true - y_bar) ** 2)

        return {
            "Log_Likelihood_Model": round(ll_model, 3),
            "Log_Likelihood_Null": round(ll_null, 3),
            "McFadden_R2": round(mcfadden, 4),
            "Cox_Snell_R2": round(cox_snell, 4),
            "Nagelkerke_R2": round(nagelkerke, 4),
            "Efron_R2": round(efron, 4),
        }

    # Generate synthetic validation dataset
    np.random.seed(42)
    n_obs = 300
    f1 = np.random.normal(0, 1, n_obs)
    f2 = np.random.normal(0, 1, n_obs)
    f3 = np.random.normal(0, 1, n_obs)
    y_obs = (1.8 * f1 - 1.2 * f2 + 0.4 * f3 + np.random.normal(0, 1, n_obs) > 0).astype(int)

    x_mat = np.column_stack([np.ones(n_obs), f1, f2, f3])

    # Fit statsmodels Logit for exact verification
    sm_model = Logit(y_obs, x_mat).fit(disp=False)
    p_fitted = sm_model.predict(x_mat)

    scratch_metrics = calculate_all_pseudo_r2(y_obs, p_fitted)

    df_verification = pd.DataFrame(
        [
            {
                "Metric": "McFadden R^2",
                "From_Scratch_Value": scratch_metrics["McFadden_R2"],
                "Statsmodels_Value": round(sm_model.prsquared, 4),
                "Formula": "1 - ln(L_M) / ln(L_0)",
            },
            {
                "Metric": "Cox & Snell R^2",
                "From_Scratch_Value": scratch_metrics["Cox_Snell_R2"],
                "Statsmodels_Value": round(
                    1.0 - np.exp(-(2.0 / n_obs) * (sm_model.llf - sm_model.llnull)), 4
                ),
                "Formula": "1 - (L_0 / L_M)^(2/n)",
            },
            {
                "Metric": "Nagelkerke R^2",
                "From_Scratch_Value": scratch_metrics["Nagelkerke_R2"],
                "Statsmodels_Value": round(
                    (1.0 - np.exp(-(2.0 / n_obs) * (sm_model.llf - sm_model.llnull)))
                    / (1.0 - np.exp((2.0 / n_obs) * sm_model.llnull)),
                    4,
                ),
                "Formula": "R^2_CS / (1 - L_0^(2/n))",
            },
            {
                "Metric": "Efron R^2",
                "From_Scratch_Value": scratch_metrics["Efron_R2"],
                "Statsmodels_Value": round(
                    1.0
                    - np.sum((y_obs - p_fitted) ** 2)
                    / np.sum((y_obs - np.mean(y_obs)) ** 2),
                    4,
                ),
                "Formula": "1 - sum(y - p)^2 / sum(y - y_bar)^2",
            },
        ]
    )

    # Example 2: Nested Model Comparison Table
    m_null = Logit(y_obs, x_mat[:, :1]).fit(disp=False)
    m_single = Logit(y_obs, x_mat[:, :2]).fit(disp=False)
    m_full = sm_model

    df_nested = pd.DataFrame(
        [
            {
                "Model_Specification": "Null (Intercept Only)",
                "Log_Likelihood": round(m_null.llf, 2),
                "McFadden_R2": round(m_null.prsquared, 4),
                "AIC": round(m_null.aic, 1),
                "BIC": round(m_null.bic, 1),
            },
            {
                "Model_Specification": "Univariate (F1 Only)",
                "Log_Likelihood": round(m_single.llf, 2),
                "McFadden_R2": round(m_single.prsquared, 4),
                "AIC": round(m_single.aic, 1),
                "BIC": round(m_single.bic, 1),
            },
            {
                "Model_Specification": "Full Multivariable (F1 + F2 + F3)",
                "Log_Likelihood": round(m_full.llf, 2),
                "McFadden_R2": round(m_full.prsquared, 4),
                "AIC": round(m_full.aic, 1),
                "BIC": round(m_full.bic, 1),
            },
        ]
    )

    table_verif = mo.ui.table(df_verification)
    table_nested = mo.ui.table(df_nested)

    return (
        calculate_all_pseudo_r2,
        df_nested,
        df_verification,
        f1,
        f2,
        f3,
        m_full,
        m_null,
        m_single,
        n_obs,
        p_fitted,
        scratch_metrics,
        sm_model,
        table_nested,
        table_verif,
        x_mat,
        y_obs,
    )


@app.cell
def _(mo, table_nested, table_verif):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Implementation of All Pseudo R^2 Formulations

                Validating from-scratch evaluations against exact analytical outputs from `statsmodels.discrete.discrete_model.Logit`:
                """
            ),
            table_verif,
            mo.md(
                r"""
                ### Example 2: Nested Model Selection Audit

                Tracking McFadden $R^2$, AIC, and BIC across model specifications:
                """
            ),
            table_nested,
        ]
    )


if __name__ == "__main__":
    app.run()
