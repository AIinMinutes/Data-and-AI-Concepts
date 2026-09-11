import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import dice_ml
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from sklearn.datasets import load_wine
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    return (
        LogisticRegression,
        StandardScaler,
        accuracy_score,
        dice_ml,
        go,
        load_wine,
        make_subplots,
        mo,
        np,
        pd,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 45 Shapley Values](45_shapley.py) | [Index](../index.html) | [47 GELU Activation →](47_gelu.py)

        # Model Counterfactuals: Actionable Algorithmic Recourse and Diverse Counterfactual Explanations (DiCE)

        ## [a] Why do you need to know these concepts?

        Feature attribution frameworks (such as SHAP and Permutation Importance) explain model behavior by answering a backward-looking diagnostic question: *"Which features contributed most heavily to the model's decision?"*

        However, in high-stakes automated decisions (such as credit lending, job candidate screening, college admissions, and criminal justice risk scoring), an applicant whose loan is rejected is not helped by knowing that their credit score was the dominant negative factor. The applicant requires an actionable, forward-looking answer to a different question:

        > **"What minimal, feasible changes can I make to my application to turn this rejection into an approval?"**

        #### Algorithmic Recourse
        **Counterfactual Explanations** (formalized by Wachter, Mittelstadt, and Russell in 2017) identify the closest possible hypothetical input vector $x^* = x + \Delta x$ that flips the model's prediction across the decision boundary to the desired target class $y^*$.

        #### Model Counterfactuals vs Causal Counterfactuals
        - **Model Counterfactuals**: Search the empirical decision manifold of a frozen predictive classifier $\hat{f}(x)$ to find boundary crossings. They operate within the model's internal feature space without requiring an underlying causal Directed Acyclic Graph (DAG).
        - **Causal Counterfactuals**: Formulated under Judea Pearl's Structural Causal Models (SCMs) and the Potential Outcomes framework ($do$-calculus). They model the true physical consequences of real-world interventions, accounting for downstream ripple effects.

        #### Diverse Counterfactual Explanations (DiCE)
        Providing a single counterfactual is often inadequate in practice. A proposed change might demand an action that is impossible or prohibitively expensive for a specific user (e.g. moving to a new zip code or increasing annual income by $100,000 in one month).

        **DiCE (Mothilal et al., 2020)** addresses this limitation by formulating a multi-objective optimization problem. It finds a diverse set of $K$ distinct counterfactuals, optimizing for:
        1. **Proximity**: Minimizing the overall perturbation distance between the factual point and counterfactuals.
        2. **Sparsity**: Changing as few features as possible.
        3. **Diversity**: Ensuring the counterfactuals offer qualitatively different pathways to approval via Determinantal Point Processes (DPP).
        4. **Feasibility**: Enforcing immutable features (e.g. age cannot decrease, race and gender cannot be altered) and realistic numerical bounds.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Optimization Mechanics

        ### 1. The Wachter Minimal Perturbation Formulation

        Let $f: \mathbb{R}^p \to [0, 1]$ be a trained binary classifier outputting the probability of a positive outcome. Let $x \in \mathbb{R}^p$ denote the factual observation where $f(x) < 0.5$ (e.g. loan rejected). Let $y^* = 1$ denote the desired target label.

        Finding a counterfactual point $x^* \in \mathbb{R}^p$ is formulated as an unconstrained optimization problem:

        $$x^* = \arg\min_{x'} \operatorname{dist}(x, x') + \lambda \cdot \ell(f(x'), y^*)$$

        where:
        - $\ell(f(x'), y^*)$ is a loss function penalizing deviation from the desired prediction:

        $$\ell(f(x'), 1) = \max\left(0, \ 0.5 - f(x')\right)^2 \quad \text{or} \quad -\ln(f(x'))$$

        - $\operatorname{dist}(x, x')$ is a normalized distance metric in feature space.
        - $\lambda > 0$ is a regularizing hyperparameter that balances proximity to the original factual instance against confidence in crossing the decision boundary.

        ### 2. Distance Metrics: The MAD-Normalized Manhattan Metric

        Using standard Euclidean distance ($L_2$) causes high-variance features to dominate the optimization and produces dense solutions where every feature is perturbed by a tiny fraction.

        To encourage **sparsity** (changing as few attributes as possible) and account for feature scales, Wachter proposed the $L_1$ distance normalized by the **Median Absolute Deviation (MAD)** of each feature across the training population:

        $$\operatorname{dist}_{\text{MAD}}(x, x') = \sum_{j=1}^p \frac{|x_j - x'_j|}{\operatorname{MAD}_j}$$

        where the Median Absolute Deviation for feature $j$ is:

        $$\operatorname{MAD}_j = \operatorname{median}_{i=1,\dots,N} \left| x_{ij} - \operatorname{median}_{k=1,\dots,N}(x_{kj}) \right|$$

        Unlike the standard deviation, MAD is robust to extreme training outliers.

        ### 3. DiCE: Multi-Objective Diversity Optimization

        Rather than finding a single counterfactual, DiCE generates $K$ counterfactuals $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ by optimizing a joint objective function:

        $$\min_{c_1, \dots, c_K} \frac{1}{K}\sum_{k=1}^K \ell(f(c_k), y^*) + \frac{\lambda_1}{K}\sum_{k=1}^K \operatorname{dist}_{\text{MAD}}(x, c_k) - \lambda_2 \cdot \operatorname{dpp\_diversity}(c_1, \dots, c_K)$$

        The diversity term utilizes a **Determinantal Point Process (DPP)**. Let $\mathbf{K} \in \mathbb{R}^{K \times K}$ be a kernel similarity matrix between counterfactual candidates, with entries:

        $$K_{i, j} = \frac{1}{1 + \operatorname{dist}(c_i, c_j)}$$

        The diversity reward is defined as the determinant of $\mathbf{K}$:

        $$\operatorname{dpp\_diversity}(c_1, \dots, c_K) = \det(\mathbf{K})$$

        - If two counterfactual candidates $c_i$ and $c_j$ are nearly identical, the rows of $\mathbf{K}$ become linearly dependent, driving $\det(\mathbf{K}) \to 0$.
        - Maximizing $\det(\mathbf{K})$ repels counterfactuals from one another in feature space, providing the user with distinct recourse strategies.

        ### 4. Feasibility and Actionability Constraints

        Real-world recourse must respect domain realities:

        $$\begin{aligned}
        c_{k, j} &= x_j \quad \forall j \in \mathcal{F}_{\text{immutable}} \quad &\text{(e.g. Country of Origin, Race)} \\
        c_{k, j} &\ge x_j \quad \forall j \in \mathcal{F}_{\text{monotonic}} \quad &\text{(e.g. Education Level, Work Experience)} \\
        L_j &\le c_{k, j} \le U_j \quad \forall j \in \{1, \dots, p\} \quad &\text{(Physical bounds, e.g. } 0 \le \text{Credit Score} \le 850)
        \end{aligned}$$
        """
    )
    return


@app.cell
def _(
    LogisticRegression,
    StandardScaler,
    dice_ml,
    load_wine,
    np,
    pd,
    train_test_split,
):
    # Load standardized tabular dataset (Wine classification: Class 0 vs Other classes)
    wine = load_wine(as_frame=True)
    df_raw = wine.frame

    # Focus on two primary continuous features for visual geometric intuition:
    # 'alcohol' and 'flavanoids'
    features_subset = ["alcohol", "flavanoids"]
    x_data = df_raw[features_subset].copy()
    # Binary outcome: Class 0 (Cultivar 1) vs Other Cultivars
    y_binary = (df_raw["target"] == 0).astype(int)

    # Standardize features for linear model training
    scaler = StandardScaler()
    x_scaled_arr = scaler.fit_transform(x_data)
    df_scaled = pd.DataFrame(x_scaled_arr, columns=["Alcohol_Std", "Flavanoids_Std"])
    df_scaled["Target"] = y_binary.values

    # Train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        df_scaled[["Alcohol_Std", "Flavanoids_Std"]],
        df_scaled["Target"],
        test_size=0.25,
        random_state=42,
        stratify=df_scaled["Target"],
    )

    # Train logistic regression classifier
    clf_model = LogisticRegression(random_state=42)
    clf_model.fit(x_train, y_train)

    # Select a rejected factual instance (Target = 0 with high confidence)
    # Find a test sample with low predicted probability
    test_probs = clf_model.predict_proba(x_test)[:, 1]
    neg_idx = np.argmin(test_probs)
    factual_sample = x_test.iloc[neg_idx : neg_idx + 1].copy()
    factual_prob = float(test_probs[neg_idx])

    # Setup DiCE framework
    d_data = dice_ml.Data(
        dataframe=pd.concat([x_train, y_train], axis=1),
        continuous_features=["Alcohol_Std", "Flavanoids_Std"],
        outcome_name="Target",
    )
    d_model = dice_ml.Model(model=clf_model, backend="sklearn", model_type="classifier")
    dice_exp = dice_ml.Dice(d_data, d_model, method="random")

    # Generate 3 diverse counterfactuals targeting Class 1 (approval threshold p >= 0.5)
    cf_res = dice_exp.generate_counterfactuals(
        factual_sample, total_CFs=3, desired_class=1
    )
    df_cfs = cf_res.cf_examples_list[0].final_cfs_df

    # Extract coordinates
    factual_coords = factual_sample[["Alcohol_Std", "Flavanoids_Std"]].values[0]
    cf_coords = df_cfs[["Alcohol_Std", "Flavanoids_Std"]].values

    return (
        cf_coords,
        cf_res,
        clf_model,
        d_data,
        d_model,
        df_cfs,
        df_raw,
        df_scaled,
        dice_exp,
        factual_coords,
        factual_prob,
        factual_sample,
        features_subset,
        neg_idx,
        scaler,
        test_probs,
        wine,
        x_data,
        x_scaled_arr,
        x_test,
        x_train,
        y_binary,
        y_test,
        y_train,
    )


@app.cell
def _(
    cf_coords,
    clf_model,
    factual_coords,
    go,
    make_subplots,
    mo,
    np,
    x_train,
    y_train,
):
    # 2D Grid for Decision Boundary
    x_min, x_max = x_train["Alcohol_Std"].min() - 0.8, x_train["Alcohol_Std"].max() + 0.8
    y_min, y_max = x_train["Flavanoids_Std"].min() - 0.8, x_train["Flavanoids_Std"].max() + 0.8

    grid_x, grid_y = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    grid_points = np.c_[grid_x.ravel(), grid_y.ravel()]
    grid_df = pd.DataFrame(grid_points, columns=["Alcohol_Std", "Flavanoids_Std"])
    prob_surface = clf_model.predict_proba(grid_df)[:, 1].reshape(grid_x.shape)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Decision Boundary and Counterfactual Recourse Vectors</b>",
            "<b>Feature Perturbation Magnitude by Recourse Option</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Left: Decision surface contour
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=prob_surface,
            colorscale="RdBu",
            contours=dict(start=0.1, end=0.9, size=0.1, showlabels=True),
            colorbar=dict(title="P(Approved)", x=0.44, len=0.8),
            name="Model Probability P(Y=1)",
        ),
        row=1,
        col=1,
    )

    # Decision Boundary Line (p = 0.5)
    fig.add_trace(
        go.Contour(
            x=np.linspace(x_min, x_max, 100),
            y=np.linspace(y_min, y_max, 100),
            z=prob_surface,
            contours=dict(type="constraint", value=0.5),
            line=dict(color="#10B981", width=3),
            name="Decision Boundary (p=0.5)",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Plot factual rejected sample
    fig.add_trace(
        go.Scatter(
            x=[factual_coords[0]],
            y=[factual_coords[1]],
            mode="markers+text",
            marker=dict(color="#DC2626", size=14, symbol="diamond"),
            text=["Factual (Rejected)"],
            textposition="bottom right",
            name="Factual Instance",
        ),
        row=1,
        col=1,
    )

    # Plot counterfactual approved points
    cf_colors = ["#2563EB", "#8B5CF6", "#F59E0B"]
    for i, (cf, color) in enumerate(zip(cf_coords, cf_colors)):
        # Point
        fig.add_trace(
            go.Scatter(
                x=[cf[0]],
                y=[cf[1]],
                mode="markers+text",
                marker=dict(color=color, size=12, symbol="star"),
                text=[f"CF {i+1}"],
                textposition="top left",
                name=f"Counterfactual {i+1}",
            ),
            row=1,
            col=1,
        )

        # Vector line connecting factual to counterfactual
        fig.add_trace(
            go.Scatter(
                x=[factual_coords[0], cf[0]],
                y=[factual_coords[1], cf[1]],
                mode="lines",
                line=dict(color=color, width=2, dash="dot"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Right Panel: Feature shift delta bar chart across counterfactuals
    features = ["Alcohol (Std)", "Flavanoids (Std)"]
    for i, (cf, color) in enumerate(zip(cf_coords, cf_colors)):
        deltas = cf - factual_coords
        fig.add_trace(
            go.Bar(
                x=features,
                y=deltas,
                name=f"Recourse Option {i+1}",
                marker_color=color,
            ),
            row=1,
            col=2,
        )

    fig.update_xaxes(title_text="Standardized Alcohol", row=1, col=1)
    fig.update_yaxes(title_text="Standardized Flavanoids", row=1, col=1)
    fig.update_xaxes(title_text="Feature Dimension", row=1, col=2)
    fig.update_yaxes(title_text="Required Shift Delta (x* - x)", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        barmode="group",
    )

    viz = mo.ui.plotly(fig)
    return (
        cf,
        cf_colors,
        color,
        deltas,
        features,
        fig,
        grid_points,
        grid_x,
        grid_y,
        i,
        prob_surface,
        viz,
        x_max,
        x_min,
        y_max,
        y_min,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below reveals how algorithmic recourse navigates model decision boundaries:

                1. **Left Panel (Boundary Crossing Vectors)**: The rejected applicant (red diamond, $P(\text{Approved}) \approx 0.01$) lies deep within the negative class region. Generated counterfactuals (CF 1, 2, 3) identify alternative shortest vectors across the green decision boundary ($p = 0.5$).
                2. **Right Panel (Actionable Trade-Offs)**: Different counterfactuals offer qualitatively different trade-offs: Recourse Option 1 achieves approval through a large increase in Flavanoids alone, while Option 3 balances smaller simultaneous increases across both Alcohol and Flavanoids.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    cf_coords,
    clf_model,
    factual_coords,
    factual_prob,
    mo,
    np,
    pd,
    x_train,
):
    # Example 1: Pure NumPy Gradient Descent Solver for Wachter Counterfactual
    def wachter_counterfactual_solver(model, x_fact, lr=0.08, max_iter=200, lam=2.5):
        w = model.coef_[0]
        b = model.intercept_[0]
        x_cf = x_fact.copy()

        for _ in range(max_iter):
            # Probability: sigma(w^T x + b)
            logit_val = np.dot(w, x_cf) + b
            prob = 1.0 / (1.0 + np.exp(-logit_val))

            if prob >= 0.505:
                break

            # Distance loss: ||x_cf - x_fact||_2^2
            grad_dist = 2.0 * (x_cf - x_fact)

            # Target loss: (0.5 - prob)^2
            # d/dx prob = prob * (1 - prob) * w
            d_loss_d_prob = -2.0 * (0.5 - prob)
            grad_target = d_loss_d_prob * (prob * (1.0 - prob)) * w

            total_grad = grad_dist + lam * grad_target
            x_cf = x_cf - lr * total_grad

        final_prob = 1.0 / (1.0 + np.exp(-(np.dot(w, x_cf) + b)))
        return x_cf, final_prob

    scratch_cf, scratch_prob = wachter_counterfactual_solver(clf_model, factual_coords)

    df_solver_verif = pd.DataFrame(
        [
            {
                "Candidate": "Factual (Rejected)",
                "Alcohol_Std": round(factual_coords[0], 3),
                "Flavanoids_Std": round(factual_coords[1], 3),
                "Model_Probability": f"{factual_prob * 100:.2f}%",
                "Decision": "Rejected",
            },
            {
                "Candidate": "Wachter Solver (From Scratch)",
                "Alcohol_Std": round(scratch_cf[0], 3),
                "Flavanoids_Std": round(scratch_cf[1], 3),
                "Model_Probability": f"{scratch_prob * 100:.2f}%",
                "Decision": "Approved (Crossed p=0.5)",
            },
        ]
    )

    # Example 2: MAD-Normalized Distance and Recourse Metric Comparison
    mad_values = np.median(
        np.abs(x_train.values - np.median(x_train.values, axis=0)), axis=0
    )
    # Prevent divide by zero
    mad_values = np.maximum(mad_values, 1e-4)

    recourse_records = []
    for idx, cf_pt in enumerate(cf_coords):
        cf_df_pt = pd.DataFrame([cf_pt], columns=["Alcohol_Std", "Flavanoids_Std"])
        p_cf = clf_model.predict_proba(cf_df_pt)[0, 1]
        l1_dist = np.sum(np.abs(cf_pt - factual_coords))
        mad_dist = np.sum(np.abs(cf_pt - factual_coords) / mad_values)
        num_changes = int(np.sum(np.abs(cf_pt - factual_coords) > 0.05))

        recourse_records.append(
            {
                "Recourse_Option": f"Counterfactual {idx + 1}",
                "Alcohol_Target": round(cf_pt[0], 3),
                "Flavanoids_Target": round(cf_pt[1], 3),
                "Predicted_Probability": f"{p_cf * 100:.2f}%",
                "L1_Distance": round(l1_dist, 3),
                "MAD_Normalized_Cost": round(mad_dist, 3),
                "Features_Altered": f"{num_changes} / 2",
            }
        )

    df_recourse = pd.DataFrame(recourse_records)

    # Example 3: Comparison between Model Counterfactuals and Causal Counterfactuals
    df_comparison = pd.DataFrame(
        [
            {
                "Dimension": "Operational Context",
                "Model_Counterfactual": "Inside frozen ML model boundary",
                "Causal_Counterfactual": "In real-world physical universe",
            },
            {
                "Dimension": "Primary Objective",
                "Model_Counterfactual": "Explain/reverse model prediction",
                "Causal_Counterfactual": "Estimate intervention effect",
            },
            {
                "Dimension": "Structural Assumption",
                "Model_Counterfactual": "Smooth classifier manifold",
                "Causal_Counterfactual": "Structural Causal Model / DAG",
            },
            {
                "Dimension": "Typical Use Case",
                "Model_Counterfactual": "User algorithmic recourse",
                "Causal_Counterfactual": "Policy impact and medical treatment",
            },
        ]
    )

    table_solver = mo.ui.table(df_solver_verif)
    table_recourse = mo.ui.table(df_recourse)
    table_comp = mo.ui.table(df_comparison)

    return (
        df_comparison,
        df_recourse,
        df_solver_verif,
        l1_dist,
        mad_dist,
        mad_values,
        num_changes,
        p_cf,
        recourse_records,
        scratch_cf,
        scratch_prob,
        table_comp,
        table_recourse,
        table_solver,
        wachter_counterfactual_solver,
    )


@app.cell
def _(mo, table_comp, table_recourse, table_solver):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Wachter Gradient Optimization

                Validating our from-scratch Wachter gradient descent optimizer crossing the decision threshold:
                """
            ),
            table_solver,
            mo.md(
                r"""
                ### Example 2: DiCE Diverse Recourse Options and Cost Audit

                Evaluating alternative pathways by L1 distance, MAD-normalized effort, and sparsity:
                """
            ),
            table_recourse,
            mo.md(
                r"""
                ### Example 3: Model Counterfactuals vs Causal Counterfactuals

                Contrasting model-level perturbation with physical real-world causal interventions:
                """
            ),
            table_comp,
        ]
    )


if __name__ == "__main__":
    app.run()
