import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from imblearn.over_sampling import ADASYN, SMOTE, RandomOverSampler
    from plotly.subplots import make_subplots
    from sklearn.datasets import make_moons
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        balanced_accuracy_score,
        f1_score,
        precision_score,
        recall_score,
    )
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import NearestNeighbors

    return (
        ADASYN,
        NearestNeighbors,
        RandomForestClassifier,
        RandomOverSampler,
        SMOTE,
        balanced_accuracy_score,
        f1_score,
        go,
        make_moons,
        make_subplots,
        mo,
        np,
        pd,
        precision_score,
        recall_score,
        train_test_split,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 37 Natural Breaks](37_natural_breaks.py) | [Index](../index.html) | [39 Permutation Importance →](39_permutation_importance.py)

        # Synthetic Minority Over-sampling: SMOTE, ADASYN, and Decision Boundary Topology

        ## [a] Why do you need to know these concepts?

        Severe class imbalance represents a pervasive challenge in applied machine learning, appearing routinely in fraud detection (where fraudulent transactions comprise less than 0.1% of traffic), rare disease diagnosis, click-through rate modeling, and anomaly detection.

        #### The Failure of Standard Empirical Risk Minimization
        Standard classification models optimize overall empirical loss (such as binary cross-entropy or 0-1 misclassification error). When faced with a 99:1 imbalance ratio, a trivial classifier that predicts the majority class for every sample achieves 99% accuracy while exhibiting 0% recall on the minority class of interest.

        #### The Hazards of Naive Random Oversampling
        The simplest heuristic to balance class frequencies is naive random oversampling (duplicating existing minority samples with replacement). However, exact point duplication artificially shrinks the variance of minority decision regions. Non-parametric models (such as decision trees, Random Forests, and deep neural networks) simply memorize the replicated coordinates, producing tightly localized, overfitted decision boundaries that fail to generalize.

        #### Synthetic Interpolation: SMOTE and ADASYN
        To expand the support of the minority class without exact duplication, Chawla et al. (2002) introduced **SMOTE** (Synthetic Minority Over-sampling Technique). SMOTE creates convex combinations between neighboring minority points, establishing connected linear paths in feature space.

        Building upon SMOTE, **ADASYN** (Adaptive Synthetic Sampling, He et al., 2008) adaptively adjusts the generation rate: it synthesizes more samples for minority observations located in ambiguous regions near the decision boundary (high proportion of majority neighbors) and fewer samples for observations in safe, well-separated cluster interiors.

        #### Preventing Data Leakage
        A critical rule in applied validation: **Resampling must never be applied to the validation or test sets**. Over-sampling the full dataset prior to train-test splitting leaks synthesized variants of test points directly into the training partition, producing artificially inflated metrics.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Sampling Mechanics

        ### 1. The Geometry of SMOTE

        Let $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^N$ be a dataset with $x_i \in \mathbb{R}^p$ and binary labels $y_i \in \{0, 1\}$. Let $\mathcal{S}_{\min} = \{x_i : y_i = 1\}$ denote the minority set with cardinality $N_{\min} = |\mathcal{S}_{\min}|$ and $\mathcal{S}_{\text{maj}} = \{x_i : y_i = 0\}$ denote the majority set ($N_{\text{maj}} \gg N_{\min}$).

        For every observation $x_i \in \mathcal{S}_{\min}$:
        1. Compute the $k$-nearest neighbors of $x_i$ exclusively within $\mathcal{S}_{\min}$ using Euclidean metric $\| \cdot \|_2$:

        $$\mathcal{N}_k(x_i) = \{x_{i}^{(1)}, x_{i}^{(2)}, \dots, x_{i}^{(k)}\} \subset \mathcal{S}_{\min}$$

        2. Randomly select one neighbor $\hat{x}_i \in \mathcal{N}_k(x_i)$.
        3. Draw a uniform random scalar $\lambda \sim \mathcal{U}(0, 1)$.
        4. Synthesize a new continuous observation along the directed line segment:

        $$x_{\text{syn}} = x_i + \lambda (\hat{x}_i - x_i)$$

        The new point $x_{\text{syn}}$ lies on the line segment connecting $x_i$ and $\hat{x}_i$, effectively populating the convex hull of the minority class.

        ### 2. ADASYN: Adaptive Synthetic Sampling

        While SMOTE generates a uniform number of synthetic points per minority observation, ADASYN weights generation by local classification difficulty.

        1. For each $x_i \in \mathcal{S}_{\min}$, identify its $k$ nearest neighbors in the **entire dataset** $\mathcal{D}$ (both minority and majority points).
        2. Let $\Delta_i$ be the count of neighbors belonging to the majority class $\mathcal{S}_{\text{maj}}$. The local difficulty ratio is:

        $$r_i = \frac{\Delta_i}{k} \in [0, 1]$$

        - If $r_i = 0$, $x_i$ is surrounded purely by minority points (easy core sample).
        - If $r_i \approx 1$, $x_i$ is heavily surrounded by majority points (hard boundary sample).

        3. Normalize the difficulty ratios into a discrete probability mass function:

        $$\hat{r}_i = \frac{r_i}{\sum_{j \in \mathcal{S}_{\min}} r_j}$$

        4. Let $G = (N_{\text{maj}} - N_{\min}) \times \beta$ be the total synthetic quota required to achieve balance ratio $\beta \in (0, 1]$. The count of synthetic samples allocated to $x_i$ is:

        $$g_i = \text{round}\left(\hat{r}_i \cdot G\right)$$

        Each of the $g_i$ samples is synthesized using the standard linear interpolation formula from a randomly chosen neighbor in $\mathcal{N}_k(x_i) \cap \mathcal{S}_{\min}$.

        ### 3. Borderline-SMOTE Sample Categorization

        Borderline-SMOTE refines sample selection by classifying every minority point $x_i$ based on its majority neighbor count $m_i \in \{0, \dots, k\}$:

        - **SAFE ($0 \le m_i < k/2$)**: Well within the interior of the minority region. Generating synthetic samples here provides minimal value.
        - **DANGER ($k/2 \le m_i < k$)**: Located right on the decision boundary. Only points in this subset are used as base samples for synthesis.
        - **NOISE ($m_i = k$)**: Completely isolated minority outliers surrounded entirely by majority points. Generating samples from noise points would create spurious minority bridges deep inside the majority region.
        """
    )
    return


@app.cell
def _(
    ADASYN,
    RandomOverSampler,
    SMOTE,
    make_moons,
    np,
    train_test_split,
):
    np.random.seed(42)

    # Generate synthetic 2D non-linear moon dataset
    x_raw, y_raw = make_moons(n_samples=1000, noise=0.28, random_state=42)

    # Impose strong 10:1 class imbalance: 450 majority (Class 0) vs 45 minority (Class 1)
    idx_maj = np.where(y_raw == 0)[0][:450]
    idx_min = np.where(y_raw == 1)[0][:45]

    x_imbalanced = np.vstack([x_raw[idx_maj], x_raw[idx_min]])
    y_imbalanced = np.hstack([y_raw[idx_maj], y_raw[idx_min]])

    # Train-test split (stratified)
    x_train, x_test, y_train, y_test = train_test_split(
        x_imbalanced,
        y_imbalanced,
        test_size=0.30,
        random_state=42,
        stratify=y_imbalanced,
    )

    # Apply resamplers ONLY to training partition
    # 1. SMOTE
    smote_resampler = SMOTE(random_state=42, k_neighbors=5)
    x_train_smote, y_train_smote = smote_resampler.fit_resample(x_train, y_train)

    # 2. ADASYN
    adasyn_resampler = ADASYN(random_state=42, n_neighbors=5)
    x_train_adasyn, y_train_adasyn = adasyn_resampler.fit_resample(x_train, y_train)

    # 3. Random Oversampling
    ros_resampler = RandomOverSampler(random_state=42)
    x_train_ros, y_train_ros = ros_resampler.fit_resample(x_train, y_train)

    return (
        adasyn_resampler,
        idx_maj,
        idx_min,
        ros_resampler,
        smote_resampler,
        x_imbalanced,
        x_raw,
        x_test,
        x_train,
        x_train_adasyn,
        x_train_ros,
        x_train_smote,
        y_imbalanced,
        y_raw,
        y_test,
        y_train,
        y_train_adasyn,
        y_train_ros,
        y_train_smote,
    )


@app.cell
def _(
    go,
    make_subplots,
    mo,
    x_train,
    x_train_adasyn,
    x_train_smote,
    y_train,
    y_train_adasyn,
    y_train_smote,
):
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[
            "<b>(a) Original Imbalanced (10:1)</b>",
            "<b>(b) SMOTE Resampled (Uniform Interpolation)</b>",
            "<b>(c) ADASYN Resampled (Boundary Focused)</b>",
        ],
        horizontal_spacing=0.06,
    )

    # (a) Original
    fig.add_trace(
        go.Scatter(
            x=x_train[y_train == 0, 0],
            y=x_train[y_train == 0, 1],
            mode="markers",
            marker=dict(color="#3B82F6", size=5, opacity=0.6),
            name="Majority Class 0",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_train[y_train == 1, 0],
            y=x_train[y_train == 1, 1],
            mode="markers",
            marker=dict(color="#DC2626", size=8, symbol="diamond"),
            name="Minority Class 1",
        ),
        row=1,
        col=1,
    )

    # (b) SMOTE
    n_orig_min = len(x_train[y_train == 1])
    n_total_train = len(x_train)

    fig.add_trace(
        go.Scatter(
            x=x_train_smote[y_train_smote == 0, 0],
            y=x_train_smote[y_train_smote == 0, 1],
            mode="markers",
            marker=dict(color="#3B82F6", size=5, opacity=0.4),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Original minority
    fig.add_trace(
        go.Scatter(
            x=x_train[y_train == 1, 0],
            y=x_train[y_train == 1, 1],
            mode="markers",
            marker=dict(color="#DC2626", size=7, symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    # Synthesized points
    fig.add_trace(
        go.Scatter(
            x=x_train_smote[n_total_train:, 0],
            y=x_train_smote[n_total_train:, 1],
            mode="markers",
            marker=dict(color="#10B981", size=6, opacity=0.75, symbol="circle"),
            name="SMOTE Synthesized",
        ),
        row=1,
        col=2,
    )

    # (c) ADASYN
    fig.add_trace(
        go.Scatter(
            x=x_train_adasyn[y_train_adasyn == 0, 0],
            y=x_train_adasyn[y_train_adasyn == 0, 1],
            mode="markers",
            marker=dict(color="#3B82F6", size=5, opacity=0.4),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x_train[y_train == 1, 0],
            y=x_train[y_train == 1, 1],
            mode="markers",
            marker=dict(color="#DC2626", size=7, symbol="diamond"),
            showlegend=False,
        ),
        row=1,
        col=3,
    )
    fig.add_trace(
        go.Scatter(
            x=x_train_adasyn[n_total_train:, 0],
            y=x_train_adasyn[n_total_train:, 1],
            mode="markers",
            marker=dict(color="#8B5CF6", size=6, opacity=0.75, symbol="cross"),
            name="ADASYN Synthesized",
        ),
        row=1,
        col=3,
    )

    fig.update_xaxes(title_text="Feature 1")
    fig.update_yaxes(title_text="Feature 2")

    fig.update_layout(
        template="plotly_white",
        height=480,
        margin=dict(l=40, r=40, t=60, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=-0.22, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return fig, n_orig_min, n_total_train, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The tripartite comparison below reveals how different over-sampling techniques transform the training feature topology:

                1. **Panel (a) Original Imbalance**: The 31 training minority samples (red diamonds) are sparsely scattered, leaving large empty voids across their crescent manifold.
                2. **Panel (b) SMOTE**: Uniformly interpolates along nearest-neighbor vectors (green dots), filling out the convex hull across the entire minority crescent.
                3. **Panel (c) ADASYN**: Concentrates synthesized instances (purple crosses) in high-density pockets along the overlapping boundary where minority points are surrounded by majority points, while generating fewer instances in clear minority regions.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    NearestNeighbors,
    RandomForestClassifier,
    balanced_accuracy_score,
    f1_score,
    mo,
    np,
    pd,
    precision_score,
    recall_score,
    x_test,
    x_train,
    x_train_adasyn,
    x_train_ros,
    x_train_smote,
    y_test,
    y_train,
    y_train_adasyn,
    y_train_ros,
    y_train_smote,
):
    # Example 1: Pure NumPy Implementation of SMOTE
    def manual_smote(x_min_arr, n_samples_to_create, k_nn=5):
        nn = NearestNeighbors(n_neighbors=k_nn + 1).fit(x_min_arr)
        indices = nn.kneighbors(x_min_arr, return_distance=False)
        synthetic_samples = []
        for _ in range(n_samples_to_create):
            # Select random base sample
            idx_base = np.random.randint(0, len(x_min_arr))
            # Pick a random neighbor (excluding the point itself)
            idx_neighbor = np.random.choice(indices[idx_base, 1:])
            x_base = x_min_arr[idx_base]
            x_neigh = x_min_arr[idx_neighbor]
            # Convex combination
            lam = np.random.uniform(0.0, 1.0)
            x_syn = x_base + lam * (x_neigh - x_base)
            synthetic_samples.append(x_syn)
        return np.array(synthetic_samples)

    minority_train = x_train[y_train == 1]
    np.random.seed(42)
    sample_syn_pts = manual_smote(minority_train, n_samples_to_create=5, k_nn=4)

    df_smote_manual = pd.DataFrame(
        {
            "Synthetic_ID": [f"Syn_{i+1}" for i in range(5)],
            "Feature_1": sample_syn_pts[:, 0].round(4),
            "Feature_2": sample_syn_pts[:, 1].round(4),
            "Interpolation_Method": "x_base + lambda * (x_neighbor - x_base)",
        }
    )

    # Example 2: Borderline-SMOTE Sample Categorization from Scratch
    nn_all = NearestNeighbors(n_neighbors=6).fit(x_train)
    _, all_indices = nn_all.kneighbors(minority_train)
    # Check labels of the 5 nearest neighbors (excluding self at column 0)
    neighbor_labels = y_train[all_indices[:, 1:]]
    maj_neighbor_counts = np.sum(neighbor_labels == 0, axis=1)

    categories = []
    for count in maj_neighbor_counts:
        if count == 5:
            categories.append("NOISE (Ignored)")
        elif count >= 3:
            categories.append("DANGER (Boundary)")
        else:
            categories.append("SAFE (Interior)")

    df_borderline = pd.DataFrame(
        {
            "Minority_Sample_ID": [f"Min_{i+1}" for i in range(len(minority_train[:8]))],
            "Feature_1": minority_train[:8, 0].round(3),
            "Feature_2": minority_train[:8, 1].round(3),
            "Majority_Neighbors_k5": maj_neighbor_counts[:8],
            "Category": categories[:8],
        }
    )

    # Example 3: Comprehensive Model Evaluation Benchmark on Unseen Test Partition
    strategies = [
        ("No Resampling (Imbalanced)", x_train, y_train),
        ("Random Oversampling (Naive)", x_train_ros, y_train_ros),
        ("SMOTE (Uniform Convex Hull)", x_train_smote, y_train_smote),
        ("ADASYN (Adaptive Boundary)", x_train_adasyn, y_train_adasyn),
    ]

    benchmark_records = []
    for name, xtr, ytr in strategies:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(xtr, ytr)
        ypred = rf.predict(x_test)

        bal_acc = balanced_accuracy_score(y_test, ypred)
        prec = precision_score(y_test, ypred, zero_division=0)
        rec = recall_score(y_test, ypred, zero_division=0)
        f1 = f1_score(y_test, ypred, zero_division=0)

        benchmark_records.append(
            {
                "Resampling_Strategy": name,
                "Training_Set_Size": len(xtr),
                "Balanced_Accuracy": f"{bal_acc * 100:.2f}%",
                "Minority_Precision": f"{prec * 100:.2f}%",
                "Minority_Recall": f"{rec * 100:.2f}%",
                "Minority_F1_Score": f"{f1 * 100:.2f}%",
            }
        )

    df_benchmark = pd.DataFrame(benchmark_records)

    table_manual = mo.ui.table(df_smote_manual)
    table_border = mo.ui.table(df_borderline)
    table_bench = mo.ui.table(df_benchmark)

    return (
        all_indices,
        benchmark_records,
        categories,
        count,
        df_benchmark,
        df_borderline,
        df_smote_manual,
        maj_neighbor_counts,
        manual_smote,
        minority_train,
        name,
        neighbor_labels,
        nn_all,
        rf,
        sample_syn_pts,
        strategies,
        table_bench,
        table_border,
        table_manual,
        xtr,
        ypred,
        ytr,
    )


@app.cell
def _(mo, table_bench, table_border, table_manual):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy SMOTE Synthesis

                Synthesizing points along nearest-neighbor chords in pure NumPy:
                """
            ),
            table_manual,
            mo.md(
                r"""
                ### Example 2: Borderline-SMOTE Sample Categorization

                Evaluating local neighborhoods to partition minority instances into SAFE, DANGER, and NOISE subsets:
                """
            ),
            table_border,
            mo.md(
                r"""
                ### Example 3: Resampling Strategy Performance on Unseen Test Set

                Comparing generalization metrics (Balanced Accuracy, Precision, Recall, F1) across strategies on a fixed unseen test partition:
                """
            ),
            table_bench,
        ]
    )


if __name__ == "__main__":
    app.run()
