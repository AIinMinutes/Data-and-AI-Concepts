import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import jenkspy
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    return go, jenkspy, make_subplots, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 36 Agglomerative Clustering](36_agglomerative_clustering.py) | [Index](../index.html) | [38 Oversampling →](38_oversampling.py)

        # Jenks Natural Breaks: Fisher-Jenks 1D Optimal Partitioning and Goodness of Variance Fit

        ## [a] Why do you need to know these concepts?

        Continuous variables frequently require discretization into discrete intervals for thematic choropleth mapping, credit risk scorecards, medical diagnostic tiers, and decision rule induction. In practice, practitioners often reach for naive heuristics such as:
        - **Equal Interval Discretization**: Divides the range $[x_{\min}, x_{\max}]$ into bins of equal width. When data is skewed, heavy-tailed, or multimodal, this results in multiple nearly empty bins and clumps the vast majority of observations into a single bin.
        - **Quantile (Equal Frequency) Discretization**: Places an equal count of observations into every bin. While it guarantees populated classes, it artificially fractures tight natural clusters across arbitrary boundaries and groups wildly distant values together in sparse tail regions.

        #### The Fisher-Jenks Global Optimization
        Formulated by cartographer George Jenks and statistician Walter D. Fisher, **Jenks Natural Breaks** treats discretization as a formal one-dimensional variance minimization problem. It identifies class boundaries that simultaneously minimize variance within each class while maximizing variance between classes.

        #### Global Optimality via 1D Dynamic Programming
        In multi-dimensional spaces, finding variance-minimizing partitions ($k$-means) is known to be NP-hard. Multi-dimensional heuristic algorithms can easily become trapped in poor local optima. In one dimension, however, sorting the data imposes an exact ordering constraint: optimal clusters must be contiguous slices of the sorted array. By leveraging this ordering property, the Fisher-Jenks algorithm employs **Dynamic Programming** to find the provably **global minimum-variance partition** in polynomial time.

        #### Measuring Partition Quality: Goodness of Variance Fit (GVF)
        Just as $R^2$ measures explained variance in linear regression, the **Goodness of Variance Fit (GVF)** measures the fraction of total variance explained by the classification. Plotting GVF across different values of $k$ provides an exact mathematical elbow criterion to choose the optimal number of bins.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Dynamic Programming Mechanics

        ### 1. Variance Decomposition: SDAM and SDCM

        Let a sorted one-dimensional dataset be denoted by:

        $$x_1 \le x_2 \le \dots \le x_n$$

        The overall sample mean is $\bar{x} = \frac{1}{n} \sum_{i=1}^n x_i$. The **Sum of Squared Deviations from the Array Mean (SDAM)** represents the total baseline variance:

        $$\text{SDAM} = \sum_{i=1}^n (x_i - \bar{x})^2$$

        Suppose the array is partitioned into $k$ contiguous classes $C_1, C_2, \dots, C_k$, defined by partition boundary indices $0 = s_0 < s_1 < s_2 < \dots < s_k = n$, such that class $C_j$ contains elements $\{x_{s_{j-1}+1}, \dots, x_{s_j}\}$.

        The mean of class $C_j$ is:

        $$\bar{x}_j = \frac{1}{|C_j|} \sum_{i \in C_j} x_i$$

        The **Sum of Squared Deviations from Class Means (SDCM)** (or Within-Class Sum of Squares) is:

        $$\text{SDCM} = \sum_{j=1}^k \sum_{i \in C_j} (x_i - \bar{x}_j)^2$$

        By the Law of Total Variance, total sum of squares decomposes into within-class and between-class components:

        $$\text{SDAM} = \text{SDCM} + \text{SSB}, \quad \text{where } \text{SSB} = \sum_{j=1}^k |C_j| (\bar{x}_j - \bar{x})^2$$

        Minimizing SDCM is strictly equivalent to maximizing between-class variance SSB.

        ### 2. Goodness of Variance Fit (GVF)

        The Goodness of Variance Fit measures the proportion of variance explained by the $k$ classes:

        $$\text{GVF} = \frac{\text{SDAM} - \text{SDCM}}{\text{SDAM}} = 1 - \frac{\text{SDCM}}{\text{SDAM}}$$

        Properties of GVF:
        - $\text{GVF} \in [0, 1]$.
        - $\text{GVF} = 0$ when $k = 1$ (all points in a single class, $\text{SDCM} = \text{SDAM}$).
        - $\text{GVF} = 1$ when $k = n$ (each point is its own class, $\text{SDCM} = 0$).
        - The optimal number of classes $k^*$ is typically selected at the elbow of the GVF curve, where marginal increase drops below a desired threshold (often targeting $\text{GVF} \ge 0.85$ or $0.90$).

        ### 3. The Fisher Dynamic Programming Recurrence

        Let $D(i, j)$ denote the sum of squared deviations of the subarray $x[i \dots j]$ from its local mean:

        $$D(i, j) = \sum_{m=i}^j \left(x_m - \bar{x}_{i:j}\right)^2 = \sum_{m=i}^j x_m^2 - \frac{1}{j - i + 1} \left( \sum_{m=i}^j x_m \right)^2$$

        Let $V(c, m)$ be the minimal SDCM achievable by partitioning the prefix $x[1 \dots m]$ into $c$ classes.

        **Base Case ($c = 1$)**:
        $$V(1, m) = D(1, m) \quad \text{for } 1 \le m \le n$$

        **Recurrence Relation ($c \ge 2$)**:
        $$V(c, m) = \min_{c-1 \le p < m} \left\{ V(c - 1, p) + D(p + 1, m) \right\}$$

        Here, $p$ represents the candidate boundary index where the $(c-1)$-th class ends, and the $c$-th class spans $x[p+1 \dots m]$. By recording the optimal split index $P^*(c, m) = \arg\min_p \{\dots\}$, the globally optimal class boundaries are reconstructed via standard backward backtracking from $P^*(k, n)$.
        """
    )
    return


@app.cell
def _(jenkspy, np):
    np.random.seed(42)

    # Generate synthetic multimodal distribution simulating real-world vulnerability / risk scores
    # Three distinct natural clusters with unequal densities
    c1 = np.random.normal(loc=2.2, scale=0.45, size=150)
    c2 = np.random.normal(loc=5.8, scale=0.60, size=220)
    c3 = np.random.normal(loc=9.4, scale=0.50, size=130)

    raw_scores = np.concatenate([c1, c2, c3])
    raw_scores = np.sort(raw_scores[raw_scores > 0])
    n_pts = len(raw_scores)

    # Target number of classes
    k_classes = 3

    # 1. Jenks Natural Breaks
    jenks_cutoffs = np.array(jenkspy.jenks_breaks(raw_scores, n_classes=k_classes))

    # 2. Equal Interval Breaks
    min_v, max_v = raw_scores.min(), raw_scores.max()
    equal_interval_cutoffs = np.linspace(min_v, max_v, k_classes + 1)

    # 3. Quantile (Equal Count) Breaks
    quantile_probs = np.linspace(0.0, 1.0, k_classes + 1)
    quantile_cutoffs = np.quantile(raw_scores, quantile_probs)

    # Compute GVF across k from 1 to 7 for the elbow scree plot
    gvf_values = []
    sdam = np.sum((raw_scores - np.mean(raw_scores)) ** 2)

    for k in range(1, 8):
        if k == 1:
            gvf_values.append(0.0)
        else:
            b = jenkspy.jenks_breaks(raw_scores, n_classes=k)
            # Compute SDCM
            sdcm_k = 0.0
            for j in range(k):
                sub = raw_scores[(raw_scores >= b[j]) & (raw_scores <= b[j + 1])]
                if len(sub) > 0:
                    sdcm_k += np.sum((sub - np.mean(sub)) ** 2)
            gvf_k = 1.0 - (sdcm_k / sdam)
            gvf_values.append(max(0.0, min(1.0, gvf_k)))

    return (
        c1,
        c2,
        c3,
        equal_interval_cutoffs,
        gvf_values,
        jenks_cutoffs,
        k_classes,
        max_v,
        min_v,
        n_pts,
        quantile_cutoffs,
        quantile_probs,
        raw_scores,
        sdam,
    )


@app.cell
def _(
    equal_interval_cutoffs,
    go,
    gvf_values,
    jenks_cutoffs,
    make_subplots,
    mo,
    np,
    quantile_cutoffs,
    raw_scores,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Partition Boundaries on Multimodal Distribution</b>",
            "<b>Goodness of Variance Fit (GVF) vs Number of Classes k</b>",
        ],
        horizontal_spacing=0.12,
    )

    # Left: Histogram of data with vertical cutoffs
    fig.add_trace(
        go.Histogram(
            x=raw_scores,
            nbinsx=45,
            marker_color="#94A3B8",
            opacity=0.65,
            name="Observed Distribution",
            showlegend=True,
        ),
        row=1,
        col=1,
    )

    # Add Jenks breaks (excluding min and max endpoints)
    for idx, j_val in enumerate(jenks_cutoffs[1:-1]):
        fig.add_vline(
            x=j_val,
            line=dict(color="#10B981", width=2.5, dash="solid"),
            annotation_text=f"Jenks {idx+1}: {j_val:.2f}",
            annotation_position="top right",
            row=1,
            col=1,
        )

    # Add Equal Interval breaks
    for idx, e_val in enumerate(equal_interval_cutoffs[1:-1]):
        fig.add_vline(
            x=e_val,
            line=dict(color="#F59E0B", width=2, dash="dash"),
            annotation_text=f"Equal {idx+1}: {e_val:.2f}",
            annotation_position="bottom right",
            row=1,
            col=1,
        )

    # Add Quantile breaks
    for idx, q_val in enumerate(quantile_cutoffs[1:-1]):
        fig.add_vline(
            x=q_val,
            line=dict(color="#3B82F6", width=2, dash="dot"),
            annotation_text=f"Quant {idx+1}: {q_val:.2f}",
            annotation_position="top left",
            row=1,
            col=1,
        )

    # Dummy legend items for the lines
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#10B981", width=2.5),
            name="Jenks Natural Breaks (Valley Aligned)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#F59E0B", width=2, dash="dash"),
            name="Equal Interval (Arbitrary Cuts)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color="#3B82F6", width=2, dash="dot"),
            name="Quantiles (Equal Frequency Cuts)",
        ),
        row=1,
        col=1,
    )

    # Right: GVF curve across k
    k_range = list(range(1, 8))
    fig.add_trace(
        go.Scatter(
            x=k_range,
            y=gvf_values,
            mode="lines+markers",
            line=dict(color="#8B5CF6", width=2.5),
            marker=dict(size=8, color="#6D28D9"),
            name="GVF Curve",
        ),
        row=1,
        col=2,
    )

    # Add threshold reference line at GVF = 0.90
    fig.add_hline(
        y=0.90,
        line=dict(color="#EF4444", width=1.5, dash="dash"),
        annotation_text="90% Variance Explained (Target)",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Continuous Score", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Number of Classes (k)", row=1, col=2)
    fig.update_yaxes(title_text="Goodness of Variance Fit (GVF)", range=[0, 1.05], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return fig, k_range, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates the mechanics of optimal 1D partitioning:

                1. **Left (Valley Alignment vs Heuristic Bins)**: Jenks Natural Breaks (emerald solid) automatically locks onto the natural low-density valleys ($x \approx 3.7$ and $x \approx 7.6$) separating the clusters. In contrast, Equal Interval (orange dashed) cuts directly across the second mode, and Quantiles (blue dotted) shifts boundaries inwards to force equal observation counts.
                2. **Right (Elbow Detection via GVF)**: At $k=1$, GVF is 0. Moving from $k=2$ to $k=3$ creates a steep climb past $0.94$, exactly where the 3 genuine clusters are resolved. Beyond $k=3$, the curve flattens into an elbow of diminishing returns.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    equal_interval_cutoffs,
    jenks_cutoffs,
    jenkspy,
    k_classes,
    mo,
    np,
    pd,
    quantile_cutoffs,
    raw_scores,
    sdam,
):
    # Example 1: Fisher Dynamic Programming from scratch in Pure Python
    def fisher_jenks_dp(arr, n_classes):
        sorted_x = np.sort(arr)
        n = len(sorted_x)

        # Precompute prefix sums and prefix sum of squares for O(1) interval variance
        prefix_sum = np.zeros(n + 1)
        prefix_sq = np.zeros(n + 1)
        for i in range(n):
            prefix_sum[i + 1] = prefix_sum[i] + sorted_x[i]
            prefix_sq[i + 1] = prefix_sq[i] + sorted_x[i] ** 2

        # Fast function to compute D(i, j) = sum_{m=i}^j (x_m - mean)^2 using 0-indexed indices [i, j]
        def get_d(i, j):
            count = j - i + 1
            s = prefix_sum[j + 1] - prefix_sum[i]
            sq = prefix_sq[j + 1] - prefix_sq[i]
            return sq - (s**2) / count

        # DP tables: V[c, m] stores min SDCM for prefix 0..m with c classes (1-indexed classes)
        v = np.zeros((n_classes + 1, n))
        p_table = np.zeros((n_classes + 1, n), dtype=int)

        # Base case c = 1
        for m in range(n):
            v[1, m] = get_d(0, m)

        # Fill DP for c = 2 .. n_classes
        for c in range(2, n_classes + 1):
            for m in range(c - 1, n):
                min_val = float("inf")
                best_p = -1
                for p in range(c - 2, m):
                    cost = v[c - 1, p] + get_d(p + 1, m)
                    if cost < min_val:
                        min_val = cost
                        best_p = p
                v[c, m] = min_val
                p_table[c, m] = best_p

        # Backtrack to find boundaries
        breaks = [sorted_x[-1]]
        curr_m = n - 1
        for c in range(n_classes, 1, -1):
            split_p = p_table[c, curr_m]
            breaks.append(sorted_x[split_p])
            curr_m = split_p
        breaks.append(sorted_x[0])
        breaks.reverse()
        return breaks, v[n_classes, n - 1]

    # Run on a representative subset to demonstrate exact mathematical match
    test_data = raw_scores[::5]
    my_breaks, my_sdcm = fisher_jenks_dp(test_data, 3)
    jenkspy_breaks = jenkspy.jenks_breaks(test_data, n_classes=3)

    df_verification = pd.DataFrame(
        {
            "Class_Boundary": ["Minimum", "Split 1", "Split 2", "Maximum"],
            "From_Scratch_DP": np.round(my_breaks, 4),
            "Jenkspy_Library": np.round(jenkspy_breaks, 4),
            "Absolute_Difference": np.abs(np.array(my_breaks) - np.array(jenkspy_breaks)).round(8),
        }
    )

    # Example 2: Metric comparison between Equal Interval, Quantile, and Jenks
    def compute_metrics(boundaries, data_arr):
        sdcm_val = 0.0
        counts = []
        for j in range(len(boundaries) - 1):
            if j == len(boundaries) - 2:
                in_bin = data_arr[(data_arr >= boundaries[j]) & (data_arr <= boundaries[j + 1])]
            else:
                in_bin = data_arr[(data_arr >= boundaries[j]) & (data_arr < boundaries[j + 1])]
            counts.append(len(in_bin))
            if len(in_bin) > 0:
                sdcm_val += np.sum((in_bin - np.mean(in_bin)) ** 2)
        gvf_val = 1.0 - (sdcm_val / sdam)
        return sdcm_val, gvf_val, counts

    jenks_sdcm, jenks_gvf, jenks_counts = compute_metrics(jenks_cutoffs, raw_scores)
    eq_sdcm, eq_gvf, eq_counts = compute_metrics(equal_interval_cutoffs, raw_scores)
    quant_sdcm, quant_gvf, quant_counts = compute_metrics(quantile_cutoffs, raw_scores)

    df_comparison = pd.DataFrame(
        [
            {
                "Method": "Jenks Natural Breaks (Optimal)",
                "SDCM (Within-Class Variance)": round(jenks_sdcm, 2),
                "GVF (Explained Variance)": f"{jenks_gvf * 100:.2f}%",
                "Class_Distribution": str(jenks_counts),
                "Optimality": "Provably minimal SDCM in 1D",
            },
            {
                "Method": "Quantiles (Equal Frequency)",
                "SDCM (Within-Class Variance)": round(quant_sdcm, 2),
                "GVF (Explained Variance)": f"{quant_gvf * 100:.2f}%",
                "Class_Distribution": str(quant_counts),
                "Optimality": "Forces equal counts, ignores valleys",
            },
            {
                "Method": "Equal Interval (Fixed Width)",
                "SDCM (Within-Class Variance)": round(eq_sdcm, 2),
                "GVF (Explained Variance)": f"{eq_gvf * 100:.2f}%",
                "Class_Distribution": str(eq_counts),
                "Optimality": "Heuristic interval, ignores density",
            },
        ]
    )

    table_verif = mo.ui.table(df_verification)
    table_comp = mo.ui.table(df_comparison)

    return (
        compute_metrics,
        df_comparison,
        df_verification,
        eq_counts,
        eq_gvf,
        eq_sdcm,
        fisher_jenks_dp,
        jenks_counts,
        jenks_gvf,
        jenks_sdcm,
        jenkspy_breaks,
        my_breaks,
        my_sdcm,
        quant_counts,
        quant_gvf,
        quant_sdcm,
        table_comp,
        table_verif,
        test_data,
    )


@app.cell
def _(mo, table_comp, table_verif):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Dynamic Programming Implementation from Scratch

                Validating our from-scratch Fisher-Jenks dynamic programming recurrence against the optimized C-extension `jenkspy`:
                """
            ),
            table_verif,
            mo.md(
                r"""
                ### Example 2: Statistical Comparison Across Discretization Strategies

                Comparing the resulting within-class variance (SDCM) and overall Goodness of Variance Fit (GVF) between the three methods:
                """
            ),
            table_comp,
        ]
    )


if __name__ == "__main__":
    app.run()
