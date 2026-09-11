import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import time
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from scipy.cluster.hierarchy import cophenet, dendrogram, linkage
    from scipy.spatial.distance import pdist
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.datasets import make_blobs, make_moons
    from sklearn.metrics import calinski_harabasz_score, silhouette_score

    return (
        AgglomerativeClustering,
        calinski_harabasz_score,
        cophenet,
        dendrogram,
        go,
        linkage,
        make_blobs,
        make_moons,
        make_subplots,
        mo,
        np,
        pd,
        pdist,
        silhouette_score,
        time,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 35 Gini Impurity vs Entropy](35_gini_impurity_vs_entropy.py) | [Index](../index.html) | [37 Natural Breaks →](37_natural_breaks.py)

        # Agglomerative Hierarchical Clustering: Linkage Criteria, Lance-Williams Recurrence, and Dendrogram Geometry

        ## [a] Why do you need to know these concepts?

        Clustering algorithms partition unlabeled data into cohesive subgroups. While flat partitioning methods (such as $k$-means and Gaussian Mixture Models) require pre-specifying the number of clusters $k$ and assume convex, isotropic cluster geometries, hierarchical agglomerative clustering constructs an entire hierarchy of nested groupings from the bottom up.

        #### The Role of Linkage in Determining Cluster Geometry
        Agglomerative clustering starts with every observation as an individual singleton cluster and sequentially merges the closest pair of clusters. The core distinguishing factor between hierarchical clustering variants is the **linkage criterion**, which defines what distance between two sets of points means:
        - **Single Linkage (Minimum Distance)**: Merges clusters based on their closest pair of points. It can uncover non-convex manifolds and concentric geometries, but is vulnerable to the **chaining effect**, where a single string of noise points merges two distinct clusters.
        - **Complete Linkage (Maximum Distance)**: Merges clusters based on their most distant pair of points. It aggressively resists chaining and produces compact, equal-diameter clusters, but is sensitive to isolated outliers.
        - **Average Linkage (UPGMA)**: Evaluates the average pairwise distance between all member points. It provides a balanced, robust trade-off and is widely used in computational biology and phylogenetics.
        - **Ward's Linkage (Minimum Variance)**: Merges the pair of clusters that minimizes the increase in total within-cluster sum of squared errors. Like $k$-means, Ward's method seeks compact, spherical clusters with approximately equal sizes.

        #### The Lance-Williams Recurrence: $O(1)$ Distance Updates
        Naively recomputing all pairwise inter-cluster distances after every merge requires $O(N^3)$ operations. The **Lance-Williams recurrence formula** enables updating the distance from a newly merged cluster to all other clusters in $O(1)$ time using an exact parametric linear formula. This reduction makes hierarchical clustering computationally tractable for practical dataset sizes.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Algorithmic Mechanics

        ### 1. The General Agglomerative Framework

        Let $\mathcal{D} = \{x_1, x_2, \dots, x_N\} \subset \mathbb{R}^p$ be a collection of $N$ observations.
        1. **Initialization**: Form $N$ singleton clusters $\mathcal{C}_1 = \{x_1\}, \dots, \mathcal{C}_N = \{x_N\}$.
        2. **Iterative Merging**: At step $t$, identify the pair of clusters $(A, B)$ satisfying:

        $$(A, B) = \arg\min_{i \neq j} D(\mathcal{C}_i, \mathcal{C}_j)$$

        3. **Union**: Form the merged cluster $\mathcal{C}_{\text{new}} = A \cup B$ and remove $A$ and $B$ from the active cluster list.
        4. **Distance Update**: Compute the distance $D(\mathcal{C}_{\text{new}}, K)$ for every remaining active cluster $K$.
        5. **Termination**: Repeat until all observations are merged into a single root cluster $\mathcal{C}_{\text{root}} = \mathcal{D}$.

        ### 2. Formulations of Linkage Criteria

        Let $A$ and $B$ be two disjoint non-empty clusters with cardinalities $|A|$ and $|B|$.

        #### Single Linkage (Nearest Neighbor)
        $$D_{\text{single}}(A, B) = \min_{u \in A, v \in B} \|u - v\|_2$$

        #### Complete Linkage (Furthest Neighbor)
        $$D_{\text{complete}}(A, B) = \max_{u \in A, v \in B} \|u - v\|_2$$

        #### Average Linkage (UPGMA)
        $$D_{\text{average}}(A, B) = \frac{1}{|A| |B|} \sum_{u \in A} \sum_{v \in B} \|u - v\|_2$$

        #### Ward's Minimum Variance Linkage
        Let $\mu_A = \frac{1}{|A|}\sum_{u \in A} u$ denote the centroid of cluster $A$. The sum of squared errors within cluster $A$ is:

        $$\text{SSE}(A) = \sum_{u \in A} \|u - \mu_A\|_2^2$$

        Ward's criterion selects the pair of clusters that minimizes the increase in total variance $\Delta \text{SSE}(A, B) = \text{SSE}(A \cup B) - \text{SSE}(A) - \text{SSE}(B)$. Applying Huygens' parallel axis theorem yields:

        $$\Delta \text{SSE}(A, B) = \frac{|A| |B|}{|A| + |B|} \|\mu_A - \mu_B\|_2^2$$

        Ward's distance metric is defined as $D_{\text{Ward}}(A, B) = \sqrt{2 \Delta \text{SSE}(A, B)}$.

        ### 3. The Lance-Williams Recurrence Formula

        When clusters $A$ and $B$ are merged into $A \cup B$, the distance between $A \cup B$ and any external cluster $K$ can be computed from known pairwise distances $D(A, K)$, $D(B, K)$, and $D(A, B)$ using:

        $$D(A \cup B, K) = \alpha_A D(A, K) + \alpha_B D(B, K) + \beta D(A, B) + \gamma |D(A, K) - D(B, K)|$$

        The parameters $\alpha_A, \alpha_B, \beta, \gamma$ for the four primary linkages are:

        - **Single**: $\alpha_A = \frac{1}{2}, \alpha_B = \frac{1}{2}, \beta = 0, \gamma = -\frac{1}{2}$
        - **Complete**: $\alpha_A = \frac{1}{2}, \alpha_B = \frac{1}{2}, \beta = 0, \gamma = \frac{1}{2}$
        - **Average**: $\alpha_A = \frac{|A|}{|A| + |B|}, \alpha_B = \frac{|B|}{|A| + |B|}, \beta = 0, \gamma = 0$
        - **Ward**: $\alpha_A = \frac{|A| + |K|}{|A| + |B| + |K|}, \alpha_B = \frac{|B| + |K|}{|A| + |B| + |K|}, \beta = -\frac{|K|}{|A| + |B| + |K|}, \gamma = 0$

        ### 4. Cophenetic Correlation Coefficient

        The dendrogram represents hierarchical tree distances. The **cophenetic distance** $t_{ij}$ between observations $x_i$ and $x_j$ is defined as the height in the dendrogram where clusters containing $x_i$ and $x_j$ first merge. The cophenetic correlation coefficient $c$ measures the Pearson correlation between true Euclidean distances $d_{ij} = \|x_i - x_j\|_2$ and dendrogram cophenetic distances $t_{ij}$:

        $$c = \frac{\sum_{i < j} (d_{ij} - \bar{d})(t_{ij} - \bar{t})}{\sqrt{\sum_{i < j} (d_{ij} - \bar{d})^2 \sum_{i < j} (t_{ij} - \bar{t})^2}}$$

        A cophenetic correlation closer to $1.0$ indicates that the dendrogram faithfully preserves original metric distances.
        """
    )
    return


@app.cell
def _(make_blobs, make_moons, np):
    np.random.seed(42)

    # 1. Dataset with bridging noise to illustrate Single Linkage chaining vs Ward/Complete
    blob_pts, _ = make_blobs(
        n_samples=[80, 80],
        centers=[[-3.0, 0.0], [3.0, 0.0]],
        cluster_std=0.7,
        random_state=42,
    )
    # Bridge of 6 noise points connecting the two blobs
    bridge_x = np.linspace(-1.5, 1.5, 6)
    bridge_y = np.random.normal(0.0, 0.1, 6)
    bridge_pts = np.column_stack([bridge_x, bridge_y])
    chaining_data = np.vstack([blob_pts, bridge_pts])

    # 2. Non-convex Two Moons dataset
    moons_data, _ = make_moons(n_samples=160, noise=0.08, random_state=42)

    return (
        blob_pts,
        bridge_pts,
        bridge_x,
        bridge_y,
        chaining_data,
        moons_data,
    )


@app.cell
def _(
    AgglomerativeClustering,
    chaining_data,
    go,
    make_subplots,
    mo,
    moons_data,
):
    # Fit Single, Complete, Average, and Ward on Chaining Data
    cluster_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(
        chaining_data
    )
    cluster_single = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(
        chaining_data
    )

    # Fit Single and Ward on Two Moons Data
    moon_single = AgglomerativeClustering(n_clusters=2, linkage="single").fit_predict(moons_data)
    moon_ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit_predict(moons_data)

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            "<b>Ward Linkage on Chaining Data (Compact Partitions)</b>",
            "<b>Single Linkage on Chaining Data (Chaining Vulnerability)</b>",
            "<b>Ward Linkage on Two Moons (Fails Non-Convex Shape)</b>",
            "<b>Single Linkage on Two Moons (Recovers Non-Convex Shape)</b>",
        ],
        horizontal_spacing=0.10,
        vertical_spacing=0.14,
    )

    color_map = {0: "#2563EB", 1: "#DC2626"}

    # Top-Left: Ward on Chaining
    fig.add_trace(
        go.Scatter(
            x=chaining_data[:, 0],
            y=chaining_data[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[c] for c in cluster_ward],
                size=7,
                opacity=0.85,
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Top-Right: Single on Chaining
    fig.add_trace(
        go.Scatter(
            x=chaining_data[:, 0],
            y=chaining_data[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[c] for c in cluster_single],
                size=7,
                opacity=0.85,
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Bottom-Left: Ward on Moons
    fig.add_trace(
        go.Scatter(
            x=moons_data[:, 0],
            y=moons_data[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[c] for c in moon_ward],
                size=7,
                opacity=0.85,
            ),
            showlegend=False,
        ),
        row=2,
        col=1,
    )

    # Bottom-Right: Single on Moons
    fig.add_trace(
        go.Scatter(
            x=moons_data[:, 0],
            y=moons_data[:, 1],
            mode="markers",
            marker=dict(
                color=[color_map[c] for c in moon_single],
                size=7,
                opacity=0.85,
            ),
            showlegend=False,
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        template="plotly_white",
        height=620,
        margin=dict(l=40, r=40, t=60, b=40),
    )

    viz = mo.ui.plotly(fig)
    return (
        cluster_single,
        cluster_ward,
        color_map,
        fig,
        moon_single,
        moon_ward,
        viz,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The multi-panel visualization below contrasts the geometric behaviors of Ward's minimum variance linkage versus Single linkage:

                1. **Top Row (Chaining Sensitivity)**: On two isolated Gaussian blobs connected by a thin sparse line of noise, Ward (left) resists the noise bridge and splits the two clusters evenly. Single linkage (right) suffers from the chaining effect: the nearest-neighbor logic absorbs all bridge points into a single giant cluster, isolating a singleton or small edge point as cluster 2.
                2. **Bottom Row (Non-Convex Manifolds)**: On the interlocking two moons dataset, Ward's method (left) enforces convex partitions and cuts across the crescent shapes. Single linkage (right) traces continuous nearest-neighbor proximity, perfectly recovering the non-convex crescent manifolds.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    AgglomerativeClustering,
    calinski_harabasz_score,
    chaining_data,
    cophenet,
    linkage,
    mo,
    np,
    pd,
    pdist,
    silhouette_score,
    time,
):
    # Example 1: Verification of Lance-Williams Recurrence against Exact Pairwise Computation
    pts_A = np.array([[0.0, 0.0], [0.0, 1.0]])
    pts_B = np.array([[1.0, 0.0], [1.0, 1.0]])
    pts_K = np.array([[3.0, 0.0], [3.0, 2.0], [4.0, 1.0]])

    pts_AB = np.vstack([pts_A, pts_B])

    # Direct average distance
    direct_avg_dist = np.mean(
        [np.linalg.norm(u - v) for u in pts_AB for v in pts_K]
    )
    dist_AK = np.mean([np.linalg.norm(u - v) for u in pts_A for v in pts_K])
    dist_BK = np.mean([np.linalg.norm(u - v) for u in pts_B for v in pts_K])

    # Lance-Williams for Average Linkage: (|A| * d(A,K) + |B| * d(B,K)) / (|A| + |B|)
    lw_avg_dist = (len(pts_A) * dist_AK + len(pts_B) * dist_BK) / (len(pts_A) + len(pts_B))

    df_lw_verif = pd.DataFrame(
        [
            {
                "Linkage_Criterion": "Average (UPGMA)",
                "Direct_Pairwise_Calculation": round(direct_avg_dist, 6),
                "Lance_Williams_Recurrence": round(lw_avg_dist, 6),
                "Difference": round(abs(direct_avg_dist - lw_avg_dist), 10),
                "Update_Speedup": "O(1) vs O(|A||B||K|)",
            }
        ]
    )

    # Example 2: Cophenetic Correlation Analysis across Linkages
    pairwise_distances = pdist(chaining_data)
    linkage_types = ["ward", "complete", "average", "single"]
    cophenet_records = []

    for l_type in linkage_types:
        z_matrix = linkage(chaining_data, method=l_type)
        c_score, _ = cophenet(z_matrix, pairwise_distances)
        cophenet_records.append(
            {
                "Linkage": l_type.capitalize(),
                "Cophenetic_Correlation": round(c_score, 4),
                "Preservation_Quality": (
                    "High (Faithfully retains pairwise metric distances)"
                    if c_score > 0.8
                    else "Moderate (Imposes strong geometric deformation)"
                ),
            }
        )

    df_cophenet = pd.DataFrame(cophenet_records)

    # Example 3: Comparative Performance Benchmark across Linkages
    benchmark_records = []
    n_benchmark_runs = 50

    for l_type in linkage_types:
        t0 = time.perf_counter()
        for _ in range(n_benchmark_runs):
            model = AgglomerativeClustering(n_clusters=2, linkage=l_type)
            labels = model.fit_predict(chaining_data)
        elapsed_ms = (time.perf_counter() - t0) * 1000 / n_benchmark_runs

        # Evaluate quality metrics
        if len(np.unique(labels)) > 1:
            sil = silhouette_score(chaining_data, labels)
            ch = calinski_harabasz_score(chaining_data, labels)
        else:
            sil, ch = 0.0, 0.0

        benchmark_records.append(
            {
                "Linkage": l_type.capitalize(),
                "Execution_Time_ms": round(elapsed_ms, 3),
                "Silhouette_Score": round(sil, 4),
                "Calinski_Harabasz_Index": round(ch, 2),
                "Cluster_Sizes": str(list(np.bincount(labels))),
            }
        )

    df_benchmark = pd.DataFrame(benchmark_records)

    table_lw = mo.ui.table(df_lw_verif)
    table_coph = mo.ui.table(df_cophenet)
    table_bench = mo.ui.table(df_benchmark)

    return (
        benchmark_records,
        c_score,
        cophenet_records,
        df_benchmark,
        df_cophenet,
        df_lw_verif,
        direct_avg_dist,
        dist_AK,
        dist_BK,
        elapsed_ms,
        l_type,
        labels,
        linkage_types,
        lw_avg_dist,
        model,
        n_benchmark_runs,
        pairwise_distances,
        pts_A,
        pts_AB,
        pts_B,
        pts_K,
        table_bench,
        table_coph,
        table_lw,
        z_matrix,
    )


@app.cell
def _(mo, table_bench, table_coph, table_lw):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Lance-Williams Recurrence Mathematical Verification

                Validating that the $O(1)$ parametric recurrence formula matches exhaustive pairwise distance computation down to exact floating-point precision:
                """
            ),
            table_lw,
            mo.md(
                r"""
                ### Example 2: Cophenetic Correlation Coefficient Evaluation

                Measuring how faithfully each linkage preserves original high-dimensional metric distances in its hierarchical tree representation:
                """
            ),
            table_coph,
            mo.md(
                r"""
                ### Example 3: Comparative Clustering Benchmark

                Quantitative evaluation of execution runtime, silhouette separation, and resulting cluster sample balance on the bridge-contaminated dataset:
                """
            ),
            table_bench,
        ]
    )


if __name__ == "__main__":
    app.run()
