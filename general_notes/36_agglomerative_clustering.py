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
    ### **Agglomerative Clustering Notes**

    Agglomerative Clustering is a bottom-up hierarchical clustering method. It begins with each data point as its own cluster and iteratively merges the closest clusters based on a distance metric. The process continues until all points are grouped into a single cluster or the desired number of clusters is achieved.

    ---

    ### **Agglomerative Clustering Algorithm**:

    1. **Initialization**:
       $$ \forall i, \ C_i = \{x_i\}, \ i = 1, 2, \dots, n $$
       _Each data point is initially its own cluster._

    2. **Distance Calculation**:
       $$ D(C_i, C_j) $$
       _The distance between two clusters $ C_i $ and $ C_j $ is calculated using a chosen distance metric (usually Euclidean)._

    3. **Choose Linkage**:
       - **Single Linkage**
       - **Complete Linkage**
       - **Average Linkage**
       - **Ward’s Linkage**

    4. **Cluster Merging**:
       $$ (i, j) = \arg \min_{i \neq j} D(C_i, C_j) $$
       $$ C_{\text{new}} = C_i \cup C_j $$
       _The two clusters with the smallest distance are merged._

    5. **Repeat**:
       - Continue merging clusters until stopping criterion is met, which can be:
         - Achieving the desired number of clusters
         - Merging all points into one cluster

    ---

    ### **Types of Linkages**:

    1. **Single Linkage**:
       $$ D_{\text{single}}(C_i, C_j) = \min_{p \in C_i, q \in C_j} \| p - q \|_2 $$
       _The distance between two clusters is the minimum pairwise distance between their points._

    2. **Complete Linkage**:
       $$ D_{\text{complete}}(C_i, C_j) = \max_{p \in C_i, q \in C_j} \| p - q \|_2 $$
       _The distance between two clusters is the maximum pairwise distance between their points._

    3. **Average Linkage**:
       $$ D_{\text{average}}(C_i, C_j) = \frac{1}{|C_i||C_j|} \sum_{p \in C_i, q \in C_j} \| p - q \|_2 $$
       _The distance between two clusters is the average of the pairwise distances between their points._

    4. **Ward’s Linkage**:
       $$ D_{\text{Ward}}(C_i, C_j) = \sqrt{\frac{|C_i| \cdot |C_j|}{|C_i| + |C_j|} \| \mu_{C_i} - \mu_{C_j} \|_2^2} $$
       _The distance between two clusters is based on the squared Euclidean distance between their centroids, weighted by their sizes._

    ---

    ### **Advantages and Disadvantages**:

    1. **Single Linkage**:
       - **Advantages**:
         - Suitable for irregular-shaped clusters.
       - **Disadvantages**:
         - Sensitive to outliers; may lead to "chaining" of distant clusters.

    2. **Complete Linkage**:
       - **Advantages**:
         - Produces compact, spherical clusters.
       - **Disadvantages**:
         - Sensitive to outliers and may split large clusters.

    3. **Average Linkage**:
       - **Advantages**:
         - Provides a balanced approach to clustering.
       - **Disadvantages**:
         - Less sensitive to irregular-shaped clusters.

    4. **Ward’s Linkage**:
       - **Advantages**:
         - Minimizes within-cluster variance, resulting in more compact clusters.
       - **Disadvantages**:
         - Sensitive to large cluster sizes and may not work well with non-spherical clusters.
    """)
    return


@app.cell
def _():
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.cluster.hierarchy import dendrogram, linkage
    from sklearn.cluster import AgglomerativeClustering, KMeans
    from sklearn.datasets import make_blobs

    np.random.seed(47)
    plt.style.use("dark_background")

    # Generate synthetic data
    X, y_true = make_blobs(n_samples=1000, centers=4, cluster_std=1.0, random_state=47)
    return AgglomerativeClustering, X, dendrogram, linkage, np, plt, y_true


@app.cell
def _(X, plt, y_true):
    # Plot original clusters
    plt.figure(figsize=(6, 5), dpi=300)
    plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap="coolwarm_r", s=30)
    plt.title("Original Clusters (True Labels)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()
    return


@app.cell
def _(AgglomerativeClustering, X, plt):
    # Define linkage types
    linkages = ["ward", "complete", "average", "single"]
    cluster_results = {}
    plt.figure(figsize=(12, 10), dpi=300)
    # Perform Agglomerative Clustering
    for _i, _linkage_type in enumerate(linkages):
        clustering = AgglomerativeClustering(n_clusters=4, linkage=_linkage_type)
        cluster_labels = clustering.fit_predict(X)  # Perform hierarchical clustering
        cluster_results[_linkage_type] = cluster_labels
        plt.subplot(2, 2, _i + 1)
        plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis", s=30)
        plt.title(f"Agglomerative Clustering - {_linkage_type.capitalize()} Linkage")
        plt.xlabel("Feature 1")  # Plot clustering results
        plt.ylabel("Feature 2")
    plt.tight_layout()
    plt.savefig("clustering_methods.png", dpi=300)
    plt.show()
    return (linkages,)


@app.cell
def _(X, dendrogram, linkage, linkages, plt):
    # Plot dendrograms for hierarchical clustering
    plt.figure(figsize=(12, 10), dpi=300)
    for _i, _linkage_type in enumerate(linkages):
        linkage_matrix = linkage(X, method=_linkage_type)  # Compute linkage matrix
        plt.subplot(2, 2, _i + 1)
        dendrogram(linkage_matrix, truncate_mode="level", p=10, no_labels=True)
        plt.title(f"Dendrogram - {_linkage_type.capitalize()} Linkage")  # Plot dendrogram
        plt.xlabel("Cluster Size")
        plt.ylabel("Distance")
    plt.tight_layout()
    plt.savefig("dendrograms.png")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ### Spectral Clustering

    Spectral clustering groups points by **connectivity** rather than by convex shape. It builds a similarity graph, takes the spectral decomposition of the graph Laplacian, and clusters in that eigenspace. That is why it recovers the two moons, where $k$-means and compact-linkage agglomerative methods typically fail.

    The graph Laplacian of the similarity matrix $W$ is $L = D - W$ (unnormalized) or $L_{\text{sym}} = I - D^{-1/2} W D^{-1/2}$ (normalized), with $D_{ii} = \sum_j W_{ij}$. The first $k$ eigenvectors of $L$ are the embedding that $k$-means then partitions.

    **IQ:** Why can spectral clustering separate the two moons when Euclidean centroid methods cannot?
    """)
    return


@app.cell
def _(np, plt):
    from matplotlib.colors import ListedColormap
    from sklearn.cluster import SpectralClustering
    from sklearn.datasets import make_moons

    X_sc, y_sc = make_moons(n_samples=200, noise=0.1, random_state=47)
    spectral = SpectralClustering(
        n_clusters=2, affinity="rbf", n_neighbors=100, assign_labels="kmeans", random_state=47
    )
    y_sc_pred = spectral.fit_predict(X_sc)
    if np.mean(y_sc == y_sc_pred) < 0.5:
        y_sc_pred = 1 - y_sc_pred
    fig, axes = plt.subplots(1, 3, figsize=(12, 6), dpi=300)
    cmap = ListedColormap(["red", "blue"])
    axes[0].scatter(X_sc[:, 0], X_sc[:, 1], c="magenta")
    axes[0].set_title("Original Data")
    axes[1].scatter(X_sc[:, 0], X_sc[:, 1], c=y_sc_pred, cmap=cmap)
    axes[1].set_title("Spectral Clustering")
    axes[2].scatter(X_sc[:, 0], X_sc[:, 1], c=y_sc, cmap=cmap)
    # Cluster IDs are arbitrary; flip them if they are inverted relative to the true labels.
    axes[2].set_title("True Clusters")
    for ax in axes:
        ax.set_xlabel("Feature-1")
        ax.set_ylabel("Feature-2")
    plt.tight_layout()
    plt.show()
    print(f"Clustering accuracy: {np.mean(y_sc == y_sc_pred):.3f}")
    for _i in range(2):
        print(f"Cluster {_i} size: {np.sum(y_sc_pred == _i)}")
    return


if __name__ == "__main__":
    app.run()
