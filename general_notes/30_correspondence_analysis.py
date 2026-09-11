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
    import prince

    return go, make_subplots, mo, np, pd, prince


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Note 30: Correspondence Analysis, Contingency Tables, and the Chi-Square Geometry

    &larr; Previous Note: [29 Canonical Correlation Analysis](29_canonical_correlation_analysis.py) | Next Note: [31 Gaussian Mixture Models](31_gaussian_mixture_models.py) &rarr;

    ---

    ## [a] Why do you need to know these concepts?

    Principal Component Analysis (PCA) assumes continuous numerical variables endowed with standard Euclidean geometry. However, tabular data in market research, natural language processing, sports analytics, and survey science often consists of contingency tables (cross-tabulated non-negative counts or frequency matrices).

    Applying ordinary Euclidean distance to frequency tables produces severe distortions: categories with large absolute counts dominate the projection, while rare but highly diagnostic categories are crushed.

    **Correspondence Analysis (CA)**, formulated by Jean-Paul Benzécri (1973), is the foundational dimensionality reduction method for categorical contingency tables:
    1. **The Chi-Square ($\chi^2$) Metric**: CA replaces Euclidean distance with the $\chi^2$-distance between relative frequency profiles. By weighting each column coordinate by the inverse of its marginal frequency ($1 / c_j$), rare features are properly scaled, satisfying the "principle of distributional equivalence".
    2. **Decomposition of Total Inertia**: The total variance in a contingency table is quantified by the total inertia $\phi^2 = \frac{\chi^2}{n}$, measuring the global departure from statistical independence. CA uses the Generalized Singular Value Decomposition (GSVD) to decompose this total $\chi^2$ into orthogonal principal components.
    3. **Dual Symmetric Biplots**: CA projects both row categories (e.g. Football Clubs or Brands) and column categories (e.g. Match Performance Metrics or Consumer Traits) into a shared low-dimensional map. Proximity between a row and a column indicates positive association (over-representation relative to independence).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [b] Concept explanation with their role in ML/AI/Stats?

    ### 1. The Correspondence Matrix and Marginals

    Let $\mathbf{N} \in \mathbb{R}^{I \times J}$ be an $I \times J$ contingency table of non-negative counts, with grand total $n = \sum_{i=1}^I \sum_{j=1}^J N_{ij} = \mathbf{1}_I^\top \mathbf{N} \mathbf{1}_J$.

    The **correspondence matrix** $\mathbf{P} \in \mathbb{R}^{I \times J}$ represents the empirical joint probability distribution:

    $$
    \mathbf{P} = \frac{1}{n} \mathbf{N}
    $$

    The row masses $\mathbf{r} \in \mathbb{R}^I$ and column masses $\mathbf{c} \in \mathbb{R}^J$ are the marginal probability vectors:

    $$
    \mathbf{r} = \mathbf{P} \mathbf{1}_J, \quad \mathbf{c} = \mathbf{P}^\top \mathbf{1}_I
    $$

    Let $\mathbf{D}_r = \operatorname{diag}(\mathbf{r})$ and $\mathbf{D}_c = \operatorname{diag}(\mathbf{c})$ denote the diagonal mass matrices.

    ---

    ### 2. Row and Column Profiles

    - The **row profiles** $\mathbf{R} \in \mathbb{R}^{I \times J}$ are the conditional distributions of columns given each row:

    $$
    \mathbf{R} = \mathbf{D}_r^{-1} \mathbf{P}, \quad R_{ij} = \frac{P_{ij}}{r_i} = \frac{N_{ij}}{N_{i\cdot}}
    $$

    - The **column profiles** $\mathbf{C} \in \mathbb{R}^{J \times I}$ are the conditional distributions of rows given each column:

    $$
    \mathbf{C} = \mathbf{D}_c^{-1} \mathbf{P}^\top, \quad C_{ji} = \frac{P_{ij}}{c_j} = \frac{N_{ij}}{N_{\cdot j}}
    $$

    The average row profile is the column marginal $\mathbf{c}^\top$, and the average column profile is the row marginal $\mathbf{r}^\top$.

    ---

    ### 3. The Chi-Square Metric and Total Inertia

    The squared $\chi^2$-distance between two row profiles $i$ and $i'$ is weighted by the inverse of column masses:

    $$
    d_{\chi^2}^2(i, i') = \sum_{j=1}^J \frac{1}{c_j} \left(R_{ij} - R_{i'j}\right)^2 = \left(\mathbf{R}_{i:} - \mathbf{R}_{i':}\right) \mathbf{D}_c^{-1} \left(\mathbf{R}_{i:} - \mathbf{R}_{i':}\right)^\top
    $$

    The **total inertia** $\phi^2$ measures the weighted sum of squared distances of all row profiles from their centroid $\mathbf{c}$:

    $$
    \phi^2 = \sum_{i=1}^I r_i \, d_{\chi^2}^2(\mathbf{R}_{i:}, \mathbf{c}) = \sum_{i=1}^I \sum_{j=1}^J \frac{(P_{ij} - r_i c_j)^2}{r_i c_j} = \frac{\chi^2}{n}
    $$

    Total inertia is identically Pearson's $\chi^2$ statistic divided by the grand total $n$.

    ---

    ### 4. Standardized Residuals and Generalized SVD

    To find the optimal low-dimensional representation, CA computes the Singular Value Decomposition of the **standardized residual matrix** $\mathbf{S}$:

    $$
    \mathbf{S} = \mathbf{D}_r^{-1/2} \left(\mathbf{P} - \mathbf{r} \mathbf{c}^\top\right) \mathbf{D}_c^{-1/2}
    $$

    Notice that $S_{ij} = \frac{P_{ij} - r_i c_j}{\sqrt{r_i c_j}} = \frac{N_{ij} - E_{ij}}{\sqrt{n E_{ij}}}$, which is the standardized Pearson residual scaled by $\frac{1}{\sqrt{n}}$.

    Computing the thin SVD:

    $$
    \mathbf{S} = \mathbf{U} \boldsymbol{\Gamma} \mathbf{V}^\top
    $$

    where $\boldsymbol{\Gamma} = \operatorname{diag}(\gamma_1, \gamma_2, \dots, \gamma_K)$ contains the singular values. The **principal inertias** (eigenvalues) are:

    $$
    \lambda_k = \gamma_k^2, \quad \sum_{k=1}^K \lambda_k = \phi^2
    $$

    ---

    ### 5. Principal Coordinates and the Symmetric Biplot

    To project both row and column points into the same orthogonal coordinate space:
    - **Principal Row Coordinates ($\mathbf{F}$)**:
      $$
      \mathbf{F} = \mathbf{D}_r^{-1/2} \mathbf{U} \boldsymbol{\Gamma}
      $$
    - **Principal Column Coordinates ($\mathbf{G}$)**:
      $$
      \mathbf{G} = \mathbf{D}_c^{-1/2} \mathbf{V} \boldsymbol{\Gamma}
      $$

    In the **Symmetric Biplot**, rows and columns are simultaneously plotted along the first two principal axes:
    - Proximity between two row points indicates that they share similar proportional distributions across column categories.
    - Proximity between a row point and a column point indicates positive association (the count $N_{ij}$ exceeds the expected count under independence $E_{ij}$).
    """)
    return


@app.cell
def _(pd, prince):
    # Data Cell: Premier League 2023-24 Season Performance Contingency Table
    _raw_dataset = prince.datasets.load_premier_league()
    _table_2024 = _raw_dataset.loc[:, ["2023-24"]].copy()
    _table_2024.columns = _table_2024.columns.droplevel(0)
    _table_2024.columns = ["Wins", "Draws", "Losses", "Goals", "Conceded", "Points"]
    _table_2024.columns.name = "Metric"

    # Select representative clubs for a clean, highly interpretable biplot
    _clubs = [
        "Arsenal",
        "Manchester City",
        "Liverpool",
        "Aston Villa",
        "Tottenham Hotspur",
        "Chelsea",
        "Newcastle United",
        "Manchester United",
        "West Ham United",
        "Everton",
        "Brentford",
        "Wolverhampton Wanderers",
    ]
    df_pl = _table_2024.loc[_clubs].copy()

    return (df_pl,)


@app.cell
def _(df_pl, go, make_subplots, mo, np):
    # Interactive Visualizations Cell:
    # Subplot 1: Symmetric Biplot (Clubs in blue, Performance Metrics in red)
    # Subplot 2: Scree Plot of Principal Inertias & Cumulative Variance Explained
    # Subplot 3: Standardized Chi-Square Residuals Heatmap (N_ij - E_ij) / sqrt(E_ij)

    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=(
            "1. Correspondence Analysis Symmetric Biplot",
            "2. Principal Inertia Scree Plot",
            "3. Standardized Chi-Square Residuals",
        ),
        horizontal_spacing=0.09,
    )

    # Step-by-step CA Computation in pure NumPy
    _N = df_pl.to_numpy()
    _n = np.sum(_N)
    _P = _N / _n

    _r = np.sum(_P, axis=1)
    _c = np.sum(_P, axis=0)

    _Dr_inv_sqrt = np.diag(1.0 / np.sqrt(_r))
    _Dc_inv_sqrt = np.diag(1.0 / np.sqrt(_c))

    # Standardized residual matrix S
    _S = _Dr_inv_sqrt @ (_P - np.outer(_r, _c)) @ _Dc_inv_sqrt
    _U, _gamma, _Vt = np.linalg.svd(_S, full_matrices=False)

    # Principal coordinates
    _F = _Dr_inv_sqrt @ _U @ np.diag(_gamma)
    _G = _Dc_inv_sqrt @ _Vt.T @ np.diag(_gamma)

    _inertias = _gamma**2
    _total_inertia = np.sum(_inertias)
    _pve = _inertias / _total_inertia

    _club_names = list(df_pl.index)
    _metric_names = list(df_pl.columns)

    # Subplot 1: Biplot
    # Plot Row Points (Clubs)
    _fig.add_trace(
        go.Scatter(
            x=_F[:, 0],
            y=_F[:, 1],
            mode="markers+text",
            marker=dict(size=9, color="#2563eb"),
            text=_club_names,
            textposition="top right",
            textfont=dict(size=8, color="#1e3a8a"),
            name="Clubs (Row Profiles)",
            hovertemplate="Club: %{text}<br>Dim 1: %{x:.3f}<br>Dim 2: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Plot Column Points (Metrics)
    _fig.add_trace(
        go.Scatter(
            x=_G[:, 0],
            y=_G[:, 1],
            mode="markers+text",
            marker=dict(size=11, symbol="diamond", color="#dc2626"),
            text=_metric_names,
            textposition="bottom left",
            textfont=dict(size=9, color="#991b1b"),
            name="Metrics (Col Profiles)",
            hovertemplate="Metric: %{text}<br>Dim 1: %{x:.3f}<br>Dim 2: %{y:.3f}<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Reference axis lines
    _fig.add_shape(type="line", x0=-0.6, x1=0.6, y0=0, y1=0, line=dict(color="#cbd5e1", width=1), row=1, col=1)
    _fig.add_shape(type="line", x0=0, x1=0, y0=-0.4, y1=0.4, line=dict(color="#cbd5e1", width=1), row=1, col=1)

    # Subplot 2: Scree Plot
    _dim_labels = [f"Dim {i+1}" for i in range(len(_inertias))]
    _cum_pve = np.cumsum(_pve) * 100.0

    _fig.add_trace(
        go.Bar(
            x=_dim_labels,
            y=_pve * 100.0,
            marker_color="#6366f1",
            name="Inertia PVE (%)",
            text=[f"{p*100:.1f}%" for p in _pve],
            textposition="auto",
        ),
        row=1,
        col=2,
    )

    _fig.add_trace(
        go.Scatter(
            x=_dim_labels,
            y=_cum_pve,
            mode="lines+markers",
            line=dict(color="#f59e0b", width=2),
            name="Cumulative PVE (%)",
        ),
        row=1,
        col=2,
    )

    # Subplot 3: Standardized Pearson Residuals (N - E) / sqrt(E)
    _E = np.outer(_r, _c) * _n
    _pearson_residuals = (_N - _E) / np.sqrt(_E)

    _fig.add_trace(
        go.Heatmap(
            z=_pearson_residuals,
            x=_metric_names,
            y=_club_names,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Residual", x=1.02, len=0.7),
            name="Pearson Residuals",
            hovertemplate="%{y} - %{x}: Residual = %{z:.2f}<extra></extra>",
        ),
        row=1,
        col=3,
    )

    _fig.update_layout(
        template="plotly_white",
        height=500,
        title=dict(
            text=f"Correspondence Analysis: Premier League 2023-24 (Total Inertia = {_total_inertia:.4f})",
            x=0.5,
            xanchor="center",
            font=dict(size=16, family="Inter, system-ui, sans-serif"),
        ),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(l=40, r=40, t=75, b=80),
    )

    _fig.update_xaxes(title_text=f"Dimension 1 ({_pve[0]*100:.1f}%)", row=1, col=1)
    _fig.update_yaxes(title_text=f"Dimension 2 ({_pve[1]*100:.1f}%)", row=1, col=1)

    _fig.update_xaxes(title_text="Principal Dimension", row=1, col=2)
    _fig.update_yaxes(title_text="Explained Inertia (%)", range=[0, 110], row=1, col=2)

    _fig.update_xaxes(title_text="Performance Metric", tickangle=45, row=1, col=3)
    _fig.update_yaxes(title_text="Football Club", row=1, col=3)

    return (mo.ui.plotly(_fig),)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## [d] Code Examples

    Below we implement two rigorous, production-grade demonstrations:
    1. **Full Correspondence Analysis Algorithm from Scratch**: Step-by-step computation of correspondence matrix $\mathbf{P}$, marginal masses $\mathbf{r}, \mathbf{c}$, standardized residual matrix $\mathbf{S}$, SVD decomposition, principal inertias, and principal coordinates $\mathbf{F}, \mathbf{G}$, verified against `prince.CA`.
    2. **Premier League Archetype Interpretation Table**: Formatted table quantifying each club's coordinate along Dimension 1 (Attacking Power & Victory vs. Defensive Vulnerability) and Dimension 2 (Draws vs. Polarized Outcomes).
    """)
    return


@app.cell
def _(df_pl, mo, np, pd, prince):
    # Example 1: Pure NumPy Step-by-Step CA from Scratch vs Prince
    _N = df_pl.to_numpy()
    _n = np.sum(_N)
    _P = _N / _n

    _r = np.sum(_P, axis=1)
    _c = np.sum(_P, axis=0)

    _Dr_inv_sqrt = np.diag(1.0 / np.sqrt(_r))
    _Dc_inv_sqrt = np.diag(1.0 / np.sqrt(_c))

    _S = _Dr_inv_sqrt @ (_P - np.outer(_r, _c)) @ _Dc_inv_sqrt
    _U, _gamma, _Vt = np.linalg.svd(_S, full_matrices=False)

    _inertias_scratch = _gamma**2
    _total_inertia = np.sum(_inertias_scratch)
    _chi2_stat = _total_inertia * _n

    # Reference fit from prince library
    _ca_prince = prince.CA(n_components=2, engine="sklearn", random_state=42)
    _ca_prince.fit(df_pl)
    _eigen_summary = _ca_prince.eigenvalues_summary

    _df_inertias = pd.DataFrame(
        [
            {
                "Principal Dimension": f"Dimension {i + 1}",
                "Principal Inertia (lambda)": f"{_inertias_scratch[i]:.5f}",
                "Percent of Inertia": f"{(_inertias_scratch[i] / _total_inertia) * 100.0:.2f}%",
                "Cumulative Inertia": f"{(np.sum(_inertias_scratch[:i+1]) / _total_inertia) * 100.0:.2f}%",
            }
            for i in range(len(_inertias_scratch))
        ]
    )

    _df_chi2_summary = pd.DataFrame(
        [
            {"Metric": "Grand Total Table Sum (n)", "Value": str(int(_n)), "Description": "Sum of all table cells"},
            {"Metric": "Total Inertia (phi^2)", "Value": f"{_total_inertia:.5f}", "Description": "Sum of all principal inertias"},
            {"Metric": "Pearson Chi-Square Statistic", "Value": f"{_chi2_stat:.2f}", "Description": "Total departure from independence"},
            {"Metric": "Degrees of Freedom (I-1)(J-1)", "Value": str((df_pl.shape[0] - 1) * (df_pl.shape[1] - 1)), "Description": "Table degrees of freedom"},
        ]
    )

    return (
        mo.md("#### Principal Inertia Decomposition from Scratch"),
        mo.ui.table(_df_inertias),
        mo.md("#### Table Independence Test Summary"),
        mo.ui.table(_df_chi2_summary),
    )


@app.cell
def _(df_pl, mo, np, pd):
    # Example 2: Club Archetype Interpretation along Dimension 1 and 2
    _N = df_pl.to_numpy()
    _n = np.sum(_N)
    _P = _N / _n
    _r = np.sum(_P, axis=1)
    _c = np.sum(_P, axis=0)

    _Dr_inv_sqrt = np.diag(1.0 / np.sqrt(_r))
    _Dc_inv_sqrt = np.diag(1.0 / np.sqrt(_c))

    _S = _Dr_inv_sqrt @ (_P - np.outer(_r, _c)) @ _Dc_inv_sqrt
    _U, _gamma, _ = np.linalg.svd(_S, full_matrices=False)
    _F = _Dr_inv_sqrt @ _U @ np.diag(_gamma)

    _records = []
    for _i, _club in enumerate(df_pl.index):
        _d1 = _F[_i, 0]
        _d2 = _F[_i, 1]

        if _d1 < -0.15:
            _archetype = "Elite Title Contender (High Wins, Goals, Points)"
        elif _d1 > 0.15:
            _archetype = "Relegation Struggler (High Losses, Conceded)"
        else:
            _archetype = "Mid-Table Stability"

        _records.append({
            "Football Club": _club,
            "Dimension 1 Coordinate": f"{_d1:+.3f}",
            "Dimension 2 Coordinate": f"{_d2:+.3f}",
            "Relative Row Mass (r_i)": f"{_r[_i]:.3f}",
            "Identified Performance Cluster": _archetype,
        })

    _df_archetypes = pd.DataFrame(_records).sort_values("Dimension 1 Coordinate")

    return (
        mo.md("#### Club Performance Archetype Mapping (CA Principal Coordinates)"),
        mo.ui.table(_df_archetypes),
    )


if __name__ == "__main__":
    app.run()
