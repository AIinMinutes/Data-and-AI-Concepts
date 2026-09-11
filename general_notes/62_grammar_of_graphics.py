import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    from plotnine.data import penguins
    from scipy import stats

    return go, make_subplots, mo, np, pd, penguins, stats


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    # The Layered Grammar of Graphics

    [← 61 User-Item Interaction Matrix](61_user_item_interaction_matrix.py) | [Index](../index.html) | [63 Einsum →](63_einsum.py)

    A statistical graphic is not a standalone chart type selected from an ad-hoc menu (e.g., "pie chart", "scatter plot"). Instead, following **Leland Wilkinson** (*The Grammar of Graphics*, 2005) and **Hadley Wickham** (*A Layered Grammar of Graphics*, 2010), a visualization is a formal composition of independent, modular layers:

    $$
    \text{Graphic} = \text{Data} + \text{Aesthetic Mappings} + \text{Geometries} + \text{Statistical Transformations} + \text{Scales} + \text{Coordinates} + \text{Facets}
    $$

    This notebook dissects the theoretical foundations of the grammar, demonstrates declarative specification using `plotnine` (Python's `ggplot2` implementation), and illustrates how the grammar resolves profound statistical phenomena such as **Simpson's Paradox** on the Palmer Penguins dataset.
    """)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## The Seven Grammatical Components

    | Layer | Formal Role | Palmer Penguins Example |
    | :--- | :--- | :--- |
    | **1. Data** | The tidy rectangular table $\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N$ | Palmer Penguins ($N = 342$ complete rows) |
    | **2. Aesthetics (`aes`)** | Mapping from variables to visual properties ($\mathcal{X} \to \mathcal{V}$) | $x \leftarrow \text{bill length}$, $y \leftarrow \text{bill depth}$, $\text{color} \leftarrow \text{species}$ |
    | **3. Geometries (`geom`)** | The geometric marks representing data points | Points (`geom_point`), lines (`geom_line`), ribbons |
    | **4. Statistics (`stat`)** | Statistical summaries or model fits computed from data | Linear regression smoother (`stat_smooth`, method=`lm`) |
    | **5. Scales (`scale`)** | Bijective mapping from data domain to perceptual space | Color palette (`scale_color_manual`), axis limits |
    | **6. Coordinates (`coord`)** | The spatial geometry mapping plane to canvas | 2D Cartesian plane ($\mathbb{R}^2$), Polar, Log-scale |
    | **7. Facets (`facet`)** | Small multiples partitioning data into subplots | Splitting by species (`facet_wrap("species")`) |

    ### Mathematical Definition of Aesthetic Mapping

    An aesthetic mapping $\phi$ is a function that projects a column vector $\mathbf{v} \in \mathcal{D}$ into a visual channel $\mathcal{A}$:

    $$
    \phi: \operatorname{dom}(\mathbf{v}) \longrightarrow \mathcal{A} \subset \mathbb{R}^d
    $$

    where $\mathcal{A}$ may represent spatial coordinates $(x, y) \in \mathbb{R}^2$, chromatic coordinates $(r, g, b) \in [0, 1]^3$, point size $s \in \mathbb{R}^+$, or glyph shape $\sigma \in \{1, \dots, K\}$.
    """)


@app.cell
def _(np, pd, penguins, stats):
    # Load and clean Palmer Penguins dataset
    df_penguins = penguins.dropna(subset=["bill_length_mm", "bill_depth_mm", "species"]).copy()
    df_penguins["species"] = df_penguins["species"].astype(str)

    species_list = sorted(df_penguins["species"].unique().tolist())
    species_colors = {
        "Adelie": "#f97316",     # Orange
        "Chinstrap": "#8b5cf6",  # Purple
        "Gentoo": "#06b6d4",     # Teal
    }

    # Compute pooled correlation and linear regression
    x_pooled = df_penguins["bill_length_mm"].values
    y_pooled = df_penguins["bill_depth_mm"].values
    slope_pooled, intercept_pooled, r_pooled, p_pooled, stderr_pooled = stats.linregress(x_pooled, y_pooled)

    # Compute within-species statistics
    within_stats = {}
    for _sp in species_list:
        _sub = df_penguins[df_penguins["species"] == _sp]
        _x = _sub["bill_length_mm"].values
        _y = _sub["bill_depth_mm"].values
        _sl, _ic, _r, _p, _se = stats.linregress(_x, _y)
        within_stats[_sp] = {
            "n": len(_sub),
            "slope": _sl,
            "intercept": _ic,
            "r": _r,
            "r_squared": _r ** 2,
            "p_val": _p,
            "mean_x": float(np.mean(_x)),
            "mean_y": float(np.mean(_y)),
        }

    return (
        df_penguins,
        intercept_pooled,
        p_pooled,
        r_pooled,
        slope_pooled,
        species_colors,
        species_list,
        stderr_pooled,
        within_stats,
        x_pooled,
        y_pooled,
    )


@app.cell(hide_code=True)
def _(
    mo,
    r_pooled,
    slope_pooled,
    species_list,
    within_stats,
):
    _rows = [
        f"| **Pooled (All Species)** | **{within_stats['Adelie']['n'] + within_stats['Chinstrap']['n'] + within_stats['Gentoo']['n']}** | **{slope_pooled:.3f}** | **{r_pooled:.3f}** | **{r_pooled**2:.3f}** | Negative Slope (Confounded) |"
    ]
    for _sp in species_list:
        _ws = within_stats[_sp]
        _rows.append(
            f"| {_sp} | {_ws['n']} | {_ws['slope']:+.3f} | {_ws['r']:+.3f} | {_ws['r_squared']:.3f} | Positive Slope (True Anatomy) |"
        )
    _table_str = "\n".join(_rows)

    _table_md = mo.md(
        "| Group | Sample Size $N$ | OLS Slope $\\beta$ | Correlation $r$ | Coefficient of Det $R^2$ | Interpretation |\n"
        "| :--- | :--- | :--- | :--- | :--- | :--- |\n"
        f"{_table_str}"
    )

    _text_md = mo.md(r"""
    ---

    ## Simpson's Paradox: A Case Study in Aesthetic Grouping

    The Palmer Penguins dataset illustrates **Simpson's Paradox** with extraordinary clarity:
    - **Pooled correlation**: $r \approx -0.235$ (longer bills appear to have *shallower* depths).
    - **Within-species correlation**: $r > 0$ for all three species ($+0.39$ for Adelie, $+0.65$ for Chinstrap, $+0.64$ for Gentoo).

    ### Mathematical Decomposition of Omitted Grouping Bias

    Let $Y$ denote bill depth, $X$ bill length, and $Z \in \{1, 2, 3\}$ the latent species indicator. Under linear assumptions:

    $$
    Y_i = \alpha + \beta_{\text{within}} X_i + \sum_{k=1}^K \gamma_k \mathbf{1}(Z_i = k) + \epsilon_i
    $$

    When we omit the grouping variable $Z$ and fit the univariate regression $Y_i = \alpha_0 + \beta_{\text{pooled}} X_i + u_i$, the estimated coefficient is:

    $$
    \beta_{\text{pooled}} = \beta_{\text{within}} + \sum_{k=1}^K \gamma_k \frac{\operatorname{Cov}(\mathbf{1}(Z_i = k), X_i)}{\operatorname{Var}(X_i)}
    $$

    Because Gentoo penguins have significantly longer bills ($\mu_X \approx 47.5$ mm vs $38.8$ mm for Adelie) but substantially shallower bills ($\mu_Y \approx 15.0$ mm vs $18.3$ mm for Adelie), $\operatorname{Cov}(\mathbf{1}(Z_i = \text{Gentoo}), X_i) > 0$ while $\gamma_{\text{Gentoo}} < 0$. This negative covariance term overwhelms the positive biological slope $\beta_{\text{within}} > 0$!
    """)

    return mo.vstack([_text_md, _table_md])


@app.cell(hide_code=True)
def _(
    df_penguins,
    go,
    intercept_pooled,
    make_subplots,
    mo,
    np,
    slope_pooled,
    species_colors,
    species_list,
    within_stats,
):
    # Create side-by-side Plotly comparison: Pooled vs Grouped Grammar Layer
    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "Layer 1-4: Data + Geom Point + Pooled Stat Smooth",
            "Layer 5: Aesthetic Mapping color = species + Grouped Smooth",
        ],
        horizontal_spacing=0.10,
    )

    # Left Plot: Pooled scatter and regression line
    _fig.add_trace(
        go.Scatter(
            x=df_penguins["bill_length_mm"],
            y=df_penguins["bill_depth_mm"],
            mode="markers",
            marker=dict(size=7, color="#64748b", opacity=0.65),
            name="Penguins (Ungrouped)",
            hovertemplate="Bill Length: %{x} mm<br>Bill Depth: %{y} mm<extra></extra>",
        ),
        row=1,
        col=1,
    )

    _x_grid_pooled = np.linspace(df_penguins["bill_length_mm"].min(), df_penguins["bill_length_mm"].max(), 100)
    _y_pred_pooled = slope_pooled * _x_grid_pooled + intercept_pooled

    _fig.add_trace(
        go.Scatter(
            x=_x_grid_pooled,
            y=_y_pred_pooled,
            mode="lines",
            line=dict(color="#dc2626", width=2.5, dash="solid"),
            name="Pooled OLS (Slope: -0.08)",
            hovertemplate="Length: %{x:.1f} mm<br>Predicted Depth: %{y:.1f} mm<extra></extra>",
        ),
        row=1,
        col=1,
    )

    # Right Plot: Grouped by species with colored points and per-species regression lines
    for _sp in species_list:
        _sub = df_penguins[df_penguins["species"] == _sp]
        _color = species_colors[_sp]
        _ws = within_stats[_sp]

        # Points
        _fig.add_trace(
            go.Scatter(
                x=_sub["bill_length_mm"],
                y=_sub["bill_depth_mm"],
                mode="markers",
                marker=dict(size=7, color=_color, opacity=0.75),
                name=f"{_sp} (r = {_ws['r']:.2f})",
                hovertemplate=f"<b>{_sp}</b><br>Length: %{{x}} mm<br>Depth: %{{y}} mm<extra></extra>",
            ),
            row=1,
            col=2,
        )

        # Fitted regression line
        _x_grid_sp = np.linspace(_sub["bill_length_mm"].min(), _sub["bill_length_mm"].max(), 80)
        _y_pred_sp = _ws["slope"] * _x_grid_sp + _ws["intercept"]

        _fig.add_trace(
            go.Scatter(
                x=_x_grid_sp,
                y=_y_pred_sp,
                mode="lines",
                line=dict(color=_color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=2,
        )

    _fig.update_layout(
        title="The Layered Grammar in Action: Resolving Simpson's Paradox",
        template="plotly_white",
        height=480,
        margin=dict(l=60, r=40, t=70, b=60),
        legend=dict(x=0.80, y=0.05, bgcolor="rgba(255,255,255,0.9)", bordercolor="#cbd5e1", borderwidth=1),
    )

    _fig.update_xaxes(title_text="Bill Length (mm)", row=1, col=1)
    _fig.update_xaxes(title_text="Bill Length (mm)", row=1, col=2)
    _fig.update_yaxes(title_text="Bill Depth (mm)", row=1, col=1)
    _fig.update_yaxes(title_text="Bill Depth (mm)", row=1, col=2)

    _md = mo.md(r"""
    ### Interactive Comparison: Pooled vs Grouped Grammar Layer

    In the grammar of graphics, grouping is **not** a preprocessing step (such as splitting the dataset into separate data structures). Instead, it is an **aesthetic mapping**:
    ```python
    aes(x="bill_length_mm", y="bill_depth_mm", color="species")
    ```
    Downstream statistical layers (`geom_smooth`) inherit this mapping automatically, fitting separate regressors across the data partition defined by the scale.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(df_penguins, go, make_subplots, mo, np, species_colors, species_list, within_stats):
    # Faceted Small Multiples visualization
    _fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[f"Species: {_sp} (N = {within_stats[_sp]['n']})" for _sp in species_list],
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    for _idx, _sp in enumerate(species_list, start=1):
        _sub = df_penguins[df_penguins["species"] == _sp]
        _color = species_colors[_sp]
        _ws = within_stats[_sp]

        # Points
        _fig.add_trace(
            go.Scatter(
                x=_sub["bill_length_mm"],
                y=_sub["bill_depth_mm"],
                mode="markers",
                marker=dict(size=7, color=_color, opacity=0.75),
                name=_sp,
                showlegend=False,
                hovertemplate=f"<b>{_sp}</b><br>Length: %{{x}} mm<br>Depth: %{{y}} mm<extra></extra>",
            ),
            row=1,
            col=_idx,
        )

        # Fitted regression line
        _x_grid = np.linspace(_sub["bill_length_mm"].min(), _sub["bill_length_mm"].max(), 50)
        _y_pred = _ws["slope"] * _x_grid + _ws["intercept"]

        _fig.add_trace(
            go.Scatter(
                x=_x_grid,
                y=_y_pred,
                mode="lines",
                line=dict(color=_color, width=2.5),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=_idx,
        )

        _fig.update_xaxes(title_text="Bill Length (mm)", range=[30, 60], row=1, col=_idx)

    _fig.update_yaxes(title_text="Bill Depth (mm)", range=[12, 23], row=1, col=1)

    _fig.update_layout(
        title="Layer 7: Faceting (Small Multiples Conditioning with Shared Coordinate Scales)",
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=40, t=70, b=60),
    )

    _md = mo.md(r"""
    ---

    ## Layer 7: Faceting and Small Multiples

    **Faceting** partitions the dataset into disjoint subsets conditioned on discrete categorical variables, rendering identical sub-geometries across a shared coordinate grid (`facet_wrap` or `facet_grid`).

    By aligning the $x$ and $y$ axis limits identically across all facets, the user can immediately perceive both:
    1. The **within-group covariance**: Every species exhibits a positive morphological relationship between length and depth.
    2. The **between-group centroid displacement**: Gentoo penguins form a distinct cluster with significantly higher bill length and shallower depth.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Declarative Specification in `plotnine`

    In Python, the `plotnine` library provides a 1-to-1 implementation of Wickham's Grammar of Graphics. The entire visualization pipeline is expressed as additive algebraic compositions:

    ```python
    from plotnine import (
        ggplot, aes, geom_point, geom_smooth,
        scale_color_manual, labs, facet_wrap, theme_minimal
    )

    # 1. Base grammar: Data + Aesthetics + Points + Linear Smoother
    p_pooled = (
        ggplot(df_penguins, aes(x="bill_length_mm", y="bill_depth_mm"))
        + geom_point(alpha=0.6, size=2.2)
        + geom_smooth(method="lm", se=True, color="#dc2626")
        + labs(title="Pooled OLS across species")
        + theme_minimal()
    )

    # 2. Layering aesthetic color mapping: resolves Simpson's Paradox
    p_grouped = (
        ggplot(df_penguins, aes(x="bill_length_mm", y="bill_depth_mm", color="species"))
        + geom_point(alpha=0.7, size=2.2)
        + geom_smooth(method="lm", se=True)
        + scale_color_manual(values={"Adelie": "#f97316", "Chinstrap": "#8b5cf6", "Gentoo": "#06b6d4"})
        + labs(title="Within-species OLS: Positive morphological slopes")
        + theme_minimal()
    )

    # 3. Layering faceting: Small multiples
    p_faceted = p_grouped + facet_wrap("~species")
    ```

    ### Key Differences: Imperative vs Declarative Visualization

    | Aspect | Imperative (`matplotlib`) | Declarative Grammar (`plotnine`, `ggplot2`) |
    | :--- | :--- | :--- |
    | **Mental Model** | Step-by-step drawing instructions on a canvas | Mathematical mapping from relations to visual channels |
    | **Grouping** | Manual loops (`for name, group in df.groupby(...)`) | Declared in `aes(color="variable")` |
    | **Statistical Fits** | Precomputed externally with `scipy` or `statsmodels` | Integrated directly via `geom_smooth(method="lm")` |
    | **Legends & Scales** | Manually constructed handles and labels | Automatically inferred from variable types and domains |
    | **Faceting** | Manual `plt.subplots` grid management and index math | Handled declaratively with `facet_wrap` / `facet_grid` |
    """)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Key Takeaways and Principles

    1. **Graphics as Language**: Plots are compositions of data, mappings, geometries, statistics, scales, coordinates, and facets.
    2. **Aesthetic Mapping vs Fixed Attribute**:
       - `geom_point(color="blue")` sets a fixed aesthetic property (ink color).
       - `aes(color="species")` establishes an aesthetic mapping that partitions the dataset and informs downstream statistical layers.
    3. **Visual Proof of Statistical Phenomena**: Proper aesthetic mappings expose critical confounds (such as Simpson's Paradox) that remain concealed in unstratified aggregate views.
    """)


if __name__ == "__main__":
    app.run()
