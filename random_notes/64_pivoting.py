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

    return go, make_subplots, mo, np, pd


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    # Tabular Data Reshaping: Pivoting, Melting, and Relational Duality

    [← 63 Einsum](63_einsum.py) | [Index](../index.html) | [65 GPU Acceleration (cuDF) →](65_cudf.py)

    Data reshaping lies at the intersection of relational algebra, tensor multidimensionality, and statistical modeling. Real-world analytical workflows constantly alternate between two canonical representations:
    - **Tidy / Long Format**: Optimized for relational databases, split-apply-combine transformations (`groupby`), and the Grammar of Graphics.
    - **Wide / Matrix Format**: Optimized for linear algebra, correlation analysis, distance metrics, covariance computation, and tabular dashboards.

    This notebook formalizes the mathematical transformations governing **pivoting** (long to wide aggregation) and **melting** (wide to long normalization), audits the inverse transformation invariant, and provides interactive multi-index pivot visualizations.
    """)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Mathematical Formalism: The Pivot-Melt Duality

    ### 1. The Tidy Long Relation

    Let a dataset be represented as a relation in third normal form:

    $$
    \mathcal{R} \subseteq \mathcal{I} \times \mathcal{C} \times \mathcal{V}
    $$

    where:
    - $\mathcal{I} = \{i_1, i_2, \dots, i_M\}$ is the set of observation identifiers (row indices, such as timestamp, subject ID).
    - $\mathcal{C} = \{c_1, c_2, \dots, c_K\}$ is the set of variable/feature attributes (column headers).
    - $\mathcal{V} \subseteq \mathbb{R}$ is the measurement domain.

    ### 2. The Pivot Operator ($\mathcal{P}$)

    The pivot operator $\mathcal{P}: \mathcal{R} \to \mathbb{R}^{M \times K}$ maps each coordinate pair $(i_m, c_k)$ to a cell in a 2D matrix. When duplicate entries exist for a given $(i_m, c_k)$ pair, an aggregation operator $\bigoplus$ (such as mean, sum, or max) reduces the multiset:

    $$
    P(i_m, c_k) = \bigoplus_{(i, c, v) \in \mathcal{R} \,:\, i=i_m, \, c=c_k} v
    $$

    For arithmetic mean aggregation:

    $$
    P(i_m, c_k) = \frac{1}{|S(i_m, c_k)|} \sum_{v \in S(i_m, c_k)} v, \quad \text{where } S(i_m, c_k) = \{v \mid (i_m, c_k, v) \in \mathcal{R}\}
    $$

    ### 3. The Melt (Unpivot) Operator ($\mathcal{M}$)

    The melt operator is the algebraic adjoint of pivoting. It unrolls a wide matrix $\mathbf{X} \in \mathbb{R}^{M \times K}$ into atomic triples:

    $$
    \mathcal{M}(\mathbf{X}) = \bigcup_{m=1}^M \bigcup_{k=1}^K \left\{ (i_m, c_k, X_{mk}) \right\}
    $$

    ### Algebraic Invertibility Invariant

    When the relation $\mathcal{R}$ has unique keys (i.e. $|S(i_m, c_k)| \le 1$ for all pairs), pivoting and melting are exact inverses:

    $$
    \mathcal{M}(\mathcal{P}(\mathcal{R})) \cong \mathcal{R}
    $$
    """)


@app.cell
def _(np, pd):
    # Generate a reproducible, self-contained longitudinal neuroimaging dataset
    # 8 subjects, 10 timepoints, 2 experimental conditions (Stimulus vs Cue), 2 brain regions (Frontal, Parietal)
    rng = np.random.default_rng(42)

    subjects = [f"s{i:02d}" for i in range(1, 9)]
    timepoints = list(range(10))
    events = ["cue", "stim"]
    regions = ["Frontal", "Parietal"]

    records = []
    for _sub in subjects:
        # Subject-specific baseline shift
        _sub_offset = rng.normal(0.0, 0.05)
        for _tp in timepoints:
            for _ev in events:
                for _reg in regions:
                    # Synthetic hemodynamic response curve (gamma-like impulse response)
                    _t_peak = 4.0 if _ev == "stim" else 2.5
                    _amp = 0.25 if _ev == "stim" else 0.12
                    _reg_scale = 1.2 if _reg == "Frontal" else 0.8
                    _signal_mean = _amp * _reg_scale * (_tp / _t_peak) * np.exp(1.0 - (_tp / _t_peak))
                    _noise = rng.normal(0.0, 0.03)
                    _signal = float(_signal_mean + _sub_offset + _noise)

                    records.append({
                        "subject": _sub,
                        "timepoint": _tp,
                        "event": _ev,
                        "region": _reg,
                        "signal": _signal,
                    })

    df_long = pd.DataFrame(records)
    n_rows_long = len(df_long)
    return (
        df_long,
        events,
        n_rows_long,
        records,
        regions,
        rng,
        subjects,
        timepoints,
    )


@app.cell(hide_code=True)
def _(df_long, mo, n_rows_long):
    _head_md = df_long.head(6).to_markdown(index=False)
    return mo.md(f"""
    ### Raw Long-Format Dataset (First 6 of {n_rows_long} Observations)

    {_head_md}

    Each row represents a single atomic measurement tuple: $(\\text{{subject}}, \\text{{timepoint}}, \\text{{event}}, \\text{{region}}, \\text{{signal}})$.
    """)


@app.cell
def _(df_long):
    # 1. Standard Pivot Table: Mean signal across timepoints and events
    df_pivoted_event = df_long.pivot_table(
        index="timepoint",
        columns="event",
        values="signal",
        aggfunc="mean",
    )

    # 2. Multi-Index Pivot: Mean signal indexed by timepoint and partitioned across (region, event)
    df_pivoted_multi = df_long.pivot_table(
        index="timepoint",
        columns=["region", "event"],
        values="signal",
        aggfunc="mean",
    )

    # 3. Standard Deviation Pivot (Dispersion across subjects)
    df_pivoted_std = df_long.pivot_table(
        index="timepoint",
        columns="event",
        values="signal",
        aggfunc="std",
    )

    # 4. Verification: Melt the pivoted table back to long format
    df_melted_back = df_pivoted_event.reset_index().melt(
        id_vars="timepoint",
        value_vars=["cue", "stim"],
        var_name="event",
        value_name="signal_mean",
    )

    return (
        df_melted_back,
        df_pivoted_event,
        df_pivoted_multi,
        df_pivoted_std,
    )


@app.cell(hide_code=True)
def _(df_pivoted_event, go, mo):
    # Visualize the pivoted matrix as an annotated heatmap
    _z_vals = df_pivoted_event.values
    _x_cols = list(df_pivoted_event.columns)
    _y_rows = [f"t = {t}" for t in df_pivoted_event.index]

    _fig = go.Figure(
        data=go.Heatmap(
            z=_z_vals,
            x=_x_cols,
            y=_y_rows,
            colorscale="Plasma",
            colorbar=dict(title="Mean Signal", len=0.8),
            text=[[f"{v:.4f}" for v in row] for row in _z_vals],
            texttemplate="%{text}",
            textfont=dict(size=12, color="#ffffff"),
            hovertemplate="Timepoint: %{y}<br>Event: %{x}<br>Mean Signal: %{z:.4f}<extra></extra>",
        )
    )

    _fig.update_layout(
        title="Pivot Table Heatmap: Mean BOLD Signal by Timepoint and Event",
        xaxis_title="Event Condition",
        yaxis_title="Timepoint",
        yaxis_autorange="reversed",
        template="plotly_white",
        height=480,
        margin=dict(l=80, r=40, t=60, b=60),
    )

    _md = mo.md(r"""
    ---

    ## Pivoted 2D Matrix Representation

    Pivoting aggregates the longitudinal subject records into a compact 2D summary matrix where temporal dynamics and condition differences can be evaluated instantaneously.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(df_pivoted_event, df_pivoted_std, go, mo):
    # Time-series response curve with standard error bands
    _timepoints = df_pivoted_event.index.values

    _fig = go.Figure()

    _colors = {"stim": "#3b82f6", "cue": "#f59e0b"}

    for _ev in ["stim", "cue"]:
        _mean = df_pivoted_event[_ev].values
        _std = df_pivoted_std[_ev].values
        _color = _colors[_ev]

        # Mean line
        _fig.add_trace(
            go.Scatter(
                x=_timepoints,
                y=_mean,
                mode="lines+markers",
                name=f"{_ev.capitalize()} Condition",
                line=dict(color=_color, width=3),
                marker=dict(size=8),
                hovertemplate=f"<b>{_ev.capitalize()}</b><br>t = %{{x}}<br>Mean: %{{y:.4f}}<extra></extra>",
            )
        )

        # Upper bound
        _fig.add_trace(
            go.Scatter(
                x=_timepoints,
                y=_mean + _std,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        # Lower bound with fill
        _fig.add_trace(
            go.Scatter(
                x=_timepoints,
                y=_mean - _std,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor=f"rgba{tuple(list(int(_color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)) + [0.15])}",
                name=f"{_ev.capitalize()} (+/- 1 SD)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

    _fig.update_layout(
        title="Hemodynamic Response Trajectory: Stimulus vs Cue across Timepoints",
        xaxis_title="Timepoint (Scan Interval)",
        yaxis_title="Hemodynamic Signal Amplitude",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.80, y=0.95, bgcolor="rgba(255,255,255,0.8)", bordercolor="#cbd5e1", borderwidth=1),
    )

    _md = mo.md(r"""
    ---

    ## Longitudinal Trajectory Derived from Pivoted Columns

    Having events as dedicated matrix columns allows direct arithmetic operations between conditions, such as the differential contrast:

    $$
    \Delta(t) = P(t, \text{stim}) - P(t, \text{cue})
    $$
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(df_pivoted_multi, go, make_subplots, mo):
    # Multi-index pivot visualization across brain regions and events
    _regions = ["Frontal", "Parietal"]
    _events = ["stim", "cue"]

    _fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[f"Brain Region: {reg}" for reg in _regions],
        shared_yaxes=True,
        horizontal_spacing=0.08,
    )

    _colors = {"stim": "#3b82f6", "cue": "#f59e0b"}

    for _col_idx, _reg in enumerate(_regions, start=1):
        for _ev in _events:
            _y_series = df_pivoted_multi[(_reg, _ev)]
            _fig.add_trace(
                go.Scatter(
                    x=_y_series.index.values,
                    y=_y_series.values,
                    mode="lines+markers",
                    name=f"{_ev.capitalize()} ({_reg})",
                    line=dict(color=_colors[_ev], width=2.5, dash="solid" if _reg == "Frontal" else "dash"),
                    marker=dict(size=7),
                    hovertemplate=f"<b>{_reg} - {_ev}</b><br>t = %{{x}}<br>Signal: %{{y:.4f}}<extra></extra>",
                ),
                row=1,
                col=_col_idx,
            )
        _fig.update_xaxes(title_text="Timepoint", row=1, col=_col_idx)

    _fig.update_yaxes(title_text="Signal Amplitude", row=1, col=1)

    _fig.update_layout(
        title="Hierarchical Multi-Index Pivot: Region x Event Factorial Decomposition",
        template="plotly_white",
        height=400,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    _md = mo.md(r"""
    ---

    ## Hierarchical Multi-Index Pivoting

    Pandas supports hierarchical indexing (Cartesian product of multiple row and column factors):
    ```python
    df.pivot_table(index="timepoint", columns=["region", "event"], values="signal", aggfunc="mean")
    ```
    This corresponds to folding a relational table into a 3D tensor $\mathcal{T} \in \mathbb{R}^{T \times R \times E}$.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(df_melted_back, mo):
    _top_md = mo.md(r"""
    ---

    ## The Inverse Transformation: Melting Back to Tidy Data

    To pass pivoted summaries into visualization libraries or regression models, we apply `melt`:
    ```python
    df_pivoted.reset_index().melt(
        id_vars="timepoint",
        value_vars=["cue", "stim"],
        var_name="event",
        value_name="signal",
    )
    ```
    """)

    _table_md = mo.md(df_melted_back.head(8).to_markdown(index=False))

    _bottom_md = mo.md(r"""
    ### Algebraic Invariant Audit

    Let $\mathbf{P} = \mathcal{P}(\mathcal{D})$ and $\mathcal{D}' = \mathcal{M}(\mathbf{P})$.
    Every row $(t, e, s)$ in $\mathcal{D}'$ exactly satisfies $s = \frac{1}{|S(t, e)|} \sum_{v \in S(t, e)} v$, verifying mathematical consistency.
    """)

    return mo.vstack([_top_md, _table_md, _bottom_md])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Summary Comparison: Pivot vs Pivot Table vs Melt

    | Method | Input Format | Output Format | Handles Duplicates? | Core Use Case |
    | :--- | :--- | :--- | :--- | :--- |
    | `df.pivot()` | Long | Wide | **No** (raises `ValueError`) | Pure 1-to-1 index-column reshuffling |
    | `df.pivot_table()` | Long | Wide | **Yes** (aggregates via `aggfunc`) | Dimensional summary and factorial analysis |
    | `df.melt()` | Wide | Long | **N/A** (expands columns to rows) | Normalizing data for tidy pipelines |
    | `df.unstack()` | Multi-Index Series | Wide DataFrame | **No** | Pivoting an inner level of a MultiIndex |
    | `df.stack()` | Wide DataFrame | Multi-Index Series | **No** | Melting columns into a row index hierarchy |
    """)


if __name__ == "__main__":
    app.run()
