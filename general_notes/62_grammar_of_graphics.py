import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grammar of graphics in plotnine

    plotnine is a Python port of ggplot2. A plot is not a chart type chosen from a menu. It is a **mapping** from a table to visual properties, built by adding layers (Wilkinson; Wickham).

    The data are the Palmer penguins: 344 birds, three species, a rectangular table of measurements. The sentence we will draw is *bill length against bill depth*.

    | Layer | Role | Here |
    |-------|------|------|
    | data | the table | penguins |
    | aes | map columns to $x$, $y$, colour, … | length, depth, species |
    | geom | the mark | points |
    | stat | a computed layer | linear smooth |
    | scale | values $\to$ ink | species colours |
    | theme | non-data ink | the rest of the figure |

    First the two bill columns with no grouping. Then the same plot with `color="species"`. The smoother inherits that mapping, so the fitted lines change without any extra wrangling.
    """)
    return


@app.cell
def _():
    from plotnine import (
        aes,
        element_blank,
        element_line,
        element_rect,
        element_text,
        geom_point,
        geom_smooth,
        ggplot,
        labs,
        scale_color_manual,
        theme,
        theme_minimal,
    )
    from plotnine.data import penguins

    df = penguins.dropna(subset=["bill_length_mm", "bill_depth_mm"]).copy()

    species_colors = {
        "Adelie": "#D65A31",
        "Chinstrap": "#6B4C9A",
        "Gentoo": "#1B7A7A",
    }

    def theme_note(legend=True):
        return theme_minimal(base_size=11) + theme(
            figure_size=(7.2, 4.5),
            dpi=120,
            plot_background=element_rect(fill="white", color="none"),
            panel_background=element_rect(fill="white", color="none"),
            panel_grid_major=element_line(color="#ECECEC", size=0.4),
            panel_grid_minor=element_blank(),
            axis_ticks=element_blank(),
            axis_title=element_text(size=11),
            plot_title=element_text(size=13, face="bold"),
            plot_subtitle=element_text(size=10, color="#444444"),
            plot_caption=element_text(size=8, color="#666666"),
            legend_title=element_text(size=10),
            legend_position="right" if legend else "none",
        )

    pooled_r = df["bill_length_mm"].corr(df["bill_depth_mm"])
    within_r = {name: g["bill_length_mm"].corr(g["bill_depth_mm"]) for name, g in df.groupby("species", observed=True)}
    print(f"{len(df)} penguins with both bill measurements (dropped {len(penguins) - len(df)} rows)")
    print(f"pooled r = {pooled_r:.2f}")
    print("within species: " + ", ".join(f"{name} {r:.2f}" for name, r in within_r.items()))
    return (
        aes,
        df,
        geom_point,
        geom_smooth,
        ggplot,
        labs,
        scale_color_manual,
        species_colors,
        theme_note,
    )


@app.cell
def _(aes, df, geom_point, geom_smooth, ggplot, labs, theme_note):
    (
        ggplot(df, aes("bill_length_mm", "bill_depth_mm"))
        + geom_point(color="#4A4A4A", alpha=0.55, size=2.2, stroke=0)
        + geom_smooth(
            method="lm",
            se=True,
            color="#1f4e79",
            fill="#1f4e79",
            alpha=0.15,
            size=0.9,
        )
        + labs(
            x="Bill length (mm)",
            y="Bill depth (mm)",
            title="Pooled across species",
            subtitle="The fitted line slopes down: longer bill, shallower depth.",
            caption="Palmer penguins  ·  Horst, Hill, and Gorman",
        )
        + theme_note(legend=False)
    )
    return


@app.cell
def _(
    aes,
    df,
    geom_point,
    geom_smooth,
    ggplot,
    labs,
    scale_color_manual,
    species_colors,
    theme_note,
):
    (
        ggplot(df, aes("bill_length_mm", "bill_depth_mm", color="species"))
        + geom_point(alpha=0.7, size=2.2, stroke=0)
        + geom_smooth(method="lm", se=True, alpha=0.12, size=0.9, show_legend=False)
        + scale_color_manual(values=species_colors, name="Species")
        + labs(
            x="Bill length (mm)",
            y="Bill depth (mm)",
            title="The same points, species as a mapping",
            subtitle="Each species slopes up. Gentoo sits to the right and down, and tilts the pooled line.",
            caption="Palmer penguins  ·  Horst, Hill, and Gorman",
        )
        + theme_note(legend=True)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The pooled correlation is $r \approx -0.24$. Within each species it is positive (Adelie $0.39$, Chinstrap $0.65$, Gentoo $0.64$). Gentoo bills are long and shallow, so that cluster sits to the right and down and tilts the pooled line.

    Grouping is therefore a mapping, not a preprocessing step: `aes(..., color="species")` changes which rows the smoother sees together. `facet_wrap("species")` would split the same slopes into three panels; colour keeps them in one picture so the pooled cloud and the within-species lines are both visible.

    That is the grammar. Matplotlib can draw the same ink; plotnine is the vocabulary that makes the grouping explicit.
    """)
    return


if __name__ == "__main__":
    app.run()
