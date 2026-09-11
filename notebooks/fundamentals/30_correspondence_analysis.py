import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import prince

    dataset = prince.datasets.load_premier_league()
    dataset = dataset.loc[:, ['2023-24']]
    dataset.columns = dataset.columns.droplevel(0)
    dataset.columns = [
        "Wins", "Draws", "Losses",
                "Goals", "Conceded", "Points"
    ]
    dataset.columns.name = 'Result'
    dataset.sample(4, random_state=47)
    return dataset, prince


@app.cell
def _(dataset, prince):
    ca = prince.CA(
        n_components=2, n_iter=100,
        copy=True, check_input=True,
        engine='sklearn', random_state=47
    )

    ca.fit(dataset)
    return (ca,)


@app.cell
def _(ca):
    ca.eigenvalues_summary
    return


@app.cell
def _(ca, dataset):
    ca.row_coordinates(dataset).sample(2)
    return


@app.cell
def _(ca, dataset):
    ca.column_coordinates(dataset).sample(2)
    return


@app.cell
def _(ca, dataset):
    chart = ca.plot(
        dataset,
        x_component=0,
        y_component=1,
        show_row_markers=True,
        show_column_markers=True,
        show_row_labels=True,
        show_column_labels=True
    )

    chart = chart.properties(
        width=400,
        height=300,
    )

    chart.save('correspondence_analysis.png', ppi=300)
    return


if __name__ == "__main__":
    app.run()
