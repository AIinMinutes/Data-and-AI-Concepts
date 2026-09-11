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
    # **Speeding Up Pandas with GPU Acceleration Using cuDF**

    Pandas is an incredibly flexible and powerful library for data manipulation, but it often struggles with performance, especially when working with large datasets. While pandas can handle various operations efficiently, certain limitations—such as single-threaded execution and memory management—can lead to slow processing. If you have a suitable NVIDIA GPU, **cuDF**, a part of the RAPIDS ecosystem, can help accelerate your pandas code without requiring major changes.

    ## Why Pandas Struggles with Performance

    Despite its flexibility, pandas faces performance issues due to the following main reasons:

    ~> **Single-Threaded Operations**: Most operations in pandas are single-threaded, meaning the CPU remains underutilized, especially for large datasets.
    ~> **Memory Handling**: Pandas loads entire datasets into memory and may swap data to disk when the dataset exceeds available memory, which can significantly slow down operations.

    ## GPU Acceleration with cuDF

    If you have access to an NVIDIA GPU, you can use **cuDF** to accelerate your pandas operations:

    ~> **cuDF** is part of the **RAPIDS** ecosystem and is designed to leverage the power of **NVIDIA GPUs** for data processing.
    ~> You can use cuDF to accelerate your pandas code without making significant changes, as it mimics the pandas API.

    ## How cuDF Works

    ~> **cuDF** provides a pandas-like API, but it runs computations on the GPU using **CUDA**.
    ~> This allows for substantial performance improvements, especially when dealing with large datasets, by utilizing the parallelism and speed of the GPU.

    ## Minimal Code Changes Required

    ~> You don’t need to rewrite your entire codebase to use cuDF. Simply replace:
      ```python
      import pandas as pd
      import cudf as pd
      ```
    """)
    return


@app.cell
def _():
    # !pip3 install --extra-index-url=https://pypi.nvidia.com polars[gpu] cudf-cu12
    return


@app.cell
def _():
    import time

    import cudf
    import numpy as np
    import pandas as pd
    import polars as pl
    from sklearn.datasets import load_diabetes

    return cudf, load_diabetes, pd, pl, time


@app.cell
def _(load_diabetes):
    X, _ = load_diabetes(return_X_y=True, as_frame=True)
    X = X[["age", "bmi", "bp"]]
    return (X,)


@app.cell
def _(X, pd):
    repeat = 100000
    X_big = pd.concat([X for _ in range(repeat)])
    X_big.shape[0]
    return (X_big,)


@app.cell
def _(X_big, time):
    start_time = time.time()
    (X_big.groupby("age").agg({"bmi": "mean", "bp": "max"}).sort_values(by="bmi"))
    end_time = time.time()
    return end_time, start_time


@app.cell
def _(end_time, start_time):
    print(f"Pandas took {end_time - start_time:.2f}s")
    return


@app.cell
def _(X_big, cudf):
    # Or alternatively load cudf.pandas extension
    X_cudf = cudf.DataFrame.from_pandas(X_big)
    return (X_cudf,)


@app.cell
def _(X_cudf, time):
    start_time_1 = time.time()
    X_cudf.groupby("age").agg({"bmi": "mean", "bp": "max"}).sort_values(by="bmi")
    end_time_1 = time.time()
    return end_time_1, start_time_1


@app.cell
def _(end_time_1, start_time_1):
    print(f"cuDF took {end_time_1 - start_time_1:.2f}s")
    return


@app.cell
def _(X_big, pl):
    # Polars (may require API changes)
    X_polars = pl.from_dataframe(X_big).lazy()
    return (X_polars,)


@app.cell
def _(X_polars, pl, time):
    start_time_2 = time.time()
    X_polars.group_by("age").agg([pl.col("bmi").mean(), pl.col("bp").max()]).sort("bmi")
    end_time_2 = time.time()
    return end_time_2, start_time_2


@app.cell
def _(end_time_2, start_time_2):
    print(f"Polars took {end_time_2 - start_time_2:.4f}s")
    return


@app.cell
def _(X_big, pl):
    # In fact, you can use polars with GPU (requires tuning)
    X_polars_1 = pl.from_dataframe(X_big).lazy()
    return (X_polars_1,)


@app.cell
def _(X_polars_1, pl, time):
    start_time_3 = time.time()
    X_polars_1.group_by("age").agg([pl.col("bmi").mean(), pl.col("bp").max()]).sort("bmi").collect(engine="gpu")
    end_time_3 = time.time()
    return end_time_3, start_time_3


@app.cell
def _(end_time_3, start_time_3):
    print(f"Polars (on GPU) took {end_time_3 - start_time_3:.4f}s")
    return


if __name__ == "__main__":
    app.run()
