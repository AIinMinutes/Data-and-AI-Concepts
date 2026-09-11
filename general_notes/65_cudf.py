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
    import polars as pl

    # Check for optional GPU libraries
    try:
        import cudf

        HAS_CUDF = True
    except ImportError:
        cudf = None
        HAS_CUDF = False

    return HAS_CUDF, cudf, go, make_subplots, mo, np, pd, pl, time


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    # GPU Acceleration and Columnar Runtimes (cuDF & Polars)

    [← 64 Pivoting](64_pivoting.py) | [Index](../index.html)

    As tabular datasets grow from gigabytes to terabytes, traditional dataframe libraries like Pandas encounter severe performance bottlenecks. Built primarily around single-threaded Python execution and row- or block-oriented memory structures, Pandas struggles to fully utilize modern multi-core CPUs and massively parallel GPUs.

    This notebook explores the hardware architecture, memory bandwidth roofline model, Apache Arrow columnar representation, and GPU acceleration via **cuDF** (NVIDIA RAPIDS) and multi-threaded CPU execution via **Polars**.
    """)


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Why Tabular Analytics is Memory-Bandwidth Bound

    Most relational dataframe operators (filtering, projection, hashing, grouping, sorting) perform very few arithmetic operations per byte read from memory. The **arithmetic intensity** $I$ is defined as:

    $$
    I = \frac{\text{FLOPs (Floating-Point Operations)}}{\text{Memory Traffic (Bytes Accessed)}}
    $$

    For an aggregation like `df["salary"].mean()`:
    - 1 addition per 8-byte float64: $I = \frac{1}{8} = 0.125 \text{ FLOPs/Byte}$.
    - Modern CPUs achieve $> 1000 \text{ GFLOPs/s}$ of compute, but system DDR memory provides only $\approx 50\text{--}100 \text{ GB/s}$ bandwidth.
    - Thus, tabular operations are overwhelmingly **memory-bandwidth bound**.

    ### Hardware Bandwidth Disparity

    Under the **Roofline Model**, the maximum attainable throughput $P_{\max}$ is bounded by memory bandwidth:

    $$
    P_{\max} \le B \times I
    $$

    where $B$ is memory bandwidth (GB/s).

    | Hardware Tier | Memory Type | Typical Bandwidth | Compute Cores |
    | :--- | :--- | :--- | :--- |
    | **CPU Host RAM** | Dual-Channel DDR5 | $60\text{--}90 \text{ GB/s}$ | $8\text{--}64$ cores |
    | **PCIe Interconnect** | PCIe 4.0 / 5.0 x16 | $31.5\text{--}63 \text{ GB/s}$ | Point-to-point bus |
    | **Workstation GPU** | GDDR6 / GDDR6X | $500\text{--}1000 \text{ GB/s}$ | $5000\text{--}16000$ CUDA cores |
    | **Data Center GPU** | HBM2e / HBM3 / HBM3e | $2000\text{--}3350 \text{ GB/s}$ | $14000\text{--}18000$ CUDA cores |

    Because GPU High-Bandwidth Memory (HBM) delivers **20x to 50x higher memory bandwidth** than CPU DDR memory, memory-bound dataframe aggregations experience massive acceleration when executed on GPUs.
    """)


@app.cell(hide_code=True)
def _(go, mo):
    # Visualize Hardware Memory Bandwidth Comparison
    _devices = [
        "PCIe 4.0 x16",
        "Host CPU DDR5",
        "PCIe 5.0 x16",
        "NVIDIA RTX 4090 (GDDR6X)",
        "NVIDIA A100 (HBM2e)",
        "NVIDIA H100 (HBM3)",
    ]
    _bandwidths = [31.5, 84.0, 63.0, 1008.0, 2039.0, 3350.0]
    _colors = ["#94a3b8", "#3b82f6", "#94a3b8", "#10b981", "#059669", "#047857"]

    _fig = go.Figure(
        go.Bar(
            x=_bandwidths,
            y=_devices,
            orientation="h",
            marker=dict(color=_colors),
            text=[f"{bw:.1f} GB/s" for bw in _bandwidths],
            textposition="outside",
        )
    )

    _fig.update_layout(
        title="Hardware Memory Bandwidth Comparison (Roofline Ceiling for DataFrames)",
        xaxis_title="Peak Memory Bandwidth (GB/s) - Log Scale",
        xaxis_type="log",
        yaxis_title="Architecture",
        template="plotly_white",
        height=380,
        margin=dict(l=180, r=60, t=60, b=60),
    )

    _md = mo.md(r"""
    ### Hardware Memory Bandwidth Comparison

    Notice that PCIe bandwidth is lower than Host DDR5 memory bandwidth. This creates a critical tradeoff governed by **Amdahl's Law**.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Amdahl's Law and the PCIe Transfer Bottleneck

    When transferring a dataset of size $S$ bytes from Host RAM to GPU VRAM across the PCIe bus, the total execution time $T_{\text{GPU}}$ is:

    $$
    T_{\text{GPU}} = t_{\text{H2D}} + t_{\text{kernel}} + t_{\text{D2H}} = \frac{S}{B_{\text{PCIe}}} + \frac{S}{B_{\text{GPU}}} + \frac{S_{\text{out}}}{B_{\text{PCIe}}}
    $$

    where $t_{\text{H2D}}$ is Host-to-Device transfer time, $t_{\text{kernel}}$ is GPU computation time, and $t_{\text{D2H}}$ is Device-to-Host transfer time.

    ### Break-Even Criterion

    For GPU acceleration to achieve a net speedup over native CPU execution ($T_{\text{GPU}} < T_{\text{CPU}}$):

    $$
    \frac{S}{B_{\text{PCIe}}} + \frac{S}{B_{\text{GPU}}} < \frac{S}{B_{\text{CPU}}} \implies \frac{1}{B_{\text{PCIe}}} + \frac{1}{B_{\text{GPU}}} < \frac{1}{B_{\text{CPU}}}
    $$

    If $B_{\text{PCIe}} < B_{\text{CPU}}$ (e.g., PCIe 4.0 at $31.5 \text{ GB/s}$ vs DDR5 at $84 \text{ GB/s}$), **a single isolated query on a small dataframe is slower on GPU** due to transfer overhead!

    GPU dataframe acceleration yields decisive advantages when:
    1. **Data Stays in GPU Memory**: Chained queries, iterative machine learning pipelines (e.g. RAPIDS cuML, XGBoost), or direct GPU Parquet loading via GPUDirect Storage bypass CPU memory completely.
    2. **High Compute/Compression Intensity**: Heavy string regex parsing, complex joins, or sorting where GPU parallel compute dwarfs transfer latency.
    """)


@app.cell
def _(np, pd, pl, time):
    # Benchmark: Pandas (Single-threaded) vs Polars (Multi-threaded Columnar Rust)
    # Scale: N = 500,000 synthetic records
    n_records = 500000
    rng = np.random.default_rng(42)

    ages = rng.integers(18, 80, size=n_records)
    bmis = rng.normal(27.0, 5.0, size=n_records)
    bps = rng.normal(120.0, 15.0, size=n_records)

    # 1. Pandas DataFrame
    df_pd = pd.DataFrame({"age": ages, "bmi": bmis, "bp": bps})

    t0 = time.perf_counter()
    res_pd = df_pd.groupby("age").agg({"bmi": "mean", "bp": "max"}).sort_values(by="bmi")
    t_pandas = (time.perf_counter() - t0) * 1000.0

    # 2. Polars DataFrame (Multi-threaded columnar execution)
    df_pl = pl.DataFrame({"age": ages, "bmi": bmis, "bp": bps})

    t0 = time.perf_counter()
    res_pl = (
        df_pl.group_by("age")
        .agg([pl.col("bmi").mean(), pl.col("bp").max()])
        .sort("bmi")
    )
    t_polars = (time.perf_counter() - t0) * 1000.0

    speedup_polars = t_pandas / t_polars if t_polars > 0 else 1.0

    return (
        ages,
        bmis,
        bps,
        df_pd,
        df_pl,
        n_records,
        res_pd,
        res_pl,
        rng,
        speedup_polars,
        t_pandas,
        t_polars,
        t0,
    )


@app.cell(hide_code=True)
def _(go, mo, n_records, speedup_polars, t_pandas, t_polars):
    _fig = go.Figure()

    _fig.add_trace(
        go.Bar(
            x=["Pandas (Single-Threaded)", "Polars (Multi-Threaded Rust)"],
            y=[t_pandas, t_polars],
            marker_color=["#ef4444", "#3b82f6"],
            text=[f"{t_pandas:.1f} ms", f"{t_polars:.1f} ms"],
            textposition="outside",
        )
    )

    _fig.update_layout(
        title=f"Execution Latency: Groupby Aggregation & Sort (N = {n_records:,} rows)",
        xaxis_title="Dataframe Engine",
        yaxis_title="Execution Time (ms)",
        template="plotly_white",
        height=380,
        margin=dict(l=60, r=40, t=60, b=60),
    )

    _md = mo.md(f"""
    ---

    ## Live Benchmark: Pandas vs Polars

    Tested on **{n_records:,} rows**:
    - **Pandas**: **{t_pandas:.1f} ms**
    - **Polars**: **{t_polars:.1f} ms** (Speedup: **{speedup_polars:.1f}x**)

    Even on CPU, columnar formats (Apache Arrow) combined with vectorized multi-core query execution in Polars eliminate Python interpreter overhead and maximize memory bus utilization.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(HAS_CUDF, mo):
    _cudf_status = (
        "**cuDF is available** in this runtime. You can run `cudf.DataFrame.from_pandas(df)`."
        if HAS_CUDF
        else "**cuDF is not installed / NVIDIA GPU not detected** in this environment. cuDF requires an NVIDIA GPU with CUDA drivers."
    )

    return mo.md(f"""
    ---

    ## cuDF and the RAPIDS Ecosystem

    {_cudf_status}

    ### How `cudf.pandas` Works (Zero-Code-Change Acceleration)

    RAPIDS provides an accelerator extension for Pandas:

    ```python
    # Enable cuDF acceleration for all subsequent pandas code
    %load_ext cudf.pandas
    import pandas as pd

    # This identical pandas code now executes on the GPU
    df = pd.read_parquet("large_dataset.parquet")
    summary = df.groupby("category").agg({"amount": "sum", "latency": "mean"})
    ```

    The `cudf.pandas` proxy engine operates via a **hybrid fallback architecture**:
    1. When an operation is supported by cuDF, data resides in GPU VRAM and executes across thousands of CUDA threads.
    2. If an unsupported Pandas edge case is encountered, `cudf.pandas` automatically copies the required slice back to CPU memory, executes standard Pandas, and returns control without throwing an exception.
    """)


@app.cell(hide_code=True)
def _(go, mo, np):
    # Simulated scaling curves: Pandas vs Polars vs cuDF (GPU)
    _sizes = np.array([10_000, 50_000, 200_000, 1_000_000, 5_000_000, 20_000_000])

    # Estimated latencies (ms) based on empirical benchmarks and memory bandwidth modeling
    _lat_pandas = _sizes * 0.00018 + 5.0
    _lat_polars = _sizes * 0.000025 + 2.0
    # cuDF has ~15ms base PCIe transfer and launch latency, but scales with slope 1/20th of Polars
    _lat_cudf = _sizes * 0.0000015 + 12.0

    _fig = go.Figure()

    _fig.add_trace(
        go.Scatter(
            x=_sizes,
            y=_lat_pandas,
            mode="lines+markers",
            name="Pandas (CPU Single-Thread)",
            line=dict(color="#ef4444", width=2.5),
            marker=dict(size=7),
            hovertemplate="Rows: %{x:,}<br>Latency: %{y:.1f} ms<extra></extra>",
        )
    )

    _fig.add_trace(
        go.Scatter(
            x=_sizes,
            y=_lat_polars,
            mode="lines+markers",
            name="Polars (CPU Multi-Thread)",
            line=dict(color="#3b82f6", width=2.5),
            marker=dict(size=7),
            hovertemplate="Rows: %{x:,}<br>Latency: %{y:.1f} ms<extra></extra>",
        )
    )

    _fig.add_trace(
        go.Scatter(
            x=_sizes,
            y=_lat_cudf,
            mode="lines+markers",
            name="cuDF (GPU Accelerated)",
            line=dict(color="#10b981", width=2.5, dash="solid"),
            marker=dict(size=7),
            hovertemplate="Rows: %{x:,}<br>Latency: %{y:.1f} ms<extra></extra>",
        )
    )

    _fig.update_layout(
        title="Throughput Scaling: End-to-End Latency vs Row Count",
        xaxis_title="Number of Rows",
        yaxis_title="Latency (ms) - Log Scale",
        xaxis_type="log",
        yaxis_type="log",
        template="plotly_white",
        height=420,
        margin=dict(l=60, r=40, t=60, b=60),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.8)"),
    )

    _md = mo.md(r"""
    ### Scaling Regimes Across DataFrame Architectures

    1. **Small Datasets ($N < 10^5$ rows)**: CPU Polars or Pandas is fastest. GPU kernel launch overhead and PCIe bus transfers dominate.
    2. **Medium Datasets ($10^5 < N < 10^7$ rows)**: Polars provides outstanding multi-threaded CPU performance without requiring specialized GPU hardware.
    3. **Large Datasets ($N > 10^7$ rows)**: cuDF and GPU acceleration dominate, delivering order-of-magnitude faster queries by fully saturating terabyte-per-second memory buses.
    """)

    return mo.vstack([_md, _fig])


@app.cell(hide_code=True)
def _(mo):
    return mo.md(r"""
    ---

    ## Key Takeaways and Architectural Guide

    1. **DataFrames are Memory Bound**: Aggregation speed is governed by memory bandwidth, not raw CPU clock speed or floating-point capability.
    2. **Columnar Memory Layout**: Systems built on Apache Arrow (Polars, cuDF, DuckDB) avoid Python pointer-chasing and leverage SIMD vectorization.
    3. **Mind the PCIe Bus**: Keep data on the GPU across multiple transformation steps to amortize Host-to-Device copy costs.
    4. **Tool Selection Strategy**:
       - Standard exploratory analysis ($< 1 \text{ GB}$): **Pandas** or **Polars**.
       - Medium-to-large pipelines ($1\text{--}100 \text{ GB}$): **Polars** or **DuckDB**.
       - Massive GPU-resident pipelines, feature engineering for cuML/PyTorch: **cuDF / RAPIDS**.
    """)


if __name__ == "__main__":
    app.run()
