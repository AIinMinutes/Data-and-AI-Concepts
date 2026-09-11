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
    import torch
    import torch.nn as nn

    return go, make_subplots, mo, nn, np, pd, time, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 52 Multi-Head Attention](52_multi_head_attention.py) | [Index](../index.html) | [54 Decoding Strategies →](54_decoding_strategies.py)

        # 53. Layer Normalization vs RMSNorm: Internal Covariate Shift and Modern Activation Scaling

        ### Executive Summary

        In deep neural architectures, continuous parameter updates during optimization cause the distribution of each layer's inputs to drift throughout training—a phenomenon known as internal covariate shift. While **Batch Normalization** (Ioffe & Szegedy, 2015) mitigated this in convolutional networks by computing mini-batch statistics, it critically degrades in sequential modeling due to dynamic token lengths and inference batch-size variability.

        **Layer Normalization (LayerNorm)** (Ba, Kiros, & Hinton, 2016) resolved this by computing mean and variance across the feature dimension for each individual token. However, modern frontier architectures (LLaMA-3, Mistral, Gemma, DeepSeek) have universally replaced LayerNorm with **Root Mean Square Normalization (RMSNorm)** (Zhang & Sennrich, 2019). RMSNorm demonstrates that the training stabilization of LayerNorm arises almost entirely from scaling invariance rather than mean-centering, enabling a computationally streamlined formulation that cuts memory operations and improves training throughput by up to $50\%$.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Invariance Derivations

        ### 1. Classical Layer Normalization (LayerNorm)

        Let $x \in \mathbb{R}^d$ represent an activation vector for a single token at a given layer. LayerNorm normalizes $x$ across its $d$ hidden dimensions using both the sample mean $\mu$ and sample variance $\sigma^2$:

        $$\mu = \frac{1}{d} \sum_{i=1}^d x_i, \qquad \sigma^2 = \frac{1}{d} \sum_{i=1}^d (x_i - \mu)^2$$

        The normalized representation $\hat{x}$ and final affine-transformed output $y \in \mathbb{R}^d$ are:

        $$\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}, \qquad y_i = \gamma_i \hat{x}_i + \beta_i$$

        where $\epsilon > 0$ is a small numerical stabilization constant (typically $10^{-5}$ or $10^{-6}$), $\gamma \in \mathbb{R}^d$ is a learnable gain vector initialized to $\mathbf{1}$, and $\beta \in \mathbb{R}^d$ is a learnable bias vector initialized to $\mathbf{0}$.

        #### Computational Complexity of LayerNorm:
        LayerNorm requires **two sequential reduction passes** across the $d$-dimensional feature vector:
        1. Pass 1: Sum elements to calculate $\mu = \frac{1}{d}\sum x_i$.
        2. Pass 2: Accumulate squared differences $(x_i - \mu)^2$ to calculate $\sigma^2$.
        On modern GPUs, this introduces global memory synchronization overhead that throttles memory bandwidth.

        ### 2. Root Mean Square Normalization (RMSNorm)

        Zhang & Sennrich (2019) hypothesized that the regularizing power of LayerNorm stems from its **scale invariance** property rather than its shift invariance (mean-centering). By discarding the mean-centering step and the bias parameter $\beta$, RMSNorm scales activations strictly by their root-mean-square magnitude:

        $$\operatorname{RMS}(x) = \sqrt{\frac{1}{d} \sum_{i=1}^d x_i^2 + \epsilon}$$

        The normalized vector $\bar{x}$ and scaled output $y$ are:

        $$\bar{x}_i = \frac{x_i}{\operatorname{RMS}(x)}, \qquad y_i = \gamma_i \bar{x}_i$$

        where $\gamma \in \mathbb{R}^d$ is the sole learnable gain parameter.

        #### Architectural Advantages of RMSNorm:
        1. **Single-Pass Reduction**: Computing $\operatorname{RMS}(x)$ requires only a single pass $\sum_{i=1}^d x_i^2$, completely bypassing the mean calculation and eliminating intermediate GPU synchronizations.
        2. **Reduced Parameter Count**: By eliminating the bias vector $\beta \in \mathbb{R}^d$, RMSNorm saves $d$ parameters per normalization layer. Across an 80-layer model with $d = 8192$ (two norms per Transformer block), this saves over $1.3$ million parameters with zero expressive penalty.
        3. **Fused Kernel Speedup**: Fused CUDA / Triton kernels execute RMSNorm $10\%$ to $50\%$ faster than LayerNorm, yielding significant wall-clock speedups over trillions of training tokens.

        ### 3. Invariance Analysis: Scale vs Shift

        A normalization operator $\mathcal{T}(x)$ is characterized by its mathematical invariance properties:

        #### Scale Invariance ($\mathcal{T}(\alpha x) = \mathcal{T}(x)$ for $\alpha > 0$):
        - **LayerNorm**:
          $$\mu(\alpha x) = \alpha \mu(x), \quad \sigma(\alpha x) = \alpha \sigma(x) \implies \frac{\alpha x_i - \alpha \mu(x)}{\alpha \sigma(x)} = \frac{x_i - \mu(x)}{\sigma(x)} = \hat{x}_i$$
        - **RMSNorm**:
          $$\operatorname{RMS}(\alpha x) = \sqrt{\frac{1}{d}\sum (\alpha x_i)^2} = \alpha \operatorname{RMS}(x) \implies \frac{\alpha x_i}{\alpha \operatorname{RMS}(x)} = \frac{x_i}{\operatorname{RMS}(x)} = \bar{x}_i$$
        Both LayerNorm and RMSNorm exhibit strict scale invariance. If forward activations are scaled by $\alpha$, the normalized activations remain invariant, bounding forward signal propagation and stabilizing backpropagated gradients.

        #### Shift Invariance ($\mathcal{T}(x + c) = \mathcal{T}(x)$ for $c \in \mathbb{R}$):
        - **LayerNorm**:
          $$\mu(x + c) = \mu(x) + c \implies (x_i + c) - (\mu(x) + c) = x_i - \mu(x)$$
          LayerNorm is strictly shift-invariant.
        - **RMSNorm**:
          $$\operatorname{RMS}(x + c) \neq \operatorname{RMS}(x)$$
          RMSNorm is **not** shift-invariant. However, in deep transformers, representations are naturally zero-centered around residual stream additions, rendering shift invariance redundant.
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Simulate an unnormalized activation vector across 32 hidden dimensions
    np.random.seed(42)
    dim_demo = 32
    # Skewed activation with large positive mean shift (simulating uncentered feed-forward output)
    raw_activations = np.random.normal(loc=3.5, scale=2.0, size=dim_demo)
    raw_activations[5] = 12.0  # Outlier spike
    raw_activations[18] = -4.0

    # 1. Pure NumPy LayerNorm
    ln_mean = np.mean(raw_activations)
    ln_var = np.var(raw_activations)
    ln_norm = (raw_activations - ln_mean) / np.sqrt(ln_var + 1e-6)

    # 2. Pure NumPy RMSNorm
    rms_val = np.sqrt(np.mean(raw_activations**2) + 1e-6)
    rms_norm = raw_activations / rms_val

    # Theoretical runtime comparison across dimension sizes (simulated FLOPs / memory reads)
    dim_bench = [256, 512, 1024, 2048, 4096, 8192]
    # Relative memory access passes (LayerNorm = 2 passes + 2 params, RMSNorm = 1 pass + 1 param)
    ln_ops = [d * 4 for d in dim_bench]  # 2 reads, 1 write, 1 affine
    rms_ops = [d * 2.5 for d in dim_bench]  # 1 read, 1 write, 0.5 affine

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Activation Profiles: Raw vs LayerNorm vs RMSNorm</b>",
            "<b>Theoretical Memory Access Passes vs Hidden Dimension</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Activation Comparison
    x_axis = np.arange(1, dim_demo + 1)
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=raw_activations,
            mode="markers+lines",
            line=dict(color="#9CA3AF", dash="dot", width=1.5),
            marker=dict(size=6),
            name="Raw Input (Mean=3.5)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=ln_norm,
            mode="markers+lines",
            line=dict(color="#1D4ED8", width=2.5),
            marker=dict(size=7),
            name="LayerNorm (Mean=0, Std=1)",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=x_axis,
            y=rms_norm,
            mode="markers+lines",
            line=dict(color="#0D9488", width=2.5),
            marker=dict(size=7),
            name="RMSNorm (RMS=1.0)",
        ),
        row=1,
        col=1,
    )

    # Zero line on panel 1
    fig.add_hline(y=0.0, line=dict(color="#6B7280", width=1, dash="dash"), row=1, col=1)

    # Panel 2: Computational / Memory Access Overhead
    fig.add_trace(
        go.Bar(
            x=[str(d) for d in dim_bench],
            y=ln_ops,
            name="LayerNorm Memory Ops",
            marker_color="#1D4ED8",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Bar(
            x=[str(d) for d in dim_bench],
            y=rms_ops,
            name="RMSNorm Memory Ops (37.5% Reduction)",
            marker_color="#0D9488",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Hidden Feature Dimension Index", row=1, col=1)
    fig.update_yaxes(title_text="Activation Value", row=1, col=1)
    fig.update_xaxes(title_text="Hidden Dimension d_model", row=1, col=2)
    fig.update_yaxes(title_text="Relative Memory Operations (kOps)", row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        barmode="group",
    )

    viz = mo.ui.plotly(fig)
    return (
        dim_bench,
        dim_demo,
        fig,
        ln_mean,
        ln_norm,
        ln_ops,
        ln_var,
        raw_activations,
        rms_norm,
        rms_ops,
        rms_val,
        viz,
        x_axis,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates activation normalization and operational efficiency:

                1. **Left Panel (Normalized Activation Curves)**: The raw input (grey dashed line) has a positive baseline shift ($\mu \approx 3.5$). LayerNorm (blue) forces the mean precisely to $0.0$ and unit variance. RMSNorm (teal) scales the vector down to an RMS radius of $1.0$ without enforcing zero-centering. Notice that the relative geometry of peaks and valleys is nearly identical between LayerNorm and RMSNorm.
                2. **Right Panel (Memory Operations Overhead)**: LayerNorm requires two independent sequential reduction passes (computing $\mu$, then $\sigma^2$). RMSNorm cuts this to a single reduction pass, reducing memory operations by $37.5\%$ to $50\%$ across all hidden dimensions.
                """
            ),
            viz,
        ]
    )


@app.cell
def _(mo, nn, np, pd, time, torch):
    # Vectorized NumPy Implementations
    def numpy_layernorm(x, gamma=None, beta=None, eps=1e-6):
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        x_hat = (x - mean) / np.sqrt(var + eps)
        if gamma is not None:
            x_hat = x_hat * gamma
        if beta is not None:
            x_hat = x_hat + beta
        return x_hat

    def numpy_rmsnorm(x, gamma=None, eps=1e-6):
        rms = np.sqrt(np.mean(x**2, axis=-1, keepdims=True) + eps)
        x_hat = x / rms
        if gamma is not None:
            x_hat = x_hat * gamma
        return x_hat

    # Verification 1: Scale Invariance and Shift Invariance Axioms
    np.random.seed(1337)
    _x = np.random.normal(2.0, 3.0, size=(10, 64))
    _scale_factor = 42.5
    _shift_constant = 17.8

    # Scale invariance test
    _ln_base = numpy_layernorm(_x)
    _ln_scaled = numpy_layernorm(_x * _scale_factor)
    _ln_scale_diff = np.max(np.abs(_ln_base - _ln_scaled))

    _rms_base = numpy_rmsnorm(_x)
    _rms_scaled = numpy_rmsnorm(_x * _scale_factor)
    _rms_scale_diff = np.max(np.abs(_rms_base - _rms_scaled))

    # Shift invariance test
    _ln_shifted = numpy_layernorm(_x + _shift_constant)
    _ln_shift_diff = np.max(np.abs(_ln_base - _ln_shifted))

    _rms_shifted = numpy_rmsnorm(_x + _shift_constant)
    _rms_shift_diff = np.max(np.abs(_rms_base - _rms_shifted))

    df_invariance = pd.DataFrame(
        [
            {
                "Mathematical_Property": "Scale Invariance: Norm(alpha * x) == Norm(x)",
                "LayerNorm_Max_Diff": f"{_ln_scale_diff:.8e}",
                "RMSNorm_Max_Diff": f"{_rms_scale_diff:.8e}",
                "Theoretical_Status": "Both Strictly Scale Invariant",
            },
            {
                "Mathematical_Property": "Shift Invariance: Norm(x + c) == Norm(x)",
                "LayerNorm_Max_Diff": f"{_ln_shift_diff:.8e}",
                "RMSNorm_Max_Diff": f"{_rms_shift_diff:.4f}",
                "Theoretical_Status": "LayerNorm Invariant | RMSNorm Non-Invariant (By Design)",
            },
            {
                "Mathematical_Property": "Mean-Centering Property: Mean(Norm(x)) == 0",
                "LayerNorm_Max_Diff": f"{np.max(np.abs(np.mean(_ln_base, axis=-1))):.8e}",
                "RMSNorm_Max_Diff": f"{np.max(np.abs(np.mean(_rms_base, axis=-1))):.4f}",
                "Theoretical_Status": "LayerNorm Exactly 0 | RMSNorm Inherits Scaled Mean",
            },
        ]
    )

    # Verification 2: PyTorch Modules and Gradient Verification
    class PyTorchRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            norm_x = torch.mean(x * x, dim=-1, keepdim=True)
            x_normed = x * torch.rsqrt(norm_x + self.eps)
            return self.weight * x_normed

    torch.manual_seed(42)
    _d = 128
    pt_ln = nn.LayerNorm(_d)
    pt_rms = PyTorchRMSNorm(_d)

    pt_x = torch.randn(4, 16, _d, requires_grad=True)

    # Run PyTorch modules
    out_ln = pt_ln(pt_x)
    loss_ln = out_ln.sum()
    loss_ln.backward()
    grad_ln_norm = pt_x.grad.norm().item()

    pt_x.grad.zero_()
    out_rms = pt_rms(pt_x)
    loss_rms = out_rms.sum()
    loss_rms.backward()
    grad_rms_norm = pt_x.grad.norm().item()

    df_modules = pd.DataFrame(
        [
            {
                "Normalization_Layer": "PyTorch LayerNorm (nn.LayerNorm)",
                "Parameters_per_Layer": f"{2 * _d} (gamma + beta)",
                "Requires_Bias_Vector": "Yes (beta in R^d)",
                "Backprop_Grad_Norm": f"{grad_ln_norm:.4f}",
                "Reduction_Passes": "2 passes (Mean and Variance)",
            },
            {
                "Normalization_Layer": "PyTorch RMSNorm (Modern LLM)",
                "Parameters_per_Layer": f"{_d} (gamma only)",
                "Requires_Bias_Vector": "No (Zero Bias)",
                "Backprop_Grad_Norm": f"{grad_rms_norm:.4f}",
                "Reduction_Passes": "1 pass (Mean Square)",
            },
        ]
    )

    # Verification 3: Empirical Forward-Pass Latency Benchmark
    # Simulate LLM tensor: Batch=8, Sequence Length=128, Hidden Dim=1024
    _b_test, _s_test, _d_test = 8, 128, 1024
    bench_x = torch.randn(_b_test, _s_test, _d_test)
    bench_ln = nn.LayerNorm(_d_test)
    bench_rms = PyTorchRMSNorm(_d_test)

    # Warmup
    for _ in range(50):
        _ = bench_ln(bench_x)
        _ = bench_rms(bench_x)

    # Benchmark 500 iterations
    n_iters = 500
    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = bench_ln(bench_x)
    t_ln = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    for _ in range(n_iters):
        _ = bench_rms(bench_x)
    t_rms = (time.perf_counter() - t0) * 1000.0

    speedup = (t_ln - t_rms) / t_ln * 100.0

    df_benchmark = pd.DataFrame(
        [
            {
                "Normalization_Type": "Standard LayerNorm",
                "Tensor_Dimension": f"({_b_test}, {_s_test}, {_d_test})",
                "Total_500_Iter_Time": f"{t_ln:.2f} ms",
                "Mean_Latency_per_Pass": f"{t_ln / n_iters * 1000:.2f} us",
                "Speedup_vs_LayerNorm": "Baseline (1.0x)",
            },
            {
                "Normalization_Type": "Streamlined RMSNorm",
                "Tensor_Dimension": f"({_b_test}, {_s_test}, {_d_test})",
                "Total_500_Iter_Time": f"{t_rms:.2f} ms",
                "Mean_Latency_per_Pass": f"{t_rms / n_iters * 1000:.2f} us",
                "Speedup_vs_LayerNorm": f"{speedup:.1f}% faster",
            },
        ]
    )

    table_invariance = mo.ui.table(df_invariance)
    table_modules = mo.ui.table(df_modules)
    table_bench = mo.ui.table(df_benchmark)

    return (
        table_bench,
        table_invariance,
        table_modules,
    )


@app.cell
def _(mo, table_bench, table_invariance, table_modules):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Pure NumPy Invariance Verification

                Validating Scale and Shift Invariance properties across LayerNorm vs RMSNorm:
                """
            ),
            table_invariance,
            mo.md(
                r"""
                ### Example 2: PyTorch Module Architecture and Parameter Comparison

                Comparing parameter footprints and gradient flow in modern LLM tensor blocks:
                """
            ),
            table_modules,
            mo.md(
                r"""
                ### Example 3: Forward Pass Latency Benchmark Across 500 Iterations

                Measuring wall-clock forward pass execution time on standard Transformer dimensions:
                """
            ),
            table_bench,
        ]
    )


if __name__ == "__main__":
    app.run()
