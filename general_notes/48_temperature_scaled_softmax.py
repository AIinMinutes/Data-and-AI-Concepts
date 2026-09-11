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
    from scipy.optimize import minimize_scalar

    return go, make_subplots, minimize_scalar, mo, np, pd


@app.cell
def _(mo):
    mo.md(
        r"""
        [← 47 GELU Activation](47_gelu.py) | [Index](../index.html) | [49 Focal Loss Balanced →](49_focal_loss_balanced.py)

        # Temperature-Scaled Softmax: Boltzmann Distributions, Model Calibration, and LLM Decoding Dynamics

        ## [a] Why do you need to know these concepts?

        Every generative sequence model—from Large Language Models (such as GPT-4, Claude, Gemini, and LLaMA) predicting next-token probability distributions over vocabulary sizes exceeding $100,000$ to reinforcement learning agents selecting discrete actions—outputs an unnormalized real-valued logits vector $z \in \mathbb{R}^V$.

        #### The Role of Temperature in Generative Decoding
        In standard inference, applying the unmodified Softmax function ($T = 1.0$) converts raw logits directly into probabilities. However, direct sampling under $T = 1.0$ often yields either repetitive sequences or wild hallucinations depending on the sharpness of the model's training distribution.

        Introducing a positive scalar parameter $T > 0$ (the **temperature**) provides a continuous control mechanism over the entropy of the predicted distribution:
        - **Low Temperature ($T \to 0$)**: The distribution sharpens into a near-degenerate spike on the single highest logit. As $T \to 0^+$, sampling converges to greedy deterministic argmax decoding ($\lim_{T \to 0^+} p_{\text{max}} = 1$). Ideal for factual question-answering, code synthesis, and structured JSON generation.
        - **Standard Temperature ($T = 1.0$)**: Preserves the native calibrated log-odds output by the model.
        - **High Temperature ($T > 1.0$)**: Compresses differences between logits, flattening the distribution toward a uniform distribution. Higher temperatures increase Shannon entropy, introducing diversity and unexpected vocabulary selections for creative writing and brainstorming.

        #### Neural Network Probability Calibration (Guo et al., 2017)
        Modern deep neural networks with batch normalization, residual connections, and massive parameter counts are notoriously **overconfident**. A vision or classification model might output a predicted probability of $98\%$ on examples where its true empirical accuracy is only $75\%$.

        Post-hoc **Temperature Scaling** provides a simple method for probability calibration: by optimizing a single scalar parameter $T > 0$ to minimize validation negative log-likelihood (NLL), the network's confidence is aligned with empirical frequency without altering classification accuracy or top-1 predictions.
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Thermodynamic Mechanics

        ### 1. The Temperature-Scaled Softmax Function

        Let $z = (z_1, z_2, \dots, z_n)^\top \in \mathbb{R}^n$ denote an unnormalized vector of logits. For any temperature $T > 0$, the temperature-scaled softmax operator $\sigma_T: \mathbb{R}^n \to \Delta^{n-1}$ is defined as:

        $$p_i(T) = \sigma_T(z)_i = \frac{\exp\left( \frac{z_i}{T} \right)}{\sum_{j=1}^n \exp\left( \frac{z_j}{T} \right)}, \quad \forall i \in \{1, \dots, n\}$$

        For numerical stability, subtracting the maximum logit prevents floating-point overflow:

        $$p_i(T) = \frac{\exp\left( \frac{z_i - \max_k z_k}{T} \right)}{\sum_{j=1}^n \exp\left( \frac{z_j - \max_k z_k}{T} \right)}$$

        ### 2. Connection to Statistical Physics (Boltzmann Distribution)

        The temperature-scaled softmax is equivalent to the **Boltzmann (Gibbs) distribution** from thermodynamics. In statistical mechanics, the probability of a physical system occupying an energy state $E_i$ at thermodynamic temperature $\tau = k_B T$ is:

        $$P(E_i) = \frac{e^{-E_i / (k_B T)}}{\mathcal{Z}}, \quad \text{where } \mathcal{Z} = \sum_{j=1}^n e^{-E_j / (k_B T)} \text{ is the partition function}$$

        Setting energy to negative log-odds ($E_i = -z_i$) establishes direct mathematical equivalence: logits represent negative energy states, and the denominator represents the thermodynamic partition function.

        ### 3. Limiting Properties

        #### Zero Temperature Limit ($T \to 0^+$, Argmax / Greedy)
        Let $\mathcal{M} = \arg\max_{k} z_k$ denote the set of maximal logit indices. Factoring out the maximum logit $z_{\max}$:

        $$\lim_{T \to 0^+} p_i(T) = \begin{cases} \frac{1}{|\mathcal{M}|} & \text{if } i \in \mathcal{M} \\ 0 & \text{if } z_i < z_{\max} \end{cases}$$

        When the maximal logit is unique ($|\mathcal{M}| = 1$), the distribution collapses to a Kronecker delta centered on the argmax.

        #### Infinite Temperature Limit ($T \to \infty$, Uniform Distribution)
        As $T \to \infty$, the exponent approaches zero: $\lim_{T \to \infty} \frac{z_i}{T} = 0$. Therefore:

        $$\lim_{T \to \infty} p_i(T) = \frac{e^0}{\sum_{j=1}^n e^0} = \frac{1}{n}$$

        ### 4. Entropy Dynamics and Partial Derivatives

        The Shannon entropy of the scaled distribution in bits is:

        $$H(p(T)) = -\sum_{i=1}^n p_i(T) \log_2 p_i(T)$$

        For any non-degenerate logit vector, $H(p(T))$ is a **strictly monotonically increasing function** of temperature $T$:

        $$\lim_{T \to 0^+} H(p(T)) = 0 \text{ bits}, \quad \lim_{T \to \infty} H(p(T)) = \log_2(n) \text{ bits}$$

        Taking the partial derivative of probability $p_i$ with respect to temperature:

        $$\frac{\partial p_i}{\partial T} = -\frac{p_i}{T^2} \left( z_i - \sum_{j=1}^n p_j z_j \right) = -\frac{p_i}{T^2} \left( z_i - \mathbb{E}_p[z] \right)$$

        - If token $i$ has an above-average logit ($z_i > \mathbb{E}_p[z]$), its probability **decreases** as $T$ increases ($\frac{\partial p_i}{\partial T} < 0$).
        - If token $i$ has a below-average logit ($z_i < \mathbb{E}_p[z]$), its probability **increases** as $T$ increases ($\frac{\partial p_i}{\partial T} > 0$).
        """
    )
    return


@app.cell
def _(np):
    # Candidate tokens simulating next-token completion for prompt: "The capital of France is"
    token_vocab = ["Paris", "Lyon", "Marseille", "Europe", "London"]
    # Logits reflecting high model confidence in "Paris"
    logits_arr = np.array([4.8, 2.1, 1.2, -0.4, -1.8])

    # Temperature grid for entropy trajectory
    temp_grid = np.linspace(0.08, 4.0, 150)

    def stable_temp_softmax(logits, temp):
        scaled = (logits - np.max(logits)) / temp
        exp_z = np.exp(scaled)
        return exp_z / np.sum(exp_z)

    # Compute entropy curve across temperature range
    entropy_trajectory = []
    prob_trajectories = {tok: [] for tok in token_vocab}

    for _t in temp_grid:
        _p = stable_temp_softmax(logits_arr, _t)
        # Shannon entropy in bits
        _ent = -np.sum(_p * np.log2(np.maximum(_p, 1e-15)))
        entropy_trajectory.append(_ent)
        for _idx, _tok in enumerate(token_vocab):
            prob_trajectories[_tok].append(_p[_idx])

    entropy_trajectory = np.array(entropy_trajectory)
    max_theoretical_entropy = float(np.log2(len(token_vocab)))

    # Selected discrete temperatures for comparison
    demo_temps = [0.25, 0.7, 1.0, 2.5]
    demo_distributions = {_t: stable_temp_softmax(logits_arr, _t) for _t in demo_temps}

    return (
        demo_distributions,
        demo_temps,
        entropy_trajectory,
        logits_arr,
        max_theoretical_entropy,
        prob_trajectories,
        stable_temp_softmax,
        temp_grid,
        token_vocab,
    )


@app.cell
def _(
    demo_distributions,
    demo_temps,
    entropy_trajectory,
    go,
    make_subplots,
    max_theoretical_entropy,
    mo,
    temp_grid,
    token_vocab,
):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Token Probability Distributions Across Temperature Levels</b>",
            "<b>Shannon Entropy vs Temperature (Creativity Curve)</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Bar chart comparing temperatures
    temp_colors = {0.25: "#1D4ED8", 0.7: "#2563EB", 1.0: "#8B5CF6", 2.5: "#EC4899"}

    for _t in demo_temps:
        fig.add_trace(
            go.Bar(
                x=token_vocab,
                y=demo_distributions[_t],
                name=f"T = {_t}",
                marker_color=temp_colors[_t],
            ),
            row=1,
            col=1,
        )

    # Panel 2: Entropy Trajectory
    fig.add_trace(
        go.Scatter(
            x=temp_grid,
            y=entropy_trajectory,
            mode="lines",
            line=dict(color="#2563EB", width=2.5),
            name="Shannon Entropy H(T)",
        ),
        row=1,
        col=2,
    )

    # Add theoretical maximum entropy line
    fig.add_hline(
        y=max_theoretical_entropy,
        line=dict(color="#DC2626", width=1.5, dash="dash"),
        annotation_text=f"Max Entropy log2(5) = {max_theoretical_entropy:.2f} bits",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Candidate Next Token", row=1, col=1)
    fig.update_yaxes(title_text="Probability P(Token)", range=[0, 1.05], row=1, col=1)
    fig.update_xaxes(title_text="Temperature Parameter T", row=1, col=2)
    fig.update_yaxes(title_text="Shannon Entropy (Bits)", range=[0, 2.5], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=500,
        margin=dict(l=40, r=40, t=70, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        barmode="group",
    )

    viz = mo.ui.plotly(fig)
    return fig, temp_colors, viz


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below captures the impact of temperature scaling on probability mass and information entropy:

                1. **Left Panel (Distribution Modulation)**: At $T = 0.25$ (dark blue), the model assigns $>99\%$ probability to the top token ("Paris"), suppressing all alternatives. As temperature rises to $T = 2.5$ (pink), probability mass redistributes evenly across secondary candidates ("Lyon", "Marseille", "Europe").
                2. **Right Panel (Information Entropy Trajectory)**: Entropy starts at $0$ bits near $T = 0$ (perfect certainty) and climbs monotonically toward the theoretical maximum $\log_2(5) \approx 2.32$ bits as $T \to \infty$ (complete randomness).
                """
            ),
            viz,
        ]
    )


@app.cell
def _(
    demo_distributions,
    demo_temps,
    logits_arr,
    minimize_scalar,
    mo,
    np,
    pd,
    stable_temp_softmax,
    token_vocab,
):
    # Example 1: Pure NumPy Temperature-Scaled Softmax and Entropy Table
    entropy_records = []
    for _t in demo_temps:
        _probs = demo_distributions[_t]
        _ent = -np.sum(_probs * np.log2(np.maximum(_probs, 1e-15)))
        entropy_records.append(
            {
                "Temperature_T": _t,
                "Top_Token_Prob (Paris)": f"{_probs[0] * 100:.2f}%",
                "Second_Token_Prob (Lyon)": f"{_probs[1] * 100:.2f}%",
                "Lowest_Token_Prob (London)": f"{_probs[4] * 100:.2f}%",
                "Shannon_Entropy_Bits": round(_ent, 3),
                "Decoding_Profile": (
                    "Deterministic / Greedy"
                    if _t < 0.5
                    else ("Balanced / Recommended" if _t <= 1.0 else "High Entropy / Creative")
                ),
            }
        )

    df_entropy = pd.DataFrame(entropy_records)

    # Example 2: Multinomial Sampling Simulation (10,000 Generation Trials)
    np.random.seed(42)
    _n_trials = 10000

    sim_records = []
    for _sim_t in [0.3, 1.0, 2.0]:
        _probs = stable_temp_softmax(logits_arr, _sim_t)
        _sampled_indices = np.random.choice(len(token_vocab), size=_n_trials, p=_probs)
        _counts = np.bincount(_sampled_indices, minlength=len(token_vocab))

        sim_records.append(
            {
                "Temperature": f"T = {_sim_t}",
                "Paris_Count": f"{_counts[0]} ({_counts[0] / _n_trials * 100:.1f}%)",
                "Lyon_Count": f"{_counts[1]} ({_counts[1] / _n_trials * 100:.1f}%)",
                "Marseille_Count": f"{_counts[2]} ({_counts[2] / _n_trials * 100:.1f}%)",
                "Europe_Count": f"{_counts[3]} ({_counts[3] / _n_trials * 100:.1f}%)",
                "London_Count": f"{_counts[4]} ({_counts[4] / _n_trials * 100:.1f}%)",
            }
        )

    df_sim = pd.DataFrame(sim_records)

    # Example 3: Platt Temperature Scaling for Overconfident Neural Networks
    # Simulate an overconfident validation set of 200 samples
    np.random.seed(42)
    _n_val = 200
    # Overconfident raw logits: large magnitude causing uncalibrated 99% confidence
    raw_val_logits = np.random.normal(0, 1, (_n_val, 2))
    raw_val_logits[:, 1] += np.random.choice([-3.5, 3.5], size=_n_val)  # Extreme margins
    val_labels = (raw_val_logits[:, 1] > 0).astype(int)
    # Introduce label noise: 15% random flips so true accuracy is ~85%, while confidence is ~98%
    noise_mask = np.random.uniform(0, 1, _n_val) < 0.15
    val_labels[noise_mask] = 1 - val_labels[noise_mask]

    # Uncalibrated baseline (T = 1.0)
    uncal_probs = stable_temp_softmax(raw_val_logits.T, 1.0).T
    uncal_conf = np.max(uncal_probs, axis=1)
    uncal_acc = np.mean(np.argmax(uncal_probs, axis=1) == val_labels)

    # Loss function for temperature calibration: Negative Log-Likelihood
    def nll_cost(temp):
        probs_t = stable_temp_softmax(raw_val_logits.T, temp).T
        correct_probs = probs_t[np.arange(_n_val), val_labels]
        return -np.mean(np.log(np.maximum(correct_probs, 1e-15)))

    opt_res = minimize_scalar(nll_cost, bounds=(0.1, 10.0), method="bounded")
    best_temp = float(opt_res.x)

    # Calibrated probabilities under optimal T*
    cal_probs = stable_temp_softmax(raw_val_logits.T, best_temp).T
    cal_conf = np.max(cal_probs, axis=1)
    cal_acc = np.mean(np.argmax(cal_probs, axis=1) == val_labels)

    df_calibration = pd.DataFrame(
        [
            {
                "Model_State": "Uncalibrated (Default T = 1.0)",
                "Optimal_Temperature_T": "1.00",
                "Mean_Model_Confidence": f"{np.mean(uncal_conf) * 100:.2f}%",
                "Actual_Validation_Accuracy": f"{uncal_acc * 100:.2f}%",
                "Confidence_Calibration_Gap": f"{abs(np.mean(uncal_conf) - uncal_acc) * 100:.2f}%",
                "Negative_Log_Likelihood": f"{nll_cost(1.0):.4f}",
            },
            {
                "Model_State": "Temperature Scaled (Calibrated T*)",
                "Optimal_Temperature_T": f"{best_temp:.2f}",
                "Mean_Model_Confidence": f"{np.mean(cal_conf) * 100:.2f}%",
                "Actual_Validation_Accuracy": f"{cal_acc * 100:.2f}%",
                "Confidence_Calibration_Gap": f"{abs(np.mean(cal_conf) - cal_acc) * 100:.2f}%",
                "Negative_Log_Likelihood": f"{nll_cost(best_temp):.4f}",
            },
        ]
    )

    table_entropy = mo.ui.table(df_entropy)
    table_sim = mo.ui.table(df_sim)
    table_cal = mo.ui.table(df_calibration)

    return (
        table_cal,
        table_entropy,
        table_sim,
    )


@app.cell
def _(mo, table_cal, table_entropy, table_sim):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Temperature Scaling and Entropy Quantification

                Evaluating probability concentration and Shannon entropy as a function of temperature:
                """
            ),
            table_entropy,
            mo.md(
                r"""
                ### Example 2: Multinomial Sampling Simulation Across 10,000 Generation Trials

                Observing empirical token frequencies across 10,000 multinomial sampling draws under varying temperature regimes:
                """
            ),
            table_sim,
            mo.md(
                r"""
                ### Example 3: Post-Hoc Temperature Scaling for Neural Network Probability Calibration

                Optimizing $T^*$ via validation negative log-likelihood (NLL) to eliminate overconfidence while preserving exact classification accuracy:
                """
            ),
            table_cal,
        ]
    )


if __name__ == "__main__":
    app.run()
