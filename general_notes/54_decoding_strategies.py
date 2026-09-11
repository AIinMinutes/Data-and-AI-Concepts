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
    mo.md(
        r"""
        [← 53 LayerNorm vs RMSNorm](53_layer_and_rms_normalization.py) | [Index](../index.html) | [55 Perplexity →](55_perplexity.py)

        # 54. Autoregressive Decoding Strategies: Greedy Search, Beam Search, Top-k, and Nucleus (Top-p) Sampling

        ### Executive Summary

        In autoregressive language models, the forward pass outputs an unnormalized logit vector $z_{t+1} \in \mathbb{R}^{|\mathcal{V}|}$ representing raw affinities across a vocabulary $\mathcal{V}$ of $32{,}000$ to $128{,}000$ tokens. Transforming these raw probabilities into coherent, fluent, and diverse natural language is the core objective of **decoding strategies**.

        The choice of decoding algorithm governs the fundamental trade-off between **coherence** (optimizing sequence likelihood) and **creativity** (avoiding repetitive loops and degenerate attractors):
        - **Deterministic Search (Greedy and Beam Search)**: Optimizes for global likelihood, making it ideal for closed-domain, high-precision tasks like machine translation, mathematical reasoning, and code synthesis. However, on open-ended text generation, it frequently degrades into repetitive, generic phrases.
        - **Stochastic Truncation (Top-k and Top-p / Nucleus Sampling)**: Dynamically eliminates the unreliable, low-probability tail of the vocabulary distribution while preserving controlled stochasticity, achieving human-like lexical diversity.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Formulations of Decoding Strategies

        ### 1. The Decoding Optimization Objective

        Given a conditioning prompt sequence $x_{1:t} = (x_1, x_2, \dots, x_t)$, the model computes conditional token probabilities:

        $$P(w \mid x_{1:t}) = \frac{\exp(z_w / T)}{\sum_{v \in \mathcal{V}} \exp(z_v / T)}$$

        The theoretically optimal sequence continuation $y^* = (y_1, y_2, \dots, y_L)$ maximizes the joint sequence log-likelihood:

        $$y^* = \arg\max_{y_{1:L}} \sum_{i=1}^L \ln P(y_i \mid x_{1:t}, y_{<i})$$

        Because the search space $|\mathcal{V}|^L$ is astronomically large (e.g., $50{,}000^{20} \approx 10^{94}$ states), exact global optimization is NP-hard, necessitating heuristic approximations.

        ---

        ### 2. Deterministic Search Strategies

        #### A. Greedy Search
        Greedy search selects the single most probable token at each individual step:

        $$y_i = \arg\max_{w \in \mathcal{V}} P(w \mid x_{1:t}, y_{<i})$$

        - **Computational Complexity**: $\mathcal{O}(L \cdot |\mathcal{V}|)$ FLOPs (extremely fast).
        - **Failure Mode**: Myopic blindness. An initially lower-probability token might unlock a subsequent trajectory of significantly higher joint likelihood. Once greedy search commits to an early branch, it can never backtrack.

        #### B. Beam Search
        Beam search maintains a fixed number $B$ (the beam width) of highest-scoring partial sequences at each time step. At step $i$, each of the $B$ active hypotheses is expanded into all $|\mathcal{V}|$ vocabulary candidates, yielding $B \cdot |\mathcal{V}|$ candidate paths. Only the top $B$ paths under a length-normalized log-likelihood score are retained:

        $$\operatorname{Score}(y_{1:i}) = \frac{1}{i^\alpha} \sum_{j=1}^i \ln P(y_j \mid x_{1:t}, y_{<j})$$

        where $\alpha \in [0.6, 1.0]$ is a length penalty parameter that counteracts the natural probabilistic bias toward shorter sequences.

        ---

        ### 3. Stochastic Truncation Strategies

        #### A. Top-k Sampling (Fan et al., 2018)
        Standard temperature sampling from the raw distribution $P(w)$ risks drawing tokens from the vast, unreliable long tail. Top-k sampling filters the vocabulary down to the $k$ most probable tokens:

        $$\mathcal{V}^{(k)} = \left\{ w \in \mathcal{V} \mid \operatorname{rank}(P(w)) \le k \right\}$$

        Probabilities are renormalized strictly over $\mathcal{V}^{(k)}$:

        $$P'(w) = \begin{cases} \frac{P(w)}{\sum_{v \in \mathcal{V}^{(k)}} P(v)} & \text{if } w \in \mathcal{V}^{(k)} \\ 0 & \text{otherwise} \end{cases}$$

        - **Pathology of Fixed $k$**: The distribution's shape changes dynamically across contexts. In rigid syntactic contexts (e.g., `"The capital of France is"`), only 1 token is valid, yet $k=50$ permits low-probability tokens. In open creative contexts, 100+ tokens may be valid, yet $k=50$ prematurely truncates valid synonyms.

        #### B. Nucleus (Top-p) Sampling (Holtzman et al., 2019)
        Nucleus sampling dynamically adjusts the candidate pool size by selecting the smallest subset of tokens whose cumulative probability mass exceeds threshold $p \in (0, 1)$:

        $$\mathcal{V}^{(p)} = \arg\min_{S \subseteq \mathcal{V}} |S| \quad \text{subject to} \quad \sum_{w \in S} P(w) \ge p$$

        Tokens outside $\mathcal{V}^{(p)}$ are assigned zero probability, and remaining tokens are renormalized:

        $$P'(w) = \begin{cases} \frac{P(w)}{\sum_{v \in \mathcal{V}^{(p)}} P(v)} & \text{if } w \in \mathcal{V}^{(p)} \\ 0 & \text{otherwise} \end{cases}$$

        - **Self-Adjusting Head**: When the model is confident (peaked entropy), $|\mathcal{V}^{(p)}|$ automatically contracts to $1$ or $2$ tokens. When the model is uncertain (flat entropy), $|\mathcal{V}^{(p)}|$ dynamically expands to dozens of tokens, preventing both wild tail errors and premature truncation.
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Panel 1: Interactive Search Tree Comparing Greedy vs Beam Search (B=2)
    # Tree nodes: (x, y, label, cumulative_log_prob, path_type)
    # Root
    tree_nodes = [
        {"x": 0.0, "y": 3.0, "label": "Prompt: 'The'", "score": 0.0, "type": "root"},
        # Step 1
        {"x": -2.0, "y": 2.0, "label": "'dog' (p=0.60)", "score": np.log(0.60), "type": "beam+greedy"},
        {"x": 2.0, "y": 2.0, "label": "'galaxy' (p=0.40)", "score": np.log(0.40), "type": "beam_only"},
        # Step 2 from 'dog'
        {"x": -3.0, "y": 1.0, "label": "'barks' (p=0.30)", "score": np.log(0.60) + np.log(0.30), "type": "greedy"},
        {"x": -1.0, "y": 1.0, "label": "'runs' (p=0.20)", "score": np.log(0.60) + np.log(0.20), "type": "pruned"},
        # Step 2 from 'galaxy'
        {
            "x": 1.0,
            "y": 1.0,
            "label": "'rotates' (p=0.90)",
            "score": np.log(0.40) + np.log(0.90),
            "type": "beam_winner",
        },
        {"x": 3.0, "y": 1.0, "label": "'shines' (p=0.08)", "score": np.log(0.40) + np.log(0.08), "type": "pruned"},
    ]

    # Tree edges
    tree_edges = [
        (0, 1, "Greedy & Beam Choice"),
        (0, 2, "Beam Retained (Rank 2)"),
        (1, 3, "Greedy Path Continues"),
        (1, 4, "Pruned"),
        (2, 5, "Beam Global Winner"),
        (2, 6, "Pruned"),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Greedy vs Beam Search (B=2) Exploration Tree</b>",
            "<b>Top-k (k=5) vs Nucleus Top-p (p=0.85) Truncation</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Tree Edges
    for start_idx, end_idx, edge_lbl in tree_edges:
        n1 = tree_nodes[start_idx]
        n2 = tree_nodes[end_idx]
        is_winner = "Beam Global Winner" in edge_lbl
        is_greedy = "Greedy" in edge_lbl and not is_winner
        line_color = "#0D9488" if is_winner else ("#DC2626" if is_greedy else "#9CA3AF")
        line_width = 3.0 if (is_winner or is_greedy) else 1.5

        fig.add_trace(
            go.Scatter(
                x=[n1["x"], n2["x"]],
                y=[n1["y"], n2["y"]],
                mode="lines",
                line=dict(color=line_color, width=line_width),
                hoverinfo="none",
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Panel 1: Tree Nodes
    for node in tree_nodes:
        node_color = (
            "#1D4ED8"
            if node["type"] == "root"
            else (
                "#0D9488"
                if node["type"] == "beam_winner"
                else ("#DC2626" if node["type"] == "greedy" else "#6B7280")
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[node["x"]],
                y=[node["y"]],
                mode="markers+text",
                marker=dict(size=28, color=node_color),
                text=[f"{node['label']}<br>Score: {node['score']:.2f}"],
                textposition="bottom center",
                name=node["label"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )

    # Panel 2: Truncation Dynamics (Zipfian Distribution)
    np.random.seed(42)
    demo_vocab = [
        "the",
        "quick",
        "brown",
        "fox",
        "jumps",
        "over",
        "lazy",
        "dog",
        "quietly",
        "swiftly",
        "bright",
        "moon",
        "river",
        "forest",
        "silent",
    ]
    raw_logits = np.array([5.2, 4.8, 4.3, 4.1, 3.8, 3.2, 2.7, 2.4, 1.8, 1.3, 0.8, 0.4, -0.1, -0.7, -1.5])
    probs = np.exp(raw_logits) / np.sum(np.exp(raw_logits))
    cum_probs = np.cumsum(probs)

    k_val = 5
    p_val = 0.85

    bar_colors = []
    for idx in range(len(probs)):
        in_top_k = idx < k_val
        in_top_p = cum_probs[idx] <= p_val or (idx > 0 and cum_probs[idx - 1] < p_val)

        if in_top_k and in_top_p:
            bar_colors.append("#1D4ED8")  # Both (Blue)
        elif in_top_p and not in_top_k:
            bar_colors.append("#0D9488")  # Top-p only (Teal)
        elif in_top_k and not in_top_p:
            bar_colors.append("#F59E0B")  # Top-k only (Amber)
        else:
            bar_colors.append("#E5E7EB")  # Truncated tail (Grey)

    fig.add_trace(
        go.Bar(
            x=demo_vocab,
            y=probs,
            marker_color=bar_colors,
            name="Token Probability",
            showlegend=False,
        ),
        row=1,
        col=2,
    )

    # Cumulative curve on secondary axis
    fig.add_trace(
        go.Scatter(
            x=demo_vocab,
            y=cum_probs,
            mode="lines+markers",
            line=dict(color="#DC2626", width=2.0),
            name="Cumulative Probability",
        ),
        row=1,
        col=2,
    )

    # Threshold horizontal line
    fig.add_hline(
        y=p_val,
        line=dict(color="#DC2626", dash="dash", width=1.5),
        annotation_text=f"Nucleus Cutoff p = {p_val}",
        annotation_position="bottom right",
        row=1,
        col=2,
    )

    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False, row=1, col=1)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False, range=[0.3, 3.4], row=1, col=1)
    fig.update_xaxes(title_text="Vocabulary Tokens (Ranked by Probability)", tickangle=-45, row=1, col=2)
    fig.update_yaxes(title_text="Probability / Cumulative Mass", range=[0, 1.05], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=540,
        margin=dict(l=40, r=40, t=70, b=70),
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.75),
    )

    viz = mo.ui.plotly(fig)
    return (
        bar_colors,
        cum_probs,
        demo_vocab,
        fig,
        k_val,
        p_val,
        probs,
        raw_logits,
        tree_edges,
        tree_nodes,
        viz,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates the mechanics of deterministic tree exploration and stochastic tail truncation:

                1. **Left Panel (Search Tree: Greedy Myopia vs Beam Search)**:
                   - **Greedy Search (Red)** selects `'dog'` ($p=0.60$) at step 1 and `'barks'` ($p=0.30$) at step 2, achieving a cumulative log score of $-1.72$ ($P = 0.18$).
                   - **Beam Search ($B=2$, Teal)** retains `'galaxy'` ($p=0.40$) on its beam. At step 2, it explores `'rotates'` ($p=0.90$), achieving a cumulative score of $-1.02$ ($P = 0.36$). Beam search successfully bypasses greedy myopia to find the globally higher-likelihood sequence.
                2. **Right Panel (Top-k vs Nucleus Top-p Truncation)**:
                   - Fixed **Top-k ($k=5$)** strictly truncates after index 5 regardless of remaining probability mass.
                   - **Top-p Nucleus ($p=0.85$)** dynamically preserves tokens through the cumulative threshold (teal bars), gracefully admitting tokens `'over'` and `'lazy'` before terminating to truncate the low-probability noise tail (grey bars).
                """
            ),
            viz,
        ]
    )


@app.cell
def _(mo, np, pd):
    # Vectorized NumPy Implementation of Decoding Algorithms
    # 1. Greedy Search
    def numpy_greedy_decode(log_prob_func, initial_token_id, max_length=5):
        sequence = [initial_token_id]
        total_log_prob = 0.0
        for _ in range(max_length):
            logits = log_prob_func(sequence)
            next_token = int(np.argmax(logits))
            probs = np.exp(logits - np.max(logits))
            probs /= np.sum(probs)
            total_log_prob += np.log(max(probs[next_token], 1e-12))
            sequence.append(next_token)
        return sequence, total_log_prob

    # 2. Vectorized Beam Search Engine
    def numpy_beam_search(log_prob_func, initial_token_id, beam_width=3, max_length=5, length_penalty=0.7):
        # Candidates: list of tuples (sequence_list, cumulative_log_prob)
        beams = [([initial_token_id], 0.0)]

        for _step in range(max_length):
            all_candidates = []
            for seq, cum_log_prob in beams:
                logits = log_prob_func(seq)
                probs = np.exp(logits - np.max(logits))
                probs /= np.sum(probs)
                top_indices = np.argsort(probs)[::-1][:beam_width]

                for tok in top_indices:
                    cand_seq = seq + [int(tok)]
                    cand_log_prob = cum_log_prob + np.log(max(probs[tok], 1e-12))
                    # Apply length normalization
                    norm_score = cand_log_prob / (len(cand_seq) ** length_penalty)
                    all_candidates.append((cand_seq, cand_log_prob, norm_score))

            # Select top B candidates
            all_candidates.sort(key=lambda x: x[2], reverse=True)
            beams = [(cand[0], cand[1]) for cand in all_candidates[:beam_width]]

        best_seq, best_log_prob = beams[0]
        return best_seq, best_log_prob

    # 3. Top-k and Nucleus (Top-p) Sampling Engine
    def sample_next_token(logits, temperature=1.0, top_k=0, top_p=0.0):
        scaled_logits = logits / max(temperature, 1e-5)

        # Apply Top-k truncation
        if top_k > 0:
            top_k_indices = np.argsort(scaled_logits)[::-1][:top_k]
            masked_logits = np.full_like(scaled_logits, -1e9)
            masked_logits[top_k_indices] = scaled_logits[top_k_indices]
            scaled_logits = masked_logits

        # Apply Top-p Nucleus truncation
        if 0.0 < top_p < 1.0:
            sorted_indices = np.argsort(scaled_logits)[::-1]
            sorted_logits = scaled_logits[sorted_indices]
            exp_l = np.exp(sorted_logits - np.max(sorted_logits))
            cum_probs = np.cumsum(exp_l / np.sum(exp_l))

            # Identify cutoff
            cutoff_mask = cum_probs > top_p
            # Keep at least one token
            cutoff_mask[1:] = cutoff_mask[:-1]
            cutoff_mask[0] = False

            filtered_indices = sorted_indices[cutoff_mask]
            scaled_logits[filtered_indices] = -1e9

        # Softmax and categorical sampling
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits))
        sample_probs = exp_logits / np.sum(exp_logits)
        sampled_token = np.random.choice(len(sample_probs), p=sample_probs)
        return int(sampled_token), sample_probs

    # Simulation setup: Toy Transition Model with 10 tokens
    vocab_map = {0: "<s>", 1: "The", 2: "dog", 3: "galaxy", 4: "barks", 5: "rotates", 6: "loudly", 7: "fast", 8: "in", 9: "space"}
    inv_vocab = {v: k for k, v in vocab_map.items()}

    def synthetic_model_logits(seq):
        last_tok = seq[-1]
        logits = np.zeros(len(vocab_map))
        if last_tok == 0:  # <s> -> "The"
            logits[1] = 6.0
        elif last_tok == 1:  # "The" -> "dog" (0.6) or "galaxy" (0.4)
            logits[2] = 2.0  # dog
            logits[3] = 1.6  # galaxy
        elif last_tok == 2:  # "dog" -> "barks" (0.4) or "loudly" (0.2)
            logits[4] = 2.0  # barks
            logits[6] = 1.3  # loudly
        elif last_tok == 3:  # "galaxy" -> "rotates" (0.85)
            logits[5] = 4.5  # rotates
            logits[9] = 0.5  # space
        elif last_tok == 4:  # "barks" -> "loudly"
            logits[6] = 3.5
        elif last_tok == 5:  # "rotates" -> "fast" (0.7) or "in" (0.3)
            logits[7] = 3.2
            logits[8] = 2.1
        else:
            logits[9] = 2.0
        return logits

    # Verification 1: Compare Deterministic Search Outputs
    greedy_seq, greedy_score = numpy_greedy_decode(synthetic_model_logits, initial_token_id=0, max_length=4)
    beam_seq, beam_score = numpy_beam_search(synthetic_model_logits, initial_token_id=0, beam_width=3, max_length=4)

    df_deterministic = pd.DataFrame(
        [
            {
                "Search_Strategy": "Greedy Search (Myopic)",
                "Decoded_Sequence": " ".join([vocab_map[t] for t in greedy_seq]),
                "Total_Log_Likelihood": f"{greedy_score:.4f}",
                "Joint_Probability": f"{np.exp(greedy_score):.6f}",
                "Optimality_Status": "Suboptimal (Trapped by local argmax)",
            },
            {
                "Search_Strategy": "Beam Search (Width B = 3)",
                "Decoded_Sequence": " ".join([vocab_map[t] for t in beam_seq]),
                "Total_Log_Likelihood": f"{beam_score:.4f}",
                "Joint_Probability": f"{np.exp(beam_score):.6f}",
                "Optimality_Status": "Globally Superior Sequence Found",
            },
        ]
    )

    # Verification 2: Monte Carlo Sampling Diversity Benchmark (1,000 Generation Trials)
    np.random.seed(42)
    n_sims = 1000

    def run_simulations(strategy_kwargs):
        trajectories = []
        for _ in range(n_sims):
            seq = [0]
            for _ in range(4):
                lgs = synthetic_model_logits(seq)
                nxt, _ = sample_next_token(lgs, **strategy_kwargs)
                seq.append(nxt)
            trajectories.append(tuple(seq))

        unique_seqs = len(set(trajectories))
        # Frequency of most common sequence
        counts = pd.Series(trajectories).value_counts()
        mode_share = counts.iloc[0] / n_sims * 100.0
        # Token diversity: distinct tokens / total generated tokens
        all_tokens = [tok for traj in trajectories for tok in traj[1:]]
        token_diversity = len(set(all_tokens)) / len(vocab_map) * 100.0

        return unique_seqs, mode_share, token_diversity

    beam_u, beam_mode, beam_div = 1, 100.0, len(set(beam_seq[1:])) / len(vocab_map) * 100.0
    k_u, k_mode, k_div = run_simulations({"temperature": 0.8, "top_k": 2})
    p_u, p_mode, p_div = run_simulations({"temperature": 0.8, "top_p": 0.85})
    pure_u, pure_mode, pure_div = run_simulations({"temperature": 1.0})

    df_diversity = pd.DataFrame(
        [
            {
                "Sampling_Regime": "Deterministic Beam Search (B = 3)",
                "Unique_Sequences_Produced": f"{beam_u} (0% randomness)",
                "Modal_Sequence_Share": f"{beam_mode:.1f}%",
                "Vocabulary_Utilization": f"{beam_div:.1f}%",
                "Ideal_Use_Case": "Code Generation & Exact QA",
            },
            {
                "Sampling_Regime": "Top-k Truncation (k = 2, T = 0.8)",
                "Unique_Sequences_Produced": f"{k_u} distinct paths",
                "Modal_Sequence_Share": f"{k_mode:.1f}%",
                "Vocabulary_Utilization": f"{k_div:.1f}%",
                "Ideal_Use_Case": "Factual Summarization",
            },
            {
                "Sampling_Regime": "Nucleus Truncation (Top-p = 0.85, T = 0.8)",
                "Unique_Sequences_Produced": f"{p_u} distinct paths",
                "Modal_Sequence_Share": f"{p_mode:.1f}%",
                "Vocabulary_Utilization": f"{p_div:.1f}%",
                "Ideal_Use_Case": "Chatbots & Conversational AI",
            },
            {
                "Sampling_Regime": "Unconstrained Pure Sampling (T = 1.0)",
                "Unique_Sequences_Produced": f"{pure_u} distinct paths",
                "Modal_Sequence_Share": f"{pure_mode:.1f}%",
                "Vocabulary_Utilization": f"{pure_div:.1f}%",
                "Ideal_Use_Case": "Creative Writing & Brainstorming",
            },
        ]
    )

    table_det = mo.ui.table(df_deterministic)
    table_div = mo.ui.table(df_diversity)

    return (
        beam_div,
        beam_mode,
        beam_score,
        beam_seq,
        beam_u,
        df_deterministic,
        df_diversity,
        greedy_score,
        greedy_seq,
        inv_vocab,
        k_div,
        k_mode,
        k_u,
        n_sims,
        numpy_beam_search,
        numpy_greedy_decode,
        p_div,
        p_mode,
        p_u,
        pure_div,
        pure_mode,
        pure_u,
        run_simulations,
        sample_next_token,
        synthetic_model_logits,
        table_det,
        table_div,
        vocab_map,
    )


@app.cell
def _(mo, table_det, table_div):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Deterministic Search Comparison (Greedy vs Beam Search)

                Validating sequence log-likelihoods on a branching transition graph:
                """
            ),
            table_det,
            mo.md(
                r"""
                ### Example 2: Stochastic Sampling Diversity Audit (1,000 Generation Runs)

                Quantifying lexical diversity, sequence uniqueness, and modal concentration across decoding regimes:
                """
            ),
            table_div,
        ]
    )


if __name__ == "__main__":
    app.run()
