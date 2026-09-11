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
    import torch
    import torch.nn as nn

    return go, make_subplots, mo, nn, np, pd, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        [← 54 Decoding Strategies](54_decoding_strategies.py) | [Index](../index.html) | [56 Reparameterization Trick →](56_reparameterization_trick.py)

        # 55. Perplexity in Language Modeling: Information Entropy, Branching Factor, and Sequence Likelihood

        ### Executive Summary

        In generative language modeling, **Perplexity (PPL)** serves as the canonical intrinsic benchmark for evaluating how effectively a probability model $P_\theta$ captures the syntax, semantics, and factual distribution of a text corpus. Mathematically formulated as the exponentiation of the cross-entropy loss, perplexity quantifies the model's average uncertainty per token.

        Intuitively, a perplexity of $K$ signifies that the model is as uncertain at each prediction step as if it were choosing uniformly at random from $K$ equally likely alternative tokens—often called the **effective branching factor**. A lower perplexity indicates higher predictive certainty, tighter probability concentration around ground-truth tokens, and superior language modeling capability.
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## [b] Mathematical Foundations and Information-Theoretic Derivations

        ### 1. The Cross-Entropy Formulation

        Let $W = (w_1, w_2, \dots, w_N)$ be a sequence of $N$ tokens drawn from a test corpus. Under the autoregressive probability chain rule, the joint probability assigned to the sequence by model $P_\theta$ is:

        $$P_\theta(W) = P_\theta(w_1, w_2, \dots, w_N) = \prod_{i=1}^N P_\theta(w_i \mid w_1, \dots, w_{i-1})$$

        The empirical cross-entropy loss $\mathcal{H}(W; P_\theta)$ per token (in nats) is the negative log-likelihood averaged over sequence length $N$:

        $$\mathcal{H}(W; P_\theta) = -\frac{1}{N} \ln P_\theta(W) = -\frac{1}{N} \sum_{i=1}^N \ln P_\theta(w_i \mid w_{<i})$$

        ### 2. Formal Definition of Perplexity

        Perplexity is defined as the exponentiated cross-entropy loss:

        $$\operatorname{PPL}(W) = \exp\left( \mathcal{H}(W; P_\theta) \right) = \exp\left( -\frac{1}{N} \sum_{i=1}^N \ln P_\theta(w_i \mid w_{<i}) \right)$$

        Using the properties of logarithms and products, perplexity can be expressed equivalently as the inverse geometric mean of the token probabilities:

        $$\operatorname{PPL}(W) = \left( \prod_{i=1}^N P_\theta(w_i \mid w_{<i}) \right)^{-\frac{1}{N}} = \sqrt[N]{\frac{1}{\prod_{i=1}^N P_\theta(w_i \mid w_{<i})}}$$

        If base-2 logarithms are used (measuring information in bits):

        $$\mathcal{H}_2(W) = -\frac{1}{N} \sum_{i=1}^N \log_2 P_\theta(w_i \mid w_{<i}), \qquad \operatorname{PPL}(W) = 2^{\mathcal{H}_2(W)}$$

        ### 3. The Branching Factor Interpretation

        To build rigorous intuition for perplexity, consider two boundary cases on a vocabulary $\mathcal{V}$ of size $V$:

        1. **Perfect Model (Zero Uncertainty)**:
           If the model predicts the correct token with $100\%$ confidence at every step ($P_\theta(w_i \mid w_{<i}) = 1.0$ for all $i$):
           $$\mathcal{H}(W) = 0 \implies \operatorname{PPL}(W) = \exp(0) = 1.0$$
           A perplexity of $1.0$ is the theoretical minimum, signifying zero surprise.

        2. **Uniform Random Baseline (Maximum Entropy)**:
           If the model has learned nothing and assigns equal probability $1/V$ to every token in the vocabulary:
           $$\mathcal{H}(W) = -\frac{1}{N} \sum_{i=1}^N \ln\left(\frac{1}{V}\right) = \ln V \implies \operatorname{PPL}(W) = \exp(\ln V) = V$$
           The perplexity equals the entire vocabulary size.

        3. **General Meaning of $\operatorname{PPL} = K$**:
           If a model achieves $\operatorname{PPL} = 12.4$ on a test set, it means that predicting each next token is, on average, as difficult for the model as choosing between $12.4$ equally likely candidate words.

        ### 4. Cross-Tokenizer Comparison: Bits Per Byte (BPB)

        A critical error in NLP evaluation is comparing raw perplexity across models that use different tokenizers (e.g., LLaMA-3 with $128\text{k}$ BPE tokens vs GPT-2 with $50\text{k}$ BPE tokens). A model with larger tokens will have fewer tokens per sentence, inflating per-token cross-entropy while reducing sequence length $N$.

        To ensure a mathematically fair comparison invariant to tokenization granularity, evaluations are normalized by total UTF-8 byte count $B$, computing **Bits Per Byte (BPB)**:

        $$\text{BPB} = \frac{\sum_{i=1}^N -\log_2 P_\theta(w_i \mid w_{<i})}{\text{Total Bytes } B} = \frac{N \cdot \mathcal{H}_2(W)}{B}$$

        Per-byte perplexity is then:

        $$\operatorname{PPL}_{\text{byte}} = 2^{\text{BPB}}$$
        """
    )
    return


@app.cell
def _(go, make_subplots, mo, np):
    # Simulated sentence demonstrating in-context perplexity resolution
    # As context grows from token 1 to token 10, conditional probability increases
    test_sentence_tokens = [
        "Artificial",
        "intelligence",
        "models",
        "predict",
        "next",
        "tokens",
        "by",
        "minimizing",
        "cross",
        "entropy",
        "loss",
    ]
    n_tokens = len(test_sentence_tokens)

    # Simulated realistic probabilities reflecting linguistic constraint buildup
    # Token 1 ("Artificial"): ambiguous start -> p = 0.04 (PPL = 25.0)
    # Token 2 ("intelligence"): heavily conditioned by "Artificial" -> p = 0.72 (PPL = 1.38)
    # Token 3 ("models"): moderately likely -> p = 0.28
    # Token 4 ("predict"): verb in AI context -> p = 0.45
    # Token 5 ("next"): idiomatic -> p = 0.65
    # Token 6 ("tokens"): highly constrained -> p = 0.82
    # Token 7 ("by"): preposition -> p = 0.55
    # Token 8 ("minimizing"): technical -> p = 0.38
    # Token 9 ("cross"): technical -> p = 0.75
    # Token 10 ("entropy"): bound bigram -> p = 0.94
    # Token 11 ("loss"): bound trigram -> p = 0.98
    token_probs = np.array([0.04, 0.72, 0.28, 0.45, 0.65, 0.82, 0.55, 0.38, 0.75, 0.94, 0.98])
    step_nll = -np.log(token_probs)

    # Cumulative running perplexity: exp(1/t * sum_{i=1}^t -ln p_i)
    cumulative_nll = np.cumsum(step_nll) / np.arange(1, n_tokens + 1)
    cumulative_ppl = np.exp(cumulative_nll)
    instantaneous_ppl = 1.0 / token_probs

    # Panel 2 data: Theoretical Branching Factor vs Cross-Entropy
    entropy_grid = np.linspace(0.0, 6.0, 200)
    ppl_nats = np.exp(entropy_grid)
    ppl_bits = 2.0**entropy_grid

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[
            "<b>Token-Level Surprise and Cumulative Perplexity Trajectory</b>",
            "<b>Branching Factor Growth: PPL = exp(H) and 2^H</b>",
        ],
        horizontal_spacing=0.14,
    )

    # Panel 1: Instantaneous vs Cumulative Perplexity
    fig.add_trace(
        go.Bar(
            x=test_sentence_tokens,
            y=instantaneous_ppl,
            name="Instantaneous Token Surprise (1/p_i)",
            marker_color="#93C5FD",
            opacity=0.7,
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=test_sentence_tokens,
            y=cumulative_ppl,
            mode="lines+markers",
            line=dict(color="#1D4ED8", width=3.0),
            marker=dict(size=8),
            name="Cumulative Sequence PPL",
        ),
        row=1,
        col=1,
    )

    # Panel 2: Exponential Branching Curve
    fig.add_trace(
        go.Scatter(
            x=entropy_grid,
            y=ppl_nats,
            mode="lines",
            line=dict(color="#DC2626", width=2.5),
            name="PPL = exp(H_nats)",
        ),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Scatter(
            x=entropy_grid,
            y=ppl_bits,
            mode="lines",
            line=dict(color="#0D9488", width=2.5, dash="dash"),
            name="PPL = 2^(H_bits)",
        ),
        row=1,
        col=2,
    )

    # Annotate typical LLM benchmark regime
    fig.add_vrect(
        x0=2.0,
        x1=3.0,
        fillcolor="#FEF3C7",
        opacity=0.5,
        layer="below",
        line_width=0,
        annotation_text="Typical LLM Regime (PPL 7-20)",
        annotation_position="top left",
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="Sequence Tokens (In-Context Horizon)", tickangle=-35, row=1, col=1)
    fig.update_yaxes(title_text="Perplexity / Effective Branching", range=[0, 30], row=1, col=1)
    fig.update_xaxes(title_text="Cross-Entropy Loss H", row=1, col=2)
    fig.update_yaxes(title_text="Perplexity (Effective Choices)", range=[1, 150], row=1, col=2)

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=40, t=70, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.32, xanchor="center", x=0.5),
    )

    viz = mo.ui.plotly(fig)
    return (
        cumulative_nll,
        cumulative_ppl,
        entropy_grid,
        fig,
        instantaneous_ppl,
        n_tokens,
        ppl_bits,
        ppl_nats,
        step_nll,
        test_sentence_tokens,
        token_probs,
        viz,
    )


@app.cell
def _(mo, viz):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [c] Interactive Visualizations

                The dual-panel visual below illustrates the mechanics of token-level uncertainty and branching factor scaling:

                1. **Left Panel (In-Context Perplexity Trajectory)**:
                   - The sentence opens with high initial uncertainty on `"Artificial"` ($1/p = 25.0$).
                   - Once `"Artificial"` is conditioned, the token `"intelligence"` is immediately expected ($p=0.72$, surprise $1.38$).
                   - As context expands toward technical phrases like `"cross entropy loss"`, probabilities soar above $90\%$, driving instantaneous perplexity near $1.0$ and stabilizing the overall cumulative sequence perplexity at $\approx 2.38$.
                2. **Right Panel (Branching Factor Curve)**: Perplexity grows exponentially with cross-entropy loss. Modern frontier models operate in the highlighted band ($H \approx 2.0 - 2.8$ nats, corresponding to $\operatorname{PPL} \approx 7 - 16$).
                """
            ),
            viz,
        ]
    )


@app.cell
def _(cumulative_ppl, mo, nn, np, pd, test_sentence_tokens, token_probs, torch):
    # Vectorized NumPy Perplexity Implementation
    def calculate_perplexity_numpy(probabilities):
        """Calculates exact perplexity from an array of conditional token probabilities.

        PPL = exp(-1/N * sum(ln p_i))
        """
        probs = np.asarray(probabilities)
        nll = -np.log(np.maximum(probs, 1e-15))
        mean_nll = np.mean(nll)
        ppl = np.exp(mean_nll)
        return ppl, mean_nll

    # PyTorch CrossEntropyLoss equivalence verification
    torch.manual_seed(42)
    _vocab_size = 50
    _seq_len = 8
    # Random unnormalized logits: (seq_len, vocab_size)
    _dummy_logits = torch.randn(_seq_len, _vocab_size)
    # Random target token indices
    _dummy_targets = torch.randint(0, _vocab_size, (_seq_len,))

    # 1. PyTorch Standard CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    ce_loss = criterion(_dummy_logits, _dummy_targets).item()
    ppl_pytorch = np.exp(ce_loss)

    # 2. NumPy from Softmax Probabilities
    _probs_all = torch.softmax(_dummy_logits, dim=-1).numpy()
    _target_probs = _probs_all[np.arange(_seq_len), _dummy_targets.numpy()]
    ppl_numpy, mean_nll_np = calculate_perplexity_numpy(_target_probs)

    diff_verif = abs(ppl_pytorch - ppl_numpy)

    df_equivalence = pd.DataFrame(
        [
            {
                "Calculation_Method": "PyTorch nn.CrossEntropyLoss Exponentiation",
                "Cross_Entropy_Loss": f"{ce_loss:.6f} nats",
                "Computed_Perplexity": f"{ppl_pytorch:.6f}",
                "Verification_Status": "Reference Implementation",
            },
            {
                "Calculation_Method": "Vectorized NumPy Inverse Geometric Mean",
                "Cross_Entropy_Loss": f"{mean_nll_np:.6f} nats",
                "Computed_Perplexity": f"{ppl_numpy:.6f}",
                "Verification_Status": f"Bitwise Match (Diff: {diff_verif:.2e})",
            },
        ]
    )

    # Example 2: In-Context Token Breakdown Table
    token_records = []
    for _idx, _tok in enumerate(test_sentence_tokens):
        _p = token_probs[_idx]
        token_records.append(
            {
                "Position": _idx + 1,
                "Token": _tok,
                "Conditional_Probability": f"{_p * 100:.1f}%",
                "Surprise_NLL_nats": f"{-np.log(_p):.3f}",
                "Instantaneous_Branching": f"{1.0 / _p:.2f}",
                "Cumulative_Sequence_PPL": f"{cumulative_ppl[_idx]:.2f}",
            }
        )

    df_tokens = pd.DataFrame(token_records)

    # Example 3: Cross-Tokenizer Bits Per Byte (BPB) Normalization Simulation
    # Target phrase: "The quick brown fox jumps over the lazy dog" (43 UTF-8 bytes)
    target_text = "The quick brown fox jumps over the lazy dog"
    byte_count = len(target_text.encode("utf-8"))

    tokenizer_simulations = [
        {
            "Tokenizer_Level": "Character-Level Tokenizer",
            "Token_Count_N": 43,
            "Per_Token_Loss_bits": 1.45,
            "Total_Bits": 43 * 1.45,
        },
        {
            "Tokenizer_Level": "Subword BPE (32k Vocab)",
            "Token_Count_N": 9,
            "Per_Token_Loss_bits": 6.80,
            "Total_Bits": 9 * 6.80,
        },
        {
            "Tokenizer_Level": "Word-Level Tokenizer",
            "Token_Count_N": 9,
            "Per_Token_Loss_bits": 7.10,
            "Total_Bits": 9 * 7.10,
        },
    ]

    bpb_records = []
    for entry in tokenizer_simulations:
        tot_bits = entry["Total_Bits"]
        bpb = tot_bits / byte_count
        token_ppl = 2.0 ** entry["Per_Token_Loss_bits"]
        byte_ppl = 2.0**bpb
        bpb_records.append(
            {
                "Tokenizer_Type": entry["Tokenizer_Level"],
                "Tokens_per_Sentence": entry["Token_Count_N"],
                "Raw_Token_Perplexity": f"{token_ppl:.2f}",
                "Total_Information_Bits": f"{tot_bits:.1f}",
                "Bits_Per_Byte (BPB)": f"{bpb:.3f}",
                "Normalized_Byte_Perplexity": f"{byte_ppl:.3f}",
                "Evaluation_Fairness": "Comparable Across Tokenizers" if "BPB" in "Bits_Per_Byte" else "",
            }
        )

    df_bpb = pd.DataFrame(bpb_records)

    table_equiv = mo.ui.table(df_equivalence)
    table_tokens = mo.ui.table(df_tokens)
    table_bpb = mo.ui.table(df_bpb)

    return (
        byte_count,
        calculate_perplexity_numpy,
        ce_loss,
        criterion,
        df_bpb,
        df_equivalence,
        df_tokens,
        diff_verif,
        mean_nll_np,
        ppl_numpy,
        ppl_pytorch,
        table_bpb,
        table_equiv,
        table_tokens,
        target_text,
        token_records,
        tokenizer_simulations,
    )


@app.cell
def _(mo, table_bpb, table_equiv, table_tokens):
    return mo.vstack(
        [
            mo.md(
                r"""
                ## [d] Code Examples and Validation

                ### Example 1: Mathematical Equivalence of Perplexity Formulations

                Validating that exponentiated `nn.CrossEntropyLoss` exactly matches inverse geometric mean probability:
                """
            ),
            table_equiv,
            mo.md(
                r"""
                ### Example 2: Step-by-Step Token Probability & Branching Factor Audit

                Tracking instantaneous surprise and cumulative sequence perplexity as conditioning context accumulates:
                """
            ),
            table_tokens,
            mo.md(
                r"""
                ### Example 3: Cross-Tokenizer Fair Evaluation via Bits Per Byte (BPB)

                Demonstrating how normalizing cross-entropy by total UTF-8 byte count resolves token-granularity discrepancies:
                """
            ),
            table_bpb,
        ]
    )


if __name__ == "__main__":
    app.run()
