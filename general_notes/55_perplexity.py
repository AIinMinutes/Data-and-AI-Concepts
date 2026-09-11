# /// script
# dependencies = ["lmppl"]
# ///

import marimo

__generated_with = "0.24.0"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _():
    # packages added via marimo's package management: lmppl !pip3 install -q lmppl

    import warnings

    import lmppl
    import matplotlib.pyplot as plt
    import numpy as np

    warnings.filterwarnings("ignore")
    plt.style.use("dark_background")
    scorer = lmppl.LM("gpt2")
    return plt, scorer


@app.cell
def _(scorer):
    # Sentence and corresponding word splits
    sentence = "We celebrate Christmas on the 25th of every"
    words = sentence.split(" ")

    # Get perplexities for different context sizes
    ppl_values = [
        scorer.get_perplexity(" ".join(words[:2])),
        scorer.get_perplexity(" ".join(words[:3])),
        scorer.get_perplexity(" ".join(words[:4])),
        scorer.get_perplexity(" ".join(words)),
    ]

    print(ppl_values)
    return ppl_values, words


@app.cell
def _(plt, ppl_values, words):
    x_labels = [2, 3, 4, len(words)]

    # Plotting the perplexity values
    plt.figure(figsize=(8, 5), dpi=300)
    plt.plot(x_labels, ppl_values, marker="o", color="magenta", linestyle="-", linewidth=2, markersize=8)
    plt.title("Perplexity and Context", fontsize=16)
    plt.xlabel("Number of Words", fontsize=12)
    plt.ylabel("Perplexity", fontsize=12)
    plt.xticks(x_labels)
    plt.grid(True, alpha=0.2, linestyle="--")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### Perplexity in Language Models

    **Definition:**
    $$
    P(W) = \exp\left( -\frac{1}{N} \sum_{i=1}^{N} \log P(w_i | w_1, w_2, \dots, w_{i-1}) \right)
    $$
    Where:
    - $ P(w_i | w_1, \dots, w_{i-1}) $: Conditional probability of $ w_i $
    - $ N $: Number of words in the sequence

    **Log-Likelihood Sum:**
    $$
    \sum_{i=1}^{N} \log P(w_i | w_1, \dots, w_{i-1})
    $$
    Represents the cumulative uncertainty of the model's predictions.

    ---

    **Interpretation:**
    - **Lower Perplexity**: Indicates better model performance, meaning the model is more confident in its predictions and less surprised by the data.
    - **Higher Perplexity**: Indicates worse model performance, meaning the model is less confident or struggles to predict the data accurately.

    **What It Interprets:**
    Perplexity interprets the **uncertainty** of the language model: the lower the perplexity, the less uncertainty the model has in predicting the next word in a sequence.
    """)
    return


if __name__ == "__main__":
    app.run()
