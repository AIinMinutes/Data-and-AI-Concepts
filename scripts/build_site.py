#!/usr/bin/env python3
"""Export Marimo notes to the static site for learnaiinminutes.com with sequential order links."""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# Metadata for all 69 notes in sequential reading order
NOTE_METADATA: dict[str, tuple[str, str, str]] = {
    "00_introduction.py": ("Linear Algebra & Matrix Foundations", "Systems of Linear Equations", "Systems of linear equations, Ax = b, and vector representations"),
    "01_inner_product.py": ("Linear Algebra & Matrix Foundations", "Inner Products", "Dot product, cosine similarity, angles, and projection geometry"),
    "02_norm_and_metric.py": ("Linear Algebra & Matrix Foundations", "Norms and Metrics", "Lp norms, distance metrics, and the Minkowski inequality"),
    "03_hyperplanes.py": ("Linear Algebra & Matrix Foundations", "Hyperplanes", "Decision boundaries, half-spaces, and linear separation geometry"),
    "04_rank_one_matrices.py": ("Linear Algebra & Matrix Foundations", "Rank-One Matrices", "Outer products, low-rank factorization, and matrix approximations"),
    "05_orthogonality.py": ("Linear Algebra & Matrix Foundations", "Orthogonality", "Orthogonal vectors, bases, projections, and Gram-Schmidt process"),
    "06_moore_penrose_inverse.py": ("Linear Algebra & Matrix Foundations", "Moore-Penrose Pseudoinverse", "Pseudoinverse and least-squares solutions for overdetermined systems"),
    "07_spectral_decomposition.py": ("Linear Algebra & Matrix Foundations", "Spectral Decomposition", "Eigendecomposition, symmetric matrices, and singular value decomposition"),
    "08_matrix_calculus_short.py": ("Linear Algebra & Matrix Foundations", "Matrix Calculus", "Derivatives of vector and matrix expressions for optimization"),
    "09_condition_number.py": ("Linear Algebra & Matrix Foundations", "Condition Number", "Numerical stability, matrix sensitivity, and multicollinearity diagnostics"),
    "10_chebyshev_inequality.py": ("Probability & Statistical Foundations", "Chebyshev Inequality", "Distribution-free probability bounds and concentration"),
    "11_ecdf.py": ("Probability & Statistical Foundations", "Empirical CDF", "Empirical distribution functions and non-parametric inference"),
    "12_multivariate_normal_distribution.py": ("Probability & Statistical Foundations", "Multivariate Normal Distribution", "Multivariate Gaussian geometry, covariance matrices, and density contours"),
    "13_unbiased_vs_consistent.py": ("Probability & Statistical Foundations", "Unbiased vs Consistent Estimators", "Core properties of statistical estimators and sample size behavior"),
    "14_dist_of_minimum.py": ("Probability & Statistical Foundations", "Distribution of Minimum", "Order statistics and extreme value distributions"),
    "15_mutual_information.py": ("Probability & Statistical Foundations", "Mutual Information", "Information theory, entropy, and non-linear feature dependence"),
    "16_point_biserial.py": ("Probability & Statistical Foundations", "Point-Biserial Correlation", "Measuring association between continuous and binary variables"),
    "17_jensen_inequality.py": ("Probability & Statistical Foundations", "Jensen's Inequality", "Convexity, expectation inequalities, and bounds in learning algorithms"),
    "18_cramer_v.py": ("Applied Statistics & Correlation", "Cramer's V", "Strength of association between categorical variables"),
    "19_kendalltaub.py": ("Applied Statistics & Correlation", "Kendall's Tau-b", "Non-parametric rank correlation robust to ties"),
    "20_spurious_correlation.py": ("Applied Statistics & Correlation", "Spurious Correlation", "Confounders, lurking variables, and Simpson's paradox"),
    "21_kruskal_wallis.py": ("Applied Statistics & Correlation", "Kruskal-Wallis Test", "Non-parametric ANOVA for comparing multiple groups"),
    "22_acf_and_pacf.py": ("Applied Statistics & Correlation", "ACF and PACF", "Autocorrelation and partial autocorrelation for time series modeling"),
    "23_ewa_and_bias_correction.py": ("Applied Statistics & Correlation", "Exponential Moving Averages", "EMA smoothing, momentum, and initial bias correction"),
    "24_adjusted_r_squared.py": ("Applied Statistics & Correlation", "Adjusted R-Squared", "Penalizing model complexity in linear regression"),
    "25_predictive_r2.py": ("Applied Statistics & Correlation", "Predictive R-Squared", "Leave-one-out cross-validation and generalization performance"),
    "26_hotelling.py": ("Applied Statistics & Correlation", "Hotelling's T-Squared", "Multivariate hypothesis testing and group mean comparisons"),
    "27_principal_component_analysis.py": ("Multivariate Methods & Dimensionality", "Principal Component Analysis", "Dimensionality reduction via covariance eigendecomposition"),
    "28_factor_analysis.py": ("Multivariate Methods & Dimensionality", "Factor Analysis", "Latent variable modeling and unobserved factor estimation"),
    "29_canonical_correlation_analysis.py": ("Multivariate Methods & Dimensionality", "Canonical Correlation Analysis", "Maximizing correlation between two multidimensional variable sets"),
    "30_correspondence_analysis.py": ("Multivariate Methods & Dimensionality", "Correspondence Analysis", "Geometric visualization of contingency tables and categorical associations"),
    "31_gaussian_mixture_models.py": ("Multivariate Methods & Dimensionality", "Gaussian Mixture Models", "Soft clustering, Expectation-Maximization, and density estimation"),
    "32_elastic_net.py": ("Machine Learning Models & Diagnostics", "Elastic Net Regression", "Balancing L1 and L2 penalties for correlated feature selection"),
    "33_huber_loss.py": ("Machine Learning Models & Diagnostics", "Huber Loss", "Robust regression combining squared and absolute error penalties"),
    "34_mahalanobis_distance.py": ("Machine Learning Models & Diagnostics", "Mahalanobis Distance", "Covariance-scaled distance metrics for outlier detection"),
    "35_gini_impurity_vs_entropy.py": ("Machine Learning Models & Diagnostics", "Gini Impurity vs Entropy", "Split evaluation criteria for decision trees"),
    "36_agglomerative_clustering.py": ("Machine Learning Models & Diagnostics", "Agglomerative Clustering", "Hierarchical clustering, distance metrics, and dendrograms"),
    "37_natural_breaks.py": ("Machine Learning Models & Diagnostics", "Natural Breaks (Jenks)", "1D clustering optimization for histogram and choropleth binning"),
    "38_oversampling.py": ("Machine Learning Models & Diagnostics", "Oversampling and SMOTE", "Synthesizing minority class samples for imbalanced classification"),
    "39_permutation_importance.py": ("Machine Learning Models & Diagnostics", "Permutation Feature Importance", "Model-agnostic feature importance via shuffling evaluation"),
    "40_pca_vs_feat_ag.py": ("Machine Learning Models & Diagnostics", "PCA vs Feature Agglomeration", "Linear dimensionality reduction versus hierarchical feature clustering"),
    "41_pseudo_r2.py": ("Machine Learning Models & Diagnostics", "Pseudo R-Squared", "Goodness-of-fit metrics for logistic regression and GLMs"),
    "42_multiclass_classification.py": ("Machine Learning Models & Diagnostics", "Multiclass Classification", "Softmax functions, cross-entropy loss, and decision regions"),
    "43_energy.py": ("Machine Learning Models & Diagnostics", "Energy-Based Models", "Energy landscapes, Boltzmann distributions, and score matching"),
    "44_logistic_regression.py": ("Interpretable AI", "Logistic Regression Interpretability", "Log-odds, odds ratios, and marginal feature effects"),
    "45_shapley.py": ("Interpretable AI", "Shapley Values and SHAP", "Game-theoretic feature attributions and local model explanations"),
    "46_model_counterfactuals.py": ("Interpretable AI", "Model Counterfactuals", "Actionable recourse and minimal changes to alter model predictions"),
    "47_gelu.py": ("Deep Learning & Generative AI", "GELU Activation", "Gaussian Error Linear Units in modern Transformer models"),
    "48_temperature_scaled_softmax.py": ("Deep Learning & Generative AI", "Temperature-Scaled Softmax", "Calibrating confidence and diversity in probability distributions"),
    "49_focal_loss_balanced.py": ("Deep Learning & Generative AI", "Focal Loss", "Down-weighting easy examples for dense object detection and hard mining"),
    "50_attention_mechanism.py": ("Deep Learning & Generative AI", "Attention Mechanism", "Scaled dot-product attention as value weighting by query-key similarity"),
    "51_causal_attention.py": ("Deep Learning & Generative AI", "Causal Attention", "Autoregressive masking in decoder-only generative models"),
    "52_multi_head_attention.py": ("Deep Learning & Generative AI", "Multi-Head Attention", "Parallel representation subspaces in Transformer blocks"),
    "53_layer_and_rms_normalization.py": ("Deep Learning & Generative AI", "LayerNorm and RMSNorm", "Internal activation scaling and modern variance normalization"),
    "54_decoding_strategies.py": ("Deep Learning & Generative AI", "Decoding Strategies", "Greedy search, beam search, top-k, and nucleus (top-p) sampling"),
    "55_perplexity.py": ("Deep Learning & Generative AI", "Perplexity", "Information-theoretic evaluation metric for autoregressive language models"),
    "56_reparametrization_trick.py": ("Deep Learning & Generative AI", "Reparameterization Trick", "Differentiable sampling through stochastic nodes via auxiliary noise"),
    "57_autoencoder_latent_space.py": ("Deep Learning & Generative AI", "Autoencoder Latent Space", "Deterministic bottleneck compression and feature representation"),
    "58_pca_for_anomaly_detection.py": ("Deep Learning & Generative AI", "PCA for Anomaly Detection", "Reconstruction error in reduced eigenspaces as anomaly scoring"),
    "59_vae_on_mnist.py": ("Deep Learning & Generative AI", "VAE on MNIST", "Variational Autoencoders with evidence lower bound (ELBO) optimization"),
    "60_vae_anomaly_detection.py": ("Deep Learning & Generative AI", "VAE Anomaly Detection", "Probabilistic reconstruction likelihood for out-of-distribution detection"),
    "61_user_item_interaction_matrix.py": ("Graphs & Applied Pipelines", "User-Item Interaction Matrix", "Bipartite graph representations for recommendation systems"),
    "62_grammar_of_graphics.py": ("Graphs & Applied Pipelines", "Grammar of Graphics", "Layered visualization specifications with plotnine"),
    "63_einsum.py": ("Programming Patterns & Tools", "Einstein Summation (einsum)", "Succinct multidimensional array contractions in NumPy and PyTorch"),
    "64_pivoting.py": ("Programming Patterns & Tools", "Data Pivoting", "Reshaping and aggregating tabular datasets in Pandas"),
    "65_cudf.py": ("Programming Patterns & Tools", "GPU Acceleration (cuDF)", "Accelerating dataframe pipelines with GPU memory and parallelism"),
    "66_prefix_sum.py": ("Programming Patterns & Tools", "Prefix Sum Pattern", "Constant-time range sum queries with precomputed cumulative arrays"),
    "67_kadanes.py": ("Programming Patterns & Tools", "Kadane's Algorithm", "Linear-time maximum contiguous subarray sum via dynamic programming"),
    "68_two_pointer.py": ("Programming Patterns & Tools", "Two-Pointer Technique", "In-place array processing and linear scan search optimizations"),
}


@dataclass(frozen=True)
class Note:
    source: Path
    slug: str
    title: str
    topic: str
    blurb: str
    number: int


def get_all_notes() -> list[Note]:
    """Discover all 69 numbered notes in sequential order."""
    notes_dir = ROOT / "notebooks" / "fundamentals"
    py_files = sorted(notes_dir.glob("[0-9][0-9]_*.py"))
    notes = []
    for file in py_files:
        filename = file.name
        match = re.match(r"^(\d+)_", filename)
        num = int(match.group(1)) if match else 0
        topic, clean_title, blurb = NOTE_METADATA.get(
            filename,
            ("Fundamentals", filename.replace(".py", "").replace("_", " ").title(), "Data and AI concept note"),
        )
        notes.append(
            Note(
                source=file.relative_to(ROOT),
                slug=f"fundamentals/{file.stem}",
                title=f"Note {num:02d}: {clean_title}",
                topic=topic,
                blurb=blurb,
                number=num,
            )
        )
    return notes


def export_note(note: Note, prev_note: Note | None, next_note: Note | None, output_dir: Path) -> Path:
    html_path = output_dir / note.slug / "index.html"
    html_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "marimo",
        "export",
        "html",
        "--force",
        str(ROOT / note.source),
        "-o",
        str(html_path),
    ]
    print(f"export {note.source} -> {html_path.relative_to(output_dir)}")
    subprocess.run(cmd, cwd=ROOT, check=True)

    # Inject order links into the exported HTML
    if html_path.is_file():
        content = html_path.read_text(encoding="utf-8")
        prev_html = (
            f'<a href="/{prev_note.slug}/" style="color: #1f4e79; text-decoration: none; font-weight: 500;">&larr; {prev_note.title}</a>'
            if prev_note
            else '<span style="color: #bbb;">&larr; Start</span>'
        )
        next_html = (
            f'<a href="/{next_note.slug}/" style="color: #1f4e79; text-decoration: none; font-weight: 500;">{next_note.title} &rarr;</a>'
            if next_note
            else '<span style="color: #bbb;">End &rarr;</span>'
        )
        nav_header = f"""
        <header id="note-order-nav" style="background: #ffffff; border-bottom: 1px solid #e5e5e5; padding: 12px 24px; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; font-size: 14px; display: flex; justify-content: space-between; align-items: center; position: sticky; top: 0; z-index: 9999;">
          <div>{prev_html}</div>
          <div><a href="/" style="color: #333333; text-decoration: none; font-weight: 600;">Learn AI in Minutes &bull; All Notes</a></div>
          <div>{next_html}</div>
        </header>
        """
        # Inject right after <body> or at top of html
        if "<body" in content:
            content = re.sub(r"(<body[^>]*>)", r"\1" + nav_header, content, count=1)
            html_path.write_text(content, encoding="utf-8")

    return html_path


def render_index(notes: list[Note]) -> str:
    # Group notes by Topic / Part
    topics: dict[str, list[Note]] = {}
    for note in notes:
        topics.setdefault(note.topic, []).append(note)

    sections_html = []
    for topic, topic_notes in topics.items():
        items = "\n".join(
            f'        <li>\n'
            f'          <a href="/{note.slug}/">{note.title}</a>\n'
            f'          <p class="blurb">{note.blurb}</p>\n'
            f'        </li>'
            for note in topic_notes
        )
        sections_html.append(
            f'      <h2>{topic}</h2>\n'
            f'      <ol class="note-list">\n'
            f'{items}\n'
            f'      </ol>'
        )

    content_body = "\n".join(sections_html)

    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Learn AI in Minutes - All Notes</title>
    <style>
      :root {{
        --ink: #1a1a1a;
        --muted: #5a5a5a;
        --accent: #1f4e79;
        --rule: #e6e6e6;
        --bg-hover: #f9fbfd;
      }}
      * {{ box-sizing: border-box; }}
      body {{
        margin: 0;
        color: var(--ink);
        background: #fff;
        font: 18px/1.55 "Iowan Old Style", Palatino, "Palatino Linotype",
          "Book Antiqua", Georgia, serif;
      }}
      main {{
        max-width: 48rem;
        margin: 0 auto;
        padding: 3.5rem 1.25rem 4rem;
      }}
      h1 {{
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem;
      }}
      .lede {{
        color: var(--muted);
        font-size: 1.1rem;
        margin: 0 0 2.5rem;
      }}
      h2 {{
        font-size: 0.85rem;
        font-weight: 700;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        border-top: 1px solid var(--rule);
        padding-top: 1.6rem;
        margin: 2rem 0 1.2rem;
      }}
      ol.note-list {{
        list-style: none;
        padding: 0;
        margin: 0;
      }}
      ol.note-list li {{
        padding: 0.75rem 0.5rem;
        border-radius: 4px;
        transition: background-color 0.15s ease;
      }}
      ol.note-list li:hover {{
        background-color: var(--bg-hover);
      }}
      ol.note-list li + li {{
        margin-top: 0.5rem;
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
      a:hover {{ text-decoration: underline; }}
      ol.note-list li > a {{
        font-size: 1.12rem;
        font-weight: 600;
      }}
      .blurb {{
        color: #444;
        font-size: 0.95rem;
        margin: 0.25rem 0 0;
      }}
      .foot {{
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--rule);
        color: var(--muted);
        font-size: 0.92rem;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Learn AI in Minutes</h1>
      <p class="lede">
        Foundational concepts in data, statistics, machine learning, and generative AI.
        Every note is a standalone interactive Marimo notebook with concept intuition, mathematical formulations, code examples, and practical takeaways.
      </p>
{content_body}
      <p class="foot">
        Published at <a href="https://learnaiinminutes.com">learnaiinminutes.com</a>.
        Source code: <a href="https://github.com/AIinMinutes/Data-and-AI-Concepts">Data-and-AI-Concepts</a>.
      </p>
    </main>
  </body>
</html>
"""


def build(output_dir: Path, only: str | None) -> None:
    all_notes = get_all_notes()
    selected = [
        note
        for note in all_notes
        if only is None or only in {note.source.stem, note.slug, note.source.name, f"{note.number:02d}"}
    ]
    if not selected:
        raise SystemExit(f"no notes matched {only!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".nojekyll").write_text("")
    (output_dir / "CNAME").write_text("learnaiinminutes.com\n")

    for i, note in enumerate(selected):
        if not (ROOT / note.source).is_file():
            raise SystemExit(f"missing notebook: {note.source}")
        prev_note = all_notes[note.number - 1] if note.number > 0 else None
        next_note = all_notes[note.number + 1] if note.number < len(all_notes) - 1 else None
        export_note(note, prev_note, next_note, output_dir)

    # Render complete homepage index
    (output_dir / "index.html").write_text(render_index(all_notes), encoding="utf-8")
    print(f"site built -> {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_site",
        help="directory for the built site (default: _site)",
    )
    parser.add_argument(
        "--only",
        help="export a single notebook (number, stem, filename, or slug)",
    )
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.only)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
