#!/usr/bin/env python3
"""Export Marimo notes to the static site for learnaiinminutes.com with sequential order links."""

from __future__ import annotations

import argparse
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NOTE_METADATA: dict[str, tuple[str, str, str]] = {
    "00_introduction.py": (
        "Linear Algebra & Matrix Foundations",
        "Systems of Linear Equations",
        "Linear combinations, row vs column perspectives, and invertibility",
    ),
    "01_inner_product.py": (
        "Linear Algebra & Matrix Foundations",
        "Inner Products and Angles",
        "Dot products, projection, geometric angles, and Hilbert spaces",
    ),
    "02_norm_and_metric.py": (
        "Linear Algebra & Matrix Foundations",
        "Vector Norms and Metrics",
        "L1, L2, Lp norms, distances, and unit ball geometries",
    ),
    "03_hyperplanes.py": (
        "Linear Algebra & Matrix Foundations",
        "Hyperplanes and Halfspaces",
        "Decision boundaries, affine sets, and separating hyperplanes",
    ),
    "04_rank_one_matrices.py": (
        "Linear Algebra & Matrix Foundations",
        "Rank-One Matrices",
        "Outer products, low-rank factorization, and matrix approximations",
    ),
    "05_orthogonality.py": (
        "Linear Algebra & Matrix Foundations",
        "Orthogonality",
        "Orthogonal vectors, bases, projections, and Gram-Schmidt process",
    ),
    "06_moore_penrose_inverse.py": (
        "Linear Algebra & Matrix Foundations",
        "Moore-Penrose Pseudoinverse",
        "Pseudoinverse and least-squares solutions for overdetermined systems",
    ),
    "07_spectral_decomposition.py": (
        "Linear Algebra & Matrix Foundations",
        "Spectral Decomposition",
        "Eigendecomposition, symmetric matrices, and singular value decomposition",
    ),
    "08_matrix_calculus_short.py": (
        "Linear Algebra & Matrix Foundations",
        "Matrix Calculus",
        "Derivatives of vector and matrix expressions for optimization",
    ),
    "09_condition_number.py": (
        "Linear Algebra & Matrix Foundations",
        "Condition Number",
        "Numerical stability, matrix sensitivity, and multicollinearity diagnostics",
    ),
    "10_chebyshev_inequality.py": (
        "Probability & Statistical Foundations",
        "Chebyshev Inequality",
        "Distribution-free probability bounds and concentration",
    ),
    "11_ecdf.py": (
        "Probability & Statistical Foundations",
        "Empirical CDF",
        "Empirical distribution functions and non-parametric inference",
    ),
    "12_multivariate_normal_distribution.py": (
        "Probability & Statistical Foundations",
        "Multivariate Normal Distribution",
        "Multivariate Gaussian geometry, covariance matrices, and density contours",
    ),
    "13_unbiased_vs_consistent.py": (
        "Probability & Statistical Foundations",
        "Unbiased vs Consistent Estimators",
        "Core properties of statistical estimators and sample size behavior",
    ),
    "14_dist_of_minimum.py": (
        "Probability & Statistical Foundations",
        "Distribution of Minimum",
        "Order statistics and extreme value distributions",
    ),
    "15_mutual_information.py": (
        "Probability & Statistical Foundations",
        "Mutual Information",
        "Information theory, entropy, and non-linear feature dependence",
    ),
    "16_point_biserial.py": (
        "Probability & Statistical Foundations",
        "Point-Biserial Correlation",
        "Measuring association between continuous and binary variables",
    ),
    "17_jensen_inequality.py": (
        "Probability & Statistical Foundations",
        "Jensen's Inequality",
        "Convexity, expectation inequalities, and bounds in learning algorithms",
    ),
    "18_cramer_v.py": (
        "Applied Statistics & Correlation",
        "Cramer's V",
        "Strength of association between categorical variables",
    ),
    "19_kendalltaub.py": (
        "Applied Statistics & Correlation",
        "Kendall's Tau-b",
        "Non-parametric rank correlation robust to ties",
    ),
    "20_spurious_correlation.py": (
        "Applied Statistics & Correlation",
        "Spurious Correlation",
        "Confounders, lurking variables, and Simpson's paradox",
    ),
    "21_kruskal_wallis.py": (
        "Applied Statistics & Correlation",
        "Kruskal-Wallis Test",
        "Non-parametric ANOVA for comparing multiple groups",
    ),
    "22_acf_and_pacf.py": (
        "Applied Statistics & Correlation",
        "ACF and PACF",
        "Autocorrelation and partial autocorrelation for time series modeling",
    ),
    "23_ewa_and_bias_correction.py": (
        "Applied Statistics & Correlation",
        "Exponential Moving Averages",
        "EMA smoothing, momentum, and initial bias correction",
    ),
    "24_adjusted_r_squared.py": (
        "Applied Statistics & Correlation",
        "Adjusted R-Squared",
        "Penalizing model complexity in linear regression",
    ),
    "25_predictive_r2.py": (
        "Applied Statistics & Correlation",
        "Predictive R-Squared",
        "Leave-one-out cross-validation and generalization performance",
    ),
    "26_hotelling.py": (
        "Applied Statistics & Correlation",
        "Hotelling's T-Squared",
        "Multivariate hypothesis testing and group mean comparisons",
    ),
    "27_principal_component_analysis.py": (
        "Multivariate Methods & Dimensionality",
        "Principal Component Analysis",
        "Dimensionality reduction via covariance eigendecomposition",
    ),
    "28_factor_analysis.py": (
        "Multivariate Methods & Dimensionality",
        "Factor Analysis",
        "Latent variable modeling and unobserved factor estimation",
    ),
    "29_canonical_correlation_analysis.py": (
        "Multivariate Methods & Dimensionality",
        "Canonical Correlation Analysis",
        "Maximizing correlation between two multidimensional variable sets",
    ),
    "30_correspondence_analysis.py": (
        "Multivariate Methods & Dimensionality",
        "Correspondence Analysis",
        "Geometric visualization of contingency tables and categorical associations",
    ),
    "31_gaussian_mixture_models.py": (
        "Multivariate Methods & Dimensionality",
        "Gaussian Mixture Models",
        "Soft clustering, Expectation-Maximization, and density estimation",
    ),
    "32_elastic_net.py": (
        "Machine Learning Models & Diagnostics",
        "Elastic Net Regression",
        "Balancing L1 and L2 penalties for correlated feature selection",
    ),
    "33_huber_loss.py": (
        "Machine Learning Models & Diagnostics",
        "Huber Loss",
        "Robust regression combining squared and absolute error penalties",
    ),
    "34_mahalanobis_distance.py": (
        "Machine Learning Models & Diagnostics",
        "Mahalanobis Distance",
        "Covariance-scaled distance metrics for outlier detection",
    ),
    "35_gini_impurity_vs_entropy.py": (
        "Machine Learning Models & Diagnostics",
        "Gini Impurity vs Entropy",
        "Split evaluation criteria for decision trees",
    ),
    "36_agglomerative_clustering.py": (
        "Machine Learning Models & Diagnostics",
        "Agglomerative Clustering",
        "Hierarchical clustering, distance metrics, and dendrograms",
    ),
    "37_natural_breaks.py": (
        "Machine Learning Models & Diagnostics",
        "Natural Breaks (Jenks)",
        "1D clustering optimization for histogram and choropleth binning",
    ),
    "38_oversampling.py": (
        "Machine Learning Models & Diagnostics",
        "Oversampling and SMOTE",
        "Synthesizing minority class samples for imbalanced classification",
    ),
    "39_permutation_importance.py": (
        "Machine Learning Models & Diagnostics",
        "Permutation Feature Importance",
        "Model-agnostic feature importance via shuffling evaluation",
    ),
    "40_pca_vs_feat_ag.py": (
        "Machine Learning Models & Diagnostics",
        "PCA vs Feature Agglomeration",
        "Linear dimensionality reduction versus hierarchical feature clustering",
    ),
    "41_pseudo_r2.py": (
        "Machine Learning Models & Diagnostics",
        "Pseudo R-Squared",
        "Goodness-of-fit metrics for logistic regression and GLMs",
    ),
    "42_multiclass_classification.py": (
        "Machine Learning Models & Diagnostics",
        "Multiclass Classification",
        "Softmax functions, cross-entropy loss, and decision regions",
    ),
    "43_energy.py": (
        "Machine Learning Models & Diagnostics",
        "Energy-Based Models",
        "Energy landscapes, Boltzmann distributions, and score matching",
    ),
    "44_logistic_regression.py": (
        "Interpretable AI",
        "Logistic Regression Interpretability",
        "Log-odds, odds ratios, and marginal feature effects",
    ),
    "45_shapley.py": (
        "Interpretable AI",
        "Shapley Values and SHAP",
        "Game-theoretic feature attributions and local model explanations",
    ),
    "46_model_counterfactuals.py": (
        "Interpretable AI",
        "Model Counterfactuals",
        "Actionable recourse and minimal changes to alter model predictions",
    ),
    "47_gelu.py": (
        "Deep Learning & Generative AI",
        "GELU Activation",
        "Gaussian Error Linear Units in modern Transformer models",
    ),
    "48_temperature_scaled_softmax.py": (
        "Deep Learning & Generative AI",
        "Temperature-Scaled Softmax",
        "Calibrating confidence and diversity in probability distributions",
    ),
    "49_focal_loss_balanced.py": (
        "Deep Learning & Generative AI",
        "Focal Loss",
        "Down-weighting easy examples for dense object detection and hard mining",
    ),
    "50_attention_mechanism.py": (
        "Deep Learning & Generative AI",
        "Attention Mechanism",
        "Scaled dot-product attention as value weighting by query-key similarity",
    ),
    "51_causal_attention.py": (
        "Deep Learning & Generative AI",
        "Causal Attention",
        "Autoregressive masking in decoder-only generative models",
    ),
    "52_multi_head_attention.py": (
        "Deep Learning & Generative AI",
        "Multi-Head Attention",
        "Parallel representation subspaces in Transformer blocks",
    ),
    "53_layer_and_rms_normalization.py": (
        "Deep Learning & Generative AI",
        "LayerNorm and RMSNorm",
        "Internal activation scaling and modern variance normalization",
    ),
    "54_decoding_strategies.py": (
        "Deep Learning & Generative AI",
        "Decoding Strategies",
        "Greedy search, beam search, top-k, and nucleus (top-p) sampling",
    ),
    "55_perplexity.py": (
        "Deep Learning & Generative AI",
        "Perplexity",
        "Information-theoretic evaluation metric for autoregressive language models",
    ),
    "56_reparametrization_trick.py": (
        "Deep Learning & Generative AI",
        "Reparameterization Trick",
        "Differentiable sampling through stochastic nodes via auxiliary noise",
    ),
    "57_autoencoder_latent_space.py": (
        "Deep Learning & Generative AI",
        "Autoencoder Latent Space",
        "Deterministic bottleneck compression and feature representation",
    ),
    "58_pca_for_anomaly_detection.py": (
        "Deep Learning & Generative AI",
        "PCA for Anomaly Detection",
        "Reconstruction error in reduced eigenspaces as anomaly scoring",
    ),
    "59_vae_on_mnist.py": (
        "Deep Learning & Generative AI",
        "VAE on MNIST",
        "Variational Autoencoders with evidence lower bound (ELBO) optimization",
    ),
    "60_vae_anomaly_detection.py": (
        "Deep Learning & Generative AI",
        "VAE Anomaly Detection",
        "Probabilistic reconstruction likelihood for out-of-distribution detection",
    ),
    "61_user_item_interaction_matrix.py": (
        "Graphs & Applied Pipelines",
        "User-Item Interaction Matrix",
        "Bipartite graph representations for recommendation systems",
    ),
    "62_grammar_of_graphics.py": (
        "Graphs & Applied Pipelines",
        "Grammar of Graphics",
        "Layered visualization specifications with plotnine",
    ),
    "63_einsum.py": (
        "Programming Patterns & Tools",
        "Einstein Summation (einsum)",
        "Succinct multidimensional array contractions in NumPy and PyTorch",
    ),
    "64_pivoting.py": (
        "Programming Patterns & Tools",
        "Data Pivoting",
        "Reshaping and aggregating tabular datasets in Pandas",
    ),
    "65_cudf.py": (
        "Programming Patterns & Tools",
        "GPU Acceleration (cuDF)",
        "Accelerating dataframe pipelines with GPU memory and parallelism",
    ),
}


PROCESSED_NOTES: list[str] = [
    "00_introduction.py",
    "01_inner_product.py",
    "02_norm_and_metric.py",
    "03_hyperplanes.py",
    "04_rank_one_matrices.py",
    "05_orthogonality.py",
    "06_moore_penrose_inverse.py",
    "07_spectral_decomposition.py",
    "08_matrix_calculus_short.py",
    "09_condition_number.py",
    "10_chebyshev_inequality.py",
    "11_ecdf.py",
    "12_multivariate_normal_distribution.py",
    "13_unbiased_vs_consistent.py",
    "14_dist_of_minimum.py",
    "15_mutual_information.py",
    "16_point_biserial.py",
    "17_jensen_inequality.py",
    "18_cramer_v.py",
    "19_kendalltaub.py",
]


@dataclass(frozen=True)
class Note:
    source: Path
    slug: str
    title: str
    topic: str
    blurb: str
    number: int
    category: str = "General Notes"


def get_all_notes(include_all: bool = False) -> list[Note]:
    """Discover notes in sequential order. By default, exports only processed notes."""
    notes: list[Note] = []

    # 1. General Notes
    notes_dir = ROOT / "general_notes"
    if include_all:
        py_files = sorted(notes_dir.glob("[0-9][0-9]_*.py"))
    else:
        py_files = [notes_dir / name for name in PROCESSED_NOTES if (notes_dir / name).is_file()]

    for file in py_files:
        filename = file.name
        match = re.match(r"^(\d+)_", filename)
        num = int(match.group(1)) if match else 0
        topic, clean_title, blurb = NOTE_METADATA.get(
            filename,
            ("General Notes", filename.replace(".py", "").replace("_", " ").title(), "Data and AI concept note"),
        )
        notes.append(
            Note(
                source=file.relative_to(ROOT),
                slug=f"fundamentals/{file.stem}",
                title=f"Note {num:02d}: {clean_title}",
                topic=topic,
                blurb=blurb,
                number=num,
                category="General Notes",
            )
        )

    # 2. Random / Algorithmic Notes (only if include_all is True)
    if include_all:
        random_dir = ROOT / "random"
        if random_dir.is_dir():
            random_files = sorted(random_dir.glob("*.py"))
            for i, file in enumerate(random_files, start=100):
                title = file.stem.replace("_", " ").title()
                notes.append(
                    Note(
                        source=file.relative_to(ROOT),
                        slug=f"random/{file.stem}",
                        title=f"Random: {title}",
                        topic="Random & Programming Patterns",
                        blurb="Algorithmic problem-solving and programming techniques.",
                        number=i,
                        category="Random",
                    )
                )

    return notes


def export_note(
    note: Note, prev_note: Note | None, next_note: Note | None, output_dir: Path, force: bool = False
) -> Path:
    html_path = output_dir / note.slug / "index.html"
    source_file = ROOT / note.source

    # Skip export if HTML exists and source hasn't been modified since
    if not force and html_path.is_file():
        if html_path.stat().st_mtime >= source_file.stat().st_mtime:
            print(f"skip (up to date): {note.source}")
            return html_path

    html_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "marimo",
        "export",
        "html",
        "--force",
        str(source_file),
        "-o",
        str(html_path),
    ]
    print(f"export {note.source} -> {html_path.relative_to(output_dir)}")
    res = subprocess.run(cmd, cwd=ROOT, check=False)
    if res.returncode != 0:
        print(f"notice: {note.source} exported with code {res.returncode}")

    # Create general_notes/ alias for backwards and URL compatibility
    stem = note.slug.split("/")[-1]
    compat_path = output_dir / "general_notes" / stem / "index.html"
    compat_path.parent.mkdir(parents=True, exist_ok=True)
    redirect_html = f'<!DOCTYPE html><html><head><meta http-equiv="refresh" content="0; url=/{note.slug}/"><link rel="canonical" href="/{note.slug}/"></head><body>Redirecting to <a href="/{note.slug}/">/{note.slug}/</a>...</body></html>'
    compat_path.write_text(redirect_html, encoding="utf-8")

    # Inject order links and rewrite .py links in the exported HTML
    if html_path.is_file():
        content = html_path.read_text(encoding="utf-8")

        # Rewrite internal note .py links (e.g. 01_inner_product.py) to web URLs (/fundamentals/01_inner_product/)
        content = re.sub(
            r'href="(?:\./)?(\d{2}_[a-zA-Z0-9_]+)\.py"',
            r'href="/fundamentals/\1/"',
            content,
        )

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
        if "<body" in content:
            content = re.sub(r"(<body[^>]*>)", r"\1" + nav_header, content, count=1)
        html_path.write_text(content, encoding="utf-8")

    return html_path


def render_index(notes: list[Note]) -> str:
    # Group notes by Topic / Part
    general_notes = [n for n in notes if n.category == "General Notes"]
    random_notes = [n for n in notes if n.category == "Random"]

    topics: dict[str, list[Note]] = {}
    for note in general_notes:
        topics.setdefault(note.topic, []).append(note)

    sections_html = []
    for topic, topic_notes in topics.items():
        items = "\n".join(
            f"        <li>\n"
            f'          <a href="/{note.slug}/">{note.title}</a>\n'
            f'          <p class="blurb">{note.blurb}</p>\n'
            f"        </li>"
            for note in topic_notes
        )
        sections_html.append(f'      <h2>{topic}</h2>\n      <ol class="note-list">\n{items}\n      </ol>')

    # Subject Notes Section
    subject_section = """
      <h2>Subject Notes (Deep Dives)</h2>
      <ol class="note-list">
        <li>
          <span style="font-weight: 600; color: #1f4e79;">Multivariate Analysis</span>
          <p class="blurb">Advanced multivariate statistical techniques and geometric formulations (in progress).</p>
        </li>
        <li>
          <span style="font-weight: 600; color: #1f4e79;">Functional Data Analysis</span>
          <p class="blurb">Infinite-dimensional representations, smoothing, and functional principal components (in progress).</p>
        </li>
      </ol>
    """
    sections_html.append(subject_section)

    # Research Paper Notes Section
    paper_section = """
      <h2>Research Paper Notes</h2>
      <ol class="note-list">
        <li>
          <span style="font-weight: 600; color: #1f4e79;">TabICLv2</span>
          <p class="blurb">Tabular foundation model for in-context learning, classification, and regression (research notes).</p>
        </li>
        <li>
          <span style="font-weight: 600; color: #1f4e79;">LeJEPA</span>
          <p class="blurb">Joint-Embedding Predictive Architecture formulations and explorations (research notes).</p>
        </li>
      </ol>
    """
    sections_html.append(paper_section)

    # Random Notes Section
    if random_notes:
        random_items = "\n".join(
            f"        <li>\n"
            f'          <a href="/{note.slug}/">{note.title}</a>\n'
            f'          <p class="blurb">{note.blurb}</p>\n'
            f"        </li>"
            for note in random_notes
        )
        sections_html.append(
            f'      <h2>Random & Programming Patterns</h2>\n      <ol class="note-list">\n{random_items}\n      </ol>'
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
        background: var(--bg-hover);
      }}
      ol.note-list a {{
        color: var(--accent);
        text-decoration: none;
        font-weight: 600;
        font-size: 1.05rem;
      }}
      ol.note-list a:hover {{
        text-decoration: underline;
      }}
      .blurb {{
        margin: 0.25rem 0 0;
        font-size: 0.95rem;
        color: var(--muted);
      }}
      .foot {{
        margin-top: 4rem;
        padding-top: 1.5rem;
        border-top: 1px solid var(--rule);
        font-size: 0.9rem;
        color: var(--muted);
      }}
      .foot a {{
        color: var(--accent);
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


def build(output_dir: Path, only: str | None, force: bool = False, include_all: bool = False) -> None:
    all_notes = get_all_notes(include_all=include_all)
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

    general_notes = [n for n in all_notes if n.category == "General Notes"]
    for i, note in enumerate(selected):
        if not (ROOT / note.source).is_file():
            raise SystemExit(f"missing notebook: {note.source}")
        if note.category == "General Notes":
            idx = general_notes.index(note)
            prev_note = general_notes[idx - 1] if idx > 0 else None
            next_note = general_notes[idx + 1] if idx < len(general_notes) - 1 else None
        else:
            prev_note = None
            next_note = None
        export_note(note, prev_note, next_note, output_dir, force=force)

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
    parser.add_argument(
        "--force",
        action="store_true",
        help="force export of all notebooks even if unchanged",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="export all 66 notes including unreviewed ones (default: only processed notes)",
    )
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.only, force=args.force, include_all=args.include_all)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
