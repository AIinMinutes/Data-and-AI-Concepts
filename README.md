# Data, Stats, and AI Concepts

This repository is a curated collection of data, stats, and AI concepts drawn from highly cited research papers. Each note is a short, runnable [Marimo](https://marimo.io) notebook: the idea, the math, and a small experiment.

**Currently working on:** [TabICLv2](https://arxiv.org/abs/2602.11139) — a tabular foundation model for in-context learning (classification and regression).

Notes are published at [learnaiinminutes.com](https://learnaiinminutes.com/).

## Running the notebooks

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Python 3.10–3.12 is required. From the repo root:

```bash
uv sync
```

That creates `.venv` and installs everything, including [Marimo](https://marimo.io).

```bash
uv run marimo edit
# or a single file:
uv run marimo edit notebooks/fundamentals/applied-statistics/acf_and_pacf.py
```

`notebooks/fundamentals/programming/optimization/cudf.py` additionally needs Linux + CUDA 12:

```bash
uv pip install --extra-index-url https://pypi.nvidia.com cudf-cu12 "polars[gpu]"
```

## Fundamentals

Notes that are **not** tied to a specific paper live in [`notebooks/fundamentals/`](notebooks/fundamentals/). They cover stats, ML, deep learning, and generative-AI primitives.

Paper-specific work (starting with TabICLv2) will sit next to that folder.

## Contributing

Contributions are welcome. Open an issue or a pull request if you have a concept, a paper, or a fix.

## License

- **Code** is licensed under the **MIT License**.
- **Content** (text, explanations, visualizations) is licensed under **Creative Commons Attribution 4.0 (CC BY 4.0)**. You may reuse it with attribution.

See [LICENSE.md](LICENSE.md).
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

## Contact

Email: AIinMinutes@icloud.com

Threads: [@AIinMinutes](https://www.threads.net/@AIinMinutes)
