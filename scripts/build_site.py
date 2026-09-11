#!/usr/bin/env python3
"""Export selected Marimo notebooks to the static site for learnaiinminutes.com."""

from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Note:
    source: Path
    slug: str
    title: str
    topic: str
    blurb: str


# Allowlist: a notebook is published only when it is listed here.
NOTES: tuple[Note, ...] = (
    Note(
        source=Path("notebooks/fundamentals/visualization/grammar_of_graphics.py"),
        slug="fundamentals/visualization/grammar-of-graphics",
        title="Grammar of graphics in plotnine",
        topic="Visualization",
        blurb="A plot is a mapping from a table to visual properties. Bill length against bill depth, then the same points with species as a mapping.",
    ),
)


def export_note(note: Note, output_dir: Path) -> Path:
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
    return html_path


def render_index(notes: list[Note]) -> str:
    items = "\n".join(
        (
            f'        <li>\n'
            f'          <a href="/{note.slug}/">{note.title}</a>\n'
            f'          <p class="blurb">{note.blurb}</p>\n'
            f'          <p class="meta">{note.topic}</p>\n'
            f'        </li>'
        )
        for note in notes
    )
    return f"""<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Learn AI in Minutes</title>
    <style>
      :root {{
        --ink: #1a1a1a;
        --muted: #5a5a5a;
        --accent: #1f4e79;
        --rule: #e6e6e6;
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
        max-width: 40rem;
        margin: 0 auto;
        padding: 3.5rem 1.25rem 4rem;
      }}
      h1 {{
        font-size: 1.85rem;
        font-weight: 600;
        letter-spacing: -0.02em;
        margin: 0 0 0.6rem;
      }}
      .lede {{
        color: var(--muted);
        margin: 0 0 2.25rem;
      }}
      h2 {{
        font-size: 0.78rem;
        font-weight: 600;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--muted);
        border-top: 1px solid var(--rule);
        padding-top: 1.4rem;
        margin: 0 0 1rem;
      }}
      ol {{
        list-style: none;
        padding: 0;
        margin: 0;
      }}
      li + li {{
        margin-top: 1.35rem;
      }}
      a {{
        color: var(--accent);
        text-decoration: none;
      }}
      a:hover {{ text-decoration: underline; }}
      li > a {{
        font-size: 1.15rem;
        font-weight: 600;
      }}
      .blurb {{
        margin: 0.3rem 0 0.15rem;
      }}
      .meta, .foot {{
        color: var(--muted);
        font-size: 0.92rem;
        margin: 0;
      }}
      .foot {{
        margin-top: 2.5rem;
      }}
    </style>
  </head>
  <body>
    <main>
      <h1>Learn AI in Minutes</h1>
      <p class="lede">
        Data, stats, and AI concepts from research papers. Each note is a short
        Marimo notebook: the idea, the math, and a small experiment.
      </p>
      <h2>Notes</h2>
      <ol>
{items}
      </ol>
      <p class="foot">
        Earlier Jupyter Book notes remain at
        <a href="/content/intro.html">/content/intro.html</a>.
        Source:
        <a href="https://github.com/AIinMinutes/Data-and-AI-Concepts">Data-and-AI-Concepts</a>.
      </p>
    </main>
  </body>
</html>
"""


def build(output_dir: Path, only: str | None) -> None:
    selected = [
        note
        for note in NOTES
        if only is None or only in {note.source.stem, note.slug, note.source.name}
    ]
    if not selected:
        raise SystemExit(f"no notes matched {only!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / ".nojekyll").write_text("")
    (output_dir / "CNAME").write_text("learnaiinminutes.com\n")

    for note in selected:
        if not (ROOT / note.source).is_file():
            raise SystemExit(f"missing notebook: {note.source}")
        export_note(note, output_dir)

    # Homepage lists the full allowlist, even when --only exports a subset.
    (output_dir / "index.html").write_text(render_index(list(NOTES)))
    print(f"site -> {output_dir}")


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
        help="export a single notebook (stem, filename, or slug)",
    )
    args = parser.parse_args()
    try:
        build(args.output.resolve(), args.only)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
