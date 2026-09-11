#!/usr/bin/env python3
"""Build the site and push it to AIinMinutes.github.io (learnaiinminutes.com)."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAGES_REPO = "https://github.com/AIinMinutes/AIinMinutes.github.io.git"
PAGES_BRANCH = "gh-pages"
CLONE = ROOT / ".pages"


def run(cmd: list[str], cwd: Path | None = None) -> None:
    subprocess.run(cmd, cwd=cwd or ROOT, check=True)


def ensure_clone() -> Path:
    if (CLONE / ".git").is_dir():
        run(["git", "fetch", "origin", PAGES_BRANCH], cwd=CLONE)
        run(["git", "checkout", PAGES_BRANCH], cwd=CLONE)
        run(["git", "reset", "--hard", f"origin/{PAGES_BRANCH}"], cwd=CLONE)
        return CLONE
    CLONE.parent.mkdir(parents=True, exist_ok=True)
    run(
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            PAGES_BRANCH,
            PAGES_REPO,
            str(CLONE),
        ]
    )
    return CLONE


def copy_site(site: Path, dest: Path) -> None:
    shutil.copy2(site / "index.html", dest / "index.html")
    shutil.copy2(site / "CNAME", dest / "CNAME")
    (dest / ".nojekyll").write_text("")
    for folder in ["general_notes", "fundamentals", "random", "subject_notes", "research_paper_notes"]:
        src = site / folder
        dst = dest / folder
        if src.exists():
            if dst.exists():
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def publish(site: Path, push: bool) -> None:
    dest = ensure_clone()
    copy_site(site, dest)
    run(["git", "add", "-A"], cwd=dest)
    dirty = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=dest).returncode
    if dirty == 0:
        print("nothing to publish")
        return
    if subprocess.run(["git", "config", "user.email"], cwd=dest, capture_output=True).returncode != 0:
        run(["git", "config", "user.name", "@AIinMinutes"], cwd=dest)
        run(["git", "config", "user.email", "aiinminutes.aim@gmail.com"], cwd=dest)
    run(
        [
            "git",
            "commit",
            "-m",
            "Publish Marimo notes to learnaiinminutes.com",
        ],
        cwd=dest,
    )
    if push:
        run(["git", "push", "origin", PAGES_BRANCH], cwd=dest)
        print("pushed to AIinMinutes.github.io gh-pages")
    else:
        print(f"commit ready in {dest} (pass --push to publish)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "_site",
        help="build directory (default: _site)",
    )
    parser.add_argument(
        "--skip-build",
        action="store_true",
        help="use an existing _site instead of rebuilding",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="git push to AIinMinutes.github.io (gh-pages)",
    )
    args = parser.parse_args()
    site = args.output.resolve()
    if not args.skip_build:
        run([sys.executable, str(ROOT / "scripts" / "build_site.py"), "--output", str(site)])
    if not (site / "index.html").is_file():
        raise SystemExit(f"missing built site: {site}")
    try:
        publish(site, push=args.push)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc


if __name__ == "__main__":
    main()
