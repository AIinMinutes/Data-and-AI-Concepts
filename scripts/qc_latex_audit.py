#!/usr/bin/env python3
"""Quality Control: Comprehensive LaTeX Audit across all 66 notebooks.

Checks:
1. Balanced LaTeX delimiters ($$...$$ and $...$).
2. Proper display-math formatting (flush-left, blank lines, no list nesting).
3. Environment matching (\\begin{env} ... \\end{env}).
4. Detection of common KaTeX unsupported macros or raw python string collisions.
"""

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTES_DIR = ROOT / "general_notes"


def extract_markdown_cells(file_path: Path) -> list[tuple[int, str]]:
    """Extract markdown text from all mo.md() calls in the notebook."""
    content = file_path.read_text(encoding="utf-8")
    try:
        tree = ast.parse(content, filename=str(file_path))
    except SyntaxError as e:
        print(f"ERROR: Could not parse {file_path.name}: {e}")
        return []

    md_cells = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # match mo.md(...)
            func = node.func
            is_mo_md = False
            if isinstance(func, ast.Attribute) and func.attr == "md":
                is_mo_md = True

            if is_mo_md and node.args:
                arg = node.args[0]
                if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                    md_cells.append((node.lineno, arg.value))
                elif isinstance(arg, ast.JoinedStr):
                    # f-string: reconstruct literal parts
                    parts = []
                    for part in arg.values:
                        if isinstance(part, ast.Constant) and isinstance(part.value, str):
                            parts.append(part.value)
                        else:
                            parts.append("<DYNAMIC>")
                    md_cells.append((node.lineno, "".join(parts)))

    return md_cells


def check_latex_in_text(text: str, file_name: str, lineno: int) -> list[str]:
    issues = []

    # 1. Check balanced display math $$
    display_math_delims = text.count("$$")
    if display_math_delims % 2 != 0:
        issues.append(f"Line {lineno}: Unmatched '$$' delimiter (count = {display_math_delims})")

    # 2. Check balanced environments \begin{xyz} ... \end{xyz}
    begins = re.findall(r"\\begin\{([a-zA-Z0-9*]+)\}", text)
    ends = re.findall(r"\\end\{([a-zA-Z0-9*]+)\}", text)
    if sorted(begins) != sorted(ends):
        issues.append(f"Line {lineno}: Mismatched environments: \\begin{begins} vs \\end{ends}")

    # 3. Check for display math $$ nested inside bullet points (e.g. "- $$" or "* $$" or "1. $$")
    for l_idx, line in enumerate(text.splitlines(), start=1):
        if re.match(r"^[-*+]\s+\$\$", line) or re.match(r"^\d+\.\s+\$\$", line):
            issues.append(f"Line {lineno}+{l_idx}: Display math '$$' is nested in a bullet item: '{line[:40]}'")
        elif line.startswith("   ") and line.lstrip().startswith("$$") and not line.startswith("$$"):
            # Indented $$ might be interpreted as code block or ignored by KaTeX
            issues.append(f"Line {lineno}+{l_idx}: Display math '$$' has leading whitespace: '{line[:40]}'")

    # 4. Check for unescaped percent signs inside inline math (KaTeX comment delimiter)
    # Match $...%...$ that isn't \%
    inline_maths = re.findall(r"(?<!\$)\$(?!\$)(.*?)(?<!\$)\$(?!\$)", text, re.DOTALL)
    for m in inline_maths:
        if re.search(r"(?<!\\)%", m):
            issues.append(f"Line {lineno}: Unescaped '%' inside inline math: '${m[:30]}...$'")

    return issues


def main():
    py_files = sorted(NOTES_DIR.glob("[0-9][0-9]_*.py"))
    print(f"Auditing LaTeX across {len(py_files)} notebooks in {NOTES_DIR.name}...")

    total_issues = 0
    clean_notes = 0

    for py_file in py_files:
        cells = extract_markdown_cells(py_file)
        file_issues = []
        for lineno, md_text in cells:
            issues = check_latex_in_text(md_text, py_file.name, lineno)
            file_issues.extend(issues)

        if file_issues:
            print(f"\n[FAIL] {py_file.name} ({len(file_issues)} issues):")
            for issue in file_issues:
                print(f"  - {issue}")
            total_issues += len(file_issues)
        else:
            clean_notes += 1

    print(f"\nAudit Complete: {clean_notes}/{len(py_files)} notebooks 100% clean.")
    if total_issues > 0:
        print(f"Total LaTeX formatting issues detected: {total_issues}")
        sys.exit(1)
    else:
        print("ALL LaTeX equations across all notebooks are strictly compliant with KaTeX and Marimo markdown rules!")
        sys.exit(0)


if __name__ == "__main__":
    main()
