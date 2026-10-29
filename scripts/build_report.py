"""Build report.pdf from report_src/report.md (Markdown -> HTML -> PDF via headless Chrome).

    uv run --with markdown python scripts/build_report.py
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "report_src" / "report.md"
HTML = ROOT / "report_src" / "report.html"
PDF = ROOT / "report.pdf"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: Letter; margin: 0.7in 0.75in; }
body { font-family: "Charter", "Georgia", serif; font-size: 10.5pt; line-height: 1.42;
       color: #0b0b0b; background: #ffffff; }
h1 { font-size: 19pt; margin: 0 0 4pt; }
h2 { font-size: 14pt; margin: 18pt 0 6pt; break-after: avoid; }
h3 { font-size: 12pt; margin: 14pt 0 5pt; break-after: avoid; }
p, li { text-align: justify; }
hr { border: none; border-top: 1px solid #d9d8d3; margin: 14pt 0; }
img { width: 100%; display: block; margin: 6pt auto 8pt; break-inside: avoid; }
table { border-collapse: collapse; margin: 6pt 0 10pt; font-size: 9pt; width: 100%;
        break-inside: avoid; }
th, td { border-bottom: 1px solid #d9d8d3; padding: 3pt 5pt; text-align: left;
         vertical-align: top; }
th { border-bottom: 1.5px solid #52514e; font-weight: 600; }
code { font-family: "Menlo", monospace; font-size: 8.6pt; background: #f2f1ed;
       padding: 0 2pt; border-radius: 2px; }
h2 + p img, h3 + p img { margin-top: 2pt; }
"""


LIST_ITEM = re.compile(r"^\s*(?:[-*]|\d+\.)\s")


def normalize_lists(text: str) -> str:
    """Python-Markdown needs a blank line before a list and 4-space nesting;
    accept the common GitHub style (no blank line, 3-space nesting) too."""
    out: list[str] = []
    for line in text.splitlines():
        if LIST_ITEM.match(line):
            indent = len(line) - len(line.lstrip(" "))
            if 0 < indent < 4:
                line = "    " + line.lstrip(" ")
            if out and out[-1].strip() and not LIST_ITEM.match(out[-1]) \
                    and not out[-1].startswith("    "):
                out.append("")
        out.append(line)
    return "\n".join(out) + "\n"


def main() -> None:
    body = markdown.markdown(normalize_lists(SRC.read_text()), extensions=["tables"])
    HTML.write_text(f"<!doctype html><html><head><meta charset='utf-8'>"
                    f"<title>Assignment 3 Report</title><style>{CSS}</style></head>"
                    f"<body>{body}</body></html>")
    subprocess.run([CHROME, "--headless", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={PDF}", HTML.as_uri()], check=True,
                   capture_output=True)
    print(f"wrote {PDF}")


if __name__ == "__main__":
    main()
