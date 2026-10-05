"""docs/report.md -> docs/report.pdf (A4). Optional; needs pandoc and playwright (Chromium).

    python scripts/build_report_pdf.py docs/report.md docs/report.pdf
"""
import subprocess, sys, os
from playwright.sync_api import sync_playwright

src, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
css = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "docs", "report.css")
html = out.replace(".pdf", ".html")
subprocess.run(["pandoc", src, "-s", "--metadata", "title=roomplan: technical report", "-c", css,
                "--embed-resources", "--standalone", "-o", html], check=True,
               cwd=os.path.dirname(os.path.abspath(src)))
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page()
    pg.goto("file://" + os.path.abspath(html))
    pg.pdf(path=out, format="A4", margin={"top": "14mm", "bottom": "14mm", "left": "15mm", "right": "15mm"},
           print_background=True)
    b.close()
import re
n = len(re.findall(rb"/Type\s*/Page[^s]", open(out, "rb").read()))
print("pages", n)
