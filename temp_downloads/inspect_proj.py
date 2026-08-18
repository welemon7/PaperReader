"""Inspect sec-project rendering in a generated HTML."""
import re
import sys
from pathlib import Path

html = Path(sys.argv[1]).read_text(encoding="utf-8")
print("len:", len(html))
for m in re.finditer(r'<div class="section-block" id="([^"]+)"', html):
    print("panel:", m.group(1))
print("qr count:", html.count("qr-placeholder"))
print("code-cta count:", html.count("code-cta"))
i = html.find("sec-project")
print(html[i - 80 : i + 1500] if i >= 0 else "sec-project NOT FOUND")
