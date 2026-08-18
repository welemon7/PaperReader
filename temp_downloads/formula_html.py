"""Find the formula-box HTML in the generated poster."""
import re
import sys
from pathlib import Path

html = Path(sys.argv[1]).read_text(encoding="utf-8")
for m in re.finditer(r'<div class="formula-box">.{0,600}?</div>\s*</div>\s*</div>', html, re.DOTALL):
    print(repr(m.group(0)[:400]))
    print("----")
