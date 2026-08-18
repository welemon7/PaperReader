"""Print the full sec-project section HTML."""
import sys
from pathlib import Path

html = Path(sys.argv[1]).read_text(encoding="utf-8")
start = html.find('id="sec-project"')
if start < 0:
    print("not found")
    sys.exit(0)
# find the matching closing div for the panel (7 levels deep is overkill; take a generous slice)
print(html[start : start + 2600])
