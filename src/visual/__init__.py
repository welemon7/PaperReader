"""Visual pipeline: Chromium probing, true-size poster capture, deterministic audit.

Phase 0/1 of the poster quality loop: the harness must *prove* a real browser
rendered the poster at its true print size before a VLM is ever asked to judge
it, and every round persists full-res PNGs, grid overlays, section/figure crops,
review JSON, applied actions and before/after diffs.
"""
