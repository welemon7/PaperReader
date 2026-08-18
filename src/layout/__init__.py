"""Scene graph and controlled-patch model for poster re-rendering.

The poster is no longer a flat list of sections mutated by CSS hacks.  A
``PosterScene`` is a panel-element graph with explicit constraints; reviewers
may only emit controlled ``ScenePatch`` objects (condense text, resize a
figure, reflow a panel, swap a figure, remove an element), which the harness
validates by re-rendering and re-auditing before keeping them.
"""
