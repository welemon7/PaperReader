"""Lightweight package exports for agent modules.

Importing this package should not eagerly import optional runtime-heavy
dependencies, so tests can import individual agent modules without pulling
in the whole pipeline stack.
"""

__all__ = [
    "run_parse_paper",
    "run_understand_paper",
    "generate_blueprint",
    "run_pipeline",
    "optimize_poster",
    "validate_poster",
]


def __getattr__(name: str):
    if name == "run_parse_paper":
        from .parse_agent import run_parse_paper
        return run_parse_paper
    if name == "run_understand_paper":
        from .understand_agent import run_understand_paper
        return run_understand_paper
    if name == "generate_blueprint":
        from .poster_planner import generate_blueprint
        return generate_blueprint
    if name == "run_pipeline":
        from .pipeline_agent import run_pipeline
        return run_pipeline
    if name == "optimize_poster":
        from .optimizer_agent import optimize_poster
        return optimize_poster
    if name == "validate_poster":
        from .validation_agent import validate_poster
        return validate_poster
    raise AttributeError(name)
