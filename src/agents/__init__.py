"""Lightweight package exports for agent modules.

Importing this package should not eagerly import optional runtime-heavy
dependencies, so tests can import individual agent modules without pulling
in the whole pipeline stack.
"""

__all__ = [
    "run_parse_paper",
    "run_understand_paper",
    "generate_blueprint",
    "analyze_content_importance",
    "plan_poster_story",
    "build_layout_tree",
    "layout_tree_to_blueprint",
    "render_layout_tree",
    "run_poster_v2",
    "run_poster_harness",
    "run_pipeline",
    "validate_poster",
    "optimize_html_with_llm",
    "batch_optimize_html",
    "optimize_with_feedback",
    "optimize_html_string",
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
    if name == "analyze_content_importance":
        from .content_importance import analyze_content_importance
        return analyze_content_importance
    if name == "plan_poster_story":
        from .poster_story_planner import plan_poster_story
        return plan_poster_story
    if name == "build_layout_tree":
        from .poster_v2 import build_layout_tree
        return build_layout_tree
    if name == "layout_tree_to_blueprint":
        from .poster_v2 import layout_tree_to_blueprint
        return layout_tree_to_blueprint
    if name == "render_layout_tree":
        from .poster_v2 import render_layout_tree
        return render_layout_tree
    if name == "run_poster_v2":
        from .poster_v2 import run_poster_v2
        return run_poster_v2
    if name == "run_poster_harness":
        from .poster_harness import run_poster_harness
        return run_poster_harness
    if name == "run_pipeline":
        from .pipeline_agent import run_pipeline
        return run_pipeline
    if name == "validate_poster":
        from .validation_agent import validate_poster
        return validate_poster
    if name == "optimize_html_with_llm":
        from .html_optimizer import optimize_html_with_llm
        return optimize_html_with_llm
    if name == "batch_optimize_html":
        from .html_optimizer import batch_optimize_html
        return batch_optimize_html
    if name == "optimize_with_feedback":
        from .html_optimizer import optimize_with_feedback
        return optimize_with_feedback
    if name == "optimize_html_string":
        from .html_optimizer import optimize_html_string
        return optimize_html_string
    raise AttributeError(name)
