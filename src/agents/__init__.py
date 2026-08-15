"""Lightweight package exports for agent modules.

Importing this package should not eagerly import optional runtime-heavy
dependencies, so tests can import individual agent modules without pulling
in the whole pipeline stack.
"""

__all__ = [
    "run_parse_paper",
    "run_understand_paper",
    "generate_blueprint",
    "build_layout_tree",
    "layout_tree_to_blueprint",
    "render_layout_tree",
    "review_rendered_poster",
    "evaluate_poster_qa",
    "generate_paperquiz_questions",
    "run_poster_v2",
    "run_poster_harness",
    "run_pipeline",
    "optimize_poster",
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
    if name == "build_layout_tree":
        from .poster_v2 import build_layout_tree
        return build_layout_tree
    if name == "layout_tree_to_blueprint":
        from .poster_v2 import layout_tree_to_blueprint
        return layout_tree_to_blueprint
    if name == "render_layout_tree":
        from .poster_v2 import render_layout_tree
        return render_layout_tree
    if name == "review_rendered_poster":
        from .poster_harness import review_rendered_poster
        return review_rendered_poster
    if name == "evaluate_poster_qa":
        from .poster_harness import evaluate_poster_qa
        return evaluate_poster_qa
    if name == "generate_paperquiz_questions":
        from .poster_harness import generate_paperquiz_questions
        return generate_paperquiz_questions
    if name == "run_poster_v2":
        from .poster_v2 import run_poster_v2
        return run_poster_v2
    if name == "run_poster_harness":
        from .poster_harness import run_poster_harness
        return run_poster_harness
    if name == "run_pipeline":
        from .pipeline_agent import run_pipeline
        return run_pipeline
    if name == "optimize_poster":
        from .optimizer_agent import optimize_poster
        return optimize_poster
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