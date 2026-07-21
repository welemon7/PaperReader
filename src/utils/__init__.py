try:
    from .arxiv import ArxivDownloader
except ModuleNotFoundError:
    # Keep utility submodules importable even when optional arxiv dependency is absent.
    ArxivDownloader = None

__all__ = ["ArxivDownloader"]
