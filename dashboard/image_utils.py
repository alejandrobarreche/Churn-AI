"""
Utility for embedding local images as base64 data-URLs in Streamlit HTML.
Falls back to an emoji string if the file does not exist.
"""
import base64
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=None)
def _b64(path: str) -> str:
    p = Path(path)
    if not p.exists():
        return ""
    suffix = p.suffix.lstrip(".").lower()
    mime = "image/svg+xml" if suffix == "svg" else f"image/{suffix}"
    data = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{data}"


def img_html(path: str, size: str = "32px", style: str = "", fallback: str = "") -> str:
    """
    Return an <img> tag with the image embedded as base64.
    If the file is missing, returns `fallback` (e.g. an emoji string).
    """
    src = _b64(path)
    if not src:
        return fallback
    return (
        f'<img src="{src}" width="{size}" height="{size}" alt="" '
        f'style="vertical-align:middle;{style}">'
    )
