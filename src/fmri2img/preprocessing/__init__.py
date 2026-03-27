"""Canonical preprocessing API for decoder workflows."""

from __future__ import annotations

from typing import Any

from fmri2img.data.preprocess import NSDPreprocessor


def describe_preprocessing(preprocessor: NSDPreprocessor | None) -> dict[str, Any]:
    """Serialize the fitted preprocessing contract for reproducible exports."""
    if preprocessor is None:
        return {"enabled": False}
    if hasattr(preprocessor, "summary"):
        summary = preprocessor.summary()
    else:
        summary = {}
    return {
        "enabled": True,
        "subject": getattr(preprocessor, "subject", None),
        "roi_mode": getattr(preprocessor, "roi_mode", None),
        "artifacts_dir": str(getattr(preprocessor, "out_dir", "")),
        "summary": summary,
    }


__all__ = ["NSDPreprocessor", "describe_preprocessing"]
