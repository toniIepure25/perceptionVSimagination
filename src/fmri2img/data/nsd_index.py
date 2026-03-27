"""
Backward-compatible NSD index access.

The canonical code path now prefers ``read_subject_index`` from
``fmri2img.data.nsd_index_reader``. This module preserves the older
``NSDIndex`` name so legacy imports remain stable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .nsd_index_reader import read_subject_index


@dataclass
class NSDIndex:
    """Compatibility wrapper around a subject-specific NSD index."""

    index_root_or_file: str
    subject: str
    allow_fallback_index: bool = False

    def load(self) -> pd.DataFrame:
        return read_subject_index(
            self.index_root_or_file,
            self.subject,
            allow_fallback_index=self.allow_fallback_index,
        )

    @property
    def path(self) -> Path:
        root = Path(self.index_root_or_file)
        if root.suffix == ".parquet":
            return root
        return root / f"subject={self.subject}" / "index.parquet"

