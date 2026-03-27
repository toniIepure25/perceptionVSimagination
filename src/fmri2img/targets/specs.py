from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LatentTargetSpec:
    """Canonical latent target description for decoding workflows."""

    name: str = "vit_l14_image_768"
    dimension: int = 768
    normalized: bool = True
    embedding_column: Optional[str] = None

    @property
    def column_candidates(self) -> tuple[str, ...]:
        if self.embedding_column:
            return (self.embedding_column,)
        return (
            "clip_target_768",
            "clip768",
            "vit_l14_image_768",
            "final",
            "embedding",
        )


class LatentTargetStore:
    """
    Generic latent target store for canonical 768-D ViT-L/14 decoding.

    The existing repo contains multiple embedding cache formats. This loader
    provides a single, explicit contract for the new workflows without making
    hidden assumptions about legacy column names.
    """

    def __init__(
        self,
        cache_path: str | Path,
        spec: LatentTargetSpec | None = None,
        id_column: Optional[str] = None,
    ):
        self.cache_path = Path(cache_path)
        self.spec = spec or LatentTargetSpec()
        self.id_column = id_column
        self._df: Optional[pd.DataFrame] = None
        self._embedding_column: Optional[str] = None

    def load(self) -> "LatentTargetStore":
        if self._df is not None:
            return self
        if not self.cache_path.exists():
            raise FileNotFoundError(
                f"Canonical target cache not found: {self.cache_path}. "
                "Build a 768-D target cache or update targets.cache_path."
            )
        self._df = pd.read_parquet(self.cache_path)
        self._embedding_column = self._resolve_embedding_column()
        id_column = self._resolve_id_column()
        if self._df[id_column].duplicated().any():
            duplicate_ids = self._df.loc[self._df[id_column].duplicated(), id_column].astype(int).tolist()[:10]
            raise ValueError(
                f"Target cache {self.cache_path} contains duplicate {id_column} values, e.g. {duplicate_ids}. "
                "Canonical target lookup requires one embedding per stimulus."
            )
        self._df = self._df.set_index(id_column)
        return self

    def _resolve_id_column(self) -> str:
        if self.id_column is not None:
            if self.id_column not in self._df.columns:
                raise KeyError(
                    f"Configured id column '{self.id_column}' not found in {self.cache_path}"
                )
            return self.id_column
        for candidate in ("nsdId", "nsd_id", "pair_id"):
            if candidate in self._df.columns:
                return candidate
        raise KeyError(f"Could not infer id column for target store {self.cache_path}")

    def _resolve_embedding_column(self) -> str:
        for candidate in self.spec.column_candidates:
            if candidate in self._df.columns:
                return candidate
        raise KeyError(
            f"Could not find a 768-D embedding column in {self.cache_path}. "
            f"Tried: {', '.join(self.spec.column_candidates)}"
        )

    def has(self, nsd_id: int) -> bool:
        self.load()
        return int(nsd_id) in self._df.index

    def get(self, nsd_id: int) -> np.ndarray:
        self.load()
        if int(nsd_id) not in self._df.index:
            raise KeyError(f"nsd_id={nsd_id} missing from {self.cache_path}")
        row = self._df.loc[int(nsd_id), self._embedding_column]
        vector = np.asarray(row, dtype=np.float32).reshape(-1)
        if vector.shape[0] != self.spec.dimension:
            raise ValueError(
                f"Target dimension mismatch for nsd_id={nsd_id}: "
                f"expected {self.spec.dimension}, got {vector.shape[0]}"
            )
        if self.spec.normalized:
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = vector / norm
        return vector

    def get_many(self, nsd_ids: Iterable[int]) -> dict[int, np.ndarray]:
        return {int(nsd_id): self.get(int(nsd_id)) for nsd_id in nsd_ids if self.has(int(nsd_id))}

    def describe(self) -> dict[str, Any]:
        self.load()
        return {
            "path": str(self.cache_path),
            "target_name": self.spec.name,
            "dimension": self.spec.dimension,
            "embedding_column": self._embedding_column,
            "id_column": self._df.index.name,
            "count": int(len(self._df)),
        }
