"""
Data Package
============

Data loading, preprocessing, indexing, and dataset classes for the NSD
perception and imagery pipelines.
"""

from .preprocess import NSDPreprocessor
from .clip_cache import CLIPCache
from .loaders import (
    DataLoaderFactory,
    FMRIDataset,
    train_val_test_split,
    extract_features_and_targets,
)
from .nsd_index_reader import read_subject_index
from .nsd_imagery import (
    ImageryTrial,
    NSDImageryDataset,
    build_nsd_imagery_index,
)
from .torch_dataset import NSDIterableDataset

__all__ = [
    # Preprocessing
    "NSDPreprocessor",
    # CLIP cache
    "CLIPCache",
    # Data loaders
    "DataLoaderFactory",
    "FMRIDataset",
    "train_val_test_split",
    "extract_features_and_targets",
    # Index
    "read_subject_index",
    # Imagery
    "ImageryTrial",
    "NSDImageryDataset",
    "build_nsd_imagery_index",
    # Torch dataset
    "NSDIterableDataset",
]
