"""
Utils Package
=============

Configuration loading, structured logging, CLIP model utilities,
and experiment manifest generation.
"""

from .config_loader import ConfigDict, load_config, save_config, print_config
from .logging_utils import (
    setup_logging,
    silence_library_loggers,
    log_time,
    log_memory,
    log_dict,
    create_experiment_logger,
    get_logger,
)
from .clip_utils import (
    load_clip_config,
    load_clip_model,
    encode_images,
    verify_embedding_dimension,
)
from .manifest import gather_env_info, write_manifest, load_manifest

__all__ = [
    # Configuration
    "ConfigDict",
    "load_config",
    "save_config",
    "print_config",
    # Logging
    "setup_logging",
    "silence_library_loggers",
    "log_time",
    "log_memory",
    "log_dict",
    "create_experiment_logger",
    "get_logger",
    # CLIP utilities
    "load_clip_config",
    "load_clip_model",
    "encode_images",
    "verify_embedding_dimension",
    # Manifest
    "gather_env_info",
    "write_manifest",
    "load_manifest",
]
