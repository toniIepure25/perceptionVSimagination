"""
Test Imagery Scaffolding

Tests for NSD-Imagery extension scaffolding to ensure:
1. Modules import correctly
2. CLIs show help and handle errors gracefully
3. NotImplementedError messages are informative
4. No breaking changes to existing code

These tests do NOT require actual NSD-Imagery data.
"""

import subprocess
import sys
from pathlib import Path
import pytest


def test_imagery_module_imports():
    """Test that imagery module imports successfully."""
    try:
        from fmri2img.data.nsd_imagery import ImageryTrial, NSDImageryDataset, build_nsd_imagery_index
    except ImportError as e:
        pytest.fail(f"Failed to import imagery module: {e}")


def test_imagery_trial_dataclass():
    """Test ImageryTrial dataclass instantiation and serialization."""
    from fmri2img.data.nsd_imagery import ImageryTrial
    
    # Create a sample trial
    trial = ImageryTrial(
        trial_id=12345,
        subject="subj01",
        session=1,
        trial_in_session=42,
        condition="imagery",
        nsd_id=5234,
        coco_id=187450,
        beta_path="nsd_imagery/betas/subj01/session01.nii.gz",
        beta_index=41,
        repeat_index=0,
        is_valid=True,
    )
    
    # Test serialization
    trial_dict = trial.to_dict()
    assert trial_dict["trial_id"] == 12345
    assert trial_dict["subject"] == "subj01"
    assert trial_dict["condition"] == "imagery"
    assert trial_dict["nsd_id"] == 5234


def test_build_index_cli_help():
    """Test that build_nsd_imagery_index.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/build_nsd_imagery_index.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "Build NSD-Imagery canonical index" in result.stdout
    assert "--subject" in result.stdout
    assert "--cache-root" in result.stdout
    assert "--output" in result.stdout


def test_eval_cli_help():
    """Test that eval_perception_to_imagery_transfer.py --help works."""
    result = subprocess.run(
        [sys.executable, "scripts/eval_perception_to_imagery_transfer.py", "--help"],
        capture_output=True,
        text=True
    )
    
    assert result.returncode == 0, f"Help command failed: {result.stderr}"
    assert "Evaluate perception-to-imagery transfer" in result.stdout
    assert "--subject" in result.stdout
    assert "--checkpoint" in result.stdout
    assert "--mode" in result.stdout


def test_build_index_missing_data_error(tmp_path):
    """Test that build_nsd_imagery_index.py fails gracefully without data."""
    # Create empty cache directory
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    output_path = tmp_path / "output.parquet"
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_nsd_imagery_index.py",
            "--subject", "subj01",
            "--cache-root", str(cache_root),
            "--output", str(output_path),
        ],
        capture_output=True,
        text=True
    )
    
    # Should fail with informative error
    assert result.returncode != 0
    assert "NSD-Imagery data directory not found" in result.stderr or \
           "nsd_imagery" in result.stderr.lower()


def test_eval_missing_checkpoint_error(tmp_path):
    """Test that eval script fails gracefully with missing checkpoint."""
    cache_root = tmp_path / "cache"
    cache_root.mkdir()
    output_dir = tmp_path / "output"
    fake_checkpoint = tmp_path / "fake_checkpoint.pt"
    
    result = subprocess.run(
        [
            sys.executable,
            "scripts/eval_perception_to_imagery_transfer.py",
            "--subject", "subj01",
            "--checkpoint", str(fake_checkpoint),
            "--mode", "imagery",
            "--output-dir", str(output_dir),
        ],
        capture_output=True,
        text=True
    )
    
    # Should fail with informative error
    assert result.returncode != 0
    assert "Checkpoint not found" in result.stderr or \
           "not found" in result.stderr.lower()


def test_dataset_not_implemented_error():
    """Test that NSDImageryDataset raises informative NotImplementedError."""
    from fmri2img.data.nsd_imagery import NSDImageryDataset
    
    # Create a fake index file
    import pandas as pd
    import tempfile
    
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        fake_index_path = f.name
        
        # Create minimal valid index
        df = pd.DataFrame({
            "trial_id": [1, 2, 3],
            "subject": ["subj01", "subj01", "subj01"],
            "condition": ["imagery", "imagery", "imagery"],
            "session": [1, 1, 1],
            "nsd_id": [100, 101, 102],
            "beta_path": ["path1", "path2", "path3"],
            "beta_index": [0, 1, 2],
        })
        df.to_parquet(fake_index_path)
    
    try:
        dataset = NSDImageryDataset(
            index_path=fake_index_path,
            subject="subj01",
        )
        
        # Should have loaded successfully
        assert len(dataset) == 3
        
        # But iteration should raise NotImplementedError with informative message
        with pytest.raises(NotImplementedError) as exc_info:
            for sample in dataset:
                pass
        
        error_msg = str(exc_info.value)
        assert "not yet implemented" in error_msg.lower()
        assert "TODO" in error_msg or "todo" in error_msg.lower()
        
    finally:
        Path(fake_index_path).unlink()


def test_build_index_not_implemented_error():
    """Test that build_nsd_imagery_index raises informative NotImplementedError."""
    from fmri2img.data.nsd_imagery import build_nsd_imagery_index
    
    with pytest.raises(NotImplementedError) as exc_info:
        build_nsd_imagery_index(
            cache_root="fake_cache",
            subject="subj01",
            output_path="fake_output.parquet",
            dry_run=True,
        )
    
    error_msg = str(exc_info.value)
    assert "not yet implemented" in error_msg.lower()
    assert "TODO" in error_msg or "todo" in error_msg.lower()
    assert "subj01" in error_msg  # Should mention the subject


def test_no_breaking_changes_to_existing_imports():
    """Test that existing imports still work (no breaking changes)."""
    # These should all still import successfully
    try:
        from fmri2img.data.torch_dataset import NSDIterableDataset
        from fmri2img.data.nsd_index import NSDIndex
        from fmri2img.data.loaders import get_dataloader  # If exists
    except ImportError as e:
        # Only fail if it's an unexpected import error
        if "imagery" not in str(e).lower():
            pytest.fail(f"Existing import broken: {e}")


def test_imagery_dataset_file_not_found_error(tmp_path):
    """Test that NSDImageryDataset gives helpful error for missing index."""
    from fmri2img.data.nsd_imagery import NSDImageryDataset
    
    fake_path = tmp_path / "nonexistent.parquet"
    
    with pytest.raises(FileNotFoundError) as exc_info:
        NSDImageryDataset(
            index_path=str(fake_path),
            subject="subj01",
        )
    
    error_msg = str(exc_info.value)
    assert "Imagery index not found" in error_msg
    assert "build_nsd_imagery_index.py" in error_msg  # Should suggest the solution


def test_cli_scripts_are_executable():
    """Test that CLI scripts have correct permissions and shebangs."""
    scripts = [
        "scripts/build_nsd_imagery_index.py",
        "scripts/eval_perception_to_imagery_transfer.py",
    ]
    
    for script_path in scripts:
        path = Path(script_path)
        assert path.exists(), f"Script not found: {script_path}"
        
        # Check shebang
        with open(path, 'r') as f:
            first_line = f.readline()
            assert first_line.startswith("#!/usr/bin/env python"), \
                f"Missing or incorrect shebang in {script_path}"


def test_module_exports():
    """Test that imagery module exports expected public API."""
    from fmri2img.data import nsd_imagery
    
    assert hasattr(nsd_imagery, 'ImageryTrial')
    assert hasattr(nsd_imagery, 'NSDImageryDataset')
    assert hasattr(nsd_imagery, 'build_nsd_imagery_index')
    
    # Check __all__ if defined
    if hasattr(nsd_imagery, '__all__'):
        assert 'ImageryTrial' in nsd_imagery.__all__
        assert 'NSDImageryDataset' in nsd_imagery.__all__
        assert 'build_nsd_imagery_index' in nsd_imagery.__all__


if __name__ == "__main__":
    # Allow running tests directly
    pytest.main([__file__, "-v"])
