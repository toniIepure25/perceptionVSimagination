"""
Integration tests for NSD-Imagery extension with fake data.

Tests the complete vertical slice:
1. Build index from fake data
2. Load dataset
3. Evaluate with dry-run mode

This test does NOT require real NSD-Imagery data.
"""

import json
import tempfile
from pathlib import Path
import numpy as np
from PIL import Image
import pytest


def create_fake_imagery_dataset(root_dir: Path, num_trials: int = 10):
    """
    Create a minimal fake NSD-Imagery dataset for testing.
    
    Structure:
        root_dir/
            subj01/
                imagery/
                    run01/
                        trial_001_simple_beta.npy
                        trial_001_simple_image.png
                        trial_002_complex_beta.npy
                        trial_002_complex_image.png
                        metadata.json
                perception/
                    run01/
                        trial_001_beta.npy
                        trial_001_image.png
                        metadata.json
    """
    subj_dir = root_dir / "subj01"
    
    # Create imagery trials
    imagery_dir = subj_dir / "imagery" / "run01"
    imagery_dir.mkdir(parents=True, exist_ok=True)
    
    imagery_trials = []
    for i in range(1, num_trials // 2 + 1):
        trial_id = f"trial_{i:03d}"
        stimulus_type = "simple" if i % 2 == 1 else "complex"
        
        # Beta file (random voxel data)
        beta_path = imagery_dir / f"{trial_id}_{stimulus_type}_beta.npy"
        np.save(beta_path, np.random.randn(15724).astype(np.float32))
        
        # Image file
        img_path = imagery_dir / f"{trial_id}_{stimulus_type}_image.png"
        img = Image.new('RGB', (224, 224), color=(i * 20 % 256, i * 30 % 256, i * 40 % 256))
        img.save(img_path)
        
        imagery_trials.append({
            'trial_id': trial_id,
            'stimulus_type': stimulus_type,
            'run_id': 'run01',
            'condition': 'imagery',
            'text_prompt': f'A {stimulus_type} scene with objects',
        })
    
    # Metadata for imagery
    metadata_path = imagery_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({'trials': imagery_trials}, f, indent=2)
    
    # Create perception trials
    perception_dir = subj_dir / "perception" / "run01"
    perception_dir.mkdir(parents=True, exist_ok=True)
    
    perception_trials = []
    for i in range(1, num_trials // 2 + 1):
        trial_id = f"trial_{i:03d}"
        
        # Beta file
        beta_path = perception_dir / f"{trial_id}_beta.npy"
        np.save(beta_path, np.random.randn(15724).astype(np.float32))
        
        # Image file
        img_path = perception_dir / f"{trial_id}_image.png"
        img = Image.new('RGB', (224, 224), color=(i * 50 % 256, i * 60 % 256, i * 70 % 256))
        img.save(img_path)
        
        perception_trials.append({
            'trial_id': trial_id,
            'run_id': 'run01',
            'condition': 'perception',
            'text_prompt': f'Natural scene {i}',
        })
    
    # Metadata for perception
    metadata_path = perception_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump({'trials': perception_trials}, f, indent=2)
    
    return root_dir


def test_build_index_with_fake_data():
    """Test building index from fake data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake dataset
        data_root = tmpdir / "fake_data"
        create_fake_imagery_dataset(data_root, num_trials=20)
        
        # Build index
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index
        
        output_path = tmpdir / "index.parquet"
        build_nsd_imagery_index(
            data_root=str(data_root),
            subject="subj01",
            output_path=str(output_path),
            cache_root=str(tmpdir / "cache"),
        )
        
        assert output_path.exists(), "Index file not created"
        
        # Validate index
        import pandas as pd
        df = pd.read_parquet(output_path)
        
        assert len(df) == 20, f"Expected 20 trials, got {len(df)}"
        assert 'trial_id' in df.columns
        assert 'condition' in df.columns
        assert 'stimulus_type' in df.columns
        assert 'split' in df.columns
        
        # Check splits
        assert df['split'].value_counts()['train'] >= 10  # 80% of 20 = 16
        assert df['split'].value_counts()['test'] >= 1   # 10% of 20 = 2
        
        print("✓ Index build successful")
        print(f"  Total trials: {len(df)}")
        print(f"  Splits: {df['split'].value_counts().to_dict()}")
        print(f"  Conditions: {df['condition'].value_counts().to_dict()}")


def test_dataset_loading_with_fake_data():
    """Test loading dataset from fake index."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake dataset and index
        data_root = tmpdir / "fake_data"
        create_fake_imagery_dataset(data_root, num_trials=20)
        
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index, NSDImageryDataset
        
        index_path = tmpdir / "index.parquet"
        build_nsd_imagery_index(
            data_root=str(data_root),
            subject="subj01",
            output_path=str(index_path),
            cache_root=str(tmpdir / "cache"),
        )
        
        # Load dataset
        dataset = NSDImageryDataset(
            index_path=str(index_path),
            subject="subj01",
            condition="imagery",
            split_filter="test",
            cache_root=str(tmpdir / "cache"),
            shuffle=False,
        )
        
        # Iterate through samples
        samples = list(dataset)
        assert len(samples) > 0, "Dataset is empty"
        
        # Check first sample
        sample = samples[0]
        assert 'voxels' in sample
        assert 'target_image' in sample
        assert 'target_text' in sample
        assert 'trial_id' in sample
        assert 'stimulus_type' in sample
        
        assert sample['voxels'].shape[0] == 15724, f"Wrong voxel count: {sample['voxels'].shape}"
        assert isinstance(sample['target_image'], Image.Image), "Target image not PIL Image"
        
        print("✓ Dataset loading successful")
        print(f"  Test set size: {len(samples)}")
        print(f"  Sample keys: {list(sample.keys())}")
        print(f"  Voxel shape: {sample['voxels'].shape}")


def test_eval_script_dry_run():
    """Test evaluation script in dry-run mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake dataset and index
        data_root = tmpdir / "fake_data"
        create_fake_imagery_dataset(data_root, num_trials=20)
        
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index
        
        index_path = tmpdir / "index.parquet"
        build_nsd_imagery_index(
            data_root=str(data_root),
            subject="subj01",
            output_path=str(index_path),
            cache_root=str(tmpdir / "cache"),
        )
        
        # Run evaluation in dry-run mode
        import subprocess
        result = subprocess.run([
            'python3', 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', str(index_path),
            '--checkpoint', 'dummy.pt',
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(tmpdir / "outputs"),
            '--dry-run',
        ], capture_output=True, text=True)
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.returncode != 0:
            print("STDERR:")
            print(result.stderr)
        
        assert result.returncode == 0, f"Eval script failed with code {result.returncode}"
        
        # Check outputs
        output_dir = tmpdir / "outputs"
        assert (output_dir / "metrics.json").exists(), "metrics.json not created"
        assert (output_dir / "per_trial.csv").exists(), "per_trial.csv not created"
        assert (output_dir / "README.md").exists(), "README.md not created"
        
        # Validate metrics
        with open(output_dir / "metrics.json") as f:
            metrics = json.load(f)
        
        assert 'clip_cosine_mean' in metrics
        assert 'retrieval@1' in metrics
        
        print("✓ Eval script dry-run successful")
        print(f"  CLIP cosine mean: {metrics['clip_cosine_mean']:.4f}")
        print(f"  Outputs: {list(output_dir.glob('*'))}")


if __name__ == "__main__":
    print("=" * 80)
    print("NSD-IMAGERY INTEGRATION TESTS WITH FAKE DATA")
    print("=" * 80)
    print()
    
    print("Test 1: Build index")
    test_build_index_with_fake_data()
    print()
    
    print("Test 2: Load dataset")
    test_dataset_loading_with_fake_data()
    print()
    
    print("Test 3: Eval script (dry-run)")
    test_eval_script_dry_run()
    print()
    
    print("=" * 80)
    print("✓ ALL TESTS PASSED")
    print("=" * 80)
