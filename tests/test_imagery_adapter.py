"""
Test Imagery Adapter Training with Synthetic Data
=================================================

Tests the complete adapter training pipeline without requiring real NSD data.
Creates a minimal fake dataset, trains a tiny adapter, and validates outputs.
"""

import json
import sys
import tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
import pandas as pd
import pytest


def create_dummy_base_model(input_dim=50, embed_dim=512):
    """Create a tiny dummy base model for testing."""
    class DummyBaseModel(nn.Module):
        def __init__(self, input_dim, embed_dim):
            super().__init__()
            self.fc = nn.Linear(input_dim, embed_dim)
        
        def forward(self, x):
            out = self.fc(x)
            return torch.nn.functional.normalize(out, dim=-1)
    
    return DummyBaseModel(input_dim, embed_dim)


def create_fake_imagery_dataset(data_root: Path, num_trials=12, input_dim=50):
    """
    Create minimal fake imagery dataset.
    
    Structure:
        data_root/
            subj01/
                run01/
                    beta_001.npy (with target image)
                    beta_002.npy (with target image)
                    ...
                    image_001.png
                    image_002.png
                    ...
                    metadata.json
    """
    subj_dir = data_root / 'subj01'
    run_dir = subj_dir / 'run01'
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Create fake beta files (fMRI data)
    for i in range(1, num_trials + 1):
        beta_data = np.random.randn(input_dim).astype(np.float32)
        np.save(run_dir / f'beta_{i:03d}.npy', beta_data)
    
    # Create fake images (colored squares)
    for i in range(1, num_trials + 1):
        img = Image.new('RGB', (64, 64), color=(i*20 % 255, i*30 % 255, i*40 % 255))
        img.save(run_dir / f'image_{i:03d}.png')
    
    # Create metadata
    metadata = {
        'run_id': 'run01',
        'subject': 'subj01',
        'condition': 'imagery',
        'trials': []
    }
    
    for i in range(1, num_trials + 1):
        trial = {
            'trial_id': f'trial_{i:03d}',
            'stimulus_type': ['simple', 'complex', 'conceptual'][i % 3],
            'has_image': i <= 10,  # First 10 have images
            'has_text': i > 10,    # Last 2 have text prompts
            'text_prompt': f'a test image {i}' if i > 10 else None
        }
        metadata['trials'].append(trial)
    
    with open(run_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return data_root


def save_dummy_checkpoint(checkpoint_path: Path, model: nn.Module, model_type='two_stage'):
    """Save a dummy checkpoint compatible with the loader."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    
    if model_type == 'two_stage':
        # Match TwoStageEncoder checkpoint format
        checkpoint = {
            'state_dict': model.state_dict(),
            'meta': {
                'model_type': 'two_stage',
                'input_dim': 50,
                'latent_dim': 64,
                'embed_dim': 512,
                'n_blocks': 1,
                'head_type': 'linear'
            }
        }
    else:
        checkpoint = {
            'state_dict': model.state_dict(),
            'meta': {
                'model_type': model_type,
                'input_dim': 50,
                'hidden': 64,
                'dropout': 0.1
            }
        }
    
    torch.save(checkpoint, checkpoint_path)


def test_adapter_modules():
    """Test adapter module creation and forward passes."""
    from fmri2img.models.adapters import (
        LinearAdapter, MLPAdapter, ConditionEmbedding, create_adapter
    )
    
    # Test LinearAdapter
    linear_adapter = LinearAdapter(embed_dim=512, use_condition=False)
    x = torch.randn(4, 512)
    out = linear_adapter(x)
    assert out.shape == (4, 512)
    assert torch.allclose(torch.norm(out, dim=-1), torch.ones(4), atol=1e-5)
    
    # Test MLPAdapter
    mlp_adapter = MLPAdapter(embed_dim=512, use_condition=False)
    out = mlp_adapter(x)
    assert out.shape == (4, 512)
    assert torch.allclose(torch.norm(out, dim=-1), torch.ones(4), atol=1e-5)
    
    # Test with condition embedding
    cond_adapter = MLPAdapter(embed_dim=512, use_condition=True)
    condition_idx = torch.tensor([0, 1, 0, 1])
    out = cond_adapter(x, condition_idx=condition_idx)
    assert out.shape == (4, 512)
    
    # Test factory function
    adapter = create_adapter('linear', embed_dim=512)
    assert isinstance(adapter, LinearAdapter)
    
    print("✓ Adapter module tests passed")


def test_training_pipeline_synthetic():
    """Test full training pipeline with synthetic data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create fake dataset
        data_root = tmpdir / 'data'
        create_fake_imagery_dataset(data_root, num_trials=12, input_dim=50)
        
        # Build index
        print("\n" + "="*60)
        print("Building index...")
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index
        
        index_path = tmpdir / 'index.parquet'
        build_nsd_imagery_index(
            data_root=str(data_root),
            subject='subj01',
            cache_root=str(tmpdir / 'cache'),
            output_path=str(index_path),
            stimulus_root=None,
            verbose=False
        )
        
        assert index_path.exists()
        df = pd.read_parquet(index_path)
        assert len(df) > 0
        print(f"✓ Index built: {len(df)} trials")
        
        # Create dummy base model and save checkpoint
        print("\nCreating dummy base model...")
        base_model = create_dummy_base_model(input_dim=50, embed_dim=512)
        checkpoint_path = tmpdir / 'checkpoint.pt'
        save_dummy_checkpoint(checkpoint_path, base_model, model_type='mlp')
        print("✓ Dummy checkpoint saved")
        
        # Train adapter (just 1 epoch on CPU for testing)
        print("\nTraining adapter (1 epoch)...")
        import subprocess
        
        output_dir = tmpdir / 'adapter_output'
        
        cmd = [
            sys.executable, 'scripts/train_imagery_adapter.py',
            '--index', str(index_path),
            '--checkpoint', str(checkpoint_path),
            '--model-type', 'mlp',
            '--adapter', 'linear',
            '--output-dir', str(output_dir),
            '--epochs', '1',
            '--lr', '0.001',
            '--batch-size', '4',
            '--device', 'cpu',
            '--seed', '42',
            '--cache-root', str(tmpdir / 'cache')
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Training failed with return code {result.returncode}")
        
        print("✓ Training completed")
        
        # Verify outputs
        assert (output_dir / 'checkpoints' / 'adapter_best.pt').exists()
        assert (output_dir / 'checkpoints' / 'adapter_last.pt').exists()
        assert (output_dir / 'metrics_train.json').exists()
        assert (output_dir / 'metrics_val.json').exists()
        assert (output_dir / 'config_resolved.yaml').exists()
        
        print("✓ All expected outputs created")
        
        # Load and verify metrics
        with open(output_dir / 'metrics_train.json', 'r') as f:
            train_metrics = json.load(f)
        assert len(train_metrics) > 0
        assert 'loss' in train_metrics[0]
        
        print("✓ Training metrics valid")
        
        # Test evaluation with adapter
        print("\nTesting evaluation...")
        eval_output_dir = tmpdir / 'eval_output'
        
        cmd_eval = [
            sys.executable, 'scripts/eval_perception_to_imagery_transfer.py',
            '--index', str(index_path),
            '--checkpoint', str(checkpoint_path),
            '--model-type', 'mlp',
            '--adapter-checkpoint', str(output_dir / 'checkpoints' / 'adapter_best.pt'),
            '--adapter-type', 'linear',
            '--mode', 'imagery',
            '--split', 'test',
            '--output-dir', str(eval_output_dir),
            '--device', 'cpu',
            '--cache-root', str(tmpdir / 'cache')
        ]
        
        result_eval = subprocess.run(cmd_eval, capture_output=True, text=True)
        
        if result_eval.returncode != 0:
            print(f"STDOUT:\n{result_eval.stdout}")
            print(f"STDERR:\n{result_eval.stderr}")
            pytest.fail(f"Evaluation failed with return code {result_eval.returncode}")
        
        print("✓ Evaluation completed")
        
        # Verify eval outputs
        assert (eval_output_dir / 'metrics.json').exists()
        assert (eval_output_dir / 'per_trial.csv').exists()
        assert (eval_output_dir / 'README.md').exists()
        
        with open(eval_output_dir / 'metrics.json', 'r') as f:
            eval_metrics = json.load(f)
        
        assert 'clip_cosine_mean' in eval_metrics
        assert isinstance(eval_metrics['clip_cosine_mean'], (int, float))
        
        print("✓ Evaluation metrics valid")
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)


def test_ablation_runner_dry_run():
    """Test ablation runner in dry-run mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create minimal fake data
        data_root = tmpdir / 'data'
        create_fake_imagery_dataset(data_root, num_trials=6)
        
        from fmri2img.data.nsd_imagery import build_nsd_imagery_index
        index_path = tmpdir / 'index.parquet'
        build_nsd_imagery_index(
            data_root=str(data_root),
            subject='subj01',
            cache_root=str(tmpdir / 'cache'),
            output_path=str(index_path),
            stimulus_root=None,
            verbose=False
        )
        
        # Create dummy checkpoint
        base_model = create_dummy_base_model()
        checkpoint_path = tmpdir / 'checkpoint.pt'
        save_dummy_checkpoint(checkpoint_path, base_model)
        
        # Run ablation in dry-run mode
        print("\n" + "="*60)
        print("Testing ablation runner (dry-run)...")
        
        import subprocess
        output_dir = tmpdir / 'ablation_output'
        
        cmd = [
            sys.executable, 'scripts/run_imagery_ablations.py',
            '--index', str(index_path),
            '--checkpoint', str(checkpoint_path),
            '--model-type', 'two_stage',
            '--output-dir', str(output_dir),
            '--epochs', '5',
            '--device', 'cpu',
            '--dry-run'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Ablation dry-run failed with return code {result.returncode}")
        
        # Verify commands.txt was created
        assert (output_dir / 'commands.txt').exists()
        
        with open(output_dir / 'commands.txt', 'r') as f:
            commands = f.read()
        assert 'train_imagery_adapter.py' in commands
        assert 'eval_perception_to_imagery_transfer.py' in commands
        
        print("✓ Ablation runner dry-run passed")
        print("="*60)


if __name__ == "__main__":
    print("Running imagery adapter tests...")
    print("")
    
    # Run tests
    test_adapter_modules()
    test_training_pipeline_synthetic()
    test_ablation_runner_dry_run()
    
    print("\n" + "="*80)
    print("✓ ALL TESTS PASSED")
    print("="*80)
