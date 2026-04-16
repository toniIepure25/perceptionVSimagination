"""Per-condition (imagery/perception) breakdown for all seed-stability models."""
import json
import sys
import numpy as np
import torch
from pathlib import Path


def get_neural_per_condition(config_path, checkpoint_path):
    from fmri2img.workflows.common import (
        build_datasets,
        build_loaders,
        instantiate_model_from_dataset,
        load_workflow_config,
        resolve_runtime_device,
    )
    from fmri2img.training.canonical import load_canonical_checkpoint

    config = load_workflow_config(config_path)
    device = resolve_runtime_device("cuda")
    train_ds, val_ds, test_ds, _, _, _ = build_datasets(config)
    _, _, test_loader = build_loaders(config, train_ds, val_ds, test_ds)
    model = instantiate_model_from_dataset(config, train_ds)
    load_canonical_checkpoint(
        model, checkpoint_path, map_location=device, device=device
    )

    model.to(device)
    model.eval()
    all_preds, all_targets, all_conditions = [], [], []
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to_device(device)
            outputs = model(batch)
            all_preds.append(outputs.content_pred.cpu().numpy())
            all_targets.append(batch.targets.clip_target_768.cpu().numpy())
            all_conditions.append(batch.condition.cpu().numpy())

    pred = np.vstack(all_preds)
    target = np.vstack(all_targets)
    condition = np.concatenate(all_conditions)
    pred_norm = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
    target_norm = target / (np.linalg.norm(target, axis=1, keepdims=True) + 1e-8)
    cosines = np.sum(pred_norm * target_norm, axis=1)

    perc_mask = condition == 0
    imag_mask = condition == 1
    return {
        "overall": float(cosines.mean()),
        "perception": float(cosines[perc_mask].mean()) if perc_mask.sum() > 0 else None,
        "imagery": float(cosines[imag_mask].mean()) if imag_mask.sum() > 0 else None,
        "n_perception": int(perc_mask.sum()),
        "n_imagery": int(imag_mask.sum()),
    }


def main():
    # Ridge per-condition
    with open(
        "outputs/canonical/baselines/full_imagery_overlap_ridge_legacy/test_scores.json"
    ) as f:
        ridge_trials = json.load(f)

    ridge_perc = [t["cosine"] for t in ridge_trials if t["condition"] == "perception"]
    ridge_imag = [t["cosine"] for t in ridge_trials if t["condition"] == "imagery"]
    ridge_all = [t["cosine"] for t in ridge_trials]

    print(
        f"Ridge: overall={np.mean(ridge_all):.5f}, "
        f"perception={np.mean(ridge_perc):.5f} (n={len(ridge_perc)}), "
        f"imagery={np.mean(ridge_imag):.5f} (n={len(ridge_imag)})"
    )

    results = {
        "ridge": {
            "overall": float(np.mean(ridge_all)),
            "perception": float(np.mean(ridge_perc)),
            "imagery": float(np.mean(ridge_imag)),
            "n_perception": len(ridge_perc),
            "n_imagery": len(ridge_imag),
        }
    }

    for model_name, config, seeds in [
        (
            "shared_only",
            "configs/canonical/full_imagery_overlap_shared_only.yaml",
            [0, 42, 123],
        ),
        (
            "sp_p16",
            "configs/canonical/threshold_shared_private_p16.yaml",
            [0, 42, 123],
        ),
    ]:
        folder = model_name
        all_seed_results = []
        for seed in seeds:
            ckpt = f"outputs/seed_stability/{folder}/seed{seed}/best_decoder.pt"
            r = get_neural_per_condition(config, ckpt)
            all_seed_results.append(r)
            print(
                f"{model_name} seed={seed}: overall={r['overall']:.5f}, "
                f"perception={r['perception']:.5f}, imagery={r['imagery']:.5f}"
            )

        avg = {
            "overall": float(np.mean([r["overall"] for r in all_seed_results])),
            "perception": float(np.mean([r["perception"] for r in all_seed_results])),
            "imagery": float(np.mean([r["imagery"] for r in all_seed_results])),
            "overall_std": float(np.std([r["overall"] for r in all_seed_results])),
            "perception_std": float(
                np.std([r["perception"] for r in all_seed_results])
            ),
            "imagery_std": float(np.std([r["imagery"] for r in all_seed_results])),
            "n_perception": all_seed_results[0]["n_perception"],
            "n_imagery": all_seed_results[0]["n_imagery"],
            "per_seed": all_seed_results,
        }
        results[model_name] = avg
        print(
            f"  AVG: overall={avg['overall']:.5f}+/-{avg['overall_std']:.5f}, "
            f"perception={avg['perception']:.5f}+/-{avg['perception_std']:.5f}, "
            f"imagery={avg['imagery']:.5f}+/-{avg['imagery_std']:.5f}"
        )

    out_path = Path("outputs/seed_stability/per_condition_breakdown.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
