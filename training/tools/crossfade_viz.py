import argparse
import random
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _add_training_to_syspath() -> None:
    """Ensure `training` package root is importable so we can import `modules.*`."""
    tools_dir = Path(__file__).resolve().parent
    training_root = tools_dir.parent  # .../training
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))

def build_targets(target_frames, num_classes, segments, crossfade_frames):
    # segments: list of (viseme_id, start_frame, end_frame), end exclusive
    targets = np.zeros((target_frames, num_classes), dtype=np.float32)
    for viseme_id, start, end in segments:
        if start >= end:
            continue
        start = max(0, start)
        end = min(target_frames, end)
        # Plateau 1.0
        targets[start:end, viseme_id] = np.maximum(targets[start:end, viseme_id], 1.0)
        # Symmetric crossfade ramps
        if crossfade_frames > 0:
            # Leading ramp 0->1
            lead_start = max(0, start - crossfade_frames)
            lead_len = start - lead_start
            if lead_len > 0:
                alpha = np.linspace(0.0, 1.0, num=lead_len, dtype=np.float32)
                targets[lead_start:start, viseme_id] = np.maximum(
                    targets[lead_start:start, viseme_id], alpha
                )
            # Trailing ramp 1->0
            tail_end = min(target_frames, end + crossfade_frames)
            tail_len = tail_end - end
            if tail_len > 0:
                alpha = np.linspace(1.0, 0.0, num=tail_len, dtype=np.float32)
                targets[end:tail_end, viseme_id] = np.maximum(
                    targets[end:tail_end, viseme_id], alpha
                )
    # Silence (class 0) only where no other viseme is active
    if num_classes > 0:
        non_silence = targets[:, 1:].sum(axis=1) if num_classes > 1 else np.zeros(target_frames, dtype=np.float32)
        silence_active = (non_silence <= 0.0).astype(np.float32)
        targets[:, 0] = np.maximum(targets[:, 0], silence_active)
    return targets

def _demo_mode(output_path: Path) -> None:
    # Example: fps=100, crossfade_ms=30 -> 3 frames
    fps = 100
    crossfade_ms = 30
    crossfade_frames = int(round((crossfade_ms / 1000.0) * fps))

    T = 80
    C = 5  # 0=silence, 1.. are visemes
    # Overlapping segments to illustrate independent ramps and multi-hot overlap
    segments = [
        (1, 10, 30),  # V1
        (2, 28, 45),  # V2 overlaps with V1 (28-30)
        (3, 44, 60),  # V3 overlaps with V2 (44-45)
    ]

    Y = build_targets(T, C, segments, crossfade_frames)

    plt.figure(figsize=(10, 4))
    colors = ["#888888", "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for c in range(C):
        plt.plot(Y[:, c], label=f"V{c}", color=colors[c % len(colors)], linewidth=2 if c else 1, alpha=0.9 if c else 0.7)
    for _, s, e in segments:
        plt.axvline(s, color="#aaaaaa", linestyle="--", linewidth=0.8)
        plt.axvline(e, color="#aaaaaa", linestyle="--", linewidth=0.8)
    plt.title(f"Crossfade visualization (demo, crossfade_frames={crossfade_frames})")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right", ncol=5, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")


def _dataset_mode(config_path: Path, split: str, index: int | None, seed: int | None, output_path: Path) -> None:
    _add_training_to_syspath()
    from modules.config import load_config
    from modules.data_pipeline import LibriSpeechDataset
    import torch

    config = load_config(str(config_path))
    ds = LibriSpeechDataset(config=config, split=split, is_training=False)

    if index is None:
        if seed is not None:
            random.seed(seed)
        index = random.randrange(len(ds))

    sample = ds[index]
    Y = sample.viseme_targets
    if Y.dim() == 1:
        # Convert single-label to one-hot for plotting
        num_classes = config.model.num_visemes
        Y = torch.nn.functional.one_hot(Y, num_classes=num_classes).to(torch.float32)
    T, C = Y.shape
    Y_np = Y.cpu().numpy()

    plt.figure(figsize=(12, 5))
    colors = plt.cm.get_cmap('tab20', C)
    for c in range(C):
        plt.plot(Y_np[:, c], label=f"V{c}", color=colors(c), linewidth=1.5 if c else 1.0, alpha=0.9 if c else 0.7)
    plt.title(f"{split} idx={index}  utterance={getattr(sample, 'utterance_id', 'N/A')}  (T={T}, C={C})")
    plt.xlabel("Frame")
    plt.ylabel("Activation")
    plt.ylim(-0.05, 1.05)
    plt.legend(loc="upper right", ncol=min(8, C))
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    print(f"Saved {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize viseme target crossfade")
    # Default to training/recipes/tcn_config.toml relative to this file
    default_cfg = Path(__file__).resolve().parents[1] / "recipes" / "tcn_config.toml"
    parser.add_argument("--config", type=str, default=str(default_cfg), help="Path to training config TOML")
    parser.add_argument("--split", type=str, default="dev-clean", help="Dataset split (e.g., dev-clean)")
    parser.add_argument("--index", type=int, default=None, help="Specific sample index in split (default: random)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for index selection")
    parser.add_argument("--output", type=str, default="crossfade_viz.png", help="Output PNG path")
    parser.add_argument("--demo", action="store_true", help="Run synthetic demo instead of dataset mode")
    args = parser.parse_args()

    out_path = Path(args.output)
    if args.demo:
        _demo_mode(out_path)
    else:
        _dataset_mode(config_path=Path(args.config), split=args.split, index=args.index, seed=args.seed, output_path=out_path)


if __name__ == "__main__":
    main()