import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import sys
# Ensure project root is on sys.path so `training` package can be imported when run via Streamlit
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from training.modules.config import load_config, TrainingConfiguration
from training.modules.data_pipeline import AudioProcessor
from training.modules.tcn_model import create_model


# -------------------------------
# Helpers: data discovery
# -------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PREPARED_DIR = PROJECT_ROOT / "training" / "data" / "prepared"
RUNS_DIR = PROJECT_ROOT / "training" / "runs"
DEFAULT_CONFIG = PROJECT_ROOT / "training" / "recipes" / "tcn_config.toml"


def list_splits() -> List[str]:
    if not PREPARED_DIR.exists():
        return []
    return sorted([p.name for p in PREPARED_DIR.iterdir() if p.is_dir()])


def list_samples(split: str) -> List[str]:
    split_dir = PREPARED_DIR / split
    if not split_dir.exists():
        return []
    return sorted([p.stem for p in split_dir.glob("*.wav")])


def find_run_checkpoints() -> Dict[str, Path]:
    runs: Dict[str, Path] = {}
    if not RUNS_DIR.exists():
        return runs
    for run_dir in RUNS_DIR.iterdir():
        if not run_dir.is_dir():
            continue
        ckpt_dir = run_dir / "checkpoints"
        if ckpt_dir.exists():
            best = ckpt_dir / "best_model.pt"
            if best.exists():
                runs[f"{run_dir.name} (best)"] = best
            # pick latest checkpoint too
            all_ckpts = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), key=lambda p: p.stat().st_mtime)
            if all_ckpts:
                runs[f"{run_dir.name} (latest)"] = all_ckpts[-1]
    return runs


def load_alignment(json_path: Path) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float, str]]]:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    words = []
    phones = []
    tiers = data.get("tiers", {})
    for start, end, label in tiers.get("words", {}).get("entries", []):
        words.append((float(start), float(end), str(label)))
    for start, end, label in tiers.get("phones", {}).get("entries", []):
        phones.append((float(start), float(end), str(label)))
    return words, phones


def phonemes_to_viseme_intervals(
    phones: List[Tuple[float, float, str]], mapping: Dict[str, int]
) -> List[Tuple[float, float, int]]:
    intervals: List[Tuple[float, float, int]] = []
    for start, end, ph in phones:
        vis = mapping.get(ph, 0)
        if intervals and intervals[-1][2] == vis and abs(intervals[-1][1] - start) < 1e-6:
            # merge contiguous same-viseme interval
            prev = intervals[-1]
            intervals[-1] = (prev[0], end, vis)
        else:
            intervals.append((start, end, vis))
    return intervals


def load_viseme_index_to_name(config: TrainingConfiguration) -> Dict[int, str]:
    """Build viseme index -> name mapping from the configured viseme map JSON."""
    cfg_path = Path(config.config_path)
    mapping_path = cfg_path.parent / config.data.phoneme_viseme_map
    if not mapping_path.exists():
        alt = Path(config.data.phoneme_viseme_map)
        mapping_path = alt if alt.exists() else PROJECT_ROOT / config.data.phoneme_viseme_map
    index_to_name: Dict[int, str] = {}
    try:
        with open(mapping_path, "r", encoding="utf-8") as f:
            mapping_data = json.load(f)
        name_to_index: Dict[str, int] = mapping_data["viseme_set"]["visemes"]
        for name, idx in name_to_index.items():
            index_to_name[int(idx)] = name
    except Exception:
        # Fallback to numeric labels if mapping can't be loaded
        pass
    return index_to_name


def load_run_config_for_checkpoint(ckpt_path: Path) -> Optional[TrainingConfiguration]:
    # Prefer logger-saved config.json at run root
    run_dir = ckpt_path.parent.parent
    run_config_json = run_dir / "config.json"
    print(f"[DEBUG] Looking for config at: {run_config_json}")
    if run_config_json.exists():
        try:
            with open(run_config_json, "r") as f:
                cfg = json.load(f)
            print(f"[DEBUG] Loaded config.json, multi_label = {cfg.get('training', {}).get('multi_label', 'NOT_FOUND')}")
            
            # Handle backward compatibility for field name changes
            audio_cfg = cfg["audio"].copy()
            # Remove old field names that are no longer supported
            removed_fields = []
            if "hop_length_samples" in audio_cfg:
                audio_cfg.pop("hop_length_samples")
                removed_fields.append("hop_length_samples")
            if "window_length_samples" in audio_cfg:
                audio_cfg.pop("window_length_samples") 
                removed_fields.append("window_length_samples")
            if "fps" in audio_cfg:
                audio_cfg.pop("fps")
                removed_fields.append("fps")
            if removed_fields:
                print(f"[DEBUG] Removed old audio fields: {removed_fields}")
            
            from training.modules.config import (
                TrainingConfiguration,
                ModelConfig,
                AudioConfig,
                TrainingConfig,
                DataConfig,
                EvaluationConfig,
                HardwareConfig,
                LoggingConfig,
                TensorBoardConfig,
                ExperimentConfig,
            )

            config = TrainingConfiguration(
                model=ModelConfig(**cfg["model"]),
                audio=AudioConfig(**audio_cfg),
                training=TrainingConfig(**cfg["training"]),
                data=DataConfig(**cfg["data"]),
                evaluation=EvaluationConfig(**cfg["evaluation"]),
                hardware=HardwareConfig(**cfg["hardware"]),
                logging=LoggingConfig(**cfg["logging"]),
                tensorboard=TensorBoardConfig(**cfg["tensorboard"]),
                experiment=ExperimentConfig(**cfg["experiment"]),
                config_path=str(DEFAULT_CONFIG),
            )
            print(f"[DEBUG] Successfully created TrainingConfiguration from checkpoint, multi_label = {config.training.multi_label}")
            return config
        except Exception as e:
            print(f"[DEBUG] Exception loading config: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[DEBUG] Config file does not exist: {run_config_json}")
    print(f"[DEBUG] Falling back to default config")
    return None


@st.cache_resource(show_spinner=False)
def get_base_config() -> TrainingConfiguration:
    config = load_config(DEFAULT_CONFIG)
    print(f"[DEBUG] Loaded default config from {DEFAULT_CONFIG}, multi_label = {config.training.multi_label}")
    return config


def compute_mel(audio_path: Path, base_config: TrainingConfiguration) -> np.ndarray:
    processor = AudioProcessor(base_config)
    with torch.no_grad():
        feats = processor.load_and_process_audio(audio_path)
    return feats.cpu().numpy()  # (T, n_mels)


def run_model_on_mel(
    mel: np.ndarray, config: TrainingConfiguration, ckpt_path: Path
) -> Tuple[np.ndarray, np.ndarray]:
    model = create_model(config)
    try:
        checkpoint = torch.load(ckpt_path, map_location=config.hardware.device, weights_only=True)
        state = checkpoint.get("model_state_dict", checkpoint)
    except Exception:
        # Fall back to full unpickling for locally created checkpoints
        # Some older checkpoints reference module path 'modules.*'; alias it to 'training.modules'
        try:
            import sys
            import training.modules as _tm
            sys.modules.setdefault("modules", _tm)  # alias for pickle compatibility
        except Exception:
            pass
        checkpoint = torch.load(ckpt_path, map_location=config.hardware.device, weights_only=False)
        state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        x = torch.from_numpy(mel).float().unsqueeze(0).to(config.hardware.device)
        logits = model(x)
        # Respect training mode: multi-label uses sigmoid; single-label uses softmax
        multi_label_setting = bool(getattr(config.training, "multi_label", False))
        print(f"[DEBUG] Model inference using multi_label = {multi_label_setting}")
        if multi_label_setting:
            probs_t = torch.sigmoid(logits)
            print(f"[DEBUG] Applied sigmoid activation")
        else:
            probs_t = torch.softmax(logits, dim=-1)
            print(f"[DEBUG] Applied softmax activation")
        probs = probs_t.squeeze(0).cpu().numpy()  # (T, C)
        print(f"[DEBUG] First frame probabilities sum: {probs[0].sum():.4f}")
        print(f"[DEBUG] Sample frame probabilities: {probs[0][:5]}")
        preds = np.argmax(probs, axis=-1)
    # Ensure model outputs align with mel length
    t_mel = mel.shape[0]
    t_model = preds.shape[0]
    t = min(t_mel, t_model)
    if t != t_model:
        preds = preds[:t]
        probs = probs[:t, :]
    return preds, probs  # (T,), (T, C)


def ema_smooth_probs(probs: np.ndarray, alpha: float) -> np.ndarray:
    """Apply EMA smoothing along the time axis on model probability outputs.

    Args:
        probs: Array shaped (T, C) containing per-frame class probabilities.
        alpha: Smoothing amount in [0, 1]. 0 = no smoothing, 1 = max smoothing.

    Returns:
        Smoothed array of same shape as probs.
    """
    if probs is None or probs.size == 0:
        return probs
    alpha = float(np.clip(alpha, 0.0, 1.0))
    if alpha == 0.0:
        return probs
    smoothed = probs.copy()
    for t in range(1, smoothed.shape[0]):
        smoothed[t] = (1.0 - alpha) * probs[t] + alpha * smoothed[t - 1]
    return smoothed


def median_filter_preds(preds: np.ndarray, window_size: int) -> np.ndarray:
    """Apply 1D median filter to integer label sequence.

    Ensures odd window size; pads with edge values.
    """
    if preds is None or preds.size == 0:
        return preds
    w = int(window_size)
    if w <= 1:
        return preds
    if w % 2 == 0:
        w += 1
    pad = w // 2
    padded = np.pad(preds, (pad, pad), mode="edge")
    out = np.empty_like(preds)
    for i in range(preds.shape[0]):
        out[i] = int(np.median(padded[i:i + w]))
    return out


def sanitize_probs(probs: np.ndarray, silence_index: Optional[int] = None, multi_label: bool = False) -> np.ndarray:
    """Clean up model probabilities before smoothing.

    - Replace NaN/Inf with 0
    - Clip to [0, 1]
    - For single-label: renormalize each frame to sum to 1; if a frame sums to 0, set one-hot to `silence_index` if provided, otherwise uniform
    - For multi-label: do not renormalize (allow multiple visemes); just return clipped values
    """
    if probs is None or probs.size == 0:
        return probs
    cleaned = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
    cleaned = np.clip(cleaned, 0.0, 1.0)
    if bool(multi_label):
        # Keep independent probabilities; no per-frame renormalization
        return cleaned
    row_sums = cleaned.sum(axis=1)
    bad_rows = ~np.isfinite(row_sums) | (row_sums <= 0.0)
    if np.any(bad_rows):
        if silence_index is not None and 0 <= silence_index < cleaned.shape[1]:
            cleaned[bad_rows, :] = 0.0
            cleaned[bad_rows, silence_index] = 1.0
        else:
            cleaned[bad_rows, :] = 1.0 / cleaned.shape[1]
        row_sums = cleaned.sum(axis=1)
    cleaned = cleaned / np.maximum(row_sums[:, None], 1e-8)
    return cleaned


def sharpen_threshold_probs(
    probs: np.ndarray,
    gamma: float,
    threshold: float,
    silence_index: Optional[int] = None,
    norm_mode: str = "sum",
) -> np.ndarray:
    """Exponentially sharpen, threshold, and renormalize probabilities.

    Args:
        probs: (T, C) probabilities.
        gamma: exponent >= 1.0 to sharpen distributions.
        threshold: absolute cutoff; values below set to 0 before renorm.
        silence_index: fallback class when a row becomes all-zero.
    """
    if probs is None or probs.size == 0:
        return probs
    g = max(1.0, float(gamma))
    t = max(0.0, float(threshold))
    # Exponential boost
    eps = 1e-8
    boosted = np.power(np.clip(probs, 0.0, 1.0) + eps, g)
    # Threshold
    if t > 0.0:
        boosted[boosted < t] = 0.0
    # Normalize per-frame according to mode
    mode = (norm_mode or "sum").lower()
    if mode == "sum":
        row_sums = boosted.sum(axis=1)
        bad_rows = ~np.isfinite(row_sums) | (row_sums <= 0.0)
        if np.any(bad_rows):
            if silence_index is not None and 0 <= silence_index < boosted.shape[1]:
                boosted[bad_rows, :] = 0.0
                boosted[bad_rows, silence_index] = 1.0
            else:
                boosted[bad_rows, :] = 1.0 / boosted.shape[1]
            row_sums = boosted.sum(axis=1)
        boosted = boosted / np.maximum(row_sums[:, None], 1e-8)
    elif mode == "max":
        row_max = boosted.max(axis=1)
        bad_rows = ~np.isfinite(row_max) | (row_max <= 0.0)
        if np.any(bad_rows):
            if silence_index is not None and 0 <= silence_index < boosted.shape[1]:
                boosted[bad_rows, :] = 0.0
                boosted[bad_rows, silence_index] = 1.0
            else:
                boosted[bad_rows, :] = 1.0 / boosted.shape[1]
            row_max = boosted.max(axis=1)
        boosted = boosted / np.maximum(row_max[:, None], 1e-8)
    else:
        # None: keep values as-is (already clipped to [0,1]); ensure not-all-zero rows
        row_sums = boosted.sum(axis=1)
        bad_rows = ~np.isfinite(row_sums) | (row_sums <= 0.0)
        if np.any(bad_rows):
            if silence_index is not None and 0 <= silence_index < boosted.shape[1]:
                boosted[bad_rows, :] = 0.0
                boosted[bad_rows, silence_index] = 1.0
            else:
                boosted[bad_rows, :] = 1.0 / boosted.shape[1]
    return boosted


def plot_tracks(
    mel: np.ndarray,
    words: List[Tuple[float, float, str]],
    phones: List[Tuple[float, float, str]],
    visemes: List[Tuple[float, float, int]],
    preds: Optional[np.ndarray],
    fps: float,
    viseme_index_to_name: Dict[int, str],
    show_words: bool,
    show_phones: bool,
    show_visemes: bool,
    show_preds: bool,
) -> go.Figure:
    def compress_sequence_to_intervals(seq: np.ndarray, fps_val: float) -> List[Tuple[float, float, int]]:
        intervals: List[Tuple[float, float, int]] = []
        if seq is None or len(seq) == 0:
            return intervals
        start_idx = 0
        current = int(seq[0])
        for i in range(1, len(seq)):
            val = int(seq[i])
            if val != current:
                intervals.append((start_idx / fps_val, i / fps_val, current))
                start_idx = i
                current = val
        intervals.append((start_idx / fps_val, len(seq) / fps_val, current))
        return intervals

    time_axis = np.arange(mel.shape[0]) / fps
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.133, 0.133, 0.133],
        vertical_spacing=0.02,
        subplot_titles=("Log-Mel", "Words", "Phonemes", "Visemes / Model")
    )

    # Mel heatmap
    fig.add_trace(
        go.Heatmap(
            x=time_axis,
            y=np.arange(mel.shape[1]),
            z=mel.T,
            colorscale="Viridis",
            showscale=False,
        ),
        row=1, col=1,
    )

    def add_intervals(intervals, row, label_key: str):
        for s, e, lbl in intervals:
            if s == e:
                continue
            fig.add_shape(
                type="rect",
                x0=s, x1=e, y0=0, y1=1,
                xref="x", yref=f"y{row}",
                line=dict(width=0), fillcolor="rgba(0, 150, 255, 0.2)",
            )
            fig.add_annotation(
                x=(s + e) / 2.0, y=0.5, text=str(lbl), showarrow=False,
                xref="x", yref=f"y{row}", font=dict(size=14)
            )

    # Words
    if show_words:
        add_intervals(words, row=2, label_key="word")
        fig.update_yaxes(range=[0, 1], showticklabels=False, row=2, col=1)

    # Phonemes
    if show_phones:
        add_intervals(phones, row=3, label_key="phone")
        fig.update_yaxes(range=[0, 1], showticklabels=False, row=3, col=1)

    # Visemes (targets)
    if show_visemes:
        for s, e, v in visemes:
            if s == e:
                continue
            fig.add_shape(
                type="rect",
                x0=s, x1=e, y0=0.0, y1=0.48,
                xref="x", yref="y4",
                line=dict(width=0), fillcolor="rgba(0, 200, 100, 0.30)",
            )
            fig.add_annotation(
                x=(s + e) / 2.0, y=0.24, text=viseme_index_to_name.get(int(v), str(v)), showarrow=False,
                xref="x", yref="y4", font=dict(size=14)
            )

    # Predicted visemes rendered as intervals in the upper half of the lane
    if show_preds and preds is not None and len(preds) > 0:
        pred_intervals = compress_sequence_to_intervals(preds, fps)
        for s, e, v in pred_intervals:
            if s == e:
                continue
            fig.add_shape(
                type="rect",
                x0=s, x1=e, y0=0.52, y1=1.0,
                xref="x", yref="y4",
                line=dict(width=0), fillcolor="rgba(0, 100, 255, 0.30)",
            )
            fig.add_annotation(
                x=(s + e) / 2.0, y=0.76, text=viseme_index_to_name.get(int(v), str(v)), showarrow=False,
                xref="x", yref="y4", font=dict(size=14)
            )

        # Add legend entries for clarity
        fig.add_trace(
            go.Scatter(x=[time_axis[0]], y=[0.0], mode="lines",
                       line=dict(color="rgba(0, 200, 100, 0.9)", width=10),
                       name="Target viseme"),
            row=4, col=1,
        )
        fig.add_trace(
            go.Scatter(x=[time_axis[0]], y=[0.0], mode="lines",
                       line=dict(color="rgba(0, 100, 255, 0.9)", width=10),
                       name="Predicted viseme"),
            row=4, col=1,
        )
    fig.update_yaxes(title_text="Mel bin", row=1, col=1)
    fig.update_yaxes(range=[0, 1], showticklabels=False, row=4, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4, col=1)
    fig.update_layout(height=900, hovermode="x unified")
    return fig


def viseme_intervals_to_sequence(
    intervals: List[Tuple[float, float, int]], total_frames: int, fps: float
) -> np.ndarray:
    seq = np.zeros((total_frames,), dtype=int)
    for s, e, v in intervals:
        start_idx = max(0, int(np.floor(s * fps)))
        end_idx = min(total_frames, int(np.ceil(e * fps)))
        if end_idx > start_idx:
            seq[start_idx:end_idx] = int(v)
    return seq


def plot_per_viseme_timelines(
    target_seq: np.ndarray,
    probs: Optional[np.ndarray],
    fps: float,
    viseme_index_to_name: Dict[int, str],
) -> go.Figure:
    num_frames = target_seq.shape[0]
    num_visemes = max(viseme_index_to_name.keys()) + 1 if viseme_index_to_name else int(probs.shape[1] if probs is not None else target_seq.max() + 1)
    time_axis = np.arange(num_frames) / fps

    # Build two layered heatmaps: targets (binary) in green, model probs in blue
    rows = num_visemes * 2
    target_grid = np.full((rows, num_frames), np.nan, dtype=float)
    probs_grid = np.full((rows, num_frames), np.nan, dtype=float)

    for v in range(num_visemes):
        top_row = 2 * v
        bot_row = 2 * v + 1
        target_grid[top_row, :] = (target_seq == v).astype(float)
        if probs is not None and v < probs.shape[1]:
            probs_grid[bot_row, :] = probs[:, v]

    # y positions
    y_vals = np.arange(rows)
    y_centers = np.array([2 * v + 0.5 for v in range(num_visemes)])
    y_center_labels = [viseme_index_to_name.get(v, str(v)) for v in range(num_visemes)]

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            x=time_axis,
            y=y_vals,
            z=target_grid,
            colorscale=[
                [0.0, "rgb(0,0,0)"],
                [1.0, "rgb(0,255,0)"]
            ],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            name="Target (top)",
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Heatmap(
            x=time_axis,
            y=y_vals,
            z=probs_grid,
            colorscale=[
                [0.0, "rgb(0,0,0)"],
                [1.0, "rgb(255,0,0)"]
            ],
            zmin=0.0,
            zmax=1.0,
            showscale=False,
            name="Model (bottom)",
            showlegend=True,
        )
    )

    fig.update_yaxes(
        tickmode="array", tickvals=y_centers, ticktext=y_center_labels,
        autorange="reversed",
        gridcolor="#333333",
        tickfont=dict(color="#FFFFFF"),
    )
    fig.update_xaxes(title_text="Time (s)", gridcolor="#333333", tickfont=dict(color="#FFFFFF"))
    fig.update_layout(
        height=max(500, num_visemes * 70),
        title="Per-viseme timelines (top: target, bottom: model confidence)",
        plot_bgcolor="#000000",
        paper_bgcolor="#000000",
        font=dict(color="#FFFFFF"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
        margin=dict(t=60, r=10, b=40, l=80),
    )
    
    # Add separator lines between visemes (positioned between the rows)
    if num_frames > 0 and len(time_axis) > 0:
        x0 = time_axis[0]
        x1 = time_axis[-1]
        for k in range(1, num_visemes):
            y = 2 * k - 0.5  # positioned between the bottom row of viseme k-1 and top row of viseme k
            fig.add_shape(
                type="line",
                x0=x0, x1=x1, y0=y, y1=y,
                xref="x", yref="y",
                line=dict(color="lightgray", width=1),
            )
    return fig


def plot_tracks_with_per_viseme(
    mel: np.ndarray,
    words: List[Tuple[float, float, str]],
    phones: List[Tuple[float, float, str]],
    viseme_intervals: List[Tuple[float, float, int]],
    preds: Optional[np.ndarray],
    probs: Optional[np.ndarray],
    fps: float,
    viseme_index_to_name: Dict[int, str],
    show_words: bool,
    show_phones: bool,
    show_visemes: bool,
    show_preds: bool,
    probs_raw: Optional[np.ndarray] = None,
    show_raw_model: bool = False,
) -> go.Figure:
    include_per_viseme = (
        show_visemes and (
            (show_preds and probs is not None) or (show_raw_model and probs_raw is not None)
        )
    )
    num_visemes = (max(viseme_index_to_name.keys()) + 1) if viseme_index_to_name else int(probs.shape[1] if probs is not None else 0)

    base_rows = 4
    total_rows = base_rows + (1 if include_per_viseme else 0)
    row_heights = [0.6, 0.133, 0.133, 0.133]
    if include_per_viseme:
        # Allocate proportional height for per-viseme panel
        extra_height = max(0.3, min(0.9, num_visemes * 0.06))
        # Normalize to keep sum ~ 1.0
        scale = (1.0 - extra_height) / sum(row_heights)
        row_heights = [h * scale for h in row_heights] + [extra_height]

    time_axis = np.arange(mel.shape[0]) / fps
    fig = make_subplots(
        rows=total_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.02,
        subplot_titles=("Log-Mel", "Words", "Phonemes", "Visemes") + (("Per-viseme timelines",) if include_per_viseme else tuple()),
    )

    # Row 1: Mel
    fig.add_trace(
        go.Heatmap(x=time_axis, y=np.arange(mel.shape[1]), z=mel.T, colorscale="Viridis", showscale=False),
        row=1, col=1,
    )

    def add_intervals(intervals, row):
        for s, e, lbl in intervals:
            if s == e:
                continue
            fig.add_shape(type="rect", x0=s, x1=e, y0=0, y1=1, xref="x", yref=f"y{row}", line=dict(width=0), fillcolor="rgba(0, 150, 255, 0.2)")
            fig.add_annotation(x=(s + e) / 2.0, y=0.5, text=str(lbl), showarrow=False, xref="x", yref=f"y{row}", font=dict(size=14))

    # Row 2: Words
    if show_words:
        add_intervals(words, row=2)
        fig.update_yaxes(range=[0, 1], showticklabels=False, row=2, col=1)

    # Row 3: Phones
    if show_phones:
        add_intervals(phones, row=3)
        fig.update_yaxes(range=[0, 1], showticklabels=False, row=3, col=1)

    # Row 4: Visemes/Model
    if show_visemes:
        for s, e, v in viseme_intervals:
            if s == e:
                continue
            fig.add_shape(type="rect", x0=s, x1=e, y0=0.0, y1=0.48, xref="x", yref="y4", line=dict(width=0), fillcolor="rgba(0, 200, 100, 0.30)")
            fig.add_annotation(x=(s + e) / 2.0, y=0.24, text=viseme_index_to_name.get(int(v), str(v)), showarrow=False, xref="x", yref="y4", font=dict(size=14))

    # Final labels
    fig.update_yaxes(title_text="Mel bin", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=4 if not include_per_viseme else total_rows, col=1)

    # Row 5: Per-viseme timelines (single panel containing all visemes)
    if include_per_viseme:
        # Build grids
        total_frames = mel.shape[0]
        target_seq = viseme_intervals_to_sequence(viseme_intervals, total_frames=total_frames, fps=fps)
        include_raw = (show_raw_model and probs_raw is not None)
        rows_per = 2 + (1 if include_raw else 0)
        rows = num_visemes * rows_per
        target_grid = np.full((rows, total_frames), np.nan, dtype=float)
        probs_grid = np.full((rows, total_frames), np.nan, dtype=float)
        probs_raw_grid = np.full((rows, total_frames), np.nan, dtype=float) if include_raw else None
        for v in range(num_visemes):
            base_row = rows_per * v
            top_row = base_row
            mid_row = base_row + 1
            bot_row = base_row + 2 if include_raw else None
            if v <= target_seq.max():
                target_grid[top_row, :] = (target_seq == v).astype(float)
            if probs is not None and v < probs.shape[1]:
                probs_grid[mid_row, :] = probs[:total_frames, v]
            if include_raw and probs_raw is not None and v < probs_raw.shape[1]:
                probs_raw_grid[bot_row, :] = probs_raw[:total_frames, v]

        y_vals = np.arange(rows)
        y_centers = np.array([rows_per * v + (rows_per - 1) / 2.0 for v in range(num_visemes)])
        y_center_labels = [viseme_index_to_name.get(v, str(v)) for v in range(num_visemes)]

        fig.add_trace(
            go.Heatmap(
                x=time_axis,
                y=y_vals,
                z=target_grid,
                colorscale=[[0.0, "rgb(0,0,0)"], [1.0, "rgb(0,255,0)"]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                name="Target",
                showlegend=False,
            ),
            row=total_rows, col=1,
        )
        fig.add_trace(
            go.Heatmap(
                x=time_axis,
                y=y_vals,
                z=probs_grid,
                colorscale=[[0.0, "rgb(0,0,0)"], [1.0, "rgb(255,0,0)"]],
                zmin=0.0,
                zmax=1.0,
                showscale=False,
                name="Model (smoothed)",
                showlegend=False,
            ),
            row=total_rows, col=1,
        )
        if include_raw and probs_raw_grid is not None:
            fig.add_trace(
                go.Heatmap(
                    x=time_axis,
                    y=y_vals,
                    z=probs_raw_grid,
                    colorscale=[[0.0, "rgb(0,0,0)"], [1.0, "rgb(0,100,255)"]],
                    zmin=0.0,
                    zmax=1.0,
                    showscale=False,
                    name="Model (raw)",
                    showlegend=False,
                ),
                row=total_rows, col=1,
            )

        fig.update_yaxes(
            tickmode="array", tickvals=y_centers, ticktext=y_center_labels,
            autorange="reversed",
            row=total_rows, col=1,
        )

        # Separator lines
        x0 = time_axis[0]
        x1 = time_axis[-1]
        for k in range(1, num_visemes):
            y = rows_per * k - 0.5
            fig.add_shape(type="line", x0=x0, x1=x1, y0=y, y1=y, xref="x", yref=f"y{total_rows}", line=dict(color="lightgray", width=1))

    fig.update_layout(height=900 + (num_visemes * 60 if include_per_viseme else 0), hovermode="x unified")
    return fig

def main():
    st.set_page_config(page_title="OpenLipSync Debugger", layout="wide")
    st.title("OpenLipSync GUI Debugger")

    base_cfg = get_base_config()

    # Sidebar controls
    st.sidebar.header("Data")
    splits = list_splits()
    split = st.sidebar.selectbox("Split", splits, index=splits.index("dev-clean") if "dev-clean" in splits else 0)

    samples = list_samples(split)
    random_clicked = st.sidebar.button("Pick random sample")
    default_idx = random.randrange(len(samples)) if samples and random_clicked else 0
    sample_id = st.sidebar.selectbox("Sample", samples, index=default_idx if samples else 0)

    st.sidebar.header("Model")
    quick_runs = find_run_checkpoints()
    quick_label = st.sidebar.selectbox("Quick load run", ["(none)"] + list(quick_runs.keys()))
    ckpt_path: Optional[Path] = None
    if quick_label != "(none)":
        ckpt_path = quick_runs[quick_label]

    user_ckpt = st.sidebar.file_uploader("Or upload checkpoint (.pt)", type=["pt"])
    if user_ckpt is not None:
        ckpt_path = Path(user_ckpt.name)
        # Save uploaded buffer to a temp file in session state directory
        tmpfile = Path(st.experimental_user().get("upload_dir", str(PROJECT_ROOT))) / ckpt_path.name
        with open(tmpfile, "wb") as f:
            f.write(user_ckpt.getbuffer())
        ckpt_path = tmpfile

    st.sidebar.header("Tracks")
    show_words = st.sidebar.checkbox("Words", value=True)
    show_phones = st.sidebar.checkbox("Phonemes", value=True)
    show_visemes = st.sidebar.checkbox("Visemes", value=True)
    show_preds = st.sidebar.checkbox("Model output", value=True)
    show_raw_model = st.sidebar.checkbox("Raw model output", value=False)

    # EMA smoothing controls (only applied to model output probabilities)
    st.sidebar.header("Smoothing")
    enable_ema = st.sidebar.checkbox("Enable EMA", value=False)
    ema_amount = st.sidebar.slider("EMA amount (0–1)", min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    enable_median = st.sidebar.checkbox("Enable median filter (labels)", value=False)
    median_window = st.sidebar.slider("Median window (frames)", min_value=1, max_value=31, value=5, step=1)

    st.sidebar.header("Sharpen / Threshold")
    enable_sharpen = st.sidebar.checkbox("Enable sharpen/threshold", value=False)
    sharpen_gamma = st.sidebar.slider("Gamma (>=1.0)", min_value=1.0, max_value=6.0, value=2.0, step=0.1)
    sharpen_thresh = st.sidebar.slider("Threshold", min_value=0.0, max_value=0.2, value=0.01, step=0.005)
    norm_mode = st.sidebar.selectbox("Normalization mode", ["sum", "max", "none"], index=0, help="'sum' keeps rows summing to 1 (exclusive); 'max' scales peak to 1 allowing coexistence; 'none' keeps magnitudes after thresholding.")

    if not samples:
        st.warning("No samples found in prepared data.")
        return

    split_dir = PREPARED_DIR / split
    wav_path = split_dir / f"{sample_id}.wav"
    json_path = split_dir / f"{sample_id}.json"

    # Load alignment
    words, phones = load_alignment(json_path)
    viseme_intervals = phonemes_to_viseme_intervals(phones, base_cfg.phoneme_to_viseme_mapping)

    # Mel
    mel = compute_mel(wav_path, base_cfg)

    # Model predictions
    preds = None
    probs = None
    probs_raw = None
    if ckpt_path is not None and ckpt_path.exists():
        cfg_from_ckpt = load_run_config_for_checkpoint(ckpt_path)
        run_cfg = cfg_from_ckpt or base_cfg
        if cfg_from_ckpt is not None:
            print(f"[DEBUG] Using checkpoint's original config, multi_label = {run_cfg.training.multi_label}")
        else:
            print(f"[DEBUG] Using fallback config (tcn_test.toml), multi_label = {run_cfg.training.multi_label}")
        preds, probs = run_model_on_mel(mel, run_cfg, ckpt_path)
        probs_raw = probs.copy() if probs is not None else None

        # Sanitize probabilities to handle NaN/Inf/noise before smoothing
        silence_idx = 0  # per viseme map, 0 == silence
        if probs is not None:
            probs = sanitize_probs(
                probs,
                silence_index=silence_idx,
                multi_label=bool(getattr(run_cfg.training, "multi_label", False)),
            )
            preds = np.argmax(probs, axis=-1)

            # Optional sharpen -> threshold -> renormalize (pre-smoothing)
            if enable_sharpen:
                probs = sharpen_threshold_probs(probs, gamma=sharpen_gamma, threshold=sharpen_thresh, silence_index=silence_idx, norm_mode=norm_mode)
                preds = np.argmax(probs, axis=-1)

        # Optionally smooth only the model outputs (probs) with EMA
        if enable_ema and probs is not None and ema_amount > 0.0:
            probs = ema_smooth_probs(probs, alpha=ema_amount)
            # Update preds from smoothed probabilities for consistency where used
            preds = np.argmax(probs, axis=-1)

        # Optional median filter on discrete labels
        if enable_median and preds is not None and median_window > 1:
            preds = median_filter_preds(preds, window_size=median_window)

    fig = plot_tracks_with_per_viseme(
        mel=mel,
        words=words,
        phones=phones,
        viseme_intervals=viseme_intervals,
        preds=preds,
        probs=probs,
        probs_raw=probs_raw,
        fps=base_cfg.audio.fps,
        viseme_index_to_name=load_viseme_index_to_name(base_cfg),
        show_words=show_words,
        show_phones=show_phones,
        show_visemes=show_visemes,
        show_preds=show_preds and preds is not None,
        show_raw_model=show_raw_model and probs_raw is not None,
    )
    st.plotly_chart(fig, use_container_width=True)


    # Simple audio player
    try:
        import torchaudio
        if wav_path.exists():
            wav_bytes = wav_path.read_bytes()
            st.audio(wav_bytes, format="audio/wav")
    except Exception:
        pass


if __name__ == "__main__":
    main()


