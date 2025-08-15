import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
    if run_config_json.exists():
        try:
            with open(run_config_json, "r") as f:
                cfg = json.load(f)
            # Rehydrate via TOML loader substitute: dump tmp TOML-like dict is not needed;
            # Instead, build TrainingConfiguration by feeding dicts into dataclasses is done by load_config only.
            # Use a small adapter: write a temporary file to pass through existing loader expectations if necessary.
            # Simpler: construct TrainingConfiguration using same shape as saved dict.
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

            return TrainingConfiguration(
                model=ModelConfig(**cfg["model"]),
                audio=AudioConfig(**cfg["audio"]),
                training=TrainingConfig(**cfg["training"]),
                data=DataConfig(**cfg["data"]),
                evaluation=EvaluationConfig(**cfg["evaluation"]),
                hardware=HardwareConfig(**cfg["hardware"]),
                logging=LoggingConfig(**cfg["logging"]),
                tensorboard=TensorBoardConfig(**cfg["tensorboard"]),
                experiment=ExperimentConfig(**cfg["experiment"]),
                config_path=str(DEFAULT_CONFIG),
            )
        except Exception:
            pass
    return None


@st.cache_resource(show_spinner=False)
def get_base_config() -> TrainingConfiguration:
    return load_config(DEFAULT_CONFIG)


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
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()  # (T, C)
        preds = np.argmax(probs, axis=-1)
    # Ensure model outputs align with mel length
    t_mel = mel.shape[0]
    t_model = preds.shape[0]
    t = min(t_mel, t_model)
    if t != t_model:
        preds = preds[:t]
        probs = probs[:t, :]
    return preds, probs  # (T,), (T, C)


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
) -> go.Figure:
    include_per_viseme = (show_visemes and show_preds and probs is not None)
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
        rows = num_visemes * 2
        target_grid = np.full((rows, total_frames), np.nan, dtype=float)
        probs_grid = np.full((rows, total_frames), np.nan, dtype=float)
        for v in range(num_visemes):
            top_row = 2 * v
            bot_row = 2 * v + 1
            if v <= target_seq.max():
                target_grid[top_row, :] = (target_seq == v).astype(float)
            if probs is not None and v < probs.shape[1]:
                probs_grid[bot_row, :] = probs[:total_frames, v]

        y_vals = np.arange(rows)
        y_centers = np.array([2 * v + 0.5 for v in range(num_visemes)])
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
                name="Target (top)",
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
                name="Model (bottom)",
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
            y = 2 * k - 0.5
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
    if ckpt_path is not None and ckpt_path.exists():
        run_cfg = load_run_config_for_checkpoint(ckpt_path) or base_cfg
        preds, probs = run_model_on_mel(mel, run_cfg, ckpt_path)

    fig = plot_tracks_with_per_viseme(
        mel=mel,
        words=words,
        phones=phones,
        viseme_intervals=viseme_intervals,
        preds=preds,
        probs=probs,
        fps=base_cfg.audio.fps,
        viseme_index_to_name=load_viseme_index_to_name(base_cfg),
        show_words=show_words,
        show_phones=show_phones,
        show_visemes=show_visemes,
        show_preds=show_preds and preds is not None,
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


