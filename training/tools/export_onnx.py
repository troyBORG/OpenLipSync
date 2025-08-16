#!/usr/bin/env python3
"""
ONNX Export Tool for OpenLipSync Models

Exports trained models from training/runs to ONNX format with configuration.
Usage:
  python export_onnx.py --list                    # List available runs
  python export_onnx.py --run RUN_NAME            # Export specific run
  python export_onnx.py --run RUN_NAME --checkpoint best  # Use best model (default)
  python export_onnx.py --run RUN_NAME --checkpoint latest # Use latest checkpoint
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.onnx
import numpy as np

# Ensure training package is importable
def _add_training_to_syspath() -> None:
    """Ensure `training` package root is importable."""
    tools_dir = Path(__file__).resolve().parent
    training_root = tools_dir.parent  # .../training
    project_root = training_root.parent  # .../OpenLipSync
    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))
    return project_root

def find_available_runs() -> Dict[str, Dict[str, Path]]:
    """Find all available training runs with their checkpoints.
    
    Returns:
        Dict mapping run_name -> {'best': path, 'latest': path, 'config': path}
    """
    project_root = _add_training_to_syspath()
    runs_dir = project_root / "training" / "runs"
    
    runs = {}
    if not runs_dir.exists():
        return runs
    
    for run_dir in runs_dir.iterdir():
        if not run_dir.is_dir():
            continue
            
        ckpt_dir = run_dir / "checkpoints"
        if not ckpt_dir.exists():
            continue
            
        run_info = {}
        
        # Look for best model
        best_model = ckpt_dir / "best_model.pt"
        if best_model.exists():
            run_info['best'] = best_model
            
        # Look for latest checkpoint
        checkpoints = sorted(ckpt_dir.glob("checkpoint_epoch_*.pt"), 
                           key=lambda p: p.stat().st_mtime)
        if checkpoints:
            run_info['latest'] = checkpoints[-1]
            
        # Look for saved config
        config_json = run_dir / "config.json"
        if config_json.exists():
            run_info['config'] = config_json
            
        if run_info:  # Only add if we found at least one checkpoint
            runs[run_dir.name] = run_info
            
    return runs

def load_model_and_config(checkpoint_path: Path, 
                         config_path: Optional[Path] = None) -> Tuple[torch.nn.Module, Dict]:
    """Load model and configuration from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint
        config_path: Optional path to config JSON (will try to find automatically)
        
    Returns:
        Tuple of (model, config_dict)
    """
    from modules.config import load_config, TrainingConfiguration
    from modules.tcn_model import create_model
    
    # Try to load config from multiple sources
    config = None
    config_dict = None
    
    if config_path and config_path.exists():
        # Load from provided JSON config
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        # Try to find config in run directory
        run_dir = checkpoint_path.parent.parent
        config_json = run_dir / "config.json"
        if config_json.exists():
            with open(config_json, 'r') as f:
                config_dict = json.load(f)
    
    if config_dict:
        # Reconstruct TrainingConfiguration from saved dict
        from modules.config import (
            TrainingConfiguration, ModelConfig, AudioConfig, TrainingConfig,
            DataConfig, EvaluationConfig, HardwareConfig, LoggingConfig,
            TensorBoardConfig, ExperimentConfig
        )
        
        # Filter config sections to only include fields that exist in current dataclasses
        def filter_config_section(section_dict, config_class):
            """Filter dict to only include fields that exist in the dataclass."""
            import dataclasses
            if not dataclasses.is_dataclass(config_class):
                return section_dict
            
            valid_fields = {field.name for field in dataclasses.fields(config_class)}
            return {k: v for k, v in section_dict.items() if k in valid_fields}
        
        config = TrainingConfiguration(
            model=ModelConfig(**filter_config_section(config_dict["model"], ModelConfig)),
            audio=AudioConfig(**filter_config_section(config_dict["audio"], AudioConfig)),
            training=TrainingConfig(**filter_config_section(config_dict["training"], TrainingConfig)),
            data=DataConfig(**filter_config_section(config_dict["data"], DataConfig)),
            evaluation=EvaluationConfig(**filter_config_section(config_dict["evaluation"], EvaluationConfig)),
            hardware=HardwareConfig(**filter_config_section(config_dict["hardware"], HardwareConfig)),
            logging=LoggingConfig(**filter_config_section(config_dict["logging"], LoggingConfig)),
            tensorboard=TensorBoardConfig(**filter_config_section(config_dict["tensorboard"], TensorBoardConfig)),
            experiment=ExperimentConfig(**filter_config_section(config_dict["experiment"], ExperimentConfig)),
            config_path=""  # Not needed for export
        )
    else:
        # Fallback to default config
        project_root = Path(__file__).resolve().parents[2]
        default_config = project_root / "training" / "recipes" / "tcn_config.toml"
        config = load_config(default_config)
        # Convert to dict for consistency
        config_dict = serialize_config(config)
    
    # Create model
    model = create_model(config)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    except Exception:
        # Fallback for older checkpoints
        try:
            # Handle module path aliasing for compatibility
            import modules as _modules
            sys.modules.setdefault("modules", _modules)
        except:
            pass
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
    
    model.load_state_dict(state_dict)
    model.eval()
    
    return model, config_dict

def serialize_config(config) -> Dict:
    """Convert TrainingConfiguration to serializable dict."""
    def to_dict(obj):
        if hasattr(obj, '__dict__'):
            return {k: to_dict(v) for k, v in obj.__dict__.items() 
                   if not k.startswith('_')}
        elif isinstance(obj, dict):
            return {k: to_dict(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [to_dict(item) for item in obj]
        else:
            return obj
    return to_dict(config)

def export_onnx(model: torch.nn.Module, 
                config_dict: Dict,
                output_dir: Path,
                model_name: str = "model") -> None:
    """Export model to ONNX format.
    
    Args:
        model: PyTorch model to export
        config_dict: Configuration dictionary
        output_dir: Output directory for exported files
        model_name: Base name for the model file
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prepare dummy input based on config
    batch_size = 1
    seq_length = 100  # 1 second at 100fps
    n_mels = config_dict['audio']['n_mels']
    
    dummy_input = torch.randn(batch_size, seq_length, n_mels)
    
    # Export to ONNX
    onnx_path = output_dir / f"{model_name}.onnx"
    
    used_exporter = "dynamo"
    try:
        # Prefer the new torch.export-based ONNX exporter
        from torch.export import Dim
        dynamic_shapes = {
            "audio_features": {0: Dim("batch", min=1, max=16), 1: Dim("sequence_length", min=1, max=2048)}
        }
        torch.onnx.export(
            model,                          # model being run
            dummy_input,                    # model input
            str(onnx_path),                 # where to save the model
            export_params=True,             # store the trained parameter weights
            opset_version=11,               # ONNX version to export to
            do_constant_folding=True,       # whether to execute constant folding
            input_names=['audio_features'], # input names
            output_names=['viseme_logits'], # output names
            dynamic_shapes=dynamic_shapes,  # dynamic shapes with new exporter
            dynamo=True                     # use new torch.export-based exporter
        )
    except Exception:
        # Fallback to legacy exporter for compatibility
        used_exporter = "torchscript"
        torch.onnx.export(
            model,                          # model being run
            dummy_input,                    # model input
            str(onnx_path),                 # where to save the model
            export_params=True,             # store the trained parameter weights
            opset_version=11,               # ONNX version to export to
            do_constant_folding=True,       # whether to execute constant folding
            input_names=['audio_features'], # input names
            output_names=['viseme_logits'], # output names
            dynamic_axes={
                'audio_features': {1: 'sequence_length'},  # variable length sequences
                'viseme_logits': {1: 'sequence_length'}    # variable length outputs
            }
        )
    
    # Save configuration
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Create metadata file
    metadata = {
        "model_format": "onnx",
        "opset_version": 11,
        "input_shape": [batch_size, "variable", n_mels],
        "output_shape": [batch_size, "variable", config_dict['model']['num_visemes']],
        "input_names": ["audio_features"],
        "output_names": ["viseme_logits"],
        "description": "OpenLipSync TCN model for viseme prediction from mel-spectrogram features",
        "exporter": used_exporter
    }
    
    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Model exported to: {onnx_path}")
    print(f"Config saved to: {config_path}")
    print(f"Metadata saved to: {metadata_path}")

def list_runs(runs: Dict[str, Dict[str, Path]]) -> None:
    """Print available runs and their checkpoints."""
    if not runs:
        print("No training runs found in training/runs/")
        return
    
    print(f"Available runs ({len(runs)} found):")
    print("-" * 60)
    
    for run_name, info in sorted(runs.items()):
        print(f"{run_name}")
        if 'best' in info:
            print(f"   best_model.pt")
        if 'latest' in info:
            latest_name = info['latest'].name
            print(f"   {latest_name}")
        if 'config' in info:
            print(f"   config.json")
        print()

def interactive_mode():
    """Run the export tool in interactive mode."""
    print("OpenLipSync ONNX Export Tool")
    print("=" * 50)
    
    # Find available runs
    runs = find_available_runs()
    
    if not runs:
        print("No training runs found in training/runs/")
        print("   Make sure you have completed at least one training run.")
        sys.exit(1)
    
    # Show available runs
    print(f"\nAvailable runs ({len(runs)} found):")
    run_names = list(runs.keys())
    for i, run_name in enumerate(run_names, 1):
        info = runs[run_name]
        checkpoints = []
        if 'best' in info:
            checkpoints.append("best")
        if 'latest' in info:
            checkpoints.append("latest")
        print(f"  {i}. {run_name}")
        print(f"     Checkpoints: {', '.join(checkpoints)}")
        print()
    
    # Select run
    while True:
        try:
            choice = input(f"Select a run (1-{len(run_names)}) or 'q' to quit: ").strip()
            if choice.lower() == 'q':
                print("Cancelled.")
                sys.exit(0)
            
            run_idx = int(choice) - 1
            if 0 <= run_idx < len(run_names):
                selected_run = run_names[run_idx]
                break
            else:
                print(f"Please enter a number between 1 and {len(run_names)}")
        except ValueError:
            print("Please enter a valid number or 'q' to quit")
    
    run_info = runs[selected_run]
    print(f"\nSelected: {selected_run}")
    
    # Select checkpoint
    available_checkpoints = []
    if 'best' in run_info:
        available_checkpoints.append('best')
    if 'latest' in run_info:
        available_checkpoints.append('latest')
    
    if len(available_checkpoints) == 1:
        checkpoint_key = available_checkpoints[0]
        print(f"Using checkpoint: {checkpoint_key}")
    else:
        print(f"\nAvailable checkpoints:")
        for i, ckpt in enumerate(available_checkpoints, 1):
            print(f"  {i}. {ckpt}")
        
        while True:
            try:
                choice = input(f"Select checkpoint (1-{len(available_checkpoints)}) [default: 1]: ").strip()
                if not choice:
                    choice = "1"
                
                ckpt_idx = int(choice) - 1
                if 0 <= ckpt_idx < len(available_checkpoints):
                    checkpoint_key = available_checkpoints[ckpt_idx]
                    break
                else:
                    print(f"Please enter a number between 1 and {len(available_checkpoints)}")
            except ValueError:
                print("Please enter a valid number")
    
    checkpoint_path = run_info[checkpoint_key]
    config_path = run_info.get('config')
    
    # Determine output directory
    project_root = _add_training_to_syspath()
    default_output = project_root / "export" / selected_run
    
    print(f"\nDefault output directory: {default_output}")
    custom_output = input("Custom output directory (press Enter for default): ").strip()
    
    if custom_output:
        output_dir = Path(custom_output)
    else:
        output_dir = default_output
    
    # Model name
    model_name = "model"
    
    # Confirmation
    print(f"\nExport Summary:")
    print(f"   Run: {selected_run}")
    print(f"   Checkpoint: {checkpoint_key}")
    print(f"   Output: {output_dir}")
    
    confirm = input(f"\nProceed with export? [Y/n]: ").strip().lower()
    if confirm and confirm not in ['y', 'yes']:
        print("Cancelled.")
        sys.exit(0)
    
    return selected_run, checkpoint_path, config_path, output_dir, model_name

def main():
    parser = argparse.ArgumentParser(
        description="Export OpenLipSync models to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                     # Interactive mode
  %(prog)s --list                              # List available runs
  %(prog)s --run my_tcn_run                    # Export using best model
  %(prog)s --run my_tcn_run --checkpoint latest # Export using latest checkpoint
  %(prog)s --run my_tcn_run --output /path/to/export  # Custom export directory
        """
    )
    
    parser.add_argument('--list', action='store_true',
                       help='List available training runs')
    parser.add_argument('--run', type=str,
                       help='Name of the training run to export')
    parser.add_argument('--checkpoint', choices=['best', 'latest'], default='best',
                       help='Which checkpoint to use (default: best)')
    parser.add_argument('--output', type=str,
                       help='Custom output directory (default: export/RUN_NAME)')
    parser.add_argument('--model-name', type=str, default='model',
                       help='Base name for exported model file (default: model)')
    parser.add_argument('--interactive', action='store_true',
                       help='Force interactive mode (default when no args provided)')
    
    args = parser.parse_args()
    
    # Find available runs
    runs = find_available_runs()
    
    if args.list:
        list_runs(runs)
        return
    
    # Use interactive mode if no run specified or explicitly requested
    if not args.run or args.interactive:
        try:
            selected_run, checkpoint_path, config_path, output_dir, model_name = interactive_mode()
        except KeyboardInterrupt:
            print("\n\nCancelled by user.")
            sys.exit(0)
    else:
        # CLI mode
        if args.run not in runs:
            print(f"Error: Run '{args.run}' not found")
            print("Available runs:")
            list_runs(runs)
            sys.exit(1)
        
        run_info = runs[args.run]
        
        # Select checkpoint
        checkpoint_key = args.checkpoint
        if checkpoint_key not in run_info:
            available = list(run_info.keys())
            print(f"Error: {checkpoint_key} checkpoint not found for run '{args.run}'")
            print(f"Available checkpoints: {available}")
            sys.exit(1)
        
        checkpoint_path = run_info[checkpoint_key]
        config_path = run_info.get('config')
        
        # Determine output directory
        if args.output:
            output_dir = Path(args.output)
        else:
            project_root = _add_training_to_syspath()
            output_dir = project_root / "export" / args.run
        
        selected_run = args.run
        model_name = args.model_name
    
    print(f"\nExporting run: {selected_run}")
    print(f"   Checkpoint: {checkpoint_path}")
    print(f"   Output: {output_dir}")
    print("-" * 50)
    
    try:
        # Load model and config
        print("Loading model and configuration...")
        model, config_dict = load_model_and_config(checkpoint_path, config_path)
        
        # Export to ONNX
        print("Exporting to ONNX...")
        export_onnx(model, config_dict, output_dir, model_name)
        
        print("\nExport completed successfully.")
        
    except Exception as e:
        print(f"\nExport failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
