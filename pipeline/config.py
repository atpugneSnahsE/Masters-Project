import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Sequence


@dataclass
class PipelineConfig:
    map_name: str = "Town10HD"
    weather: str = "night"
    model_path: str = "models/lane_model_best.pth"
    output_dir: str = "reports/segmentation_plots"
    sim_length: int = 300
    save_every: int = 10
    seed: int = 42
    headless: bool = True
    show_gui: bool = False


def build_config(argv: Optional[Sequence[str]] = None) -> PipelineConfig:
    parser = argparse.ArgumentParser(description="Run the CARLA lane and localization pipeline")
    parser.add_argument("--map", default="Town10HD", help="CARLA map to load")
    parser.add_argument("--weather", default="night", choices=["night", "day", "rain", "fog"], help="Weather preset")
    parser.add_argument("--model-path", default="models/lane_model_best.pth", help="Path to the perception checkpoint")
    parser.add_argument("--sim-length", type=int, default=300, help="Number of simulation frames to process")
    parser.add_argument("--output-dir", default="reports/segmentation_plots", help="Base directory for outputs")
    parser.add_argument("--save-every", type=int, default=10, help="Persist log CSV every N frames")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--headless", action="store_true", help="Force headless rendering")
    parser.add_argument("--show-gui", action="store_true", help="Attempt to display GUI windows")
    args = parser.parse_args(list(argv) if argv is not None else None)

    return PipelineConfig(
        map_name=args.map,
        weather=args.weather,
        model_path=args.model_path,
        output_dir=args.output_dir,
        sim_length=args.sim_length,
        save_every=args.save_every,
        seed=args.seed,
        headless=args.headless or not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"),
        show_gui=args.show_gui,
    )


def save_config(config: PipelineConfig, output_dir: str) -> None:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(asdict(config), handle, indent=2)
