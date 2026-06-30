import sys
from pathlib import Path

import cv2
import numpy as np
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from perception import LaneState


class HUDRenderer:
    def __init__(self) -> None:
        self.font = cv2.FONT_HERSHEY_DUPLEX

    def draw(self, overlay: np.ndarray, lane_label: str, state: LaneState, speed: float, gap: Optional[float], trend: Optional[str], width: float, num_v: int, num_w: int) -> None:
        h, w = overlay.shape[:2]
        bar = np.zeros((130, w, 3), dtype=np.uint8)
        overlay[0:130] = cv2.addWeighted(overlay[0:130], 0.35, bar, 0.65, 0)
        cv2.putText(overlay, f"Lane: {lane_label}", (20, 38), self.font, 0.85, (0, 255, 255), 2)
        cv2.putText(overlay, f"State: {state.value} | Speed: {speed:.1f} km/h", (20, 68), self.font, 0.72, (0, 255, 180), 2)
        info = f"Gap: {gap:.1f}px | Trend: {trend} | Width: {int(width)}px" if gap is not None else f"Width: {int(width)}px"
        cv2.putText(overlay, info, (20, 98), self.font, 0.68, (255, 255, 100), 2)
        traffic_txt = f"NPC Cars: {num_v}  Walkers: {num_w}"
        txt_size = cv2.getTextSize(traffic_txt, self.font, 0.58, 1)[0]
        cv2.putText(overlay, traffic_txt, (w - txt_size[0] - 14, 28), self.font, 0.58, (200, 200, 255), 1)
