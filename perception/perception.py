import cv2
import numpy as np
import torch
from enum import Enum
from typing import Deque, Dict, Optional, Tuple


NUM_CLASSES = 9
EGO_CLASS = 1
_INF_H, _INF_W = 256, 512
MIN_LANE_PIXELS = 6
MAX_LANE_WIDTH_PX = 500
MIN_LANE_WIDTH_PX = 60
BEV_SRC_POINTS = np.float32([[0.18, 0.92], [0.82, 0.92], [0.42, 0.58], [0.58, 0.58]])
BEV_DST_POINTS = np.float32([[0.20, 1.00], [0.80, 1.00], [0.20, 0.00], [0.80, 0.00]])
_BEV_TRANSFORMS: Dict[Tuple[int, int], np.ndarray] = {}


class LaneState(Enum):
    NORMAL = "NORMAL"
    CROSSWALK = "CROSSWALK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MERGING = "MERGING"
    CURVE = "CURVE"


_clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))
_left_committed, _right_committed = None, None
_left_contrary, _right_contrary = 0, 0


def night_enhance(rgb: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    l = _clahe.apply(l)
    rgb_out = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    inv_gamma = 1.0 / 0.72
    lut = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    rgb_out = cv2.LUT(rgb_out, lut)
    rgb_out = cv2.convertScaleAbs(rgb_out, alpha=1.35, beta=8)
    return rgb_out


def preprocess_gpu(rgb_uint8: np.ndarray, device: str, mean: torch.Tensor, std: torch.Tensor, is_half: bool) -> torch.Tensor:
    small = cv2.resize(rgb_uint8, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(small).permute(2, 0, 1).to(device)
    t = t.half() if is_half else t.float()
    t = t.div_(255.0).unsqueeze(0)
    t = (t - mean) / std
    return t


def filter_horizontal_noise(mask: np.ndarray) -> np.ndarray:
    mask_bool = mask > 0
    h, w = mask.shape
    if h == 0 or w == 0:
        return mask.copy()

    row_counts = mask_bool.sum(axis=1)
    valid_rows = row_counts > 0
    if not np.any(valid_rows):
        return mask.copy()

    row_starts = np.argmax(mask_bool[valid_rows], axis=1)
    row_ends = w - 1 - np.argmax(mask_bool[valid_rows][:, ::-1], axis=1)
    row_spans = row_ends - row_starts
    invalid_rows = np.zeros(h, dtype=bool)
    invalid_rows[valid_rows] = (row_counts[valid_rows] < MIN_LANE_PIXELS) | (row_spans > w * 0.58)

    clean = mask.copy()
    clean[invalid_rows] = 0
    return clean


def detect_crosswalk(mask: np.ndarray) -> bool:
    h, w = mask.shape
    roi = mask[int(h * 0.55):int(h * 0.85), :]
    rows = np.sum(roi > 0, axis=1)
    return bool(np.sum(rows > w * 0.35) > 12)


def bev(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    key = (w, h)
    if key not in _BEV_TRANSFORMS:
        src = np.float32([
            [w * BEV_SRC_POINTS[0, 0], h * BEV_SRC_POINTS[0, 1]],
            [w * BEV_SRC_POINTS[1, 0], h * BEV_SRC_POINTS[1, 1]],
            [w * BEV_SRC_POINTS[2, 0], h * BEV_SRC_POINTS[2, 1]],
            [w * BEV_SRC_POINTS[3, 0], h * BEV_SRC_POINTS[3, 1]],
        ])
        dst = np.float32([
            [w * BEV_DST_POINTS[0, 0], h * BEV_DST_POINTS[0, 1]],
            [w * BEV_DST_POINTS[1, 0], h * BEV_DST_POINTS[1, 1]],
            [w * BEV_DST_POINTS[2, 0], h * BEV_DST_POINTS[2, 1]],
            [w * BEV_DST_POINTS[3, 0], h * BEV_DST_POINTS[3, 1]],
        ])
        _BEV_TRANSFORMS[key] = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(mask, _BEV_TRANSFORMS[key], (w, h))


def get_line_type(band: np.ndarray) -> str:
    occ = np.mean(np.sum(band > 0, axis=1) > 2)
    return "solid" if occ > 0.68 else "dashed" if occ > 0.16 else "unknown"


def classify_boundary(side: np.ndarray, hist: Deque[Tuple[str, float]], side_id: str) -> str:
    global _left_committed, _right_committed, _left_contrary, _right_contrary
    h, w = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col = int(np.argmax(col_profile))
    peak_val = int(col_profile[peak_col])

    if peak_val < 4:
        raw, conf = "unknown", 0.25
    else:
        search = col_profile.astype(float).copy()
        suppress = max(4, int(w * 0.055))
        search[max(0, peak_col - suppress):min(w, peak_col + suppress)] = 0
        second_col = int(np.argmax(search))
        second_val = int(search[second_col])
        distance = abs(second_col - peak_col)
        is_double = (second_val >= peak_val * 0.55 and 8 <= distance <= int(w * 0.11))

        bh = max(6, int(w * 0.11))
        main_band = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
        main_type = get_line_type(main_band)

        if is_double:
            sb_band = side[:, max(0, second_col - bh):min(w, second_col + bh)]
            second_type = get_line_type(sb_band)
            raw = f"double {main_type}" if main_type == second_type != "unknown" else main_type
        else:
            raw = main_type

        conf = min(1.0, peak_val / 25.0)
        conf *= 0.78 if "solid" in raw else 0.92

    hist.append((raw, conf))
    vote = max(set([x[0] for x in hist]), key=[x[0] for x in hist].count) if len(hist) >= 5 else raw

    committed = _left_committed if side_id == 'left' else _right_committed
    contrary = _left_contrary if side_id == 'left' else _right_contrary

    if committed is None:
        committed, contrary = vote, 0
    elif vote == committed:
        contrary = 0
    else:
        contrary += 1
        if contrary >= 14:
            committed, contrary = vote, 0

    if side_id == 'left':
        _left_committed, _left_contrary = committed, contrary
    else:
        _right_committed, _right_contrary = committed, contrary
    return committed


def compute_lane_width(mask: np.ndarray) -> float:
    h, w, mid = mask.shape[0], mask.shape[1], mask.shape[1] // 2
    widths = []
    for y in range(int(h * 0.6), h, 4):
        row = np.where(mask[y] > 0)[0]
        if len(row) == 0:
            continue
        L, R = row[row < mid], row[row >= mid]
        if len(L) > 3 and len(R) > 3:
            widths.append(int(np.min(R)) - int(np.max(L)))
    return float(np.median(widths)) if widths else 0.0


_kf_gap_x, _kf_gap_p = 0.0, 1.0


def measure_dash_trend(side: np.ndarray, trend_buf: Deque[Tuple[int, float]], trend_frame: int) -> Tuple[Optional[float], Optional[str], int]:
    global _kf_gap_x, _kf_gap_p
    h, w = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col = int(np.argmax(col_profile))
    if col_profile[peak_col] < 4:
        return None, None, trend_frame

    trend_frame += 1
    bh = max(6, int(w * 0.13))
    band = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
    signal = (np.sum(band > 0, axis=1) > 2).astype(np.int8)
    diff = np.diff(signal)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0:
        return None, None, trend_frame
    if ends[0] < starts[0]:
        ends = ends[1:]
    n = min(len(starts), len(ends))
    if n < 2:
        return None, None, trend_frame

    gaps = [int(starts[i + 1]) - int(ends[i]) for i in range(n - 1) if 5 < (starts[i + 1] - ends[i]) < h * 0.6]
    if not gaps:
        return None, None, trend_frame

    _kf_gap_p += 0.08
    k = _kf_gap_p / (_kf_gap_p + 1.2)
    _kf_gap_x = _kf_gap_x + k * (float(np.mean(gaps)) - _kf_gap_x)
    _kf_gap_p = (1 - k) * _kf_gap_p

    trend_buf.append((trend_frame, _kf_gap_x))
    if len(trend_buf) < 12:
        return _kf_gap_x, "calculating...", trend_frame

    data = np.array(list(trend_buf)[-15:])
    slope = np.polyfit(data[:, 0], data[:, 1], 1)[0]
    trend = "diverging" if slope > 0.20 else "converging" if slope < -0.20 else "constant"
    return _kf_gap_x, trend, trend_frame
