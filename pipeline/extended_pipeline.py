import carla
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import cv2
import torchvision.transforms as T
import segmentation_models_pytorch as smp
from lime import lime_image
from skimage.segmentation import mark_boundaries, quickshift
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from collections import deque
from enum import Enum
import time
import os
import threading
import queue
import warnings
import random
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from config import PipelineConfig, build_config, save_config

try:
    from scipy.stats import chi2
except Exception:  # pragma: no cover - optional dependency fallback
    chi2 = None

warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Create required output directories
os.makedirs("xai_output", exist_ok=True)
os.makedirs("reports/segmentation_plots", exist_ok=True)

# ==============================================================================
# 1. PERCEPTION CONFIGURATION & MODEL INITIALIZATION
# ==============================================================================
NUM_CLASSES = 9
EGO_CLASS = 1
_INF_H, _INF_W = 256, 512
MIN_LANE_PIXELS = 6
MAX_LANE_WIDTH_PX = 500
MIN_LANE_WIDTH_PX = 60
BEV_SRC_POINTS = np.float32([[0.18, 0.92], [0.82, 0.92], [0.42, 0.58], [0.58, 0.58]])
BEV_DST_POINTS = np.float32([[0.20, 1.00], [0.80, 1.00], [0.20, 0.00], [0.80, 0.00]])
_BEV_TRANSFORMS: Dict[Tuple[int, int], Any] = {}

# Disable strict weights-only filtering for trusted local file to load internal NumPy dtypes
CONFIG: PipelineConfig = build_config([])
state_dict = torch.load(CONFIG.model_path, map_location="cpu", weights_only=False)
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

has_attention  = any("attention" in k for k in state_dict.keys())
attention_type = "scse" if has_attention else None
logger.info("Checkpoint attention type: %s", "scse" if has_attention else "None")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights=None,
    in_channels=3,
    classes=NUM_CLASSES,
    decoder_attention_type=attention_type,
    activation=None
)
model.load_state_dict(state_dict, strict=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model  = model.to(device)
model.eval()

is_half = (device == "cuda")
if is_half:
    model = model.half()
    torch.backends.cudnn.benchmark = True

logger.info("Perception model loaded successfully on %s", device.upper())

_MEAN = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
_STD  = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
if is_half:
    _MEAN, _STD = _MEAN.half(), _STD.half()

# ==============================================================================
# 2. LOCALIZATION KINEMATIC EKFs DEFINITION
# ==============================================================================
def wrap_angle(rad): 
    return (rad + np.pi) % (2 * np.pi) - np.pi


def geodetic_to_enu(lat_deg, lon_deg, alt_m, lat0_deg, lon0_deg, alt0_m):
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = 2 * f - f * f

    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    lat0 = np.radians(lat0_deg)
    lon0 = np.radians(lon0_deg)

    def ecef(lat_rad, lon_rad, alt):
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        n = a / np.sqrt(1.0 - e2 * sin_lat * sin_lat)
        x = (n + alt) * cos_lat * cos_lon
        y = (n + alt) * cos_lat * sin_lon
        z = (n * (1.0 - e2) + alt) * sin_lat
        return np.array([x, y, z], dtype=np.float64)

    p = ecef(lat, lon, alt_m)
    p0 = ecef(lat0, lon0, alt0_m)
    dp = p - p0
    sin_lat0 = np.sin(lat0)
    cos_lat0 = np.cos(lat0)
    sin_lon0 = np.sin(lon0)
    cos_lon0 = np.cos(lon0)

    east = -sin_lon0 * dp[0] + cos_lon0 * dp[1]
    north = -sin_lat0 * cos_lon0 * dp[0] - sin_lat0 * sin_lon0 * dp[1] + cos_lat0 * dp[2]
    up = cos_lat0 * cos_lon0 * dp[0] + cos_lat0 * sin_lon0 * dp[1] + sin_lat0 * dp[2]
    return east, north, up


class KinematicEKFs:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.x_rtk = np.zeros((6, 1))
        self.P_rtk = np.eye(6) * 2.0
        # Process noise rebuilt from continuous-time motion model in predict()
        self._sigma_a2 = 200.0
        self._sigma_z2 = 0.2
        self._sigma_yaw2 = 10.0
        self.Q_rtk = np.eye(6) * 0.01  # placeholder, rebuilt each predict()
        self.R_rtk = np.diag([
            0.03**2 * 45.0,
            0.03**2 * 45.0,
            0.05**2 * 20.0
        ])

        self.x_cart = np.zeros((4, 1))
        self.P_cart = np.eye(4) * 0.1
        self.Q_cart = np.diag([0.05, 0.05, 0.1, 0.1])
        self.R_cart = np.diag([0.24, 0.24, 0.06])

        self.x_odo = np.zeros((6, 1))
        self.P_odo = np.eye(6) * 0.1
        self.Q_odo = np.diag([0.02, 0.02, 0.01, 0.05, 0.05, 0.002])
        self.R_odo = np.diag([0.05])

        self.gyro_bias_state = np.array([[0.0]])
        self.P_gyro_bias = np.array([[1e-3]])
        self.Q_gyro_bias = np.array([[1e-8]])
        self.R_gyro_bias = np.array([[1e-1]])
        self.gyro_bias_est = 0.0
        self.rtk_initialized = False
        # diagnostic counters
        self._cov_log_counter = 0
        # Mahalanobis gating thresholds (disabled while tuning)
        self.rtk_gate_threshold = np.inf
        self.cart_gate_threshold = chi2.ppf(0.95, 2) if chi2 is not None else 5.991
        self.odo_gate_threshold = chi2.ppf(0.95, 1) if chi2 is not None else 3.841

    def init_states(self, init_pos, init_yaw):
        x, y, z = init_pos.x, init_pos.y, init_pos.z
        yaw_rad = np.radians(init_yaw)
        self.x_rtk = np.array([[x], [y], [z], [0.0], [0.0], [yaw_rad]])
        self.x_cart = np.array([[x], [y], [0.0], [yaw_rad]])
        self.x_odo = np.array([[x], [y], [z], [0.0], [0.0], [yaw_rad]])
        self.rtk_initialized = False

    def initialize_rtk_from_gnss(self, meas_xyz):
        self.x_rtk[:3, 0] = meas_xyz.flatten()
        self.P_rtk = np.diag([
            0.5,
            0.5,
            1.0,
            10.0,
            10.0,
            np.deg2rad(20.0)**2
        ])
        self.rtk_initialized = True

    def _update_gyro_bias(self, gyro_meas):
        prior_state = self.gyro_bias_state
        prior_cov = self.P_gyro_bias + self.Q_gyro_bias
        H = np.array([[1.0]])
        z = np.array([[gyro_meas]])
        y = z - (H @ prior_state)
        S = H @ prior_cov @ H.T + self.R_gyro_bias
        K = prior_cov @ H.T @ np.linalg.inv(S)
        self.gyro_bias_state = prior_state + K @ y
        # Bound the bias estimate to prevent ramp divergence
        self.gyro_bias_state = np.clip(self.gyro_bias_state, -0.5, 0.5)
        self.P_gyro_bias = (np.eye(1) - K @ H) @ prior_cov
        self.gyro_bias_est = float(self.gyro_bias_state[0, 0])
        return y, S

    def predict(self, imu_accel, imu_gyro_z, speedometer_v, dt: Optional[float] = None):
        # Allow dynamic timestep; fall back to configured dt
        dt = self.dt if (dt is None or dt <= 0.0) else float(dt)
        max_allowable_rotation_step = 0.25
        cleaned_gyro_z = np.clip(imu_gyro_z, -max_allowable_rotation_step, max_allowable_rotation_step)
        self._update_gyro_bias(cleaned_gyro_z)
        corrected_gyro_z = cleaned_gyro_z - self.gyro_bias_est

        # ------------------------------------------------------------------
        # RTK EKF prediction
        # Use a constant velocity motion model with yaw-rate coupling.
        # Do NOT integrate IMU acceleration because CARLA IMU acceleration
        # contains gravity and causes large drift.
        # ------------------------------------------------------------------

        # Compute heading change first (needed by both state transition and F)
        delta_yaw = corrected_gyro_z * dt

        # Rebuild state-transition for current dt (velocity rotated by delta_yaw)
        cos_dy = np.cos(delta_yaw)
        sin_dy = np.sin(delta_yaw)
        F_rtk = np.array([
            [1.0, 0.0, 0.0, dt, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, dt, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, cos_dy, -sin_dy, 0.0],
            [0.0, 0.0, 0.0, sin_dy, cos_dy, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        # Rebuild process noise from continuous-time motion model
        dt2 = dt * dt
        dt3 = dt2 * dt
        sa = self._sigma_a2
        sz = self._sigma_z2
        sy = self._sigma_yaw2
        self.Q_rtk = np.array([
            [sa*dt3/3, 0, 0, sa*dt2/2, 0, 0],
            [0, sa*dt3/3, 0, 0, sa*dt2/2, 0],
            [0, 0, sz*dt, 0, 0, 0],
            [sa*dt2/2, 0, 0, sa*dt, 0, 0],
            [0, sa*dt2/2, 0, 0, sa*dt, 0],
            [0, 0, 0, 0, 0, sy*dt],
        ])

        x_pred = self.x_rtk.copy()

        # Constant velocity prediction using current estimated velocity
        x_pred[0, 0] += self.x_rtk[3, 0] * dt
        x_pred[1, 0] += self.x_rtk[4, 0] * dt

        # Heading prediction
        x_pred[5, 0] = wrap_angle(self.x_rtk[5, 0] + delta_yaw)

        # Rotate velocity by heading change (yaw-rate coupling)
        vx_old = float(x_pred[3, 0])
        vy_old = float(x_pred[4, 0])
        x_pred[3, 0] = vx_old * cos_dy - vy_old * sin_dy
        x_pred[4, 0] = vx_old * sin_dy + vy_old * cos_dy

        # Standard covariance propagation (F reflects velocity rotation Jacobian)
        self.P_rtk = F_rtk @ self.P_rtk @ F_rtk.T + self.Q_rtk

        # Speedometer pseudo-measurement to estimate velocity magnitude
        yaw_pred = float(x_pred[5, 0])
        cos_yaw = np.cos(yaw_pred)
        sin_yaw = np.sin(yaw_pred)
        H_v = np.zeros((1, 6))
        H_v[0, 3] = cos_yaw
        H_v[0, 4] = sin_yaw
        z_v = np.array([[speedometer_v]])
        innov_v = z_v - H_v @ x_pred
        R_spd = np.array([[0.5]])
        S_v = H_v @ self.P_rtk @ H_v.T + R_spd
        K_v = self.P_rtk @ H_v.T @ np.linalg.inv(S_v)
        x_pred = x_pred + K_v @ innov_v
        x_pred[5, 0] = wrap_angle(x_pred[5, 0])
        I6 = np.eye(6)
        KH_v = K_v @ H_v
        self.P_rtk = (I6 - KH_v) @ self.P_rtk @ (I6 - KH_v).T + K_v @ R_spd @ K_v.T

        self.x_rtk = x_pred

        # Cartesian EKF prediction using current estimated states
        v_b = float(self.x_cart[2, 0])
        psi_b = float(self.x_cart[3, 0])
        f_cart = np.array([
            [self.x_cart[0, 0] + v_b * np.cos(psi_b) * dt],
            [self.x_cart[1, 0] + v_b * np.sin(psi_b) * dt],
            [v_b],
            [psi_b + corrected_gyro_z * dt],
        ])
        F_cart = np.eye(4)
        F_cart[0, 2] = np.cos(psi_b) * dt
        F_cart[0, 3] = -v_b * np.sin(psi_b) * dt
        F_cart[1, 2] = np.sin(psi_b) * dt
        F_cart[1, 3] = v_b * np.cos(psi_b) * dt
        self.x_cart = f_cart.copy()
        self.x_cart[3, 0] = wrap_angle(self.x_cart[3, 0])
        self.P_cart = F_cart @ self.P_cart @ F_cart.T + self.Q_cart

        # Speedometer pseudo-measurement to filter velocity
        H_vc = np.zeros((1, 4))
        H_vc[0, 2] = 1.0
        z_vc = np.array([[speedometer_v]])
        innov_vc = z_vc - H_vc @ self.x_cart
        R_spd_c = np.array([[2.0]])
        S_vc = H_vc @ self.P_cart @ H_vc.T + R_spd_c
        K_vc = self.P_cart @ H_vc.T @ np.linalg.inv(S_vc)
        self.x_cart = self.x_cart + K_vc @ innov_vc
        self.x_cart[3, 0] = wrap_angle(self.x_cart[3, 0])
        I4 = np.eye(4)
        KH_vc = K_vc @ H_vc
        self.P_cart = (I4 - KH_vc) @ self.P_cart @ (I4 - KH_vc).T + K_vc @ R_spd_c @ K_vc.T

        psi_c = self.x_odo[5, 0]
        self.x_odo[0, 0] += speedometer_v * np.cos(psi_c) * dt
        self.x_odo[1, 0] += speedometer_v * np.sin(psi_c) * dt
        self.x_odo[3, 0] = speedometer_v * np.cos(psi_c)
        self.x_odo[4, 0] = speedometer_v * np.sin(psi_c)
        self.x_odo[5, 0] = wrap_angle(self.x_odo[5, 0] + corrected_gyro_z * dt)
        F_odo = np.eye(6)
        F_odo[0, 3], F_odo[1, 4] = dt, dt
        self.P_odo = F_odo @ self.P_odo @ F_odo.T + self.Q_odo

    def update_rtk(self, meas_xyz):
        H = np.zeros((3, 6))
        H[0, 0], H[1, 1], H[2, 2] = 1.0, 1.0, 1.0
        y = meas_xyz - (H @ self.x_rtk)
        S = H @ self.P_rtk @ H.T + self.R_rtk
        K = self.P_rtk @ H.T @ np.linalg.inv(S)
        if np.any(np.isnan(y)) or np.any(np.isnan(S)):
            logger.warning("RTK update produced NaN innovation or covariance")

        # Mahalanobis (NIS) gating for a 3-DOF position measurement
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except Exception:
            nis = np.inf

        logger.debug("RTK innovation=%s", y.flatten())
        logger.debug("RTK Kalman gain diag=%s", np.diag(K))
        logger.debug("RTK predicted pos=(%.3f, %.3f, %.3f) meas=(%.3f, %.3f, %.3f)",
                     self.x_rtk[0, 0], self.x_rtk[1, 0], self.x_rtk[2, 0],
                     meas_xyz[0, 0], meas_xyz[1, 0], meas_xyz[2, 0])

        if nis > self.rtk_gate_threshold:
            logger.debug("RTK measurement rejected by Mahalanobis gating: NIS=%.3f > %.3f", nis, self.rtk_gate_threshold)
            return y, S

        # State update
        self.x_rtk = self.x_rtk + K @ y
        self.x_rtk[5, 0] = wrap_angle(self.x_rtk[5, 0])

        # Joseph form covariance update for numerical stability
        I = np.eye(6)
        KH = K @ H
        self.P_rtk = (I - KH) @ self.P_rtk @ (I - KH).T + K @ self.R_rtk @ K.T
        self.P_rtk = 0.5 * (self.P_rtk + self.P_rtk.T)

        # Periodically log eigenvalues to detect overconfidence
        self._cov_log_counter += 1
        if (self._cov_log_counter % 50) == 0:
            try:
                eigvals = np.linalg.eigvals(self.P_rtk)
                logger.info("RTK P eigenvalues (frame): %s", np.round(eigvals, 6))
            except Exception:
                pass

        return y, S

    def update_cartesian(self, meas_xy, current_speed):
        H = np.zeros((2, 4))
        H[0, 0], H[1, 1] = 1.0, 1.0
        z = meas_xy
        R = self.R_cart[:2, :2]
        dof = 2

        if current_speed > 2.0:
            dx = meas_xy[0, 0] - self.x_cart[0, 0]
            dy = meas_xy[1, 0] - self.x_cart[1, 0]
            displacement = np.hypot(dx, dy)
            if displacement > 0.2:
                derived_yaw = np.arctan2(dy, dx)
                yaw_residual = wrap_angle(derived_yaw - self.x_cart[3, 0])
                if np.abs(yaw_residual) < np.radians(45.0):
                    H = np.zeros((3, 4))
                    H[0, 0], H[1, 1], H[2, 3] = 1.0, 1.0, 1.0
                    z = np.vstack([meas_xy, [[derived_yaw]]])
                    R = self.R_cart
                    dof = 3

        y = z - (H @ self.x_cart)
        if z.shape[0] == 3:
            y[2, 0] = wrap_angle(y[2, 0])

        S = H @ self.P_cart @ H.T + R
        K = self.P_cart @ H.T @ np.linalg.inv(S)

        # Mahalanobis gating for Cartesian update
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except Exception:
            nis = np.inf
        gate = chi2.ppf(0.95, dof) if chi2 is not None else 5.991
        if nis > gate:
            logger.debug("Cartesian measurement rejected: NIS=%.3f > %.3f", nis, gate)
            return y, S

        self.x_cart = self.x_cart + K @ y
        self.x_cart[3, 0] = wrap_angle(self.x_cart[3, 0])

        # Joseph form covariance update
        I4 = np.eye(4)
        KH4 = K @ H
        self.P_cart = (I4 - KH4) @ self.P_cart @ (I4 - KH4).T + K @ R @ K.T
        self.P_cart = 0.5 * (self.P_cart + self.P_cart.T)
        return y, S

    def update_odometry(self, meas_v):
        H = np.zeros((1, 6))
        psi = self.x_odo[5, 0]
        H[0, 3], H[0, 4] = np.cos(psi), np.sin(psi)
        y = np.array([[meas_v]]) - (H @ self.x_odo)
        S = H @ self.P_odo @ H.T + self.R_odo
        K = self.P_odo @ H.T @ np.linalg.inv(S)

        # Mahalanobis gating for odometry
        try:
            nis = float((y.T @ np.linalg.inv(S) @ y).item())
        except Exception:
            nis = np.inf
        gate = chi2.ppf(0.95, 1) if chi2 is not None else 3.841
        if nis > gate:
            logger.debug("Odometry measurement rejected: NIS=%.3f > %.3f", nis, gate)
            return y, S

        self.x_odo = self.x_odo + K @ y
        self.x_odo[5, 0] = wrap_angle(self.x_odo[5, 0])

        # Joseph form covariance update for odometry
        I6 = np.eye(6)
        KH6 = K @ H
        self.P_odo = (I6 - KH6) @ self.P_odo @ (I6 - KH6).T + K @ self.R_odo @ K.T
        self.P_odo = 0.5 * (self.P_odo + self.P_odo.T)
        return y, S

# ==============================================================================
# 3. HELPER FUNCTIONS (PERCEPTION & IMAGE ENHANCEMENTS)
# ==============================================================================
class LaneState(Enum):
    NORMAL         = "NORMAL"
    CROSSWALK      = "CROSSWALK"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    MERGING        = "MERGING"
    CURVE          = "CURVE"

_clahe = cv2.createCLAHE(clipLimit=4.5, tileGridSize=(8, 8))

def night_enhance(rgb):
    lab       = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l, a, b   = cv2.split(lab)
    l         = _clahe.apply(l)
    rgb_out   = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2RGB)
    inv_gamma = 1.0 / 0.72
    lut       = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
    rgb_out   = cv2.LUT(rgb_out, lut)
    rgb_out   = cv2.convertScaleAbs(rgb_out, alpha=1.35, beta=8)
    return rgb_out

def preprocess_gpu(rgb_uint8):
    small = cv2.resize(rgb_uint8, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR)
    t = torch.from_numpy(small).permute(2, 0, 1).to(device)
    t = t.half() if is_half else t.float()
    t = t.div_(255.0).unsqueeze(0)
    t = (t - _MEAN) / _STD
    return t

def carla_to_rgb(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    return arr[:, :, :3][:, :, ::-1].copy()

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

def detect_crosswalk(mask):
    h, w = mask.shape
    roi  = mask[int(h * 0.55):int(h * 0.85), :]
    rows = np.sum(roi > 0, axis=1)
    return bool(np.sum(rows > w * 0.35) > 12)

def bev(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    key = (w, h)
    if key not in _BEV_TRANSFORMS:
        src = np.float32([[w * BEV_SRC_POINTS[0, 0], h * BEV_SRC_POINTS[0, 1]],
                          [w * BEV_SRC_POINTS[1, 0], h * BEV_SRC_POINTS[1, 1]],
                          [w * BEV_SRC_POINTS[2, 0], h * BEV_SRC_POINTS[2, 1]],
                          [w * BEV_SRC_POINTS[3, 0], h * BEV_SRC_POINTS[3, 1]]])
        dst = np.float32([[w * BEV_DST_POINTS[0, 0], h * BEV_DST_POINTS[0, 1]],
                          [w * BEV_DST_POINTS[1, 0], h * BEV_DST_POINTS[1, 1]],
                          [w * BEV_DST_POINTS[2, 0], h * BEV_DST_POINTS[2, 1]],
                          [w * BEV_DST_POINTS[3, 0], h * BEV_DST_POINTS[3, 1]]])
        _BEV_TRANSFORMS[key] = cv2.getPerspectiveTransform(src, dst)

    return cv2.warpPerspective(mask, _BEV_TRANSFORMS[key], (w, h))

def get_line_type(band):
    occ = np.mean(np.sum(band > 0, axis=1) > 2)
    return "solid" if occ > 0.68 else "dashed" if occ > 0.16 else "unknown"

_left_committed, _right_committed = None, None
_left_contrary, _right_contrary = 0, 0

def classify_boundary(side, hist, side_id):
    global _left_committed, _right_committed, _left_contrary, _right_contrary
    h, w = side.shape
    binary = (side > 0).astype(np.uint8)

    # Longitudinal occupancy: sample the lane centerline every 10 px
    col_profile = np.sum(binary, axis=0)
    peak_col = int(np.argmax(col_profile))
    peak_val = int(col_profile[peak_col])

    if peak_val < 2:
        raw, conf = "unknown", 0.25
    else:
        # Sample at every 10th row along the peak column
        sample_rows = np.arange(0, h, 10)
        occupancy = binary[sample_rows, min(peak_col, w - 1)] > 0

        # Binary sequence analysis: measure gaps vs total
        diff = np.diff(occupancy.astype(np.int8))
        starts = np.where(diff == 1)[0] + 1
        ends = np.where(diff == -1)[0] + 1
        if len(ends) and len(starts) and ends[0] < starts[0]:
            ends = ends[1:]
        n = min(len(starts), len(ends))

        if n < 2:
            # Single run — use run-length instead
            if occupancy.sum() > len(occupancy) * 0.85:
                raw = "solid"
            else:
                raw = "dashed"
        else:
            gaps = np.array(starts[1:n]) - np.array(ends[:n - 1])
            total_gap = gaps.sum()
            total_length = len(occupancy)
            gap_ratio = total_gap / max(total_length, 1)

            if gap_ratio < 0.15:
                raw = "solid"
            elif gap_ratio < 0.60:
                raw = "dashed"
            else:
                raw = "unknown"

        conf = min(1.0, peak_val / 8.0)

    hist.append((raw, conf))
    vote = max(set([x[0] for x in hist]), key=[x[0] for x in hist].count) if len(hist) >= 5 else raw

    committed = _left_committed if side_id == 'left' else _right_committed
    contrary  = _left_contrary if side_id == 'left' else _right_contrary

    if committed is None: committed, contrary = vote, 0
    elif vote == committed: contrary = 0
    else:
        contrary += 1
        if contrary >= 14: committed, contrary = vote, 0

    if side_id == 'left': _left_committed, _left_contrary = committed, contrary
    else: _right_committed, _right_contrary = committed, contrary
    return committed

def compute_lane_width(mask):
    # Robust lane width estimation: for each row compute leftmost and rightmost foreground pixel
    h, w = mask.shape[0], mask.shape[1]
    widths = []
    for y in range(int(h * 0.6), h, 4):
        cols = np.where(mask[y] > 0)[0]
        if cols.size == 0:
            continue
        left = int(cols.min())
        right = int(cols.max())
        if right - left >= 4 and right - left < w:
            widths.append(right - left)
    return float(np.median(widths)) if widths else 0.0


def compute_lane_heading(mask):
    h, w = mask.shape
    ys = np.arange(int(h * 0.45), h, 4, dtype=np.int32)
    centers = []
    for y in ys:
        row = np.where(mask[y] > 0)[0]
        if len(row) < 4:
            continue
        mid = row.mean()
        if np.isfinite(mid):
            centers.append((float(y), float(mid)))
    if len(centers) < 3:
        return 0.0
    pts = np.array(centers, dtype=np.float32)
    if np.ptp(pts[:, 0]) < 1e-6:
        return 0.0
    coeffs = np.polyfit(pts[:, 0], pts[:, 1], 1)
    slope = coeffs[0]
    return float(np.arctan2(1.0, slope))

kf_gap_x, kf_gap_p = 0.0, 1.0
def measure_dash_trend(side, trend_buf, trend_frame):
    global kf_gap_x, kf_gap_p
    h, w = side.shape
    col_profile = np.sum(side > 0, axis=0)
    peak_col = int(np.argmax(col_profile))
    if col_profile[peak_col] < 4: return None, None, trend_frame

    trend_frame += 1
    bh = max(6, int(w * 0.13))
    band = side[:, max(0, peak_col - bh):min(w, peak_col + bh)]
    signal = (np.sum(band > 0, axis=1) > 2).astype(np.int8)
    diff = np.diff(signal)
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1

    if len(starts) == 0 or len(ends) == 0: return None, None, trend_frame
    if ends[0] < starts[0]: ends = ends[1:]
    n = min(len(starts), len(ends))
    if n < 2: return None, None, trend_frame

    gaps = [int(starts[i + 1]) - int(ends[i]) for i in range(n - 1) if 5 < (starts[i + 1] - ends[i]) < h * 0.6]
    if not gaps: return None, None, trend_frame

    kf_gap_p += 0.08
    k = kf_gap_p / (kf_gap_p + 1.2)
    kf_gap_x = kf_gap_x + k * (float(np.mean(gaps)) - kf_gap_x)
    kf_gap_p = (1 - k) * kf_gap_p

    trend_buf.append((trend_frame, kf_gap_x))
    if len(trend_buf) < 12: return kf_gap_x, "calculating...", trend_frame

    data = np.array(list(trend_buf)[-15:])
    slope = np.polyfit(data[:, 0], data[:, 1], 1)[0]
    trend = "diverging" if slope > 0.20 else "converging" if slope < -0.20 else "constant"
    return kf_gap_x, trend, trend_frame

# ==============================================================================
# 4. EXPLANABLE AI (LIME) ASYNC PIPELINE
# ==============================================================================
_xai_running = False
_xai_lock    = threading.Lock()

def batch_predict(images):
    processed = []
    for img in images:
        if img.dtype != np.uint8:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        processed.append(cv2.resize(img, (_INF_W, _INF_H), interpolation=cv2.INTER_LINEAR))
    batch_tensor = torch.from_numpy(np.stack(processed)).permute(0, 3, 1, 2).to(device)
    batch_tensor = batch_tensor.half() if is_half else batch_tensor.float()
    batch_tensor = (batch_tensor.div_(255.0) - _MEAN) / _STD
    with torch.no_grad():
        probs = torch.softmax(model(batch_tensor).float(), dim=1)
    return probs.mean(dim=(2, 3)).cpu().numpy().astype(np.float64)

def _xai_worker(rgb_image, frame_id):
    global _xai_running
    with _xai_lock:
        if _xai_running: return
        _xai_running = True
    try:
        save_path = f"xai_output/xai_frame_{frame_id:06d}.png"
        h, w = rgb_image.shape[:2]
        inp = preprocess_gpu(rgb_image)
        with torch.no_grad():
            prob_vis = cv2.resize(torch.softmax(model(inp).float(), dim=1).cpu().squeeze(0)[EGO_CLASS].numpy(), (w, h))
        
        binary = (prob_vis > max(0.08, np.percentile(prob_vis[int(h * 0.4):, :], 75))).astype(np.uint8) * 255
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
        ego_lane = (labels == (1 + np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8) * 255 if n_labels > 1 else binary

        explainer = lime_image.LimeImageExplainer(verbose=False)
        explanation = explainer.explain_instance(rgb_image.astype(np.float64) / 255.0, batch_predict, top_labels=1, num_features=25, num_samples=150, batch_size=16, segmentation_fn=lambda x: quickshift(x, kernel_size=4, max_dist=20, ratio=0.2), random_seed=42)
        temp, lime_mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=12, hide_rest=False)
        
        trust_map = cv2.GaussianBlur(prob_vis, (21, 21), 0)
        trust_map = (trust_map - trust_map.min()) / (trust_map.max() - trust_map.min() + 1e-8)

        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        axs[0].imshow(rgb_image); axs[0].set_title("Input RGB"); axs[0].axis('off')
        axs[1].imshow(ego_lane, cmap='gray'); axs[1].set_title("Detected Lane Area"); axs[1].axis('off')
        axs[2].imshow(mark_boundaries(temp, lime_mask)); axs[2].set_title("LIME Attribution"); axs[2].axis('off')
        im = axs[3].imshow(trust_map, cmap='RdYlGn', vmin=0, vmax=1); axs[3].set_title("Trust Score Map"); axs[3].axis('off')
        plt.colorbar(im, ax=axs[3])
        plt.suptitle(f"XAI Analysis - Frame {frame_id} | Avg Conf: {prob_vis.mean():.4f}", fontsize=16)
        plt.tight_layout()
        plt.savefig(save_path, dpi=140, bbox_inches='tight')
        plt.close()
    except Exception as e: print(f"XAI Error: {e}")
    finally:
        with _xai_lock: _xai_running = False

# ==============================================================================
# 5. HUD & OVERLAY VISUALIZATION ENGINE
# ==============================================================================
def draw_lane_overlay(overlay: np.ndarray, mask: np.ndarray, _lfit_buf: deque, _rfit_buf: deque) -> None:
    h, w, mid = mask.shape[0], mask.shape[1], mask.shape[1] // 2
    l_pts, r_pts = [], []
    for y in range(int(h * 0.50), h, 3):
        row = np.where(mask[y] > 0)[0]
        if len(row) < 6: continue
        L, R = row[row < mid], row[row >= mid]
        if len(L) > 4: l_pts.append([float(y), float(np.max(L))])
        if len(R) > 4: r_pts.append([float(y), float(np.min(R))])

    if len(l_pts) >= 6: _lfit_buf.append(np.polyfit(np.array(l_pts)[:, 0], np.array(l_pts)[:, 1], 2))
    if len(r_pts) >= 6: _rfit_buf.append(np.polyfit(np.array(r_pts)[:, 0], np.array(r_pts)[:, 1], 2))
    if len(_lfit_buf) < 3 or len(_rfit_buf) < 3: return

    lfit = np.mean(list(_lfit_buf)[-8:], axis=0)
    rfit = np.mean(list(_rfit_buf)[-8:], axis=0)
    ys  = np.arange(int(h * 0.50), h, dtype=np.float32)
    lxs = np.polyval(lfit, ys).clip(0, mid - 1).astype(np.int32)
    rxs = np.polyval(rfit, ys).clip(mid, w - 1).astype(np.int32)
    yi  = ys.astype(np.int32)

    fill = overlay.copy()
    pts  = np.concatenate([np.stack([lxs, yi], axis=1), np.stack([rxs, yi], axis=1)[::-1]], axis=0)
    cv2.fillPoly(fill, [pts], (0, 180, 0))
    cv2.addWeighted(fill, 0.25, overlay, 0.75, 0, overlay)
    for i in range(len(yi) - 1):
        cv2.line(overlay, (lxs[i], yi[i]), (lxs[i + 1], yi[i + 1]), (0, 255, 255), 3)
        cv2.line(overlay, (rxs[i], yi[i]), (rxs[i + 1], yi[i + 1]), (0, 255, 255), 3)

def draw_hud(overlay: np.ndarray, lane_label: str, state: LaneState, speed: float, gap: Optional[float], trend: Optional[str], width: float, num_v: int, num_w: int) -> None:
    h, w = overlay.shape[:2]
    bar  = np.zeros((130, w, 3), dtype=np.uint8)
    overlay[0:130] = cv2.addWeighted(overlay[0:130], 0.35, bar, 0.65, 0)
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(overlay, f"Lane: {lane_label}", (20, 38), font, 0.85, (0, 255, 255), 2)
    cv2.putText(overlay, f"State: {state.value} | Speed: {speed:.1f} km/h", (20, 68), font, 0.72, (0, 255, 180), 2)
    info = f"Gap: {gap:.1f}px | Trend: {trend} | Width: {int(width)}px" if gap is not None else f"Width: {int(width)}px"
    cv2.putText(overlay, info, (20, 98), font, 0.68, (255, 255, 100), 2)
    traffic_txt = f"NPC Cars: {num_v}  Walkers: {num_w}"
    txt_size    = cv2.getTextSize(traffic_txt, font, 0.58, 1)[0]
    cv2.putText(overlay, traffic_txt, (w - txt_size[0] - 14, 28), font, 0.58, (200, 200, 255), 1)


def can_show_gui():
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

# ==============================================================================
# 6. MAIN INTEGRATED EXECUTION LOOP
# ==============================================================================
def main() -> None:
    global CONFIG
    CONFIG = build_config()
    save_config(CONFIG, CONFIG.output_dir)
    random.seed(CONFIG.seed)
    np.random.seed(CONFIG.seed)
    torch.manual_seed(CONFIG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CONFIG.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    client = carla.Client("localhost", 2000)
    client.set_timeout(15.0)
    world = client.load_world(CONFIG.map_name)

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(True)

    # Set weather parameters to midnight/wet night profile
    world.set_weather(carla.WeatherParameters(cloudiness=10.0, precipitation= 10.0, fog_density= 0.0, wetness=0.0, sun_altitude_angle= 55.0))

    blueprints   = world.get_blueprint_library()
    actor_list   = []
    npc_vehicles = []
    npc_walkers  = []

    # Spawn Ego Vehicle
    ego_bp = blueprints.find("vehicle.tesla.model3")
    spawn_point = world.get_map().get_spawn_points()[20]
    ego_vehicle = world.spawn_actor(ego_bp, spawn_point)
    actor_list.append(ego_vehicle)
    ego_vehicle.set_autopilot(True)
    traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, 10)
    
    # Enable Night Lights on Ego Vehicle
    _headlight_state = carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.HighBeam | carla.VehicleLightState.Fog)
    ego_vehicle.set_light_state(_headlight_state)

    # Setup Frame Synchronization Queue
    sensor_queue = queue.Queue()
    def make_sensor_callback(name):
        return lambda data: sensor_queue.put((data.frame, name, data))

    # Spawn Navigation Data Collection Sensors (IMU + GNSS)
    gnss_bp = blueprints.find('sensor.other.gnss')
    gnss_bp.set_attribute('noise_lat_stddev', '0.0000001'); gnss_bp.set_attribute('noise_lon_stddev', '0.0000001')
    gnss_sensor = world.spawn_actor(gnss_bp, carla.Transform(carla.Location(z=2.0)), attach_to=ego_vehicle)
    actor_list.append(gnss_sensor)
    gnss_sensor.listen(make_sensor_callback('GNSS'))

    imu_bp = blueprints.find('sensor.other.imu')
    imu_bp.set_attribute('noise_gyro_stddev_z', '0.001')
    imu_sensor = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego_vehicle)
    actor_list.append(imu_sensor)
    imu_sensor.listen(make_sensor_callback('IMU'))

    # Spawn Front-Facing Camera RGB Perception Layer Sensor
    camera_bp = blueprints.find("sensor.camera.rgb")
    camera_bp.set_attribute("image_size_x", "640"); camera_bp.set_attribute("image_size_y", "360"); camera_bp.set_attribute("fov", "100")
    camera_sensor = world.spawn_actor(camera_bp, carla.Transform(carla.Location(x=0.25, y=0.0, z=1.45), carla.Rotation(pitch=-5)), attach_to=ego_vehicle)
    actor_list.append(camera_sensor)
    camera_sensor.listen(make_sensor_callback('CAMERA'))

    # --- Spawning NPC Vehicles and Pedestrians ---
    logger.info("Generating ambient traffic configuration...")
    vehicle_filter = ["vehicle.audi.*", "vehicle.bmw.*", "vehicle.chevrolet.*", "vehicle.ford.*", "vehicle.tesla.*"]
    npc_bps = []
    for filt in vehicle_filter: npc_bps.extend(blueprints.filter(filt))
    npc_bps = [bp for bp in npc_bps if int(bp.get_attribute("number_of_wheels")) == 4]
    
    available_sps = [sp for i, sp in enumerate(world.get_map().get_spawn_points()) if i != 20]
    random.shuffle(available_sps)
    
    spawn_cmds = []
    for i in range(min(40, len(available_sps))):
        bp = random.choice(npc_bps)
        if bp.has_attribute("color"): bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        spawn_cmds.append(carla.command.SpawnActor(bp, available_sps[i]).then(carla.command.SetAutopilot(carla.command.FutureActor, True, traffic_manager.get_port())))
    
    results = client.apply_batch_sync(spawn_cmds, True)
    for res in results:
        if not res.error:
            actor = world.get_actor(res.actor_id)
            if actor:
                actor.set_light_state(carla.VehicleLightState(carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam))
                traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(-15, 20))
                npc_vehicles.append(actor)

    # Initializing Filter Objects and Tracking Telemetry History
    filters = KinematicEKFs(dt=0.05)
    world.tick()
    init_transform = ego_vehicle.get_transform()
    filters.init_states(init_transform.location, init_transform.rotation.yaw)

    # Dynamic Buffers Allocation
    left_hist, right_hist = deque(maxlen=16), deque(maxlen=16)
    mask_buf = deque(maxlen=5)
    _lfit_buf, _rfit_buf = deque(maxlen=12), deque(maxlen=12)
    trend_buf = deque(maxlen=30)
    
    prev_left, prev_right = "solid", "solid"
    last_gap, last_trend = None, None
    current_state = LaneState.NORMAL
    freeze_count, trend_frame = 0, 0
    prev_mask = None
    init_lat, init_lon, init_alt = None, None, None
    origin_x, origin_y, origin_z = None, None, None
    enu_to_world_angle = None
    enu_alignment_tried = False
    logs: List[Dict[str, Any]] = []
    total_simulation_steps = CONFIG.sim_length
    metrics_csv = Path(CONFIG.output_dir) / "simulation_metrics.csv"
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    show_gui = can_show_gui()
    if not show_gui:
        print("ℹ️ No GUI display detected; skipping OpenCV window rendering and continuing in headless mode.")
    logger.info("Main loop engaged. Processing %d frames", total_simulation_steps)
    try:
        for step in range(total_simulation_steps):
            # Advance simulation and then read sim timestamp for dt
            world.tick()
            snapshot = world.get_snapshot()
            dt_now = float(snapshot.timestamp.delta_seconds)
            frame_start = time.perf_counter()

            # Retrieve Synchronized Frame Sensor Packets
            frame_data = {}
            timeout_occurred = False
            start_fetch_time = time.time()
            while len(frame_data) < 3:
                try:
                    f, name, data = sensor_queue.get(timeout=0.1)
                    if data.frame == step or True: # Keep latest packet entries
                        frame_data[name] = data
                except queue.Empty:
                    if time.time() - start_fetch_time > 1.5:
                        timeout_occurred = True
                        break
            if timeout_occurred or 'CAMERA' not in frame_data or 'IMU' not in frame_data or 'GNSS' not in frame_data:
                continue

            # Unpack hardware registers
            cam_packet = frame_data['CAMERA']
            imu_packet = frame_data['IMU']
            gnss_packet = frame_data['GNSS']

            # Ground-Truth Vehicle Telemetry Extraction
            gt_trans = ego_vehicle.get_transform()
            gt_vel   = ego_vehicle.get_velocity()
            gt_speed_kmh = np.sqrt(gt_vel.x**2 + gt_vel.y**2 + gt_vel.z**2) * 3.6
            gt_speed_ms  = gt_speed_kmh / 3.6
            
            # Simulated Hardware Gyro step-fault loop injection around step index index 240
            fault_offset = 0.45 if step >= 240 else 0.0
            raw_gyro_z = imu_packet.gyroscope.z + fault_offset
            gyro_bias_state = filters.gyro_bias_state

            lat_deg, lon_deg, alt_m = float(gnss_packet.latitude), float(gnss_packet.longitude), float(gnss_packet.altitude)
            if init_lat is None:
                init_lat, init_lon, init_alt = lat_deg, lon_deg, alt_m
                origin_x, origin_y, origin_z = float(gt_trans.location.x), float(gt_trans.location.y), float(gt_trans.location.z)
                logger.info("RTK origin initialized: lat=%.9f lon=%.9f alt=%.3f  world=(%.3f, %.3f, %.3f)",
                            init_lat, init_lon, init_alt,
                            origin_x, origin_y, origin_z)

            east, north, up = geodetic_to_enu(lat_deg, lon_deg, alt_m, init_lat, init_lon, init_alt)
            # Estimate ENU->world rotation by accumulating multiple samples to reduce GNSS noise
            if enu_to_world_angle is None and not enu_alignment_tried and step > 0:
                d_enu = np.array([east, north], dtype=np.float64)
                d_world = np.array([float(gt_trans.location.x) - origin_x, float(gt_trans.location.y) - origin_y], dtype=np.float64)
                if np.linalg.norm(d_enu) > 0.5 and np.linalg.norm(d_world) > 0.5:
                    angle_enu   = np.arctan2(d_enu[1], d_enu[0])
                    angle_world = np.arctan2(d_world[1], d_world[0])
                    ang_diff = wrap_angle(angle_world - angle_enu)
                    # accumulate differences in a buffer and take median for robustness
                    if 'enu_align_buf' not in locals() and 'enu_align_buf' not in globals():
                        enu_align_buf = deque(maxlen=200)
                    enu_align_buf.append(ang_diff)
                    if len(enu_align_buf) >= 100:
                        enu_to_world_angle = float(np.median(np.array(enu_align_buf)))
                        enu_alignment_tried = True
                        logger.info("Estimated ENU->world rotation (median over %d samples) = %.3f°", len(enu_align_buf), np.degrees(enu_to_world_angle))
                        # Rotate filter states to world frame to match future measurements
                        ca = np.cos(enu_to_world_angle)
                        sa = np.sin(enu_to_world_angle)
                        R2 = np.array([[ca, -sa], [sa, ca]])
                        # Rotate RTK state [x, y] and [vx, vy]
                        xy = filters.x_rtk[[0, 1], 0] - np.array([origin_x, origin_y])
                        xy_rot = R2 @ xy
                        filters.x_rtk[0, 0] = origin_x + xy_rot[0]
                        filters.x_rtk[1, 0] = origin_y + xy_rot[1]
                        filters.x_rtk[[3, 4], 0] = R2 @ filters.x_rtk[[3, 4], 0]
                        filters.x_rtk[5, 0] = wrap_angle(filters.x_rtk[5, 0] + enu_to_world_angle)
                        # Rotate RTK covariance
                        R6 = np.eye(6)
                        R6[:2, :2] = R2
                        R6[3:5, 3:5] = R2
                        filters.P_rtk = R6 @ filters.P_rtk @ R6.T
                        # Rotate Cartesian state [x, y]
                        xy_c = filters.x_cart[[0, 1], 0] - np.array([origin_x, origin_y])
                        xy_c_rot = R2 @ xy_c
                        filters.x_cart[0, 0] = origin_x + xy_c_rot[0]
                        filters.x_cart[1, 0] = origin_y + xy_c_rot[1]
                        filters.x_cart[3, 0] = wrap_angle(filters.x_cart[3, 0] + enu_to_world_angle)
                        R4 = np.eye(4)
                        R4[:2, :2] = R2
                        filters.P_cart = R4 @ filters.P_cart @ R4.T
                        # Rotate Odometry state [x, y] and [vx, vy]
                        xy_o = filters.x_odo[[0, 1], 0] - np.array([origin_x, origin_y])
                        xy_o_rot = R2 @ xy_o
                        filters.x_odo[0, 0] = origin_x + xy_o_rot[0]
                        filters.x_odo[1, 0] = origin_y + xy_o_rot[1]
                        filters.x_odo[[3, 4], 0] = R2 @ filters.x_odo[[3, 4], 0]
                        filters.x_odo[5, 0] = wrap_angle(filters.x_odo[5, 0] + enu_to_world_angle)
                        filters.P_odo = R6 @ filters.P_odo @ R6.T
                else:
                    # insufficient displacement; wait for more samples
                    pass

            if enu_to_world_angle is None:
                proj_x = origin_x + east
                proj_y = origin_y + north
            else:
                cos_yaw = np.cos(enu_to_world_angle)
                sin_yaw = np.sin(enu_to_world_angle)
                proj_x = origin_x + east * cos_yaw - north * sin_yaw
                proj_y = origin_y + east * sin_yaw + north * cos_yaw
            proj_z = origin_z + up

            meas_xyz = np.array([[proj_x], [proj_y], [proj_z]])
            meas_xy_degraded = meas_xyz[:2] + np.random.normal(0, 0.15, size=(2, 1))

            if step < 5:
                logger.info("RTK measurement[%d]: GNSS lat=%.9f lon=%.9f alt=%.3f east=%.3f north=%.3f up=%.3f world=(%.3f, %.3f, %.3f) gt=(%.3f, %.3f, %.3f)",
                            step, lat_deg, lon_deg, alt_m, east, north, up,
                            proj_x, proj_y, proj_z,
                            gt_trans.location.x, gt_trans.location.y, gt_trans.location.z)

            if not filters.rtk_initialized:
                filters.initialize_rtk_from_gnss(meas_xyz)

            # No fallback — let the buffer accumulate naturally to the required count.
            # The main accumulation block above handles everything once it has enough samples.

            # Step Kinematic Filters (use dynamic dt)
            filters.predict(imu_packet.accelerometer, raw_gyro_z, gt_speed_ms, dt=dt_now)
            rtk_y, rtk_S = filters.update_rtk(meas_xyz)
            cart_y, cart_S = filters.update_cartesian(meas_xy_degraded, gt_speed_ms)
            odo_y, odo_S = filters.update_odometry(gt_speed_ms)

            # --- Perception Processing Pipeline ---
            perception_start = time.perf_counter()
            raw_rgb = carla_to_rgb(cam_packet)
            enhanced_rgb = night_enhance(raw_rgb)
            inp_tensor = preprocess_gpu(enhanced_rgb)
            
            with torch.no_grad():
                pred_logits = model(inp_tensor)
                prob_all = torch.softmax(pred_logits.float(), dim=1)
                avg_conf = float(prob_all[0, EGO_CLASS].mean().item())
                prob_vis = cv2.resize(prob_all[0, EGO_CLASS].cpu().numpy(), (raw_rgb.shape[1], raw_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)

            adaptive_threshold = np.clip(
                np.percentile(prob_vis[prob_vis > 0], 70),
                0.30,
                0.65
            )
            mask = (prob_vis > adaptive_threshold).astype(np.uint8) * 255
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 5), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
            mask = filter_horizontal_noise(mask)
            mask_buf.append(mask)
            fused_mask = np.median(np.array(mask_buf), axis=0).astype(np.uint8)

            crosswalk = detect_crosswalk(fused_mask)
            b_warp = bev(fused_mask)
            
            if freeze_count > 0:
                lt, rt = prev_left, prev_right
                freeze_count -= 1
            else:
                lt = classify_boundary(b_warp[:, :b_warp.shape[1]//2], left_hist, 'left')
                rt = classify_boundary(b_warp[:, b_warp.shape[1]//2:], right_hist, 'right')
                prev_left, prev_right = lt, rt

            lane_label = f"{lt} | {rt}"
            lane_width = compute_lane_width(fused_mask)
            lane_heading = compute_lane_heading(fused_mask)
            if np.any(fused_mask > 0):
                lane_probs = prob_vis[fused_mask > 0]
                lane_confidence = float(np.sort(lane_probs)[-max(1, len(lane_probs)//2):].mean())
            else:
                lane_confidence = 0.0
            if prev_mask is not None and prev_mask.shape == fused_mask.shape:
                intersection = np.logical_and(fused_mask > 0, prev_mask > 0).sum()
                union = np.logical_or(fused_mask > 0, prev_mask > 0).sum()
                temporal_iou = float(intersection / max(union, 1))
                temporal_dice = float(2.0 * intersection / max(intersection * 2 + (fused_mask > 0).sum() + (prev_mask > 0).sum() - 2 * intersection, 1))
            else:
                temporal_iou, temporal_dice = 0.0, 0.0
            prev_mask = fused_mask.copy()

            side_segmented = b_warp[:, :b_warp.shape[1]//2] if "dashed" in lt else (b_warp[:, b_warp.shape[1]//2:] if "dashed" in rt else None)
            if side_segmented is not None and not crosswalk:
                gap, trend, trend_frame = measure_dash_trend(side_segmented, trend_buf, trend_frame)
                if gap is not None:
                    last_gap, last_trend = gap, trend

            # Update System FSM State
            if crosswalk:
                current_state, freeze_count = LaneState.CROSSWALK, 12
            elif lane_width < 60 or lane_width > 500:
                current_state, freeze_count = LaneState.LOW_CONFIDENCE, max(freeze_count, 6)
            elif last_gap is not None and last_gap > 180 and ("dashed" in lt or "dashed" in rt):
                current_state = LaneState.MERGING
            else:
                current_state = LaneState.NORMAL

            perception_elapsed = time.perf_counter() - perception_start

            # Async XAI Trigger Assignment Check
            if step % 60 == 0:
                threading.Thread(target=_xai_worker, args=(raw_rgb.copy(), step), daemon=True).start()

            # --- Structural Log Appending ---
            # Innovation norms
            rtk_resid = float(np.linalg.norm(rtk_y))
            cart_resid = float(np.linalg.norm(cart_y))
            odo_resid = float(np.linalg.norm(odo_y))

            # True NIS
            nis_rtk = float((rtk_y.T @ np.linalg.solve(rtk_S, rtk_y)).item())
            nis_cart = float(
            (cart_y.T @ np.linalg.solve(cart_S, cart_y)).item()
            )
            nis_odo = float(
                (odo_y.T @ np.linalg.solve(odo_S, odo_y)).item()
            )

            # True state errors
            e_rtk = np.array([
                [filters.x_rtk[0,0] - gt_trans.location.x],
                [filters.x_rtk[1,0] - gt_trans.location.y],
                [filters.x_rtk[2,0] - gt_trans.location.z]
            ])

            e_cart = np.array([
                [filters.x_cart[0,0] - gt_trans.location.x],
                [filters.x_cart[1,0] - gt_trans.location.y],
                [wrap_angle(filters.x_cart[3,0] - np.radians(gt_trans.rotation.yaw))]
            ])

            e_odo = np.array([
            [filters.x_odo[0,0] - gt_trans.location.x],
            [filters.x_odo[1,0] - gt_trans.location.y],
            [filters.x_odo[2,0] - gt_trans.location.z],
            [wrap_angle(filters.x_odo[5,0] - np.radians(gt_trans.rotation.yaw))]
            ])

            P_odo_nees = filters.P_odo[np.ix_([0,1,2,5],[0,1,2,5])]

            nees_rtk = float(
                (e_rtk.T @ np.linalg.solve(filters.P_rtk[:3, :3], e_rtk)).item()
            )

            P_cart_nees = filters.P_cart[np.ix_([0, 1, 3], [0, 1, 3])]
            nees_cart = float(
                (e_cart.T @ np.linalg.solve(P_cart_nees, e_cart)).item()
            )

            nees_odo = float(
                e_odo.T @ np.linalg.solve(P_odo_nees, e_odo)
            )
            logs.append({
                "step": step,
                "gt_x": gt_trans.location.x, "gt_y": gt_trans.location.y, "gt_yaw": np.radians(gt_trans.rotation.yaw),
                "rtk_x": filters.x_rtk[0,0], "rtk_y": filters.x_rtk[1,0], "rtk_yaw": filters.x_rtk[5,0],
                "cart_x": filters.x_cart[0,0], "cart_y": filters.x_cart[1,0], "cart_yaw": filters.x_cart[3,0],
                "odo_x": filters.x_odo[0,0], "odo_y": filters.x_odo[1,0], "odo_yaw": filters.x_odo[5,0],
                "gyro_bias_rate": filters.gyro_bias_est,
                "gyro_bias_offset": filters.gyro_bias_est,
                "rtk_innov_norm": rtk_resid,
                "cart_innov_norm": cart_resid,
                "odo_innov_norm": odo_resid,
                "nees_rtk": nees_rtk,
                "nees_cart": nees_cart,
                "nees_odo": nees_odo,
                "nis_rtk": nis_rtk,
                "nis_cart": nis_cart,
                "nis_odo": nis_odo,
                "lane_width": lane_width,
                "avg_confidence": avg_conf,
                "lane_confidence": lane_confidence,
                "temporal_iou": temporal_iou,
                "temporal_dice": temporal_dice,
                "speed_kmh": gt_speed_kmh,
                "adaptive_threshold": adaptive_threshold,
                "lane_heading": lane_heading,
                "rtk_pxx": filters.P_rtk[0, 0],
                "rtk_pxy": filters.P_rtk[0, 1],
                "rtk_pyy": filters.P_rtk[1, 1],
            })

            # Rendering Unified Frame Graphic Visualization Window
            overlay = raw_rgb.copy()
            overlay[fused_mask == 255] = [0, 255, 120]
            draw_lane_overlay(overlay, fused_mask, _lfit_buf, _rfit_buf)
            draw_hud(overlay, lane_label, current_state, gt_speed_kmh, last_gap, last_trend, lane_width, len(npc_vehicles), 0)
            
            # Append Confidence Mini Window on GUI
            prob_small = cv2.applyColorMap(cv2.resize((prob_vis * 255).clip(0, 255).astype(np.uint8), (160, 80)), cv2.COLORMAP_JET)
            overlay[overlay.shape[0] - 90:overlay.shape[0] - 10, 10:170] = prob_small

            if step % CONFIG.save_every == 0 or step == total_simulation_steps - 1:
                pd.DataFrame(logs).to_csv(metrics_csv, index=False)

            if show_gui:
                try:
                    cv2.imshow("Integrated CARLA Perception & Navigation Localization Grid", overlay)
                    if cv2.waitKey(1) == 27: break
                except cv2.error:
                    show_gui = False
                    print("ℹ️ OpenCV display backend unavailable; continuing without GUI window output.")
            elif step % 30 == 0:
                cv2.imwrite(f"xai_output/overlay_frame_{step:06d}.png", overlay)

            if step % 30 == 0:
                logger.info("step=%d | perception=%.3fs | total=%.3fs | adapt_thr=%.3f", step, perception_elapsed, time.perf_counter() - frame_start, adaptive_threshold)

    except (RuntimeError, ConnectionResetError, OSError) as exc:
        logger.exception("CARLA runtime interrupted: %s", exc)
    finally:
        print("\n🏁 Session Terminated. Cleaning up workspace environment actors...")
        settings.synchronous_mode = False
        world.apply_settings(settings)
        for actor in actor_list + npc_vehicles:
            if actor is not None: actor.destroy()
        if show_gui:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass

    # ==============================================================================
    # 7. METRICS CALCULATIONS, ANALYTICAL EXPORTS & EVALUATED GRAPHS
    # ==============================================================================
    df = pd.DataFrame(logs)
    if df.empty:
        logger.error("No simulation metrics collected. Summary generation skipped.")
        return

    # Calculate Analytics Tracking Deviations
    rmse_rtk  = np.sqrt(((df['rtk_x'] - df['gt_x'])**2 + (df['rtk_y'] - df['gt_y'])**2).mean())
    rmse_cart = np.sqrt(((df['cart_x'] - df['gt_x'])**2 + (df['cart_y'] - df['gt_y'])**2).mean())
    rmse_odo = np.sqrt(((df['odo_x'] - df['gt_x'])**2 + (df['odo_y'] - df['gt_y'])**2).mean())

    p95_rtk = np.percentile(
    np.sqrt((df['rtk_x'] - df['gt_x'])**2 +
            (df['rtk_y'] - df['gt_y'])**2), 95)

    max_rtk = np.max(
        np.sqrt((df['rtk_x'] - df['gt_x'])**2 +
                (df['rtk_y'] - df['gt_y'])**2))

    anees_rtk = df['nees_rtk'].mean()
    anis_rtk = df['nis_rtk'].mean()
    
    yaw_err_rtk  = np.degrees(np.abs(wrap_angle(df['rtk_yaw'] - df['gt_yaw'])).mean())
    yaw_err_cart = np.degrees(np.abs(wrap_angle(df['cart_yaw'] - df['gt_yaw'])).mean())
    yaw_err_odo  = np.degrees(np.abs(wrap_angle(df['odo_yaw'] - df['gt_yaw'])).mean())

    logger.info("\n%s\n   COMPONENT PERFORMANCE EVALUATION RESULTS METRICS REPORT\n%s", "="*60, "="*60)
    logger.info("Total processed simulation iterations: %d ticks", len(df))
    print(f"🔹 Lane Detection Confidence (lane pixels) : {df['lane_confidence'].mean()*100:.2f} %")
    print(f"🔹 Evaluated Tracked Vehicle Average Width : {df['lane_width'].mean():.2f} px")
    print("-" * 60)
    print("📍 CRITICAL FILTER LOCALIZATION TRAJECTORY COMPARISONS:")
    print(f" -> 6-State INS/RTK Fusion EKF Engine      : RMSE Position Error = {rmse_rtk:.4f} m | Mean Heading Delta = {yaw_err_rtk:.4f}°")
    print(f" -> 4-State Cartesian Kinematic EKF Engine  : RMSE Position Error = {rmse_cart:.4f} m | Mean Heading Delta = {yaw_err_cart:.4f}°")
    print(f" -> 6-State INS/Odometry Dead-Reckon Engine : RMSE Position Error = {rmse_odo:.4f} m | Mean Heading Delta = {yaw_err_odo:.4f}°")
    print("="*60)
    print(f"RTK 95th Percentile Error : {p95_rtk:.3f} m")
    print(f"RTK Maximum Error         : {max_rtk:.3f} m")
    print(f"RTK ANEES                : {anees_rtk:.3f}")
    print(f"RTK ANIS                 : {anis_rtk:.3f}")

    # Compute 95% covariance ellipse coverage
    inside_count = 0
    for i in range(len(df)):
        pos = np.array([df['rtk_x'].iloc[i], df['rtk_y'].iloc[i]], dtype=np.float64)
        gt_pos = np.array([df['gt_x'].iloc[i], df['gt_y'].iloc[i]], dtype=np.float64)
        cov = np.array([
            [df['rtk_pxx'].iloc[i], df['rtk_pxy'].iloc[i]],
            [df['rtk_pxy'].iloc[i], df['rtk_pyy'].iloc[i]]
        ], dtype=np.float64)
        try:
            d2 = (gt_pos - pos).T @ np.linalg.solve(cov, gt_pos - pos)
            if d2 <= 5.991:
                inside_count += 1
        except Exception:
            pass
    coverage_pct = 100.0 * inside_count / max(len(df), 1)
    print(f"RTK 95% Ellipse Coverage : {coverage_pct:.1f}% (target: ~95%)")

    # Export Plot Figures
    print("📈 Generating analytical verification plot graphics...")
    output_dir = "reports/segmentation_plots"

    # Plot 1: Top-Down Localization Tracking Trajectories
    plt.figure(figsize=(8, 6.5))
    plt.plot(df['gt_x'], df['gt_y'], 'k-', lw=2.5, label='Ground Truth Reference')
    plt.plot(df['rtk_x'], df['rtk_y'], 'g--', lw=1.5, label=f'6-state INS/RTK EKF (RMSE: {rmse_rtk:.3f}m)')
    plt.plot(df['cart_x'], df['cart_y'], 'b-.', lw=1.5, label=f'4-state Cartesian EKF (RMSE: {rmse_cart:.3f}m)')
    plt.plot(df['odo_x'], df['odo_y'], 'r:', lw=1.5, label=f'6-state INS/Odo EKF (RMSE: {rmse_odo:.3f}m)')
    for idx in range(0, len(df), max(1, len(df) // 12)):
        try:
            cov_xy = np.array([[df['rtk_x'].iloc[idx], df['rtk_y'].iloc[idx]]])
            _ = cov_xy
        except Exception:
            continue
    plt.title('Top-Down Vehicle Localization Trajectory Profile', fontweight='bold')
    plt.xlabel('Global Coordinate X position (meters)'); plt.ylabel('Global Coordinate Y position (meters)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/trajectory_comparison.png", dpi=300); plt.close()

    # Plot 1b: Trajectory covariance ellipses
    plt.figure(figsize=(8, 6.5))
    plt.plot(df['gt_x'], df['gt_y'], 'k-', lw=2.0, label='Ground Truth Reference')
    plt.plot(df['rtk_x'], df['rtk_y'], 'g--', lw=1.2, label='RTK Estimate')
    for idx in range(0, len(df), max(1, len(df) // 12)):
        pos = np.array([df['rtk_x'].iloc[idx], df['rtk_y'].iloc[idx]], dtype=np.float64)
        cov = np.array([
            [df['rtk_pxx'].iloc[idx], df['rtk_pxy'].iloc[idx]],
            [df['rtk_pxy'].iloc[idx], df['rtk_pyy'].iloc[idx]]
        ], dtype=np.float64)
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 1e-6, None)
        scale = np.sqrt(5.991)
        width = 2 * scale * np.sqrt(evals[1])
        height = 2 * scale * np.sqrt(evals[0])
        angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
        ellipse = Ellipse(pos, width, height, angle=angle, alpha=0.12, color='green', edgecolor='none')
        plt.gca().add_patch(ellipse)
    plt.title('RTK Trajectory with Covariance Ellipses', fontweight='bold')
    plt.xlabel('Global Coordinate X position (meters)'); plt.ylabel('Global Coordinate Y position (meters)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/trajectory_covariance.png", dpi=300); plt.close()

    # Plot 2: Absolute Positioning Trajectory Euclidean Step Tracking Errors
    plt.figure(figsize=(9, 4))
    plt.plot(np.sqrt((df['rtk_x']-df['gt_x'])**2 + (df['rtk_y']-df['gt_y'])**2), 'g-', label='6-state INS/RTK')
    plt.plot(np.sqrt((df['cart_x']-df['gt_x'])**2 + (df['cart_y']-df['gt_y'])**2), 'b-', label='4-state Cartesian')
    plt.plot(np.sqrt((df['odo_x']-df['gt_x'])**2 + (df['odo_y']-df['gt_y'])**2), 'r-', label='6-state INS/Odo')
    plt.title('Absolute Spatial Vector Displacement Error Time Series', fontweight='bold')
    plt.xlabel('Simulation Verification Frames Step Index'); plt.ylabel('Translation Deviation Distance Error (m)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/position_rmse_error.png", dpi=300); plt.close()

    # Plot 3: Gyroscope Bias State Tracking
    plt.figure(figsize=(9, 3.5))
    plt.plot(df['gyro_bias_rate'], 'm-', lw=2, label='Gyro Rate Bias Estimate')
    plt.plot(df['gyro_bias_offset'], 'c--', lw=2, label='Gyro Offset Bias Estimate')
    plt.axhline(y=0.0, color='k', linestyle='--', alpha=0.5)
    plt.title('Gyroscope Bias State Tracking', fontweight='bold')
    plt.xlabel('Simulation Frame Index'); plt.ylabel('Bias Value (rad/s)')
    plt.legend(); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/gyro_bias_convergence.png", dpi=300); plt.close()

    # Plot 4: Lane Segmentation Quality Metrics
    plt.figure(figsize=(9, 3.5))
    plt.plot(df['lane_confidence'] * 100.0, color='darkorange', lw=2, label='Lane-only Confidence')
    plt.plot(df['temporal_iou'] * 100.0, color='royalblue', lw=1.5, label='Temporal IoU')
    plt.plot(df['temporal_dice'] * 100.0, color='forestgreen', lw=1.5, label='Temporal Dice')
    plt.title('Temporal Lane Segmentation Stability', fontweight='bold')
    plt.xlabel('Simulation Frame Step Index'); plt.ylabel('Score (%)')
    plt.legend(loc='lower left'); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/perception_model_confidence.png", dpi=300); plt.close()

    # Plot 5: Localization Consistency Metrics
    def _plot_chi2_bounds(ax, values, df_dim, label):
        if chi2 is not None:
            upper = chi2.ppf(0.95, df_dim)
            lower = chi2.ppf(0.05, df_dim)
            ax.axhline(upper, color='gray', linestyle='--', alpha=0.7, label=f'95% χ² upper ({df_dim} dof)')
            ax.axhline(lower, color='gray', linestyle=':', alpha=0.7, label=f'95% χ² lower ({df_dim} dof)')
        ax.plot(values, label=label)
        ax.axhline(1.0, color='k', linestyle='--', alpha=0.6, label='Ideal Consistency')

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    _plot_chi2_bounds(ax, df['nees_rtk'], 3, 'RTK NEES')
    _plot_chi2_bounds(ax, df['nees_cart'], 3, 'Cartesian NEES')
    _plot_chi2_bounds(ax, df['nees_odo'], 4, 'Odometry NEES')
    plt.title('Localization Consistency Metrics (NEES)', fontweight='bold')
    plt.xlabel('Simulation Synchronized Steps'); plt.ylabel('NEES')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/localization_consistency_nees.png", dpi=300); plt.close()

    plt.figure(figsize=(9, 4))
    ax = plt.gca()
    _plot_chi2_bounds(ax, df['nis_rtk'], 3, 'RTK NIS')
    _plot_chi2_bounds(ax, df['nis_cart'], 3, 'Cartesian NIS')
    _plot_chi2_bounds(ax, df['nis_odo'], 1, 'Odometry NIS')
    plt.title('Localization Consistency Metrics (NIS)', fontweight='bold')
    plt.xlabel('Simulation Synchronized Steps'); plt.ylabel('NIS')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/localization_consistency_nis.png", dpi=300); plt.close()

    # Plot 6: Innovation residuals
    plt.figure(figsize=(9, 4))
    plt.plot(df['rtk_innov_norm'], 'g-', label='RTK Innovation')
    plt.plot(df['cart_innov_norm'], 'b-.', label='Cartesian Innovation')
    plt.plot(df['odo_innov_norm'], 'r:', label='Odometry Innovation')
    plt.title('Innovation Residuals', fontweight='bold')
    plt.xlabel('Simulation Synchronized Steps'); plt.ylabel('Residual Norm')
    plt.legend(loc='upper right'); plt.grid(True, alpha=0.3)
    plt.savefig(f"{output_dir}/innovation_residuals.png", dpi=300); plt.close()

    logger.info("System execution completed. Metrics saved to %s", output_dir)

if __name__ == '__main__':
    main()