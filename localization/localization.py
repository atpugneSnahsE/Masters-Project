import numpy as np
from typing import Optional


class WrapAngle:
    @staticmethod
    def wrap(rad: float) -> float:
        return (rad + np.pi) % (2 * np.pi) - np.pi


class KinematicEKFs:
    def __init__(self, dt: float = 0.05) -> None:
        self.dt = dt
        self.x_rtk = np.zeros((6, 1))
        self.P_rtk = np.eye(6) * 0.01
        self.Q_rtk = np.diag([0.01, 0.01, 0.01, 0.05, 0.05, 0.001])
        self.R_rtk = np.diag([0.01, 0.01, 0.02])

        self.x_cart = np.zeros((4, 1))
        self.P_cart = np.eye(4) * 0.1
        self.Q_cart = np.diag([0.05, 0.05, 0.1, 0.02])
        self.R_cart = np.diag([0.2, 0.2, 0.05])

        self.x_odo = np.zeros((6, 1))
        self.P_odo = np.eye(6) * 0.1
        self.Q_odo = np.diag([0.02, 0.02, 0.01, 0.05, 0.05, 0.002])
        self.R_odo = np.diag([0.05])

    def init_states(self, init_pos: Optional[object], init_yaw: float) -> None:
        x, y, z = init_pos.x, init_pos.y, init_pos.z
        yaw_rad = np.radians(init_yaw)
        self.x_rtk = np.array([[x], [y], [z], [0.0], [0.0], [yaw_rad]])
        self.x_cart = np.array([[x], [y], [0.0], [yaw_rad]])
        self.x_odo = np.array([[x], [y], [z], [0.0], [0.0], [yaw_rad]])

    def predict(self, imu_accel: object, imu_gyro_z: float, speedometer_v: float) -> None:
        dt = self.dt
        max_allowable_rotation_step = 0.25
        cleaned_gyro_z = np.clip(imu_gyro_z, -max_allowable_rotation_step, max_allowable_rotation_step)

        psi_a = self.x_rtk[5, 0]
        accel_x_g = imu_accel.x * np.cos(psi_a) - imu_accel.y * np.sin(psi_a)
        accel_y_g = imu_accel.x * np.sin(psi_a) + imu_accel.y * np.cos(psi_a)

        f_rtk = np.array([
            [self.x_rtk[0, 0] + self.x_rtk[3, 0] * dt + 0.5 * accel_x_g * dt**2],
            [self.x_rtk[1, 0] + self.x_rtk[4, 0] * dt + 0.5 * accel_y_g * dt**2],
            [self.x_rtk[2, 0]],
            [self.x_rtk[3, 0] + accel_x_g * dt],
            [self.x_rtk[4, 0] + accel_y_g * dt],
            [self.x_rtk[5, 0] + cleaned_gyro_z * dt],
        ])
        F_rtk = np.eye(6)
        F_rtk[0, 3], F_rtk[1, 4] = dt, dt
        self.x_rtk = f_rtk
        self.P_rtk = F_rtk @ self.P_rtk @ F_rtk.T + self.Q_rtk

        v_b = self.x_cart[2, 0]
        psi_b = self.x_cart[3, 0]
        f_cart = np.array([
            [self.x_cart[0, 0] + v_b * np.cos(psi_b) * dt],
            [self.x_cart[1, 0] + v_b * np.sin(psi_b) * dt],
            [v_b],
            [psi_b + cleaned_gyro_z * dt],
        ])
        F_cart = np.eye(4)
        F_cart[0, 2] = np.cos(psi_b) * dt
        F_cart[0, 3] = -v_b * np.sin(psi_b) * dt
        F_cart[1, 2] = np.sin(psi_b) * dt
        F_cart[1, 3] = v_b * np.cos(psi_b) * dt
        self.x_cart = f_cart
        self.P_cart = F_cart @ self.P_cart @ F_cart.T + self.Q_cart

        psi_c = self.x_odo[5, 0]
        self.x_odo[0, 0] += speedometer_v * np.cos(psi_c) * dt
        self.x_odo[1, 0] += speedometer_v * np.sin(psi_c) * dt
        self.x_odo[3, 0] = speedometer_v * np.cos(psi_c)
        self.x_odo[4, 0] = speedometer_v * np.sin(psi_c)
        self.x_odo[5, 0] += cleaned_gyro_z * dt
        F_odo = np.eye(6)
        F_odo[0, 3], F_odo[1, 4] = dt, dt
        self.P_odo = F_odo @ self.P_odo @ F_odo.T + self.Q_odo

    def update_rtk(self, meas_xyz: np.ndarray) -> None:
        H = np.zeros((3, 6))
        H[0, 0], H[1, 1], H[2, 2] = 1.0, 1.0, 1.0
        y = meas_xyz - (H @ self.x_rtk)
        S = H @ self.P_rtk @ H.T + self.R_rtk
        K = self.P_rtk @ H.T @ np.linalg.inv(S)
        self.x_rtk = self.x_rtk + K @ y
        self.P_rtk = (np.eye(6) - K @ H) @ self.P_rtk

    def update_cartesian(self, meas_xy: np.ndarray, current_speed: float) -> None:
        H = np.zeros((2, 4))
        H[0, 0], H[1, 1] = 1.0, 1.0
        z = meas_xy
        R = self.R_cart[:2, :2]

        if current_speed > 2.0:
            dx = meas_xy[0, 0] - self.x_cart[0, 0]
            dy = meas_xy[1, 0] - self.x_cart[1, 0]
            derived_yaw = np.arctan2(dy, dx)
            yaw_residual = WrapAngle.wrap(derived_yaw - self.x_cart[3, 0])
            if np.abs(yaw_residual) < np.radians(45.0):
                H = np.zeros((3, 4))
                H[0, 0], H[1, 1], H[2, 3] = 1.0, 1.0, 1.0
                z = np.vstack([meas_xy, [[derived_yaw]]])
                R = self.R_cart

        y = z - (H @ self.x_cart)
        if z.shape[0] == 3:
            y[2, 0] = WrapAngle.wrap(y[2, 0])

        S = H @ self.P_cart @ H.T + R
        K = self.P_cart @ H.T @ np.linalg.inv(S)
        self.x_cart = self.x_cart + K @ y
        self.P_cart = (np.eye(4) - K @ H) @ self.P_cart

    def update_odometry(self, meas_v: float) -> None:
        H = np.zeros((1, 6))
        psi = self.x_odo[5, 0]
        H[0, 3], H[0, 4] = np.cos(psi), np.sin(psi)
        y = np.array([[meas_v]]) - (H @ self.x_odo)
        S = H @ self.P_odo @ H.T + self.R_odo
        K = self.P_odo @ H.T @ np.linalg.inv(S)
        self.x_odo = self.x_odo + K @ y
        self.P_odo = (np.eye(6) - K @ H) @ self.P_odo
