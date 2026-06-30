# ==========================================================
# EXPERIMENT 4: VEHICLE ORIENTATION AND LANE ALIGNMENT EVALUATION
# Tracking Full Vehicle Pose [X, Y, Velocity, Yaw, Lane Deviations]
# ==========================================================

import carla
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# MASTER CONFIGURATION & DETERMINISTIC SEEDING
# ==========================================================
GLOBAL_SPATIAL_SEED = 42
SIM_DT = 0.05
TOTAL_STEPS = 10000  
CSV_NAME = "experiment4_alignment_evaluation.csv"

NUM_VEHICLES = 35
OPEN_RTK_STD = 0.03
BASE_ACCEL_STD = 0.05
BASE_GYRO_STD = np.radians(0.08)

# ==========================================================
# ERROR-STATE COUPLED KINEMATIC EKF
# State Vector x = [x, y, v, theta]^T
# ==========================================================
class VehiclePoseEKF:
    def __init__(self):
        self.x = np.zeros((4, 1))  
        self.P = np.eye(4) * 0.1
        self.Q = np.diag([0.002, 0.002, 0.02, np.radians(0.01)])
        self.H_pos = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
        self.R_open = np.eye(2) * (OPEN_RTK_STD**2)
        self.initialized = False

    def initialize(self, x, y, v, theta):
        self.x = np.array([[x], [y], [v], [theta]])
        self.initialized = True

    def predict(self, ax_body, gyro_z, dt):
        if not self.initialized: return
        x, y, v, theta = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]

        x_next = x + v * math.cos(theta) * dt + 0.5 * ax_body * math.cos(theta) * (dt**2)
        y_next = y + v * math.sin(theta) * dt + 0.5 * ax_body * math.sin(theta) * (dt**2)
        v_next = max(0.0, v + ax_body * dt)
        theta_next = math.atan2(math.sin(theta + gyro_z * dt), math.cos(theta + gyro_z * dt))

        self.x = np.array([[x_next], [y_next], [v_next], [theta_next]])

        F = np.eye(4)
        F[0, 2] = math.cos(theta) * dt
        F[0, 3] = -v * math.sin(theta) * dt
        F[1, 2] = math.sin(theta) * dt
        F[1, 3] = v * math.cos(theta) * dt
        self.P = F @ self.P @ F.T + self.Q

    def update_position(self, mx, my):
        if not self.initialized: return
        z = np.array([[mx], [my]])
        innovation = z - self.H_pos @ self.x
        
        S = self.H_pos @ self.P @ self.H_pos.T + self.R_open
        K = self.P @ self.H_pos.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ self.H_pos) @ self.P
        self.x[3, 0] = math.atan2(math.sin(self.x[3, 0]), math.cos(self.x[3, 0]))

# ==========================================================
# MATHEMATICAL GEOMETRY ALIGNMENT ENGINE
# ==========================================================
def compute_lane_relative_metrics(ego_location, ego_yaw_deg, waypoint):
    """Computes rigorous phase-wrapped heading error and signed lateral cross-track error."""
    # 1. FIXED: Strict Phase wrapping transformation to clear out +180/-180 discontinuities
    heading_error = math.radians(ego_yaw_deg) - math.radians(waypoint.transform.rotation.yaw)
    heading_error_wrapped = math.atan2(math.sin(heading_error), math.cos(heading_error))
    heading_error_deg = math.degrees(heading_error_wrapped)

    # 2. Signed Cross-Track Displacement
    wp_loc = waypoint.transform.location
    wp_forward = waypoint.transform.get_forward_vector()
    dx = ego_location.x - wp_loc.x
    dy = ego_location.y - wp_loc.y
    
    # Orthogonal road track scalar projection
    lateral_offset = dx * (-wp_forward.y) + dy * wp_forward.x
    return heading_error_deg, lateral_offset

# ==========================================================
# CARLA CORE RUNTIME ENGINE SETUP
# ==========================================================
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.load_world("Town03")
world.wait_for_tick()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = SIM_DT
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(GLOBAL_SPATIAL_SEED)

for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
world.tick()

# Route spawn assignment
primary_spawn = carla.Transform(carla.Location(x=250.0, y=-30.0, z=3.0), carla.Rotation(yaw=-100.0))
ego = world.try_spawn_actor(world.get_blueprint_library().filter("vehicle.tesla.model3")[0], primary_spawn)
ego.set_autopilot(True, traffic_manager.get_port())
traffic_manager.ignore_lights_percentage(ego, 100)
traffic_manager.set_route(ego, ["Straight", "Straight", "Straight"])

spawn_points = world.get_map().get_spawn_points()
random.seed(GLOBAL_SPATIAL_SEED)
for sp in spawn_points[:NUM_VEHICLES]:
    try:
        npc = world.try_spawn_actor(random.choice(world.get_blueprint_library().filter("vehicle.*")), sp)
        if npc: npc.set_autopilot(True, traffic_manager.get_port())
    except: pass

for _ in range(40): world.tick()

init_loc = ego.get_location()
init_vel = ego.get_velocity()
init_speed = math.sqrt(init_vel.x**2 + init_vel.y**2)
init_yaw = math.radians(ego.get_transform().rotation.yaw)

ekf = VehiclePoseEKF()
ekf.initialize(init_loc.x, init_loc.y, init_speed, init_yaw)

logs = []
rng = np.random.default_rng(GLOBAL_SPATIAL_SEED)

print("\nRunning Experiment 4 validation loop...")
try:
    for step in range(TOTAL_STEPS):
        world.tick()
        
        transform = ego.get_transform()
        gt_x, gt_y = transform.location.x, transform.location.y
        gt_yaw = transform.rotation.yaw
        gt_speed = math.sqrt(ego.get_velocity().x**2 + ego.get_velocity().y**2)

        waypoint = world.get_map().get_waypoint(transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        
        # Guard statement ensuring lane logic captures standard road context tracking profiles
        if waypoint.is_junction:
            continue

        true_heading_error, true_lateral_deviation = compute_lane_relative_metrics(transform.location, gt_yaw, waypoint)

        # Skip logging massive transients from macro lane change maneuvers
        if abs(true_lateral_deviation) > 1.8:
            continue

        true_accel = ego.get_acceleration()
        ax_body = math.sqrt(true_accel.x**2 + true_accel.y**2) * (1.0 if gt_speed > 0.5 else 0.0)
        ax_noisy = ax_body + rng.normal(0, BASE_ACCEL_STD)
        gyro_noisy = ego.get_angular_velocity().z + rng.normal(0, BASE_GYRO_STD)

        rtk_x = gt_x + rng.normal(0, OPEN_RTK_STD)
        rtk_y = gt_y + rng.normal(0, OPEN_RTK_STD)

        ekf.predict(ax_noisy, gyro_noisy, SIM_DT)
        ekf.update_position(rtk_x, rtk_y)

        est_x, est_y, est_yaw = ekf.x[0,0], ekf.x[1,0], ekf.x[3,0]
        pos_error = math.sqrt((est_x - gt_x)**2 + (est_y - gt_y)**2)
        yaw_error_deg = math.degrees(math.atan2(math.sin(est_yaw - math.radians(gt_yaw)), math.cos(est_yaw - math.radians(gt_yaw))))

        logs.append([
            step, gt_x, gt_y, gt_yaw, pos_error, yaw_error_deg, true_heading_error, true_lateral_deviation
        ])

finally:
    settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
    for actor in world.get_actors().filter('vehicle.*'):
        try: actor.destroy()
        except: pass

# ==========================================================
# POST-PROCESSING AND COMPILATION MATRIX
# ==========================================================
columns = ["step", "gt_x", "gt_y", "gt_yaw", "pos_error", "yaw_error", "heading_error_deg", "lateral_deviation"]
df = pd.DataFrame(logs, columns=columns)
df.to_csv(CSV_NAME, index=False)

# Compute complete numerical stats for your thesis document table
h_err_abs = abs(df["heading_error_deg"])
l_dev_abs = abs(df["lateral_deviation"])

stats = {
    "Metric": ["Heading Alignment Error (deg)", "Lateral Lane Deviation (m)"],
    "Mean": [np.mean(h_err_abs), np.mean(l_dev_abs)],
    "Max": [np.max(h_err_abs), np.max(l_dev_abs)],
    "P95": [np.percentile(h_err_abs, 95), np.percentile(l_dev_abs, 95)],
    "RMSE": [np.sqrt(np.mean(np.square(df["heading_error_deg"]))), np.sqrt(np.mean(np.square(df["lateral_deviation"])))]
}
df_stats = pd.DataFrame(stats)

print("\n==========================================================================")
print("     TABLE IV: VEHICLE ORIENTATION AND ROAD ALIGNMENT PERFORMANCE METRICS ")
print("==========================================================================")
print(df_stats.to_string(index=False))
print("==========================================================================")

# ----------------------------------------------------------
# THESIS VISUALIZATION CHART PIPELINE
# ----------------------------------------------------------
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# Plot 1: True Heading Error Over Time Horizon (Wrapped)
axs[0, 0].plot(df["step"] * SIM_DT, df["heading_error_deg"], color='#d62728', lw=1.2)
axs[0, 0].set_title("Lane-Relative Heading Error Vector Over Time", fontweight='bold')
axs[0, 0].set_xlabel("Time [Seconds]")
axs[0, 0].set_ylabel("Heading Error [Degrees]")
axs[0, 0].set_ylim([-8, 8])  # Zoomed into true error variance bounds
axs[0, 0].grid(True, linestyle=":", alpha=0.6)

# Plot 2: Unimodal Heading Error Histogram Profile
axs[0, 1].hist(df["heading_error_deg"], bins=50, color='#1f77b4', edgecolor='black', alpha=0.8, range=(-6, 6))
axs[0, 1].set_title("Statistical Error Density Distribution Histogram", fontweight='bold')
axs[0, 1].set_xlabel("Heading Registration Error [Degrees]")
axs[0, 1].set_ylabel("Sample Counts")
axs[0, 1].grid(True, linestyle=":", alpha=0.6)

# Plot 3: Cross-Track Error Deviation Profile
axs[1, 0].plot(df["step"] * SIM_DT, df["lateral_deviation"], color='#2ca02c', lw=1.2)
axs[1, 0].axhline(0, color='black', linestyle='--', alpha=0.5)
axs[1, 0].set_title("Lateral Displacement From Local Lane Centerline", fontweight='bold')
axs[1, 0].set_xlabel("Time [Seconds]")
axs[1, 0].set_ylabel("Displacement Offset [Meters]")
axs[1, 0].set_ylim([-0.5, 0.5])
axs[1, 0].grid(True, linestyle=":", alpha=0.6)

# Plot 4: Trajectory Mapping Resolved via Track Deviation Degradation Profiles
sc = axs[1, 1].scatter(df["gt_x"], df["gt_y"], c=l_dev_abs, cmap='viridis', s=3, alpha=0.9, vmin=0.0, vmax=0.25)
cbar = fig.colorbar(sc, ax=axs[1, 1])
cbar.set_label("Absolute Cross-Track Deviation Error [Meters]", fontweight='bold')
axs[1, 1].set_title("Spatial Vehicle Trajectory Tracking Map", fontweight='bold')
axs[1, 1].set_xlabel("Global Easting X [m]")
axs[1, 1].set_ylabel("Global Northing Y [m]")
axs[1, 1].grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
plt.savefig("experiment4_alignment_evaluation.png", dpi=300)
plt.close()
print("\n[Complete] Performance plots exported to disk cleanly.")