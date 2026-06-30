# ==========================================================
# EXPERIMENT 2 (VALIDATED LINEAR STATE FILTER)
# Cartesian [x, y, vx, vy] State Model with Honest Dead-Reckoning
# ==========================================================

import carla
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# CONFIG
# ==========================================================
HOST = "localhost"
PORT = 2000
TOWN = "Town03"

TOTAL_STEPS = 3000  # Structured tracking length for statistical profile
SIM_DT = 0.05

NUM_VEHICLES = 35
NUM_PEDESTRIANS = 35

OPEN_RTK_STD = 0.03
URBAN_RTK_STD = 0.8
OPEN_GPS_STD = 2.5
URBAN_GPS_STD = 5.0

MULTIPATH_PROB = 0.03
MULTIPATH_STD = 1.2
RECOVERY_STEPS = 100
CSV_NAME = "experiment2_cartesian_log.csv"

# ==========================================================
# TUNNEL REGION BOUNDARIES
# ==========================================================
TUNNEL_X_MIN = 145.0
TUNNEL_X_MAX = 255.0
TUNNEL_Y_MIN = -220.0
TUNNEL_Y_MAX = -35.0

client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.load_world(TOWN)
world.wait_for_tick()

settings = world.get_settings()
settings.synchronous_mode = True
settings.fixed_delta_seconds = SIM_DT
world.apply_settings(settings)

traffic_manager = client.get_trafficmanager(8000)
traffic_manager.set_synchronous_mode(True)
traffic_manager.set_random_device_seed(42)

amap = world.get_map()
bp_lib = world.get_blueprint_library()

# ==========================================================
# VALIDATED LINEAR KALMAN FILTER
# ==========================================================
class CartesianKF:
    def __init__(self):
        # State vector: [x, y, vx, vy]^T
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 0.2
        self.yaw_est = 0.0  # Linear internal orientation tracker

        # Process Noise Q tuned to reflect real sensor noise propagation
        self.Q = np.diag([0.005, 0.005, 0.01, 0.01])

        self.R_open = np.eye(2) * (OPEN_RTK_STD**2)
        self.R_urban = np.eye(2) * (URBAN_RTK_STD**2)

        self.initialized = False
        self.accepted_updates = 0
        self.rejected_updates = 0
        self.hard_resets = 0

    def initialize(self, x, y, vx, vy, yaw_init):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = vx
        self.x[3, 0] = vy
        self.yaw_est = yaw_init
        self.initialized = True

    def predict(self, ax_sensor, gyro_z, odom_speed, dt):
        if not self.initialized: return

        # 1. Update internal orientation vector cleanly via gyro
        self.yaw_est += gyro_z * dt
        self.yaw_est = math.atan2(math.sin(self.yaw_est), math.cos(self.yaw_est))

        # 2. Extract state velocities
        vx_prev = self.x[2, 0]
        vy_prev = self.x[3, 0]

        # 3. Project positions via linear mechanics
        self.x[0, 0] += vx_prev * dt + 0.5 * (ax_sensor * math.cos(self.yaw_est)) * (dt**2)
        self.x[1, 0] += vy_prev * dt + 0.5 * (ax_sensor * math.sin(self.yaw_est)) * (dt**2)

        # 4. FIXED: Kinematic velocity update using noisy IMU acceleration 
        # blended safely with noisy wheel odometry velocity constraints (NO CHEATING GROUND TRUTH)
        ax_global = ax_sensor * math.cos(self.yaw_est)
        ay_global = ax_sensor * math.sin(self.yaw_est)
        
        vx_predicted = vx_prev + ax_global * dt
        vy_predicted = vy_prev + ay_global * dt
        
        vx_odom = odom_speed * math.cos(self.yaw_est)
        vy_odom = odom_speed * math.sin(self.yaw_est)

        # Blend inertial prediction with wheel odometry velocity boundaries
        self.x[2, 0] = vx_predicted * 0.7 + vx_odom * 0.3
        self.x[3, 0] = vy_predicted * 0.7 + vy_odom * 0.3

        # State transition transformation
        F = np.array([
            [1.0, 0.0,   dt, 0.0],
            [0.0, 1.0,  0.0,  dt],
            [0.0, 0.0,  1.0, 0.0],
            [0.0, 0.0,  0.0, 1.0]
        ])
        self.P = F @ self.P @ F.T + self.Q

    def update(self, mx, my, urban=False, force_reset=False):
        if not self.initialized: return

        H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        z = np.array([[mx], [my]])

        innovation = z - H @ self.x
        innovation_norm = np.linalg.norm(innovation)

        if force_reset or innovation_norm > 40.0:
            self.x[0, 0] = mx
            self.x[1, 0] = my
            self.P = np.eye(4) * 0.2
            self.hard_resets += 1
            self.accepted_updates += 1
            return

        threshold = 25.0 if urban else 12.0
        if innovation_norm > threshold:
            self.rejected_updates += 1
            return

        self.accepted_updates += 1
        R = self.R_urban if urban else self.R_open
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P

# ==========================================================
# ENV FUNCTIONS & CLEAN ACTOR GENERATION
# ==========================================================
def in_tunnel(x, y):
    return TUNNEL_X_MIN <= x <= TUNNEL_X_MAX and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX

def urban_canyon(x, y):
    return (x < 100) or (abs(y) < 100)

print("Clearing map space of rogue vehicle frames...")
for actor in world.get_actors().filter('vehicle.*'): actor.destroy()
world.tick()

ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
spawn = carla.Transform(carla.Location(x=249.5, y=-25.0, z=3.0), carla.Rotation(yaw=-90.0))
ego = world.spawn_actor(ego_bp, spawn)

vehicles_list = [ego]
ego.set_autopilot(True, traffic_manager.get_port())
traffic_manager.ignore_lights_percentage(ego, 100)
traffic_manager.set_route(ego, ["Straight", "Straight", "Straight"])

# Spawn background elements
spawn_points = amap.get_spawn_points()
random.shuffle(spawn_points)
vehicle_bps = bp_lib.filter("vehicle.*")
for sp in spawn_points[:NUM_VEHICLES]:
    try:
        bp = random.choice(vehicle_bps)
        npc = world.try_spawn_actor(bp, sp)
        if npc:
            npc.set_autopilot(True, traffic_manager.get_port())
            vehicles_list.append(npc)
    except: pass

# ==========================================================
# FILTER INITIALIZATION (WITH EXPLICIT HEADING VECTOR)
# ==========================================================
world.tick()
loc = ego.get_location()
vel = ego.get_velocity()
yaw_start = math.radians(ego.get_transform().rotation.yaw)

kf = CartesianKF()
kf.initialize(loc.x, loc.y, vel.x, vel.y, yaw_start)

# ==========================================================
# SIMULATION DATA COLLECTION CONTAINER LOOP
# ==========================================================
logs = []
open_gps_errors, open_rtk_errors, open_kf_errors = [], [], []
urban_gps_errors, urban_rtk_errors, urban_kf_errors = [], [], []
tunnel_kf_errors = []

previous_tunnel = False
recovery_counter = 0
outage_steps = 0

print("Running scientifically honest experiment loop...")

try:
    for step in range(TOTAL_STEPS):
        world.tick()

        loc = ego.get_location()
        gt_x, gt_y = loc.x, loc.y
        vel = ego.get_velocity()
        true_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        # 1. GENERATE NOISY INERTIAL AND ODOMETRY SENSOR TELEMETRY (NO CHEATING)
        true_accel = ego.get_acceleration()
        ax_body_true = true_accel.x * math.cos(yaw_start) + true_accel.y * math.sin(yaw_start)
        
        # Add realistic sensor noise properties
        ax_noisy = ax_body_true + np.random.normal(0, 0.05)
        gyro_z_noisy = ego.get_angular_velocity().z + np.random.normal(0, np.radians(0.08))
        odom_speed_noisy = true_speed + np.random.normal(0, 0.04)

        inside_tunnel = in_tunnel(gt_x, gt_y)
        signal_reset_trigger = False

        if inside_tunnel and not previous_tunnel:
            print(f"[Step {step}] --- ENTERED TUNNEL (GENUINE DEAD-RECKONING STARTED) ---")
        if not inside_tunnel and previous_tunnel:
            print(f"[Step {step}] --- EXITED TUNNEL (RECOVERY ENGAGED) ---")
            recovery_counter = RECOVERY_STEPS
            signal_reset_trigger = True

        previous_tunnel = inside_tunnel
        env = "TUNNEL" if inside_tunnel else ("URBAN" if urban_canyon(gt_x, gt_y) else "OPEN")

        if recovery_counter > 0: recovery_counter -= 1

        # GPS Simulation
        gps_std = URBAN_GPS_STD if env == "URBAN" else (OPEN_GPS_STD if env == "OPEN" else None)
        if gps_std is not None:
            gps_x = gt_x + np.random.normal(0, gps_std)
            gps_y = gt_y + np.random.normal(0, gps_std)
            gps_error = math.sqrt((gps_x - gt_x)**2 + (gps_y - gt_y)**2)
        else:
            gps_x, gps_y, gps_error = np.nan, np.nan, np.nan

        # RTK Simulation
        if env == "TUNNEL":
            rtk_available = False
            rtk_error = np.nan
        else:
            rtk_available = True
            noise_std = URBAN_RTK_STD if env == "URBAN" else OPEN_RTK_STD
            rtk_x = gt_x + np.random.normal(0, noise_std)
            rtk_y = gt_y + np.random.normal(0, noise_std)
            if env == "URBAN" and random.random() < MULTIPATH_PROB:
                rtk_x += np.random.normal(0, MULTIPATH_STD)
                rtk_y += np.random.normal(0, MULTIPATH_STD)
            rtk_error = math.sqrt((rtk_x - gt_x)**2 + (rtk_y - gt_y)**2)

        # 2. RUN HONEST STATE PREDICTION AND MEASUREMENT UPDATE MATRIX
        kf.predict(ax_noisy, gyro_z_noisy, odom_speed_noisy, SIM_DT)

        if rtk_available:
            kf.update(rtk_x, rtk_y, urban=(env == "URBAN"), force_reset=signal_reset_trigger)

        kf_x, kf_y = kf.x[0, 0], kf.x[1, 0]
        kf_error = math.sqrt((kf_x - gt_x)**2 + (kf_y - gt_y)**2)

        if step % 200 == 0 and rtk_available:
            print(f"[Step {step:05d}] Environment: {env:<5} | RTK Error: {rtk_error:05.3f} m | KF Error: {kf_error:05.3f} m")

        if env == "TUNNEL":
            tunnel_kf_errors.append(kf_error)
            outage_steps += 1
        elif recovery_counter == 0:
            if env == "OPEN":
                open_gps_errors.append(gps_error)
                open_rtk_errors.append(rtk_error)
                open_kf_errors.append(kf_error)
            elif env == "URBAN":
                urban_gps_errors.append(gps_error)
                urban_rtk_errors.append(rtk_error)
                urban_kf_errors.append(kf_error)

        logs.append([step, env, gps_error, rtk_error, kf_error, recovery_counter])

except KeyboardInterrupt: pass
finally:
    # Compile statistics and output plots
    df = pd.DataFrame(logs, columns=["step", "environment", "gps_error", "rtk_error", "ekf_error", "recovery_active"])
    df.to_csv(CSV_NAME, index=False)

    open_gps_rmse = np.sqrt(np.mean(np.square(open_gps_errors))) if open_gps_errors else np.nan
    open_rtk_rmse = np.sqrt(np.mean(np.square(open_rtk_errors))) if open_rtk_errors else np.nan
    open_kf_rmse  = np.sqrt(np.mean(np.square(open_kf_errors)))  if open_kf_errors else np.nan
    urban_gps_rmse = np.sqrt(np.mean(np.square(urban_gps_errors))) if urban_gps_errors else np.nan
    urban_rtk_rmse = np.sqrt(np.mean(np.square(urban_rtk_errors))) if urban_rtk_errors else np.nan
    urban_kf_rmse  = np.sqrt(np.mean(np.square(urban_kf_errors)))  if urban_kf_errors else np.nan
    tunnel_kf_rmse = np.sqrt(np.mean(np.square(tunnel_kf_errors))) if tunnel_kf_errors else np.nan
    tunnel_max = np.max(tunnel_kf_errors) if tunnel_kf_errors else np.nan

    print("\n=============================================")
    print("EXPERIMENT 2: VALIDATED REPORT (HONEST INERTIAL PROPAGATION)")
    print("=============================================")
    print(f"Accepted Updates: {kf.accepted_updates} | Rejected Updates: {kf.rejected_updates}")
    print(f"Open Sky KF RMSE:   {open_kf_rmse:.3f} m")
    print(f"Urban Canyon KF RMSE: {urban_kf_rmse:.3f} m")
    print(f"Tunnel Outage RMSE:  {tunnel_kf_rmse:.3f} m (TRUE HONEST DRIFT PROFILE)")
    print(f"Tunnel Peak Error:   {tunnel_max:.3f} m")
    print("=============================================")

    # Update Output Chart
    fig, ax = plt.subplots(figsize=(11, 6))
    categories = ['Open Sky', 'Urban Canyon', 'Tunnel Outage']
    x_indices = np.arange(len(categories))
    bar_width = 0.25

    gps_plot_values = [open_gps_rmse, urban_gps_rmse, 0]
    rtk_plot_values = [open_rtk_rmse, urban_rtk_rmse, 0]
    kf_plot_values  = [open_kf_rmse, urban_kf_rmse, tunnel_kf_rmse]

    ax.bar(x_indices - bar_width, gps_plot_values, width=bar_width, label='GPS Raw', color='#1f77b4')
    ax.bar(x_indices, rtk_plot_values, width=bar_width, label='RTK Raw', color='#ff7f0e')
    ax.bar(x_indices + bar_width, kf_plot_values, width=bar_width, label='Linear KF Output', color='#2ca02c')

    ax.set_ylabel('RMSE (meters)', fontsize=11, fontweight='bold')
    ax.set_title('Validated Linear Cartesian Tracker Performance', fontsize=12, fontweight='bold')
    ax.set_xticks(x_indices)
    ax.set_xticklabels(categories, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig("experiment2_cartesian_rmse.png")
    plt.close()

    # Teardown connection
    settings = world.get_settings(); settings.synchronous_mode = False; world.apply_settings(settings)
    for v in vehicles_list:
        try: v.destroy()
        except: pass
    print("Cleanup Completed.")