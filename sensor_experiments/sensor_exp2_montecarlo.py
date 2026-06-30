# ==========================================================
# EXPERIMENT 2 (MONTE CARLO VALIDATION VERSION)
# Cartesian Linear KF: Multi-Trial Statistical Ensemble
# ==========================================================

import carla
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ==========================================================
# GLOBAL CONFIG & CONFIGURATION SEEDING
# ==========================================================
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

HOST = "localhost"
PORT = 2000
TOWN = "Town03"

TOTAL_STEPS = 3000  
SIM_DT = 0.05
NUM_TRIALS = 10     # Structured Monte Carlo repetitions

NUM_VEHICLES = 35
NUM_PEDESTRIANS = 35

OPEN_RTK_STD = 0.03
URBAN_RTK_STD = 0.8
OPEN_GPS_STD = 2.5
URBAN_GPS_STD = 5.0

MULTIPATH_PROB = 0.03
MULTIPATH_STD = 1.2
RECOVERY_STEPS = 100
CSV_NAME = "experiment2_montecarlo_results.csv"

# ==========================================================
# TUNNEL REGION BOUNDARIES
# ==========================================================
TUNNEL_X_MIN = 145.0
TUNNEL_X_MAX = 255.0
TUNNEL_Y_MIN = -220.0
TUNNEL_Y_MAX = -35.0

# ==========================================================
# VALIDATED LINEAR KALMAN FILTER
# ==========================================================
class CartesianKF:
    def __init__(self):
        self.x = np.zeros((4, 1))
        self.P = np.eye(4) * 0.2
        self.yaw_est = 0.0  
        self.Q = np.diag([0.005, 0.005, 0.01, 0.01])
        self.R_open = np.eye(2) * (OPEN_RTK_STD**2)
        self.R_urban = np.eye(2) * (URBAN_RTK_STD**2)
        self.initialized = False

    def initialize(self, x, y, vx, vy, yaw_init):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = vx
        self.x[3, 0] = vy
        self.yaw_est = yaw_init
        self.initialized = True

    def predict(self, ax_sensor, gyro_z, odom_speed, dt):
        if not self.initialized: return

        self.yaw_est += gyro_z * dt
        self.yaw_est = math.atan2(math.sin(self.yaw_est), math.cos(self.yaw_est))

        vx_prev = self.x[2, 0]
        vy_prev = self.x[3, 0]

        self.x[0, 0] += vx_prev * dt + 0.5 * (ax_sensor * math.cos(self.yaw_est)) * (dt**2)
        self.x[1, 0] += vy_prev * dt + 0.5 * (ax_sensor * math.sin(self.yaw_est)) * (dt**2)

        ax_global = ax_sensor * math.cos(self.yaw_est)
        ay_global = ax_sensor * math.sin(self.yaw_est)
        
        vx_predicted = vx_prev + ax_global * dt
        vy_predicted = vy_prev + ay_global * dt
        
        vx_odom = odom_speed * math.cos(self.yaw_est)
        vy_odom = odom_speed * math.sin(self.yaw_est)

        self.x[2, 0] = vx_predicted * 0.7 + vx_odom * 0.3
        self.x[3, 0] = vy_predicted * 0.7 + vy_odom * 0.3

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
            return

        threshold = 25.0 if urban else 12.0
        if innovation_norm > threshold:
            return

        R = self.R_urban if urban else self.R_open
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(4) - K @ H) @ self.P

# ==========================================================
# ENVIRONMENT CORRIDORS
# ==========================================================
def in_tunnel(x, y):
    return TUNNEL_X_MIN <= x <= TUNNEL_X_MAX and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX

def urban_canyon(x, y):
    return (x < 100) or (abs(y) < 100)

# ==========================================================
# INITIALIZE CONNECTION ENGINE
# ==========================================================
client = carla.Client(HOST, PORT)
client.set_timeout(30.0)
world = client.load_world(TOWN)
world.wait_for_tick()

# Core tracking repositories 
ensemble_results = {
    "OPEN": {"GPS": [], "RTK": [], "KF": []},
    "URBAN": {"GPS": [], "RTK": [], "KF": []},
    "TUNNEL": {"KF": [], "PEAK": []}
}

spawn_transform = carla.Transform(carla.Location(x=249.5, y=-25.0, z=3.0), carla.Rotation(yaw=-90.0))

print(f"\n=======================================================")
print(f"LAUNCHING EXPERIMENT 2 MONTE CARLO BATCH ({NUM_TRIALS} TRIALS)")
print(f"=======================================================")

for trial in range(1, NUM_TRIALS + 1):
    print(f"Executing Trial Run [{trial:02d}/{NUM_TRIALS:02d}]...")
    
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DT
    world.apply_settings(settings)

    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(RANDOM_SEED + trial)

    amap = world.get_map()
    bp_lib = world.get_blueprint_library()

    # Flush environment actors
    for actor in world.get_actors().filter('vehicle.*'):
        try: actor.destroy()
        except: pass

    # Spawn primary vehicle state
    ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(ego_bp, spawn_transform)
    if ego is None:
        print(f"  [Error] Obstruction encountered during trial initialization. Skipping trial context.")
        continue

    vehicles_list = [ego]
    ego.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(ego, 100)
    traffic_manager.set_route(ego, ["Straight", "Straight", "Straight"])

    # Clutter dynamic NPCs
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

    world.tick()
    loc_start = ego.get_location()
    vel_start = ego.get_velocity()
    yaw_start = math.radians(ego.get_transform().rotation.yaw)

    kf = CartesianKF()
    kf.initialize(loc_start.x, loc_start.y, vel_start.x, vel_start.y, yaw_start)

    # Local temporary accumulation storage arrays
    run_metrics = {
        "OPEN": {"GPS": [], "RTK": [], "KF": []},
        "URBAN": {"GPS": [], "RTK": [], "KF": []},
        "TUNNEL": {"KF": []}
    }

    previous_tunnel = False
    recovery_counter = 0

    # Iterative trial step sequence
    for step in range(TOTAL_STEPS):
        world.tick()

        loc = ego.get_location()
        gt_x, gt_y = loc.x, loc.y
        vel = ego.get_velocity()
        true_speed = math.sqrt(vel.x**2 + vel.y**2 + vel.z**2)

        true_accel = ego.get_acceleration()
        ax_body_true = true_accel.x * math.cos(yaw_start) + true_accel.y * math.sin(yaw_start)
        
        ax_noisy = ax_body_true + np.random.normal(0, 0.05)
        gyro_z_noisy = ego.get_angular_velocity().z + np.random.normal(0, np.radians(0.08))
        odom_speed_noisy = true_speed + np.random.normal(0, 0.04)

        inside_tunnel = in_tunnel(gt_x, gt_y)
        signal_reset_trigger = False

        if not inside_tunnel and previous_tunnel:
            recovery_counter = RECOVERY_STEPS
            signal_reset_trigger = True

        previous_tunnel = inside_tunnel
        env = "TUNNEL" if inside_tunnel else ("URBAN" if urban_canyon(gt_x, gt_y) else "OPEN")

        if recovery_counter > 0: recovery_counter -= 1

        # Simulate Raw GPS Noise profiles
        gps_std = URBAN_GPS_STD if env == "URBAN" else (OPEN_GPS_STD if env == "OPEN" else None)
        if gps_std is not None:
            gps_x = gt_x + np.random.normal(0, gps_std)
            gps_y = gt_y + np.random.normal(0, gps_std)
            gps_error = math.sqrt((gps_x - gt_x)**2 + (gps_y - gt_y)**2)
        else:
            gps_error = np.nan

        # Simulate Raw RTK Noise profiles
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

        # Execute propagation logic steps
        kf.predict(ax_noisy, gyro_z_noisy, odom_speed_noisy, SIM_DT)
        if rtk_available:
            kf.update(rtk_x, rtk_y, urban=(env == "URBAN"), force_reset=signal_reset_trigger)

        kf_x, kf_y = kf.x[0, 0], kf.x[1, 0]
        kf_error = math.sqrt((kf_x - gt_x)**2 + (kf_y - gt_y)**2)

        # Segment logs based on environment status parameters
        if env == "TUNNEL":
            run_metrics["TUNNEL"]["KF"].append(kf_error)
        elif recovery_counter == 0:
            if env == "OPEN":
                run_metrics["OPEN"]["GPS"].append(gps_error)
                run_metrics["OPEN"]["RTK"].append(rtk_error)
                run_metrics["OPEN"]["KF"].append(kf_error)
            elif env == "URBAN":
                run_metrics["URBAN"]["GPS"].append(gps_error)
                run_metrics["URBAN"]["RTK"].append(rtk_error)
                run_metrics["URBAN"]["KF"].append(kf_error)

    # Process trial metrics context into global statistics buckets
    for region in ["OPEN", "URBAN"]:
        for metric in ["GPS", "RTK", "KF"]:
            if run_metrics[region][metric]:
                ensemble_results[region][metric].append(np.mean(run_metrics[region][metric]))

    if run_metrics["TUNNEL"]["KF"]:
        ensemble_results["TUNNEL"]["KF"].append(np.mean(run_metrics["TUNNEL"]["KF"]))
        ensemble_results["TUNNEL"]["PEAK"].append(np.max(run_metrics["TUNNEL"]["KF"]))

    # Clean execution elements
    for vehicle in vehicles_list:
        try: vehicle.destroy()
        except: pass
    world.tick()

# ==========================================================
# MONTE CARLO ANALYSIS METRIC COMPILATION
# ==========================================================
summary_rows = [
    {
        "Environment": "Open Sky",
        "GPS Raw Mean Error (m)": f"{np.mean(ensemble_results['OPEN']['GPS']):.3f} ± {np.std(ensemble_results['OPEN']['GPS']):.3f}",
        "RTK Raw Mean Error (m)": f"{np.mean(ensemble_results['OPEN']['RTK']):.3f} ± {np.std(ensemble_results['OPEN']['RTK']):.3f}",
        "Linear KF Mean Error (m)": f"{np.mean(ensemble_results['OPEN']['KF']):.3f} ± {np.std(ensemble_results['OPEN']['KF']):.3f}",
        "Max Peak Error (m)": "N/A"
    },
    {
        "Environment": "Urban Canyon",
        "GPS Raw Mean Error (m)": f"{np.mean(ensemble_results['URBAN']['GPS']):.3f} ± {np.std(ensemble_results['URBAN']['GPS']):.3f}",
        "RTK Raw Mean Error (m)": f"{np.mean(ensemble_results['URBAN']['RTK']):.3f} ± {np.std(ensemble_results['URBAN']['RTK']):.3f}",
        "Linear KF Mean Error (m)": f"{np.mean(ensemble_results['URBAN']['KF']):.3f} ± {np.std(ensemble_results['URBAN']['KF']):.3f}",
        "Max Peak Error (m)": "N/A"
    },
    {
        "Environment": "Tunnel Outage",
        "GPS Raw Mean Error (m)": "N/A",
        "RTK Raw Mean Error (m)": "N/A",
        "Linear KF Mean Error (m)": f"{np.mean(ensemble_results['TUNNEL']['KF']):.3f} ± {np.std(ensemble_results['TUNNEL']['KF']):.3f}",
        "Max Peak Error (m)": f"{np.mean(ensemble_results['TUNNEL']['PEAK']):.3f} ± {np.std(ensemble_results['TUNNEL']['PEAK']):.3f}"
    }
]

df_report = pd.DataFrame(summary_rows)
df_report.to_csv(CSV_NAME, index=False)

print("\n==============================================================================================================")
print("                           EXPERIMENT 2 MONTE CARLO STATISTICAL SUMMARY MATRIX                                ")
print("==============================================================================================================")
print(df_report.to_string(index=False))
print("==============================================================================================================")

# Generate final plot metric distribution summary
plt.figure(figsize=(9, 5))
categories = ['Open Sky KF', 'Urban Canyon KF', 'Tunnel Outage KF']
means = [np.mean(ensemble_results['OPEN']['KF']), np.mean(ensemble_results['URBAN']['KF']), np.mean(ensemble_results['TUNNEL']['KF'])]
stds = [np.std(ensemble_results['OPEN']['KF']), np.std(ensemble_results['URBAN']['KF']), np.std(ensemble_results['TUNNEL']['KF'])]

plt.bar(categories, means, yerr=stds, color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', capsize=8, alpha=0.85)
plt.ylabel("Ensemble Root Mean Square Error [RMSE] (Meters)", fontweight='bold')
plt.title("Experiment 2: Cartesian Linear KF Tracking Performance\n(10 Independent Monte Carlo Dynamic Trials)", fontweight='bold')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("experiment2_montecarlo_rmse.png")
plt.close()

# Reset context settings safely
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print("\n[Complete] Experiment 2 Monte Carlo complete. Matrix profile sheets saved to disk.")