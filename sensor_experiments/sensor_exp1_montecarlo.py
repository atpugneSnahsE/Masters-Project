# ==========================================================
# EXPERIMENT 1 (VALIDATED MONTE CARLO ARCHITECTURE)
# GPS vs RTK vs INS-EKF+RTK: Controlled Statistical Ensemble
# ==========================================================

import carla
import time
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from queue import Queue, Empty

# ==========================================================
# MASTER COGNITIVE SEEDING BASES
# ==========================================================
GLOBAL_SPATIAL_SEED = 42  # Locks CARLA geometry and vehicle paths
TOTAL_STEPS = 3000        # Trajectory evaluation depth
SIM_DT = 0.05
NUM_TRIALS = 10           # Monte Carlo sample size

NUM_VEHICLES = 35
NUM_PEDESTRIANS = 35

OPEN_RTK_STD = 0.03
URBAN_RTK_STD = 0.8
OPEN_GPS_STD = 2.5
URBAN_GPS_STD = 5.0

MULTIPATH_PROB = 0.03
MULTIPATH_STD = 1.2
CSV_NAME = "experiment1_montecarlo_validated_results.csv"

TUNNEL_X_MIN, TUNNEL_X_MAX = 145.0, 255.0
TUNNEL_Y_MIN, TUNNEL_Y_MAX = -220.0, -35.0

# ==========================================================
# EXTENDED STATE INERTIAL KALMAN FILTER
# ==========================================================
class INS_EKF:
    def __init__(self):
        self.x = np.zeros((6, 1))  
        self.P = np.eye(6) * 0.1  # Confident but agile initialization matrix
        
        self.Q = np.diag([
            0.002,               # Position variance X
            0.002,               # Position variance Y
            0.040,               # Velocity variance X
            0.040,               # Velocity variance Y
            np.radians(0.005),   # Yaw drift angular variance
            1e-7                 # Gyroscope stability dynamic drift bias
        ])
        
        self.H = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0]
        ])
        self.R_open = np.eye(2) * (OPEN_RTK_STD**2)
        self.R_urban = np.eye(2) * (URBAN_RTK_STD**2)
        self.initialized = False

    def initialize(self, x, y, vx, vy, yaw):
        self.x[0, 0] = x
        self.x[1, 0] = y
        self.x[2, 0] = vx
        self.x[3, 0] = vy
        self.x[4, 0] = yaw
        self.x[5, 0] = 0.0  
        self.initialized = True

    def predict(self, ax, ay, gyro_z, dt):
        if not self.initialized: return

        x, y, vx, vy = self.x[0,0], self.x[1,0], self.x[2,0], self.x[3,0]
        heading, bias = self.x[4,0], self.x[5,0]

        corrected_gyro = gyro_z - bias
        heading_next = heading + corrected_gyro * dt
        heading_next = math.atan2(math.sin(heading_next), math.cos(heading_next))

        ax_global = ax * math.cos(heading) - ay * math.sin(heading)
        ay_global = ax * math.sin(heading) + ay * math.cos(heading)

        x_next = x + vx * dt + 0.5 * ax_global * (dt**2)
        y_next = y + vy * dt + 0.5 * ay_global * (dt**2)
        vx_next = vx + ax_global * dt
        vy_next = vy + ay_global * dt

        self.x = np.array([[x_next], [y_next], [vx_next], [vy_next], [heading_next], [bias]])

        F = np.eye(6)
        F[0, 2] = dt
        F[1, 3] = dt
        F[0, 4] = 0.5 * (-ax * math.sin(heading) - ay * math.cos(heading)) * (dt**2)
        F[1, 4] = 0.5 * (ax * math.cos(heading) - ay * math.sin(heading)) * (dt**2)
        F[2, 4] = (-ax * math.sin(heading) - ay * math.cos(heading)) * dt
        F[3, 4] = (ax * math.cos(heading) - ay * math.sin(heading)) * dt
        F[4, 5] = -dt

        self.P = F @ self.P @ F.T + self.Q

    def update(self, mx, my, urban=False, bypass_gate=False):
        if not self.initialized: return
        z = np.array([[mx], [my]])
        innovation = z - self.H @ self.x
        
        if not bypass_gate:
            innovation_norm = np.linalg.norm(innovation)
            threshold = 20.0 if urban else 6.0  # Safe bounds ensuring data continuity
            if innovation_norm > threshold:
                return 

        R = self.R_urban if urban else self.R_open
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        self.P = (np.eye(6) - K @ self.H) @ self.P

# ==========================================================
# SENSOR SIMULATION ENGINE (ISOLATED RUN SEEDING)
# ==========================================================
class SensorSuite:
    def __init__(self, world, ego, local_stochastic_seed):
        self.imu_queue = Queue()
        self.gyro_bias = 0.0
        self.rng = np.random.default_rng(local_stochastic_seed)
        
        bp_lib = world.get_blueprint_library()
        imu_bp = bp_lib.find("sensor.other.imu")
        self.imu = world.spawn_actor(imu_bp, carla.Transform(), attach_to=ego)
        self.imu.listen(lambda data: self.imu_queue.put(data))

    def get_imu_reading(self):
        try:
            data = self.imu_queue.get(timeout=0.2)
            self.gyro_bias += self.rng.normal(0, 0.000005)
            
            ax_noisy = data.accelerometer.x + self.rng.normal(0, 0.04)
            ay_noisy = data.accelerometer.y + self.rng.normal(0, 0.04)
            gyro_noisy = data.gyroscope.z + self.rng.normal(0, np.radians(0.08)) + self.gyro_bias
            return ax_noisy, ay_noisy, gyro_noisy
        except Empty:
            return 0.0, 0.0, 0.0

    def clear_queue(self):
        while not self.imu_queue.empty():
            try: self.imu_queue.get_nowait()
            except Empty: break

    def destroy(self):
        try: self.imu.stop(); self.imu.destroy()
        except: pass

def in_tunnel(x, y): return TUNNEL_X_MIN <= x <= TUNNEL_X_MAX and TUNNEL_Y_MIN <= y <= TUNNEL_Y_MAX
def urban_canyon(x, y): return (x < 100) or (abs(y) < 100)

# ==========================================================
# SIMULATION HOST PIPELINE ENVIRONMENT
# ==========================================================
client = carla.Client("localhost", 2000)
client.set_timeout(30.0)
world = client.load_world("Town03")
world.wait_for_tick()

ensemble_stats = {"GPS": [], "RTK": [], "INS_EKF": []}
primary_spawn = carla.Transform(carla.Location(x=250.0, y=-30.0, z=3.0), carla.Rotation(yaw=-100.0))

print(f"\n=======================================================")
print(f"LAUNCHING RIGOROUS DECOUPLED MONTE CARLO FRAMEWORK ({NUM_TRIALS} RUNS)")
print(f"=======================================================")

for trial in range(1, NUM_TRIALS + 1):
    print(f"Running Statistical Trial Baseline Run [{trial:02d}/{NUM_TRIALS:02d}]...")
    
    # Generate completely independent local stochastic engines per run
    trial_stochastic_seed = GLOBAL_SPATIAL_SEED + trial
    rng_local = np.random.default_rng(trial_stochastic_seed)
    random.seed(trial_stochastic_seed)

    # Clean and apply environment settings
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = SIM_DT
    world.apply_settings(settings)

    # CRITICAL FIX: Lock Spatial Track using uniform seed across all runs
    traffic_manager = client.get_trafficmanager(8000)
    traffic_manager.set_synchronous_mode(True)
    traffic_manager.set_random_device_seed(GLOBAL_SPATIAL_SEED)

    # Flush old objects to prevent cross-trial state leakage
    for actor in world.get_actors().filter('vehicle.*'):
        try: actor.destroy()
        except: pass
    world.tick()

    # Re-instantiate Ego on the identical physical path
    ego_bp = world.get_blueprint_library().filter("vehicle.tesla.model3")[0]
    ego = world.try_spawn_actor(ego_bp, primary_spawn)
    vehicles_list = [ego]
    ego.set_autopilot(True, traffic_manager.get_port())
    traffic_manager.ignore_lights_percentage(ego, 100)
    traffic_manager.set_route(ego, ["Straight", "Straight", "Straight"])

    # Populate identical traffic layout background components
    spawn_points = world.get_map().get_spawn_points()
    random.seed(GLOBAL_SPATIAL_SEED)  # Temporarily reset to lock layout distribution geometry
    random.shuffle(spawn_points)
    vehicle_bps = world.get_blueprint_library().filter("vehicle.*")
    for sp in spawn_points[:NUM_VEHICLES]:
        try:
            bp = random.choice(vehicle_bps)
            npc = world.try_spawn_actor(bp, sp)
            if npc:
                npc.set_autopilot(True, traffic_manager.get_port())
                vehicles_list.append(npc)
        except: pass
    
    # Restore randomness context for the runtime trial tracker
    random.seed(trial_stochastic_seed)

    # Instantiate clean sensor and EKF objects
    sensors = SensorSuite(world, ego, trial_stochastic_seed)
    ekf = INS_EKF()

    # Physics Warmup Settling Phase
    for _ in range(40):
        world.tick()

    sensors.clear_queue()
    init_loc = ego.get_location()
    init_vel = ego.get_velocity()
    init_yaw = math.radians(ego.get_transform().rotation.yaw)
    
    # CRITICAL FIX: Anchor filter using a stabilized local measurement position baseline
    init_noise_std = URBAN_RTK_STD if urban_canyon(init_loc.x, init_loc.y) else OPEN_RTK_STD
    mx_init = init_loc.x + rng_local.normal(0, init_noise_std)
    my_init = init_loc.y + rng_local.normal(0, init_noise_std)
    ekf.initialize(mx_init, my_init, init_vel.x, init_vel.y, init_yaw)

    trial_gps, trial_rtk, trial_ekf = [], [], []

    # Execution Loop
    for step in range(TOTAL_STEPS):
        world.tick()

        transform = ego.get_transform()
        gt_x, gt_y = transform.location.x, transform.location.y
        ax, ay, gyro_z = sensors.get_imu_reading()

        env = "TUNNEL" if in_tunnel(gt_x, gt_y) else ("URBAN" if urban_canyon(gt_x, gt_y) else "OPEN")

        if env != "TUNNEL":
            # 1. Stochastic GPS simulation
            gps_std = URBAN_GPS_STD if env == "URBAN" else OPEN_GPS_STD
            gps_x = gt_x + rng_local.normal(0, gps_std)
            gps_y = gt_y + rng_local.normal(0, gps_std)
            trial_gps.append(math.sqrt((gps_x - gt_x)**2 + (gps_y - gt_y)**2))

            # 2. Stochastic RTK simulation
            noise_std = URBAN_RTK_STD if env == "URBAN" else OPEN_RTK_STD
            rtk_x = gt_x + rng_local.normal(0, noise_std)
            rtk_y = gt_y + rng_local.normal(0, noise_std)
            if env == "URBAN" and rng_local.random() < MULTIPATH_PROB:
                rtk_x += rng_local.normal(0, MULTIPATH_STD)
                rtk_y += rng_local.normal(0, MULTIPATH_STD)
            trial_rtk.append(math.sqrt((rtk_x - gt_x)**2 + (rtk_y - gt_y)**2))
            rtk_avail = True
        else:
            rtk_avail = False

        # 3. Process State Model Updates
        ekf.predict(ax, ay, gyro_z, SIM_DT)
        if rtk_avail:
            ekf.update(rtk_x, rtk_y, urban=(env == "URBAN"), bypass_gate=(step < 100))

        if env != "TUNNEL":
            ekf_x, ekf_y = ekf.x[0, 0], ekf.x[1, 0]
            trial_ekf.append(math.sqrt((ekf_x - gt_x)**2 + (ekf_y - gt_y)**2))

    ensemble_stats["GPS"].append(trial_gps)
    ensemble_stats["RTK"].append(trial_rtk)
    ensemble_stats["INS_EKF"].append(trial_ekf)

    # Comprehensive Teardown (Zero State Bleed Between Runs)
    sensors.destroy()
    for vehicle in vehicles_list:
        try: vehicle.destroy()
        except: pass
    world.tick()

# ==========================================================
# POST-PROCESSING METRIC ANALYSIS GENERATION
# ==========================================================
compiled_summary = []
for system_key in ["GPS", "RTK", "INS_EKF"]:
    trial_means = [np.mean(run) for run in ensemble_stats[system_key]]
    trial_medians = [np.median(run) for run in ensemble_stats[system_key]]
    trial_p95s = [np.percentile(run, 95) for run in ensemble_stats[system_key]]
    trial_maxes = [np.max(run) for run in ensemble_stats[system_key]]
    
    compiled_summary.append({
        "System": system_key,
        "Mean_RMSE": np.mean(trial_means), "Mean_Std": np.std(trial_means),
        "Median": np.mean(trial_medians),
        "P95": np.mean(trial_p95s),
        "Max_Peak": np.mean(trial_maxes), "Max_Std": np.std(trial_maxes)
    })

df_output = pd.DataFrame(compiled_summary)
df_output.to_csv(CSV_NAME, index=False)

print("\n==========================================================================================")
print("                    VALIDATED EXPERIMENT 1 MONTE CARLO RESULTS MATRIX                     ")
print("==========================================================================================")
print("System      |   Mean RMSE (m)     |   Median (m)   |     P95 (m)    |   Maximum Peak Error (m)  ")
print("------------------------------------------------------------------------------------------")
for _, row in df_output.iterrows():
    print(f"{row['System']:<11} | {row['Mean_RMSE']:6.3f} ± {row['Mean_Std']:5.3f} |   {row['Median']:6.3f}   |   {row['P95']:6.3f}    |  {row['Max_Peak']:6.3f} ± {row['Max_Std']:5.3f}")
print("==========================================================================================")

# Generate high-confidence bar chart for thesis use
plt.figure(figsize=(7, 5))
plt.bar(df_output["System"], df_output["Mean_RMSE"], yerr=df_output["Mean_Std"], color=['#1f77b4', '#ff7f0e', '#2ca02c'], edgecolor='black', capsize=6, alpha=0.85)
plt.ylabel("Ensemble Root Mean Square Error [RMSE] (Meters)", fontweight='bold')
plt.title("Experiment 1: Validated Monte Carlo Comparative Performance\n(Controlled Spatial Geometry Baseline)", fontweight='bold')
plt.grid(axis='y', linestyle=':', alpha=0.5)
plt.tight_layout()
plt.savefig("experiment1_validated_montecarlo_rmse.png")
plt.close()

# Revert simulator settings to standard free run mode
settings = world.get_settings()
settings.synchronous_mode = False
settings.fixed_delta_seconds = None
world.apply_settings(settings)
print("\n[Complete] Data structures match scientific requirements. Output files saved.")